"""Record DAgger-style takeover (recovery) demonstrations in the camera env.

Per attempt (1 env): reset with a fresh seed, --settle_steps hold steps (the first post-reset camera frame is the
previous episode's render: neither the policy nor the saved data sees it); the ACT policy (scripts/policy_server.py)
drives the arm until the takeover; then CubeInCupExpert.replan() picks the phase that fits the current state and the
expert runs to success (at most --expert_time s). Only the frames from the takeover on are saved
(observations as rendered then, expert actions), and only if the expert succeeds -- in exactly
record_demos.py's format, so scripts/convert_to_lerobot.py works unchanged:
  <out>/ep_XXXXXX/{data.npz, wrist.mp4, scene.mp4}  (+ takeover.json: when/how the takeover happened)
  data.npz: state [T,6] rad (joint positions BEFORE the action), action [T,6] rad (absolute joint
  targets), phase [T], cube_size, cube_idx, success, fps.

When to take over (each attempt draws a mode; --stuck_frac sets the mix):
  time : after t0 ~ U(--t0_min, --t0_max) s of policy control (early / mid / late task stages);
  stuck: when the policy looks stuck -- TCP nearly static for --static_s s ("static"), cube dropped from the hand
         and come to rest outside the cup ("cube_dropped"), jaw shut on nothing after opening at the cube (a missed
         grasp: "empty_grasp"), or, over --hover_s with the jaws open and the cube neither held nor in the cup, the TCP
         dithering within --dither_amp ("dither") or hovering within --hover_r of the cube without getting closer to
         it ("hover": ACT v3's dominant failure, the open hand stopping a few cm from the cube and never descending /
         closing) -- or at --stuck_max s.
         With probability --stuck_wait_p the takeover waits a further U(0, --stuck_wait_max) s after that sign
         (trigger "<sign>+wait"), so the policy carries on into its downstream failure (an empty hand hovering
         at the cup, re-grasps next to the cup, dithering).
  In both modes a cup starting to tilt (> 6 deg) triggers the takeover at once.
Attempts in which the policy succeeds, or the episode ends, before the takeover are not saved.
Statistics: every attempt is appended to <out>/takeover_log.jsonl with the takeover stage (from the history:
reach = no grasp attempt yet, regrasp = after a missed / lost grasp, carry, place; "@cup" = hand within 8 cm
of the cup axis), the privileged state at the takeover (state0) and the cup tilt / shift per expert phase.
A takeover counts as unrecoverable only if the cup is tilted > 15 deg at the takeover or within 0.5 s after
it (it was already falling), or the cube is off the desk / out of reach. The summary (every 10 attempts and at
the end) gives the expert recovery rate over ALL takeovers and over the recoverable ones, by stage, phase,
trigger and cube size. Quality filters (not saved, counted as failures): cup_bumped -- the cup tilted more than
--max_cup_tilt deg beyond its tilt at the takeover or was pushed more than --max_cup_shift mm while the expert
re-grasped / carried (phases before RELEASE), or more than --max_cup_tilt_place / --max_cup_shift_place during
release / retreat (where the expert from reset itself has a tail: p95 8.6 deg, p99 18.6 deg); ik_jump -- an arm
joint target jumping more than --max_action_jump rad in one step.
Re-running with the same --out resumes (attempt numbering, seed sequence and statistics continue; an
episode is renamed into place only when complete, and one whose log line was lost is recovered from its
takeover.json). The Isaac camera renderer occasionally crashes, so long runs belong in a retry loop -- see below.
--preview N writes <out>/takeover_preview.mp4: scene | wrist (half size) of the first N saved episodes
INCLUDING their policy part, with a POLICY / EXPERT banner and a red frame at the takeover (a resumed
run writes takeover_preview_<first ep>.mp4; takeover.json says where each episode starts).
--fail_preview N writes the first N failed takeovers the same way to takeover_failures[_<attempt>].mp4.

The policy server is started here (LeRobot venv, --ckpt, on --port, default chunking) and stopped at the
end; --ckpt '' uses a server that is already running on --port instead.
Run (heavy GPU job: through the serializing wrapper; the retry loop runs INSIDE it, so a crash does not
give the GPU lock away mid-run; between retries, drop the page-cache copies of Isaac's shader caches -- with
the unstable RAM, corrupted cached pages made every camera start crash in the shader compiler):
  /mnt/secondary/so101_r2s/gpu_run.sh <log> -- bash -c 'for i in $(seq 1 30); do ./lab.sh -p scripts/record_takeover.py \
      --episodes 300 --preview 12 --fail_preview 12 --out <out> --viz none && exit 0; echo "[retry] $i failed $(date +%T)"; \
      <drop cached pages of /mnt/secondary/isaacsim/kit/cache and extscache (posix_fadvise DONTNEED)>; sleep 5; done; exit 1'
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SO101-CubeInCup-Cam-v0")
parser.add_argument("--episodes", type=int, default=20, help="number of SUCCESSFUL takeover episodes to save")
parser.add_argument("--out", default="/mnt/secondary/so101_r2s/sim_demos/takeover1")
parser.add_argument("--preview", type=int, default=0)
parser.add_argument("--speed", type=float, default=0.20)
parser.add_argument("--noise", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=1000, help="attempt k resets the env with seed + k")
parser.add_argument("--port", type=int, default=6101)
parser.add_argument("--ckpt", default="/mnt/secondary/so101_r2s/train_runs/act_sim_v1/checkpoints/030000/pretrained_model",
                    help="policy to serve; '' = connect to a server already running on --port")
parser.add_argument("--lerobot_python", default="/mnt/secondary/so101_r2s/.venv-lerobot/bin/python",
                    help="(falls back to ~/lr/bin/python, the Brev boxes' LeRobot venv, if missing)")
parser.add_argument("--task_text", default="Put the cube in the cup")
parser.add_argument("--stuck_frac", type=float, default=0.5, help="fraction of attempts using the 'stuck' trigger")
parser.add_argument("--t0_min", type=float, default=0.3)
parser.add_argument("--t0_max", type=float, default=9.0)
parser.add_argument("--stuck_max", type=float, default=14.0, help="'stuck' mode: take over at this time at the latest")
parser.add_argument("--static_s", type=float, default=1.5, help="'stuck' mode: TCP static this long = stuck")
parser.add_argument("--hover_s", type=float, default=2.0,
                    help="'stuck' mode: window [s] of the hover / dither signs (jaws open, cube not held, not in the cup)")
parser.add_argument("--hover_r", type=float, default=0.10,
                    help="hover sign: TCP within this [m] (xy) of the cube and ...")
parser.add_argument("--hover_progress", type=float, default=0.01,
                    help="... its 3-D distance to the cube came no closer than this [m] beyond the best before the window "
                         "(hovering / dithering next to the cube without descending onto it)")
parser.add_argument("--dither_amp", type=float, default=0.04,
                    help="dither sign: TCP range over the window below this [m] anywhere (small-amplitude dithering)")
parser.add_argument("--settle_steps", type=int, default=1,
                    help="hold steps after each reset before the policy starts: the first post-reset camera frame is a "
                         "stale render of the previous episode (as scripts/eval_policy.py)")
parser.add_argument("--n_action_steps", type=int, default=None, help="passed to the policy server it starts (default: as trained)")
parser.add_argument("--save_traces", action="store_true",
                    help="write <out>/traces/attempt_XXXXXX.npz (policy part: t, tcp, cube, gripper q, held) for tuning the signs")
parser.add_argument("--stuck_wait_p", type=float, default=0.5, help="'stuck' mode: probability of waiting after the sign ...")
parser.add_argument("--stuck_wait_max", type=float, default=6.0, help="... for U(0, this) s before taking over")
parser.add_argument("--expert_time", type=float, default=15.0, help="expert time limit after the takeover [s]")
parser.add_argument("--max_attempts", type=int, default=0, help="stop after this many attempts (0 = 10 x episodes)")
parser.add_argument("--max_cup_tilt", type=float, default=8.0,
                    help="not a clean recovery (not saved, outcome cup_bumped) if the cup tilted this much more [deg] than "
                         "at the takeover while the expert re-grasped / carried (phases before RELEASE) ...")
parser.add_argument("--max_cup_shift", type=float, default=10.0, help="... or was pushed this far [mm]")
parser.add_argument("--max_cup_tilt_place", type=float, default=20.0,
                    help="the same during release / retreat (expert from reset, 252 successes: tilt median 0.1, p95 8.6, "
                         "p99 18.6 deg -- mostly when the open jaw leaves the cup; the 400 demos have the same tail) ...")
parser.add_argument("--max_cup_shift_place", type=float, default=20.0, help="... [mm] (from reset: p95 4.8, p99 16.6 mm)")
parser.add_argument("--max_action_jump", type=float, default=0.25,
                    help="not saved (outcome ik_jump) if an arm joint target jumps more than this [rad] in one step "
                         "(400 demos: max 0.198, p95 of the per-episode max 0.165)")
parser.add_argument("--fail_preview", type=int, default=0, help="write the first N failed takeovers to takeover_failures.mp4")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import ctypes
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from collections import Counter, defaultdict, deque
from multiprocessing.connection import Client
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

import so101_r2s2r  # noqa: F401
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import mdp, units
from so101_r2s2r.expert import DONE, LIFT, LOWER, PHASE_NAMES, RELEASE, TRANSPORT, CubeInCupExpert

REPO = Path(__file__).resolve().parents[1]
CUP_TILT_TRIGGER = math.cos(math.radians(6.0))
CUP_TILT_UNRECOVERABLE = math.cos(math.radians(15.0))
FALLING_S = 0.5   # [s] cup tilting past 15 deg this soon after the takeover: it was already falling (unrecoverable)
MAX_REACH = 0.31  # [m] cube farther than this from the base (horizontally) is out of the expert's reach
AT_CUP = 0.08     # [m] hand (TCP) this close to the cup axis (xy) at the takeover: stage "...@cup"


class Mp4:
    """Pipe RGB frames into ffmpeg (H.264, yuv420p, 30 fps) -- as in record_demos.py."""

    def __init__(self, path, w, h, fps=30, crf=18):
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
             "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


# ----------------------------------------------------------------------------- preview overlay (5x7 font)
_FONT = {
    "A": "01110 10001 10001 11111 10001 10001 10001", "B": "11110 10001 10001 11110 10001 10001 11110",
    "C": "01110 10001 10000 10000 10000 10001 01110", "D": "11110 10001 10001 10001 10001 10001 11110",
    "E": "11111 10000 10000 11110 10000 10000 11111", "F": "11111 10000 10000 11110 10000 10000 10000",
    "G": "01110 10001 10000 10111 10001 10001 01111", "H": "10001 10001 10001 11111 10001 10001 10001",
    "I": "01110 00100 00100 00100 00100 00100 01110", "J": "00111 00010 00010 00010 00010 10010 01100",
    "K": "10001 10010 10100 11000 10100 10010 10001", "L": "10000 10000 10000 10000 10000 10000 11111",
    "M": "10001 11011 10101 10101 10001 10001 10001", "N": "10001 10001 11001 10101 10011 10001 10001",
    "O": "01110 10001 10001 10001 10001 10001 01110", "P": "11110 10001 10001 11110 10000 10000 10000",
    "Q": "01110 10001 10001 10001 10101 10010 01101", "R": "11110 10001 10001 11110 10100 10010 10001",
    "S": "01111 10000 10000 01110 00001 00001 11110", "T": "11111 00100 00100 00100 00100 00100 00100",
    "U": "10001 10001 10001 10001 10001 10001 01110", "V": "10001 10001 10001 10001 10001 01010 00100",
    "W": "10001 10001 10001 10101 10101 10101 01010", "X": "10001 10001 01010 00100 01010 10001 10001",
    "Y": "10001 10001 01010 00100 00100 00100 00100", "Z": "11111 00001 00010 00100 01000 10000 11111",
    "0": "01110 10001 10011 10101 11001 10001 01110", "1": "00100 01100 00100 00100 00100 00100 01110",
    "2": "01110 10001 00001 00010 00100 01000 11111", "3": "11111 00010 00100 00010 00001 10001 01110",
    "4": "00010 00110 01010 10010 11111 00010 00010", "5": "11111 10000 11110 00001 00001 10001 01110",
    "6": "00110 01000 10000 11110 10001 10001 01110", "7": "11111 00001 00010 00100 01000 01000 01000",
    "8": "01110 10001 10001 01110 10001 10001 01110", "9": "01110 10001 10001 01111 00001 00010 01100",
    " ": "00000 00000 00000 00000 00000 00000 00000", "-": "00000 00000 00000 11111 00000 00000 00000",
    "+": "00000 00100 00100 11111 00100 00100 00000", ".": "00000 00000 00000 00000 00000 01100 01100",
    ":": "00000 01100 01100 00000 01100 01100 00000", "_": "00000 00000 00000 00000 00000 00000 11111",
    "/": "00001 00010 00010 00100 01000 01000 10000", "=": "00000 00000 11111 00000 11111 00000 00000",
}
_GLYPH = {c: np.array([[b == "1" for b in row] for row in g.split()], bool) for c, g in _FONT.items()}


def draw_text(img, text, x, y, color=(255, 255, 255), scale=2):
    for ch in text.upper():
        g = _GLYPH.get(ch, _GLYPH[" "])
        m = np.kron(g, np.ones((scale, scale), bool))
        h, w = m.shape
        if x + w > img.shape[1] or y + h > img.shape[0]:
            break
        img[y:y + h, x:x + w][m] = color
        x += w + scale


def preview_frame(scene, wrist, banner, color, flash=False):
    """Half-size scene | wrist with a coloured banner; a thick red frame marks the takeover."""
    f = np.concatenate([scene[::2, ::2], wrist[::2, ::2]], axis=1).copy()
    f[:18] = color
    draw_text(f, banner, 4, 2)
    if flash:
        f[:, :8] = f[:, -8:] = f[-8:] = (255, 0, 0)
        f[:18] = (255, 0, 0)
        draw_text(f, "TAKEOVER  " + banner, 4, 2)
    return f


# ----------------------------------------------------------------------------- policy server
def _die_with_parent():
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG: never orphaned


def start_server(out_dir):
    with socket.socket() as s:  # refuse to silently talk to somebody else's server on this port
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # (but a crashed run's TIME_WAIT is no reason)
        try:
            s.bind(("127.0.0.1", args.port))
        except OSError:
            raise SystemExit(f"[takeover] port {args.port} is busy; pick another --port or pass --ckpt '' to use that server")
    keep = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "TMPDIR", "CUDA_VISIBLE_DEVICES", "HF_HOME",
            "HF_LEROBOT_HOME", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR")
    env = {k: os.environ[k] for k in keep if k in os.environ}  # NOT Isaac's PYTHONPATH / LD_* / carb settings
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    log = open(os.path.join(out_dir, "policy_server.log"), "a")
    py = args.lerobot_python if os.path.exists(args.lerobot_python) else os.path.expanduser("~/lr/bin/python")
    cmd = [py, "-u", str(REPO / "scripts" / "policy_server.py"), "--ckpt", args.ckpt, "--port", str(args.port)]
    if args.n_action_steps is not None:
        cmd += ["--n_action_steps", str(args.n_action_steps)]
    srv = subprocess.Popen(cmd, env=env, cwd=str(REPO), stdout=log, stderr=subprocess.STDOUT, preexec_fn=_die_with_parent)
    print(f"[takeover] policy server pid {srv.pid} on :{args.port} ({args.ckpt})", flush=True)
    return srv


def connect(srv, timeout=300.0):
    t = time.time()
    while True:
        if srv is not None and srv.poll() is not None:
            raise SystemExit(f"[takeover] policy server exited rc={srv.returncode} (see policy_server.log)")
        try:
            return Client(("127.0.0.1", args.port), authkey=b"so101")
        except OSError:
            if time.time() - t > timeout:
                raise
            time.sleep(1.0)


def stop_server(srv, conn):
    if conn is not None:
        try:
            if srv is not None:
                conn.send({"cmd": "quit"})
                conn.recv()
            conn.close()
        except (OSError, EOFError):
            pass
    if srv is not None:
        try:
            srv.wait(timeout=15)
        except subprocess.TimeoutExpired:
            srv.terminate()  # by PID
            try:
                srv.wait(timeout=10)
            except subprocess.TimeoutExpired:
                srv.kill()
        print(f"[takeover] policy server pid {srv.pid} stopped (rc={srv.returncode})", flush=True)


# ----------------------------------------------------------------------------- privileged state helpers (env 0)
def cup_up_z(u):
    up = math_utils.quat_apply(u.scene["cup"].data.root_quat_w.torch, torch.tensor([[0.0, 0.0, 1.0]], device=u.device))
    return float(up[0, 2])


def cube_speed(u):
    v = torch.stack([u.scene[n].data.root_lin_vel_w.torch for n in mdp.CUBE_NAMES], 1)[0, mdp.active_cube_idx(u)[0]]
    return float(torch.linalg.vector_norm(v))


class StuckMonitor:
    """Signs that the policy is stuck (privileged state; evaluated before every policy action), and the history
    of the attempt (was the cube ever held / did the jaw ever shut on nothing) for the stage label."""

    def __init__(self, u):
        self.u, self.dt = u, u.step_dt
        self.tcp = deque(maxlen=max(2, int(round(args.static_s / self.dt))))
        nh = max(2, int(round(args.hover_s / self.dt)))
        self.h_tcp, self.h_dist, self.h_ok = deque(maxlen=nh), deque(maxlen=nh), deque(maxlen=nh)  # hover / dither window
        self.best_before = float("inf")  # closest TCP-cube distance before the hover window
        self.held_n = self.lost_n = self.empty_n = 0
        self.was_held = self.jaw_was_open = self.ever_empty = False
        self.trace = []  # (t, tcp xyz, cube xyz, gripper q, held) in env-local coordinates (--save_traces)

    def check(self, t):
        u = self.u
        cube_p, _ = mdp.active_cube_pose_w(u)
        size = float(mdp.active_cube_size(u)[0])
        tcp = mdp.tcp_pos_w(u)[0]
        q = float(mdp.gripper_q(u)[0])
        q_contact = (size - 0.017) / 0.066
        held = bool(mdp.cube_grasped(u)[0])
        self.held_n = self.held_n + 1 if held else 0
        self.was_held |= self.held_n >= 3
        dist = float(torch.linalg.vector_norm(cube_p[0] - tcp))
        # (opened AT the cube, i.e. a grasp attempt: from the folded rest pose ACT v3 opens / shuts the jaw a little on the
        # way up, ~10 cm from the cube -- counted, that fired a false "empty_grasp" at ~1.3 s)
        self.jaw_was_open |= q > q_contact + 0.15 and dist < 0.06
        far = dist > 0.035
        shut = q < q_contact - 0.10
        lost = self.was_held and not held and (far or shut) and not bool(mdp.cube_in_cup(u)[0]) and cube_speed(u) < 0.05
        self.lost_n = self.lost_n + 1 if lost else 0
        self.empty_n = self.empty_n + 1 if (self.jaw_was_open and shut and not held) else 0
        self.ever_empty |= self.empty_n >= 3
        self.tcp.append(tcp.cpu().numpy())
        # hover / dither (ACT v3's dominant failure: the open hand stops a few cm from the cube -- or dithers there with a
        # small amplitude as each new chunk re-aims -- and never descends / closes): over the last --hover_s, jaws open,
        # cube neither held nor in the cup, and either the TCP stayed within --dither_amp, or it is within --hover_r (xy) of
        # the cube and came no closer to it than --hover_progress beyond its best distance before the window.
        if len(self.h_dist) == self.h_dist.maxlen and t - args.hover_s > 1.0:  # (not the climb out of the rest pose:
            self.best_before = min(self.best_before, self.h_dist[0])          # from there the hand starts near the cube)
        self.h_tcp.append(self.tcp[-1])
        self.h_dist.append(dist)
        self.h_ok.append(q > q_contact + 0.15 and not held and not bool(mdp.cube_in_cup(u)[0]))
        if args.save_traces:
            o = self.u.scene.env_origins[0]
            self.trace.append(np.concatenate([[t], (tcp - o).cpu().numpy(), (cube_p[0] - o).cpu().numpy(), [q, float(held)]]))
        if self.lost_n >= 6:
            return "cube_dropped"
        if self.empty_n >= 10:
            return "empty_grasp"
        if t > 1.0 and len(self.tcp) == self.tcp.maxlen and np.ptp(np.stack(self.tcp), axis=0).max() < 0.015:
            return "static"
        if t > 1.0 and len(self.h_ok) == self.h_ok.maxlen and all(self.h_ok):
            if np.ptp(np.stack(self.h_tcp), axis=0).max() < args.dither_amp:
                return "dither"
            xy = float(torch.linalg.vector_norm(cube_p[0, :2] - tcp[:2]))
            if xy < args.hover_r and min(self.h_dist) > self.best_before - args.hover_progress:
                return "hover"
        return None


def cup_tilt_deg(u):
    return math.degrees(math.acos(max(-1.0, min(1.0, cup_up_z(u)))))


def classify(u):
    """Is the takeover state recoverable at all? (None = yes, else the reason). A cup that is merely tilted (6..15
    deg: the cup_tilt trigger) and still rocking is NOT written off here -- whether it was already falling is decided
    after the fact (tilt > 15 deg within FALLING_S of the takeover)."""
    if cup_up_z(u) < CUP_TILT_UNRECOVERABLE:
        return "cup_tilted"
    cube_p, _ = mdp.active_cube_pose_w(u)
    if float(cube_p[0, 2] - u.scene.env_origins[0, 2]) < -0.01:
        return "cube_off_desk"
    rel = cube_p[0, :2] - u.scene["robot"].data.root_pos_w.torch[0, :2]
    if float(torch.linalg.vector_norm(rel)) > MAX_REACH or float(rel[0]) < 0.0:
        return "cube_out_of_reach"
    return None


def takeover_state(u):
    """Privileged state at the takeover (for the log)."""
    cube_p, _ = mdp.active_cube_pose_w(u)
    cube_p = cube_p[0]
    cup = u.scene["cup"].data.root_pos_w.torch[0]
    dz = u.scene.env_origins[0, 2]
    tcp = mdp.tcp_pos_w(u)[0]
    base = u.scene["robot"].data.root_pos_w.torch[0]
    size = float(mdp.active_cube_size(u)[0])
    mm = lambda v: round(1000 * float(v), 1)
    xy = lambda a, b: torch.linalg.vector_norm(a[:2] - b[:2])
    return dict(cube_cup_mm=mm(xy(cube_p, cup)), cube_z_mm=mm(cube_p[2] - dz), cube_base_mm=mm(xy(cube_p, base)),
                tcp_z_mm=mm(tcp[2] - dz), tcp_cube_xy_mm=mm(xy(tcp, cube_p)), tcp_cup_xy_mm=mm(xy(tcp, cup)),
                grip_q=round(float(mdp.gripper_q(u)[0]), 3), q_contact=round((size - 0.017) / 0.066, 3),
                held=bool(mdp.cube_grasped(u)[0]), in_cup=bool(mdp.cube_in_cup(u)[0]))


def stage_group(phase, s0, mon):
    """Task stage at the takeover, from the history of the attempt (not just the jaw state at that moment)."""
    if s0["in_cup"] or phase >= RELEASE:
        g = "place"  # the cube is in the cup (release / retreat / done)
    elif s0["held"] or phase in (LIFT, TRANSPORT, LOWER):
        g = "carry"
    else:
        g = "regrasp" if (mon.was_held or mon.ever_empty) else "reach"  # after a missed / lost grasp, or not yet tried
    return g + ("@cup" if g in ("reach", "regrasp") and s0["tcp_cup_xy_mm"] < 1000 * AT_CUP else "")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def new_stats():
    cell = lambda: dict(n=0, ok=0)
    return dict(attempts=0, max_attempt=-1, saved=0, takeovers=0, all_ok=0, all_reached=0, pre=Counter(), unrec=Counter(),
                unrec_ok=Counter(), fail=Counter(), rec=defaultdict(cell), stage=defaultdict(cell), phase=defaultdict(cell),
                trigger=defaultdict(cell), cube=defaultdict(cell))


def account(st, rec):
    """Add one attempt record (a takeover_log.jsonl line) to the statistics."""
    st["attempts"] += 1
    st["max_attempt"] = max(st["max_attempt"], int(rec["attempt"]))
    if "trigger" not in rec:  # the episode ended (policy success / failure) before any takeover
        st["pre"][rec["outcome"]] += 1
        return
    st["takeovers"] += 1
    st["saved"] += "episode" in rec
    ok = rec["outcome"] == "success"
    st["all_ok"] += ok
    st["all_reached"] += ok or rec["outcome"] in ("cup_bumped", "ik_jump")  # got the cube in, before the quality filters
    reason = rec.get("unrecoverable")
    if reason is not None:
        st["unrec"][reason] += 1
        st["unrec_ok"][reason] += ok
        return
    for key, val in (("rec", "all"), ("stage", rec["stage"]), ("phase", rec["phase"]), ("trigger", rec["trigger"]),
                     ("cube", f"{rec['cube_mm']}mm")):
        st[key][val]["n"] += 1
        st[key][val]["ok"] += ok
    if not ok:
        st["fail"][f"{rec['phase']}:{rec['outcome']}"] += 1


def summary(st):
    a, T = st["attempts"], st["takeovers"]
    print(f"[takeover] ---- {a} attempts: {st['saved']} saved; before any takeover: {dict(st['pre'])}; {T} takeovers", flush=True)
    if T:
        lo, hi = wilson(st["all_ok"], T)
        print(f"[takeover]   expert recovery, ALL takeovers: {st['all_ok']}/{T} = {st['all_ok'] / T:.0%} (95% CI {lo:.0%}-{hi:.0%}); "
              f"cube in the cup before the quality filters {st['all_reached']}/{T}", flush=True)
    print(f"[takeover]   unrecoverable (cup tilted > 15 deg at / within {FALLING_S} s of the takeover, cube off the desk / out "
          f"of reach): {dict(st['unrec'])}, of which recovered anyway {sum(st['unrec_ok'].values())}", flush=True)
    c = st["rec"]["all"]
    if c["n"]:
        lo, hi = wilson(c["ok"], c["n"])
        print(f"[takeover]   expert recovery, recoverable takeovers: {c['ok']}/{c['n']} = {c['ok'] / c['n']:.0%} "
              f"(95% CI {lo:.0%}-{hi:.0%}); failures {dict(st['fail'])}", flush=True)
    for key in ("stage", "phase", "trigger", "cube"):
        parts = [f"{k} {v['ok']}/{v['n']}" for k, v in sorted(st[key].items())]
        print(f"[takeover]   by {key}: " + ", ".join(parts), flush=True)


# ----------------------------------------------------------------------------- main
env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = 1
env_cfg.seed = args.seed
env_cfg.terminations.success = None  # as record_demos.py: the expert finishes (release + retreat); we end episodes
env_cfg.episode_length_s = max(args.t0_max, args.stuck_max) + args.expert_time + 2.0
os.makedirs(args.out, exist_ok=True)
srv = start_server(args.out) if args.ckpt else None
conn = None


def load_log(out, log_path):
    """Attempt records of earlier runs into <out>; an episode renamed into place whose log line was lost (crash in
    between) is recovered from its takeover.json; half-written episodes (.tmp_ep_*) of a crashed run are removed."""
    for d in os.listdir(out):
        if d.startswith(".tmp_ep_"):
            shutil.rmtree(os.path.join(out, d), ignore_errors=True)
            print(f"[takeover] removed the half-written {d} of a crashed run", flush=True)
    recs = [json.loads(line) for line in open(log_path) if line.strip()] if os.path.exists(log_path) else []
    seen = {int(r["attempt"]) for r in recs}
    for d in sorted(os.listdir(out)):
        tj = os.path.join(out, d, "takeover.json")
        if re.fullmatch(r"ep_\d{6}", d) and os.path.exists(tj):
            r = json.load(open(tj))
            if int(r["attempt"]) not in seen:
                print(f"[takeover] {d}: attempt {r['attempt']} missing from the log, recovered from takeover.json", flush=True)
                with open(log_path, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
                recs.append(r)
                seen.add(int(r["attempt"]))
    return recs


class Preview:
    """An mp4 of (policy part + expert part) episodes with banners; opened on the first write."""

    def __init__(self, path):
        self.path, self.mp4, self.frames = path, None, 0

    def write(self, pv, buf, tag, color):
        h, w = buf["wrist"][0].shape[:2]
        if self.mp4 is None:
            self.mp4 = Mp4(self.path, w, h // 2)
        for f in pv:
            self.mp4.write(f)
        for i, (fs, fw) in enumerate(zip(buf["scene"], buf["wrist"])):
            self.mp4.write(preview_frame(fs, fw, tag, color, flash=i < 15))
        self.frames += len(pv) + len(buf["scene"])

    def close(self):
        if self.mp4 is not None:
            self.mp4.close()


with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    env.reset(seed=args.seed)
    u = env.unwrapped
    dt = u.step_dt
    expert = CubeInCupExpert(u, speed=args.speed, noise=args.noise, seed=args.seed)
    # resume (e.g. after an Isaac crash): continue the attempt / seed sequence and the statistics of <out>
    log_path = os.path.join(args.out, "takeover_log.jsonl")
    st = new_stats()
    for r in load_log(args.out, log_path):
        account(st, r)
    prior = st["max_attempt"] + 1
    rng = np.random.default_rng(args.seed + 7919 * prior)
    expert.gen.manual_seed(args.seed + 7919 * prior)  # (waypoint jitter: no repeat of the first run's sequence)
    eps = [int(d[3:]) for d in os.listdir(args.out) if re.fullmatch(r"ep_\d{6}", d)]
    saved, next_idx = len(eps), max(eps, default=-1) + 1
    if prior:
        print(f"[takeover] resuming {args.out}: {st['attempts']} attempts (next {prior}) / {saved} episodes so far", flush=True)
    max_attempts = args.max_attempts or 10 * args.episodes  # (total, including earlier runs)
    preview = Preview(os.path.join(args.out, "takeover_preview.mp4" if saved == 0 else f"takeover_preview_{saved:06d}.mp4"))
    fails = Preview(os.path.join(args.out, "takeover_failures.mp4" if prior == 0 else f"takeover_failures_{prior:06d}.mp4"))
    n_fail_pv, n_new, saved_new, t_start, frames_saved, steps_total = 0, 0, 0, time.time(), 0, 0
    try:
        conn = connect(srv)
        print(f"[takeover] connected to the policy server on :{args.port}", flush=True)
        while saved < args.episodes and st["attempts"] < max_attempts:
            k = prior + n_new
            n_new += 1
            seed = args.seed + k
            obs, _ = env.reset(seed=seed)
            for _ in range(args.settle_steps):  # (the first post-reset frame is the previous episode's render)
                obs, *_ = env.step(u.scene["robot"].data.joint_pos.torch.clone())
            expert.reset()
            conn.send({"cmd": "reset"}); conn.recv()
            mode = "stuck" if rng.random() < args.stuck_frac else "time"
            t0 = float(rng.uniform(args.t0_min, args.t0_max)) if mode == "time" else args.stuck_max
            wait = float(rng.uniform(0.0, args.stuck_wait_max)) if rng.random() < args.stuck_wait_p else 0.0
            size_mm = int(round(float(mdp.active_cube_size(u)[0]) * 1000))
            rec = dict(attempt=k, seed=seed, mode=mode, t0=round(t0, 2), cube_mm=size_mm)
            if mode == "stuck":
                rec["wait"] = round(wait, 2)
            want_pv = saved < args.preview or n_fail_pv < args.fail_preview
            pv = []
            mon = StuckMonitor(u)
            step, trigger, pre_outcome, sign0, sign_t = 0, None, None, None, None
            # ------------------------------------------------ policy part
            while True:
                t = step * dt
                sign = mon.check(t)
                if sign is not None and sign0 is None:
                    sign0, sign_t = sign, t
                if cup_up_z(u) < CUP_TILT_TRIGGER:
                    trigger = "cup_tilt"
                elif mode == "time" and t >= t0:
                    trigger = "time"
                elif mode == "stuck" and sign0 is not None and t >= sign_t + wait:
                    trigger = sign0 + ("+wait" if wait > 0 else "")
                elif mode == "stuck" and t >= t0:
                    trigger = "time_limit"
                if trigger is not None:
                    break
                q = u.scene["robot"].data.joint_pos.torch[0].cpu().numpy()
                wrist = obs["images"]["wrist"][0, ..., :3].to(torch.uint8).cpu().numpy()
                scene = obs["images"]["scene"][0, ..., :3].to(torch.uint8).cpu().numpy()
                conn.send({"cmd": "act", "task": args.task_text, "obs": {
                    "observation.state": units.sim_to_lerobot(q),
                    "observation.images.wrist": wrist, "observation.images.scene": scene}})
                a = units.lerobot_to_sim(conn.recv())
                if want_pv:
                    pv.append(preview_frame(scene, wrist, f"POLICY {t:4.1f}S", (40, 90, 200)))
                obs, rew, term, trunc, info = env.step(torch.tensor(a, device=u.device)[None])
                step += 1
                if bool(term[0] | trunc[0]):  # (the env has auto-reset: check this first)
                    tm = u.termination_manager
                    pre_outcome = "policy_" + next((n for n in ("cup_tipped", "cube_dropped", "physics_blowup")
                                                    if bool(tm.get_term(n)[0])), "time_out")
                elif bool(mdp.task_success(u, open_q=1.0)[0]):
                    pre_outcome = "policy_success"
                if pre_outcome is not None:
                    break
            steps_total += step
            rec.update(policy_steps=step)
            if args.save_traces and mon.trace:
                os.makedirs(os.path.join(args.out, "traces"), exist_ok=True)
                np.savez_compressed(os.path.join(args.out, "traces", f"attempt_{k:06d}.npz"), trace=np.stack(mon.trace),
                                    cols=np.array("t tcp_x tcp_y tcp_z cube_x cube_y cube_z grip_q held".split()))
            if sign0 is not None:
                rec.update(first_sign=sign0, first_sign_t=round(sign_t, 2))
            if pre_outcome is not None:
                rec.update(outcome=pre_outcome)
            else:
                # ------------------------------------------------ takeover: expert from the current state
                reason = classify(u)
                s0 = takeover_state(u)
                tilt0 = cup_tilt_deg(u)
                ph0 = int(expert.replan()[0])
                label = PHASE_NAMES[ph0] + ("+rise" if bool(expert.rise[0]) else "")
                group = stage_group(ph0, s0, mon)
                rec.update(trigger=trigger, t_takeover=round(step * dt, 2), phase=label, stage=group, unrecoverable=reason,
                           cup_tilt0_deg=round(tilt0, 1), state0=s0)
                buf = dict(state=[], action=[], phase=[], wrist=[], scene=[])
                done_for, n, outcome = 0.0, 0, None
                cup_xy0 = u.scene["cup"].data.root_pos_w.torch[0, :2].clone()
                by_ph = {}  # phase -> [max cup tilt beyond the takeover's (deg), max cup shift from the takeover (mm)]
                early_tilt = 0.0
                while outcome is None:
                    q = u.scene["robot"].data.joint_pos.torch.clone()
                    wrist = obs["images"]["wrist"][..., :3].to(torch.uint8).cpu().numpy()
                    scene = obs["images"]["scene"][..., :3].to(torch.uint8).cpu().numpy()
                    action = expert.act()
                    buf["state"].append(q[0].cpu().numpy()); buf["action"].append(action[0].cpu().numpy())
                    buf["phase"].append(int(expert.phase[0])); buf["wrist"].append(wrist[0]); buf["scene"].append(scene[0])
                    size_now, idx_now = mdp.active_cube_size(u).clone(), mdp.active_cube_idx(u).clone()
                    obs, rew, term, trunc, info = env.step(action)
                    n += 1
                    if not bool(term[0] | trunc[0]):  # (an env-side reset would show the next episode's cup)
                        tilt = cup_tilt_deg(u)  # did the expert bump the cup? (per phase)
                        shift = 1000 * float(torch.linalg.vector_norm(u.scene["cup"].data.root_pos_w.torch[0, :2] - cup_xy0))
                        m_ = by_ph.setdefault(PHASE_NAMES[int(expert.phase[0])], [0.0, 0.0])
                        m_[0], m_[1] = max(m_[0], tilt - tilt0), max(m_[1], shift)
                        if n * dt <= FALLING_S:
                            early_tilt = max(early_tilt, tilt)
                    elif n * dt <= FALLING_S and bool(u.termination_manager.get_term("cup_tipped")[0]):
                        early_tilt = 90.0
                    done_for = done_for + dt if int(expert.phase[0]) == DONE else 0.0
                    if done_for > 0.4:  # 0.4 s after the retreat (as record_demos.py)
                        ok = bool(mdp.cube_in_cup(u)[0]) and bool(mdp.cup_upright(u)[0])
                        outcome = "success" if ok else ("cup_tipped" if not bool(mdp.cup_upright(u)[0]) else "not_in_cup")
                    elif bool(term[0] | trunc[0]):
                        tm = u.termination_manager
                        outcome = next((m for m in ("cup_tipped", "cube_dropped", "physics_blowup") if bool(tm.get_term(m)[0])), "time_out")
                    elif n * dt > args.expert_time:
                        outcome = "expert_timeout"
                steps_total += n
                if reason is None and early_tilt > 15.0:
                    reason = rec["unrecoverable"] = "cup_falling"  # tipped within FALLING_S: it was already on its way over
                appr = [v for p_, v in by_ph.items() if PHASE_NAMES.index(p_) < RELEASE]
                plc = [v for p_, v in by_ph.items() if PHASE_NAMES.index(p_) >= RELEASE]
                t_a, s_a = max([v[0] for v in appr], default=0.0), max([v[1] for v in appr], default=0.0)
                t_p, s_p = max([v[0] for v in plc], default=0.0), max([v[1] for v in plc], default=0.0)
                acts = np.stack(buf["action"])[:, :5]
                jump = float(np.abs(np.diff(acts, axis=0)).max()) if len(acts) > 1 else 0.0
                if outcome == "success" and (t_a > args.max_cup_tilt or s_a > args.max_cup_shift
                                             or t_p > args.max_cup_tilt_place or s_p > args.max_cup_shift_place):
                    outcome = "cup_bumped"  # it got there, but by knocking the cup about: not a demo to imitate
                elif outcome == "success" and jump > args.max_action_jump:
                    outcome = "ik_jump"  # an IK chatter step: not a demo to imitate
                ok = outcome == "success"
                rec.update(outcome=outcome, expert_steps=n, auto_replans=int(expert.n_replans[0]),
                           cup_tilt_max_deg=round(tilt0 + max(t_a, t_p, 0.0), 1), cup_shift_mm=round(max(s_a, s_p), 1),
                           cup_appr=[round(t_a, 1), round(s_a, 1)], cup_place=[round(t_p, 1), round(s_p, 1)],
                           cup_by_phase={p_: [round(v[0], 1), round(v[1], 1)] for p_, v in by_ph.items()},
                           max_action_jump=round(jump, 3))
                if ok and saved < args.episodes:
                    # written to a hidden temp dir and renamed when complete: an Isaac crash mid-write leaves no
                    # half episode behind for the resume / convert_to_lerobot.py (which globs ep_*)
                    d = os.path.join(args.out, f"ep_{next_idx:06d}")
                    tmp = os.path.join(args.out, f".tmp_ep_{os.getpid()}")
                    os.makedirs(tmp, exist_ok=True)
                    np.savez_compressed(os.path.join(tmp, "data.npz"), state=np.stack(buf["state"]), action=np.stack(buf["action"]),
                                        phase=np.array(buf["phase"]), cube_size=float(size_now[0]), cube_idx=int(idx_now[0]),
                                        success=True, fps=round(1 / dt))
                    h, w = buf["wrist"][0].shape[:2]
                    for cam in ("wrist", "scene"):
                        vw = Mp4(os.path.join(tmp, f"{cam}.mp4"), w, h)
                        for f in buf[cam]:
                            vw.write(f)
                        vw.close()
                    rec["episode"] = os.path.basename(d)
                    in_pv = saved < args.preview
                    if in_pv:  # where this episode is in the preview (policy part first, then the expert's)
                        rec.update(preview=os.path.basename(preview.path), preview_start=preview.frames,
                                   preview_takeover=preview.frames + len(pv))
                    with open(os.path.join(tmp, "takeover.json"), "w") as fh:
                        json.dump(rec, fh, indent=1)
                    os.rename(tmp, d)  # (a crash before the log line below: load_log recovers it from takeover.json)
                    if in_pv:
                        preview.write(pv, buf, f"EXPERT {trigger} {label}", (220, 120, 20))
                    saved += 1
                    saved_new += 1
                    next_idx += 1
                    frames_saved += n
                    print(f"[takeover] saved ep {saved}/{args.episodes} ({os.path.basename(d)}): {n} frames, cube {size_mm} mm, "
                          f"{mode}/{trigger} at {step * dt:.1f}s -> {label} [{group}] "
                          f"({steps_total / (time.time() - t_start):.1f} env-steps/s)", flush=True)
                elif not ok and n_fail_pv < args.fail_preview:
                    rec.update(preview=os.path.basename(fails.path), preview_start=fails.frames,
                               preview_takeover=fails.frames + len(pv))
                    fails.write(pv, buf, f"FAIL {outcome} {trigger} {label}", (200, 30, 30))
                    n_fail_pv += 1
            print(f"[takeover] attempt {k}: " + json.dumps(rec), flush=True)
            with open(log_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            account(st, rec)
            if st["attempts"] % 10 == 0:
                summary(st)
        summary(st)
        el = time.time() - t_start
        print(f"[takeover] done: {saved} episodes in {args.out} ({saved_new} new, {frames_saved} frames, from {n_new} "
              f"attempts in {el:.0f}s = {el / max(saved_new, 1):.1f}s per saved episode)", flush=True)
    finally:
        preview.close()
        fails.close()
        stop_server(srv, conn)
    env.close()

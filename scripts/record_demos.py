"""Record scripted-expert demonstrations in the camera env (successful episodes only).

Per episode: <out>/ep_XXXXXX/{data.npz, wrist.mp4, scene.mp4}
  data.npz: state [T,6] rad (joint positions BEFORE the action), action [T,6] rad (absolute joint
  targets), phase [T], cube_size, cube_idx, success. Units are converted to LeRobot's later.
With --preview N, also writes preview.mp4 (scene | wrist side by side) of the first N episodes.

DART-style action noise (--action_noise STD, default 0 = off, old behaviour unchanged): the EXECUTED arm joint targets
are the expert's plus smooth, temporally correlated noise (white noise through two cascaded first-order low-pass
filters, --noise_tau ~ its correlation time, scaled to STD rad per joint; gripper never perturbed), applied in the
phases --noise_phases while the expert's Cartesian command is moving (gain ramped over 0.2 s). Once a move's command has
arrived the noise fades out, so the arm settles and the expert's arrival / settle checks end the phase as without noise
(kept on, the jiggling arm missed the 12 mm tolerance and every move ran into the expert's 3 s time box: dwelling above
the cube -- the very hover the policy must unlearn). The SAVED action stays the expert's clean target --
computed by its IK from the CURRENT, perturbed joint state -- so the episodes visit off-nominal states and carry the
corrective labels. data.npz then also holds exec_action [T,6] (what was actually commanded) and action_noise (STD).

Run: ./lab.sh -p scripts/record_demos.py --episodes 20 --preview 4 --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SO101-CubeInCup-Cam-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=20, help="number of SUCCESSFUL episodes to save")
parser.add_argument("--out", default="/mnt/secondary/so101_r2s/sim_demos/run0")
parser.add_argument("--preview", type=int, default=0)
parser.add_argument("--speed", type=float, default=0.20)
parser.add_argument("--noise", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--action_noise", type=float, default=0.0, help="DART: std [rad] of the executed arm-target noise (0 = off)")
parser.add_argument("--noise_tau", type=float, default=0.4, help="DART: correlation time [s] of the noise")
parser.add_argument("--noise_phases", default="pregrasp,descend,close,lift,transport",
                    help="DART: expert phases in which the noise is applied (comma-separated PHASE_NAMES)")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import os
import shutil
import subprocess
import time

import gymnasium as gym
import numpy as np
import torch

import so101_r2s2r  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import mdp
from so101_r2s2r.expert import DONE, PHASE_NAMES, CubeInCupExpert


class DartNoise:
    """Smooth zero-mean noise on the 5 arm joint targets: white noise through two cascaded first-order low-pass filters
    (time constant tau/2 each), normalised to ``std`` [rad] (stationary); per-phase gain ramped over 0.2 s."""

    def __init__(self, n, dev, std, tau, dt, phases, seed):
        self.std, self.dt = std, dt
        self.a = float(np.exp(-dt / max(tau / 2, 1e-6)))
        k = np.arange(20000)
        self.norm = 1.0 / float(np.sqrt(((k + 1.0) ** 2 * self.a ** (2 * k)).sum())) / (1 - self.a) ** 2  # unit variance
        self.mask = torch.tensor([PHASE_NAMES[i] in phases for i in range(len(PHASE_NAMES))], device=dev, dtype=torch.float32)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(seed + 4242)
        self.x1 = torch.zeros(n, 5, device=dev)
        self.x2 = torch.zeros(n, 5, device=dev)
        self.gain = torch.zeros(n, device=dev)

    def reset(self, ids):
        self.x1[ids] = 0.0
        self.x2[ids] = 0.0
        self.gain[ids] = 0.0

    def __call__(self, phase, moving):
        w = torch.randn(self.x1.shape, device=self.x1.device, generator=self.gen)
        self.x1 = self.a * self.x1 + (1 - self.a) * w
        self.x2 = self.a * self.x2 + (1 - self.a) * self.x1
        self.gain += (self.mask[phase] * moving.float() - self.gain).clamp(-self.dt / 0.2, self.dt / 0.2)
        out = torch.zeros(self.x1.shape[0], 6, device=self.x1.device)
        out[:, :5] = self.std * self.norm * self.x2 * self.gain[:, None]
        return out


class Mp4:
    """Pipe RGB frames into ffmpeg (H.264, yuv420p, 30 fps)."""

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


env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env_cfg.terminations.success = None  # let the expert finish (release + retreat); the script ends episodes
env_cfg.episode_length_s = 20.0
os.makedirs(args.out, exist_ok=True)

with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset(seed=args.seed)
    u = env.unwrapped
    N = u.num_envs
    expert = CubeInCupExpert(u, speed=args.speed, noise=args.noise, seed=args.seed)
    expert.reset()
    phases = [p_.strip() for p_ in args.noise_phases.split(",") if p_.strip()]
    assert all(p_ in PHASE_NAMES for p_ in phases), f"--noise_phases must be among {PHASE_NAMES}"
    dart = DartNoise(N, u.device, args.action_noise, args.noise_tau, u.step_dt, phases, args.seed) if args.action_noise > 0 else None
    for d in os.listdir(args.out):  # half-written episode of a crashed run (episodes are renamed into place when complete)
        if d.startswith(".tmp_ep_"):
            shutil.rmtree(os.path.join(args.out, d), ignore_errors=True)
    saved = len([d for d in os.listdir(args.out) if d.startswith("ep_")])
    if saved:
        print(f"[rec] resuming {args.out}: {saved} episodes there (use a new --seed, or the scenes repeat)", flush=True)
    attempts, t0, frames_total = 0, time.time(), 0
    buf = [dict(state=[], action=[], exec=[], phase=[], wrist=[], scene=[]) for _ in range(N)]
    done_for = torch.zeros(N, device=u.device)
    preview = None
    while saved < args.episodes:
        q = u.scene["robot"].data.joint_pos.torch.clone()
        wrist = obs["images"]["wrist"][..., :3].to(torch.uint8).cpu().numpy()
        scene = obs["images"]["scene"][..., :3].to(torch.uint8).cpu().numpy()
        cmd_prev = expert.cmd.clone()
        action = expert.act()  # clean expert target (the label), from the current -- possibly perturbed -- state
        if dart is not None:  # (noise only while the expert's TCP command is still on its way)
            executed = action + dart(expert.phase, torch.linalg.vector_norm(expert.cmd - cmd_prev, dim=-1) > 1e-4)
        else:
            executed = action
        for e in range(N):
            b = buf[e]
            b["state"].append(q[e].cpu().numpy()); b["action"].append(action[e].cpu().numpy())
            b["exec"].append(executed[e].cpu().numpy())
            b["phase"].append(int(expert.phase[e])); b["wrist"].append(wrist[e]); b["scene"].append(scene[e])
        size_now, idx_now = mdp.active_cube_size(u).clone(), mdp.active_cube_idx(u).clone()
        obs, rew, term, trunc, info = env.step(executed)
        frames_total += N
        done_for += (expert.phase == DONE).float() * u.step_dt
        finished = (done_for > 0.4) | term | trunc  # 0.4 s after the retreat, or an env-side reset
        for e in finished.nonzero().flatten().tolist():
            attempts += 1
            ok = bool(done_for[e] > 0.4) and bool(mdp.cube_in_cup(u)[e]) and bool(mdp.cup_upright(u)[e])
            # drop the first frame: its images are the obs from before the reset (the previous episode's end, or the
            # robot's USD zero pose right after the env is built), not the reset state
            b = {k: v[1:] for k, v in buf[e].items()}
            if ok and saved < args.episodes:
                d = os.path.join(args.out, f"ep_{saved:06d}")
                d_final, d = d, os.path.join(args.out, f".tmp_ep_{os.getpid()}")
                os.makedirs(d, exist_ok=True)
                extra = dict(exec_action=np.stack(b["exec"]), action_noise=args.action_noise) if dart is not None else {}
                np.savez_compressed(os.path.join(d, "data.npz"), state=np.stack(b["state"]), action=np.stack(b["action"]),
                                    phase=np.array(b["phase"]), cube_size=float(size_now[e]), cube_idx=int(idx_now[e]),
                                    success=True, fps=round(1 / u.step_dt), **extra)
                h, w = b["wrist"][0].shape[:2]
                for cam in ("wrist", "scene"):
                    vw = Mp4(os.path.join(d, f"{cam}.mp4"), w, h)
                    for f in b[cam]:
                        vw.write(f)
                    vw.close()
                os.rename(d, d_final)  # (an Isaac crash mid-write leaves no half episode for convert_to_lerobot.py)
                if saved < args.preview:
                    if preview is None:
                        preview = Mp4(os.path.join(args.out, "preview.mp4"), 2 * w, h)
                    for fs, fw in zip(b["scene"], b["wrist"]):
                        preview.write(np.concatenate([fs, fw], axis=1))
                saved += 1
                print(f"[rec] saved ep {saved}/{args.episodes} len={len(b['state'])} cube={int(size_now[e] * 1000)}mm "
                      f"(attempts {attempts}, {frames_total / (time.time() - t0):.1f} env-steps/s)", flush=True)
            buf[e] = dict(state=[], action=[], exec=[], phase=[], wrist=[], scene=[])
        if finished.any():
            ids = finished.nonzero().flatten()
            stale = ids[~(term | trunc)[ids]]  # envs the env did NOT auto-reset: reset them ourselves
            if len(stale):
                u._reset_idx(stale)
                u.scene.write_data_to_sim()
                u.sim.forward()
                u.sim.render()
                u.scene.update(u.physics_dt)
                obs = u.observation_manager.compute()
            expert.reset(ids)
            if dart is not None:
                dart.reset(ids)
            done_for[ids] = 0.0
    if preview is not None:
        preview.close()
    el = time.time() - t0
    print(f"[rec] done: {saved} successful / {attempts} attempts in {el:.0f}s ({el / max(saved, 1):.1f}s per saved episode; "
          f"action_noise {args.action_noise} rad, tau {args.noise_tau} s, phases {args.noise_phases if dart is not None else '-'})")
    env.close()

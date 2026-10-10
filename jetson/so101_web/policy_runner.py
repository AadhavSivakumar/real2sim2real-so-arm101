"""Policy deployment for the SO-101 web console: load a policy, run episodes on the arm, label and record them.

One PolicyRunner per app. Threads:
- loader: starts a LocalWorker (policy_worker.py on this machine's GPU) or connects to a remote policy server
  listed in <policy-root>/remotes.json, fetches its info and checks it against this robot;
- job: one motion job at a time -- an episode (optional slow move to the start pose, then the 30 Hz policy
  loop) or a standalone "go to start pose";
- saver: saves a labeled rollout episode (video encoding can take seconds), so labeling never blocks motion;
- monitor: pings an idle backend so a dead worker shows up before the next start.
The runner never touches the serial bus: it reads ctl.snapshot() and hands targets plus a heartbeat to the
Controller, whose control thread slews toward them (rate limit, ramp-in, stall and tracking watchdogs).
Every way an episode ends -- STOP, time limit, fault, worker death -- leaves the arm holding its pose.

Episode lifecycle: start -> [homing] -> running <-> intervention (leader takeover) -> pending (waiting for an
outcome label, or discard) -> labeled: one line in <policy>/evals.jsonl and, when recording, the episode saved to
the dataset plus a line in <dataset>/meta/so101_rollouts.jsonl.
"""
import collections
import datetime
import importlib.util
import itertools
import json
import logging
import math
import os
import queue
import re
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import mock_hw
import recorder
from policy_client import ChunkScheduler, EnsembleScheduler, LocalWorker, PolicyClient, PolicyError

log = logging.getLogger("so101_web.policy")

JOINTS = recorder.JOINTS
CAMERAS = recorder.CAMERAS
# every real (non-dry) run is recorded to rollouts_<policy> unless its start says "record": false
AUTO_RECORD = os.environ.get("SO101_AUTO_RECORD", "1") != "0"
FPS = recorder.FPS
COMMANDS = ("load", "unload", "start", "stop", "home", "takeover", "resume", "outcome", "discard")
DEFAULT_START_POSE = [0.0, -103.0, 95.0, 52.0, 0.0, 20.0]  # the folded rest pose the real demos start from
DEFAULT_TASK = "Put the cube in the cup"
DEFAULT_MAX_SPEED = 90.0
DEFAULT_TIME_LIMIT_S = 40.0
# smoothing: "chunks" = execute chunk after chunk (n_action_steps each, 5-step crossfade at the seams); "ensemble" = ACT
# temporal ensembling: a new chunk every ENSEMBLE_STRIDE ticks, each step the exp(-coeff * n)-weighted mean of every
# prediction made for it (the sidecar may set "smoothing" and "ensemble_coeff")
SMOOTHING = ("chunks", "ensemble")
ENSEMBLE_STRIDE = 2  # ACT on the Jetson: ~35 ms inference, ~38 ms round trip = just over one 33 ms tick
ENSEMBLE_COEFF = 0.01
# only for deterministic chunks: ACT's inference is deterministic (latent z = 0); flow-matching policies (smolvla, pi0,
# pi05, groot) sample fresh noise per chunk, and averaging samples from different modes gives a path none of them proposed
ENSEMBLE_TYPES = ("act",)
HOME_RATE = 30.0  # deg/s for "go to start pose"
TAKEOVER_MAX_GAP_DEG = 12.0  # leader takeover only when the leader is this close to the follower (= app.py)
CAM_SIZE = (640, 480)
NON_CHUNKING = {"diffusion", "vqbet", "tdmpc", "sac", "reward_classifier", "gaussian_actor"}
NEEDS_TRANSFORMERS = {"smolvla", "pi0", "pi05", "pi0_fast", "groot", "eo1", "lawam", "lingbot_va", "molmoact2",
                      "vla_jepa", "wall_x", "xvla", "fastwam", "multi_task_dit"}
LOAD_TIMEOUT_S = 300.0
CHUNK_TIMEOUT_S = 10.0
MAX_ERRORS_IN_ROW = 3  # error replies from the policy before the episode faults
CAMERA_FAULT_S = 0.5  # cameras unusable for this long while a chunk is due -> fault
MODE_SWITCH_S = 0.5  # time the control thread gets to apply a mode switch we requested
CONTROL_STALL_S = 0.25  # the control thread's tick hasn't advanced for this long (follower reads failing?) -> fault
_REQ_IDS = itertools.count(1)  # chunk request ids, unique for the life of the process (never reused across episodes)
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class _End(Exception):
    """The episode ends normally (STOP, time limit, user)."""


class _Fault(Exception):
    """The episode ends because something went wrong; the message is shown as the fault."""


def _refusal(msg):
    """enter_policy() refused: STOP or a pending user mode change ends the episode; anything else is a fault."""
    return (_End if ("STOP" in msg or "pending" in msg) else _Fault)(msg)


class _TickWatch:
    """Faults when the control thread's tick counter stops advancing (it can't read the follower)."""

    def __init__(self):
        self.tick, self.t = None, time.monotonic()

    def check(self, tick, now):
        if tick != self.tick:
            self.tick, self.t = tick, now
        elif now - self.t > CONTROL_STALL_S:
            raise _Fault(f"control loop stalled: no fresh follower reading for {now - self.t:.2f} s")


def _now_iso():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "on", "yes")


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _mtime(p):
    try:
        return p.stat().st_mtime_ns
    except OSError:
        return None


def _append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


_transformers = None


def transformers_available():
    global _transformers
    if _transformers is None:
        _transformers = importlib.util.find_spec("transformers") is not None
    return _transformers


# ------------------------------------------------------------------------------------- policy directories
def resolve_ckpt(d):
    """The directory with config.json: d, d/pretrained_model or d/checkpoints/last/pretrained_model."""
    for c in (d, d / "pretrained_model", d / "checkpoints" / "last" / "pretrained_model"):
        if (c / "config.json").is_file():
            return c
    return None


def read_sidecar(policy_dir, ckpt):
    for p in (policy_dir / "so101_deploy.json", ckpt / "so101_deploy.json"):
        if p.is_file():
            d = _read_json(p)
            return d if d is not None else {"_error": f"{p.name} is not valid JSON"}
    return None


def tally(evals_path, include_mock=None):
    """Labeled outcomes of real (non-dry-run) episodes in an evals.jsonl: total = success + fail. Episodes run in
    mock mode only count in a mock-mode app, so testing never changes a policy's real tally."""
    if include_mock is None:
        include_mock = mock_hw.enabled()
    s = f = 0
    try:
        lines = Path(evals_path).read_text().splitlines() if evals_path else []
    except OSError:
        lines = []
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("dry_run") or (e.get("mock") and not include_mock):
            continue
        s += e.get("outcome") == "success"
        f += e.get("outcome") == "fail"
    return {"success": s, "fail": f, "total": s + f}


def validate_deploy(deploy):
    """Why a so101_deploy.json can't be used, or None."""
    if deploy is None:
        return None
    if not isinstance(deploy, dict):
        return "so101_deploy.json must be a JSON object"
    if "_error" in deploy:
        return deploy["_error"]
    for cam, u in (deploy.get("undistort") or {}).items():
        if cam not in CAMERAS:
            return f"so101_deploy.json undistort: unknown camera '{cam}' (cameras: {', '.join(CAMERAS)})"
        try:
            k, dist = np.asarray(u["K"], float), np.asarray(u["dist"], float)
        except (KeyError, TypeError, ValueError):
            return f"so101_deploy.json undistort.{cam} needs K (9 numbers) and dist"
        if k.size != 9 or dist.size not in (4, 5, 8, 12, 14) or not (np.isfinite(k).all() and np.isfinite(dist).all()):
            return f"so101_deploy.json undistort.{cam}: K needs 9 numbers, dist 4/5/8/12/14"
    for key, feed in (deploy.get("camera_map") or {}).items():
        if feed not in CAMERAS:
            return f"so101_deploy.json camera_map: {key} -> unknown camera '{feed}'"
    sm = deploy.get("smoothing")
    if sm is not None and sm not in SMOOTHING:
        return f"so101_deploy.json smoothing must be one of: {', '.join(SMOOTHING)}"
    ec = deploy.get("ensemble_coeff")
    if ec is not None and (isinstance(ec, bool) or not isinstance(ec, (int, float)) or not math.isfinite(ec)
                           or not 0 <= ec <= 1):
        return "so101_deploy.json ensemble_coeff must be a number in 0-1"
    sp = deploy.get("start_pose")
    if sp is not None:
        try:
            ok = len(sp) == 6 and all(math.isfinite(float(x)) for x in sp)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            return "so101_deploy.json start_pose needs 6 numbers (LeRobot units)"
    for k in ("max_speed", "time_limit_s"):
        v = deploy.get(k)
        if v is not None and not (isinstance(v, (int, float)) and v > 0):
            return f"so101_deploy.json {k} must be a positive number"
    n = deploy.get("n_action_steps")
    if n is not None and not (isinstance(n, int) and n >= 1):
        return "so101_deploy.json n_action_steps must be a positive integer or null"
    return None


def check_policy(ptype, chunking, state_dim, action_dim, image_keys, other_inputs, deploy, local, fps=None):
    """(why_not | None, camera_map {image key the policy takes: camera}) for running a policy on this arm."""
    if not chunking:
        return (f"'{ptype}' is not an action-chunking policy (supported: act, smolvla, pi0, pi05, groot, ...)", {})
    if action_dim != len(JOINTS):
        return f"outputs {action_dim}-D actions; this arm takes {len(JOINTS)}", {}
    if state_dim not in (None, len(JOINTS)):
        return f"expects a {state_dim}-D observation.state; this arm has {len(JOINTS)}", {}
    if other_inputs:
        return f"needs inputs this robot doesn't provide: {', '.join(other_inputs)}", {}
    if fps is not None and abs(float(fps) - FPS) > 0.5:
        return f"trained at {fps:g} fps; policies run at {FPS} fps here", {}
    why = validate_deploy(deploy)
    if why:
        return why, {}
    if (deploy or {}).get("smoothing") == "ensemble" and ptype not in ENSEMBLE_TYPES:
        return f"so101_deploy.json smoothing 'ensemble' is for ACT policies only, not '{ptype}'", {}
    cmap_cfg = (deploy or {}).get("camera_map") or {}
    cmap = {}
    for k in image_keys:
        feed = cmap_cfg.get(k)
        if feed is None and k.startswith("observation.images."):
            feed = k[len("observation.images."):]
        if feed not in CAMERAS:
            return (f"no camera for {k}: add a camera_map to so101_deploy.json "
                    f"(cameras: {', '.join(CAMERAS)})", {})
        cmap[k] = feed
    if local and ptype in NEEDS_TRANSFORMERS and not transformers_available():
        return ("needs the transformers package, which isn't installed here: serve it from a GPU server "
                "(./so101.sh policy-server) and list it in remotes.json"), cmap
    return None, cmap


def describe_policy(d, ckpt):
    """Listing entry for a policy directory, from its JSON files only (never loads weights)."""
    cfg = _read_json(ckpt / "config.json") or {}
    tc = _read_json(ckpt / "train_config.json") or {}
    pre = _read_json(ckpt / "policy_preprocessor.json") or {}
    ptype = cfg.get("type")
    feats = cfg.get("input_features") or {}
    rename = {}
    for st in pre.get("steps", []):
        rm = (st.get("config") or {}).get("rename_map")
        if isinstance(rm, dict):
            rename.update(rm)
    inverse = {v: k for k, v in rename.items()}
    visual = [k for k, v in feats.items() if isinstance(v, dict) and v.get("type") == "VISUAL"]
    image_keys = [inverse.get(k, k) for k in visual]
    state = (feats.get("observation.state") or {}).get("shape")
    others = [k for k in feats if k != "observation.state" and k not in visual]
    act = ((cfg.get("output_features") or {}).get("action") or {}).get("shape")
    deploy = read_sidecar(d, ckpt)
    why, cmap = check_policy(ptype, ptype not in NON_CHUNKING and "chunk_size" in cfg,
                             int(np.prod(state)) if state else None, int(np.prod(act)) if act else None,
                             image_keys, others, deploy, local=True)
    step = (_read_json(ckpt.parent / "training_state" / "training_step.json") or {}).get("step")
    size = 0
    for f in ckpt.iterdir():
        if f.is_file():
            size += f.stat().st_size
    model = ckpt / "model.safetensors"
    created = datetime.datetime.fromtimestamp((model if model.exists() else ckpt / "config.json").stat().st_mtime)
    return {
        "name": d.name, "path": str(d), "ckpt": str(ckpt), "type": ptype, "size_mb": round(size / 1e6, 1),
        "steps": step if step is not None else tc.get("steps"), "dataset": (tc.get("dataset") or {}).get("repo_id"),
        "created": created.isoformat(timespec="seconds"), "chunk_size": cfg.get("chunk_size"),
        "n_action_steps": cfg.get("n_action_steps"), "image_keys": image_keys,
        "deploy": deploy if deploy is None or "_error" not in deploy else None,
        "undistort": bool((deploy or {}).get("undistort")), "runnable": why is None, "why_not": why,
        "tally": tally(d / "evals.jsonl"), "camera_map": cmap,
    }


# ------------------------------------------------------------------------------------------- helpers
class RolloutWriter:
    """Adds frames to the rollout dataset on its own thread, so encoder hiccups never delay the policy loop."""

    def __init__(self, ds, task):
        self.ds, self.task = ds, task
        self.q = queue.Queue(maxsize=FPS * 2)
        self.error = None
        self.added = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="rollout-writer")
        self.thread.start()

    def put(self, state, action, frames):
        """False if the frame could not be queued (writer failed or more than 2 s behind)."""
        if self.error is not None:
            return False
        try:
            self.q.put_nowait((state, action, frames))
            return True
        except queue.Full:
            return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                return
            if self.error is not None:
                continue
            try:
                self.ds.add_frame(recorder.make_frame(*item, self.task))
                self.added += 1
            except Exception as e:
                log.exception("rollout recording failed")
                self.error = e

    def close(self, timeout=60.0):
        try:
            self.q.put(None, timeout=timeout)
        except queue.Full:
            pass
        self.thread.join(timeout)
        return not self.thread.is_alive()


class PolicyViewFeed:
    """MJPEG source with a camera's frames as the loaded policy sees them (undistorted). Encodes only while
    someone watches. Same surface as app.Feed for Handler._stream: touch() and wait_next()."""

    def __init__(self, name, src, get_maps, fps=15):
        self.name, self.src, self.get_maps, self.fps = name, src, get_maps, fps
        self.jpeg = None
        self.seq = 0
        self.cond = threading.Condition()
        self.last_client = 0.0
        threading.Thread(target=self._run, daemon=True, name=f"feed-{name}-policy").start()

    def touch(self):
        self.last_client = time.time()

    def wait_next(self, seq, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout)
            return self.jpeg, self.seq

    def _run(self):
        period = 1.0 / self.fps
        while True:
            t0 = time.time()
            if time.time() - self.last_client > 3.0:
                time.sleep(0.1)
                continue
            try:
                rgb, _ = self.src.get_frame(max_age_ms=1000)
                maps = self.get_maps()
                if maps is not None:
                    rgb = cv2.remap(rgb, maps[0], maps[1], cv2.INTER_LINEAR)
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with self.cond:
                        self.jpeg = jpg.tobytes()
                        self.seq += 1
                        self.cond.notify_all()
            except Exception:
                time.sleep(0.2)
            time.sleep(max(0.0, period - (time.time() - t0)))


class Episode:
    def __init__(self, n, cfg, stops0, policy, evals_path, backend):
        self.n = n
        self.dry_run = cfg["dry_run"]
        self.max_speed = cfg["max_speed"]
        self.time_limit_s = cfg["time_limit_s"]
        self.n_exec = cfg["n_exec"]
        self.smoothing = cfg.get("smoothing", "chunks")
        self.ensemble_coeff, self.ensemble_stride = cfg.get("ensemble_coeff"), cfg.get("ensemble_stride")
        self.task = cfg["task"]
        self.home_first = cfg["home_first"]
        self.dataset = cfg["dataset"]
        self.start_pose = cfg["start_pose"]
        self.stops0 = stops0
        self.policy, self.evals_path, self.backend = policy, evals_path, backend
        self.phase = "homing" if self.home_first and not self.dry_run else "running"
        self.t0 = None
        self.elapsed = 0.0
        self.interventions = []  # [first, end) recorded-frame ranges under leader control
        self.n_interventions = 0
        self.iv_open = None
        self.frames = 0
        self.ds = self.writer = self.quality = self.sched = None
        self.episode_index = None
        self.end_reason = self.fault = None
        self.label_after = None  # (outcome, note) requested while running

    def settings(self):
        return {"max_speed": self.max_speed, "time_limit_s": self.time_limit_s, "n_action_steps": self.n_exec,
                "smoothing": self.smoothing, **({"ensemble_coeff": self.ensemble_coeff, "ensemble_stride":
                self.ensemble_stride} if self.smoothing == "ensemble" else {}), "task": self.task, "home_first": self.home_first, "start_pose": self.start_pose,
                "backend": self.backend}


# ------------------------------------------------------------------------------------------- the runner
class PolicyRunner:
    def __init__(self, ctl, feeds, rec, policy_root, data_root):
        self.ctl, self.feeds, self.rec = ctl, feeds, rec
        self.root = Path(policy_root).expanduser()
        self.data_root = Path(data_root).expanduser()
        self.lock = threading.RLock()
        # the loaded policy
        self.load_state = "none"  # none | loading | ready | error
        self.name = self.backend = self.info = None
        self.client = self.worker = None
        self._loading_worker = None
        self._loader = None
        self._load_gen = 0
        self.deploy, self.camera_map, self.undistort = {}, {}, {}
        self.start_pose = list(DEFAULT_START_POSE)
        self.evals_path = None
        self.backend_ok = True
        self.backend_error = None
        self.load_error = None
        self.message = "No policy loaded"
        self.fault = None
        # jobs and episodes
        self.job = None
        self.job_kind = None  # None | "episode" | "home"
        self._cmds = collections.deque()
        self.ep = None  # current or last episode
        self.pending = None  # ended episode waiting for its label
        self._saver = None
        self.saving = False
        self.episode_count = 0
        self.tally = {"success": 0, "fail": 0, "total": 0}
        self.last_outcome = None
        self.record_saved = {}
        self.target = None
        self.live = self._fresh_stats()
        self._rtt_guess = 0.15
        self._views = {}
        self._cache = {}
        self._stopping = False
        threading.Thread(target=self._monitor, daemon=True, name="policy-monitor").start()

    @staticmethod
    def _fresh_stats():
        return {"infer_last": None, "infer_sum": 0.0, "infer_n": 0, "infer_max": None, "rtt_ms": None,
                "chunks": 0, "queue": 0, "hz": 0.0, "starved": 0, "late": 0, "dropped": 0, "errors": 0,
                "last_error": None}

    # ------------------------------------------------------------------ status / listing (HTTP threads)
    def _phase(self):
        if self.job_kind == "home":
            return "homing"
        if self.job_kind == "episode":
            return self.ep.phase
        if self.pending is not None:
            return "pending"
        return self.load_state

    def status(self):
        with self.lock:
            ep, lv = self.ep, self.live
            rec = None
            if ep is not None and ep.dataset:
                rec = {"dataset": ep.dataset, "saved": self.record_saved.get(ep.dataset, 0),
                       "pending": self.pending is ep or (self.job_kind == "episode" and ep.frames > 0),
                       "frames": ep.frames}
            n = lv["infer_n"]
            return {
                "phase": self._phase(), "name": self.name, "backend": self.backend, "info": self.info,
                "dry_run": ep.dry_run if ep else None, "episode": ep.n if ep else 0,
                "elapsed_s": round(ep.elapsed, 1) if ep else 0.0,
                "time_limit_s": ep.time_limit_s if ep else None, "max_speed": ep.max_speed if ep else None,
                "infer_ms": {"last": lv["infer_last"], "avg": round(lv["infer_sum"] / n, 1) if n else None,
                             "max": lv["infer_max"]},
                "chunks": lv["chunks"], "queue": lv["queue"], "hz": round(lv["hz"], 1), "starved": lv["starved"],
                "target": dict(self.target) if self.target else None,
                "message": self.message, "fault": self.fault, "record": rec, "tally": dict(self.tally),
                "last_outcome": self.last_outcome,
                # extras
                "rtt_ms": lv["rtt_ms"], "late": lv["late"], "dropped": lv["dropped"], "errors": lv["errors"],
                "interventions": ep.n_interventions if ep else 0, "start_pose": list(self.start_pose),
                "n_action_steps": ep.n_exec if ep else None, "task": ep.task if ep else None,
                "smoothing": ep.smoothing if ep else None,
                "auto_record": self._auto_dataset(),  # where a real run records when no dataset is named
                "saving": self.saving, "backend_ok": self.backend_ok, "backend_error": self.backend_error,
                "deploy": self.deploy or None, "undistort": sorted(self.undistort),
                "load_state": self.load_state,  # the policy's own state; phase shows a job / pending episode first
                "job": self.job_kind,  # "episode" | "home" (stand-alone start-pose move: episode fields are the last one's)
            }

    def _remotes(self):
        r = _read_json(self.root / "remotes.json")
        if not isinstance(r, list):
            return []
        return [x for x in r if isinstance(x, dict) and x.get("name") and x.get("address")]

    def _describe(self, d):
        ckpt = resolve_ckpt(d)
        if ckpt is None:
            return None
        files = (ckpt / "config.json", ckpt / "train_config.json", ckpt / "policy_preprocessor.json",
                 ckpt / "model.safetensors", d / "so101_deploy.json", ckpt / "so101_deploy.json", d / "evals.jsonl")
        key = tuple(_mtime(f) for f in files)
        hit = self._cache.get(str(d))
        if hit is not None and hit[0] == key:
            return hit[1]
        try:
            entry = describe_policy(d, ckpt)
        except Exception as e:  # unreadable directory: list it as not runnable rather than failing the listing
            entry = {"name": d.name, "path": str(d), "ckpt": str(ckpt), "type": None, "size_mb": None,
                     "steps": None, "dataset": None, "created": None, "chunk_size": None, "n_action_steps": None,
                     "image_keys": [], "deploy": None, "undistort": False, "runnable": False,
                     "why_not": f"unreadable: {e}", "tally": tally(d / "evals.jsonl"), "camera_map": {}}
        self._cache[str(d)] = (key, entry)
        return entry

    def list_policies(self):
        out = []
        try:
            dirs = sorted(p for p in self.root.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))
        except OSError:
            dirs = []
        for d in dirs:
            e = self._describe(d)
            if e is not None:
                out.append(e)
        return {"root": str(self.root), "policies": out,
                "remotes": [{"name": r["name"], "address": r["address"]} for r in self._remotes()]}

    def view_feed(self, cam):
        """The ?view=policy stream for a camera, or None when it would equal the raw stream."""
        with self.lock:
            if cam not in self.undistort:
                return None
            f = self._views.get(cam)
            src = self.feeds.get(cam)
            if f is None and hasattr(src, "get_frame"):
                f = self._views[cam] = PolicyViewFeed(cam, src, lambda: self.undistort.get(cam))
            return f

    def _auto_dataset(self):
        """Rollout dataset a real run records to when the start names none (None: auto-recording is off)."""
        if not AUTO_RECORD or not self.name:
            return None
        name = ("rollouts_" + re.sub(r"[^A-Za-z0-9_-]", "_", self.name))[:64]
        return name if recorder.NAME_RE.match(name) else None

    def recording_blocker(self):
        """Why a teleop recording session can't start now, or None."""
        with self.lock:
            if self.job_kind == "episode":
                return "a policy episode is running: stop it first"
            if self.job_kind == "home":
                return "the arm is moving to the start pose: wait or press STOP"
            if self.saving:
                return "saving a policy rollout: try again in a moment"
            if self.pending is not None and self.pending.ds is not None:
                return "a policy rollout is waiting for its outcome label: label or discard it first"
        return None

    # ------------------------------------------------------------------ commands (HTTP threads)
    def command(self, action, body):
        return getattr(self, "_cmd_" + action)(body)

    def _busy_reason(self):
        if self.job_kind == "episode":
            return "an episode is running: stop it first"
        if self.job_kind == "home":
            return "the arm is moving to the start pose: wait or press STOP"
        return None

    def _cmd_load(self, body):
        with self.lock:
            busy = self._busy_reason()
            if busy:
                return False, busy
            spec, err = self._resolve(body)
            if err:
                return False, err
            self._load_gen += 1
            gen = self._load_gen
            old = (self.client, self.worker)
            self.client = self.worker = None
            self._reset_loaded()
            self.load_state, self.name, self.backend = "loading", spec["name"], spec["backend"]
            self.message = f"Loading {spec['name']}..."
            self._loader = threading.Thread(target=self._load, args=(gen, spec, old), daemon=True, name="policy-load")
            self._loader.start()
        return True, f"loading {spec['name']}"

    def _cmd_unload(self, body):
        with self.lock:
            busy = self._busy_reason()
            if busy:
                return False, busy
            self._load_gen += 1  # cancels a load in progress
            old = (self.client, self.worker)
            self.client = self.worker = None
            self._reset_loaded()
            self.load_state, self.name, self.backend = "none", None, None
            self.message = "Policy unloaded"
        threading.Thread(target=self._close_backend, args=old, daemon=True, name="policy-unload").start()
        return True, "unloaded"

    def _cmd_start(self, body):
        with self.lock:
            busy = self._busy_reason()
            if busy:
                return False, busy
            if self.load_state == "loading":
                return False, "the policy is still loading"
            if self.load_state == "error":
                return False, self.load_error or self.message
            if self.load_state != "ready" or self.client is None:
                return False, "load a policy first"
            if self.worker is not None and not self.worker.alive():
                return False, "the policy worker has exited: load the policy again"
            if self.rec.active():
                return False, "a teleop recording session is active: stop it first"
            if self.saving:
                return False, "saving the last episode: try again in a moment"
            cfg, err = self._episode_settings(body)
            if err:
                return False, err
            _, _, _, mode, _ = self.ctl.snapshot()
            if mode == "limp":
                return False, "the arm is limp: switch to Manual (torque on) first"
            note = ""
            for feed in sorted(set(self.camera_map.values()) | (set(CAMERAS) if cfg["dataset"] else set())):
                try:
                    self.feeds[feed].get_frame()
                except Exception as e:
                    if cfg["auto_record"] and feed not in self.camera_map.values():
                        cfg["dataset"], note = None, f" (not recording: {feed} camera is not ready)"
                        continue
                    return False, f"{feed} camera is not ready: {e}"
            stops0 = self.ctl.stop_count  # any STOP from now on cancels this episode before it moves
            self.episode_count += 1
            ep = Episode(self.episode_count, cfg, stops0, self.name, self.evals_path, self.backend)
            prev, self.pending = self.pending, None
            self.ep, self.fault, self.live = ep, None, self._fresh_stats()
            self.message = f"Episode {ep.n}{' (dry run)' if ep.dry_run else ''}: starting{note}"
            self._start_job("episode", self._episode_job, ep, prev)
        return True, f"episode {ep.n} started{note}"

    def _cmd_stop(self, body):
        with self.lock:
            if self.job_kind in ("episode", "home"):
                self._cmds.append("stop")
                return True, "stopping"
        return True, "nothing is running"

    def _cmd_home(self, body):
        pose = body.get("pose")
        if pose is not None:
            try:
                pose = [float(x) for x in pose]
                if len(pose) != 6 or not all(math.isfinite(x) for x in pose):
                    raise ValueError
            except (TypeError, ValueError):
                return False, "pose needs 6 numbers (LeRobot units)"
        with self.lock:
            busy = self._busy_reason()
            if busy:
                return False, busy
            if self.rec.active():
                return False, "a teleop recording session is active: stop it first"
            _, _, _, mode, _ = self.ctl.snapshot()
            if mode == "limp":
                return False, "the arm is limp: switch to Manual (torque on) first"
            pose = pose or list(self.start_pose)
            self.fault = None
            self.message = "Moving to the start pose"
            self._start_job("home", self._home_job, pose, self.ctl.stop_count)
        return True, "moving to the start pose"

    def _cmd_takeover(self, body):
        with self.lock:
            if self.job_kind != "episode" or self.ep.phase != "running":
                return False, "no running episode to take over"
            if self.ep.dry_run:
                return False, "dry run: the policy isn't moving the arm (use the Teleop mode button)"
            if self.ctl.leader is None:
                return False, "leader arm is not connected"
            gap, why = self.ctl.takeover_gap()
            if gap is None:
                return False, why
            if gap > TAKEOVER_MAX_GAP_DEG:
                return False, (f"bring the leader to the follower's pose first: {why} is {gap:.0f} deg away "
                               f"(max {TAKEOVER_MAX_GAP_DEG:.0f})")
            self._cmds.append("takeover")
        return True, "taking over with the leader arm"

    def _cmd_resume(self, body):
        with self.lock:
            if self.job_kind != "episode" or self.ep.phase != "intervention":
                return False, "not in a takeover"
            self._cmds.append("resume")
        return True, "resuming the policy"

    def _cmd_outcome(self, body):
        outcome = body.get("outcome")
        if outcome not in ("success", "fail", "none"):
            return False, "outcome must be success, fail or none"
        note = str(body.get("note") or "")[:1000]
        with self.lock:
            if self.job_kind == "episode":
                self.ep.label_after = (outcome, note)
                self._cmds.append("end")
                return True, f"ending the episode ({outcome})"
            if self.saving:
                return False, "still saving the last episode"
            ep = self.pending
            if ep is None:
                return False, "no episode to label"
            self._start_saver(self._label, ep, outcome, note)
        return True, f"labeled {outcome}"

    def _cmd_discard(self, body):
        with self.lock:
            if self.job_kind == "episode":
                self.ep.label_after = ("discard", "")
                self._cmds.append("end")
                return True, "ending and discarding the episode"
            if self.saving:
                return False, "still saving the last episode"
            ep = self.pending
            if ep is None:
                return False, "no pending episode"
            self._start_saver(self._discard, ep)
        return True, "discarded"

    # ------------------------------------------------------------------ loading
    def _resolve(self, body):
        remote = body.get("remote")
        if remote:
            r = next((x for x in self._remotes() if x["name"] == remote), None)
            if r is None:
                return None, f"no remote '{remote}' in {self.root / 'remotes.json'}"
            key = r.get("authkey") or os.environ.get("SO101_POLICY_KEY") or "so101"
            return {"kind": "remote", "name": str(r["name"]), "address": str(r["address"]),
                    "authkey": str(key), "backend": f"remote:{r['address']}"}, None
        name = str(body.get("name") or "")
        if not NAME_RE.match(name):
            return None, "give a policy name (or a remote)"
        d = self.root / name
        entry = self._describe(d) if d.is_dir() else None
        if entry is None:
            return None, f"no policy '{name}' in {self.root}"
        if not entry["runnable"]:
            return None, f"{name} can't run here: {entry['why_not']}"
        return {"kind": "local", "name": name, "dir": d, "ckpt": entry["ckpt"], "backend": "local"}, None

    def _reset_loaded(self):
        self.info = None
        self.deploy, self.camera_map, self.undistort = {}, {}, {}
        self.start_pose = list(DEFAULT_START_POSE)
        self.evals_path = None
        self.backend_ok = True
        self.backend_error = None
        self.load_error = None
        self.tally = {"success": 0, "fail": 0, "total": 0}
        self.last_outcome = None
        self.fault = None

    @staticmethod
    def _close_backend(client, worker):
        if client is not None:
            client.close()
        if worker is not None:
            worker.stop()

    def _load(self, gen, spec, old):
        self._close_backend(*old)
        client = worker = None
        cancelled = lambda: gen != self._load_gen or self._stopping  # noqa: E731
        try:
            if spec["kind"] == "local":
                worker = LocalWorker(spec["ckpt"], device="auto").start()
                self._loading_worker = worker
                worker.wait_ready(LOAD_TIMEOUT_S, cancel=cancelled)
                client = PolicyClient(worker.address, worker.authkey, chunk_timeout=CHUNK_TIMEOUT_S)
            else:
                client = PolicyClient(spec["address"], spec["authkey"], chunk_timeout=CHUNK_TIMEOUT_S)
            info = client.info(timeout=20.0)
            setup = self._prepare(info, spec)
            with self.lock:
                if cancelled():
                    raise PolicyError("cancelled")
                self.client, self.worker = client, worker
                self.info = info
                self.deploy, self.camera_map, self.undistort, self.start_pose, self.evals_path = setup
                self.tally = tally(self.evals_path)
                self.load_state = "ready"
                where = f"{info.get('device')} on {info.get('host')}" if spec["kind"] == "remote" else info.get("device")
                if info.get("precision"):
                    where += f" {info['precision']}"
                notes = [f"undistorted {', '.join(sorted(self.undistort))}"] if self.undistort else []
                sizes = {tuple(v) for v in (info.get("image_sizes") or {}).values()}
                if sizes and sizes != {(CAM_SIZE[1], CAM_SIZE[0])}:
                    notes.append(f"frames resized to {', '.join(f'{w}x{h}' for h, w in sorted(sizes))}")
                self.message = (f"Ready: {info.get('name')} ({info.get('type')}, {where}, warm-up "
                                f"{info.get('warmup_ms', 0):.0f} ms{'; ' + '; '.join(notes) if notes else ''})")
            log.info("policy %s loaded: %s", spec["name"], self.message)
        except Exception as e:
            self._close_backend(client, worker)
            with self.lock:
                if not cancelled():
                    if not isinstance(e, PolicyError):
                        log.exception("loading %s failed", spec["name"])
                    self.load_state = "error"
                    self.message = self.load_error = f"Loading {spec['name']} failed: {e}"
        finally:
            self._loading_worker = None

    def _prepare(self, info, spec):
        """Check a loaded policy against this robot -> (deploy, camera_map, undistort maps, start pose, evals)."""
        deploy = read_sidecar(spec["dir"], Path(spec["ckpt"])) if spec["kind"] == "local" else info.get("deploy")
        why, cmap = check_policy(info.get("type"), bool(info.get("chunk_size")), info.get("state_dim"),
                                 info.get("action_dim"), list(info.get("image_keys") or []),
                                 list(info.get("other_inputs") or []), deploy, local=False, fps=info.get("fps"))
        if why:
            raise PolicyError(why)
        deploy = deploy or {}
        maps = {}
        for cam, u in (deploy.get("undistort") or {}).items():
            k = np.asarray(u["K"], np.float64).reshape(3, 3)
            dist = np.asarray(u["dist"], np.float64).reshape(-1)
            maps[cam] = cv2.initUndistortRectifyMap(k, dist, None, k, CAM_SIZE, cv2.CV_16SC2)
        pose = [float(x) for x in deploy.get("start_pose") or DEFAULT_START_POSE]
        if spec["kind"] == "local":
            evals = spec["dir"] / "evals.jsonl"
        else:
            evals = self.root / "_remote" / re.sub(r"[^A-Za-z0-9._-]", "_", spec["name"]) / "evals.jsonl"
        return deploy, cmap, maps, pose, evals

    def _episode_settings(self, body):
        d, info = self.deploy or {}, self.info or {}
        dry = _bool(body.get("dry_run"), True)  # a start that doesn't say otherwise never moves the arm
        dataset = None
        rec = body.get("record")  # {"dataset": name}; null/omitted = rollouts_<policy>; false = don't record
        if rec is not False and rec:
            if not isinstance(rec, dict):
                return None, "record must be {\"dataset\": name} or null"
            dataset = str(rec.get("dataset") or "").strip() or None
            if dataset and dry:
                return None, "recording rollouts needs a real run: turn off Dry run"
            if dataset and not recorder.NAME_RE.match(dataset):
                return None, "dataset name: letters, digits, '_' and '-' only"
        auto = dataset is None and rec is not False and not dry
        if auto:
            dataset = self._auto_dataset()
        chunk = int(info.get("chunk_size") or 0)
        smoothing = str(body.get("smoothing") or d.get("smoothing") or "chunks")
        if smoothing not in SMOOTHING:
            return None, f"smoothing must be one of: {', '.join(SMOOTHING)}"
        if smoothing == "ensemble" and info.get("type") not in ENSEMBLE_TYPES:
            return None, f"Ensemble smoothing is for ACT policies only ({info.get('type')} samples a new chunk each time)"
        try:
            max_speed = float(body.get("max_speed") or d.get("max_speed") or DEFAULT_MAX_SPEED)
            limit = float(body.get("time_limit_s") or d.get("time_limit_s") or DEFAULT_TIME_LIMIT_S)
            n = body.get("n_action_steps")
            if smoothing == "ensemble":
                n = None  # ensembling uses every action of every chunk
            elif n in (None, ""):
                n = int(d.get("n_action_steps") or info.get("n_action_steps") or chunk)
            else:
                n = int(n)
        except (TypeError, ValueError):
            return None, "max speed, time limit and chunk steps must be numbers"
        if not (math.isfinite(max_speed) and 1 <= max_speed <= 300):
            return None, "max speed must be 1-300 deg/s"
        if not (math.isfinite(limit) and 1 <= limit <= 600):
            return None, "time limit must be 1-600 s"
        if n is not None and not 1 <= n <= max(chunk, 1):
            return None, f"chunk steps must be 1-{chunk} (blank = as trained)"
        ec = d.get("ensemble_coeff")
        coeff = ENSEMBLE_COEFF if ec is None else float(ec)  # 0 (uniform average) is a valid choice
        task = " ".join(str(body.get("task") or d.get("task") or DEFAULT_TASK).split())[:500]
        return {"dry_run": dry, "max_speed": max_speed, "time_limit_s": limit, "n_exec": n, "task": task,
                "smoothing": smoothing, "ensemble_coeff": coeff if smoothing == "ensemble" else None,
                "ensemble_stride": ENSEMBLE_STRIDE if smoothing == "ensemble" else None,
                "home_first": _bool(body.get("home_first"), True), "dataset": dataset,
                "auto_record": auto and dataset is not None,
                "start_pose": list(self.start_pose)}, None

    # ------------------------------------------------------------------ jobs
    def _start_job(self, kind, fn, *args):
        self.job_kind = kind
        self._cmds.clear()
        self.job = threading.Thread(target=fn, args=args, daemon=True, name=f"policy-{kind}")
        self.job.start()

    def _start_saver(self, fn, *args):
        self.saving = True
        self._saver = threading.Thread(target=self._saver_run, args=(fn,) + args, daemon=True, name="policy-save")
        self._saver.start()

    def _saver_run(self, fn, *args):
        try:
            fn(*args)
        except Exception as e:
            log.exception("saving the episode failed")
            with self.lock:
                self.message = f"Saving failed: {e}"
        finally:
            with self.lock:
                self.saving = False

    def _backend_note(self):
        """Keeps a broken backend visible after later messages (labels, discards) replace the status line."""
        if self.load_state == "error" and self.load_error:
            return ". " + self.load_error
        if self.load_state == "ready" and not self.backend_ok and self.backend_error:
            return f". {self.backend_error}"
        return ""

    def _take_cmd(self):
        with self.lock:
            return self._cmds.popleft() if self._cmds else None

    def _check_stop_cmd(self):
        """Before taking control of the arm: honour a stop / end that arrived during the setup."""
        with self.lock:
            if "stop" in self._cmds or "end" in self._cmds:
                raise _End("Stopped" if "stop" in self._cmds else "Ended for labeling")

    def _set(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def _hold(self, also_teleop=False):
        """Hand the arm back to a manual hold and wait (briefly) until the control thread has applied it, so the
        phase we publish next never shows while the policy still has the arm."""
        self.ctl.leave_policy(also_teleop=also_teleop)
        modes = ("policy", "teleop") if also_teleop else ("policy",)
        t_end = time.monotonic() + 0.25
        while self.ctl.snapshot()[3] in modes:
            if time.monotonic() > t_end:
                log.warning("controller did not leave policy mode within 0.25 s: pressing STOP")
                self.ctl.stop()  # last resort: bumps stop_count and forces a manual hold request
                return
            time.sleep(0.005)

    def _home_job(self, pose, stops0):
        try:
            self._home_loop(pose, stops0)
            self._set(message="At the start pose (holding)")
        except _End as e:
            self._set(message=str(e))
        except _Fault as e:
            self._set(message=f"Go to start pose: {e}", fault=str(e))
        except Exception as e:
            log.exception("homing failed")
            self._set(message=f"Go to start pose failed: {e}", fault=f"internal error: {e}")
        finally:
            self._hold()
            with self.lock:
                self.target = None
                self.job_kind = None

    def _home_loop(self, pose, stops0, ep=None):
        """Slow move to `pose` (HOME_RATE) in policy mode; returns when there. Raises _End / _Fault."""
        ctl = self.ctl
        target = ctl._clamp_all(dict(zip(JOINTS, pose)))
        self._check_stop_cmd()
        ok, msg = ctl.enter_policy(stops0, HOME_RATE, target)
        if not ok:
            raise _refusal(msg)
        with self.lock:
            self.target = target
        _, pos, _, _, _ = ctl.snapshot()
        t0 = time.monotonic()
        timeout = max(abs(target[j] - pos[j]) for j in JOINTS) / HOME_RATE + 5.0
        entered = False
        t_next = t0
        watch = _TickWatch()
        while True:
            if ctl.stop_count != stops0:
                raise _End("STOP pressed: holding")
            ctl.set_policy_target(target, HOME_RATE)
            c = self._take_cmd()
            if c in ("stop", "end"):
                raise _End("Stopped")
            if ctl.stop_count != stops0:
                raise _End("STOP pressed: holding")
            tick, pos, held, mode, _ = ctl.snapshot()
            now = time.monotonic()
            watch.check(tick, now)
            if mode == "policy":
                entered = True
            elif entered or now - t0 > MODE_SWITCH_S:
                if ctl.policy_fault:
                    raise _Fault(ctl.policy_fault)
                raise _End(f"Control mode changed to {mode}")
            if (entered and held is not None and all(abs(held[j] - target[j]) < 0.5 for j in JOINTS)
                    and all(abs(pos[j] - target[j]) < 3.0 for j in JOINTS[:-1])):
                return
            if now - t0 > timeout:
                far = max(JOINTS[:-1], key=lambda j: abs(pos[j] - target[j]))
                raise _Fault(f"could not reach the start pose in {timeout:.0f} s ({far} is "
                             f"{abs(pos[far] - target[far]):.0f} deg off)")
            t_next += 1.0 / FPS
            time.sleep(max(0.0, t_next - time.monotonic()))

    def _episode_job(self, ep, prev):
        try:
            if prev is not None:  # a new start saves the previous episode unlabeled
                self._set(message="Saving the previous episode (unlabeled)...")
                self._label(prev, "none", "")
            if self._saver is not None:
                self._saver.join()
            if ep.dataset:
                self._set(message=f"Opening dataset {ep.dataset}...")
                try:
                    ep.ds = recorder.open_dataset(self.data_root, ep.dataset)
                except Exception as e:
                    raise _Fault(f"cannot open dataset {ep.dataset}: {e}") from e
            if self.ctl.stop_count != ep.stops0:
                raise _End("STOP pressed: episode cancelled")
            if ep.phase == "homing":
                self._set(message=f"Episode {ep.n}: moving to the start pose")
                self._home_loop(ep.start_pose, ep.stops0, ep)
            ep.end_reason = self._run_loop(ep)
        except _End as e:
            ep.end_reason = str(e)
        except _Fault as e:
            ep.fault = str(e)
        except Exception as e:
            log.exception("policy episode failed")
            ep.fault = f"internal error: {type(e).__name__}: {e}"
        finally:
            self._finish_episode(ep)

    def _observation(self, pos, raw):
        obs = {"observation.state": np.array([pos[j] for j in JOINTS], dtype=np.float32)}
        for key, feed in self.camera_map.items():
            img = raw[feed]
            maps = self.undistort.get(feed)
            if maps is not None:
                img = cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)
            obs[key] = img
        return obs

    def _run_loop(self, ep):
        """The 30 Hz policy loop. Returns why the episode ended; raises _Fault on faults."""
        ctl, client = self.ctl, self.client
        lv = self.live
        if ep.smoothing == "ensemble":  # a fresh chunk every ENSEMBLE_STRIDE ticks, overlapping predictions averaged
            sched = ep.sched = EnsembleScheduler(stride=ep.ensemble_stride, coeff=ep.ensemble_coeff, fps=FPS,
                                                 rtt_guess_s=self._rtt_guess)
        else:
            sched = ep.sched = ChunkScheduler(ep.n_exec, fps=FPS, rtt_guess_s=self._rtt_guess)
        client.cancel()
        self._check_stop_cmd()
        if ep.dry_run:
            _, _, _, mode, _ = ctl.snapshot()
            if mode == "teleop":
                ctl.request_mode("manual")  # dry run: the arm holds while the policy only watches
        else:
            ok, msg = ctl.enter_policy(ep.stops0, ep.max_speed, None)
            if not ok:
                raise _refusal(msg)
        if ep.ds is not None:
            ep.writer = RolloutWriter(ep.ds, ep.task)
            ep.quality = recorder.EpisodeQuality()
        client.submit({"cmd": "reset"})
        need_cams = sorted(set(self.camera_map.values()) | (set(CAMERAS) if ep.writer else set()))
        expect, expect_t, expect_seen = ("manual" if ep.dry_run else "policy"), time.monotonic(), False
        errors_in_row = 0
        cam_fail_since = None
        period = 1.0 / FPS
        t_next = ep.t0 = time.monotonic()
        last = None
        watch = _TickWatch()
        with self.lock:
            ep.phase = "running"
            self.message = f"Episode {ep.n}{' (dry run)' if ep.dry_run else ''}: running"
        while True:
            now = time.monotonic()
            # ---- STOP first: a STOP that raced with a command queued this tick always wins
            if ctl.stop_count != ep.stops0:
                return "STOP pressed: holding"
            # ---- user commands
            c = self._take_cmd()
            if c == "stop":
                return "Stopped"
            if c == "end":
                return "Ended for labeling"
            if c == "takeover" and ep.phase == "running":
                ok, msg = ctl.request_takeover(ep.stops0)
                if ok:
                    sched.begin()
                    client.cancel()
                    ep.n_interventions += 1
                    ep.iv_open = ep.frames
                    expect, expect_t, expect_seen = "teleop", now, False
                    with self.lock:
                        ep.phase = "intervention"
                        self.target = None
                        self.message = f"Episode {ep.n}: leader takeover (recording continues)"
                else:
                    self._set(message=f"Takeover failed: {msg}")
            elif c == "resume" and ep.phase == "intervention":
                ok, msg = ctl.enter_policy(ep.stops0, ep.max_speed, None)
                if not ok:
                    raise _refusal(msg)
                self._close_intervention(ep)
                sched.begin()
                client.cancel()
                client.submit({"cmd": "reset"})
                expect, expect_t, expect_seen = "policy", now, False
                with self.lock:
                    ep.phase = "running"
                    self.message = f"Episode {ep.n}: policy resumed"
            # ---- safety: STOP always wins, mode changes end the episode, the backend must be alive
            if ctl.stop_count != ep.stops0:
                return "STOP pressed: holding"
            tick, pos, held, mode, _ = ctl.snapshot()
            watch.check(tick, now)
            if not ep.dry_run:
                if mode == expect:
                    expect_seen = True
                elif expect_seen or now - expect_t > MODE_SWITCH_S:
                    if expect == "policy" and ctl.policy_fault:
                        raise _Fault(ctl.policy_fault)
                    return f"Control mode changed to {mode}"
            if self.worker is not None and not self.worker.alive():
                raise _Fault(f"policy worker exited (code {self.worker.returncode})")
            # ---- replies from the policy
            for msg, reply, rtt in client.results():
                if msg.get("cmd") == "reset":
                    if "error" in reply:
                        raise _Fault(f"policy reset failed: {reply['error']}")
                    continue
                if "error" in reply:
                    if reply.get("transport"):
                        raise _Fault(f"lost the policy backend: {reply['error']}")
                    errors_in_row += 1
                    sched.failed(msg.get("id"))
                    with self.lock:
                        lv["errors"] += 1
                        lv["last_error"] = reply["error"]
                    if errors_in_row >= MAX_ERRORS_IN_ROW:
                        raise _Fault(f"policy failed {errors_in_row} times in a row: {reply['error']}")
                    continue
                acts = np.asarray(reply.get("actions"), dtype=np.float32)
                if acts.ndim != 2 or acts.shape[1] != len(JOINTS) or not np.isfinite(acts).all():
                    raise _Fault(f"policy returned unusable actions (shape {acts.shape})")
                errors_in_row = 0
                if sched.receive(msg.get("id"), acts) == "ok":
                    ms = float(reply.get("infer_ms") or 0.0)
                    with self.lock:
                        lv["infer_last"] = round(ms, 1)
                        lv["infer_sum"] += ms
                        lv["infer_n"] += 1
                        lv["infer_max"] = round(max(lv["infer_max"] or 0.0, ms), 1)
                        lv["rtt_ms"] = round(sched.ema_rtt * 1000, 1)
            # ---- act: this tick's action (None = hold until the first chunk), plus the heartbeat
            if ep.phase == "running":
                a = sched.tick()
                if sched.fault:
                    raise _Fault(sched.fault)
                target = None if a is None else {j: float(v) for j, v in zip(JOINTS, a)}
                if not ep.dry_run:
                    ctl.set_policy_target(target, reramp=sched.take_recovered())
                with self.lock:
                    self.target = target
            elif not ep.dry_run:
                ctl.set_policy_target(None)  # heartbeat until the controller has switched to teleop
            # ---- frames (one capture per tick, shared by the next request and the recording)
            want = ep.phase == "running" and sched.want_request()
            raw, stamps = {}, {}
            if want or ep.writer is not None:
                try:
                    for cam in need_cams:
                        raw[cam], stamps[cam] = self.feeds[cam].get_frame()
                    cam_fail_since = None
                except Exception as e:
                    raw = None
                    cam_fail_since = cam_fail_since or now
                    if now - cam_fail_since > CAMERA_FAULT_S:
                        raise _Fault(f"camera: {e}")
            # ---- request the next chunk with the observation of this step
            if want and raw:
                rid = next(_REQ_IDS)
                client.submit({"cmd": "chunk", "id": rid, "task": ep.task, "obs": self._observation(pos, raw)})
                sched.sent(rid)
            # ---- record (raw frames; state/action as the controller saw/sent them this tick)
            if ep.writer is not None:
                if raw and held is not None and ep.writer.put(pos, held, {c: raw[c] for c in CAMERAS}):
                    ep.frames += 1
                    ep.quality.frame({c: stamps[c] for c in CAMERAS}, now)
                else:
                    ep.quality.late += 1
            # ---- time limit, stats, pacing
            ep.elapsed = now - ep.t0
            if ep.elapsed >= ep.time_limit_s:
                return "Time limit reached"
            with self.lock:
                lv["queue"], lv["chunks"], lv["starved"] = sched.remaining, sched.chunks, sched.starved
                lv["dropped"] = sched.dropped
                if last is not None and now > last:
                    lv["hz"] = 0.9 * lv["hz"] + 0.1 / (now - last) if lv["hz"] else 1.0 / (now - last)
                last = now
            t_next += period
            behind = time.monotonic() - t_next
            if behind > period:  # more than a tick behind: skip ahead rather than burst
                with self.lock:
                    lv["late"] += int(behind / period)
                t_next = time.monotonic()
            time.sleep(max(0.0, t_next - time.monotonic()))

    @staticmethod
    def _close_intervention(ep):
        if ep.iv_open is not None:
            if ep.frames > ep.iv_open:
                ep.interventions.append([ep.iv_open, ep.frames])
            ep.iv_open = None

    def _finish_episode(self, ep):
        """Hold the arm, stop talking to the policy, flush the recording; the episode becomes pending."""
        if not ep.dry_run:
            self._hold(also_teleop=ep.phase == "intervention")
        if self.client is not None:
            self.client.cancel()
        self._close_intervention(ep)
        if ep.writer is not None:
            if not ep.writer.close():
                ep.fault = ep.fault or "rollout recording did not finish writing frames"
        started = ep.t0 is not None  # the policy loop ran; otherwise it was cancelled during setup / homing
        if not started and ep.ds is not None:
            try:
                ep.ds.finalize()
            except Exception:
                log.exception("closing the rollout dataset failed")
            ep.ds = None
        if ep.sched is not None and ep.sched.chunks > 1:
            self._rtt_guess = ep.sched.ema_rtt  # the next episode's first prefetch lead
        dead = self.worker is not None and not self.worker.alive()
        with self.lock:
            self.target = None
            self.job_kind = None
            self._cmds.clear()
            if ep.fault:
                self.fault = ep.fault
            if not started:
                self.message = f"Episode {ep.n} cancelled before the policy ran: {ep.fault or ep.end_reason}"
            elif ep.fault:
                self.message = f"Episode {ep.n} stopped by a fault: {ep.fault}. Holding."
            else:
                self.message = f"Episode {ep.n}: {ep.end_reason or 'ended'} ({ep.elapsed:.1f} s). Label it."
            if started:
                self.pending = ep
            if dead:
                self.load_state = "error"
                self.load_error = f"The policy worker exited (code {self.worker.returncode}): load the policy again."
                self.message += " " + self.load_error
            label = ep.label_after if started else None
            if label is not None and label[0] == "discard":
                self._start_saver(self._discard, ep)
            elif label is not None:
                self._start_saver(self._label, ep, *label)
        log.info("episode %d ended: %s (%.1f s, %d frames, fault=%s)", ep.n, ep.end_reason, ep.elapsed,
                 ep.frames, ep.fault)

    # ------------------------------------------------------------------ outcomes (saver thread)
    def _label(self, ep, outcome, note):
        idx, ds_msg = None, ""
        if ep.ds is not None:
            ds_msg = self._save_rollout(ep, outcome, note)
            idx = ep.episode_index
        line = {"t": _now_iso(), "policy": ep.policy, "episode": ep.n, "outcome": outcome,
                "duration_s": round(ep.elapsed, 2), "interventions": ep.n_interventions, "dry_run": ep.dry_run,
                "dataset": ep.dataset, "episode_index": idx, "settings": ep.settings(), "note": note,
                "end_reason": ep.end_reason, "fault": ep.fault, "intervention_frames": ep.interventions}
        if mock_hw.enabled():
            line["mock"] = True  # simulated arm: excluded from the real tally
        if ep.evals_path is not None:
            _append_jsonl(ep.evals_path, line)
        with self.lock:
            if self.pending is ep:
                self.pending = None
            self.last_outcome = outcome
            if ep.evals_path is not None and ep.evals_path == self.evals_path:
                self.tally = tally(self.evals_path)
            label = {"success": "success", "fail": "fail", "none": "saved without a label"}[outcome]
            self.message = f"Episode {ep.n}: {label}{ds_msg}{self._backend_note()}"

    def _save_rollout(self, ep, outcome, note):
        """Save the pending rollout into its dataset (+ so101_rollouts.jsonl). Returns a message suffix."""
        ds, ep.ds = ep.ds, None
        ep.episode_index = None
        try:
            min_frames = int(recorder.MIN_EPISODE_S * FPS)
            if ep.writer is not None and ep.writer.error is not None:
                why = f"recording failed ({ep.writer.error})"
            elif ep.frames < min_frames:
                why = f"too short ({ep.frames} frames)"
            else:
                why = None
            if why is not None:
                if ds.has_pending_frames():
                    ds.clear_episode_buffer()
                ds.finalize()
                return f"; not saved to {ep.dataset}: {why}"
            idx, msg = recorder.save_pending_episode(ds)  # finalizes on success
            if idx is None:
                ds.finalize()
                return f"; not saved: {msg}"
            ep.episode_index = idx
            _append_jsonl(self.data_root / ep.dataset / "meta" / "so101_rollouts.jsonl", {
                "episode_index": idx, "policy": ep.policy, "outcome": outcome,
                "intervention_frames": ep.interventions, "t": _now_iso(), "episode": ep.n, "frames": ep.frames,
                "duration_s": round(ep.elapsed, 2), "end_reason": ep.end_reason, "fault": ep.fault, "note": note,
                "settings": ep.settings(), "quality": ep.quality.summary() if ep.quality else None,
                **({"mock": True} if mock_hw.enabled() else {})})
            with self.lock:
                self.record_saved[ep.dataset] = self.record_saved.get(ep.dataset, 0) + 1
            q = ep.quality.summary() if ep.quality else {}
            warn = f" (warning: {q['warning']})" if q.get("warning") else ""
            return f"; saved as episode {idx} of {ep.dataset}{warn}"
        except Exception as e:
            log.exception("saving rollout failed")
            try:
                ds.finalize()
            except Exception:
                pass
            return f"; saving to {ep.dataset} failed: {e}"

    def _discard(self, ep):
        ds, ep.ds = ep.ds, None
        if ds is not None:
            try:
                if ds.has_pending_frames():
                    ds.clear_episode_buffer()
                ds.finalize()
            except Exception:
                log.exception("discarding rollout failed")
        with self.lock:
            if self.pending is ep:
                self.pending = None
            self.last_outcome = "discarded"
            self.message = f"Episode {ep.n} discarded{self._backend_note()}"

    # ------------------------------------------------------------------ monitor / shutdown
    def _monitor(self):
        last_ping = 0.0
        while not self._stopping:
            time.sleep(1.0)
            with self.lock:
                client, worker, gen = self.client, self.worker, self._load_gen
                idle = self.job_kind is None and self.load_state == "ready"
            if not idle or client is None:
                continue
            if worker is not None and not worker.alive():
                with self.lock:
                    if gen == self._load_gen and self.load_state == "ready":
                        self.load_state = "error"
                        self.message = self.load_error = (f"The policy worker exited (code {worker.returncode}): "
                                                          "load the policy again")
                continue
            if time.monotonic() - last_ping < 3.0:
                continue
            last_ping = time.monotonic()
            try:
                client.ping(timeout=5.0)
                ok, err = True, None
            except PolicyError as e:
                ok, err = False, str(e)
            with self.lock:
                if gen != self._load_gen or self.job_kind is not None:
                    continue
                if not ok and self.backend_ok:
                    self.backend_error = f"Policy backend not responding: {err}"
                    self.message = self.backend_error
                elif ok and not self.backend_ok:
                    self.backend_error = None
                    self.message = f"Policy backend is back: {self.name} ready"
                self.backend_ok = ok

    def shutdown(self):
        """App exit: end a running episode (hold), save a pending rollout unlabeled, stop the backend."""
        self._stopping = True
        with self.lock:
            self._load_gen += 1
            if self.job_kind in ("episode", "home"):
                self._cmds.append("stop")
            job = self.job
        if job is not None:
            job.join(15)
        if self._saver is not None:
            self._saver.join(60)
        with self.lock:
            ep = self.pending
        if ep is not None and ep.ds is not None:
            try:
                self._label(ep, "none", "saved at shutdown")
            except Exception:
                log.exception("saving the pending rollout at shutdown failed")
        if self._loader is not None:
            self._loader.join(3)
        with self.lock:
            client, worker = self.client, self.worker
            self.client = self.worker = None
        self._close_backend(client, worker)
        w = self._loading_worker
        if w is not None:
            w.stop(timeout=2)

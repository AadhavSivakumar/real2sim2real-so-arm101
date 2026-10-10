"""Episode recorder for the SO-101 web console.

Writes LeRobot v3.0 datasets that train ACT and pi0.5 as-is:
- observation.state: follower position read at the start of a control tick
- action: the goal sent to the follower in that same tick (what a policy has to output)
- observation.images.wrist / .scene: 640x480 RGB video
Units are degrees for joints 1-5 and 0-100 for the gripper (lerobot's default use_degrees=True), so
policies deploy with stock lerobot-rollout. Depth is view-only and never recorded.

Runs as one thread in the web app and mirrors lerobot-record: record -> reset -> save, with
next / rerecord / stop. With manual_start, each episode waits for a "next" (Space in the UI) instead of
starting on its own, and the reset period lasts until the next press. Every kept episode is saved and finalized before the next one starts, so a crash
only loses the episode in progress.

The dataset helpers (open_dataset, make_frame, EpisodeQuality, save_pending_episode) are shared with the policy
rollout recorder in policy_runner.py. Teleop recording and policy episodes are mutually exclusive: the app sets
Recorder.blocker to a callable that returns why a session can't start (or None).
"""
import logging
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("so101_web.recorder")

FPS = 30
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NAMES = [f"{j}.pos" for j in JOINTS]
CAMERAS = ("wrist", "scene")
IMG = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (6,), "names": NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": NAMES},
    **{f"observation.images.{c}": dict(IMG) for c in CAMERAS},
}
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MIN_EPISODE_S = 1.0

_lerobot = None
_import_lock = threading.Lock()


def _import_lerobot():
    """LeRobot's dataset stack takes ~6 s to import on the Jetson; do it once, in the background."""
    global _lerobot
    with _import_lock:
        if _lerobot is None:
            from lerobot.configs import RGBEncoderConfig
            from lerobot.datasets import LeRobotDataset
            _lerobot = (LeRobotDataset, RGBEncoderConfig)
    return _lerobot


def warm_up():
    threading.Thread(target=_import_lerobot, daemon=True, name="lerobot-import").start()


def list_datasets(root):
    import json
    out = []
    for info in sorted(Path(root).expanduser().glob("*/meta/info.json")):
        try:
            meta = json.loads(info.read_text())
            out.append({"name": info.parent.parent.name, "episodes": meta.get("total_episodes", 0),
                        "frames": meta.get("total_frames", 0), "fps": meta.get("fps")})
        except (OSError, ValueError):
            continue
    return out


def open_dataset(root, name):
    """Create the dataset root/name, or resume it if it exists (same features and fps required)."""
    LeRobotDataset, RGBEncoderConfig = _import_lerobot()
    # x264 ultrafast: ~0.2 core per camera here. Never "auto" (picks NVENC, which this Jetson lacks) and
    # not LeRobot's default libsvtav1 (saturates all 6 cores with two cameras).
    kw = dict(rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast", crf=23, g=2),
              encoder_threads=1, streaming_encoding=True, encoder_queue_maxsize=60, video_backend="pyav")
    root, repo = Path(root).expanduser() / name, f"local/{name}"
    if (root / "meta" / "info.json").exists():
        shutil.rmtree(root / "images", ignore_errors=True)  # leftovers from a crash
        for p in root.glob("tmp*"):
            shutil.rmtree(p, ignore_errors=True)
        ds = LeRobotDataset.resume(repo, root=root, **kw)
        for k, f in FEATURES.items():
            have = ds.meta.features.get(k)
            if have is None or have["dtype"] != f["dtype"] or tuple(have["shape"]) != f["shape"]:
                ds.finalize()
                raise RuntimeError(f"existing dataset '{name}' has different features ({k}); use a new name")
        if ds.meta.fps != FPS:
            ds.finalize()
            raise RuntimeError(f"existing dataset '{name}' is {ds.meta.fps} fps, not {FPS}")
        return ds
    root.parent.mkdir(parents=True, exist_ok=True)
    return LeRobotDataset.create(repo, FPS, FEATURES, root=root, robot_type="so_follower", use_videos=True, **kw)


def make_frame(state, action, frames, task):
    """One dataset row: joint dicts (LeRobot units) + {camera: RGB frame}."""
    return {
        "observation.state": np.array([state[j] for j in JOINTS], dtype=np.float32),
        "action": np.array([action[j] for j in JOINTS], dtype=np.float32),
        **{f"observation.images.{cam}": frames[cam] for cam in CAMERAS},
        "task": task,
    }


class EpisodeQuality:
    """Timing and camera health of one recorded episode."""

    def __init__(self, t0=None):
        self.n = self.late = 0
        self.max_gap = 0.0
        self.dup = dict.fromkeys(CAMERAS, 0)
        self._last_ts = dict.fromkeys(CAMERAS)
        self._t_prev = t0

    def frame(self, timestamps, now):
        """Count a recorded frame: {camera: capture timestamp} and the time it was taken."""
        for cam in CAMERAS:
            self.dup[cam] += timestamps[cam] == self._last_ts[cam]
            self._last_ts[cam] = timestamps[cam]
        self.n += 1
        if self._t_prev is not None:
            self.max_gap = max(self.max_gap, now - self._t_prev)
        self._t_prev = now

    def summary(self):
        n = self.n
        q = {"frames": n, "late_ticks": self.late, "max_gap_ms": round(self.max_gap * 1000, 1), "dup_frames": dict(self.dup)}
        warn = []
        if self.late > 0.01 * max(n, 1):
            warn.append(f"{self.late} late ticks (the loop fell behind; time is compressed in the data)")
        for cam in CAMERAS:
            if self.dup[cam] > 0.1 * max(n, 1):
                warn.append(f"{cam} camera repeated {self.dup[cam]} frames (running below {FPS} fps?)")
        q["warning"] = "; ".join(warn) or None
        return q


def save_pending_episode(ds):
    """Save the buffered episode and finalize the files. Returns (episode_index | None, message)."""
    enc = getattr(ds.writer, "_streaming_encoder", None)  # private in lerobot 0.6.2
    dropped = sum(getattr(enc, "_dropped_frames", {}).values()) if enc else 0
    if dropped:
        # A dropped frame silently shifts the video against the parquet rows: never keep such an episode.
        ds.clear_episode_buffer()
        return None, f"Episode discarded: the video encoder dropped {dropped} frames"
    index = ds.meta.total_episodes
    t0 = time.time()
    ds.save_episode()
    ds.finalize()
    return index, f"{time.time() - t0:.1f} s"


class _Abort(Exception):
    pass


class Recorder:
    def __init__(self, ctl, feeds, root):
        self.ctl, self.feeds, self.root = ctl, feeds, Path(root).expanduser()
        self.lock = threading.Lock()
        self.thread = None
        self.cmd = None  # "next" | "rerecord" | "stop", consumed by the session thread
        self.st = {"phase": "idle", "message": ""}
        self.blocker = None  # () -> reason a session can't start now (a policy episode), or None

    # ------------------------------------------------------------------ called from HTTP threads
    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, p):
        if self.active():
            return False, "a recording session is already running"
        why = self.blocker() if self.blocker else None
        if why:
            return False, why
        name = str(p.get("dataset", "")).strip()
        task = " ".join(str(p.get("task", "")).split())
        if not NAME_RE.match(name):
            return False, "dataset name: letters, digits, '_' and '-' only"
        if len(task) < 3:
            return False, "enter a task description (pi0.5 uses it as its language prompt)"
        try:
            cfg = {"name": name, "task": task, "episode_s": float(p.get("episode_s", 30)),
                   "reset_s": float(p.get("reset_s", 10)), "num_episodes": int(p.get("num_episodes", 10)),
                   "manual_start": str(p.get("manual_start", "")).lower() in ("1", "true", "on", "yes")}
        except (TypeError, ValueError):
            return False, "episode / reset seconds and episode count must be numbers"
        if not (1 <= cfg["episode_s"] <= 600 and 0 <= cfg["reset_s"] <= 300 and 1 <= cfg["num_episodes"] <= 1000):
            return False, "episode 1-600 s, reset 0-300 s, 1-1000 episodes"
        with self.lock:
            self.cmd = None
            self.st = {"phase": "opening", "message": "Opening dataset...", "dataset": name, "task": task,
                       "episode_s": cfg["episode_s"], "reset_s": cfg["reset_s"],
                       "num_episodes": cfg["num_episodes"], "manual_start": cfg["manual_start"],
                       "saved": 0, "last": None}
            self.thread = threading.Thread(target=self._session, args=(cfg,), daemon=True, name="recorder")
            self.thread.start()
        return True, "ok"

    def command(self, c):
        if c not in ("next", "rerecord", "stop"):
            return False, "unknown command"
        if not self.active():
            return False, "no recording session"
        with self.lock:
            self.cmd = c
        return True, "ok"

    def status(self):
        with self.lock:
            return dict(self.st)

    # ------------------------------------------------------------------ session thread
    def _set(self, **kw):
        with self.lock:
            self.st.update(kw)

    def _take_cmd(self):
        with self.lock:
            c, self.cmd = self.cmd, None
            return c

    def _open(self, cfg):
        return open_dataset(self.root, cfg["name"])

    def _session(self, cfg):
        ds = None
        try:
            ds = self._open(cfg)
            saved, stop, armed = 0, False, False
            while saved < cfg["num_episodes"] and not stop:
                if not self._wait_ready():
                    break
                if cfg["manual_start"] and not armed and not self._wait_start(saved):
                    break
                armed = False
                result, n, quality = self._record(ds, cfg, saved)
                if result.startswith("abort:"):
                    if ds.has_pending_frames():
                        ds.clear_episode_buffer()
                    self._set(message=f"Episode discarded: {result[6:]}")
                    continue  # back to waiting; the user fixes the problem and recording resumes
                if result == "stop":
                    stop = True
                pending = result != "rerecord" and n >= MIN_EPISODE_S * FPS
                if not pending and ds.has_pending_frames():
                    ds.clear_episode_buffer()
                if not stop and saved + int(pending) < cfg["num_episodes"]:
                    r = self._reset(cfg, pending)
                    if r == "rerecord" and pending:
                        ds.clear_episode_buffer()
                        pending = False
                        self._set(message="Previous episode discarded")
                    stop = r == "stop"
                    armed = cfg["manual_start"] and r != "stop"  # the press that ended the reset starts the next episode
                if pending:
                    if self._save(ds, n, quality):
                        saved += 1
                        self._set(saved=saved)
                        if saved < cfg["num_episodes"] and not stop:
                            ds = self._open(cfg)  # resume: next episode goes into new files
            self._set(phase="idle", message=f"Session finished: {saved} episode(s) saved to {self.root / cfg['name']}")
        except Exception as e:
            log.exception("recording session failed")
            self._set(phase="error", message=f"Recording failed: {e}")
        finally:
            if ds is not None:
                try:
                    ds.finalize()
                except Exception:
                    log.exception("finalize failed")

    def _wait_ready(self):
        """Block until the arm and cameras can be recorded; False if the user stopped the session."""
        while True:
            if self._take_cmd() == "stop":
                return False
            _, _, action, mode, synced = self.ctl.snapshot()
            reason = None
            if mode == "limp":
                reason = "Arm is limp: switch to Teleop or Manual to record"
            elif mode == "teleop" and not synced:
                reason = "Waiting for the follower to sync with the leader"
            elif action is None:
                reason = "Waiting for the control loop"
            else:
                for c in CAMERAS:
                    try:
                        self.feeds[c].get_frame()
                    except Exception as e:
                        reason = f"Waiting for the {c} camera: {e}"
                        break
            if reason is None:
                return True
            self._set(phase="waiting", message=reason)
            time.sleep(0.1)

    def _wait_start(self, saved):
        """Manual start: wait for "next". False if the user stopped the session."""
        self._take_cmd()  # drop stale presses
        while True:
            c = self._take_cmd()
            if c == "stop":
                return False
            if c == "next":
                return True
            _, _, action, mode, synced = self.ctl.snapshot()
            if mode == "limp" or action is None or (mode == "teleop" and not synced):
                if not self._wait_ready():
                    return False
            self._set(phase="ready", episode=saved + 1, message="Press Space to start recording")
            time.sleep(0.05)

    def _record(self, ds, cfg, saved):
        """Record one episode. Returns (result, frames, quality); result is done|rerecord|stop|abort:<why>."""
        ctl, period, total = self.ctl, 1.0 / FPS, int(round(cfg["episode_s"] * FPS))
        stops0 = ctl.stop_count
        _, _, _, mode0, _ = ctl.snapshot()
        n = 0
        self._set(phase="recording", episode=saved + 1, frames=0, elapsed_s=0.0, late=0,
                  message=f"Recording episode {saved + 1} ({mode0})")
        t_next = time.perf_counter()
        q = EpisodeQuality(t0=t_next)
        result = "done"
        while n < total:
            c = self._take_cmd()
            if c == "next":
                break
            if c in ("rerecord", "stop"):
                result = c
                break
            _, state, action, mode, synced = ctl.snapshot()
            if ctl.stop_count != stops0:
                return "abort:STOP was pressed", n, None
            if mode != mode0 or action is None or (mode == "teleop" and not synced):
                return f"abort:control mode changed to {mode}", n, None
            frames, stamps = {}, {}
            for cam in CAMERAS:
                try:
                    frames[cam], stamps[cam] = self.feeds[cam].get_frame()
                except Exception as e:
                    return f"abort:{cam} camera: {e}", n, None
            ds.add_frame(make_frame(state, action, frames, cfg["task"]))
            n += 1
            now = time.perf_counter()
            q.frame(stamps, now)
            t_next += period
            if now > t_next + period:  # more than a frame behind: skip ahead rather than burst
                q.late += int((now - t_next) / period)
                t_next = now
            if n % 10 == 0:
                self._set(frames=n, elapsed_s=round(n / FPS, 1), late=q.late)
            time.sleep(max(0.0, t_next - time.perf_counter()))
        quality = q.summary()
        self._set(frames=n, elapsed_s=round(n / FPS, 1), late=q.late, last=quality)
        return result, n, quality

    def _reset(self, cfg, pending):
        """Reset period: teleop keeps running, nothing recorded. Returns done|rerecord|stop."""
        end = float("inf") if cfg["manual_start"] else time.time() + cfg["reset_s"]
        result = "done"
        if cfg["manual_start"]:
            self._set(message="Reset the scene, then press Space to save and start the next episode")
        while time.time() < end:
            c = self._take_cmd()
            if c == "next":
                break
            if c == "stop":
                return "stop"
            if c == "rerecord" and pending:
                result, pending = "rerecord", False  # discard the episode just recorded, keep resetting
                self._set(message="Last episode will be discarded")
            self._set(phase="resetting", pending=pending,
                      remaining_s=None if cfg["manual_start"] else round(end - time.time(), 1))
            time.sleep(0.1)
        return result

    def _save(self, ds, expected, quality):
        self._set(phase="saving", message="Saving episode...")
        index, msg = save_pending_episode(ds)
        if index is None:
            self._set(message=msg)
            return False
        self._set(message=f"Saved episode ({expected} frames, {msg})", last=quality)
        return True

"""Fake SO-101 hardware for developing and testing the web console without the robot (SO101_MOCK=1).

Never opens /dev/serial/* or /dev/video*. Provides exactly the surface app.py uses:
- MockFollower / MockLeader: bus.sync_read("Present_Position"), bus.calibration[j].range_min/max,
  bus.enable_torque/disable_torque, send_action / get_action, connect/disconnect, is_calibrated.
  The follower tracks its goal like a stiff position servo (first order, tau 50 ms, <= 300 deg/s, small read
  noise); with torque off it stays where it is (limp = holds position). jam(joint, at) blocks a joint, for
  testing the tracking guard.
- MockCamera: the OpenCVCamera surface CameraFeed uses (frame_lock, latest_frame, latest_timestamp,
  read_latest, connect/disconnect). Loops one episode of a recorded dataset at 30 fps (decoded with PyAV),
  or a synthetic test pattern when the dataset isn't available.
The follower (and the leader) start at that episode's first observation.state, so camera views and arm agree.

Env: SO101_MOCK_DATASET (default ~/so101_datasets/cube_cup_real_v1), SO101_MOCK_EPISODE (default 0),
SO101_MOCK_LEADER=0 to simulate a missing leader arm; SO101_MOCK_LEADER=follow for a leader that a person keeps
at the follower's pose (it reads the follower's position), so leader takeovers pass the pose-gap check.
"""
import functools
import json
import logging
import os
import random
import threading
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("so101_web.mock")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REST_POSE = [0.0, -103.0, 95.0, 52.0, 0.0, 20.0]
# The real follower's calibration ranges (raw ticks), so the app computes the same joint limits as on the arm.
CALIBRATION = {"shoulder_pan": (1208, 2872), "shoulder_lift": (885, 3274), "elbow_flex": (844, 3069),
               "wrist_flex": (854, 3204), "wrist_roll": (0, 4095), "gripper": (1953, 3476)}


def enabled():
    return os.environ.get("SO101_MOCK", "0").strip().lower() in ("1", "true", "yes", "on")


@functools.lru_cache(maxsize=None)
def episode_source():
    """{"state0": [6] | None, "videos": {cam: (mp4 path, from_s, to_s)}} of the mock episode."""
    root = Path(os.environ.get("SO101_MOCK_DATASET", "~/so101_datasets/cube_cup_real_v1")).expanduser()
    ep = int(os.environ.get("SO101_MOCK_EPISODE", "0"))
    out = {"state0": None, "videos": {}, "dataset": str(root), "episode": ep}
    try:
        import pyarrow.parquet as pq
        info = json.loads((root / "meta" / "info.json").read_text())
        row = None
        for f in sorted((root / "meta" / "episodes").glob("*/*.parquet")):
            t = pq.read_table(f)
            idx = t.column("episode_index").to_pylist()
            if ep in idx:
                i = idx.index(ep)
                row = {c: t.column(c)[i].as_py() for c in t.column_names if not c.startswith("stats/")}
                break
        if row is None:
            raise LookupError(f"episode {ep} not in {root}")
        for cam in ("wrist", "scene"):
            key = f"observation.images.{cam}"
            if f"videos/{key}/file_index" not in row:
                continue
            path = root / info["video_path"].format(video_key=key, chunk_index=row[f"videos/{key}/chunk_index"],
                                                    file_index=row[f"videos/{key}/file_index"])
            out["videos"][cam] = (path, float(row[f"videos/{key}/from_timestamp"]),
                                  float(row[f"videos/{key}/to_timestamp"]))
        data = root / info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
        t = pq.read_table(data, columns=["index", "observation.state"])
        i = t.column("index").to_pylist().index(row["dataset_from_index"])
        out["state0"] = [float(x) for x in t.column("observation.state")[i].as_py()]
        log.info("mock: episode %d of %s (start pose %s)", ep, root, [round(x, 1) for x in out["state0"]])
    except Exception as e:
        log.warning("mock: no recorded episode (%s: %s); synthetic camera pattern, rest pose", type(e).__name__, e)
    return out


def start_pose():
    return dict(zip(JOINTS, episode_source()["state0"] or REST_POSE))


# ------------------------------------------------------------------------------------------------- arms
class _Cal:
    def __init__(self, lo, hi):
        self.range_min, self.range_max = lo, hi


class _ArmSim:
    """First-order position servo per joint, simulated lazily on each read."""

    TAU = 0.05  # s
    VMAX = 300.0  # deg/s (and %/s for the gripper)
    NOISE = 0.05  # deg, read noise

    def __init__(self, pose):
        self.lock = threading.Lock()
        self.pos = dict(pose)
        self.goal = dict(pose)
        self.torque = True
        self.jammed = {}  # joint -> position it is stuck at
        self.t = time.perf_counter()
        self.limits = {}
        for j, (lo, hi) in CALIBRATION.items():
            half = (hi - lo) / 2 * 360 / 4095
            self.limits[j] = (0.0, 100.0) if j == "gripper" else (-half, half)

    def _advance(self):
        now = time.perf_counter()
        dt = min(now - self.t, 0.1)
        self.t = now
        if not self.torque:
            return
        for j in JOINTS:
            if j in self.jammed:
                self.pos[j] = self.jammed[j]
                continue
            err = self.goal[j] - self.pos[j]
            step = max(-self.VMAX, min(self.VMAX, err / self.TAU)) * dt
            self.pos[j] += err if abs(step) >= abs(err) else step
            lo, hi = self.limits[j]
            self.pos[j] = min(max(self.pos[j], lo - 2), hi + 2)

    def read(self):
        with self.lock:
            self._advance()
            return {j: self.pos[j] + random.gauss(0.0, self.NOISE) for j in JOINTS}

    def set_goal(self, goal):
        with self.lock:
            self._advance()
            self.goal.update(goal)

    def set_torque(self, on):
        with self.lock:
            self._advance()
            self.torque = on


class _MockBus:
    def __init__(self, sim):
        self.sim = sim
        self.calibration = {j: _Cal(*CALIBRATION[j]) for j in JOINTS}

    def sync_read(self, item, num_retry=0):
        if item != "Present_Position":
            raise NotImplementedError(f"mock bus: sync_read({item!r})")
        return self.sim.read()

    def enable_torque(self, num_retry=0):
        self.sim.set_torque(True)

    def disable_torque(self, num_retry=0):
        self.sim.set_torque(False)


class MockFollower:
    is_calibrated = True

    def __init__(self, pose=None):
        self.sim = _ArmSim(pose or start_pose())
        self.bus = _MockBus(self.sim)
        self.is_connected = False

    def connect(self, calibrate=False):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def send_action(self, action):
        self.sim.set_goal({k.removesuffix(".pos"): float(v) for k, v in action.items()})
        return action

    def jam(self, joint, at=None):
        """Block a joint where it is (or at `at`); jam(joint, False) frees it."""
        with self.sim.lock:
            if at is False:
                self.sim.jammed.pop(joint, None)
            else:
                self.sim.jammed[joint] = self.sim.pos[joint] if at is None else float(at)


class MockLeader:
    """A leader arm resting at the episode's start pose (small read noise)."""

    is_calibrated = True

    def __init__(self, pose=None, follow=None):
        self.pose = dict(pose or start_pose())
        self.follow = follow  # a MockFollower whose present position this leader reports
        self.is_connected = False

    def connect(self, calibrate=False):
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_action(self):
        pose = self.follow.sim.read() if self.follow is not None else self.pose
        return {f"{j}.pos": v + random.gauss(0.0, 0.05) for j, v in pose.items()}


def make_leader(follower=None):
    how = os.environ.get("SO101_MOCK_LEADER", "1")
    if how == "0":
        raise RuntimeError("mock leader disabled (SO101_MOCK_LEADER=0)")
    return MockLeader(follow=follower if how == "follow" else None)


# ----------------------------------------------------------------------------------------------- camera
class MockCamera:
    """Plays the mock episode's video for one camera in a loop at `fps`, like a live camera."""

    def __init__(self, name, fps=30, width=640, height=480):
        self.name, self.fps, self.width, self.height = name, fps, width, height
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_timestamp = None
        self.thread = None
        self._stop = threading.Event()
        self.is_connected = False

    def __str__(self):
        return f"MockCamera({self.name})"

    def connect(self, warmup=True):
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"mockcam-{self.name}")
        self.thread.start()
        self.is_connected = True
        t0 = time.time()
        while self.latest_frame is None and time.time() - t0 < 5:
            time.sleep(0.02)

    def disconnect(self):
        self._stop.set()
        if self.thread is not None:
            self.thread.join(2)
        self.is_connected = False

    def read_latest(self, max_age_ms=500):
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        with self.frame_lock:
            frame, ts = self.latest_frame, self.latest_timestamp
        if frame is None or ts is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age = (time.perf_counter() - ts) * 1e3
        if age > max_age_ms:
            raise TimeoutError(f"{self} latest frame is too old: {age:.1f} ms (max allowed: {max_age_ms} ms).")
        return frame

    def _publish(self, rgb):
        if rgb.shape[:2] != (self.height, self.width):
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        with self.frame_lock:
            self.latest_frame = rgb
            self.latest_timestamp = time.perf_counter()

    def _run(self):
        src = episode_source()["videos"].get(self.name)
        self._t_next = time.perf_counter()
        if src is not None:
            try:
                self._play(*src)  # loops until disconnect()
            except Exception as e:
                log.warning("mock camera %s: cannot play %s (%s); synthetic pattern", self.name, src[0], e)
        while not self._stop.is_set():
            self._synthetic()

    def _pace(self):
        self._t_next += 1.0 / self.fps
        delay = self._t_next - time.perf_counter()
        if delay > 0:
            self._stop.wait(delay)
        elif delay < -0.5:
            self._t_next = time.perf_counter()

    def _play(self, path, t0, t1):
        """Loop the [t0, t1) segment of the video; one open container (re-opening per loop spams swscaler logs)."""
        import av
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            while not self._stop.is_set():
                c.seek(int(t0 / s.time_base), stream=s, backward=True)
                n = 0
                for frame in c.decode(s):
                    if self._stop.is_set():
                        return
                    t = frame.time if frame.time is not None else 0.0
                    if t < t0 - 1e-3:
                        continue
                    if t >= t1 - 0.5 / self.fps:
                        break
                    self._publish(frame.to_ndarray(format="rgb24"))
                    n += 1
                    self._pace()
                if n == 0:
                    raise RuntimeError("no frames decoded")

    def _synthetic(self):
        h, w = self.height, self.width
        base = np.zeros((h, w, 3), np.uint8)
        base[..., 0] = np.linspace(40, 200, w, dtype=np.uint8)[None, :]
        base[..., 1] = np.linspace(40, 160, h, dtype=np.uint8)[:, None]
        base[..., 2] = 90
        for i in range(self.fps * 10):
            if self._stop.is_set():
                return
            img = base.copy()
            x = int((i / (self.fps * 10)) * w)
            cv2.circle(img, (x, h // 2), 40, (255, 255, 255), -1)
            cv2.putText(img, f"MOCK {self.name} {i:03d}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            self._publish(img)
            self._pace()

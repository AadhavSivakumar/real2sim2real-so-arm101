#!/usr/bin/env python
"""SO-101 web console: camera views (wrist, scene, depth), leader->follower teleop, manual joint control,
episode recording and policy deployment.

    python so101_web/app.py [--port 8080] [--data-root DIR] [--policy-root DIR]
    SO101_MOCK=1 SO101_DEPTH=0 python so101_web/app.py --port 8091   (fake arm and cameras, see mock_hw.py)

Design notes
- Only the control thread touches the serial buses (they are not thread-safe). HTTP handlers just
  post requests (mode, targets) that the control thread applies on its next tick.
- Every command to the follower goes through a rate limiter, so switching modes never makes the arm jump.
- Depth runs in a subprocess (depth_worker.py) because the Orbbec driver can crash.
- Policies run in a subprocess (policy_worker.py) or on a remote GPU server; policy_runner.py turns their
  action chunks into targets at 30 Hz and the control thread slews toward them in mode "policy", with its
  own heartbeat and tracking watchdogs (any fault -> hold the present pose, torque on).
"""
import argparse
import fcntl
import hmac
import json
import logging
import math
import os
import re
import signal
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

import mock_hw
import policy_runner
import recorder

MOCK = mock_hw.enabled()  # SO101_MOCK=1: fake follower/leader/cameras, never opens /dev/serial or /dev/video
HERE = Path(__file__).resolve().parent
log = logging.getLogger("so101_web")

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LEADER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6053876-if00"  # 5V arm
FOLLOWER_PORT = "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6080770-if00"  # 12V arm
WRIST_CAM = "/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0"
SCENE_CAM = "/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0"

# Speed limits (deg/s for joints, % /s for gripper) applied to every command sent to the follower.
POSITION_P = int(os.environ.get("SO101_P", 32))  # servo position gain; lerobot default 16 is soft and laggy
MANUAL_RATE = 400.0  # deg/s cap; high enough that slider drags feel direct
SYNC_RATE = 60.0  # teleop, while the follower is still catching up to the leader
TELEOP_RATE = 1500.0  # teleop once synced: effectively direct, only guards against bad reads
SYNC_TOLERANCE = 3.0
# Policy mode (targets come from policy_runner.py; rate = the episode's max speed, gripper in %/s)
POLICY_STALL_S = 0.5  # no heartbeat from the runner for this long -> hold
RAMP_RATE = 45.0  # rate cap until the command first gets within RAMP_DONE_DEG of the target (or RAMP_MAX_S)
RAMP_DONE_DEG = 5.0
RAMP_MAX_S = 2.0
TRACK_ERR_DEG = 25.0  # an arm joint this far from its command ...
TRACK_ERR_S = 1.0  # ... for this long (blocked, overloaded, disconnected) -> hold + fault
READ_FAIL_HOLD = 6  # consecutive follower read failures in policy mode (~0.1 s) -> hold + fault
TAKEOVER_MAX_GAP_DEG = 12.0  # takeover only when the leader is this close to the follower (arm joints)
FOLLOWER_LOCK = "/tmp/so101_follower.lock"  # one console per follower bus (non-mock only)


# --------------------------------------------------------------------------------------- video feeds
class Feed:
    """Latest JPEG of a video source. Clients block on wait_next()."""

    def __init__(self, name):
        self.name = name
        self.jpeg = None
        self.seq = 0
        self.cond = threading.Condition()
        self.last_client = 0.0
        self.status = "starting"
        self.extra = {}

    def publish(self, jpeg):
        with self.cond:
            self.jpeg = jpeg
            self.seq += 1
            self.cond.notify_all()

    def wait_next(self, seq, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout)
            return self.jpeg, self.seq

    def touch(self):
        self.last_client = time.time()

    def has_clients(self):
        return time.time() - self.last_client < 3.0


class CameraFeed(Feed):
    """OpenCV camera -> JPEG. Only encodes while someone is watching."""

    def __init__(self, name, path, fps=15):
        super().__init__(name)
        self.fps = fps
        self.cam = None
        self._path = path
        self._open()
        threading.Thread(target=self._run, daemon=True, name=f"feed-{name}").start()

    def _open(self):
        """(Re)connect the camera; leaves self.cam None and a status message if it fails."""
        if self.cam is not None:
            try:
                self.cam.disconnect()
            except Exception:
                pass
            self.cam = None
        try:
            if MOCK:
                cam = mock_hw.MockCamera(self.name)
            else:
                cam = OpenCVCamera(
                    OpenCVCameraConfig(index_or_path=self._path, width=640, height=480, fps=30, fourcc="MJPG")
                )
            cam.connect()
            self.cam = cam
            self.status = "ok"
        except Exception as e:
            self.status = f"camera not available, retrying: {e}"

    def get_frame(self, max_age_ms=200):
        """(fresh RGB copy, capture perf_counter time) of the newest frame; raises if disconnected or stale."""
        cam = self.cam
        if cam is None:
            raise RuntimeError(self.status)
        with cam.frame_lock:
            frame, ts = cam.latest_frame, cam.latest_timestamp
        if frame is None:
            raise RuntimeError("no frames yet")
        age = (time.perf_counter() - ts) * 1000
        if age > max_age_ms:
            raise TimeoutError(f"newest frame is {age:.0f} ms old")
        return frame.copy(), ts

    def _run(self):
        period = 1.0 / self.fps
        while True:
            t0 = time.time()
            if self.cam is None:
                time.sleep(3)
                self._open()
                continue
            if not self.has_clients():
                time.sleep(0.1)
                continue
            try:
                rgb = self.cam.read_latest(max_age_ms=1000)
                ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    self.publish(jpg.tobytes())
                    self.status = "ok"
            except Exception as e:
                self.status = f"error, reconnecting: {e}"
                self.jpeg = None
                time.sleep(1)
                self._open()
            time.sleep(max(0.0, period - (time.time() - t0)))


class DepthFeed(Feed):
    """Runs depth_worker.py, restarts it if it dies or stalls, and reports why depth is unavailable."""

    FIRST_FRAME_TIMEOUT = 60.0  # open time grows with the sensor's reply-queue lag (~0.05 s per lag step)
    FRAME_TIMEOUT = 8.0
    LAG_WARN = 300

    def __init__(self):
        super().__init__("depth")
        self.proc = None
        self._stop = False
        self._stop_ev = threading.Event()
        self._lock = threading.Lock()
        self._last_frame = 0.0
        self.lag = None
        threading.Thread(target=self._supervise, daemon=True, name="feed-depth").start()

    def _read_exact(self, proc, n):
        buf = b""
        while len(buf) < n:
            chunk = proc.stdout.read(n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return buf

    def _stderr_reader(self, proc):
        shown = 0
        for line in proc.stderr:
            line = line.decode(errors="replace").strip()
            if line.startswith("STATUS:"):
                self.status = line[7:].strip()
                continue
            m = re.match(r"obresync: lag (\d+)", line)
            if m:
                self.lag = int(m.group(1))
                self.extra = {**self.extra, "queue_lag": self.lag}
            if line and shown < 300:
                shown += 1
                log.warning("depth worker: %s", line)

    @staticmethod
    def _stop_gently(proc):
        """SIGTERM lets the worker stop the stream and close the device; SIGKILL only if it is stuck."""
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _watchdog(self, proc):
        """Kill a worker that stops delivering frames (unplugged mid-stream, driver hang)."""
        started = time.time()
        while proc.poll() is None and not self._stop_ev.wait(1):
            if self._last_frame < started:
                if time.time() - started > self.FIRST_FRAME_TIMEOUT:
                    self.status = "depth worker produced no frames, restarting it"
                    self._stop_gently(proc)
            elif time.time() - self._last_frame > self.FRAME_TIMEOUT:
                self.status = "depth frames stopped, restarting the worker"
                self._stop_gently(proc)

    @staticmethod
    def _device_id():
        """(USB address, healthy) of the Astra depth sensor, or (None, False) if absent.

        A sensor that enumerates with 0 interfaces has broken firmware/USB state; opening it just fails.
        """
        for d in Path("/sys/bus/usb/devices").glob("*"):
            try:
                if (d / "idVendor").read_text().strip() == "2bc5" and (d / "idProduct").read_text().strip() == "0403":
                    healthy = int((d / "bNumInterfaces").read_text().strip() or 0) > 0
                    return (d / "devnum").read_text().strip(), healthy
            except (OSError, ValueError):
                continue
        return None, False

    def _wait_for_device(self, not_id=None):
        """Block until a healthy sensor is on the bus (and, if not_id is given, at a different address)."""
        while not self._stop:
            dev, healthy = self._device_id()
            if dev is None:
                self.status = "waiting for the depth sensor to appear on USB"
            elif not healthy:
                self.status = ("depth sensor is on USB but broken (no interfaces, kernel descriptor errors)"
                               " -- unplug and replug the Astra USB cable")
            elif dev != not_id:
                return dev
            self._stop_ev.wait(2)
        return None

    def _supervise(self):
        self.status = "waiting for the depth sensor to appear on USB"
        failed_id = last_id = None
        fails = 0
        while not self._stop:
            # Failed opens don't harm the sensor, so retry a few times per plug-in; after that wait for a replug.
            dev_id = self._wait_for_device(not_id=failed_id)
            if dev_id is None:
                return
            if dev_id != last_id:
                last_id, fails = dev_id, 0
            if self._stop_ev.wait(4):  # let a freshly enumerated device settle
                return
            # obresync.so keeps the Astra's lagging command/reply queue in sync for the Orbbec driver.
            trace = HERE / "logs" / "obresync.log"  # every sensor command/reply, for diagnosing depth failures
            try:
                if trace.stat().st_size > 20_000_000:
                    trace.replace(trace.with_suffix(".log.1"))
            except OSError:
                pass
            env = dict(os.environ, LD_PRELOAD=str(HERE / "obresync.so"), SO101_PARENT_PID=str(os.getpid()),
                       OBRESYNC_LOG=str(trace))
            with self._lock:
                if self._stop:
                    return
                self.status = "starting"
                proc = self.proc = subprocess.Popen(
                    [sys.executable, str(HERE / "depth_worker.py")],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=str(HERE),
                )
            err_thread = threading.Thread(target=self._stderr_reader, args=(proc,), daemon=True)
            err_thread.start()
            threading.Thread(target=self._watchdog, args=(proc,), daemon=True).start()
            got_frames = False
            try:
                while True:
                    length, center = struct.unpack("<II", self._read_exact(proc, 8))
                    self.publish(self._read_exact(proc, length))
                    self._last_frame = time.time()
                    self.extra = {**self.extra, "center_mm": center}
                    if self.lag is not None and self.lag > self.LAG_WARN:
                        self.extra["warning"] = (f"sensor reply-queue lag is {self.lag}; depth start-up is slowing down, "
                                                 "replug the Astra when convenient")
                    got_frames = True
            except EOFError:
                pass
            code = proc.wait()
            err_thread.join(1)
            if self._stop:
                return
            self.jpeg = None
            if not got_frames:
                fails += 1
                reason = self.status if self.status not in ("ok", "starting") else f"worker exited (code {code})"
                if fails >= 3:
                    failed_id, fails = dev_id, 0
                    self.status = f"{reason} -- unplug and replug the Astra USB cable"
                else:
                    self.status = f"{reason} -- retrying ({fails}/3)"
                    self._stop_ev.wait(10)
            else:
                failed_id, fails = None, 0
                if self.status == "ok":
                    self.status = "depth stream stopped, restarting"
                self._stop_ev.wait(3)
            log.warning("depth worker ended: %s", self.status)

    def stop(self):
        with self._lock:
            self._stop = True
            self._stop_ev.set()
            proc = self.proc
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(3)
            except subprocess.TimeoutExpired:
                proc.kill()


# ------------------------------------------------------------------------------------------ robot control
class Controller:
    def __init__(self, hz=60):
        self.hz = hz
        if MOCK:
            self.follower = mock_hw.MockFollower()
        else:
            self.follower = SO101Follower(SO101FollowerConfig(port=FOLLOWER_PORT, id="follower", use_degrees=True, disable_torque_on_disconnect=False,
                                position_p_coefficient=POSITION_P))
        self.follower.connect(calibrate=False)
        if not self.follower.is_calibrated:
            self.follower.disconnect()
            raise SystemExit("follower is not calibrated (or calibration doesn't match): run ./so101.sh calibrate-follower")
        self.leader = None
        try:
            if MOCK:
                self.leader = mock_hw.make_leader(self.follower)
            else:
                self.leader = SO101Leader(SO101LeaderConfig(port=LEADER_PORT, id="leader", use_degrees=True))
            self.leader.connect(calibrate=False)
            if not self.leader.is_calibrated:
                raise RuntimeError("leader is not calibrated: run ./so101.sh calibrate-leader")
        except Exception as e:
            log.warning("leader not available (%s); teleop disabled", e)
            self.leader = None

        self.limits = self._limits()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        pos = self._read_follower()
        self.mode = "manual"  # holds the current pose at startup, never moves on its own
        self.req_mode = None
        self.targets = dict(pos)
        self.cmd = dict(pos)
        self.synced = False
        self.torque = True
        self.pos = pos
        self.leader_pos = None
        self.tick = 0
        self.last_action = None  # goal the follower is holding this tick (recorded as `action`)
        self.stop_count = 0  # bumped by STOP; the recorder discards the episode in progress
        self.actual_hz = 0.0
        self.errors = 0
        self.message = "Holding current pose"
        # policy mode: written by the runner (under the lock), read by the control thread
        self.policy_target = None  # {joint: value} to slew toward, None = hold the current command
        self.policy_rate = 30.0
        self.policy_heartbeat = 0.0  # perf_counter of the runner's last update
        self.policy_fault = None  # why the control thread left policy mode on its own (or refused it)
        self._policy_stops = None  # stop_count enter_policy() was called with
        self._reramp = False  # set by the runner: restart the ramp-in (control thread consumes it)
        self._ramp_t0 = None  # control-thread state for ramp-in and the tracking guard
        self._ramped = False
        self._track_since = {}
        self.follower.send_action({f"{j}.pos": v for j, v in pos.items()})  # goal = present
        self.thread = threading.Thread(target=self._loop, daemon=True, name="control")
        self.thread.start()

    def _limits(self):
        out = {}
        for j in JOINTS:
            if j == "gripper":
                out[j] = (0.0, 100.0)
            else:
                cal = self.follower.bus.calibration[j]
                half = (cal.range_max - cal.range_min) / 2 * 360 / 4095
                out[j] = (-half, half)
        return out

    def _read_follower(self):
        raw = self.follower.bus.sync_read("Present_Position", num_retry=2)
        return {j: float(raw[j]) for j in JOINTS}

    def _clamp(self, j, v):
        v = float(v)
        if not math.isfinite(v):
            raise ValueError(f"{j}: non-finite target {v}")
        lo, hi = self.limits[j]
        return min(max(v, lo), hi)

    def _clamp_all(self, target):
        return {j: self._clamp(j, float(target[j])) for j in JOINTS}

    def _slew(self, target, rate, dt):
        step = rate * dt
        for j in JOINTS:
            t = self._clamp(j, target[j])
            d = t - self.cmd[j]
            self.cmd[j] += max(-step, min(step, d))

    # ---- called from HTTP threads
    def request_mode(self, mode):
        if mode == "policy":
            return False, "policy mode is entered by starting a policy episode"
        if mode not in ("teleop", "manual", "limp"):
            return False, "unknown mode"
        if mode == "teleop" and self.leader is None:
            return False, "leader arm is not connected"
        with self.lock:
            self.req_mode = mode
        return True, "ok"

    def stop(self):
        """Freeze where the arm currently is commanded to be, torque stays on."""
        with self.lock:
            self.stop_count += 1
            self.targets = dict(self.cmd)
            self.policy_target = None
            self.req_mode = "manual"  # always (also overrides a pending switch into policy / teleop)

    # ---- policy mode (called by the policy runner)
    def enter_policy(self, stop_count, rate, target=None):
        """Switch to policy mode (keeps the current command: no jump). Refused if STOP was pressed since the
        caller read stop_count, or if the arm is limp (the runner never grabs a limp arm)."""
        with self.lock:
            if self.stop_count != stop_count:
                return False, "STOP was pressed"
            if not self.torque or self.mode == "limp" or self.req_mode == "limp":
                return False, "the arm is limp: switch to Manual or Teleop first"
            if self.req_mode not in (None, "policy"):
                return False, f"a switch to {self.req_mode} mode is pending"
            self._policy_stops = stop_count  # re-checked by the control thread right before it switches
            self.policy_target = self._clamp_all(target) if target else None
            self.policy_rate = float(rate)
            self.policy_heartbeat = time.perf_counter()
            self.policy_fault = None
            self.req_mode = "policy"
        return True, "ok"

    def set_policy_target(self, target, rate=None, reramp=False):
        """New target (None = hold) and heartbeat; the runner calls this every tick it controls the arm.
        reramp: approach this target at the ramp-in rate again (actions resumed after the policy starved)."""
        target = self._clamp_all(target) if target else None
        with self.lock:
            self.policy_target = target
            if rate is not None:
                self.policy_rate = float(rate)
            if reramp:
                self._reramp = True
            self.policy_heartbeat = time.perf_counter()

    def leave_policy(self, also_teleop=False):
        """Back to a manual hold of the present pose (no-op if the user already switched elsewhere)."""
        ours = ("policy", "teleop") if also_teleop else ("policy",)
        with self.lock:
            self.policy_target = None
            if self.req_mode in (None, *ours) and (self.mode in ours or self.req_mode in ours):
                self.req_mode = "manual"

    def takeover_gap(self):
        """(max |leader - follower| over the arm joints, joint) from the last tick, or (None, why)."""
        with self.lock:
            lp, pos = self.leader_pos, self.pos
        if self.leader is None:
            return None, "leader arm is not connected"
        if lp is None:
            return None, "no leader reading yet"
        j = max(JOINTS[:-1], key=lambda k: abs(lp[k] - pos[k]))
        return abs(lp[j] - pos[j]), j

    def request_takeover(self, stop_count, max_gap=TAKEOVER_MAX_GAP_DEG):
        """Policy -> teleop for a leader takeover. Refused if STOP was pressed since the caller read stop_count,
        if another mode change is pending, or if the leader is more than max_gap deg from the follower."""
        gap, j = self.takeover_gap()
        if gap is None:
            return False, j
        if gap > max_gap:
            return False, (f"bring the leader to the follower's pose first: {j} is {gap:.0f} deg away "
                           f"(max {max_gap:.0f})")
        with self.lock:
            if self.stop_count != stop_count:
                return False, "STOP was pressed"
            if self.req_mode not in (None, "policy"):
                return False, f"a switch to {self.req_mode} mode is pending"
            self.req_mode = "teleop"
        return True, "ok"

    def set_targets(self, targets):
        if not isinstance(targets, dict):
            return False, "targets must be {joint: value}"
        try:
            new = {j: self._clamp(j, v) for j, v in targets.items() if j in JOINTS}
        except (TypeError, ValueError) as e:
            return False, f"bad target: {e}"
        with self.lock:
            self.targets.update(new)
        return True, "ok"

    def snapshot(self):
        """(tick, follower pos read at tick start, goal sent that tick or None, mode, synced) of the last tick.
        Mode is "manual", "teleop", "limp" or "policy"."""
        with self.lock:
            return (self.tick, dict(self.pos), dict(self.last_action) if self.last_action else None,
                    self.mode, self.synced)

    def state(self):
        with self.lock:
            return {
                "mode": self.mode,
                "torque": self.torque,
                "synced": self.synced,
                "follower": self.pos,
                "leader": self.leader_pos,
                "targets": self.targets,
                "limits": self.limits,
                "leader_connected": self.leader is not None,
                "hz": round(self.actual_hz, 1),
                "errors": self.errors,
                "message": self.message,
                "stop_count": self.stop_count,
                "mock": MOCK,
            }

    # ---- control thread
    def _apply_mode_request(self, pos):
        with self.lock:
            mode, self.req_mode = self.req_mode, None
            if mode == "policy" and self.stop_count != self._policy_stops:
                mode = None  # STOP since the runner asked for policy mode: stay where we are
        if mode is None or mode == self.mode:
            return
        bus = self.follower.bus
        if mode == "policy":
            if not self.torque:  # never grab a limp arm on the policy's behalf
                with self.lock:
                    self.policy_fault = "the arm is limp"
                return
            # Keep self.cmd: it is exactly what the servos are holding now, so entering policy mode can't jump.
            self.synced = False
            self._ramp_t0, self._ramped, self._track_since = None, False, {}
            self.message = "Policy control"
            self.mode = mode
            return
        if mode == "limp":
            bus.disable_torque(num_retry=2)
            self.torque = False
            self.message = "Torque released: arm is limp"
        else:
            if not self.torque:
                self.follower.send_action({f"{j}.pos": v for j, v in pos.items()})  # goal = present, then grab
                bus.enable_torque(num_retry=2)
                self.torque = True
            self.cmd = dict(pos)
            self.synced = False
            if mode == "manual":
                with self.lock:
                    self.targets = dict(pos)
                self.message = "Manual joint control"
            else:
                self.message = "Teleop: follower is moving to the leader's pose..."
        self.mode = mode

    def _loop(self):
        period = 1.0 / self.hz
        last = time.perf_counter()
        n_leader_fail = n_read_fail = 0
        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            dt = min(t0 - last, 0.1)
            last = t0
            try:
                pos = self._read_follower()
                n_read_fail = 0
            except Exception as e:
                self.errors += 1
                n_read_fail += 1
                log.warning("follower read failed: %s", e)
                if self.mode == "policy" and n_read_fail >= READ_FAIL_HOLD:
                    try:  # the policy watchdogs need fresh reads: give the arm back to a manual hold
                        self._hold(None, f"Follower reads failing ({n_read_fail} in a row): holding the arm")
                    except Exception as e2:
                        log.warning("hold after read failures: %s", e2)
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                continue
            try:
                self._apply_mode_request(pos)
                leader = None
                if self.leader is not None:
                    try:
                        raw = self.leader.get_action()
                        leader = {j: float(raw[f"{j}.pos"]) for j in JOINTS}
                        n_leader_fail = 0
                    except Exception as e:
                        self.errors += 1
                        n_leader_fail += 1
                        log.warning("leader read failed: %s", e)
                        if self.mode == "teleop" and n_leader_fail >= 5:
                            self.req_mode = "manual"
                            self.message = "Leader read failing; switched to manual"

                if self.mode == "manual":
                    with self.lock:
                        target = dict(self.targets)
                    self._slew(target, MANUAL_RATE, dt)
                    self.follower.send_action({f"{j}.pos": v for j, v in self.cmd.items()})
                elif self.mode == "teleop" and leader is not None:
                    if not self.synced and max(abs(leader[j] - self.cmd[j]) for j in JOINTS) < SYNC_TOLERANCE:
                        self.synced = True
                        self.message = "Teleop"
                    self._slew(leader, TELEOP_RATE if self.synced else SYNC_RATE, dt)
                    self.follower.send_action({f"{j}.pos": v for j, v in self.cmd.items()})
                elif self.mode == "policy":
                    self._policy_tick(pos, dt)

                held = dict(self.cmd) if self.mode in ("manual", "teleop", "policy") and self.torque else None
                with self.lock:
                    self.pos = pos
                    self.last_action = held
                    self.tick += 1
                    self.leader_pos = leader
                    self.actual_hz = 0.9 * self.actual_hz + 0.1 / max(dt, 1e-6)
            except Exception as e:
                self.errors += 1
                log.exception("control loop error: %s", e)
                time.sleep(0.05)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))

    def _policy_tick(self, pos, dt):
        now = time.perf_counter()
        with self.lock:
            target, rate, beat = self.policy_target, self.policy_rate, self.policy_heartbeat
            reramp, self._reramp = self._reramp, False
        if reramp:
            self._ramp_t0, self._ramped = None, False
        if now - beat > POLICY_STALL_S:  # the runner thread is stuck or dead: freeze at the current command
            return self._hold(None, f"Policy stalled (no update for {now - beat:.1f} s): holding the arm")
        if target is None:
            self._ramp_t0, self._ramped = None, False  # the next target ramps in again
        else:
            if self._ramp_t0 is None:
                self._ramp_t0 = now
            if not self._ramped:
                if (max(abs(target[j] - self.cmd[j]) for j in JOINTS) < RAMP_DONE_DEG
                        or now - self._ramp_t0 > RAMP_MAX_S):
                    self._ramped = True
                else:
                    rate = min(rate, RAMP_RATE)
            self._slew(target, rate, dt)
        self.follower.send_action({f"{j}.pos": v for j, v in self.cmd.items()})
        for j in JOINTS[:-1]:  # tracking guard (not the gripper: it legitimately stalls on a grasped object)
            err = abs(pos[j] - self.cmd[j])
            if err <= TRACK_ERR_DEG:
                self._track_since.pop(j, None)
            elif now - self._track_since.setdefault(j, now) > TRACK_ERR_S:
                # hold where the arm actually is, so it stops pushing against whatever blocks it
                return self._hold(pos, f"Tracking fault: {j} stayed {err:.0f} deg away from its command for "
                                       f"over {TRACK_ERR_S:.0f} s (blocked or overloaded?): holding the arm")

    def _hold(self, pos, msg):
        """Leave policy mode on the control thread's own initiative: manual hold (torque on) of `pos`, or of the
        current command if pos is None."""
        self.mode = "manual"
        self.synced = False
        if pos is not None:
            self.cmd = dict(pos)
        with self.lock:
            self.targets = dict(self.cmd)
            self.policy_target = None
            self.policy_fault = msg
            if self.req_mode == "policy":
                self.req_mode = None
        self.message = msg
        log.warning(msg)
        self.follower.send_action({f"{j}.pos": v for j, v in self.cmd.items()})

    def shutdown(self):
        self.stop_event.set()
        self.thread.join(2)
        for dev in (self.follower, self.leader):
            try:
                if dev is not None:
                    dev.disconnect()
            except Exception as e:
                log.warning("disconnect failed: %s", e)


# ------------------------------------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "so101web/1.0"
    protocol_version = "HTTP/1.1"

    ctl: Controller = None
    rec: recorder.Recorder = None
    runner: policy_runner.PolicyRunner = None
    feeds: dict = {}
    token: str | None = None

    def log_message(self, *args):
        pass

    def _authorized(self, query):
        if not self.token:
            return True
        cookie = self.headers.get("Cookie", "")
        have = next((c.split("=", 1)[1] for c in (x.strip() for x in cookie.split(";")) if c.startswith("token=")), "")
        given = query.get("token", [have])[0]
        return hmac.compare_digest(given, self.token)

    def _send(self, code, body=b"", ctype="application/json", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if not self._authorized(q):
            return self._send(401, {"error": "token required: open the page with ?token=..."})
        cookie = {"Set-Cookie": f"token={self.token}; Path=/; SameSite=Strict"} if self.token else None
        if url.path in ("/", "/index.html"):
            return self._send(200, (HERE / "index.html").read_bytes(), "text/html; charset=utf-8", cookie)
        if url.path == "/api/state":
            st = self.ctl.state()
            st["feeds"] = {n: {"status": f.status, **f.extra} for n, f in self.feeds.items()}
            st["recording"] = self.rec.status()
            st["policy"] = self.runner.status()
            return self._send(200, st)
        if url.path == "/api/datasets":
            return self._send(200, {"root": str(self.rec.root), "datasets": recorder.list_datasets(self.rec.root)})
        if url.path == "/api/policies":
            return self._send(200, self.runner.list_policies())
        if url.path.startswith("/stream/") and url.path.endswith(".mjpg"):
            name = url.path[len("/stream/"):-len(".mjpg")]
            feed = self.feeds.get(name)
            if feed is None:
                return self._send(404, {"error": "no such stream"})
            if q.get("view", [""])[0] == "policy":  # frames as the loaded policy sees them (undistorted)
                feed = self.runner.view_feed(name) or feed
            return self._stream(feed)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        url = urlparse(self.path)
        if not self._authorized(parse_qs(url.query)):
            return self._send(401, {"error": "token required"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send(400, {"error": "bad json"})
        if url.path == "/api/mode":
            ok, msg = self.ctl.request_mode(body.get("mode"))
            return self._send(200 if ok else 400, {"ok": ok, "message": msg})
        if url.path == "/api/target":
            ok, msg = self.ctl.set_targets(body.get("targets", {}) if isinstance(body, dict) else None)
            return self._send(200 if ok else 400, {"ok": ok, "message": msg})
        if url.path == "/api/stop":  # freeze in place, torque stays on
            self.ctl.stop()
            return self._send(200, {"ok": True})
        if url.path == "/api/record/start":
            ok, msg = self.rec.start(body)
            return self._send(200 if ok else 400, {"ok": ok, "message": msg})
        if url.path.startswith("/api/record/"):
            ok, msg = self.rec.command(url.path.rsplit("/", 1)[1])
            return self._send(200 if ok else 400, {"ok": ok, "message": msg})
        if url.path.startswith("/api/policy/"):
            action = url.path[len("/api/policy/"):]
            if action not in policy_runner.COMMANDS:
                return self._send(404, {"error": "not found"})
            try:
                ok, msg = self.runner.command(action, body if isinstance(body, dict) else {})
            except Exception as e:
                log.exception("policy %s failed", action)
                ok, msg = False, f"{type(e).__name__}: {e}"
            return self._send(200 if ok else 400, {"ok": ok, "message": msg})
        self._send(404, {"error": "not found"})

    def _stream(self, feed):
        boundary = "frame"
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary}")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        seq = -1
        try:
            while True:
                feed.touch()
                jpeg, seq = feed.wait_next(seq)
                if jpeg is None:
                    continue
                self.wfile.write(
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n".encode()
                    + jpeg + b"\r\n"
                )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        self.close_connection = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--token", default=os.environ.get("SO101_TOKEN"), help="require ?token=... (or set SO101_TOKEN)")
    ap.add_argument("--hz", type=int, default=60, help="control loop rate")
    ap.add_argument("--data-root", default=os.environ.get("SO101_DATA_ROOT", "~/so101_datasets"),
                    help="where recorded datasets go")
    ap.add_argument("--policy-root", default=os.environ.get("SO101_POLICY_ROOT", "~/so101_policies"),
                    help="where policies (one directory each) and remotes.json are")
    ap.add_argument("--real", action="store_true",
                    help="allow the real arm on a port other than 8080 (without SO101_MOCK=1, other ports refuse)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if MOCK:
        log.warning("MOCK MODE (SO101_MOCK=1): simulated arms and cameras, no hardware is used")
    else:
        if args.port != 8080 and not args.real:
            raise SystemExit(f"refusing to drive the REAL arm on port {args.port}: set SO101_MOCK=1 for a simulated "
                             f"arm, or pass --real if you really mean the hardware")
        lock_fd = os.open(FOLLOWER_LOCK, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"another console already owns the follower arm ({FOLLOWER_LOCK} is locked): stop it first")
        os.ftruncate(lock_fd, 0)
        os.write(lock_fd, f"{os.getpid()} port {args.port}\n".encode())
        main.lock_fd = lock_fd  # held (open) for the life of the process

    ctl = Controller(hz=args.hz)
    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, TimeoutError)):
                return  # a browser closed a tab or a stream; not worth a traceback
            super().handle_error(request, client_address)

    server = Server((args.host, args.port), Handler)
    server.daemon_threads = True

    feeds = {}
    for name, path in (("wrist", WRIST_CAM), ("scene", SCENE_CAM)):
        try:
            feeds[name] = CameraFeed(name, path)
        except Exception as e:
            log.error("camera %s unavailable: %s", name, e)
            feeds[name] = Feed(name)
            feeds[name].status = f"unavailable: {e}"
    if MOCK or os.environ.get("SO101_DEPTH", "1") == "0":  # mock mode never touches the depth sensor either
        feeds["depth"] = Feed("depth")
        feeds["depth"].status = "depth disabled (mock mode)" if MOCK else "depth disabled (SO101_DEPTH=0)"
        feeds["depth"].stop = lambda: None
    else:
        feeds["depth"] = DepthFeed()

    recorder.warm_up()
    Handler.ctl, Handler.feeds, Handler.token = ctl, feeds, args.token
    Handler.rec = recorder.Recorder(ctl, feeds, args.data_root)
    Handler.runner = policy_runner.PolicyRunner(ctl, feeds, Handler.rec, args.policy_root, args.data_root)
    Handler.rec.blocker = Handler.runner.recording_blocker

    def bye(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGTERM, bye)

    log.info("Serving on http://%s:%d  (leader %s)", args.host, args.port, "connected" if ctl.leader else "MISSING")
    if not args.token:
        log.warning("No token set: anyone on the network can move the arm. Use --token or SO101_TOKEN.")
    try:
        server.serve_forever()
    finally:
        log.info("Shutting down")
        Handler.runner.shutdown()  # ends an episode (hold), saves a pending rollout unlabeled, stops the worker
        if Handler.rec.active():  # save what we can: the session thread finalizes the dataset on stop
            Handler.rec.command("stop")
            Handler.rec.thread.join(15)
        feeds["depth"].stop()
        ctl.shutdown()  # torque stays on: the arm keeps holding its last pose


if __name__ == "__main__":
    main()

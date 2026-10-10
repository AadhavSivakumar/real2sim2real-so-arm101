#!/usr/bin/env python
"""End-to-end test of policy deployment in mock mode: simulated arm + cameras, the real ACT policy on the GPU.

    SO101_MOCK=1 SO101_DEPTH=0 python tests/test_e2e.py [--port 8092] [--remote-port 8093]

Run from ~/lerobot with its venv python. Never touches hardware: the app runs with SO101_MOCK=1.
Part 1 (in-process Controller): policy mode entry without a jump, ramp-in, heartbeat stall -> hold, tracking
  guard -> hold, STOP, limp refusal.
Part 2 (HTTP, app.py on --port): listing, load, dry run, real run + STOP, time limit, validation, takeover/resume,
  worker killed mid-run -> hold + fault, teleop-recording exclusivity, rollout recording + outcome sidecar +
  LeRobotDataset reload, discard, policy view stream (undistort), remote server over TCP (+ killed mid-run),
  go to start pose (+ STOP), unload, shutdown without leaks.
Test policies (symlinks to ~/so101_policies/act_real_v0) and datasets live under ~/so101_web_dev/test_datasets.
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

WEB = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WEB))
HOME = Path.home()
DATA = HOME / "so101_web_dev" / "test_datasets"
POL = DATA / "_e2e_policies"
LOGS = DATA / "_logs"
REAL = HOME / "so101_policies" / "act_real_v0"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM = JOINTS[:-1]

RESULTS, METRICS = [], {}
BASE = None
APP = None


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  -- ' + str(detail)) if detail != '' else ''}", flush=True)
    return cond


def section(fn):
    def run(*a, **kw):
        print(f"\n== {fn.__name__}", flush=True)
        try:
            return fn(*a, **kw)
        except Exception as e:
            traceback.print_exc()
            check(f"{fn.__name__}: completed", False, f"{type(e).__name__}: {e}")
            try:  # leave the app in a clean state for the next section
                api("POST", "/api/policy/stop")
                wait(lambda s: s["policy"]["phase"] not in ("running", "intervention", "homing"), 10, "episode end")
                if pol()["phase"] == "pending":
                    api("POST", "/api/policy/outcome", {"outcome": "none"})
                    wait(lambda s: not s["policy"]["saving"], 30, "saving")
            except Exception:
                pass
    run.__name__ = fn.__name__
    return run


# ------------------------------------------------------------------------------------------- helpers
def api(method, path, body=None, timeout=15):
    data = json.dumps(body if body is not None else {}).encode() if method == "POST" else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
            return r.status, (json.loads(raw) if r.headers.get_content_type() == "application/json" else raw)
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except ValueError:
            return e.code, raw


def state():
    return api("GET", "/api/state")[1]


def pol():
    return state()["policy"]


def wait(pred, timeout, what, poll=0.05):
    t_end = time.monotonic() + timeout
    s = None
    while time.monotonic() < t_end:
        s = state()
        if pred(s):
            return s
        time.sleep(poll)
    p = s["policy"] if s else {}
    raise AssertionError(f"timeout ({timeout} s) waiting for {what}: mode={s and s['mode']} phase={p.get('phase')} "
                         f"msg={p.get('message')!r} fault={p.get('fault')!r}")


def phase_is(*phases):
    return lambda s: s["policy"]["phase"] in phases


def max_drift(seconds=0.6):
    """Largest joint motion (deg) over `seconds`: the arm is holding if this stays ~0."""
    a = state()["follower"]
    time.sleep(seconds)
    b = state()["follower"]
    return max(abs(a[j] - b[j]) for j in ARM)


def start(body, ok=True):
    """Start an episode; live unless the body says dry_run (the API itself defaults to a dry run).
    No recording unless the body names a dataset (auto_recording() covers the record-by-default path)."""
    code, r = api("POST", "/api/policy/start", {"dry_run": False, "record": False, **body})
    if ok and code != 200:
        raise AssertionError(f"start refused: {r}")
    return code, r


def label(outcome="none", note=None):
    code, r = api("POST", "/api/policy/outcome", {"outcome": outcome, **({"note": note} if note else {})})
    assert code == 200, r
    return wait(lambda s: not s["policy"]["saving"] and s["policy"]["phase"] != "pending", 60, "labeled")


def load(name=None, remote=None, timeout=240):
    t0 = time.monotonic()
    code, r = api("POST", "/api/policy/load", {"name": name} if name else {"remote": remote})
    assert code == 200, r
    s = wait(lambda s: s["policy"]["phase"] in ("ready", "error"), timeout, f"load {name or remote}", poll=0.3)
    return s, time.monotonic() - t0


def worker_pids(app_pid):
    out = subprocess.run(["pgrep", "-af", "[p]olicy_worker.py"], capture_output=True, text=True).stdout
    return [int(l.split()[0]) for l in out.splitlines() if f"/tmp/so101_policy_{app_pid}_" in l]


def time_to(pred, timeout=3.0):
    """Seconds until pred(state) holds (polled as fast as the API answers)."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if pred(state()):
            return time.monotonic() - t0
        time.sleep(0.005)
    return None


def grab_jpeg(path, timeout=8):
    import cv2
    buf = b""
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        t_end = time.monotonic() + timeout
        while time.monotonic() < t_end and len(buf) < 4_000_000:
            buf += r.read1(65536)
            a = buf.find(b"\xff\xd8")
            b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
            if a >= 0 and b > a:
                return cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
    return None


# ------------------------------------------------------------------------------------- test fixtures
def make_policy_root(remote_port):
    shutil.rmtree(POL, ignore_errors=True)
    POL.mkdir(parents=True)

    def link_ckpt(d):
        d.mkdir(parents=True)
        for f in REAL.iterdir():
            if f.is_file() and f.name not in ("so101_deploy.json", "evals.jsonl"):
                (d / f.name).symlink_to(f)

    link_ckpt(POL / "act_real_v0")
    link_ckpt(POL / "act_undistort")
    k = [470.0, 0.0, 320.0, 0.0, 470.0, 240.0, 0.0, 0.0, 1.0]
    (POL / "act_undistort" / "so101_deploy.json").write_text(json.dumps({
        "undistort": {"wrist": {"K": k, "dist": [-0.32, 0.12, 0.0, 0.0, -0.02]},
                      "scene": {"K": k, "dist": [-0.05, 0.01, 0.0, 0.0, 0.0]}},
        "start_pose": [0.0, -100.0, 90.0, 60.0, 0.0, 20.0], "task": "Put the cube in the cup",
        "max_speed": 100, "time_limit_s": 20, "notes": "e2e: undistorted views"}))
    link_ckpt(POL / "nested_run" / "checkpoints" / "last" / "pretrained_model")
    cfg = json.loads((REAL / "config.json").read_text())
    feats = cfg["input_features"]
    cam1 = {("observation.images.camera1" if k == "observation.images.wrist" else k): v for k, v in feats.items()}
    for name, sidecar in (("camera1", None), ("camera1_mapped", {"camera_map": {"observation.images.camera1": "wrist"}})):
        d = POL / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps({**cfg, "input_features": cam1}))
        if sidecar:
            (d / "so101_deploy.json").write_text(json.dumps(sidecar))
    d = POL / "diffusion_fake"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"type": "diffusion", "n_action_steps": 8, "input_features": feats,
                                               "output_features": cfg["output_features"]}))
    (POL / "remotes.json").write_text(json.dumps(
        [{"name": "tcp-local", "address": f"127.0.0.1:{remote_port}", "authkey": "e2e-key"}]))
    for ds in ("e2e_rollouts", "e2e_teleop", AUTO_DS):
        shutil.rmtree(DATA / ds, ignore_errors=True)


def start_app(port):
    global APP
    LOGS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / "e2e_app.log", "w")
    # SO101_MOCK_LEADER=follow: the mock leader sits at the follower's pose, so takeovers pass the pose-gap check
    env = dict(os.environ, SO101_MOCK="1", SO101_DEPTH="0", PYTHONUNBUFFERED="1", SO101_MOCK_LEADER="follow")
    APP = subprocess.Popen([sys.executable, str(WEB / "app.py"), "--port", str(port), "--data-root", str(DATA),
                            "--policy-root", str(POL)], stdout=log, stderr=subprocess.STDOUT, env=env,
                           cwd=str(HOME / "lerobot"))
    t_end = time.monotonic() + 90
    while time.monotonic() < t_end:
        try:
            if api("GET", "/api/state", timeout=2)[0] == 200:
                return APP
        except Exception:
            pass
        if APP.poll() is not None:
            raise RuntimeError(f"app exited with {APP.returncode}: see {LOGS / 'e2e_app.log'}")
        time.sleep(0.5)
    raise RuntimeError("app did not come up")


# --------------------------------------------------------------------------------- part 1: controller
class Beat(threading.Thread):
    """The runner's side of the policy heartbeat: send the current target 30x per second while `on`."""

    def __init__(self, ctl):
        super().__init__(daemon=True)
        self.ctl, self.target, self.on, self.alive = ctl, None, True, True
        self.start()

    def run(self):
        while self.alive:
            if self.on:
                self.ctl.set_policy_target(self.target)
            time.sleep(1 / 30)


@section
def controller_safety():
    os.environ["SO101_MOCK"] = "1"
    import app
    ctl = app.Controller(hz=60)
    beat = None
    try:
        time.sleep(0.4)
        _, pos0, _, mode, _ = ctl.snapshot()
        check("controller starts in manual hold", mode == "manual")
        check("/api/mode cannot select policy", ctl.request_mode("policy")[0] is False)
        ok, msg = ctl.enter_policy(ctl.stop_count, 120.0, None)
        beat = Beat(ctl)
        t0 = time.monotonic()
        while ctl.snapshot()[3] != "policy" and time.monotonic() - t0 < 1:
            time.sleep(0.005)
        check("enter policy mode", ok and ctl.snapshot()[3] == "policy", msg)
        time.sleep(0.5)
        _, pos1, held, _, _ = ctl.snapshot()
        check("no jump on entering policy mode", max(abs(pos1[j] - pos0[j]) for j in ARM) < 0.5,
              f"{max(abs(pos1[j] - pos0[j]) for j in ARM):.2f} deg")
        check("recorded action (held) includes policy mode", held is not None)

        # ramp-in: 60 deg away at max_speed 120 -> capped at 45 deg/s until within 5 deg
        base = dict(held)
        far = dict(base, shoulder_pan=base["shoulder_pan"] + 60)
        beat.target = far
        t0 = time.monotonic()
        time.sleep(0.5)
        moved = ctl.snapshot()[2]["shoulder_pan"] - base["shoulder_pan"]
        check("ramp-in caps the first approach at 45 deg/s", 15 < moved < 26, f"{moved:.1f} deg in 0.5 s")
        while abs(ctl.snapshot()[2]["shoulder_pan"] - far["shoulder_pan"]) > 0.1 and time.monotonic() - t0 < 4:
            time.sleep(0.01)
        METRICS["ramp_in_60deg_s"] = round(time.monotonic() - t0, 2)
        beat.target = base
        c0 = ctl.snapshot()[2]["shoulder_pan"]
        time.sleep(0.3)
        moved = c0 - ctl.snapshot()[2]["shoulder_pan"]
        check("full max_speed after ramp-in", 28 < moved < 40, f"{moved:.1f} deg in 0.3 s at 120 deg/s")
        time.sleep(0.5)
        beat.target = None  # hold -> the next target ramps in again
        time.sleep(0.2)
        c0 = ctl.snapshot()[2]["shoulder_pan"]
        beat.target = dict(ctl.snapshot()[2], shoulder_pan=c0 + 50)
        time.sleep(0.5)
        moved = ctl.snapshot()[2]["shoulder_pan"] - c0
        check("ramp-in again after a hold (target None)", 15 < moved < 26, f"{moved:.1f} deg in 0.5 s")

        # heartbeat stall -> hold
        beat.on = False
        t0 = time.monotonic()
        while ctl.snapshot()[3] == "policy" and time.monotonic() - t0 < 3:
            time.sleep(0.005)
        dt = time.monotonic() - t0
        METRICS["stall_to_hold_s"] = round(dt, 3)
        check("stalled heartbeat -> hold within ~0.5 s", ctl.snapshot()[3] == "manual" and 0.4 < dt < 0.75
              and "stalled" in (ctl.policy_fault or ""), f"{dt:.2f} s: {ctl.policy_fault}")
        time.sleep(0.3)
        a = ctl.snapshot()[1]
        time.sleep(0.5)
        b = ctl.snapshot()[1]
        check("arm holds after the stall", max(abs(a[j] - b[j]) for j in ARM) < 0.5)

        # tracking guard: a blocked arm joint -> hold + fault; a stalled gripper is fine
        beat.target = None
        beat.on = True
        ok, _ = ctl.enter_policy(ctl.stop_count, 120.0, None)
        time.sleep(0.2)
        cmd = ctl.snapshot()[2]
        ctl.follower.jam("elbow_flex")
        beat.target = dict(cmd, elbow_flex=cmd["elbow_flex"] - 60)
        t0 = time.monotonic()
        while ctl.snapshot()[3] == "policy" and time.monotonic() - t0 < 5:
            time.sleep(0.01)
        dt = time.monotonic() - t0
        check("tracking guard: blocked joint -> hold + fault", ctl.snapshot()[3] == "manual"
              and "Tracking fault" in (ctl.policy_fault or "") and "elbow_flex" in ctl.policy_fault,
              f"after {dt:.2f} s: {ctl.policy_fault}")
        ctl.follower.jam("elbow_flex", False)
        _, pos, held, _, _ = ctl.snapshot()
        check("hold is at the present pose (no push into the obstacle)", abs(held["elbow_flex"] - pos["elbow_flex"]) < 1.0)
        beat.target = None
        ok, _ = ctl.enter_policy(ctl.stop_count, 120.0, None)
        time.sleep(0.2)
        cmd = ctl.snapshot()[2]
        ctl.follower.jam("gripper")
        beat.target = dict(cmd, gripper=min(cmd["gripper"] + 50, 100))
        time.sleep(2.5)
        check("stalled gripper does not trip the guard", ctl.snapshot()[3] == "policy")
        ctl.follower.jam("gripper", False)

        # STOP wins, and a start that read stop_count before the STOP is refused
        sc = ctl.stop_count
        ctl.stop()
        t0 = time.monotonic()
        while ctl.snapshot()[3] != "manual" and time.monotonic() - t0 < 1:
            time.sleep(0.002)
        check("STOP -> manual hold within a control tick or two", time.monotonic() - t0 < 0.05,
              f"{(time.monotonic() - t0) * 1000:.0f} ms")
        check("enter_policy refused after STOP", ctl.enter_policy(sc, 90, None)[0] is False)
        beat.on = False
        ctl.request_mode("limp")
        time.sleep(0.2)
        ok, msg = ctl.enter_policy(ctl.stop_count, 90, None)
        time.sleep(0.1)
        check("policy refused while limp", not ok and ctl.snapshot()[3] == "limp", msg)
        ctl.request_mode("manual")
        time.sleep(0.2)
        review_regressions(ctl, beat)
    finally:
        if beat:
            beat.alive = False
        ctl.shutdown()


def _enter(ctl, beat):
    """Policy mode holding the current command, heartbeat on."""
    beat.target, beat.on = None, True
    ok, msg = ctl.enter_policy(ctl.stop_count, 90.0, None)
    t0 = time.monotonic()
    while ctl.snapshot()[3] != "policy" and time.monotonic() - t0 < 1:
        time.sleep(0.005)
    return ok and ctl.snapshot()[3] == "policy", msg


def _modes_for(ctl, seconds):
    seen, t0 = set(), time.monotonic()
    while time.monotonic() - t0 < seconds:
        seen.add(ctl.snapshot()[3])
        time.sleep(0.002)
    return seen


def review_regressions(ctl, beat):
    """Safety-review fixes, in-process: STOP vs takeover (1), follower read failures (2), takeover pose gap (3),
    STOP during the switch into policy mode (5)."""
    import mock_hw
    import policy_runner
    # (3) takeover refused while the leader is far from the follower; fine once it's at the follower's pose
    _, pos, _, _, _ = ctl.snapshot()
    ctl.leader = mock_hw.MockLeader(pose=dict(pos, shoulder_pan=pos["shoulder_pan"] + 30))
    time.sleep(0.1)
    ok_in, msg = _enter(ctl, beat)
    check("(review 3) enter policy for the takeover tests", ok_in, msg)
    ok, msg = ctl.request_takeover(ctl.stop_count)
    time.sleep(0.1)
    check("(review 3) takeover refused: leader 30 deg away", not ok and "bring the leader" in msg
          and ctl.snapshot()[3] == "policy", msg)
    ctl.leader = mock_hw.MockLeader(follow=ctl.follower)
    time.sleep(0.1)
    ok, msg = ctl.request_takeover(ctl.stop_count)
    t0 = time.monotonic()
    while ctl.snapshot()[3] != "teleop" and time.monotonic() - t0 < 1:
        time.sleep(0.005)
    check("(review 3) takeover accepted with the leader at the follower's pose", ok and ctl.snapshot()[3] == "teleop", msg)
    ctl.leave_policy(also_teleop=True)
    time.sleep(0.1)
    check("leave_policy(also_teleop) after a takeover -> manual", ctl.snapshot()[3] == "manual")

    # (1) STOP between the runner reading stop_count and the takeover: refused, the arm stays held
    _enter(ctl, beat)
    sc = ctl.stop_count
    ctl.stop()
    ok, msg = ctl.request_takeover(sc)
    seen = _modes_for(ctl, 0.3)
    check("(review 1) takeover after STOP refused, never teleop", not ok and "STOP" in msg and "teleop" not in seen
          and ctl.snapshot()[3] == "manual", f"{msg}; modes {sorted(seen)}")
    # (1) a takeover queued and then the episode ends (STOP / time limit / fault): the pending teleop is cancelled
    _enter(ctl, beat)
    ok, _ = ctl.request_takeover(ctl.stop_count)
    ctl.leave_policy(also_teleop=True)
    time.sleep(0.15)
    check("(review 1) leave_policy cancels a pending takeover", ok and ctl.snapshot()[3] == "manual", ctl.snapshot()[3])
    # (1) a pending user mode change blocks a takeover
    for _ in range(3):  # the control thread may consume the request first (16 ms tick): retry
        _enter(ctl, beat)
        ctl.request_mode("manual")
        ok, msg = ctl.request_takeover(ctl.stop_count)
        if not ok:
            break
    check("(review 1) takeover refused while another mode change is pending", not ok and "pending" in msg, msg)
    time.sleep(0.1)

    # (5) STOP right after enter_policy: the switch into policy mode never happens
    # (the control thread ticks every 16 ms; if it slips in between the two calls, settle and retry)
    beat.on = True
    for _ in range(3):
        ok, _ = ctl.enter_policy(ctl.stop_count, 90.0, None)
        ctl.stop()
        seen = _modes_for(ctl, 0.2)
        if ok and "policy" not in seen:
            break
        ctl.stop()
        time.sleep(0.05)
    check("(review 5) STOP right after enter_policy wins", ok and "policy" not in seen, sorted(seen))
    # (5) STOP that lands between the control thread taking the request and switching (stop_count re-check)
    for _ in range(3):
        ok, _ = ctl.enter_policy(ctl.stop_count, 90.0, None)
        with ctl.lock:
            ctl.stop_count += 1  # a STOP whose req_mode write the control thread already missed
        seen = _modes_for(ctl, 0.15)
        if ok and "policy" not in seen:
            break
        ctl.stop()
        time.sleep(0.05)
    check("(review 5) control thread re-checks stop_count before entering policy mode", ok and "policy" not in seen,
          sorted(seen))
    for _ in range(3):
        ctl.request_mode("manual")
        ok, msg = ctl.enter_policy(ctl.stop_count, 90.0, None)
        if not ok:
            break
    check("(review 5) enter_policy refused while a user mode change is pending", not ok and "pending" in msg, msg)
    time.sleep(0.1)

    # (2) follower reads failing in policy mode -> hold + fault within ~0.1 s
    _enter(ctl, beat)
    bus = ctl.follower.bus
    real_read = bus.sync_read

    def failing(*a, **k):
        raise ConnectionError("mock: no status packet")
    bus.sync_read = failing
    t0 = time.monotonic()
    while ctl.mode == "policy" and time.monotonic() - t0 < 2:
        time.sleep(0.002)
    dt = time.monotonic() - t0
    bus.sync_read = real_read
    time.sleep(0.1)
    check("(review 2) follower read failures in policy mode -> hold + fault", ctl.snapshot()[3] == "manual"
          and dt < 0.3 and "reads failing" in (ctl.policy_fault or ""), f"{dt * 1000:.0f} ms: {ctl.policy_fault}")
    # (2) the runner's control-loop stall watch
    w = policy_runner._TickWatch()
    w.check(5, 100.0)
    w.check(6, 100.2)
    try:
        w.check(6, 100.5)
        stalled = None
    except policy_runner._Fault as e:
        stalled = str(e)
    check("(review 2) runner faults when the control tick stops advancing", stalled and "stalled" in stalled, stalled)
    # (10) non-finite manual targets are rejected
    ok, msg = ctl.set_targets({"shoulder_pan": float("nan")})
    check("(review 10) NaN target rejected", not ok and math.isfinite(ctl.state()["targets"]["shoulder_pan"]), msg)
    beat.on = False
    ctl.stop()
    time.sleep(0.1)


# ------------------------------------------------------------------------------------- part 2: HTTP
@section
def listing_and_validation():
    code, html = api("GET", "/")
    check("index page served", code == 200 and b"<html" in html[:500].lower())
    s = state()
    check("mock mode, manual, no policy", s["mock"] and s["mode"] == "manual" and s["policy"]["phase"] == "none",
          (s["mock"], s["mode"], s["policy"]["phase"]))
    code, d = api("GET", "/api/policies")
    ps = {p["name"]: p for p in d["policies"]}
    need = {"name", "path", "type", "size_mb", "steps", "dataset", "created", "chunk_size", "n_action_steps",
            "image_keys", "deploy", "undistort", "runnable", "why_not", "tally"}
    check("listing fields", all(need <= set(p) for p in ps.values()), sorted(need - set(next(iter(ps.values())))))
    a = ps.get("act_real_v0", {})
    check("act_real_v0 runnable", a.get("runnable") and a.get("type") == "act" and a.get("chunk_size") == 50
          and a.get("steps") == 30000 and a.get("size_mb", 0) > 200, {k: a.get(k) for k in ("runnable", "why_not", "steps", "size_mb")})
    u = ps.get("act_undistort", {})
    check("sidecar read (undistort badge, start pose)", u.get("undistort") is True and u.get("deploy", {}).get("max_speed") == 100)
    check("diffusion not runnable", ps.get("diffusion_fake", {}).get("runnable") is False
          and "chunking" in (ps["diffusion_fake"]["why_not"] or ""), ps.get("diffusion_fake", {}).get("why_not"))
    check("unmapped camera1 not runnable", ps.get("camera1", {}).get("runnable") is False
          and "camera_map" in (ps["camera1"]["why_not"] or ""), ps.get("camera1", {}).get("why_not"))
    check("camera_map makes camera1 runnable", ps.get("camera1_mapped", {}).get("runnable") is True,
          ps.get("camera1_mapped", {}).get("why_not"))
    check("checkpoints/last/pretrained_model layout", ps.get("nested_run", {}).get("runnable")
          and ps["nested_run"]["ckpt"].endswith("checkpoints/last/pretrained_model"))
    check("remotes listed without authkey", d["remotes"] == [{"name": "tcp-local", "address": d["remotes"][0]["address"]}]
          and "authkey" not in json.dumps(d["remotes"]), d["remotes"])
    check("/api/mode policy -> 400", api("POST", "/api/mode", {"mode": "policy"})[0] == 400)
    check("start without a policy -> 400", api("POST", "/api/policy/start", {})[0] == 400)
    check("load non-runnable -> 400", api("POST", "/api/policy/load", {"name": "diffusion_fake"})[0] == 400)
    check("load unknown -> 400", api("POST", "/api/policy/load", {"name": "nope"})[0] == 400)
    check("load path traversal -> 400", api("POST", "/api/policy/load", {"name": "../x"})[0] == 400)
    check("unknown policy endpoint -> 404", api("POST", "/api/policy/bogus", {})[0] == 404)
    for ep in ("takeover", "resume", "outcome", "discard"):
        check(f"{ep} without an episode -> 400", api("POST", f"/api/policy/{ep}", {"outcome": "success"})[0] == 400)
    code, r = api("POST", "/api/target", {"targets": {"shoulder_pan": float("nan")}})
    check("(review 10) /api/target NaN -> 400", code == 400, r)


@section
def load_act():
    s, dt = load("act_real_v0")
    p = s["policy"]
    METRICS["load_s"] = round(dt, 1)
    METRICS["warmup_ms"] = (p["info"] or {}).get("warmup_ms")
    check("load act_real_v0 -> ready", p["phase"] == "ready" and p["backend"] == "local", f"{dt:.1f} s: {p['message']}")
    check("status has info + start pose", p["info"]["type"] == "act" and len(p["start_pose"]) == 6)
    check("one local worker", len(worker_pids(APP.pid)) == 1)


@section
def dry_run():
    check("record + dry run -> 400", start({"dry_run": True, "record": {"dataset": "e2e_x"}}, ok=False)[0] == 400)
    code, r = api("POST", "/api/policy/start", {"time_limit_s": 2, "home_first": False})
    s = wait(lambda s: s["policy"]["phase"] in ("running", "pending"), 10, "started without dry_run")
    check("(review 8) start without dry_run is a dry run", code == 200 and s["policy"]["dry_run"] is True
          and s["mode"] == "manual", (code, s["policy"]["dry_run"], s["mode"]))
    wait(phase_is("pending"), 6, "pending")
    label("none")
    f0 = state()["follower"]
    start({"dry_run": True, "time_limit_s": 4, "home_first": True})
    s = wait(lambda s: s["policy"]["target"] is not None, 10, "dry-run target")
    check("dry run: target shown, controller stays manual", s["mode"] == "manual" and s["policy"]["dry_run"])
    drift = 0.0
    while pol()["phase"] == "running":
        f = state()["follower"]
        drift = max(drift, max(abs(f[j] - f0[j]) for j in ARM))
        time.sleep(0.1)
    p = pol()
    check("dry run: the arm never moved (home_first ignored)", drift < 1.0, f"max drift {drift:.2f} deg")
    check("dry run ends at the time limit -> pending", p["phase"] == "pending" and "Time limit" in p["message"]
          and p["chunks"] >= 2, f"{p['message']} chunks={p['chunks']}")
    s = label("none")
    check("label -> ready", s["policy"]["phase"] == "ready" and s["policy"]["last_outcome"] == "none")


@section
def real_run_and_stop():
    start({"dry_run": False, "max_speed": 90, "time_limit_s": 30, "home_first": True})
    phases = set()
    t_end = time.monotonic() + 30
    while time.monotonic() < t_end:
        p = pol()
        phases.add(p["phase"])
        if p["phase"] == "running" and p["chunks"] >= 4:
            break
        time.sleep(0.05)
    s = state()
    p = s["policy"]
    check("homing then running in policy mode", "homing" in phases and p["phase"] == "running" and s["mode"] == "policy",
          sorted(phases))
    sp = p["start_pose"]
    moved = max(abs(s["follower"][j] - sp[i]) for i, j in enumerate(JOINTS[:5]))
    check("the policy moves the (mock) arm", moved > 2.0, f"{moved:.1f} deg from the start pose")
    time.sleep(1.0)
    p = pol()
    METRICS.update(run_hz=p["hz"], infer_ms_avg=p["infer_ms"]["avg"], infer_ms_max=p["infer_ms"]["max"],
                   rtt_ms=p["rtt_ms"], starved=p["starved"], late_ticks=p["late"])
    check("loop at ~30 Hz, not starved", 27 <= p["hz"] <= 33 and p["starved"] == 0,
          f"hz {p['hz']}, infer {p['infer_ms']}, rtt {p['rtt_ms']} ms, queue {p['queue']}, starved {p['starved']}, late {p['late']}")
    t0 = time.monotonic()
    api("POST", "/api/stop")
    dt = time_to(lambda s: s["mode"] == "manual")
    METRICS["stop_to_hold_ms"] = round((time.monotonic() - t0) * 1000 if dt is None else dt * 1000, 1)
    check("STOP -> manual hold", dt is not None and dt < 0.2, f"{METRICS['stop_to_hold_ms']} ms (incl. HTTP polling)")
    s = wait(phase_is("pending"), 3, "pending after STOP")
    check("episode pending after STOP", "STOP" in s["policy"]["message"], s["policy"]["message"])
    check("arm holds after STOP", max_drift() < 1.0)
    s = label("fail")
    check("fail label counted", s["policy"]["tally"] == {"success": 0, "fail": 1, "total": 1}, s["policy"]["tally"])


@section
def time_limit_and_settings():
    start({"time_limit_s": 3, "home_first": False})
    s = wait(phase_is("pending"), 10, "time limit")
    p = s["policy"]
    check("time limit ends the episode", "Time limit" in p["message"] and 2.95 <= p["elapsed_s"] <= 3.2,
          f"{p['elapsed_s']} s")
    check("hold after the time limit", s["mode"] == "manual" and max_drift() < 1.0)
    s = label("success")
    check("success label counted", s["policy"]["tally"] == {"success": 1, "fail": 1, "total": 2}, s["policy"]["tally"])
    for body, why in (({"max_speed": 1000}, "max speed"), ({"time_limit_s": 0.5}, "time limit"),
                      ({"n_action_steps": 51}, "chunk steps"), ({"n_action_steps": "x"}, "numbers"),
                      ({"record": {"dataset": "bad name!"}}, "dataset name")):
        code, r = start(body, ok=False)
        check(f"start validation: {why}", code == 400 and why in r["message"], r.get("message"))
    start({"time_limit_s": 2, "home_first": False, "n_action_steps": 10, "max_speed": 45})
    s = wait(phase_is("running"), 5, "running")
    check("settings applied", s["policy"]["max_speed"] == 45 and s["policy"]["n_action_steps"] == 10)
    check("second start while running -> 400", start({}, ok=False)[0] == 400)
    check("load while running -> 400", api("POST", "/api/policy/load", {"name": "act_real_v0"})[0] == 400)
    check("unload while running -> 400", api("POST", "/api/policy/unload")[0] == 400)
    wait(phase_is("pending"), 6, "pending")
    label("none")



def target_jumps(seconds):
    """Largest change of any policy target joint between consecutive ~50 ms samples while an episode runs."""
    tg, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        p = pol()
        if p.get("target"):
            tg.append([p["target"][j] for j in JOINTS])
        time.sleep(0.05)
    return float(np.abs(np.diff(np.array(tg), axis=0)).max()) if len(tg) > 2 else None


@section
def ensemble_smoothing():
    code, r = start({"smoothing": "bogus"}, ok=False)
    check("start validation: smoothing", code == 400 and "smoothing" in r["message"], r.get("message"))
    # reference: the default chunk-after-chunk execution (chunk steps 10 = re-plans most often)
    start({"time_limit_s": 4, "home_first": False, "smoothing": "chunks", "n_action_steps": 10})
    wait(lambda s: s["policy"]["phase"] == "running" and s["mode"] == "policy" and s["policy"]["chunks"] >= 2, 10, "chunks running")
    METRICS["chunks_max_target_step_deg"] = target_jumps(2.0)
    wait(phase_is("pending"), 6, "pending")
    label("none")
    # (review) chunk steps is unused with ensemble: an out-of-range value must not block the start, and is not reported
    start({"time_limit_s": 4, "home_first": False, "smoothing": "ensemble", "n_action_steps": 999})
    s = wait(lambda s: s["policy"]["phase"] == "running" and s["mode"] == "policy" and s["policy"]["chunks"] >= 2, 10,
             "ensemble running")
    check("ensemble: setting applied, chunk steps ignored", s["policy"].get("smoothing") == "ensemble"
          and s["policy"].get("n_action_steps") is None, (s["policy"].get("smoothing"), s["policy"].get("n_action_steps")))
    c0, t0 = s["policy"]["chunks"], time.monotonic()
    jump = target_jumps(2.0)
    p = pol()
    rate = (p["chunks"] - c0) / (time.monotonic() - t0)
    METRICS["ensemble_chunks_per_s"] = round(rate, 1)
    METRICS["ensemble_max_target_step_deg"] = jump
    check("ensemble: a fresh chunk every 2-3 ticks (>= 8 chunks/s)", rate >= 8, f"{rate:.1f}/s")
    check("ensemble: never starved", p["starved"] == 0, p["starved"])
    check("ensemble: targets change smoothly (< 10 deg per 50 ms)", jump is not None and jump < 10, jump)
    s = wait(phase_is("pending"), 6, "pending")
    check("ensemble: hold after the episode", s["mode"] == "manual" and max_drift() < 1.0)
    check("ensemble: recorded in the episode settings", s["policy"].get("smoothing") == "ensemble")
    label("none")


@section
def takeover_resume():
    start({"time_limit_s": 30, "home_first": False})
    wait(lambda s: s["policy"]["phase"] == "running" and s["policy"]["chunks"] >= 2, 10, "running")
    code, r = api("POST", "/api/policy/takeover")
    s = wait(lambda s: s["policy"]["phase"] == "intervention" and s["mode"] == "teleop", 2, "intervention")
    check("takeover -> teleop, target cleared", code == 200 and s["policy"]["target"] is None)
    s = wait(lambda s: s["synced"], 10, "follower synced to the leader")
    check("follower syncs to the leader (existing teleop sync)", s["synced"])
    c0 = s["policy"]["chunks"]
    code, r = api("POST", "/api/policy/resume")
    s = wait(lambda s: s["policy"]["phase"] == "running" and s["mode"] == "policy" and s["policy"]["chunks"] > c0, 5,
             "resumed with a fresh chunk")
    check("resume -> policy mode, fresh chunk", code == 200 and s["policy"]["interventions"] == 1)
    code, r = api("POST", "/api/policy/stop")
    s = wait(phase_is("pending"), 3, "pending")
    check("policy stop -> pending + hold", s["mode"] == "manual" and s["policy"]["message"].find("Stopped") >= 0
          and max_drift() < 1.0, s["policy"]["message"])
    label("none")
    # (review 1) takeover and STOP arriving together: STOP wins, the arm is never handed to the leader
    start({"time_limit_s": 30, "home_first": False})
    wait(lambda s: s["policy"]["phase"] == "running" and s["mode"] == "policy" and s["policy"]["chunks"] >= 2, 10, "running")
    t = threading.Thread(target=api, args=("POST", "/api/policy/takeover"))
    t.start()
    api("POST", "/api/stop")
    t.join()
    time.sleep(0.1)  # a takeover the runner handled before the STOP may legitimately show teleop for a tick or two
    modes, t0 = set(), time.monotonic()
    while time.monotonic() - t0 < 0.8:
        modes.add(state()["mode"])
        time.sleep(0.02)
    s = wait(phase_is("pending"), 3, "pending")
    check("(review 1) takeover + STOP together -> manual hold after the STOP, never teleop", s["mode"] == "manual"
          and modes == {"manual"} and "STOP" in s["policy"]["message"], f"{sorted(modes)}: {s['policy']['message']}")
    label("none")


@section
def worker_killed():
    start({"time_limit_s": 30, "home_first": False})
    wait(lambda s: s["policy"]["phase"] == "running" and s["policy"]["chunks"] >= 2, 10, "running")
    pids = worker_pids(APP.pid)
    check("found the local worker", len(pids) == 1, pids)
    t0 = time.monotonic()
    os.kill(pids[0], signal.SIGKILL)
    dt = time_to(lambda s: s["mode"] == "manual")
    METRICS["worker_kill_to_hold_ms"] = None if dt is None else round(dt * 1000, 1)
    s = wait(phase_is("pending"), 3, "pending after worker death")
    f = s["policy"]["fault"] or ""
    check("worker killed -> hold + fault", dt is not None and dt < 0.5 and ("worker exited" in f or "lost the policy" in f),
          f"{METRICS['worker_kill_to_hold_ms']} ms: {f}")
    check("arm holds after the fault", max_drift() < 1.0)
    s = label("none")
    check("policy needs reloading", s["policy"]["phase"] == "error", s["policy"]["message"])
    code, r = start({}, ok=False)
    check("start refused until reloaded", code == 400, r.get("message"))
    s, _ = load("act_real_v0")
    check("reload after a crash", s["policy"]["phase"] == "ready")


@section
def teleop_recording_exclusive():
    start({"time_limit_s": 30, "home_first": False})
    wait(phase_is("running"), 10, "running")
    code, r = api("POST", "/api/record/start", {"dataset": "e2e_teleop", "task": "test task", "num_episodes": 1,
                                                "manual_start": True})
    check("teleop recording refused during a policy episode", code == 400 and "policy" in r["message"], r.get("message"))
    api("POST", "/api/policy/stop")
    wait(phase_is("pending"), 3, "pending")
    label("none")
    code, r = api("POST", "/api/record/start", {"dataset": "e2e_teleop", "task": "test task", "num_episodes": 1,
                                                "manual_start": True})
    check("teleop recording starts when no episode runs", code == 200, r)
    wait(lambda s: s["recording"]["phase"] in ("ready", "waiting"), 30, "recorder ready")
    code, r = start({}, ok=False)
    check("policy start refused during teleop recording", code == 400 and "recording" in r["message"], r.get("message"))
    code, r = api("POST", "/api/policy/home", {})
    check("go to start pose refused during teleop recording", code == 400, r.get("message"))
    api("POST", "/api/record/stop")
    wait(lambda s: s["recording"]["phase"] == "idle", 30, "recorder idle")
    # the refactored recorder still records teleop episodes (manual hold here: any non-limp mode records)
    code, r = api("POST", "/api/record/start", {"dataset": "e2e_teleop", "task": "test task", "num_episodes": 1,
                                                "episode_s": 2, "reset_s": 0})
    s = wait(lambda s: s["recording"]["phase"] == "idle" and "Session finished" in s["recording"]["message"], 60,
             "teleop episode recorded")
    info = json.loads((DATA / "e2e_teleop" / "meta" / "info.json").read_text())
    q = s["recording"].get("last") or {}
    check("teleop recording still works (refactored recorder)", code == 200 and info["total_episodes"] == 1
          and 55 <= info["total_frames"] <= 61 and q.get("frames") == info["total_frames"],
          f"{info['total_episodes']} episode, {info['total_frames']} frames, quality {q}")


AUTO_DS = "rollouts_act_real_v0"


@section
def auto_recording():
    """A live start that names no dataset records to rollouts_<policy>; dry runs and "record": false don't."""
    s = api("GET", "/api/state")[1]
    check("auto_record names the default dataset", s["policy"].get("auto_record") == AUTO_DS, s["policy"].get("auto_record"))
    code, r = api("POST", "/api/policy/start", {"dry_run": False, "time_limit_s": 3, "home_first": False})
    check("live start without a dataset is accepted", code == 200, r)
    s = wait(phase_is("pending"), 20, "pending")
    rec = s["policy"]["record"]
    check("live run records to rollouts_<policy> by default", rec and rec["dataset"] == AUTO_DS and rec["pending"], rec)
    s = label("fail")
    check("auto-recorded rollout saved with its label", f"saved as episode 0 of {AUTO_DS}" in s["policy"]["message"],
          s["policy"]["message"])
    side = [json.loads(l) for l in (DATA / AUTO_DS / "meta" / "so101_rollouts.jsonl").read_text().splitlines()]
    check("auto dataset sidecar", len(side) == 1 and side[0]["outcome"] == "fail" and side[0]["policy"] == "act_real_v0",
          [(x["episode_index"], x["outcome"]) for x in side])
    for body, what in (({"dry_run": True}, "dry run"), ({"dry_run": False, "record": False}, '"record": false')):
        code, r = api("POST", "/api/policy/start", {**body, "time_limit_s": 2, "home_first": False})
        s = wait(phase_is("pending"), 20, "pending")
        check(f"{what}: not recorded", code == 200 and s["policy"]["record"] is None, s["policy"]["record"])
        label("none")
    n = len((DATA / AUTO_DS / "meta" / "so101_rollouts.jsonl").read_text().splitlines())
    check("auto dataset unchanged by unrecorded runs", n == 1, n)


@section
def rollout_recording():
    start({"record": {"dataset": "e2e_rollouts"}, "time_limit_s": 6, "home_first": True})
    wait(lambda s: s["policy"]["phase"] == "running" and s["policy"]["chunks"] >= 1, 40, "recording episode running")
    time.sleep(1.5)
    api("POST", "/api/policy/takeover")
    wait(phase_is("intervention"), 2, "intervention")
    time.sleep(1.0)
    api("POST", "/api/policy/resume")
    s = wait(phase_is("pending"), 15, "pending")
    rec = s["policy"]["record"]
    check("rollout pending with frames", rec and rec["dataset"] == "e2e_rollouts" and rec["pending"]
          and 170 <= rec["frames"] <= 181, rec)
    frames0 = rec["frames"]
    code, r = api("POST", "/api/record/start", {"dataset": "e2e_teleop", "task": "test task"})
    check("teleop recording refused while a rollout awaits its label", code == 400 and "label" in r["message"], r.get("message"))
    s = label("success", note="e2e rollout")
    check("rollout saved with the label", "saved as episode 0 of e2e_rollouts" in s["policy"]["message"]
          and s["policy"]["record"]["saved"] == 1, s["policy"]["message"])
    side = (DATA / "e2e_rollouts" / "meta" / "so101_rollouts.jsonl").read_text().splitlines()
    e = json.loads(side[0])
    iv = e["intervention_frames"]
    check("so101_rollouts.jsonl: outcome + intervention frames", len(side) == 1 and e["episode_index"] == 0
          and e["outcome"] == "success" and e["policy"] == "act_real_v0" and len(iv) == 1
          and 30 <= iv[0][0] < iv[0][1] < frames0 and 20 <= iv[0][1] - iv[0][0] <= 40, {k: e[k] for k in ("episode_index", "outcome", "intervention_frames")})
    ev = json.loads((POL / "act_real_v0" / "evals.jsonl").read_text().splitlines()[-1])
    check("evals.jsonl line", ev["dataset"] == "e2e_rollouts" and ev["episode_index"] == 0 and ev["outcome"] == "success"
          and ev["interventions"] == 1 and ev["mock"] is True and ev["note"] == "e2e rollout"
          and set(ev) >= {"t", "policy", "episode", "outcome", "duration_s", "interventions", "dry_run", "dataset",
                          "episode_index", "settings", "note"}, ev)
    # discard
    start({"record": {"dataset": "e2e_rollouts"}, "time_limit_s": 3, "home_first": False})
    wait(phase_is("pending"), 20, "pending")
    code, r = api("POST", "/api/policy/discard")
    s = wait(lambda s: s["policy"]["phase"] == "ready" and not s["policy"]["saving"], 30, "discarded")
    check("discard drops the pending rollout", code == 200 and s["policy"]["last_outcome"] == "discarded"
          and s["policy"]["record"]["saved"] == 1)
    # a new start saves the pending one unlabeled
    start({"record": {"dataset": "e2e_rollouts"}, "time_limit_s": 3, "home_first": False})
    wait(phase_is("pending"), 20, "pending")
    start({"time_limit_s": 2, "home_first": False})
    wait(phase_is("running"), 30, "running after auto-save")
    wait(phase_is("pending"), 6, "pending")
    label("none")
    side = [json.loads(l) for l in (DATA / "e2e_rollouts" / "meta" / "so101_rollouts.jsonl").read_text().splitlines()]
    check("starting again saved the pending rollout unlabeled", len(side) == 2 and side[1]["episode_index"] == 1
          and side[1]["outcome"] == "none", [(x["episode_index"], x["outcome"]) for x in side])
    # reload the dataset with LeRobot
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset("local/e2e_rollouts", root=DATA / "e2e_rollouts", video_backend="pyav")
    n0 = side[0]["frames"]
    check("LeRobotDataset reload", ds.meta.total_episodes == 2 and abs(ds.num_frames - (n0 + side[1]["frames"])) == 0,
          f"{ds.meta.total_episodes} episodes, {ds.num_frames} frames")
    a, b = ds[0], ds[ds.num_frames - 1]
    check("frames decode (state, action, 2 cameras)", tuple(a["observation.images.wrist"].shape) == (3, 480, 640)
          and tuple(b["observation.images.scene"].shape) == (3, 480, 640) and tuple(a["action"].shape) == (6,)
          and a["task"] == "Put the cube in the cup")


@section
def policy_view_undistort():
    s, _ = load("act_undistort")
    p = s["policy"]
    check("undistort policy loaded", p["phase"] == "ready" and p["undistort"] == ["scene", "wrist"]
          and "undistorted" in p["message"] and p["start_pose"] == [0.0, -100.0, 90.0, 60.0, 0.0, 20.0], p["message"])
    raw = grab_jpeg("/stream/wrist.mjpg")
    und = grab_jpeg("/stream/wrist.mjpg?view=policy")
    diff = float(np.abs(raw.astype(int) - und.astype(int)).mean()) if raw is not None and und is not None else -1
    check("?view=policy stream is undistorted", und is not None and und.shape == (480, 640, 3) and diff > 3,
          f"mean |raw - policy view| = {diff:.1f}")
    start({"dry_run": True, "time_limit_s": 3})
    s = wait(phase_is("running"), 5, "running")
    check("sidecar defaults (max speed)", s["policy"]["max_speed"] == 100, s["policy"]["max_speed"])
    s = wait(phase_is("pending"), 8, "pending")
    check("policy runs on undistorted frames", s["policy"]["chunks"] >= 2 and not s["policy"]["fault"])
    label("none")


@section
def remote_backend(remote_port):
    env = dict(os.environ, SO101_POLICY_KEY="e2e-key")
    srv = subprocess.Popen([sys.executable, str(WEB / "policy_worker.py"), "--ckpt", str(POL / "act_real_v0"),
                            "--address", f"127.0.0.1:{remote_port}"], stdout=subprocess.PIPE,
                           stderr=open(LOGS / "e2e_remote.log", "w"), env=env, start_new_session=True)
    try:
        line = ""
        t_end = time.monotonic() + 300
        while time.monotonic() < t_end and not line.startswith(("READY", "ERROR")):
            line = srv.stdout.readline().decode()
            if not line and srv.poll() is not None:
                break
        check("remote server READY", line.startswith("READY"), line[:60])
        s, dt = load(remote="tcp-local")
        p = s["policy"]
        check("load remote -> ready", p["phase"] == "ready" and p["backend"] == f"remote:127.0.0.1:{remote_port}",
              f"{dt:.1f} s: {p['message']}")
        time.sleep(1.0)
        check("local worker stopped when switching to the remote", worker_pids(APP.pid) == [])
        start({"time_limit_s": 3, "home_first": False})
        s = wait(phase_is("pending"), 15, "pending")
        p = s["policy"]
        METRICS["remote_rtt_ms"] = p["rtt_ms"]
        check("episode over TCP + JPEG", p["chunks"] >= 2 and not p["fault"] and p["starved"] == 0,
              f"chunks {p['chunks']}, infer {p['infer_ms']}, rtt {p['rtt_ms']} ms")
        label("success")
        ev = POL / "_remote" / "tcp-local" / "evals.jsonl"
        check("remote evals logged under _remote/", ev.exists() and json.loads(ev.read_text().splitlines()[-1])["outcome"] == "success")
        start({"time_limit_s": 30, "home_first": False})
        wait(lambda s: s["policy"]["phase"] == "running" and s["policy"]["chunks"] >= 2, 10, "running")
        srv.kill()
        t0 = time.monotonic()
        dt = time_to(lambda s: s["mode"] == "manual", timeout=5)
        METRICS["remote_kill_to_hold_ms"] = None if dt is None else round(dt * 1000, 1)
        s = wait(phase_is("pending"), 5, "pending")
        f = s["policy"]["fault"] or ""
        check("remote server killed -> hold + fault", dt is not None and dt < 2.5 and ("lost the policy" in f or "starved" in f),
              f"{METRICS['remote_kill_to_hold_ms']} ms: {f}")
        check("arm holds", max_drift() < 1.0)
        label("none")
        s = wait(lambda s: s["policy"]["backend_ok"] is False, 10, "monitor notices the dead server")
        check("monitor reports the dead remote", "not responding" in s["policy"]["message"]
              and "not responding" in (s["policy"]["backend_error"] or ""), s["policy"]["message"])
        code, _ = api("POST", "/api/policy/unload")
        s = wait(phase_is("none"), 5, "unloaded")
        check("unload remote", code == 200 and s["policy"]["name"] is None)
    finally:
        if srv.poll() is None:
            srv.kill()
        srv.wait(5)


@section
def go_to_start_pose():
    pose = [0.0, -90.0, 80.0, 60.0, 0.0, 30.0]
    code, r = api("POST", "/api/policy/home", {"pose": pose})
    check("home accepted without a policy loaded", code == 200, r)
    wait(phase_is("homing"), 2, "homing")
    s = wait(lambda s: s["policy"]["phase"] != "homing", 30, "homing done")
    err = max(abs(s["follower"][j] - pose[i]) for i, j in enumerate(ARM))
    check("arrives at the pose and holds", s["mode"] == "manual" and err < 3.0 and "start pose" in s["policy"]["message"],
          f"max error {err:.1f} deg: {s['policy']['message']}")
    check("bad pose -> 400", api("POST", "/api/policy/home", {"pose": [1, 2, 3]})[0] == 400)
    api("POST", "/api/policy/home", {"pose": [40.0, -40.0, 40.0, 20.0, 30.0, 60.0]})
    wait(phase_is("homing"), 2, "homing")
    time.sleep(0.5)
    api("POST", "/api/stop")
    dt = time_to(lambda s: s["mode"] == "manual")
    s = wait(lambda s: s["policy"]["phase"] != "homing", 2, "homing stopped")
    check("STOP interrupts go-to-start-pose", dt is not None and dt < 0.2 and "STOP" in s["policy"]["message"],
          s["policy"]["message"])
    check("holds after STOP", max_drift() < 1.0)


@section
def unload_and_shutdown():
    s, _ = load("act_real_v0")
    pids = worker_pids(APP.pid)
    code, _ = api("POST", "/api/policy/unload")
    s = wait(phase_is("none"), 5, "unloaded")
    t0 = time.monotonic()
    while worker_pids(APP.pid) and time.monotonic() - t0 < 8:
        time.sleep(0.2)
    check("unload stops the worker", code == 200 and pids and not worker_pids(APP.pid), pids)
    load("act_real_v0")
    api("POST", "/api/policy/home", {"pose": [40.0, -40.0, 40.0, 20.0, 30.0, 60.0]})
    wait(lambda s: s["policy"]["phase"] != "homing", 30, "away from the start pose")
    start({"time_limit_s": 30, "home_first": True, "record": {"dataset": "e2e_rollouts"}})
    wait(lambda s: s["policy"]["phase"] == "homing" and s["mode"] == "policy", 15, "homing motion")
    time.sleep(0.3)
    api("POST", "/api/stop")
    s = wait(lambda s: s["policy"]["phase"] not in ("homing", "running"), 3, "cancelled")
    check("STOP during homing cancels the episode (nothing to label)", s["policy"]["phase"] == "ready"
          and "cancelled" in s["policy"]["message"] and s["mode"] == "manual", s["policy"]["message"])
    start({"time_limit_s": 30, "home_first": False, "record": {"dataset": "e2e_rollouts"}})
    wait(lambda s: s["policy"]["phase"] == "running" and s["policy"]["chunks"] >= 2, 30, "running")
    time.sleep(1.5)
    pids = worker_pids(APP.pid)
    t0 = time.monotonic()
    APP.send_signal(signal.SIGTERM)
    try:
        code = APP.wait(60)
    except subprocess.TimeoutExpired:
        code = None
    METRICS["shutdown_s"] = round(time.monotonic() - t0, 1)
    time.sleep(0.5)
    alive = [p for p in pids if os.path.exists(f"/proc/{p}")]
    socks = [p for p in os.listdir("/tmp") if p.startswith(f"so101_policy_{APP.pid}_")]
    check("SIGTERM mid-episode: clean exit, no worker or socket left", code == 0 and not alive and not socks,
          f"exit {code} in {METRICS['shutdown_s']} s, alive {alive}, sockets {socks}")
    side = [json.loads(l) for l in (DATA / "e2e_rollouts" / "meta" / "so101_rollouts.jsonl").read_text().splitlines()]
    check("pending rollout saved unlabeled at shutdown", len(side) == 3 and side[-1]["note"] == "saved at shutdown",
          [(x["episode_index"], x["outcome"], x.get("note")) for x in side])


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8092)
    ap.add_argument("--remote-port", type=int, default=8093)
    ap.add_argument("--skip-controller", action="store_true")
    args = ap.parse_args()
    if os.environ.get("SO101_MOCK") != "1":
        raise SystemExit("refusing to run without SO101_MOCK=1")
    BASE = f"http://127.0.0.1:{args.port}"
    make_policy_root(args.remote_port)
    if not args.skip_controller:
        controller_safety()
    start_app(args.port)
    try:
        listing_and_validation()
        load_act()
        dry_run()
        real_run_and_stop()
        time_limit_and_settings()
        ensemble_smoothing()
        takeover_resume()
        worker_killed()
        teleop_recording_exclusive()
        rollout_recording()
        auto_recording()
        policy_view_undistort()
        remote_backend(args.remote_port)
        go_to_start_pose()
        unload_and_shutdown()
    finally:
        if APP.poll() is None:
            APP.send_signal(signal.SIGTERM)
            try:
                APP.wait(30)
            except subprocess.TimeoutExpired:
                APP.kill()
    log = (LOGS / "e2e_app.log").read_text()
    tracebacks = len(re.findall(r"^Traceback", log, re.M))
    check("no unexpected tracebacks in the app log", tracebacks == 0, f"{tracebacks} in {LOGS / 'e2e_app.log'}")
    print("\nmetrics:", json.dumps(METRICS))
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\ntest_e2e: {len(RESULTS) - len(failed)} passed, {len(failed)} failed"
          + (f"  (FAILED: {'; '.join(failed)})" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""UI stub for the SO-101 console: serves web/index.html with canned API responses. No hardware, stdlib only.

    python3 ui_stub/stub.py [--port 8199]
    open http://127.0.0.1:8199/?phase=running      (sets the scenario, then serves the page)

Scenario: `?phase=` on the page URL, GET /_phase?p=<name>, or the file ui_stub/phase.txt (read at startup).
Scenarios: none loading ready homing running running_dry intervention pending error fault rec_session limp.
`?selftest=1` injects ui_stub/selftest.js (keyboard model checks; results land in #selftest-out).
POSTs are answered {"ok": true} and appended to ui_stub/posts.log; GET /_posts returns them.
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
WEB = HERE.parent / "web"
LOG = HERE / "posts.log"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SCENARIOS = ["none", "loading", "ready", "homing", "running", "running_dry", "intervention", "pending", "error",
             "fault", "rec_session", "limp"]
CUR = {"name": "none"}
POSTS = []

POSE = dict(zip(JOINTS, [2.4, -98.7, 91.2, 55.0, -1.3, 18.0]))
LIMITS = {j: ([0.0, 100.0] if j == "gripper" else [-150.0, 150.0]) for j in JOINTS}
INFO = {"name": "act_real_v0", "type": "act", "ckpt": "/home/asnano/so101_policies/act_real_v0", "device": "cuda",
        "input_features": {"observation.state": [6], "observation.images.wrist": [3, 480, 640],
                           "observation.images.scene": [3, 480, 640]},
        "output_features": {"action": [6]}, "chunk_size": 50, "n_action_steps": 50, "fps": 30,
        "image_keys": ["observation.images.wrist", "observation.images.scene"],
        "deploy": {"max_speed": 120, "time_limit_s": 30, "task": "Put the cube in the cup",
                   "notes": "real-only ACT, 50 demos; raw frames"}, "warmup_ms": 830.0}
TALLY = {"success": 7, "fail": 2, "total": 10}


def policy(phase, **kw):
    p = {"phase": phase, "name": None, "backend": None, "info": None, "dry_run": False, "episode": 0,
         "elapsed_s": 0.0, "time_limit_s": 30.0, "max_speed": 120.0, "infer_ms": {"last": None, "avg": None, "max": None},
         "chunks": 0, "queue": 0, "hz": 0.0, "starved": 0, "target": None, "message": "", "fault": None,
         "record": None, "tally": {"success": 0, "fail": 0, "total": 0}, "last_outcome": None}
    if phase not in ("none",):
        p.update(name="act_real_v0", backend="local", tally=dict(TALLY))
    if phase not in ("none", "loading"):
        p["info"] = INFO
    p.update(kw)
    return p


def target(off):
    return {j: round(v + off * (i + 1), 1) for i, (j, v) in enumerate(POSE.items())}


def scenario(name):
    s = {"mode": "manual", "torque": True, "synced": False, "follower": dict(POSE), "leader": target(-1.5),
         "targets": dict(POSE), "limits": LIMITS, "leader_connected": True, "hz": 59.7, "errors": 0,
         "message": "Holding current pose",
         "feeds": {"wrist": {"status": "ok"}, "scene": {"status": "ok"},
                   "depth": {"status": "disabled (SO101_DEPTH=0)"}},
         "recording": {"phase": "idle", "message": ""}}
    run = dict(episode=4, elapsed_s=12.4, infer_ms={"last": 81.5, "avg": 84.2, "max": 131.0}, chunks=9, queue=23,
               hz=29.9, starved=0)
    if name == "none":
        s["policy"] = policy("none")
    elif name == "loading":
        s["policy"] = policy("loading", message="Loading act_real_v0: starting the policy worker (5-10 s)...")
    elif name == "ready":
        s["policy"] = policy("ready", message="Ready (warm-up 830 ms)", last_outcome="success",
                             infer_ms={"last": 80.1, "avg": 80.1, "max": 830.0}, chunks=1)
    elif name == "homing":
        s.update(mode="policy", message="Policy: moving to the start pose")
        s["policy"] = policy("homing", episode=4, message="Moving to the start pose (30 deg/s)",
                             target=dict(zip(JOINTS, [0, -103, 95, 52, 0, 20])))
    elif name == "running":
        s.update(mode="policy", message="Policy")
        s["policy"] = policy("running", **run, target=target(3.0), message="Episode 4 running",
                             record={"dataset": "rollouts_act_v0", "saved": 3, "pending": False})
    elif name == "running_dry":
        s.update(mode="manual", message="Holding current pose (dry run)")
        s["policy"] = policy("running", **run, dry_run=True, target=target(6.0), message="Dry run: targets shown only")
    elif name == "intervention":
        s.update(mode="teleop", synced=True, message="Teleop")
        s["policy"] = policy("intervention", **run, message="Takeover: the leader arm is in control; policy paused",
                             record={"dataset": "rollouts_act_v0", "saved": 3, "pending": False})
    elif name == "pending":
        s["policy"] = policy("pending", **dict(run, elapsed_s=30.0, queue=0), message="Episode 4 ended: time limit",
                             record={"dataset": "rollouts_act_v0", "saved": 3, "pending": True}, last_outcome="success")
    elif name == "error":
        s["policy"] = policy("error", info=None, message="Load the policy again to continue",
                             fault="policy worker exited (code -9) during episode 5: arm held at its last pose")
    elif name == "fault":
        s["policy"] = policy("pending", **dict(run, elapsed_s=17.2, queue=0), message="Episode 4 ended by a fault",
                             fault="tracking guard: elbow_flex 31.0 deg behind its command for 1.0 s; holding pose")
    elif name == "rec_session":
        s.update(mode="teleop", synced=True, message="Teleop")
        s["policy"] = policy("ready", message="Ready")
        s["recording"] = {"phase": "recording", "message": "Recording episode 2 (teleop)", "dataset": "cube_cup_real_v2",
                          "task": "Put the cube in the cup", "episode_s": 30, "reset_s": 10, "num_episodes": 10,
                          "manual_start": True, "saved": 1, "episode": 2, "frames": 240, "elapsed_s": 8.0, "late": 0}
    elif name == "limp":
        s.update(mode="limp", torque=False, message="Torque released: arm is limp")
        s["policy"] = policy("ready", message="Ready")
        s["feeds"]["scene"] = {"status": "no frames for 3.2 s (replug?)"}
    else:
        raise KeyError(name)
    return s


POLICIES = {"root": "/home/asnano/so101_policies", "remotes": [{"name": "brev-a10g", "address": "10.8.0.12:7100"}],
            "policies": [
    {"name": "act_real_v0", "path": "/home/asnano/so101_policies/act_real_v0", "type": "act", "size_mb": 197.4,
     "steps": 20000, "dataset": "cube_cup_real_v1", "created": "2026-10-05T14:22:09", "chunk_size": 50,
     "n_action_steps": 50, "image_keys": ["observation.images.wrist", "observation.images.scene"],
     "deploy": INFO["deploy"], "undistort": False, "runnable": True, "why_not": None, "tally": TALLY},
    {"name": "act_cosim_v1", "path": "/home/asnano/so101_policies/act_cosim_v1", "type": "act", "size_mb": 197.6,
     "steps": 40000, "dataset": "cube_cup_cosim_v1", "created": 1759900000, "chunk_size": 100, "n_action_steps": 50,
     "image_keys": ["observation.images.wrist", "observation.images.scene"],
     "deploy": {"undistort": {"wrist": {}, "scene": {}}, "start_pose": [0, -100, 90, 50, 0, 20],
                "notes": "sim co-trained; needs undistorted frames"},
     "undistort": True, "runnable": True, "why_not": None, "tally": {"success": 0, "fail": 0, "total": 0}},
    {"name": "smolvla_cube", "path": "/home/asnano/so101_policies/smolvla_cube", "type": "smolvla", "size_mb": 1734.0,
     "steps": 30000, "dataset": "cube_cup_real_v1", "created": "2026-10-06T09:00:00", "chunk_size": 50,
     "n_action_steps": 50, "image_keys": ["observation.images.camera1", "observation.images.camera2"], "deploy": None,
     "undistort": False, "runnable": False,
     "why_not": "transformers is not installed on the Jetson: serve it from a GPU box (./so101.sh policy-server)",
     "tally": {"success": 0, "fail": 0, "total": 0}},
    {"name": "diffusion_test", "path": "/home/asnano/so101_policies/diffusion_test", "type": "diffusion", "size_mb": 1020.0,
     "steps": 5000, "dataset": "cube_cup_real_v1", "created": None, "chunk_size": None, "n_action_steps": 8,
     "image_keys": ["observation.images.wrist"], "deploy": None, "undistort": False, "runnable": False,
     "why_not": "diffusion is not a chunking policy", "tally": {"success": 0, "fail": 0, "total": 0}},
]}
DATASETS = {"root": "/home/asnano/so101_web_dev/test_datasets",
            "datasets": [{"name": "cube_cup_real_v1", "episodes": 50}, {"name": "rollouts_act_v0", "episodes": 3}]}


def frame_svg(cam, view):
    col = "#b98cff" if view == "policy" else "#4da3ff"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">
<rect width="640" height="480" fill="#1b2230"/><g stroke="#2c3850">{''.join(f'<line x1="{x}" y1="0" x2="{x}" y2="480"/>' for x in range(0, 640, 40))}
{''.join(f'<line x1="0" y1="{y}" x2="640" y2="{y}"/>' for y in range(0, 480, 40))}</g>
<rect x="250" y="300" width="60" height="60" fill="#e04848"/><ellipse cx="420" cy="330" rx="55" ry="35" fill="none" stroke="#ddd" stroke-width="6"/>
<text x="20" y="40" fill="{col}" font-family="sans-serif" font-size="26">{cam} · {view} (stub)</text></svg>""".encode()


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            if "phase" in q:
                CUR["name"] = q["phase"][0]
            html = (WEB / "index.html").read_text()
            if q.get("selftest"):
                html = html.replace("</body>", '<script src="/selftest.js"></script>\n</body>')
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/selftest.js":
            return self._send(200, (HERE / "selftest.js").read_bytes(), "text/javascript")
        if u.path == "/_phase":
            CUR["name"] = q.get("p", ["none"])[0]
            return self._send(200, {"ok": True, "phase": CUR["name"]})
        if u.path == "/_posts":
            return self._send(200, POSTS)
        if u.path == "/api/state":
            return self._send(200, scenario(CUR["name"]))
        if u.path == "/api/policies":
            return self._send(200, POLICIES)
        if u.path == "/api/datasets":
            return self._send(200, DATASETS)
        if u.path.startswith("/stream/") and u.path.endswith(".mjpg"):
            cam = u.path[len("/stream/"):-len(".mjpg")]
            return self._send(200, frame_svg(cam, q.get("view", ["raw"])[0]), "image/svg+xml")
        self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) or b"{}"
        try:
            body = json.loads(raw)
        except ValueError:
            body = {"_raw": raw.decode(errors="replace")}
        entry = {"t": round(time.time(), 3), "scenario": CUR["name"], "path": self.path, "body": body}
        POSTS.append(entry)
        with LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        print("POST", self.path, json.dumps(body), flush=True)
        self._send(200, {"ok": True, "message": "stub ok"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8199)
    args = ap.parse_args()
    pf = HERE / "phase.txt"
    if pf.exists():
        CUR["name"] = pf.read_text().strip() or "none"
    ThreadingHTTPServer(("127.0.0.1", args.port), H).serve_forever()


if __name__ == "__main__":
    main()

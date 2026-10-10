#!/usr/bin/env python
"""policy_worker.py + PolicyClient/LocalWorker against a real checkpoint on the GPU (no robot needed).

    python tests/test_worker.py [--ckpt ~/so101_policies/act_real_v0] [--port 8095]

Run from ~/lerobot with its venv python. Checks: READY/info, reset, chunk shape + latency (unix socket,
raw frames from a recorded episode), bad requests -> error replies (server survives), plain
multiprocessing.connection client compatibility, wrong authkey, newest-client-wins + transparent reconnect,
TCP + JPEG transport, unsupported policy type -> ERROR exit 2, worker exits with its parent, SIGTERM, quit.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from policy_client import LocalWorker, PolicyClient, PolicyError  # noqa: E402

RESULTS = []
METRICS = {}


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond), detail))
    print(f"{'PASS' if cond else 'FAIL'}  {name}{('  -- ' + str(detail)) if detail else ''}", flush=True)
    return cond


def section(fn):
    def run(*a):
        try:
            return fn(*a)
        except Exception as e:
            traceback.print_exc()
            check(f"{fn.__name__} (no exception)", False, f"{type(e).__name__}: {e}")
    run.__name__ = fn.__name__
    return run


def episode_frames(n=3):
    """A few real (state, {cam: frame}) observations from the recorded dataset the mock uses."""
    os.environ.setdefault("SO101_MOCK", "1")
    import av
    import mock_hw
    src = mock_hw.episode_source()
    state = np.array(src["state0"] or mock_hw.REST_POSE, dtype=np.float32)
    frames = {}
    for cam, (path, t0, _) in src["videos"].items():
        out = []
        with av.open(str(path)) as c:
            for f in c.decode(video=0):
                if f.time is not None and f.time < t0 - 1e-3:
                    continue
                out.append(f.to_ndarray(format="rgb24"))
                if len(out) >= n:
                    break
        frames[cam] = out
    if not frames:  # no dataset: synthetic
        frames = {c: [np.full((480, 640, 3), 100, np.uint8)] * n for c in ("wrist", "scene")}
    return state, frames


def obs_for(state, frames, i=0):
    return {"observation.state": state, **{f"observation.images.{c}": f[i] for c, f in frames.items()}}


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 1)


@section
def unix_worker(ckpt, state, frames):
    t0 = time.monotonic()
    w = LocalWorker(ckpt, device="auto").start()
    try:
        info = w.wait_ready(300)
        ready_s = time.monotonic() - t0
        METRICS["ready_s"] = round(ready_s, 1)
        METRICS["warmup_ms"] = info.get("warmup_ms")
        check("READY with info", info.get("type") == "act", f"ready in {ready_s:.1f} s, warm-up {info.get('warmup_ms')} ms")
        mode = os.stat(w.sock_path).st_mode & 0o777
        check("socket file is owner-only", mode & 0o077 == 0, oct(mode))
        cl = PolicyClient(w.address, w.authkey)
        i2 = cl.info()
        check("info fields", all(k in i2 for k in ("name", "type", "ckpt", "device", "input_features",
                                                     "output_features", "chunk_size", "n_action_steps", "fps",
                                                     "image_keys", "deploy", "warmup_ms")), sorted(i2))
        check("info values", i2["chunk_size"] == 50 and i2["n_action_steps"] == 30 and i2["fps"] == 30
              and sorted(i2["image_keys"]) == ["observation.images.scene", "observation.images.wrist"]
              and i2["state_dim"] == 6 and i2["action_dim"] == 6 and i2["name"] == Path(ckpt).name,
              {k: i2[k] for k in ("name", "chunk_size", "n_action_steps", "fps", "image_keys", "device")})
        check("reset", cl.reset().get("ok") is True)

        acts, ms = cl.chunk(1, obs_for(state, frames), "Put the cube in the cup")
        check("chunk shape (50, 6) float32", acts.shape == (50, 6) and acts.dtype == np.float32, acts.shape)
        check("actions finite", np.isfinite(acts).all())
        dev = np.abs(acts[0, :5] - state[:5]).max()
        check("first action near the current pose (sane units)", dev < 20, f"max |a0 - state| = {dev:.1f} deg")

        rtts, infers = [], []
        for i in range(30):
            t = time.perf_counter()
            _, ms = cl.chunk(100 + i, obs_for(state, frames, i % len(frames["wrist"])), "Put the cube in the cup")
            rtts.append((time.perf_counter() - t) * 1000)
            infers.append(ms)
        METRICS["unix_infer_ms_p50"], METRICS["unix_infer_ms_max"] = pct(infers, 50), round(max(infers), 1)
        METRICS["unix_rtt_ms_p50"], METRICS["unix_rtt_ms_p95"] = pct(rtts, 50), pct(rtts, 95)
        check("latency (unix, raw frames)", pct(rtts, 50) < 250,
              f"infer p50 {pct(infers, 50)} ms, round trip p50 {pct(rtts, 50)} / p95 {pct(rtts, 95)} / max {max(rtts):.0f} ms")

        # bad requests -> error replies, server keeps serving
        bad = {
            "missing image": {k: v for k, v in obs_for(state, frames).items() if "wrist" not in k},
            "state length": {**obs_for(state, frames), "observation.state": np.zeros(5, np.float32)},
            "state NaN": {**obs_for(state, frames), "observation.state": np.full(6, np.nan, np.float32)},
            "image dtype": {**obs_for(state, frames), "observation.images.wrist": np.zeros((480, 640, 3), np.float32)},
            "bad jpeg": {**obs_for(state, frames), "observation.images.wrist": ("jpeg", b"not a jpeg")},
            "obs not a dict": "hello",
        }
        for name, obs in bad.items():
            r = cl.call({"cmd": "chunk", "id": 7, "task": "x", "obs": obs}, 10)
            check(f"bad request ({name}) -> error reply", "error" in r and r.get("id") == 7, r.get("error"))
        r = cl.call({"cmd": "nope"}, 5)
        check("unknown cmd -> error", "error" in r, r.get("error"))
        r = cl.call(["not", "a", "dict"], 5)
        check("non-dict message -> error", "error" in r, r.get("error"))
        small = {**obs_for(state, frames), "observation.images.scene": frames["scene"][0][::2, ::2].copy()}
        a2, _ = cl.chunk(8, small, "Put the cube in the cup")
        check("smaller frames are resized, not rejected", a2.shape == (50, 6))
        check("server alive after bad requests", cl.ping().get("ok") is True)

        # a plain multiprocessing.connection client works (Listener-compatible protocol)
        raw = Client(w.sock_path, family="AF_UNIX", authkey=w.authkey.encode())
        raw.send({"cmd": "ping"})
        ok = raw.poll(5) and raw.recv().get("ok")
        check("plain multiprocessing.connection.Client", ok)
        raw.close()

        try:
            PolicyClient(w.address, "wrong-key", connect_timeout=5).ping()
            check("wrong authkey rejected", False)
        except PolicyError as e:
            check("wrong authkey rejected", "auth" in str(e).lower() or "handshake" in str(e).lower(), e)

        # newest client wins; the old one reconnects transparently on its next-but-one call
        c2 = PolicyClient(w.address, w.authkey)
        check("second client served", c2.ping().get("ok"))
        time.sleep(0.3)
        try:
            cl.ping()
            first = "ok"
        except PolicyError as e:
            first = f"error: {e}"
        again = cl.ping().get("ok")
        check("old client dropped, then reconnects", first.startswith("error") and again, first)
        c2.close()

        # async path as the runner uses it
        cl.cancel()
        cl.submit({"cmd": "reset"})
        cl.submit({"cmd": "chunk", "id": 55, "task": "t", "obs": obs_for(state, frames)})
        got, t_end = [], time.monotonic() + 10
        while len(got) < 2 and time.monotonic() < t_end:
            got += cl.results()
            time.sleep(0.01)
        check("async submit/results", len(got) == 2 and got[1][1].get("id") == 55 and "actions" in got[1][1],
              [g[1].keys() for g in got])

        # quit -> exits 0 and removes its socket
        cl.quit()
        try:
            code = w.proc.wait(10)
        except subprocess.TimeoutExpired:
            code = None
        check("quit -> exit 0, socket removed", code == 0 and not os.path.exists(w.sock_path), f"code {code}")
        cl.close()
    finally:
        w.stop()


@section
def tcp_worker(ckpt, port, state, frames):
    env = dict(os.environ, SO101_POLICY_KEY="tcp-test-key")
    p = subprocess.Popen([sys.executable, str(HERE / "policy_worker.py"), "--ckpt", ckpt,
                          "--address", f"127.0.0.1:{port}"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         env=env, start_new_session=True)
    try:
        line = ""
        t_end = time.monotonic() + 300
        while time.monotonic() < t_end and not line.startswith(("READY", "ERROR")):
            line = p.stdout.readline().decode()
            if not line and p.poll() is not None:
                break
        check("TCP worker READY", line.startswith("READY"), line[:80])
        cl = PolicyClient(f"127.0.0.1:{port}", "tcp-test-key")
        check("TCP client uses JPEG", cl.jpeg)
        acts, _ = cl.chunk(1, obs_for(state, frames), "Put the cube in the cup")
        rtts, infers = [], []
        for i in range(20):
            t = time.perf_counter()
            _, ms = cl.chunk(10 + i, obs_for(state, frames), "Put the cube in the cup")
            rtts.append((time.perf_counter() - t) * 1000)
            infers.append(ms)
        METRICS["tcp_jpeg_rtt_ms_p50"], METRICS["tcp_jpeg_rtt_ms_p95"] = pct(rtts, 50), pct(rtts, 95)
        check("TCP + JPEG chunk", acts.shape == (50, 6) and np.isfinite(acts).all(),
              f"round trip p50 {pct(rtts, 50)} / p95 {pct(rtts, 95)} ms (incl. JPEG encode), infer p50 {pct(infers, 50)} ms")
        cl.quit()
        try:
            code = p.wait(10)
        except subprocess.TimeoutExpired:
            code = None
        check("TCP worker quit", code == 0, code)
        cl.close()
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(5)


@section
def unsupported_and_broken(tmp):
    d = Path(tmp) / "diffusion_fake"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"type": "diffusion", "n_action_steps": 8}))
    w = LocalWorker(d).start()
    try:
        w.wait_ready(60)
        check("non-chunking policy -> ERROR", False)
    except PolicyError as e:
        code = w.proc.wait(10)
        check("non-chunking policy -> ERROR, exit 2", "not supported" in str(e) and code == 2, f"{e} (exit {code})")
    w.stop()
    w = LocalWorker(Path(tmp) / "does_not_exist").start()
    try:
        w.wait_ready(120)
        check("missing checkpoint -> ERROR", False)
    except PolicyError as e:
        code = w.proc.wait(10)
        check("missing checkpoint -> ERROR, exit 1", code == 1 and "config.json" in str(e), f"{str(e)[:90]} (exit {code})")
    w.stop()


@section
def no_leaks(ckpt):
    # the worker exits when the process that started it dies (even mid-load)
    code = ("import sys, time; sys.path.insert(0, %r); from policy_client import LocalWorker; "
            "w = LocalWorker(%r).start(); print(w.proc.pid, flush=True); time.sleep(2); import os; os._exit(0)"
            % (str(HERE), ckpt))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60).stdout.strip()
    pid = int(out.splitlines()[-1])
    t0 = time.monotonic()
    while time.monotonic() - t0 < 10:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.2)
    else:
        os.kill(pid, 9)
    gone = time.monotonic() - t0
    check("worker exits when its parent dies (during load)", gone < 5, f"gone {gone:.1f} s after the parent")

    w = LocalWorker(ckpt).start()
    time.sleep(3)
    t0 = time.monotonic()
    w.stop(timeout=5)
    check("stop() during load (SIGTERM)", w.proc.poll() is not None and time.monotonic() - t0 < 6,
          f"exit {w.proc.poll()} after {time.monotonic() - t0:.1f} s")
    check("no socket left behind", not os.path.exists(w.sock_path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/so101_policies/act_real_v0"))
    ap.add_argument("--port", type=int, default=8095)
    args = ap.parse_args()
    state, frames = episode_frames(3)
    with tempfile.TemporaryDirectory(prefix="so101_test_worker_") as tmp:
        unix_worker(args.ckpt, state, frames)
        tcp_worker(args.ckpt, args.port, state, frames)
        unsupported_and_broken(tmp)
        no_leaks(args.ckpt)
    leftovers = subprocess.run(["pgrep", "-f", f"[p]olicy_worker.py --ckpt {args.ckpt}"], capture_output=True,
                               text=True).stdout.split()
    check("no worker processes left", not leftovers, leftovers)
    print("\nmetrics:", json.dumps(METRICS))
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\ntest_worker: {len(RESULTS) - len(failed)} passed, {len(failed)} failed"
          + (f"  (FAILED: {', '.join(failed)})" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

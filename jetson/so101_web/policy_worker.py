#!/usr/bin/env python
"""Policy server for the SO-101 web console: loads one LeRobot checkpoint and serves action chunks.

    python policy_worker.py --ckpt DIR [--address unix:/path.sock | --address HOST:PORT] [--authkey KEY]
                            [--device cuda] [--fp32]

On CUDA the model runs under fp16 autocast by default; ACT additionally goes through jetson_accel.py (folded
BatchNorm, channels_last, the whole predict_action_chunk captured in one CUDA graph): SO101_ACCEL=graph (default) |
compile (torch.compile, needs Python.h, minutes to compile) | eager | off. Any acceleration failure falls back to plain
fp16 autocast; it never fails the load. --fp32 (or SO101_POLICY_FP32=1) = no acceleration, fp32.
(jetson_accel.py must sit next to this file; without it the worker uses fp16 autocast.)

The web app starts it as a subprocess (unix socket, random authkey via SO101_POLICY_KEY). On a GPU server run
it by hand: `./so101.sh policy-server <ckpt_dir> [port]`, or copy this one file next to any LeRobot install
(it imports only lerobot / torch / numpy / cv2, never the web app). The authkey defaults to $SO101_POLICY_KEY,
then "so101". Messages are pickled: only expose the port on a private network (e.g. Tailscale) and use a
non-default key.

DIR may be a pretrained_model directory (config.json, model.safetensors, policy_*processor*) or a training
run / policy directory containing pretrained_model/ or checkpoints/last/pretrained_model/.

Startup prints exactly one line on stdout: `READY {json info}`, or `ERROR {json}` and exits with code 2 if
the policy type is not supported (only action-chunking policies are) or 1 if loading failed. Then it serves
multiprocessing.connection requests (dicts), one client at a time; a newly connecting client replaces the
current one (e.g. the app was restarted):
  {"cmd": "info"}   -> info dict (name, type, ckpt, device, input_features, output_features, chunk_size,
                       n_action_steps, fps, image_keys, deploy, warmup_ms, ...)
  {"cmd": "reset"}  -> {"ok": True}                         {"cmd": "ping"} -> {"ok": True, "t": ...}
  {"cmd": "chunk", "id": int, "task": str, "obs": {"observation.state": float32[S],
                     "observation.images.<k>": ("jpeg", bytes) | uint8 HxWx3 RGB}}
                    -> {"id", "actions": float32[T, D], "infer_ms"}  or  {"id", "error": str}
  {"cmd": "quit"}   -> {"ok": True}, then exits
image_keys are the keys the client sends (before the policy's own rename step, if it has one).
"""
import argparse
import json
import logging
import os
import queue
import signal
import socket
import struct
import sys
import threading
import time
import traceback
from contextlib import nullcontext
from multiprocessing.connection import AuthenticationError, Listener, answer_challenge, deliver_challenge
from pathlib import Path

import numpy as np

log = logging.getLogger("policy_worker")

# Policies that don't predict action chunks (or don't work with predict_action_chunk the way we use it).
NON_CHUNKING = {"diffusion", "vqbet", "tdmpc", "sac", "reward_classifier", "gaussian_actor"}
ROBOT_TYPE = "so_follower"
STATE = "observation.state"


def resolve_ckpt(path):
    """The directory holding config.json: DIR, DIR/pretrained_model or DIR/checkpoints/last/pretrained_model."""
    d = Path(path).expanduser()
    for c in (d, d / "pretrained_model", d / "checkpoints" / "last" / "pretrained_model"):
        if (c / "config.json").is_file():
            return c
    raise FileNotFoundError(f"no config.json in {d} (or its pretrained_model/, checkpoints/last/pretrained_model/)")


def policy_name(given):
    """Display name: the policy / run directory, not 'pretrained_model'."""
    g = Path(given).expanduser().resolve()
    if g.name == "pretrained_model":
        g = g.parent.parent.parent if g.parent.parent.name == "checkpoints" else g.parent
    return g.name


def parse_address(a):
    a = str(a).strip()
    if a.startswith("unix:"):
        return a[5:], "AF_UNIX"
    host, sep, port = a.rpartition(":")
    if not sep or not port.isdigit():
        raise SystemExit(f"bad --address {a!r}: use unix:/path.sock or HOST:PORT")
    return (host.strip("[]") or "0.0.0.0", int(port)), "AF_INET"


def set_io_timeouts(fd, seconds):
    s = socket.socket(fileno=os.dup(fd))
    try:
        tv = struct.pack("ll", int(seconds), int((seconds % 1) * 1e6))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, tv)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, tv)
    finally:
        s.close()


def read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def dataset_fps(ckpt):
    """fps the policy was trained at: train_config dataset (or env) fps, else the dataset's info.json, else 30."""
    tc = read_json(ckpt / "train_config.json") or {}
    ds = tc.get("dataset") or {}
    for v in (ds.get("fps"), (tc.get("env") or {}).get("fps")):
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    if ds.get("root"):
        info = read_json(Path(ds["root"]).expanduser() / "meta" / "info.json") or {}
        if isinstance(info.get("fps"), (int, float)):
            return float(info["fps"])
    return 30.0


class Policy:
    """A loaded LeRobot policy + its pre/post-processors; predict() mirrors lerobot's predict_action."""

    def __init__(self, given, device, fp16=True):
        t0 = time.perf_counter()
        self.ckpt = resolve_ckpt(given)
        raw = read_json(self.ckpt / "config.json") or {}
        ptype = raw.get("type")
        if ptype in NON_CHUNKING or "chunk_size" not in raw:
            raise Unsupported(f"policy type '{ptype}' is not supported: only action-chunking policies "
                              f"(act, smolvla, pi0, pi05, groot, ...) can be deployed")

        import torch
        try:
            from lerobot.configs import PreTrainedConfig
        except ImportError:
            from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        try:
            from lerobot.policies import prepare_observation_for_inference
        except ImportError:
            prepare_observation_for_inference = None
        self.torch = torch
        self._prepare = prepare_observation_for_inference or self._prepare_fallback

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dev = torch.device(device)
        cfg = PreTrainedConfig.from_pretrained(str(self.ckpt))
        cfg.device = device
        self.cfg = cfg
        # fp16 autocast around the model call only (pre/post-processors stay fp32); weights stay fp32
        self.amp = bool(fp16 and self.dev.type == "cuda")
        self.policy = get_policy_class(cfg.type).from_pretrained(str(self.ckpt), config=cfg)
        self.policy.to(self.dev)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(self.ckpt),
            preprocessor_overrides={"device_processor": {"device": device}})
        self.accel = self._accelerate(fp16)  # the warm-up below captures the CUDA graph

        feats = cfg.input_features
        self.input_features = {k: list(v.shape) for k, v in feats.items()}
        self.output_features = {k: list(v.shape) for k, v in cfg.output_features.items()}
        rename = {}
        for step in getattr(self.pre, "steps", []):
            rm = getattr(step, "rename_map", None)
            if isinstance(rm, dict):
                rename.update(rm)
        self.rename_map = rename
        inverse = {v: k for k, v in rename.items()}
        visual = [k for k, v in feats.items() if getattr(v.type, "name", str(v.type)) == "VISUAL"]
        # client key (before the policy's rename step) -> (H, W) the policy was trained on
        self.images = {inverse.get(k, k): (int(feats[k].shape[-2]), int(feats[k].shape[-1])) for k in visual}
        self.state_dim = int(np.prod(feats[STATE].shape)) if STATE in feats else None
        other = [k for k in feats if k != STATE and k not in visual]
        self.other_inputs = other
        out = cfg.output_features.get("action")
        self.action_dim = int(np.prod(out.shape)) if out is not None else None
        self.chunk_size = int(getattr(cfg, "chunk_size", 0) or 0)
        self.n_action_steps = int(getattr(cfg, "n_action_steps", 0) or self.chunk_size)
        self.fps = dataset_fps(self.ckpt)
        self.load_s = time.perf_counter() - t0
        self._resized_warned = set()

        t1 = time.perf_counter()
        dummy = {}
        if self.state_dim:
            dummy[STATE] = np.zeros(self.state_dim, np.float32)
        for k, (h, w) in self.images.items():
            dummy[k] = np.zeros((h, w, 3), np.uint8)
        for k in other:  # zeros of whatever else the policy wants, so warm-up still exercises the model
            dummy[k] = np.zeros(feats[k].shape, np.float32)
        try:
            for _ in range(2):  # first call pays for CUDA kernels / cudnn autotuning (and the graph capture)
                self.predict(dummy, "warm-up")
        except Exception as e:
            if not self.accel.get("enabled"):
                raise
            log.warning("accelerated warm-up failed (%s: %s): falling back to fp16 autocast", type(e).__name__, e)
            self._disable_accel(f"warm-up failed: {type(e).__name__}: {e}")
            for _ in range(2):
                self.predict(dummy, "warm-up")
        self.policy.reset()
        self.warmup_ms = (time.perf_counter() - t1) * 1000

    def _accelerate(self, fp16):
        """jetson_accel.install() for ACT on CUDA unless SO101_ACCEL=off / --fp32. Never raises."""
        mode = (os.environ.get("SO101_ACCEL") or "graph").strip().lower()
        if not fp16 or self.dev.type != "cuda" or self.cfg.type != "act" or mode in ("off", "0", "none", "false"):
            return {"enabled": False, "mode": "off"}
        try:
            import jetson_accel
            if mode not in ("graph", "compile", "eager", "auto"):
                raise ValueError(f"unknown SO101_ACCEL={mode!r} (graph | compile | eager | auto | off)")
            if mode in ("compile", "auto") and not jetson_accel.python_headers_ok():
                log.warning("SO101_ACCEL=%s: Python.h is missing, so torch.compile can't build: using graph", mode)
                mode = "graph"
            info = jetson_accel.install(self.policy, mode=mode)
            info.setdefault("enabled", True)
            if info["enabled"]:
                log.info("acceleration: %s", info)
            return info
        except Exception as e:
            log.warning("acceleration disabled (%s: %s): using fp16 autocast", type(e).__name__, e)
            self.policy.__dict__.pop("predict_action_chunk", None)  # back to the class method
            return {"enabled": False, "mode": "off", "error": f"{type(e).__name__}: {e}"}

    def _disable_accel(self, why):
        self.policy.__dict__.pop("predict_action_chunk", None)
        self.accel = {"enabled": False, "mode": "off", "error": why}

    @property
    def precision(self):
        if self.accel.get("enabled"):
            return {"graph": "fp16+cudagraph", "compile": "fp16+compile", "eager": "fp16+channels_last"}.get(
                self.accel.get("mode"), "fp16+" + str(self.accel.get("mode")))
        return "fp16-autocast" if self.amp else "fp32"

    def _prepare_fallback(self, observation, device, task=None, robot_type=None):
        torch = self.torch
        for name in observation:
            t = torch.from_numpy(observation[name]).to(device)
            if "image" in name:
                if t.dtype == torch.uint8:
                    t = t.type(torch.float32) / 255
                t = t.permute(2, 0, 1).contiguous()
            observation[name] = t.unsqueeze(0)
        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""
        return observation

    def predict(self, obs, task):
        """obs: {key: np array (images HxWx3 uint8)} -> (float32[T, D] actions, infer_ms)."""
        torch = self.torch
        t0 = time.perf_counter()
        # jetson_accel applies its own fp16 autocast inside the (graphed) predict_action_chunk
        amp = (torch.autocast(device_type="cuda", dtype=torch.float16) if self.amp and not self.accel.get("enabled")
               else nullcontext())
        with torch.inference_mode():
            batch = self._prepare(dict(obs), self.dev, task, ROBOT_TYPE)
            batch = self.pre(batch)
            with amp:
                actions = self.policy.predict_action_chunk(batch)
            actions = actions.float()
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            actions = self.post(actions)
        out = actions.detach().float().cpu().numpy()
        if out.ndim == 3:
            out = out[0]
        return np.ascontiguousarray(out, dtype=np.float32), (time.perf_counter() - t0) * 1000

    def decode(self, obs):
        """Validate / decode a client observation into what predict() takes."""
        import cv2
        if not isinstance(obs, dict):
            raise ValueError("obs must be a dict")
        out = {}
        if self.state_dim:
            if STATE not in obs:
                raise ValueError(f"missing {STATE}")
            st = np.asarray(obs[STATE], dtype=np.float32).reshape(-1)
            if st.shape[0] != self.state_dim:
                raise ValueError(f"{STATE} has {st.shape[0]} values, the policy expects {self.state_dim}")
            if not np.all(np.isfinite(st)):
                raise ValueError(f"{STATE} contains NaN/inf")
            out[STATE] = st
        for k, (h, w) in self.images.items():
            v = obs.get(k)
            if v is None:
                raise ValueError(f"missing {k}")
            if isinstance(v, (tuple, list)) and len(v) == 2 and v[0] == "jpeg":
                img = cv2.imdecode(np.frombuffer(v[1], np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError(f"{k}: JPEG decoding failed")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.asarray(v)
                if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                    raise ValueError(f"{k}: expected uint8 HxWx3, got {img.dtype} {img.shape}")
            if img.shape[:2] != (h, w):
                if k not in self._resized_warned:
                    self._resized_warned.add(k)
                    log.warning("%s: resizing %dx%d frames to the trained %dx%d", k, img.shape[1], img.shape[0], w, h)
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            out[k] = np.ascontiguousarray(img)
        for k in self.other_inputs:
            if k not in obs:
                raise ValueError(f"missing {k}")
            out[k] = np.asarray(obs[k], dtype=np.float32)
        return out


class Unsupported(Exception):
    pass


class Server:
    """Accepts clients on its own thread (handshake with a timeout); the main thread serves the newest one."""

    def __init__(self, listener, authkey, handler):
        self.listener, self.authkey, self.handler = listener, authkey, handler
        self.new = queue.Queue()
        self.stop = threading.Event()
        threading.Thread(target=self._accept_loop, daemon=True, name="accept").start()

    def _accept_loop(self):
        while not self.stop.is_set():
            try:
                conn = self.listener.accept()  # no authkey on the Listener: we authenticate below, with a timeout
            except OSError:
                if self.stop.is_set():
                    return
                time.sleep(0.2)
                continue
            try:
                set_io_timeouts(conn.fileno(), 10.0)
                deliver_challenge(conn, self.authkey)
                answer_challenge(conn, self.authkey)
                set_io_timeouts(conn.fileno(), 30.0)  # backstop for a peer that stalls mid-message
            except (AuthenticationError, OSError, EOFError) as e:
                log.warning("rejected a client: %s: %s", type(e).__name__, e)
                conn.close()
                continue
            self.new.put(conn)

    def serve(self):
        conn = None
        while not self.stop.is_set():
            try:
                c = self.new.get(timeout=0.5) if conn is None else self.new.get_nowait()
                if conn is not None:
                    log.warning("a new client connected; dropping the previous one")
                    conn.close()
                conn = c
                log.info("client connected")
            except queue.Empty:
                if conn is None:
                    continue
            try:
                if not conn.poll(0.2):
                    continue
                msg = conn.recv()
            except (EOFError, OSError):
                log.info("client disconnected")
                conn.close()
                conn = None
                continue
            except Exception as e:  # unpicklable request: the stream may be out of sync, drop the client
                log.warning("bad request (%s: %s); dropping the client", type(e).__name__, e)
                conn.close()
                conn = None
                continue
            reply, quit_after = self.handler(msg)
            try:
                conn.send(reply)
            except Exception as e:
                log.info("client gone while replying (%s)", e)
                conn.close()
                conn = None
            if quit_after:
                return


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--address", default="127.0.0.1:7100", help="unix:/path.sock or HOST:PORT")
    ap.add_argument("--authkey", default=None, help="default: $SO101_POLICY_KEY, else 'so101'")
    ap.add_argument("--device", default="auto", help="cuda | cpu | auto")
    ap.add_argument("--fp32", action="store_true", default=os.environ.get("SO101_POLICY_FP32") == "1",
                    help="run the model in fp32 (default on CUDA: fp16 autocast; env SO101_POLICY_FP32=1)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")

    # stdout carries the READY/ERROR protocol line; anything libraries print goes to stderr instead.
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    address, family = parse_address(args.address)
    authkey = (args.authkey or os.environ.get("SO101_POLICY_KEY") or "so101").encode()
    sock_path = address if family == "AF_UNIX" else None

    def cleanup():
        if sock_path:
            try:
                os.unlink(sock_path)
            except OSError:
                pass

    def emit(kind, payload):
        proto.write(f"{kind} {json.dumps(payload)}\n")
        proto.flush()

    def bye(*_):
        cleanup()
        os._exit(0)

    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)

    parent = os.environ.get("SO101_PARENT_PID")
    if parent:  # started by the web app: never outlive it (prctl PDEATHSIG would fire when the spawning thread exits)
        def watch_parent():
            while True:
                if os.getppid() != int(parent):
                    log.warning("web app (pid %s) is gone; exiting", parent)
                    bye()
                time.sleep(1.0)
        threading.Thread(target=watch_parent, daemon=True, name="parent-watch").start()

    try:
        policy = Policy(args.ckpt, args.device, fp16=not args.fp32)
    except Unsupported as e:
        emit("ERROR", {"error": str(e), "code": "unsupported"})
        cleanup()
        os._exit(2)
    except BaseException as e:
        log.error("loading failed:\n%s", traceback.format_exc())
        emit("ERROR", {"error": f"loading failed: {type(e).__name__}: {e}"})
        cleanup()
        os._exit(1)

    deploy = None
    for d in (Path(args.ckpt).expanduser(), policy.ckpt):
        deploy = read_json(d / "so101_deploy.json")
        if deploy is not None:
            break
    try:
        import lerobot
        lr_version = getattr(lerobot, "__version__", None)
    except Exception:
        lr_version = None
    info = {
        "name": policy_name(args.ckpt), "type": policy.cfg.type, "ckpt": str(policy.ckpt),
        "device": policy.device, "precision": policy.precision, "accel": policy.accel,
        "host": socket.gethostname(),
        "input_features": policy.input_features, "output_features": policy.output_features,
        "chunk_size": policy.chunk_size, "n_action_steps": policy.n_action_steps, "fps": policy.fps,
        "image_keys": list(policy.images), "image_sizes": {k: list(v) for k, v in policy.images.items()},
        "state_dim": policy.state_dim, "action_dim": policy.action_dim, "other_inputs": policy.other_inputs,
        "rename_map": policy.rename_map, "deploy": deploy, "warmup_ms": round(policy.warmup_ms, 1),
        "load_s": round(policy.load_s, 2), "lerobot_version": lr_version,
    }

    if sock_path:
        os.umask(0o077)  # socket file only for this user
        if os.path.exists(sock_path):
            os.unlink(sock_path)
    try:
        listener = Listener(address, family=family, backlog=4)
    except OSError as e:
        emit("ERROR", {"error": f"cannot listen on {args.address}: {e}"})
        os._exit(1)

    stats = {"n": 0, "ms": 0.0}

    def handle(msg):
        if not isinstance(msg, dict):
            return {"error": "request must be a dict"}, False
        cmd = msg.get("cmd")
        if cmd == "chunk":
            rid = msg.get("id")
            try:
                obs = policy.decode(msg.get("obs"))
                actions, ms = policy.predict(obs, str(msg.get("task") or ""))
                if not np.all(np.isfinite(actions)):
                    return {"id": rid, "error": "policy produced NaN/inf actions"}, False
                stats["n"] += 1
                stats["ms"] += ms
                if stats["n"] in (1, 10) or stats["n"] % 500 == 0:
                    log.info("chunk %d: %s in %.0f ms (avg %.0f ms)", stats["n"], actions.shape, ms,
                             stats["ms"] / stats["n"])
                return {"id": rid, "actions": actions, "infer_ms": round(ms, 2)}, False
            except Exception as e:
                log.warning("chunk %s failed: %s: %s", rid, type(e).__name__, e)
                if "out of memory" in str(e).lower():
                    try:
                        policy.torch.cuda.empty_cache()
                    except Exception:
                        pass
                return {"id": rid, "error": f"{type(e).__name__}: {e}"}, False
        if cmd == "info":
            return info, False
        if cmd == "ping":
            return {"ok": True, "t": time.time()}, False
        if cmd == "reset":
            try:
                policy.policy.reset()
                return {"ok": True}, False
            except Exception as e:
                return {"error": f"reset failed: {type(e).__name__}: {e}"}, False
        if cmd == "quit":
            return {"ok": True}, True
        return {"error": f"unknown cmd {cmd!r}"}, False

    log.info("serving %s (%s on %s, %s) at %s; warm-up %.0f ms", info["name"], info["type"], policy.device,
             info["precision"], args.address, policy.warmup_ms)
    emit("READY", info)
    server = Server(listener, authkey, handle)
    try:
        server.serve()
    finally:
        server.stop.set()
        try:
            listener.close()
        except OSError:
            pass
        cleanup()
        log.info("quit")
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()

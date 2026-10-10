"""Policy backends for the SO-101 web console.

- PolicyClient: talks to a policy_worker.py server (local unix socket or remote TCP). Every wait is bounded:
  connect and handshake time out, replies are awaited with poll(), and SO_RCVTIMEO/SO_SNDTIMEO make even a
  half-received message or a stalled peer raise instead of hanging. After any transport failure or timeout
  the connection is dropped (the next call reconnects), so a late reply can never be mistaken for another.
- LocalWorker: runs policy_worker.py as a subprocess on this machine (random authkey, private unix socket),
  waits for its READY line, streams its stderr to the log and kills it on stop / app exit.
- ChunkScheduler: pure logic (no I/O, injectable clock) that turns asynchronously arriving action chunks
  into one action per control tick. Unit-tested in tests/test_scheduler.py.

No web-app imports: policy_runner.py builds on this module and the tests import it directly.
"""
import atexit
import collections
import itertools
import json
import logging
import math
import os
import queue
import secrets
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import weakref
from multiprocessing.connection import AuthenticationError, Connection, answer_challenge, deliver_challenge
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKER = HERE / "policy_worker.py"
log = logging.getLogger("so101_web.policy")


class PolicyError(RuntimeError):
    """A request to a policy backend failed: transport, timeout, or the backend could not start."""


def parse_address(address):
    """'unix:/path.sock' -> ('AF_UNIX', '/path.sock');  'host:port' -> ('AF_INET', (host, port))."""
    a = str(address).strip()
    if a.startswith("unix:"):
        if len(a) <= 5:
            raise ValueError(f"bad policy address {address!r}")
        return "AF_UNIX", a[5:]
    host, sep, port = a.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(f"bad policy address {address!r}: use unix:/path.sock or HOST:PORT")
    return "AF_INET", (host.strip("[]") or "127.0.0.1", int(port))


def _set_io_timeouts(sock, seconds):
    """Kernel-level send/recv timeouts: a blocking read or write on a stalled peer fails with EAGAIN."""
    tv = struct.pack("ll", int(seconds), int((seconds % 1) * 1e6))
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVTIMEO, tv)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDTIMEO, tv)


# ------------------------------------------------------------------------------------------------ client
class PolicyClient:
    """Connection to one policy server. Thread-safe: requests are serialized by a lock.

    call() is synchronous with a timeout; submit()/results() run requests on a background I/O thread so the
    30 Hz policy loop never waits on the network or the GPU (images are JPEG-encoded there for TCP servers).
    """

    def __init__(self, address, authkey, connect_timeout=10.0, io_timeout=30.0, chunk_timeout=10.0,
                 jpeg_quality=90):
        self.address = address
        self.family, self._addr = parse_address(address)
        self.authkey = authkey.encode() if isinstance(authkey, str) else bytes(authkey)
        self.jpeg = self.family != "AF_UNIX"  # raw arrays over a local socket, JPEG over the network
        self.connect_timeout, self.io_timeout, self.chunk_timeout = connect_timeout, io_timeout, chunk_timeout
        self.jpeg_quality = jpeg_quality
        self._conn = None
        self._wake = None  # dup of the connection's socket: shutdown() on it unblocks any thread using it
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._q = queue.Queue()
        self._results = queue.Queue()
        self._gen = 0  # bumped by cancel(): replies to older submissions are dropped
        self._gen_lock = threading.Lock()  # cancel() vs the I/O thread's "still current? then deliver"
        self._io = None

    # ---- connection
    def _connect(self):
        fam = socket.AF_UNIX if self.family == "AF_UNIX" else socket.AF_INET
        s = socket.socket(fam, socket.SOCK_STREAM)
        try:
            s.settimeout(self.connect_timeout)
            s.connect(self._addr)
            s.settimeout(None)  # blocking fd for Connection; the kernel timeouts below bound every wait
            if fam == socket.AF_INET:
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            _set_io_timeouts(s, self.connect_timeout)
            conn = Connection(s.detach())
        except OSError as e:
            s.close()
            raise PolicyError(f"cannot connect to {self.address}: {e}") from e
        try:
            answer_challenge(conn, self.authkey)
            deliver_challenge(conn, self.authkey)
            wake = socket.socket(fileno=os.dup(conn.fileno()))
            _set_io_timeouts(wake, self.io_timeout)  # socket options are shared with conn's fd
        except AuthenticationError as e:
            conn.close()
            raise PolicyError(f"{self.address}: authentication failed (wrong authkey?)") from e
        except (OSError, EOFError) as e:
            conn.close()
            raise PolicyError(f"{self.address}: handshake failed ({type(e).__name__}: {e})") from e
        return conn, wake

    def _drop_locked(self):
        conn, wake, self._conn, self._wake = self._conn, self._wake, None, None
        for c in (conn, wake):
            if c is not None:
                try:
                    c.close()
                except OSError:
                    pass

    def connected(self):
        return self._conn is not None

    def call(self, msg, timeout=10.0):
        """Send one request and wait (bounded) for its reply dict. Raises PolicyError on transport problems;
        an {"error": ...} reply from the server is returned as is."""
        if self._closed.is_set():
            raise PolicyError("policy client is closed")
        if not self._lock.acquire(timeout=timeout + self.io_timeout):
            raise PolicyError("policy connection is busy")
        try:
            if self._conn is None:
                self._conn, self._wake = self._connect()
            conn = self._conn
            try:
                conn.send(msg)
                deadline = time.monotonic() + timeout
                while not conn.poll(max(0.0, min(0.25, deadline - time.monotonic()))):
                    if self._closed.is_set():
                        raise PolicyError("policy client is closed")
                    if time.monotonic() >= deadline:
                        raise PolicyError(f"no reply from {self.address} within {timeout:.1f} s")
                reply = conn.recv()
            except PolicyError:
                self._drop_locked()
                raise
            except Exception as e:  # EOF, reset, kernel timeout, unpicklable garbage
                self._drop_locked()
                raise PolicyError(f"connection to {self.address} lost ({type(e).__name__}: {e})") from e
        finally:
            self._lock.release()
        if not isinstance(reply, dict):
            raise PolicyError(f"bad reply from {self.address}: {type(reply).__name__}")
        return reply

    def info(self, timeout=15.0):
        r = self.call({"cmd": "info"}, timeout)
        if "error" in r:
            raise PolicyError(r["error"])
        return r

    def ping(self, timeout=5.0):
        r = self.call({"cmd": "ping"}, timeout)
        if "error" in r:
            raise PolicyError(r["error"])
        return r

    def reset(self, timeout=10.0):
        r = self.call({"cmd": "reset"}, timeout)
        if "error" in r:
            raise PolicyError(r["error"])
        return r

    def chunk(self, req_id, obs, task, timeout=None):
        """Synchronous chunk request -> (actions float32[T, D], infer_ms). Raises PolicyError."""
        r = self.call(self._encode({"cmd": "chunk", "id": req_id, "task": task, "obs": obs}),
                      timeout or self.chunk_timeout)
        if "error" in r:
            raise PolicyError(r["error"])
        if r.get("id") != req_id:
            raise PolicyError(f"reply for request {r.get('id')} while waiting for {req_id}")
        return np.asarray(r["actions"], dtype=np.float32), float(r.get("infer_ms", 0.0))

    def quit(self, timeout=3.0):
        try:
            self.call({"cmd": "quit"}, timeout)
        except PolicyError:
            pass

    def _encode(self, msg):
        if not self.jpeg or msg.get("cmd") != "chunk":
            return msg
        import cv2
        obs = {}
        for k, v in msg["obs"].items():
            if isinstance(v, np.ndarray) and v.ndim == 3 and v.dtype == np.uint8:
                ok, buf = cv2.imencode(".jpg", cv2.cvtColor(v, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
                if not ok:
                    raise PolicyError(f"JPEG encoding of {k} failed")
                v = ("jpeg", buf.tobytes())
            obs[k] = v
        return {**msg, "obs": obs}

    # ---- asynchronous requests (used by the policy loop)
    def submit(self, msg):
        """Queue a request for the I/O thread. Its reply comes back from results()."""
        if self._io is None:
            self._io = threading.Thread(target=self._io_loop, daemon=True, name="policy-io")
            self._io.start()
        with self._gen_lock:
            self._q.put((self._gen, msg, time.monotonic()))

    def cancel(self):
        """Forget queued requests; replies to anything submitted so far are dropped."""
        with self._gen_lock:
            self._gen += 1
            for qq in (self._q, self._results):
                try:
                    while True:
                        qq.get_nowait()
                except queue.Empty:
                    pass

    def results(self):
        """[(msg, reply, rtt_s)] of finished requests. Transport failures come back as
        {"id", "error", "transport": True}."""
        out = []
        try:
            while True:
                out.append(self._results.get_nowait())
        except queue.Empty:
            return out

    def _io_loop(self):
        while not self._closed.is_set():
            try:
                gen, msg, t0 = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if gen != self._gen:
                continue
            try:
                timeout = self.chunk_timeout if msg.get("cmd") == "chunk" else 10.0
                reply = self.call(self._encode(msg), timeout)
            except PolicyError as e:
                reply = {"id": msg.get("id"), "error": str(e), "transport": True}
            except Exception as e:  # never let the I/O thread die
                log.exception("policy request failed")
                reply = {"id": msg.get("id"), "error": f"{type(e).__name__}: {e}", "transport": True}
            with self._gen_lock:
                if gen == self._gen:
                    self._results.put((msg, reply, time.monotonic() - t0))

    def close(self):
        """Close for good. Never blocks for long: shutdown() wakes a thread stuck on this connection."""
        self._closed.set()
        wake = self._wake
        if wake is not None:
            try:
                wake.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        io = self._io
        if io is not None and io is not threading.current_thread():
            io.join(2.0)
        if self._lock.acquire(timeout=2.0):
            try:
                self._drop_locked()
            finally:
                self._lock.release()


# --------------------------------------------------------------------------------------------- local worker
class LocalWorker:
    """policy_worker.py as a subprocess of this app, on a private unix socket with a random authkey."""

    _n = itertools.count(1)
    _live = weakref.WeakSet()

    def __init__(self, ckpt, device="cuda", python=None):
        self.ckpt = str(ckpt)
        self.device = device
        self.python = python or sys.executable
        self.sock_path = f"/tmp/so101_policy_{os.getpid()}_{next(LocalWorker._n)}.sock"
        self.address = "unix:" + self.sock_path
        self.authkey = secrets.token_hex(16)
        self.proc = None
        self.info = None
        self.stderr_tail = collections.deque(maxlen=30)
        self._lines = queue.Queue()

    def start(self):
        # The authkey goes through the environment, not argv, so it doesn't show up in `ps`.
        env = dict(os.environ, SO101_POLICY_KEY=self.authkey, SO101_PARENT_PID=str(os.getpid()),
                   PYTHONUNBUFFERED="1")
        cmd = [self.python, "-u", str(WORKER), "--ckpt", self.ckpt, "--address", self.address,
               "--device", self.device]
        # Own session: a Ctrl-C in the app's terminal doesn't reach the worker; the app stops it itself.
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, cwd=str(HERE), start_new_session=True)
        LocalWorker._live.add(self)
        threading.Thread(target=self._read_stdout, daemon=True, name="policy-worker-out").start()
        threading.Thread(target=self._read_stderr, daemon=True, name="policy-worker-err").start()
        log.info("policy worker pid %d: %s", self.proc.pid, " ".join(cmd))
        return self

    def _read_stdout(self):
        for raw in self.proc.stdout:
            self._lines.put(raw.decode(errors="replace").rstrip("\n"))
        self._lines.put(None)  # EOF

    def _read_stderr(self):
        n = 0
        for raw in self.proc.stderr:
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            self.stderr_tail.append(line)
            n += 1
            if n <= 5000:
                log.info("policy worker: %s", line)

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    @property
    def returncode(self):
        return None if self.proc is None else self.proc.poll()

    def _tail(self, n=4):
        lines = [l for l in self.stderr_tail if "Traceback" not in l][-n:]
        return " | ".join(lines)

    def wait_ready(self, timeout=300.0, cancel=None):
        """Block until READY (returns the info dict) or ERROR / exit / timeout / cancel (raises PolicyError)."""
        deadline = time.monotonic() + timeout
        while True:
            if cancel is not None and cancel():
                raise PolicyError("loading cancelled")
            try:
                line = self._lines.get(timeout=0.25)
            except queue.Empty:
                if time.monotonic() > deadline:
                    raise PolicyError(f"policy worker not ready after {timeout:.0f} s")
                continue
            if line is None:  # stdout closed: the worker is exiting
                try:
                    code = self.proc.wait(5)
                except subprocess.TimeoutExpired:
                    code = "?"
                raise PolicyError(f"policy worker exited (code {code}) before it was ready: {self._tail()}")
            if line.startswith("READY "):
                self.info = json.loads(line[6:])
                return self.info
            if line.startswith("ERROR "):
                try:
                    err = json.loads(line[6:]).get("error", line)
                except ValueError:
                    err = line[6:]
                raise PolicyError(str(err))
            log.info("policy worker stdout: %s", line)

    def stop(self, timeout=5.0):
        p = self.proc
        if p is not None and p.poll() is None:
            for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 5.0)):
                try:
                    os.killpg(p.pid, sig)  # the worker leads its own process group
                except (ProcessLookupError, PermissionError):
                    pass
                try:
                    p.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    log.warning("policy worker %d ignored signal %d", p.pid, sig)
        if p is not None:
            log.info("policy worker %d stopped (code %s)", p.pid, p.poll())
        try:
            os.unlink(self.sock_path)
        except OSError:
            pass
        LocalWorker._live.discard(self)


@atexit.register
def _stop_all_workers():
    for w in list(LocalWorker._live):
        try:
            w.stop(timeout=2.0)
        except Exception:
            pass


# ---------------------------------------------------------------------------------------------- scheduler
class ChunkScheduler:
    """One action per control tick from asynchronously arriving action chunks.

    Steps count control ticks of one (sub-)episode. A chunk requested with the observation captured at step
    k is aligned so that chunk[0] is the action for step k. Each tick() pops the action for the current
    step; when the queue gets short (remaining <= lead, lead = ceil(ema_rtt * fps) + lead_margin) and no
    request is in flight, want_request() says so: the caller captures the observation now and calls
    sent(id). When that chunk arrives at step k + d (d = ticks executed since), the new queue is
    chunk[d : d + n_exec], linearly cross-faded with the old queue over min(remaining, xfade_steps) steps.
    A chunk that arrives too late (d >= len) is dropped and a new one requested. Before the first chunk
    tick() returns None (hold); with an empty queue it repeats the last action and counts `starved`;
    starving for starve_timeout_s (or no first chunk within first_timeout_s) sets `fault`.
    """

    def __init__(self, n_exec, fps=30.0, xfade_steps=5, starve_timeout_s=2.0, first_timeout_s=20.0,
                 lead_margin=2, rtt_guess_s=0.15, ema_alpha=0.3, clock=time.monotonic):
        if int(n_exec) < 1:
            raise ValueError("n_exec must be >= 1")
        self.n_exec = int(n_exec)
        self.fps = float(fps)
        self.xfade_steps = max(0, int(xfade_steps))
        self.starve_timeout_s = float(starve_timeout_s)
        self.first_timeout_s = float(first_timeout_s)
        self.lead_margin = int(lead_margin)
        self.ema_rtt = float(rtt_guess_s)
        self.ema_alpha = float(ema_alpha)
        self.clock = clock
        # cumulative over the episode
        self.chunks = self.starved = self.dropped = self.stale = self.errors = 0
        self.begin()

    def begin(self):
        """(Re)start: hold until a fresh first chunk arrives. Replies to earlier requests become stale."""
        self.queue = []
        self.step = 0
        self.inflight = None  # (req_id, step k of the observation, clock at send)
        self.waiting_first = True
        self.t_begin = self.clock()
        self.last = None
        self.starved_since = None
        self.fault = None
        self.recovered = False  # a chunk arrived after the queue ran dry (see take_recovered)

    def take_recovered(self):
        """True once after a chunk ended a starvation: the caller should ramp into the new actions again."""
        r, self.recovered = self.recovered, False
        return r

    @property
    def lead(self):
        return math.ceil(self.ema_rtt * self.fps) + self.lead_margin

    @property
    def remaining(self):
        return len(self.queue)

    def want_request(self):
        if self.fault is not None or self.inflight is not None:
            return False
        return self.waiting_first or len(self.queue) <= self.lead

    def sent(self, req_id):
        """The caller just captured the observation (during the step that tick() last returned) and sent it."""
        self.inflight = (req_id, max(self.step - 1, 0), self.clock())

    def failed(self, req_id):
        """The request got an error reply; allow a new request right away."""
        if self.inflight is not None and self.inflight[0] == req_id:
            self.inflight = None
            self.errors += 1

    def receive(self, req_id, actions):
        """Feed a chunk reply. Returns 'ok', 'stale' (not the request in flight) or 'late' (dropped)."""
        if self.inflight is None or self.inflight[0] != req_id:
            self.stale += 1
            return "stale"
        _, k, t_sent = self.inflight
        self.inflight = None
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or len(actions) == 0:
            raise ValueError(f"chunk must be [T, D], got shape {actions.shape}")
        rtt = self.clock() - t_sent
        if self.waiting_first:
            # The arm held still while this chunk was computed, so its observation is still current.
            self.queue = list(actions[: self.n_exec])
            self.waiting_first = False
            self.step = 0
        else:
            self.ema_rtt += self.ema_alpha * (rtt - self.ema_rtt)
            d = self.step - k
            if d >= len(actions):
                self.dropped += 1
                return "late"
            new = list(actions[d: d + self.n_exec])
            # starved (empty queue): blend from the action the arm is holding, and let the controller re-ramp
            old = self.queue or ([self.last] * self.xfade_steps if self.last is not None else [])
            if not self.queue and self.last is not None:
                self.recovered = True
            m = min(len(old), self.xfade_steps, len(new))
            for i in range(m):
                w = (i + 1) / (m + 1)
                new[i] = (1.0 - w) * old[i] + w * new[i]
            self.queue = new
        self.chunks += 1
        return "ok"

    def tick(self):
        """Advance one control step: the action for this step, the last one while starved, or None before
        the first chunk (= hold the current command)."""
        now = self.clock()
        if self.fault is not None:
            return self.last
        if self.waiting_first:
            if now - self.t_begin > self.first_timeout_s:
                self.fault = f"no action chunk from the policy within {self.first_timeout_s:.0f} s"
            return None
        if self.queue:
            self.last = self.queue.pop(0)
            self.starved_since = None
        else:
            self.starved += 1
            if self.starved_since is None:
                self.starved_since = now
            elif now - self.starved_since >= self.starve_timeout_s:
                self.fault = (f"policy starved: no new actions for {now - self.starved_since:.1f} s "
                              f"(inference too slow or stuck)")
        self.step += 1
        return self.last


class EnsembleScheduler(ChunkScheduler):
    """ACT temporal ensembling (Zhao et al. 2023) instead of executing chunks one after another.

    A new chunk is requested every `stride` control ticks (or sooner when the predictions run short; one request in
    flight at a time, so a slow backend simply asks less often). Every action of every chunk is kept, aligned to the
    absolute step it is meant for (chunk[0] = the step of its observation, as in ChunkScheduler), and the action for a
    step is the weighted mean of all predictions made for it, w = exp(-coeff * n) for the n-th prediction of that step
    (n = 0 the oldest; ACT's temporal_ensemble_coeff, 0.01). Neighbouring steps then share most of their predictions,
    so the command moves smoothly instead of switching plans at chunk boundaries. n_exec is not used.
    Same interface and fault behaviour as ChunkScheduler: hold (None) until the first chunk, repeat the last action
    while starved (fault after starve_timeout_s), stale / late replies are counted and ignored.
    """

    def __init__(self, stride=2, coeff=0.01, **kw):
        kw.pop("n_exec", None)
        kw.pop("xfade_steps", None)
        self.stride = max(1, int(stride))
        self.coeff = float(coeff)
        super().__init__(n_exec=1, xfade_steps=0, **kw)

    def begin(self):
        super().begin()
        self.acc = {}  # absolute step -> [weighted sum of predictions (float64), sum of weights, number of predictions]
        self.last_req_step = None

    @property
    def remaining(self):
        """Steps from now on that have at least one prediction (coverage is contiguous: chunks are contiguous and a
        reply only ever extends it from the current step)."""
        return max(self.acc) - self.step + 1 if self.acc else 0

    def want_request(self):
        if self.fault is not None or self.inflight is not None:
            return False
        if self.waiting_first or self.remaining <= self.lead:
            return True
        return self.last_req_step is None or self.step - self.last_req_step >= self.stride

    def sent(self, req_id):
        super().sent(req_id)
        self.last_req_step = self.step

    def receive(self, req_id, actions):
        if self.inflight is None or self.inflight[0] != req_id:
            self.stale += 1
            return "stale"
        _, k, t_sent = self.inflight
        self.inflight = None
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or len(actions) == 0:
            raise ValueError(f"chunk must be [T, D], got shape {actions.shape}")
        rtt = self.clock() - t_sent
        if self.waiting_first:
            k = self.step  # the arm held still while this chunk was computed: chunk[0] is for the current step
            self.waiting_first = False
        else:
            self.ema_rtt += self.ema_alpha * (rtt - self.ema_rtt)
            if k + len(actions) <= self.step:
                self.dropped += 1
                return "late"
            if self.step not in self.acc and self.last is not None:
                self.recovered = True  # it had run dry: the controller ramps into the new actions again
        for i in range(max(0, self.step - k), len(actions)):
            e = self.acc.get(k + i)
            if e is None:
                self.acc[k + i] = [actions[i].astype(np.float64), 1.0, 1]
            else:
                w = math.exp(-self.coeff * e[2])
                e[0] = e[0] + w * actions[i]
                e[1] += w
                e[2] += 1
        self.chunks += 1
        return "ok"

    def tick(self):
        now = self.clock()
        if self.fault is not None:
            return self.last
        if self.waiting_first:
            if now - self.t_begin > self.first_timeout_s:
                self.fault = f"no action chunk from the policy within {self.first_timeout_s:.0f} s"
            return None
        e = self.acc.pop(self.step, None)
        if e is not None:
            self.last = (e[0] / e[1]).astype(np.float32)
            self.starved_since = None
        else:
            self.starved += 1
            if self.starved_since is None:
                self.starved_since = now
            elif now - self.starved_since >= self.starve_timeout_s:
                self.fault = (f"policy starved: no new actions for {now - self.starved_since:.1f} s "
                              f"(inference too slow or stuck)")
        self.step += 1
        return self.last

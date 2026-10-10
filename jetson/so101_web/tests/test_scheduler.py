#!/usr/bin/env python
"""ChunkScheduler unit tests with a fake clock (no GPU, no hardware).

    python tests/test_scheduler.py

Chunks are built so that action value == the absolute step it is meant for; then any misalignment (wrong
latency skip, off-by-one) shows up as a jump in the executed sequence.
"""
import math
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from policy_client import ChunkScheduler, EnsembleScheduler  # noqa: E402

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


DT = 1.0 / 30


def ramp_chunk(k, n=50, d=6):
    """Chunk for an observation taken at step k: chunk[i] is meant for step k + i, so its value is k + i."""
    return np.repeat(np.arange(k, k + n, dtype=np.float32)[:, None], d, axis=1)


def const_chunk(v, n=50, d=6):
    return np.full((n, d), v, dtype=np.float32)


def first(s, clock, chunk, req=1):
    assert s.want_request(), "should want the first chunk"
    s.sent(req)
    clock.advance(0.1)
    assert s.receive(req, chunk) == "ok"


def close(a, b, tol=1e-4):
    return a is not None and abs(float(np.asarray(a).reshape(-1)[0]) - b) <= tol


@test
def first_chunk_holds_then_starts_at_zero():
    c = Clock()
    s = ChunkScheduler(n_exec=30, clock=c)
    assert s.tick() is None, "must hold (None) before the first chunk"
    assert s.want_request()
    s.sent(1)
    assert not s.want_request(), "only one request in flight"
    for _ in range(5):  # arm holds while the first chunk is computed
        c.advance(DT)
        assert s.tick() is None
    assert s.receive(1, ramp_chunk(0)) == "ok"
    assert s.remaining == 30
    vals = [float(s.tick()[0]) for _ in range(3)]
    assert vals == [0.0, 1.0, 2.0], vals


@test
def prefetch_when_remaining_reaches_lead():
    c = Clock()
    s = ChunkScheduler(n_exec=30, rtt_guess_s=0.1, clock=c)  # lead = ceil(0.1 * 30) + 2 = 5
    assert s.lead == 5, s.lead
    first(s, c, ramp_chunk(0))
    asked_at = None
    for i in range(30):
        s.tick()
        c.advance(DT)
        if s.want_request():
            asked_at = (i, s.remaining)
            break
    assert asked_at == (24, 5), f"expected the request after the 25th pop with 5 left, got {asked_at}"


@test
def latency_skip_aligns_with_time():
    c = Clock()
    s = ChunkScheduler(n_exec=30, rtt_guess_s=0.1, xfade_steps=0, clock=c)
    first(s, c, ramp_chunk(0))
    executed = []
    while not s.want_request():
        executed.append(float(s.tick()[0]))
        c.advance(DT)
    k = executed[-1]  # the observation is captured during the step just executed
    s.sent(2)
    for _ in range(3):  # three more ticks pass while the policy computes
        executed.append(float(s.tick()[0]))
        c.advance(DT)
    assert s.receive(2, ramp_chunk(int(k))) == "ok"
    nxt = float(s.tick()[0])
    assert nxt == executed[-1] + 1, f"after the skip the next action must be step {executed[-1] + 1}, got {nxt}"
    assert s.remaining == 29


@test
def crossfade_is_linear_over_xfade_steps():
    c = Clock()
    s = ChunkScheduler(n_exec=30, rtt_guess_s=0.1, xfade_steps=5, clock=c)
    first(s, c, const_chunk(0.0))
    while not s.want_request():
        s.tick()
        c.advance(DT)
    s.sent(2)
    s.tick()
    assert s.remaining == 4
    assert s.receive(2, const_chunk(10.0)) == "ok"
    got = [float(s.tick()[0]) for _ in range(6)]
    want = [10 * 1 / 5, 10 * 2 / 5, 10 * 3 / 5, 10 * 4 / 5, 10.0, 10.0]  # m = min(remaining 4, 5) = 4
    assert all(abs(a - b) < 1e-4 for a, b in zip(got, want)), (got, want)


@test
def crossfade_full_window():
    c = Clock()
    s = ChunkScheduler(n_exec=30, rtt_guess_s=0.3, xfade_steps=5, clock=c)  # lead = 9 + 2 = 11
    first(s, c, const_chunk(0.0))
    while not s.want_request():
        s.tick()
        c.advance(DT)
    s.sent(2)
    assert s.remaining == 11
    assert s.receive(2, const_chunk(6.0)) == "ok"
    got = [float(s.tick()[0]) for _ in range(7)]
    want = [1, 2, 3, 4, 5, 6, 6]  # w = i / 6 for i = 1..5
    assert all(abs(a - b) < 1e-4 for a, b in zip(got, want)), got


@test
def starvation_holds_last_then_faults():
    c = Clock()
    s = ChunkScheduler(n_exec=3, rtt_guess_s=0.0, lead_margin=0, starve_timeout_s=2.0, clock=c)
    first(s, c, ramp_chunk(0))
    for _ in range(3):
        s.tick()
        c.advance(DT)
    assert s.remaining == 0
    s.sent(2)  # never answered
    last = None
    for i in range(59):  # < 2 s
        last = s.tick()
        c.advance(DT)
        assert s.fault is None, f"fault too early at tick {i}"
    assert float(last[0]) == 2.0, "must hold the last action while starved"
    assert s.starved == 59
    c.advance(0.1)
    s.tick()
    assert s.fault and "starved" in s.fault, s.fault


@test
def starvation_recovers_when_chunk_arrives():
    c = Clock()
    s = ChunkScheduler(n_exec=3, rtt_guess_s=0.0, lead_margin=0, starve_timeout_s=2.0, xfade_steps=0, clock=c)
    first(s, c, ramp_chunk(0))
    for _ in range(3):
        s.tick()
        c.advance(DT)
    s.sent(2)  # observation of step 2
    for _ in range(30):  # 1 s starving
        s.tick()
        c.advance(DT)
    assert s.receive(2, ramp_chunk(2)) == "ok"
    v = float(s.tick()[0])
    assert v == 33.0, f"time-aligned resume expected step 33, got {v}"
    assert s.starved_since is None and s.fault is None
    for _ in range(80):  # well past the timeout, but fed
        if s.want_request():
            k = s.step - 1
            s.sent(99)
            s.receive(99, ramp_chunk(k))
        s.tick()
        c.advance(DT)
    assert s.fault is None


@test
def crossfade_from_held_action_after_starvation():
    c = Clock()
    s = ChunkScheduler(n_exec=3, rtt_guess_s=0.0, lead_margin=0, starve_timeout_s=5.0, xfade_steps=5, clock=c)
    first(s, c, const_chunk(0.0))
    for _ in range(3):
        s.tick(); c.advance(1 / 30)
    s.sent(2)
    for _ in range(10):  # starving: holds 0.0
        assert float(s.tick()[0]) == 0.0
        c.advance(1 / 30)
    assert not s.take_recovered()
    assert s.receive(2, const_chunk(10.0, n=50)) == "ok"
    a = [float(s.tick()[0]) for _ in range(3)]
    # blend from the held 0.0 toward 10.0 (w = 1/4, 2/4, 3/4 over the 3 actions), no jump to 10.0
    assert close(a[0], 2.5) and close(a[1], 5.0) and close(a[2], 7.5), a
    assert s.take_recovered() and not s.take_recovered(), "recovery is reported once (controller re-ramps)"


@test
def stale_and_late_chunks():
    c = Clock()
    s = ChunkScheduler(n_exec=10, rtt_guess_s=0.0, lead_margin=0, xfade_steps=0, clock=c)
    first(s, c, ramp_chunk(0))
    assert s.receive(7, ramp_chunk(0)) == "stale", "unknown id must be ignored"
    for _ in range(10):
        s.tick()
        c.advance(DT)
    s.sent(2)
    for _ in range(60):  # 60 steps later the 50-step chunk is entirely in the past
        s.tick()
        c.advance(DT)
    q_before = s.remaining
    assert s.receive(2, ramp_chunk(9)) == "late"
    assert s.dropped == 1 and s.remaining == q_before
    assert s.want_request(), "a dropped chunk must be re-requested"
    # begin() (resume after a takeover) makes the old request's reply stale
    s.sent(3)
    s.begin()
    assert s.receive(3, ramp_chunk(0)) == "stale"
    assert s.tick() is None and s.want_request()


@test
def first_chunk_timeout_faults():
    c = Clock()
    s = ChunkScheduler(n_exec=10, first_timeout_s=5.0, clock=c)
    s.sent(1)
    for _ in range(int(5.0 / DT)):
        assert s.tick() is None
        c.advance(DT)
    c.advance(0.1)
    s.tick()
    assert s.fault and "no action chunk" in s.fault


@test
def error_reply_allows_new_request():
    c = Clock()
    s = ChunkScheduler(n_exec=10, clock=c)
    s.sent(1)
    assert not s.want_request()
    s.failed(1)
    assert s.want_request() and s.errors == 1


@test
def n_exec_limits_and_validation():
    c = Clock()
    s = ChunkScheduler(n_exec=100, clock=c)
    first(s, c, ramp_chunk(0, n=50))
    assert s.remaining == 50, "n_exec larger than the chunk: use the whole chunk"
    try:
        ChunkScheduler(n_exec=0)
        raise AssertionError("n_exec=0 must be rejected")
    except ValueError:
        pass
    try:
        s.sent(5)
        s.receive(5, np.zeros((0, 6)))
        raise AssertionError("empty chunk must be rejected")
    except ValueError:
        pass


@test
def ema_lead_adapts_to_slow_inference():
    c = Clock()
    s = ChunkScheduler(n_exec=30, rtt_guess_s=0.1, ema_alpha=0.5, clock=c)
    first(s, c, ramp_chunk(0))
    lead0 = s.lead
    for r in range(4):
        while not s.want_request():
            s.tick()
            c.advance(DT)
        k = s.step - 1
        s.sent(10 + r)
        c.advance(0.5)  # 0.5 s round trips (the policy got slow); ticks are not needed for the measurement
        s.receive(10 + r, ramp_chunk(k))
    assert s.ema_rtt > 0.4, s.ema_rtt
    assert s.lead == math.ceil(s.ema_rtt * 30) + 2 and s.lead > lead0


@test
def simulated_run_is_continuous():
    """300 ticks against a fake policy with 2-5 tick latency: the executed actions must step by exactly 1."""
    rng = np.random.default_rng(0)
    c = Clock()
    s = ChunkScheduler(n_exec=30, rtt_guess_s=0.12, clock=c)
    pending = None  # (arrival step, id, obs step)
    rid = 0
    executed = []
    s.sent(0)
    s.receive(0, ramp_chunk(0))
    for step in range(300):
        if pending and step >= pending[0]:
            assert s.receive(pending[1], ramp_chunk(pending[2])) == "ok"
            pending = None
        a = s.tick()
        executed.append(float(a[0]))
        if s.want_request():
            rid += 1
            s.sent(rid)
            pending = (step + int(rng.integers(2, 6)), rid, step)
        c.advance(DT)
    diffs = np.diff(executed)
    assert np.allclose(diffs, 1.0, atol=1e-3), f"discontinuity at {np.flatnonzero(~np.isclose(diffs, 1.0, atol=1e-3))[:5]}"
    assert s.starved == 0 and s.fault is None and s.chunks > 5



# ------------------------------------------------------------------ EnsembleScheduler (ACT temporal ensembling)
@test
def ens_first_chunk_holds_then_starts_at_zero():
    c = Clock()
    s = EnsembleScheduler(stride=2, clock=c)
    assert s.tick() is None and s.want_request()
    s.sent(1)
    assert not s.want_request(), "only one request in flight"
    for _ in range(4):
        c.advance(DT)
        assert s.tick() is None
    assert s.receive(1, ramp_chunk(0)) == "ok"
    assert s.remaining == 50, s.remaining  # every action of the chunk is kept
    assert [float(s.tick()[0]) for _ in range(3)] == [0.0, 1.0, 2.0]


@test
def ens_requests_every_stride_ticks():
    c = Clock()
    s = EnsembleScheduler(stride=2, rtt_guess_s=0.05, clock=c)
    first(s, c, ramp_chunk(0))
    asks = []
    rid = 1
    for i in range(12):
        s.tick()
        if s.want_request():
            rid += 1
            asks.append(s.step)
            s.sent(rid)
            assert s.receive(rid, ramp_chunk(s.step - 1)) == "ok"  # instant reply
        c.advance(DT)
    assert asks == list(range(2, 13, 2)), asks


@test
def ens_alignment_with_latency_is_continuous():
    """Chunks whose value == the step they are meant for: any misalignment would break the ramp (the mean of
    correctly aligned predictions of a ramp is the ramp itself)."""
    c = Clock()
    s = EnsembleScheduler(stride=2, rtt_guess_s=0.1, clock=c)
    first(s, c, ramp_chunk(0))
    rng = np.random.default_rng(0)
    executed, pending, rid = [], None, 1
    for step in range(400):
        if pending and step >= pending[0]:
            assert s.receive(pending[1], ramp_chunk(pending[2])) == "ok"
            pending = None
        executed.append(float(s.tick()[0]))
        if pending is None and s.want_request():
            rid += 1
            k = s.step - 1  # the observation of the step just executed
            s.sent(rid)
            pending = (step + int(rng.integers(1, 5)), rid, k)
        c.advance(DT)
    assert np.allclose(np.diff(executed), 1.0, atol=1e-3), "discontinuity"
    assert s.starved == 0 and s.fault is None and s.chunks > 80, (s.starved, s.chunks)


@test
def ens_weighted_mean_of_overlapping_predictions():
    c = Clock()
    s = EnsembleScheduler(stride=1, coeff=0.01, clock=c)
    first(s, c, const_chunk(0.0))  # prediction n = 0 (oldest, weight 1) for steps 0..49
    s.tick()  # step 0 executed
    s.sent(2)  # observation of step 0
    assert s.receive(2, const_chunk(10.0)) == "ok"  # prediction n = 1 (weight exp(-0.01)) for steps 1..49
    w = math.exp(-0.01)
    want = (0.0 * 1 + 10.0 * w) / (1 + w)
    got = float(s.tick()[0])
    assert abs(got - want) < 1e-4, (got, want)


@test
def ens_starvation_holds_then_faults_and_recovers():
    c = Clock()
    s = EnsembleScheduler(stride=2, starve_timeout_s=0.5, clock=c)
    first(s, c, ramp_chunk(0, n=5))
    vals = [float(s.tick()[0]) for _ in range(5)]
    assert vals == [0, 1, 2, 3, 4]
    s.sent(2)  # observation of step 4
    assert float(s.tick()[0]) == 4.0 and s.starved == 1, "repeat the last action while starved"
    c.advance(0.2)
    assert s.receive(2, ramp_chunk(4)) == "ok"
    assert s.take_recovered(), "a chunk after running dry must ask the controller to re-ramp"
    assert float(s.tick()[0]) == 6.0  # step 6: chunk[2] of the observation at step 4
    # now starve for good
    while s.remaining:
        s.tick()
    for _ in range(40):
        c.advance(DT)
        s.tick()
    assert s.fault and "starved" in s.fault, s.fault


@test
def ens_stale_late_and_error_replies():
    c = Clock()
    s = EnsembleScheduler(stride=2, clock=c)
    first(s, c, ramp_chunk(0, n=10))
    assert s.receive(99, ramp_chunk(0)) == "stale" and s.stale == 1
    s.sent(2)  # k = 0
    for _ in range(12):
        s.tick()
    assert s.receive(2, ramp_chunk(0, n=10)) == "late" and s.dropped == 1, "a chunk entirely in the past is dropped"
    s.sent(3)
    s.failed(3)
    assert s.errors == 1 and s.want_request(), "an error reply frees the slot for a new request"


@test
def ens_begin_resets_after_takeover():
    c = Clock()
    s = EnsembleScheduler(stride=2, clock=c)
    first(s, c, ramp_chunk(0))
    for _ in range(5):
        s.tick()
    s.begin()
    assert s.tick() is None and s.remaining == 0 and s.want_request(), "begin() forgets the old predictions"
    first(s, c, ramp_chunk(100), req=7)
    assert float(s.tick()[0]) == 100.0


def main():
    passed = failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\ntest_scheduler: {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

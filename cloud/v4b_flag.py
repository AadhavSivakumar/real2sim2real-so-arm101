"""Flag sim episodes whose cup / cube sat off the real desk (or overhung its edge).

Spawn bug: the sim's polar spawn ranges were never clipped to the real desk, and the physics floor is an endless plane at
desk height, so some cups/cubes rest "on air" over the splat's shelf clutter. The raw episodes (data.npz) carry only
the joint states and the expert phase, so the object positions are recovered from the arm:
  * cup centre  = held-cube centre at the end of the RELEASE segment that follows LOWER (the expert's TRANSPORT/LOWER
                  target puts the held cube centre over the cup axis: over_cup = cup - R @ hold_off_g);
  * cube centre = cube centre in the closed hand at each CLOSE segment (the DESCEND target is the cube centre).
FK is applied to the recorded ACTION (the expert's slew-limited IK command), not the measured state: the arm sags /
lags under the PD drives, so FK(state) at the first CLOSE frame falls ~15 mm short of the cube (median, vs the sim's
ground truth in the takeover logs' state0: cube-to-base distance residual -15.2 mm with the state, -2.7 mm with the
action; cup-to-TCP0 distance residual +2.5 mm median).
The expert's TCP (grasp point) is gripper-link pose @ tcp_off (expert.reset), the cube centre in the hand
~ TCP + R @ HOLD_OFF (cube pushed onto the fixed jaw; TCP target 7 mm above the 30 mm cube's centre, desk clearance).

  python3 v4b_flag.py --out flags.jsonl SRC_DIR...      (numpy only; FK from the repo's calib/fk.py)
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO = Path("/home/asubuntudesktop/TestingGrounds/so101_r2s2r")
_spec = importlib.util.spec_from_file_location("fk", REPO / "calib" / "fk.py")
fk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fk)

# real desk footprint in the sim/desk frame (calib/export_splat.py --desk_poly), CCW
DESK = np.array([(-0.12, 0.60), (-0.12, -0.226), (0.386, -0.164), (0.339, 0.60)])
CUP_R = 0.040            # cup rim radius (assets/objects/cup_two_tone.usda: top_d 0.08, bottom_d 0.06)
EDGE_MARGIN = 0.010      # cup centre must be >= CUP_R + EDGE_MARGIN inside the desk
FIXED_JAW_X, TCP_Z = -0.0088, -0.080   # env_cfg.py
HOLD_OFF = np.array([-0.004, 0.0, -0.007])  # cube centre - TCP in the gripper frame (30 mm cube), see docstring
PREGRASP, DESCEND, CLOSE, LIFT, TRANSPORT, LOWER, RELEASE, RETREAT, DONE = range(9)


def desk_dist(xy):
    """Signed distance [m] of xy to the desk polygon edge (> 0 inside)."""
    xy = np.asarray(xy, float)
    d = np.inf
    n = len(DESK)
    for i in range(n):
        a, b = DESK[i], DESK[(i + 1) % n]
        ab = b - a
        t = np.clip(np.dot(xy - a, ab) / np.dot(ab, ab), 0, 1)
        d = min(d, np.linalg.norm(xy - (a + t * ab)))
    # inside test (CCW convex polygon: left of every edge)
    cr = lambda u, v: u[0] * v[1] - u[1] * v[0]
    inside = all(cr(DESK[(i + 1) % n] - DESK[i], xy - DESK[i]) >= 0 for i in range(n))
    return d if inside else -d


def hand(q, size):
    """(tcp, cube-in-hand centre, fingertip) in the desk frame for one sim state row (rad)."""
    T = fk.link_poses(q[:5])
    Rg, pg = T[4][:3, :3], T[4][:3, 3]  # gripper link (child of wrist_roll)
    off = np.array([FIXED_JAW_X + size / 2 + 0.004, 0.0, TCP_Z])
    tcp = pg + Rg @ off + fk.T_WORLD_URDF
    return tcp, tcp + Rg @ HOLD_OFF, T[5][:3, 3] + fk.T_WORLD_URDF


def segments(phase, ph):
    m = np.r_[False, phase == ph, False].astype(int)
    st, en = np.flatnonzero(np.diff(m) == 1), np.flatnonzero(np.diff(m) == -1)
    return list(zip(st, en))  # [start, end)


def analyse(ep: Path):
    d = np.load(ep / "data.npz")
    q, phase, size = d["action"].astype(float), d["phase"], float(d["cube_size"])  # see docstring: FK of the command
    qs = d["state"].astype(float)
    r = dict(ep=ep.name, T=int(len(q)), size=size, q0=[round(float(v), 5) for v in qs[0]],
             phases=[int(p) for i, p in enumerate(phase) if i == 0 or p != phase[i - 1]])
    # cup: RELEASE segments preceded by LOWER (over_cup releases); fall back to any RELEASE segment
    rel = segments(phase, RELEASE)
    good = [(s, e) for s, e in rel if s > 0 and phase[s - 1] == LOWER]
    r["cup_method"] = "lower_release" if good else ("release" if rel else None)
    segs = good or rel
    if segs:
        s, e = segs[-1]
        tcp_e, cube_e, tip_e = hand(q[e - 1], size)
        tcp_s, cube_s, _ = hand(q[s], size)
        r["cup_xy"] = [round(float(v), 4) for v in cube_e[:2]]
        r["cup_xy_first"] = [round(float(v), 4) for v in cube_s[:2]]
        r["cup_xy_tip"] = [round(float(v), 4) for v in tip_e[:2]]
        r["n_release_lower"], r["n_release"] = len(good), len(rel)
        r["cup_d"] = round(desk_dist(cube_e[:2]), 4)
    # cube: every CLOSE segment (first frame = end of DESCEND, as specified, and last frame = jaw shut)
    cl = segments(phase, CLOSE)
    r["cube_xys"] = []
    for s, e in cl:
        _, c_first, _ = hand(q[s], size)
        r["cube_xys"].append([round(float(v), 4) for v in c_first[:2]])
    if cl:
        r["cube_xy"] = r["cube_xys"][0]
        r["cube_d"] = round(desk_dist(r["cube_xy"]), 4)
        r["cube_d_min"] = round(min(desk_dist(c) for c in r["cube_xys"]), 4)
    tj = ep / "takeover.json"
    if tj.exists():
        t = json.load(open(tj))
        r["takeover"] = dict(phase=t.get("phase"), state0=t.get("state0"), policy_steps=t.get("policy_steps"),
                             cup_shift_mm=t.get("cup_shift_mm"))
        tcp0, cube0, _ = hand(qs[0], size)  # measured TCP at the takeover (state0 is measured too)
        r["tcp0_xy"] = [round(float(v), 4) for v in tcp0[:2]]
        s0 = t.get("state0") or {}
        if "cup_d" not in r and s0.get("tcp_cup_xy_mm") is not None:
            # takeover with the cube already in / at the cup and no release left (RETREAT/DONE only): the cup is within
            # tcp_cup_xy_mm of the takeover TCP -> its worst-case distance to the desk edge
            r["cup_method"] = "takeover_tcp0"
            r["cup_xy"] = r["tcp0_xy"]
            r["cup_d"] = round(desk_dist(tcp0[:2]) - s0["tcp_cup_xy_mm"] / 1000, 4)
    # validity
    reasons = []
    if "cup_d" not in r:
        reasons.append("cup_unknown")
    elif r["cup_d"] < 0:
        reasons.append("cup_off_desk")
    elif r["cup_d"] < CUP_R + EDGE_MARGIN:
        reasons.append("cup_at_edge")
    if cl and r["cube_d"] < 0:
        reasons.append("cube_off_desk")
    elif cl and r["cube_d_min"] < 0:
        reasons.append("cube_off_desk_regrasp")
    r["invalid"] = bool(reasons)
    r["reasons"] = reasons
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("srcs", nargs="+")
    a = ap.parse_args()
    with open(a.out, "w") as fh:
        for src in a.srcs:
            eps = sorted(p for p in Path(src).glob("ep_*") if (p / "data.npz").exists())
            n_bad = 0
            for ep in eps:
                r = analyse(ep)
                r["source"] = Path(src).name
                n_bad += r["invalid"]
                fh.write(json.dumps(r) + "\n")
            print(f"{Path(src).name}: {len(eps)} episodes, invalid {n_bad} ({n_bad / max(1, len(eps)):.1%})", flush=True)

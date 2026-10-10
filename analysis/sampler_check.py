"""Numpy replica of mdp.reset_cube_and_cup (old and desk-constrained) + spawn-distribution check.

Usage: python sampler_check.py [N]
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
rng = np.random.default_rng(0)

DESK_POLY = np.array([(-0.12, 0.60), (-0.12, -0.226), (0.386, -0.164), (0.339, 0.60)])
CUBE = dict(r=(0.12, 0.27), phi=(-1.1, 1.1))
CUP = dict(r=(0.15, 0.32), phi=(-1.2, 1.2))
MIN_SEP, KEEP = 0.11, ((0.17, 0.0), 0.10)
CUBE_HALF, CUP_R, MARGIN = 0.015, 0.040, 0.015  # 30 mm cube; cup outer rim radius (CUP_DIMS top_d / 2)


def desk_inside(p, poly=DESK_POLY):
    """Signed distance [m] of points p[..., 2] inside a convex polygon (min over the edge half-planes)."""
    a = poly
    if 0.5 * np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1]) < 0:
        a = a[::-1]  # make it CCW
    b = np.roll(a, -1, 0)
    d = b - a
    nrm = np.stack([-d[:, 1], d[:, 0]], -1) / np.linalg.norm(d, axis=-1, keepdims=True)  # inward (left) normals
    return np.min(np.einsum("...ec,ec->...e", p[..., None, :] - a, nrm), -1)


def polar(rr, shape):
    r = rng.uniform(*rr["r"], shape)
    phi = rng.uniform(*rr["phi"], shape)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], -1)


def sample(n, K, desk):
    cube, cup = polar(CUBE, (n, K)), polar(CUP, (n, K))
    ok = np.linalg.norm(cube - cup, axis=-1) > MIN_SEP
    ok &= np.linalg.norm(cup - np.array(KEEP[0]), axis=-1) > KEEP[1]
    on = np.ones_like(ok)
    if desk:
        on = (desk_inside(cube) >= CUBE_HALF + MARGIN) & (desk_inside(cup) >= CUP_R + MARGIN)
        ok &= on
    # mdp: first fully valid candidate, else the first on-desk one, else candidate 0
    score = 2 * ok + on
    first = np.argmax(score, 1)
    rows = np.arange(n)
    fb = ~ok.any(1)
    return cube[rows, first], cup[rows, first], fb, ok.mean(), on.mean()


def report(tag, cube, cup, fb):
    dc, du = desk_inside(cube), desk_inside(cup)
    print(f"[{tag}] fallback {fb.mean():.2e} ({fb.sum()} / {len(fb)})")
    print(f"   cube: centre off desk {np.mean(dc < 0):6.2%}   within half-size+1.5cm of an edge {np.mean((dc >= 0) & (dc < CUBE_HALF + MARGIN)):6.2%}")
    print(f"   cup : centre off desk {np.mean(du < 0):6.2%}   overhanging (centre < r_cup=4cm inside) {np.mean((du >= 0) & (du < CUP_R)):6.2%}"
          f"   centre < 4 cm inside {np.mean(du < 0.04):6.2%}   < r+1.5cm {np.mean(du < CUP_R + MARGIN):6.2%}")
    for nm, p in (("cube", cube), ("cup", cup)):
        r, phi = np.hypot(*p.T), np.arctan2(p[:, 1], p[:, 0])
        print(f"   {nm}: x {p[:,0].min():+.3f}..{p[:,0].max():+.3f}  y {p[:,1].min():+.3f}..{p[:,1].max():+.3f}  "
              f"r {r.min():.3f}..{r.max():.3f}  phi {phi.min():+.2f}..{phi.max():+.2f}  frac y<0 {np.mean(p[:,1] < 0):.3f}  "
              f"y<-0.10 {np.mean(p[:,1] < -0.10):.3f}")


print("=== old sampler (K=32, no desk constraint)")
cube0, cup0, fb0, acc0, _ = sample(N, 32, desk=False)
print(f"   per-candidate acceptance {acc0:.3f}")
report("old", cube0, cup0, fb0)

print("\n=== per-candidate acceptance with the desk constraint")
_, _, _, acc1, on1 = sample(200_000, 32, desk=True)
print(f"   all constraints {acc1:.3f}   desk alone {on1:.3f}")
for K in (32, 64, 128):
    print(f"   K={K:4d}: P(fallback) ~ (1-p)^K = {(1 - acc1) ** K:.1e}")

K = 64
print(f"\n=== new sampler (K={K}, desk constraint)")
cube1, cup1, fb1, _, _ = sample(N, K, desk=True)
report("new", cube1, cup1, fb1)

# what the constraint removes: share of the old spawn distribution (per object) that is no longer allowed
cube_bad = desk_inside(cube0) < CUBE_HALF + MARGIN
cup_bad = desk_inside(cup0) < CUP_R + MARGIN
print("\n=== share of the old spawns now excluded")
for nm, p, bad in (("cube", cube0, cube_bad), ("cup", cup0, cup_bad)):
    right = p[:, 1] < 0
    print(f"   {nm}: {bad.mean():.2%} of all spawns; {bad[right].mean():.2%} of the right half (y<0); "
          f"{bad[~right].mean():.2%} of the left half; excluded by edge: "
          f"right {np.mean(bad & right):.2%}, back {np.mean(bad & (p[:,0] > 0.28) & ~right):.2%}")

# histograms by y band (old vs new)
print("\n=== y-band histogram (fraction of spawns): old -> new")
bands = [(-0.40, -0.20), (-0.20, -0.15), (-0.15, -0.10), (-0.10, -0.05), (-0.05, 0.0), (0.0, 0.10), (0.10, 0.20), (0.20, 0.40)]
for nm, a, b in (("cube", cube0, cube1), ("cup", cup0, cup1)):
    print(f"   {nm}: " + "  ".join(f"[{lo:+.2f},{hi:+.2f}) {np.mean((a[:,1]>=lo)&(a[:,1]<hi)):.3f}->{np.mean((b[:,1]>=lo)&(b[:,1]<hi)):.3f}"
                             for lo, hi in bands))

# coverage of the real demos: is each real position within 2 cm of where the new sampler puts objects with
# non-negligible density, and does it pass the desk constraint?
real = np.load(HERE / "real_positions.npz")
print("\n=== real demo positions (FK at grasp / release, +-~3 cm)")
for nm, P, S0, S1, need in (("cube", real["cube"], cube0, cube1, CUBE_HALF + MARGIN),
                            ("cup", real["cup"], cup0, cup1, CUP_R + MARGIN)):
    P = P[:, :2]
    if nm == "cube":
        good = real["cube"][:, 2] < 0.035  # grasp on the desk (not a re-grasp in the air)
    else:
        good = real["cup"][:, 2] > 0.07  # released above the rim
    P = P[good]
    din = desk_inside(P)
    near0 = np.array([np.mean(np.linalg.norm(S0[:200_000] - p, axis=-1) < 0.02) for p in P])
    near1 = np.array([np.mean(np.linalg.norm(S1[:200_000] - p, axis=-1) < 0.02) for p in P])
    r = np.hypot(*P.T)
    print(f"   {nm}: n={len(P)} ; y<0: {np.sum(P[:,1] < 0)} ; inside the desk polygon by "
          f"min {din.min()*100:.1f} cm (need {need*100:.1f}) ; pass constraint {np.sum(din >= need)}/{len(P)}")
    print(f"        density within 2 cm (fraction of spawns), old median {np.median(near0):.4f} / new median {np.median(near1):.4f};"
          f" positions with ~zero new density (<1e-4): {np.sum(near1 < 1e-4)} (r of those: {np.round(r[near1 < 1e-4], 3).tolist()})")
    for p, d, a0, a1 in sorted(zip(P.tolist(), din, near0, near1), key=lambda z: z[0][1])[:6]:
        print(f"        rightmost: ({p[0]:+.3f},{p[1]:+.3f}) inside {d*100:4.1f} cm  density old {a0:.4f} new {a1:.4f}")
np.savez(HERE / "sampler_samples.npz", cube0=cube0[:100000], cup0=cup0[:100000], cube1=cube1[:100000], cup1=cup1[:100000])

"""Cube positions of the real demos (saved by desk_check/real_positions.py) vs. today's real policy rollouts.

Rollouts: cube = fingertip xy at the grasp (same rule as real_positions.py); a run with no grasp gets the
fingertip xy where the hand hovered longest with the jaw open (it never reached the cube, so that is approximate).
"""
import glob, importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path("/home/asubuntudesktop/TestingGrounds/so101_r2s2r")
def _module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
units = _module(REPO / "so101_r2s2r" / "units.py"); fk = _module(REPO / "calib" / "fk.py")
RO = Path("/mnt/secondary/so101_r2s/report/rollouts")

def positions(ds):
    t = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{ds}/data/*/*.parquet"))])
    side = [json.loads(l) for l in open(f"{ds}/meta/so101_rollouts.jsonl")]
    out = []
    for ep in sorted(t.episode_index.unique()):
        e = t[t.episode_index == ep].sort_values("frame_index")
        s = np.stack(e["observation.state"].values).astype(np.float64)
        g = s[:, 5]
        tip = np.stack([fk.fingertip_in_world(qq[:5]) for qq in units.lerobot_to_sim(s)])
        # jaw reading with the 30 mm cube held is ~14.6 (same rule as real_positions.py: within 4 of the 5th percentile)
        closed = g < np.percentile(g, 5) + 4.0
        held = np.percentile(g, 5) < 17.0
        k = next((i for i in range(len(g) - 10) if held and closed[i:i + 10].all()), None)
        if k is not None:
            p, how = np.median(tip[k:k + 3], 0), "grasp"
        else:   # longest low hover with the jaw open
            low = (tip[:, 2] < 0.08) & ~closed
            p, how = np.median(tip[low], 0) if low.any() else tip[-1], "hover"
        meta = next(m for m in side if m["episode_index"] == ep)
        out.append(dict(ds=Path(ds).name, ep=int(ep), t=meta["t"][11:16], outcome=meta["outcome"],
                        dur=meta["duration_s"], how=how, x=float(p[0]), y=float(p[1]), z=float(p[2]),
                        gmin=float(g.min())))
    return out

rows = positions(RO / "rollouts_act_real40_80k") + positions(RO / "rollouts_act_v4_80k")
for r in rows:
    print(f"{r['ds']:24s} ep{r['ep']} {r['t']} {r['outcome']:7s} {r['dur']:5.1f}s  {r['how']:5s} cube ({r['x']:+.3f},{r['y']:+.3f}) z{r['z']:+.3f}  "
          f"r {np.hypot(r['x'], r['y']):.3f} phi {np.degrees(np.arctan2(r['y'], r['x'])):+.0f} deg  gmin {r['gmin']:.1f}")
json.dump(rows, open(Path(__file__).with_name("rollout_positions.json"), "w"), indent=1)

d = np.load("/mnt/secondary/so101_r2s/videos_v4/desk_check/real_positions.npz", allow_pickle=True)
c, names = d["cube"], d["names"]
phi = np.degrees(np.arctan2(c[:, 1], c[:, 0])); r = np.hypot(c[:, 0], c[:, 1])
print(f"\nreal demos: n={len(c)}  right (y<0): {(c[:,1]<0).sum()}  left: {(c[:,1]>=0).sum()}")
for lo, hi in ((-70, -40), (-40, -20), (-20, 0), (0, 20), (20, 40), (40, 70)):
    m = (phi >= lo) & (phi < hi)
    print(f"  phi {lo:+4d}..{hi:+4d} deg: {m.sum():2d}   r {r[m].min() if m.any() else 0:.2f}-{r[m].max() if m.any() else 0:.2f}")
print("right-side demos (x, y, r, phi):")
for n, p, rr, pp in zip(names, c, r, phi):
    if p[1] < 0: print(f"  {n:22s} ({p[0]:+.3f},{p[1]:+.3f})  r {rr:.3f}  phi {pp:+.0f}")

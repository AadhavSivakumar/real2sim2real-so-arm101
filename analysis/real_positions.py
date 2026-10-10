"""Cube (grasp) and cup (release) positions of the user's 40 real demos, from the follower state via FK.

cube xy = fingertip xy at the grasp (first frame the jaw reading settles closed);
cup  xy = fingertip xy at the release (first frame after the grasp the jaw opens again).
Writes real_positions.npz next to this script.
"""
import glob
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path("/home/asubuntudesktop/TestingGrounds/so101_r2s2r")


def _module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


units = _module(REPO / "so101_r2s2r" / "units.py")
fk = _module(REPO / "calib" / "fk.py")
DS = ["/mnt/secondary/so101_r2s/lerobot_datasets/real_cube_cup_v1",
      "/mnt/secondary/so101_r2s/lerobot_datasets/cube_cup_real_v2"]

rows = []
for ds in DS:
    fs = sorted(glob.glob(f"{ds}/data/*/*.parquet"))
    import pandas as pd
    t = pd.concat([pq.read_table(f).to_pandas() for f in fs])
    for ep in sorted(t.episode_index.unique()):
        e = t[t.episode_index == ep].sort_values("frame_index")
        s = np.stack(e["observation.state"].values).astype(np.float64)
        g = s[:, 5]
        g_closed = np.percentile(g, 5)
        closed = g < g_closed + 4.0
        # grasp: first index where the jaw is closed for >= 10 consecutive frames
        k = None
        for i in range(len(g) - 10):
            if closed[i:i + 10].all():
                k = i
                break
        if k is None:
            print("no grasp", ds, ep); continue
        j = k + 10
        while j < len(g) and closed[j]:
            j += 1
        if j >= len(g):
            print("no release", ds, ep); continue
        q = units.lerobot_to_sim(s)
        tip = np.stack([fk.fingertip_in_world(qq[:5]) for qq in q])
        # cube: tip at the grasp (median of the 3 frames from the closure); cup: tip at the release frame
        cube = np.median(tip[k:k + 3], 0)
        cup = tip[j]
        rows.append((Path(ds).name, int(ep), *cube, *cup, g_closed))
        print(f"{Path(ds).name:22s} ep{ep:2d}  cube ({cube[0]:+.3f},{cube[1]:+.3f}) z{cube[2]:+.3f}   cup ({cup[0]:+.3f},{cup[1]:+.3f}) z{cup[2]:+.3f}  grip_closed {g_closed:.1f}")

arr = np.array([r[2:] for r in rows])
np.savez(Path(__file__).with_name("real_positions.npz"), cube=arr[:, 0:3], cup=arr[:, 3:6],
         names=np.array([f"{r[0]}:{r[1]}" for r in rows]))
cube, cup = arr[:, 0:3], arr[:, 3:6]
for nm, p in (("cube", cube), ("cup", cup)):
    r, phi = np.hypot(p[:, 0], p[:, 1]), np.arctan2(p[:, 1], p[:, 0])
    print(f"{nm}: n={len(p)} x {p[:,0].min():.3f}..{p[:,0].max():.3f}  y {p[:,1].min():+.3f}..{p[:,1].max():+.3f}  "
          f"r {r.min():.3f}..{r.max():.3f}  phi {phi.min():+.2f}..{phi.max():+.2f}  left(y>0) {np.mean(p[:,1]>0):.2f}")

"""Cube (first grasp) positions of every episode in cotrain_v4, split into sim and real (same rule as real_positions.py)."""
import glob, importlib.util, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq

REPO = Path("/home/asubuntudesktop/TestingGrounds/so101_r2s2r")
def _module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
units = _module(REPO / "so101_r2s2r" / "units.py"); fk = _module(REPO / "calib" / "fk.py")
DS = "/mnt/secondary/so101_r2s/lerobot_datasets/cotrain_v4"
src = json.load(open(f"{DS}/meta/v4_sources.json"))
t = pd.concat([pq.read_table(f, columns=["observation.state", "episode_index", "frame_index"]).to_pandas()
               for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))])
eps = sorted(t.episode_index.unique())
lens = t.groupby("episode_index").size()
print("episodes", len(eps), "sim_frames", src["sim_frames"], "frames in first 4013 eps", int(lens.iloc[:4013].sum()))
out = []
for ep, e in t.groupby("episode_index"):
    s = np.stack(e.sort_values("frame_index")["observation.state"].values).astype(np.float64)
    g = s[:, 5]; closed = g < np.percentile(g, 5) + 4.0
    k = next((i for i in range(len(g) - 10) if closed[i:i + 10].all()), None)
    if k is None: continue
    q = units.lerobot_to_sim(s[k:k + 3])
    tip = np.median(np.stack([fk.fingertip_in_world(qq[:5]) for qq in q]), 0)
    out.append((int(ep), *tip))
a = np.array(out)
np.save(Path(__file__).with_name("train_positions.npy"), a)
sim = a[a[:, 0] < 4013]; real = a[a[:, 0] >= 4013]
for nm, p in (("sim", sim), ("real(x19)", real)):
    print(f"{nm}: n={len(p)}  right (y<0) {np.mean(p[:,2] < 0):.0%}  z median {np.median(p[:,3]):.3f}")
fail = np.array([0.104, -0.145])
for nm, p in (("sim", sim),):
    near = np.hypot(p[:, 1] - fail[0], p[:, 2] - fail[1]) < 0.05
    print(f"{nm} within 5 cm of the real-only failure spot: {near.sum()}")
d = np.load("/mnt/secondary/so101_r2s/videos_v4/desk_check/real_positions.npz")
c = d["cube"]; near = np.hypot(c[:, 0] - fail[0], c[:, 1] - fail[1]) < 0.05
print(f"real demos within 5 cm of it: {near.sum()} of {len(c)}")

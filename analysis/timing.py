import glob, json
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
RO = Path("/mnt/secondary/so101_r2s/report/rollouts")
for ds in ("rollouts_act_real40_80k", "rollouts_act_v4_80k"):
    t = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(f"{RO/ds}/data/*/*.parquet"))])
    for ep, e in t.groupby("episode_index"):
        e = e.sort_values("frame_index"); g = np.stack(e["observation.state"].values)[:, 5]
        closed = g < np.percentile(g, 5) + 4.0
        k = next((i for i in range(len(g) - 10) if np.percentile(g, 5) < 17 and closed[i:i + 10].all()), None)
        if k is None: print(ds, ep, "no grasp"); continue
        j = k + 10
        while j < len(g) and closed[j]: j += 1
        print(f"{ds:24s} ep{ep}  grasp {k/30:5.1f}s  release {j/30:5.1f}s  episode {len(g)/30:5.1f}s")

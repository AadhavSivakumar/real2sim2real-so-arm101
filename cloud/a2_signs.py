"""Offline replay of record_takeover.py's stuck signs on --save_traces traces (policy part of each attempt)."""
import glob
import json
import os
import sys

import numpy as np

out = sys.argv[1]
recs = {}
for line in open(os.path.join(out, "takeover_log.jsonl")):
    r = json.loads(line)
    recs[r["attempt"]] = r
DT = 1 / 30
QC = (0.030 - 0.017) / 0.066


def signs(tr, static_s=1.5, hover_s=2.0, hover_r=0.10, prog=0.01, amp=0.04):
    t, tcp, cube, q, held = tr[:, 0], tr[:, 1:4], tr[:, 4:7], tr[:, 7], tr[:, 8] > 0.5
    dist = np.linalg.norm(cube - tcp, axis=1)
    xy = np.linalg.norm(cube[:, :2] - tcp[:, :2], axis=1)
    ok = (q > QC + 0.15) & ~held & (cube[:, 2] < 0.03)
    ns, nh = int(round(static_s / DT)), int(round(hover_s / DT))
    first = {}
    for i in range(len(t)):
        if t[i] <= 1.0:
            continue
        if "static" not in first and i + 1 >= ns and np.ptp(tcp[i + 1 - ns:i + 1], 0).max() < 0.015:
            first["static"] = t[i]
        if i + 1 >= nh and ok[i + 1 - nh:i + 1].all():
            if "dither" not in first and np.ptp(tcp[i + 1 - nh:i + 1], 0).max() < amp:
                first["dither"] = t[i]
            if "hover" not in first and i + 1 > nh and xy[i] < hover_r and (t[:i + 1 - nh] > 1.0).any() and dist[i + 1 - nh:i + 1].min() > dist[:i + 1 - nh][t[:i + 1 - nh] > 1.0].min() - prog:
                first["hover"] = t[i]
    return first, dist, xy, q


for f in sorted(glob.glob(os.path.join(out, "traces", "attempt_*.npz"))):
    k = int(f[-10:-4])
    tr = np.load(f)["trace"]
    r = recs.get(k, {})
    first, dist, xy, q = signs(tr, hover_s=float(os.environ.get("HS", 2.0)), prog=float(os.environ.get("PROG", 0.01)))
    end = r.get("outcome", "?")
    trig = r.get("trigger", "-")
    s0 = r.get("state0", {})
    # summary of the last 3 s of the policy part
    n3 = min(len(tr), 90)
    tail = f"last3s: dist {dist[-n3:].min()*100:.1f}-{dist[-n3:].max()*100:.1f}cm xy {xy[-1]*100:.1f}cm tcp_z {tr[-1,3]*100:.1f}cm " \
           f"tcp range {np.ptp(tr[-n3:,1:4],0).max()*100:.1f}cm q {q[-n3:].min():.2f}-{q[-n3:].max():.2f}"
    print(f"att {k} seed {r.get('seed')} T={tr[-1,0]:.1f}s trig={trig} pre/expert={end} signs={ {a: round(b,1) for a,b in first.items()} } | {tail}")

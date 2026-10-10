"""v4 recording statistics (ON r2s-sim, plain python3): per sim_v4_* dir the saved episodes and frames, and
  takeover dirs: attempts, policy successes before any takeover, takeovers, expert recoveries (all / recoverable),
            failure outcomes and trigger mix (from takeover_log.jsonl);
  dart dirs : saved / attempts (from the last "[rec] saved ep" line of ~/logs/v4_<name>.log).
  python3 v4_stats.py [--json out.json]
"""
import glob
import json
import os
import re
import sys
from collections import Counter

import numpy as np  # noqa: F401  (frames are read with numpy when available)

S = "/mnt/secondary/so101_r2s/sim_demos"
rows = {}
for d in sorted(glob.glob(f"{S}/sim_v4_*")):
    name = os.path.basename(d)[len("sim_v4_"):]
    eps = sorted(glob.glob(f"{d}/ep_*"))
    frames = 0
    for e in eps:
        with np.load(f"{e}/data.npz") as z:
            frames += len(z["state"])
    r = dict(episodes=len(eps), frames=frames, splat=name.split("_")[1].startswith("sp"))
    if name.startswith("takeover"):
        log = [json.loads(l) for l in open(f"{d}/takeover_log.jsonl") if l.strip()]
        out = Counter(x.get("outcome") for x in log)
        tk = [x for x in log if x.get("outcome") != "policy_success" and "t_takeover" in x]
        rec = [x for x in tk if not x.get("unrecoverable")]
        r.update(attempts=len(log), policy_success=out.get("policy_success", 0), takeovers=len(tk),
                 recovered=sum(x["outcome"] == "success" for x in tk),
                 recoverable=len(rec), recovered_recoverable=sum(x["outcome"] == "success" for x in rec),
                 outcomes=dict(out), triggers=dict(Counter(x.get("trigger") for x in tk)),
                 trigger_success=dict(Counter(x.get("trigger") for x in tk if x["outcome"] == "success")))
    else:
        # "attempts A" counts from 1 in every (re)started run: sum the last value of each run
        att = [int(m.group(1)) for m in re.finditer(r"\[rec\] saved ep \d+/\d+ .*?\(attempts (\d+),",
                                                     open(os.path.expanduser(f"~/logs/v4_{name}.log"), errors="ignore").read())]
        r.update(attempts_logged=sum(a for i, a in enumerate(att) if i + 1 == len(att) or att[i + 1] <= a),
                 runs=1 + sum(att[i + 1] <= a for i, a in enumerate(att[:-1])))
    rows[name] = r
    print(name, r, flush=True)

tot = Counter()
for n, r in rows.items():
    kind = "takeover" if n.startswith("takeover") else "dart"
    tot[f"{kind}_{'splat' if r['splat'] else 'nosplat'}_episodes"] += r["episodes"]
    tot[f"{kind}_frames"] += r["frames"]
print("TOTAL", dict(tot))
if len(sys.argv) > 2 and sys.argv[1] == "--json":
    json.dump(dict(rows=rows, total=dict(tot)), open(sys.argv[2], "w"), indent=1)

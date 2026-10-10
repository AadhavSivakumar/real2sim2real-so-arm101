"""ON r2s-h100 (LeRobot venv): build cotrain_v4b = cotrain_v4 minus the off-desk sim episodes (v4b_flag.py) and minus
enough real duplicates to keep the real share at ~--real_frac of frames. cotrain_v4 is only read, never modified.

Route: LeRobot's dataset_tools.delete_episodes (filters + re-indexes the data parquet, the episode metadata and the
aggregated stats), with its video step replaced by HARD LINKS of the source mp4s: every kept episode keeps its video
file and from/to timestamps, so nothing is re-encoded (frames of dropped episodes stay in the files, unreferenced).

  ~/lr/bin/python v4b_build.py --flags v4b_flags.jsonl --src cotrain_v4 --out cotrain_v4b
"""
import argparse
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import lerobot.datasets.dataset_tools as dt
from lerobot.datasets import LeRobotDataset

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path.home() / "r2s" / "lerobot_datasets"))
ap.add_argument("--src", default="cotrain_v4")
ap.add_argument("--out", default="cotrain_v4b")
ap.add_argument("--flags", required=True)
ap.add_argument("--real_frac", type=float, default=0.25)
ap.add_argument("--dry", action="store_true")
args = ap.parse_args()
root = Path(args.root)
src_root, out_root = root / args.src, root / args.out
assert not out_root.exists(), f"{out_root} exists"

t0 = time.time()
srcinfo = json.loads((src_root / "meta" / "v4_sources.json").read_text())
flags = [json.loads(line) for line in open(args.flags)]
by_src = {}
for r in flags:
    by_src.setdefault(r["source"], []).append(r)
for v in by_src.values():
    v.sort(key=lambda r: r["ep"])


def part_size(src):
    if src.startswith("sim_v4_"):
        return 40
    k = int(src.rsplit("_s", 1)[1])
    return 84 if k <= 4 else 63


# ---- 1. sim episode index -> raw episode, in the aggregate's part order
order = []
for part in srcinfo["sims"]:
    src, k = part[len("v4p_"):].rsplit("_", 1)
    P = part_size(src)
    order += [(part, r) for r in by_src[src][int(k) * P:(int(k) + 1) * P]]
eps = pd.read_parquet(sorted((src_root / "meta" / "episodes").rglob("*.parquet"))[0])
for f in sorted((src_root / "meta" / "episodes").rglob("*.parquet"))[1:]:
    eps = pd.concat([eps, pd.read_parquet(f)], ignore_index=True)
eps = eps.sort_values("episode_index").reset_index(drop=True)
n_sim = len(order)
n_tot = len(eps)
print(f"[v4b] {len(srcinfo['sims'])} parts -> {n_sim} sim episodes in the raw tables; {args.src} has {n_tot} episodes", flush=True)
assert n_sim == n_tot - 40 * srcinfo["real_dup"], (n_sim, n_tot, srcinfo["real_dup"])

# verify the mapping: episode length and first-frame observation.state (LeRobot units) against data.npz
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("units", Path(__file__).resolve().parent / "units.py")
units = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(units)
data = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"])
                  for f in sorted((src_root / "data").rglob("*.parquet"))], ignore_index=True)
first = data[data["frame_index"] == 0].set_index("episode_index")["observation.state"]
bad_len, bad_state, max_dev = [], [], 0.0
for i, (part, r) in enumerate(order):
    if int(eps.loc[i, "length"]) != r["T"]:
        bad_len.append((i, part, r["ep"], int(eps.loc[i, "length"]), r["T"]))
    dev = float(np.abs(np.asarray(first[i], float) - units.sim_to_lerobot(np.asarray(r["q0"]))).max())
    max_dev = max(max_dev, dev)
    if dev > 0.05:
        bad_state.append((i, part, r["ep"], round(dev, 3)))
print(f"[v4b] mapping check: {len(bad_len)} length mismatches, {len(bad_state)} first-state mismatches "
      f"(max |dev| {max_dev:.4f} LeRobot units)", flush=True)
if bad_len or bad_state:
    print("[v4b] MISMATCH", bad_len[:10], bad_state[:10], flush=True)
    raise SystemExit(1)
# the real block: 40 episodes per duplicate, identical lengths across duplicates
rl = eps["length"].to_numpy()[n_sim:].reshape(srcinfo["real_dup"], 40)
assert (rl == rl[0]).all() and int(rl[0].sum()) == srcinfo["real_frames"], "real duplicates are not 19 x 40 identical"

# ---- 2. what to keep
drop_sim = [i for i, (_, r) in enumerate(order) if r["invalid"]]
keep_sim = [i for i in range(n_sim) if i not in set(drop_sim)]
fs = int(eps.loc[keep_sim, "length"].sum())
fr = srcinfo["real_frames"]
dup = max(1, round(args.real_frac * fs / ((1 - args.real_frac) * fr)))
keep_real = list(range(n_sim, n_sim + 40 * dup))
drop = drop_sim + list(range(n_sim + 40 * dup, n_tot))
share = dup * fr / (fs + dup * fr)
per_src = {}
for i, (part, r) in enumerate(order):
    s = per_src.setdefault(r["source"], [0, 0])
    s[0] += 1
    s[1] += r["invalid"]
print(f"[v4b] sim: drop {len(drop_sim)}/{n_sim} episodes ({len(drop_sim) / n_sim:.1%}), keep {len(keep_sim)} = {fs} frames "
      f"(was {srcinfo['sim_frames']}); real {fr} frames x{dup} (was x{srcinfo['real_dup']}) -> real share {share:.3f}; "
      f"total {len(keep_sim) + len(keep_real)} episodes, {fs + dup * fr} frames", flush=True)
if args.dry:
    raise SystemExit(0)


# ---- 3. delete_episodes with hard-linked videos
def link_videos(src_dataset, dst_meta, episode_mapping):
    out = {new: {} for new in episode_mapping.values()}
    E = src_dataset.meta.episodes
    for vk in src_dataset.meta.video_keys:
        done = set()
        for old, new in episode_mapping.items():
            e = E[old]
            c, f = e[f"videos/{vk}/chunk_index"], e[f"videos/{vk}/file_index"]
            if (c, f) not in done:
                s = src_dataset.root / src_dataset.meta.video_path.format(video_key=vk, chunk_index=c, file_index=f)
                d = dst_meta.root / dst_meta.video_path.format(video_key=vk, chunk_index=c, file_index=f)
                d.parent.mkdir(parents=True, exist_ok=True)
                os.link(s, d)
                done.add((c, f))
            out[new].update({f"videos/{vk}/chunk_index": c, f"videos/{vk}/file_index": f,
                             f"videos/{vk}/from_timestamp": e[f"videos/{vk}/from_timestamp"],
                             f"videos/{vk}/to_timestamp": e[f"videos/{vk}/to_timestamp"]})
        print(f"[v4b] {vk}: hard-linked {len(done)} mp4s", flush=True)
    return out


@lru_cache(maxsize=1)
def _episodes_df(meta_root):
    # all meta/episodes rows, read once. (The rows' own meta/episodes/{chunk,file}_index columns are NOT trusted: the
    # real duplicates carry the Jetson recorder's cycling 0..9 file_index while all rows sit in file-000.parquet.)
    fs = sorted((Path(meta_root) / "episodes").rglob("*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True).set_index("episode_index", drop=False)


def load_episode_with_stats(src_dataset, episode_idx):  # = dt._load_episode_with_stats, without the file lookup
    return _episodes_df(str(src_dataset.root / "meta")).loc[episode_idx].to_dict()


dt._copy_and_reindex_videos = link_videos
dt._load_episode_with_stats = load_episode_with_stats
src_ds = LeRobotDataset(f"local/{args.src}", root=src_root, video_backend="torchcodec")
new_ds = dt.delete_episodes(src_ds, drop, output_dir=out_root, repo_id=f"local/{args.out}")
print(f"[v4b] wrote {out_root}: {new_ds.meta.total_episodes} episodes, {new_ds.meta.total_frames} frames "
      f"({time.time() - t0:.0f} s)", flush=True)
assert new_ds.meta.total_episodes == len(keep_sim) + len(keep_real)
assert new_ds.meta.total_frames == fs + dup * fr
(out_root / "meta" / "v4b_sources.json").write_text(json.dumps(dict(
    srcinfo, base=args.src, real_dup=dup, sim_frames=fs, real_share=share, sim_episodes=len(keep_sim),
    dropped_sim=[dict(v4_episode=i, part=order[i][0], source=order[i][1]["source"], ep=order[i][1]["ep"],
                      reasons=order[i][1]["reasons"]) for i in drop_sim],
    invalid_per_source=per_src, rule="cup centre (FK of the release command) >= 0.05 m inside the desk polygon, "
                                    "cube (FK of every grasp command) inside the desk; v4b_flag.py"), indent=1))

# ---- 4. check: kept episodes decode to exactly the source frames / states
new_ds = LeRobotDataset(f"local/{args.out}", root=out_root, video_backend="torchcodec")
kept = keep_sim + keep_real
rng = np.random.default_rng(0)
checks = sorted({0, len(keep_sim) - 1, len(keep_sim), len(kept) - 1, *rng.choice(len(kept), 6, replace=False).tolist()})
for new_i in checks:
    old_i = kept[new_i]
    a, b = new_ds.meta.episodes[new_i], src_ds.meta.episodes[old_i]
    assert a["length"] == b["length"]
    for off in (0, a["length"] // 2, a["length"] - 1):
        x, y = new_ds[a["dataset_from_index"] + off], src_ds[b["dataset_from_index"] + off]
        assert int(x["episode_index"]) == new_i and int(y["episode_index"]) == old_i
        assert int(x["frame_index"]) == int(y["frame_index"]) == off
        for k in ("observation.state", "action", "observation.images.wrist", "observation.images.scene"):
            assert np.array_equal(x[k].numpy(), y[k].numpy()), (new_i, old_i, off, k)
    print(f"[v4b] check new ep {new_i} == {args.src} ep {old_i} ({a['length']} frames): 3 frames identical "
          f"(state, action, both cameras)", flush=True)
print(f"[v4b] DONE {time.time() - t0:.0f} s", flush=True)

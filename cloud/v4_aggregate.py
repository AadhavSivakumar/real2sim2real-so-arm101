"""ON r2s-sim (LeRobot venv): aggregate converted sim parts + undistorted real datasets into one co-training dataset.

Real datasets are duplicated so that real frames are ~--real_frac of all frames. Videos are HARD-LINKED, not copied:
aggregate_datasets(concatenate_videos=False) keeps one mp4 per source file, and its shutil.copy of those files is
replaced by os.link, so neither the aggregate nor the real duplicates cost extra disk (parquet data / metadata are
written normally). The source parts can be deleted afterwards; their videos live on in the aggregate.

  ~/lr/bin/python v4_aggregate.py --out cotrain_v4 --sims <part names> --reals real_cube_cup_v1_undist cube_cup_real_v2_undist
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import lerobot.datasets.aggregate as agg

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(Path.home() / "r2s" / "lerobot_datasets"))
ap.add_argument("--out", required=True)
ap.add_argument("--sims", nargs="+", required=True)
ap.add_argument("--reals", nargs="+", required=True)
ap.add_argument("--real_frac", type=float, default=0.25)
args = ap.parse_args()
root = Path(args.root)


class _LinkingShutil:
    """Stands in for the shutil module inside lerobot.datasets.aggregate (used there only for whole-mp4 copies)."""

    def __getattr__(self, k):
        return getattr(shutil, k)

    @staticmethod
    def copy(src, dst, *a, **kw):
        try:
            os.link(src, dst)
            return dst
        except OSError:
            return shutil.copy(src, dst, *a, **kw)


agg.shutil = _LinkingShutil()


def info(p):
    return json.loads((p / "meta" / "info.json").read_text())


def fix_episode_meta_index(p):
    """The Jetson recorder's meta/episodes parquet carries meta/episodes/file_index 0..9 cycling while all episodes are
    in file-000.parquet (cube_cup_real_v2); aggregate_metadata opens the file named by those columns. Point them at the
    file each row is actually in (on the duplicate's own copy of meta/, never on the source)."""
    import pandas as pd

    for f in sorted((p / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        c, i = int(f.parent.name.split("-")[1]), int(f.stem.split("-")[1])
        df = pd.read_parquet(f)
        if (df["meta/episodes/chunk_index"] != c).any() or (df["meta/episodes/file_index"] != i).any():
            print(f"[agg] {p.name}: meta/episodes index columns -> chunk {c} file {i} in {f.name}", flush=True)
            df["meta/episodes/chunk_index"] = c
            df["meta/episodes/file_index"] = i
            df.to_parquet(f, index=False)


sims = [root / s for s in args.sims]
reals = [root / r for r in args.reals]
fs = sum(info(p)["total_frames"] for p in sims)
fr = sum(info(p)["total_frames"] for p in reals)
dup = max(1, round(args.real_frac * fs / ((1 - args.real_frac) * fr)))
print(f"[agg] {len(sims)} sim parts: {sum(info(p)['total_episodes'] for p in sims)} episodes, {fs} frames; "
      f"real {[p.name for p in reals]}: {sum(info(p)['total_episodes'] for p in reals)} episodes, {fr} frames; "
      f"real x{dup} -> real share {dup * fr / (fs + dup * fr):.3f}", flush=True)

dup_root = root / "_v4_realdup"
if dup_root.exists():
    shutil.rmtree(dup_root)
dups = []
for i in range(dup):  # each aggregated source needs its own root: meta/data copied (small), videos hard-linked
    for r in reals:
        p = dup_root / f"{r.name}_dup{i}"
        p.mkdir(parents=True)
        shutil.copytree(r / "meta", p / "meta")
        shutil.copytree(r / "data", p / "data")
        shutil.copytree(r / "videos", p / "videos", copy_function=os.link)
        fix_episode_meta_index(p)
        dups.append(p)

out = root / args.out
if out.exists():
    shutil.rmtree(out)
srcs = sims + dups
agg.aggregate_datasets([f"local/{p.name}" for p in srcs], f"local/{args.out}", roots=srcs, aggr_root=out,
                       concatenate_videos=False)
tot = {k: v for k, v in info(out).items() if k.startswith("total")}
print(f"[agg] aggregated {tot}", flush=True)
assert tot["total_frames"] == fs + dup * fr, (tot, fs, dup, fr)
(out / "meta" / "v4_sources.json").write_text(json.dumps(
    {"sims": args.sims, "reals": args.reals, "real_dup": dup, "sim_frames": fs, "real_frames": fr,
     "real_share": dup * fr / (fs + dup * fr)}, indent=1))
shutil.rmtree(dup_root)

# spot check: decode frames of the first / middle / last sim episode and of the first / last real duplicate
from lerobot.datasets import LeRobotDataset  # noqa: E402

ds = LeRobotDataset(f"local/{args.out}", root=out, video_backend="torchcodec")
n_sim = sum(info(p)["total_episodes"] for p in sims)
check = [0, n_sim // 2, n_sim - 1, n_sim, ds.meta.total_episodes - 1]
for ep in check:
    e = ds.meta.episodes[ep]
    for idx in (e["dataset_from_index"], e["dataset_to_index"] - 1):
        it = ds[idx]
        shapes = {k: tuple(v.shape) for k, v in it.items() if k.startswith("observation.images")}
        print(f"[agg] check ep {ep} frame {idx}: ep_idx {int(it['episode_index'])} {shapes} "
              f"mean {[round(float(it[k].mean()), 3) for k in shapes]}", flush=True)
print("[agg] DONE", flush=True)

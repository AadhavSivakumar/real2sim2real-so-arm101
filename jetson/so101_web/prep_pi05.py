#!/usr/bin/env python
"""Recompute true quantile stats (q01/q99, ...) for a recorded dataset so pi0.5 normalizes it correctly.

    python so101_web/prep_pi05.py <dataset_name> [--data-root ~/so101_datasets]

LeRobot keeps quantiles while recording only as a per-episode envelope; pi0.5 normalizes state/action by
q01/q99, so run this after your LAST recording session into the dataset (recording more re-envelopes them).
ACT uses mean/std and is unaffected. Calls the library function directly: the
augment_dataset_quantile_stats CLI always pushes to the Hub.
"""
import argparse
from pathlib import Path

from lerobot.datasets import LeRobotDataset, write_stats
from lerobot.scripts.augment_dataset_quantile_stats import compute_quantile_stats_for_dataset

ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("--data-root", default="~/so101_datasets")
args = ap.parse_args()

root = Path(args.data_root).expanduser() / args.name
ds = LeRobotDataset(f"local/{args.name}", root=root, video_backend="pyav")
stats = compute_quantile_stats_for_dataset(ds, use_sampling=True, skip_images=True)
for k, v in ds.meta.stats.items():
    stats.setdefault(k, v)
write_stats(stats, ds.meta.root)
print(f"{args.name}: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames; quantile stats rewritten")
for k in ("observation.state", "action"):
    print(f"  {k} q01={[round(float(x), 1) for x in stats[k]['q01']]}")
    print(f"  {k} q99={[round(float(x), 1) for x in stats[k]['q99']]}")

"""Kinematics vs camera: where the real gripper closed on the cube (FK of the real readings through units.py) against
the cube position the overhead camera detected in the episode's first frame (replay_real.py detect -> poses.json).

  ~/lr/bin/python grasp_check.py --poses /mnt/secondary/so101_r2s/replay_real_v2/poses.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--poses", required=True)
ap.add_argument("--data", default="/mnt/secondary/so101_r2s/lerobot_datasets/real_cube_cup_v1")
ap.add_argument("--repo", default=os.path.expanduser("~/so101_r2s2r"))
args = ap.parse_args()
sys.path.insert(0, os.path.join(args.repo, "calib"))
import silhouette as S  # noqa: E402  (lerobot_to_sim = the repo's units.py, incl. the joint-zero offsets)

model = S.get_model()
tcp_g = np.array([-0.0088 + 0.015, 0.0, -0.080])  # cube centre in the jaws, gripper link frame (env_cfg TCP_OFFSET)
poses = json.load(open(args.poses))["episodes"]
poses = {int(p.get("episode", i)): p for i, p in enumerate(poses)} if isinstance(poses, list) else {int(k): v for k, v in poses.items()}
rows = []
for pq in sorted(glob.glob(os.path.join(args.data, "data/chunk-000/*.parquet"))):
    df = pd.read_parquet(pq)
    ep = int(df["episode_index"].iloc[0])
    s = np.stack(df["observation.state"].values).astype(float)
    n = len(s)
    i = int(np.argmin(s[n // 5: 4 * n // 5, 5])) + n // 5  # jaw most closed (holding the cube)
    # last frame before the hand lifts: the cube still sits where it was detected
    q0 = S.lerobot_to_sim(s[i])
    T = model.link_transforms(q0)
    g = T["gripper_link"]
    p = g[:3, :3] @ tcp_g + g[:3, 3]
    det = poses[ep]
    cube = np.array(det["cube"]["xy"] if "cube" in det and isinstance(det["cube"], dict) and "xy" in det["cube"] else det["cube_xy"], float)
    d = p[:2] - cube[:2]
    rows.append((ep, i, *p, *cube[:2], *d))
    print(f"ep{ep}: grasp frame {i} gripper {s[i, 5]:.0f} | FK cube centre ({p[0]:.3f}, {p[1]:.3f}, z {p[2]:.3f}) | "
          f"detected ({cube[0]:.3f}, {cube[1]:.3f}) | diff {1000 * d[0]:+.0f}, {1000 * d[1]:+.0f} mm")
R = np.array(rows)
# fit the cube's position in the jaws (gripper frame) instead of assuming the expert's TCP: R_i p + t_i = (cube_i, 15 mm)
A, b = [], []
for pq, (ep, i, *_rest) in zip(sorted(glob.glob(os.path.join(args.data, "data/chunk-000/*.parquet"))), rows):
    s_ = np.stack(pd.read_parquet(pq)["observation.state"].values).astype(float)
    g = model.link_transforms(S.lerobot_to_sim(s_[int(i)]))["gripper_link"]
    det = poses[int(ep)]["cube"]["xy"]
    A.append(g[:3, :3]); b.append(np.array([det[0], det[1], 0.015]) - g[:3, 3])
p_fit, *_ = np.linalg.lstsq(np.concatenate(A), np.concatenate(b), rcond=None)
res = np.concatenate(A) @ p_fit - np.concatenate(b)
res = res.reshape(-1, 3) * 1000
print(f"fitted cube centre in the gripper frame: {np.round(p_fit * 1000, 1)} mm (expert TCP {np.round(tcp_g * 1000, 1)})")
print(f"residual after the fit: RMS {np.sqrt((res ** 2).sum(1).mean()):.1f} mm, per-axis RMS {np.round(np.sqrt((res ** 2).mean(0)), 1)} mm")
d = R[:, -2:] * 1000
print(f"mean diff x {d[:, 0].mean():+.1f} y {d[:, 1].mean():+.1f} mm; RMS {np.sqrt((d ** 2).sum(1).mean()):.1f} mm; "
      f"FK z at grasp {R[:, 4].mean() * 1000:.0f} mm (cube centre on the desk = 15 mm)")

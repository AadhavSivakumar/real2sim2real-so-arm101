"""Which wrist_roll offset maps the real follower's reading onto the URDF/sim joint? (overhead silhouettes)

For sampled overhead frames of the real episodes, render the robot (calib/silhouette.py, URDF meshes) at the real
joint readings + offsets with wrist_roll offsets in {0, +90, -90, 180} deg; align each render to the real frame by
the best whole-arm shift (absorbs the ~3 cm overhead-pose error), then score only the gripper region: silhouette IoU
and printed/servo label agreement (the gripper servo sits on one side of the gripper, so +90 and -90 differ).

  ~/lr/bin/python roll_sign.py --data /mnt/secondary/so101_r2s/lerobot_datasets/real_cube_cup_v1 --repo ~/so101_r2s2r
"""
import argparse
import glob
import json
import os
import sys

import av
import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--repo", default=os.path.expanduser("~/so101_r2s2r"))
ap.add_argument("--every", type=int, default=15, help="frame stride")
ap.add_argument("--shift", type=int, default=36, help="max |dx|,|dy| [px] of the whole-arm alignment")
ap.add_argument("--rolls", default="0,90,-90,180")
ap.add_argument("--off", default="0,-0.2,-4.5,-3.0,0,0", help="LeRobot-unit offsets besides the roll")
ap.add_argument("--out", default=os.path.expanduser("~/r2s_work/roll_sign"))
ap.add_argument("--fine", action="store_true", help="per-frame best roll in --rolls (shift fixed at roll -90): direction test")
args = ap.parse_args()
sys.path.insert(0, os.path.join(args.repo, "calib"))
import silhouette as S  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

os.makedirs(args.out, exist_ok=True)
cal = json.load(open(os.path.join(args.repo, "calib/data/calib.json")))
K = np.array(cal["SCENE_K"], float).reshape(3, 3)
dist = np.array(cal["SCENE_DIST"], float)
R = Rotation.from_quat(cal["SCENE_CAM_QUAT"]).as_matrix()
C = np.array(cal["SCENE_CAM_POS"], float)
model = S.get_model()
grip_links = [ln for ln in model.link_meshes if "gripper" in ln or "jaw" in ln]
print("gripper links:", grip_links, "| all links:", list(model.link_meshes))
rolls = [float(r) for r in args.rolls.split(",")]
off = np.array([float(v) for v in args.off.split(",")])


def mesh(q, links=None):
    T = model.link_transforms(q)
    Vs, Fs, Ls, o = [], [], [], 0
    for ln, (V, F, L) in model.link_meshes.items():
        if links is not None and ln not in links:
            continue
        M = T[ln]
        Vs.append(V @ M[:3, :3].T + M[:3, 3]); Fs.append(F + o); Ls.append(L); o += len(V)
    return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ls)


def shifted(img, dx, dy):
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (img.shape[1], img.shape[0]),
                          flags=cv2.INTER_NEAREST, borderValue=0)


res = {r: dict(arm_iou=[], grip_iou=[], grip_label=[]) for r in rolls}
fine = []  # (episode, frame, real roll reading [deg], best roll offset [deg], best gripper IoU)
examples = []
for pq in sorted(glob.glob(os.path.join(args.data, "data/chunk-000/*.parquet"))):
    df = pd.read_parquet(pq)
    ep = int(df["episode_index"].iloc[0])
    states = np.stack(df["observation.state"].values)
    vid = os.path.join(args.data, f"videos/observation.images.scene/chunk-000/file-{ep:03d}.mp4")
    with av.open(vid) as c:
        frames = [f.to_ndarray(format="bgr24") for i, f in enumerate(c.decode(video=0))]
    for i in range(0, min(len(frames), len(states)), args.every):
        img = frames[i]
        seg = S.segment(img)
        real_robot = seg["robot"] > 0.5
        real_lab = np.where(seg["white"] > 0.5, 1, np.where(seg["dark"] > 0.5, 2, 0)).astype(np.uint8)
        if args.fine:
            v = states[i].astype(float) + off; v[4] -= 90
            arm = S.render(*mesh(S.lerobot_to_sim(v)), R, C, K, dist, (640, 480), labels=True)
            ys, xs = np.nonzero(arm)
            box = (slice(max(ys.min() - 40, 0), ys.max() + 40), slice(max(xs.min() - 40, 0), xs.max() + 40))
            best = max(((((shifted(arm, dx, dy)[box] > 0) & real_robot[box]).sum()
                         / max(((shifted(arm, dx, dy)[box] > 0) | real_robot[box]).sum(), 1), dx, dy)
                        for dy in range(-args.shift, args.shift + 1, 4) for dx in range(-args.shift, args.shift + 1, 4)))
            _, dx, dy = best
            scores = []
            for r in rolls:
                v = states[i].astype(float) + off; v[4] += r
                g = shifted(S.render(*mesh(S.lerobot_to_sim(v), grip_links), R, C, K, dist, (640, 480), labels=True), dx, dy)
                gy, gx = np.nonzero(g)
                if len(gy) < 50:
                    scores.append(-1); continue
                gb = (slice(max(gy.min() - 15, 0), gy.max() + 15), slice(max(gx.min() - 15, 0), gx.max() + 15))
                gm, rm = g[gb] > 0, real_robot[gb]
                scores.append((gm & rm).sum() / max((gm | rm).sum(), 1))
            k = int(np.argmax(scores))
            fine.append((ep, i, float(states[i][4]), rolls[k], float(scores[k])))
            continue
        for r in rolls:
            v = states[i].astype(float) + off
            v[4] += r
            q = S.lerobot_to_sim(v)
            arm = S.render(*mesh(q), R, C, K, dist, (640, 480), labels=True)
            grip = S.render(*mesh(q, grip_links), R, C, K, dist, (640, 480), labels=True)
            if (grip > 0).sum() < 50:
                continue
            ys, xs = np.nonzero(arm)
            box = (slice(max(ys.min() - 40, 0), ys.max() + 40), slice(max(xs.min() - 40, 0), xs.max() + 40))
            best = (-1, 0, 0)
            for dy in range(-args.shift, args.shift + 1, 4):
                for dx in range(-args.shift, args.shift + 1, 4):
                    a = shifted(arm, dx, dy)[box] > 0
                    rr = real_robot[box]
                    iou = (a & rr).sum() / max((a | rr).sum(), 1)
                    if iou > best[0]:
                        best = (iou, dx, dy)
            iou, dx, dy = best
            g = shifted(grip, dx, dy)
            gy, gx = np.nonzero(g)
            gb = (slice(max(gy.min() - 15, 0), gy.max() + 15), slice(max(gx.min() - 15, 0), gx.max() + 15))
            gm, rm = g[gb] > 0, real_robot[gb]
            res[r]["arm_iou"].append(iou)
            res[r]["grip_iou"].append((gm & rm).sum() / max((gm | rm).sum(), 1))
            on = gm & (real_lab[gb] > 0)
            res[r]["grip_label"].append(float((g[gb][on] == real_lab[gb][on]).mean()) if on.sum() > 20 else np.nan)
            if i == 0 and ep in (0, 3, 7):
                vis = img.copy()
                cnt, _ = cv2.findContours((g > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(vis, cnt, -1, (0, 0, 255), 1)
                vis[g == 2] = (0.5 * vis[g == 2] + [0, 127, 255]).astype(np.uint8)  # rendered servo in orange
                ys, xs = np.nonzero(g)
                cy, cx = int(ys.mean()), int(xs.mean())
                crop = vis[max(cy - 70, 0):cy + 70, max(cx - 70, 0):cx + 70]
                crop = cv2.resize(crop, (280, 280), interpolation=cv2.INTER_CUBIC)
                cv2.putText(crop, f"ep{ep} roll{r:+.0f}", (4, 16), 0, 0.5, (0, 255, 255), 1)
                examples.append(((ep, r), crop))
    if args.fine:
        print(f"ep{ep}: {len(fine)} frames so far", flush=True)
        continue
    print(f"ep{ep}: " + "  ".join(f"roll{r:+.0f} arm {np.mean(res[r]['arm_iou']):.3f} grip {np.mean(res[r]['grip_iou']):.3f} "
                                   f"lab {np.nanmean(res[r]['grip_label']):.3f}" for r in rolls), flush=True)

if args.fine:
    F = np.array([f[2:] for f in fine])
    good = F[:, 2] > 0.4
    slope, icpt = np.polyfit(F[good, 0], F[good, 1], 1)
    print(f"[fine] {good.sum()}/{len(F)} frames with gripper IoU > 0.4; real roll readings {F[good, 0].min():.0f}..{F[good, 0].max():.0f} deg")
    print(f"[fine] best offset: median {np.median(F[good, 1]):.1f} deg, IQR {np.percentile(F[good, 1], 25):.0f}..{np.percentile(F[good, 1], 75):.0f}")
    print(f"[fine] best offset vs real reading: slope {slope:+.3f} (0 = same direction, -2 = mirrored), intercept {icpt:.1f}")
    json.dump(dict(frames=fine, slope=slope, intercept=icpt), open(os.path.join(args.out, "fine.json"), "w"))
    sys.exit(0)
summary = {f"{r:+.0f}": {k: float(np.nanmean(v)) for k, v in d.items()} | {"n": len(d["arm_iou"])} for r, d in res.items()}
print(json.dumps(summary, indent=1))
json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
if examples:
    rows = {}
    for (ep, r), crop in examples:
        rows.setdefault(ep, []).append(crop)
    cv2.imwrite(os.path.join(args.out, "examples.jpg"), np.concatenate([np.concatenate(v, 1) for v in rows.values()], 0))

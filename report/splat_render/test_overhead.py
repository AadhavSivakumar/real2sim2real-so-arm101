import json
import sys
import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/mnt/secondary/so101_r2s/report/splat_render/scripts")
import splatcpu as S  # noqa: E402

OUT = "/mnt/secondary/so101_r2s/report/splat_render/work"
PLY = "/mnt/secondary/so101_r2s/assets/splat/background.ply"
POSE = "/mnt/secondary/so101_r2s/assets/splat/splat_pose.json"
cal = json.load(open("/home/asubuntudesktop/TestingGrounds/so101_r2s2r/calib/data/calib.json"))
K = np.asarray(cal["SCENE_K"], float).reshape(3, 3)
DIST = np.asarray(cal["SCENE_DIST"], float)
C = np.asarray(cal["SCENE_CAM_POS"], float)
R_cw = Rotation.from_quat(cal["SCENE_CAM_QUAT"]).as_matrix()
W, H = 640, 480

t0 = time.time()
sp = S.Splat(PLY, POSE, "baked")
print(f"load {time.time() - t0:.1f}s n={sp.n}")
m = sp.means.numpy()
print("world bbox pct 1/99:", np.percentile(m, 1, 0).round(3), np.percentile(m, 99, 0).round(3))
print("world bbox pct 10/90:", np.percentile(m, 10, 0).round(3), np.percentile(m, 90, 0).round(3))
from matplotlib.path import Path as P  # noqa: E402
ins = P(S.DESK_POLY).contains_points(m[:, :2])
print("over desk footprint:", ins.sum(), " z-hist over desk:", np.histogram(m[ins, 2], bins=[-5, -1, -0.5, -0.3, 0, 0.35, 0.6, 1, 5])[0])
print("opacity mean", sp.op.mean().item(), "scale max pct 50/90/99", np.percentile(sp.scale_max.numpy(), [50, 90, 99]))
print("cam C", C, "cam fwd", R_cw[:, 2].round(3), "cam right", R_cw[:, 0].round(3), "cam down", R_cw[:, 1].round(3))

desk = S.Desk("/mnt/secondary/so101_r2s/assets/real2sim/desk_albedo_hr.png")

# distortion map: for each raw (distorted) pixel, its undistorted (pinhole) pixel
uu, vv = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
pts = np.stack([uu.ravel(), vv.ravel()], -1)[:, None, :]
und = cv2.undistortPoints(pts, K, DIST, P=K).reshape(H, W, 2)
mapx, mapy = und[..., 0].astype(np.float32), und[..., 1].astype(np.float32)

for mode in ["baked", "undo", "double"]:
    spm = sp if mode == "baked" else S.Splat(PLY, POSE, mode)
    for use_desk in [False, True]:
        t0 = time.time()
        img, st = S.render(spm, C, R_cw, K, W, H, desk=desk if use_desk else None, return_stats=True)
        dt = time.time() - t0
        bgr = S.to_bgr8(img)
        dist = cv2.remap(bgr, mapx, mapy, cv2.INTER_LINEAR)
        tag = f"{mode}_{'desk' if use_desk else 'nodesk'}"
        cv2.imwrite(f"{OUT}/oh_{tag}.png", dist)
        print(f"{tag}: {dt:.1f}s vis={st['n_vis']} pairs={st['pairs']} work={st['work'] / 1e6:.0f}M "
              f"mean BGR={dist.reshape(-1, 3).mean(0).round(1)}", flush=True)
    if mode == "baked":
        np.save(f"{OUT}/oh_alpha.npy", st["alpha"])
        np.save(f"{OUT}/oh_deskmask.npy", st["desk"])

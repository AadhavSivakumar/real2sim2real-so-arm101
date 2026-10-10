"""Splat (as the sim sees it: background.ply + desk-top quad) from the calibrated overhead camera vs the real frame.

Real frame = episode 0, first overhead frame of real_cube_cup_v1, undistorted with SCENE_K/SCENE_DIST (pinhole K), the
same frame as the REAL panel of brev/splat_check/cmp_ep000.png. Render = pinhole K, 2x supersampled.
"""
import json
import sys

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/mnt/secondary/so101_r2s/report/splat_render/scripts")
import splatcpu as S  # noqa: E402

D = "/mnt/secondary/so101_r2s/report/splat_render"
cal = json.load(open("/home/asubuntudesktop/TestingGrounds/so101_r2s2r/calib/data/calib.json"))
K = np.asarray(cal["SCENE_K"], float).reshape(3, 3)
C = np.asarray(cal["SCENE_CAM_POS"], float)
R_cw = Rotation.from_quat(cal["SCENE_CAM_QUAT"]).as_matrix()  # xyzw, camera-to-world, OpenCV optical axes
W, H, ss = 640, 480, 2
Ks = K.copy()
Ks[:2] *= ss
sp = S.Splat("/mnt/secondary/so101_r2s/assets/splat/background.ply",
             "/mnt/secondary/so101_r2s/assets/splat/splat_pose.json", "baked")
desk = S.Desk("/mnt/secondary/so101_r2s/assets/real2sim/desk_albedo_hr.png")
img = S.render(sp, C, R_cw, Ks, W * ss, H * ss, desk=desk)
rend = S.to_bgr8(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
real = cv2.imread(f"{D}/work/real_ep0_undist.png")
cv2.imwrite(f"{D}/work/overhead_render.png", rend)
blend = ((real.astype(np.uint16) + rend) // 2).astype(np.uint8)


def panel(im, title, sub):
    bar = np.full((56, W, 3), 24, np.uint8)
    cv2.putText(bar, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 190, 190), 1, cv2.LINE_AA)
    return np.concatenate([im, bar], 0)


gap = np.full((H + 56, 6, 3), 24, np.uint8)
out = np.concatenate([
    panel(real, "Real overhead camera", "episode 0, first frame (undistorted)"), gap,
    panel(rend, "3DGS splat, calibrated overhead pose", "background splat + sim desk-top texture"), gap,
    panel(blend, "50/50 blend", "alignment check"),
], 1)
cv2.imwrite(f"{D}/overhead_compare.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])
print("wrote", f"{D}/overhead_compare.jpg", out.shape)

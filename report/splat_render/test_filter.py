import sys, json, os
import cv2, numpy as np, torch
sys.path.insert(0, "/mnt/secondary/so101_r2s/report/splat_render/scripts")
import splatcpu as S
os.environ.setdefault("ORBIT_PATH", "{}")
import render_orbit as RO
from matplotlib.path import Path as P
sp = S.Splat("/mnt/secondary/so101_r2s/assets/splat/background.ply", "/mnt/secondary/so101_r2s/assets/splat/splat_pose.json", "baked")
desk = S.Desk("/mnt/secondary/so101_r2s/assets/real2sim/desk_albedo_hr.png")
m = sp.means.numpy()
over = P(S.DESK_POLY).contains_points(m[:, :2])
hi = over & (m[:, 2] > 0.35) & (m[:, 2] < 0.9)
print("over-desk high gaussians:", hi.sum(), "opacity mean", sp.op.numpy()[hi].mean(), "vs all", sp.op.numpy().mean())
sp_f = sp.subset(torch.from_numpy(~hi))
only = sp.subset(torch.from_numpy(hi))
W, H = 640, 360
f = 0.5 * W / np.tan(np.radians(55) / 2)
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
rows = []
for i in [0, 235, 300]:
    C, R = RO.camera(i)
    a = S.render(sp, C, R, K, W, H, desk=desk, near=0.02)
    b = S.render(sp, C, R, K, W, H, desk=desk, near=0.30)
    c = S.render(sp_f, C, R, K, W, H, desk=desk, near=0.30)
    d = S.render(only, C, R, K, W, H, desk=desk, near=0.02)
    rows.append(np.concatenate([S.to_bgr8(x) for x in (a, b, c, d)], 1))
cv2.imwrite("work/filter_test.jpg", cv2.resize(np.concatenate(rows, 0), None, fx=0.6, fy=0.6))

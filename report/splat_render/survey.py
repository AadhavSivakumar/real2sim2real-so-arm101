"""Low-res survey of orbit viewpoints: azimuth x elevation grid around the desk centre."""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/mnt/secondary/so101_r2s/report/splat_render/scripts")
import splatcpu as S  # noqa: E402

OUT = "/mnt/secondary/so101_r2s/report/splat_render/work"
sp = S.Splat("/mnt/secondary/so101_r2s/assets/splat/background.ply",
             "/mnt/secondary/so101_r2s/assets/splat/splat_pose.json", "baked")
desk = S.Desk("/mnt/secondary/so101_r2s/assets/real2sim/desk_albedo_hr.png")
W, H = 384, 216
hfov = np.radians(float(sys.argv[1]) if len(sys.argv) > 1 else 60)
f = 0.5 * W / np.tan(hfov / 2)
K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
target = np.array([float(x) for x in (sys.argv[5].split(",") if len(sys.argv) > 5 else ["0.13", "0.08", "0.0"])])
r = float(sys.argv[2]) if len(sys.argv) > 2 else 0.95
els = [float(e) for e in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["35", "55"])]
azs = [int(a) for a in (sys.argv[4].split(",") if len(sys.argv) > 4 else range(0, 360, 30))]
rows = []
t0 = time.time()
for el in els:
    row = []
    for az in azs:
        e, a = np.radians(el), np.radians(az)
        C = target + r * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        img = S.render(sp, C, S.look_at(C, target), K, W, H, desk=desk)
        im = S.to_bgr8(img).copy()
        cv2.putText(im, f"az{az} el{int(el)}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        row.append(im)
    rows.append(row)
print(f"{len(els) * len(azs)} frames in {time.time() - t0:.1f}s")
for i, row in enumerate(rows):
    grid = np.concatenate([np.concatenate(row[:len(row)//2], 1), np.concatenate(row[len(row)//2:], 1)], 0)
    cv2.imwrite(f"{OUT}/survey_el{int(els[i])}_r{r}.jpg", grid)

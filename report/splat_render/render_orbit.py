"""Render the eased orbit arc around the desk (CPU). Frames -> OUT_DIR/f_%04d.png.

  render_orbit.py OUT_DIR W H [i0 i1 step]   (frame indices of the 30 fps path; default all)
"""
import json
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/mnt/secondary/so101_r2s/report/splat_render/scripts")
import splatcpu as S  # noqa: E402

PATH = dict(fps=30, seconds=11.0, az=(78.0, 198.0), el=(48.0, 44.0), r=0.85, r_dip=0.04,
            target=(0.15, -0.01, 0.0), hfov=55.0, near_fade=(0.22, 0.38), cull_over_desk_z=(0.35, 0.9), ss=1.5)
import os as _os  # noqa: E402
PATH.update(json.loads(_os.environ.get("ORBIT_PATH", "{}")))


def camera(i, P=PATH):
    n = int(round(P["fps"] * P["seconds"]))
    p = i / (n - 1)
    e = 0.5 - 0.5 * np.cos(np.pi * p)  # ease in/out
    az = np.radians(P["az"][0] + (P["az"][1] - P["az"][0]) * e)
    el = np.radians(P["el"][0] + (P["el"][1] - P["el"][0]) * e)
    r = P["r"] - P["r_dip"] * np.sin(np.pi * e)
    tgt = np.asarray(P["target"], float)
    C = tgt + r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return C, S.look_at(C, tgt)


if __name__ == "__main__":
    out, W, H = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    n = int(round(PATH["fps"] * PATH["seconds"]))
    i0, i1, step = (int(x) for x in sys.argv[4:7]) if len(sys.argv) > 6 else (0, n, 1)
    import os
    os.makedirs(out, exist_ok=True)
    json.dump(PATH, open(os.path.join(out, "path.json"), "w"), indent=1)
    sp = S.Splat("/mnt/secondary/so101_r2s/assets/splat/background.ply",
                 "/mnt/secondary/so101_r2s/assets/splat/splat_pose.json", "baked")
    if PATH.get("cull_over_desk_z"):  # haze/floaters hanging over the desk above the band export_splat.py cropped
        import torch
        from matplotlib.path import Path as Poly
        m = sp.means.numpy()
        z0, z1 = PATH["cull_over_desk_z"]
        hi = Poly(S.DESK_POLY).contains_points(m[:, :2]) & (m[:, 2] > z0) & (m[:, 2] < z1)
        sp = sp.subset(torch.from_numpy(~hi))
        print(f"culled {hi.sum()} over-desk floaters; {sp.n} Gaussians left", flush=True)
    desk = S.Desk("/mnt/secondary/so101_r2s/assets/real2sim/desk_albedo_hr.png")
    ss = float(PATH.get("ss", 1.0))
    Wr, Hr = int(round(W * ss)), int(round(H * ss))
    f = 0.5 * Wr / np.tan(np.radians(PATH["hfov"]) / 2)
    K = np.array([[f, 0, Wr / 2], [0, f, Hr / 2], [0, 0, 1]])
    ts = []
    for i in range(i0, min(i1, n), step):
        fn = os.path.join(out, f"f_{i:04d}.png")
        if os.path.exists(fn):
            continue
        t0 = time.time()
        C, R = camera(i)
        img = S.render(sp, C, R, K, Wr, Hr, desk=desk, near=0.05, near_fade=PATH.get("near_fade"))
        if (Wr, Hr) != (W, H):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(fn, S.to_bgr8(img))
        ts.append(time.time() - t0)
        print(f"frame {i} {ts[-1]:.2f}s", flush=True)
    if ts:
        print(f"done {len(ts)} frames, mean {np.mean(ts):.2f}s/frame, total {np.sum(ts):.0f}s", flush=True)

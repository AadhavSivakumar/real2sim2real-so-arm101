"""Snapshot dirs (calib/capture.py layout: scene.jpg + state.json) for solve.py scene_robot from still moments of the
real episodes, with the joint-zero offsets already added to the readings (so solve with --joint_offsets off/check).

Per episode: the stillest frame while the jaw closes on the cube (grasp), while it opens over the cup (release), and
the first frame (rest; only from a few episodes). Also copies the sweep snapshots (snap_sw*) with the same offsets.

  ~/lr/bin/python make_snaps.py --data .../real_cube_cup_v1 --calib_data ~/so101_r2s2r/calib/data --out ~/r2s_work/snaps
"""
import argparse
import glob
import json
import os
import shutil

import av
import cv2
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--calib_data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--off", default="0,-0.2,-4.5,-3.0,-90,0", help="LeRobot-unit offsets added to the readings")
ap.add_argument("--max_speed", type=float, default=8.0, help="still = fastest arm joint below this [deg/s] over +-2 frames")
ap.add_argument("--rest_eps", default="0,4,8")
args = ap.parse_args()
off = np.array([float(v) for v in args.off.split(",")])
names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
os.makedirs(args.out, exist_ok=True)
shutil.copy(os.path.join(args.calib_data, "intrinsics_scene.json"), args.out)
made = []


def write(name, img, reading, source):
    d = os.path.join(args.out, f"snap_{name}")
    os.makedirs(d, exist_ok=True)
    cv2.imwrite(os.path.join(d, "scene.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    v = np.asarray(reading, float) + off
    json.dump({"follower": {f"{j}.pos": float(x) for j, x in zip(names, v)}, "source": source,
               "offsets_added": off.tolist()}, open(os.path.join(d, "state.json"), "w"), indent=1)
    made.append(name)


for pq in sorted(glob.glob(os.path.join(args.data, "data/chunk-000/*.parquet"))):
    df = pd.read_parquet(pq)
    ep = int(df["episode_index"].iloc[0])
    s = np.stack(df["observation.state"].values).astype(float)
    with av.open(os.path.join(args.data, f"videos/observation.images.scene/chunk-000/file-{ep:03d}.mp4")) as c:
        frames = [f.to_ndarray(format="bgr24") for f in c.decode(video=0)]
    n = min(len(frames), len(s))
    sp = np.zeros(n)
    for i in range(n):
        lo, hi = max(i - 2, 0), min(i + 2, n - 1)
        sp[i] = np.abs(s[hi, :5] - s[lo, :5]).max() * 30.0 / max(hi - lo, 1)
    g = s[:n, 5]
    dg = np.gradient(g)
    gmin = int(np.argmin(g[n // 5: 4 * n // 5])) + n // 5  # jaw most closed on the cube (mid-episode)
    picks = {}
    close = [i for i in range(max(gmin - 45, 0), gmin + 1)]  # the second before full closure
    i = min(close, key=lambda k: sp[k])
    if sp[i] < args.max_speed:
        picks["grasp"] = i
    opening = [i for i in range(gmin, n) if dg[i] > 0.3]  # jaw opening after the grasp = release over the cup
    if opening:
        win = range(opening[0], min(opening[0] + 45, n))
        i = min(win, key=lambda k: sp[k])
        if sp[i] < args.max_speed:
            picks["release"] = i
    if str(ep) in args.rest_eps.split(","):
        picks["rest"] = 0
    for tag, i in picks.items():
        write(f"ep{ep}_{tag}", frames[i], s[i], f"real_cube_cup_v1 ep{ep} frame {i} ({tag}, {sp[i]:.1f} deg/s)")
    print(f"ep{ep}: " + ", ".join(f"{t} f{i} ({sp[i]:.1f} deg/s, grip {g[i]:.0f})" for t, i in picks.items()), flush=True)

for d in sorted(glob.glob(os.path.join(args.calib_data, "snap_sw*"))):
    st = json.load(open(os.path.join(d, "state.json")))
    f = st["follower"]
    v = [f[j] if j in f else f[f"{j}.pos"] for j in names] if isinstance(f, dict) else list(f)[:6]
    write(os.path.basename(d)[5:], cv2.imread(os.path.join(d, "scene.jpg")), v, st.get("source", d))
print("snapshots:", " ".join(made))

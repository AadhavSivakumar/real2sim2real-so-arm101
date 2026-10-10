"""Copy a real LeRobot v3 dataset with every camera video undistorted to its own K (calib/data/calib.json *_K, *_DIST).

The sim renders undistorted pinholes with exactly those K (env_cfg.RENDER_LENS_DISTORTION is off), so real frames have
to look the same before sim and real data are mixed for co-training -- and the robot side undistorts the same way
(cv2.undistort(img, K, DIST)) when a sim/co-trained policy runs. Parquet data and metadata are copied unchanged
(same resolution / fps / frame count); image stats in meta/stats.json are left as recorded.

  ~/lr/bin/python scripts/undistort_real_dataset.py --src .../real_cube_cup_v1 --dst .../real_cube_cup_v1_undist
"""
import argparse
import json
import shutil
from pathlib import Path

import av
import cv2
import numpy as np

CAMS = {"observation.images.wrist": "WRIST", "observation.images.scene": "SCENE"}

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--calib", default=str(Path(__file__).resolve().parents[1] / "calib" / "data" / "calib.json"))
ap.add_argument("--crf", type=int, default=18)
args = ap.parse_args()
src, dst = Path(args.src), Path(args.dst)
cal = json.loads(Path(args.calib).read_text())
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst, ignore=shutil.ignore_patterns("videos"))
info = json.loads((src / "meta" / "info.json").read_text())

for key, name in CAMS.items():
    K = np.asarray(cal[f"{name}_K"], float).reshape(3, 3)
    D = np.asarray(cal[f"{name}_DIST"], float)
    vi = info["features"][key]["info"]
    w, h, fps = vi["video.width"], vi["video.height"], vi["video.fps"]
    mx, my = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_32FC1)
    for f in sorted((src / "videos" / key).rglob("*.mp4")):
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with av.open(str(f)) as cin, av.open(str(out), "w") as cout:
            st = cout.add_stream("libx264", rate=fps)
            st.width, st.height, st.pix_fmt = w, h, "yuv420p"
            st.options = {"crf": str(args.crf), "g": str(vi.get("video.g", 2)), "preset": "veryfast"}
            for fr in cin.decode(video=0):
                img = cv2.remap(fr.to_ndarray(format="bgr24"), mx, my, cv2.INTER_LINEAR)
                for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                    cout.mux(pkt)
                n += 1
            for pkt in st.encode():
                cout.mux(pkt)
        print(f"[undist] {key} {f.name}: {n} frames", flush=True)
print(f"[undist] wrote {dst}")

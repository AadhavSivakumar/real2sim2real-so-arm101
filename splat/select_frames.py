"""Pick the sharpest frame from every window of N consecutive video frames.

Usage: python select_frames.py <frames_dir> <out_dir> <prefix> <window>
Sharpness = variance of the Laplacian on a downscaled grayscale copy.
"""
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

src, dst, prefix, window = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
dst.mkdir(parents=True, exist_ok=True)

frames = sorted(src.glob("*.jpg"))
scores = []
for f in frames:
    g = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)
    scores.append(cv2.Laplacian(g, cv2.CV_64F).var())
scores = np.array(scores)

kept = []
for start in range(0, len(frames), window):
    best = start + int(np.argmax(scores[start : start + window]))
    kept.append(best)
    shutil.copy(frames[best], dst / f"{prefix}_{best:04d}.jpg")

print(f"{src.name}: {len(frames)} frames -> kept {len(kept)} | "
      f"sharpness all: median {np.median(scores):.1f}, min {scores.min():.1f} | "
      f"kept: median {np.median(scores[kept]):.1f}, min {scores[kept].min():.1f}")

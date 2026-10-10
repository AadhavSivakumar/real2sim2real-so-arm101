"""Structure-from-Motion with COLMAP (via pycolmap): recover camera poses + a sparse point cloud.

Usage: python run_sfm.py <scene_dir>   (expects <scene_dir>/images/v<id>_<frame>.jpg)
Writes <scene_dir>/sparse/0/{cameras,images,points3D}.bin, the layout gsplat's COLMAP loader reads.
"""
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import pycolmap

scene = Path(sys.argv[1])
images, work = scene / "images", scene / "colmap_work"
db = work / "database.db"
shutil.rmtree(work, ignore_errors=True)
work.mkdir(parents=True)

t0 = time.time()

# 1) SIFT features. Each video gets its own camera (shared intrinsics within a clip),
#    using the OPENCV model (focal, principal point, 2 radial + 2 tangential distortion).
by_video = defaultdict(list)
for f in sorted(images.glob("*.jpg")):
    by_video[f.name.split("_")[0]].append(f.name)

extraction = pycolmap.FeatureExtractionOptions()
extraction.sift.max_num_features = 8192
extraction.num_threads = 6  # CPU SIFT is RAM-hungry at 1080x1920; cap concurrency
for vid, names in by_video.items():
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "OPENCV"
    pycolmap.extract_features(db, images, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=reader, extraction_options=extraction)
    print(f"[{time.time()-t0:6.0f}s] extracted features for {vid}: {len(names)} images", flush=True)

# 2) Match every image against every other image, then geometrically verify each pair (RANSAC).
pycolmap.match_exhaustive(db)
print(f"[{time.time()-t0:6.0f}s] exhaustive matching done", flush=True)

# 3) Incremental mapping: start from a strong image pair, triangulate, add images one by one,
#    bundle-adjust as it grows.
recs = pycolmap.incremental_mapping(db, images, work / "sparse")
print(f"[{time.time()-t0:6.0f}s] mapping done: {len(recs)} model(s)", flush=True)

best_id = max(recs, key=lambda k: recs[k].num_reg_images())
for k, r in recs.items():
    print(f"  model {k}: {r.num_reg_images()} images, {r.num_points3D()} points, "
          f"mean reproj err {r.compute_mean_reprojection_error():.3f}px")
rec = recs[best_id]
for cam_id, cam in rec.cameras.items():
    print(f"  camera {cam_id}: {cam.model.name} {cam.width}x{cam.height} params={[round(p, 4) for p in cam.params]}")

out = scene / "sparse" / "0"
shutil.rmtree(scene / "sparse", ignore_errors=True)
out.mkdir(parents=True)
rec.write(out)
missing = sorted(set(f.name for f in images.glob("*.jpg")) - {im.name for im in rec.images.values()})
print(f"kept model {best_id} -> {out}; unregistered images: {len(missing)} {missing[:10]}")

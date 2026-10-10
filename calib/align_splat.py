"""Similarity transform from a splat's SfM frame into the desk frame, using the overhead Astra as the anchor.

The overhead camera is fixed and calibrated (calib.json SCENE_K, SCENE_CAM_POS/QUAT; xyzw, ROS camera axes). Its
undistorted frames (undistort_real_dataset.py output) are matched to every registered video frame with DISK + LightGlue
(SIFT fails on the dark, low-res overhead view). Each pair is verified with an essential matrix (both cameras are
calibrated); verified matches that land on a triangulated SfM point give 2D-3D pairs, and PnP over all of them gives the
overhead pose in SfM units. Rotation and translation follow from the known overhead pose; the scale from the desk: an
inlier's overhead ray hits the desk plane z = 0 at a known distance from the camera, s times its SfM distance (median
over the inliers that hit the desk near the robot). Writes SPLAT_POS / SPLAT_QUAT (xyzw) / SPLAT_SCALE (desk = s R x + t)
plus diagnostics, and an overlay of the SfM points projected into the overhead view (out with _overlay.jpg).

Coarse: the matches cluster near the robot base, so scale and tilt are weak; refine with refine_splat_align.py once the
splat is trained, then export_splat.py.

  ~/kv/bin/python calib/align_splat.py --sfm SCENE/sparse/0 --images SCENE/images --overhead 'overhead/*.png' \
      --calib calib/data/calib.json --out SCENE/align.json      (venv with torch, kornia, pycolmap, opencv, scipy)
"""
import argparse
import glob
import json

import cv2
import kornia.feature as KF
import numpy as np
import pycolmap
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("--sfm", required=True)
ap.add_argument("--images", required=True)
ap.add_argument("--overhead", required=True, help="glob of undistorted overhead frames (calib SCENE_K)")
ap.add_argument("--calib", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n_overhead", type=int, default=3, help="overhead frames used (the camera never moves)")
ap.add_argument("--max_side", type=int, default=1024, help="video frames are matched at this size")
ap.add_argument("--snap_px", type=float, default=4.0, help="match -> SfM keypoint distance (full-res px)")
ap.add_argument("--desk_center", type=float, nargs=2, default=(0.25, 0.0))
ap.add_argument("--desk_radius", type=float, default=0.45)
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"

rec = pycolmap.Reconstruction(args.sfm)
cal = json.loads(open(args.calib).read())
K = np.asarray(cal["SCENE_K"], float).reshape(3, 3)
R_rc = Rotation.from_quat(cal["SCENE_CAM_QUAT"]).as_matrix()  # overhead camera -> desk (scipy quats are xyzw)
C_r = np.asarray(cal["SCENE_CAM_POS"], float)
disk = KF.DISK.from_pretrained("depth").to(dev).eval()
lg = KF.LightGlueMatcher("disk").to(dev).eval()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


@torch.inference_mode()
def feats(img, n=4096):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])  # the overhead view is dark; equalise both sides the same way
    rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)[None].to(dev)
    f = disk(t, n=n, pad_if_not_divisible=True)[0]
    kp = f.keypoints
    laf = KF.laf_from_center_scale_ori(kp[None], torch.ones(1, len(kp), 1, 1, device=dev))
    return kp, f.descriptors, laf, img.shape[:2]


@torch.inference_mode()
def match(a, b):
    _, idx = lg(a[1], b[1], a[2], b[2], hw1=torch.tensor(a[3], device=dev), hw2=torch.tensor(b[3], device=dev))
    idx = idx.cpu().numpy()
    return a[0][idx[:, 0]].cpu().numpy(), b[0][idx[:, 1]].cpu().numpy()


ov_files = sorted(glob.glob(args.overhead))[: args.n_overhead]
ov = [feats(cv2.imread(f)) for f in ov_files]
x2d, X3d, per_img = [], [], {}
for im in rec.images.values():
    cam = rec.cameras[im.camera_id]
    fx, fy, cx, cy = cam.params[:4]
    Kc = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    dist = np.array(cam.params[4:8]) if cam.model.name == "OPENCV" else np.zeros(4)
    img = cv2.imread(f"{args.images}/{im.name}")
    sc = min(1.0, args.max_side / max(img.shape[:2]))
    fi = feats(cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA))
    pts = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
    tree = cKDTree(np.array([p[0] for p in pts]))
    for o in ov:
        po, pi = match(o, fi)
        if len(po) < 15:
            continue
        pi = pi / sc
        no = cv2.undistortPoints(po.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        ni = cv2.undistortPoints(pi.reshape(-1, 1, 2), Kc, dist).reshape(-1, 2)
        E, m = cv2.findEssentialMat(no, ni, np.eye(3), method=cv2.RANSAC, prob=0.999, threshold=2.0 / fx)
        if E is None:
            continue
        m = m.ravel().astype(bool)
        if m.sum() < 15:
            continue
        d, j = tree.query(pi[m], distance_upper_bound=args.snap_px)
        ok = np.isfinite(d)
        per_img[im.name] = per_img.get(im.name, 0) + int(ok.sum())
        x2d += list(po[m][ok])
        X3d += [rec.points3D[pts[k][1]].xyz for k in j[ok]]
x2d, X3d = np.array(x2d, float), np.array(X3d, float)
top = sorted(per_img.items(), key=lambda kv: -kv[1])[:5]
print(f"[align] {len(x2d)} verified overhead<->SfM correspondences from {len(per_img)} frames (top {top})", flush=True)
okp, rvec, tvec, inl = cv2.solvePnPRansac(X3d, x2d, K, None, reprojectionError=4.0, iterationsCount=20000,
                                          confidence=0.9999, flags=cv2.SOLVEPNP_SQPNP)
inl = inl.ravel()
rvec, tvec = cv2.solvePnPRefineLM(X3d[inl], x2d[inl], K, None, rvec, tvec)
R_cs = cv2.Rodrigues(rvec)[0]
C_s = (-R_cs.T @ tvec).ravel()
err = np.linalg.norm(cv2.projectPoints(X3d[inl], rvec, tvec, K, None)[0].reshape(-1, 2) - x2d[inl], axis=1)
print(f"[align] PnP inliers {len(inl)}/{len(x2d)}, reproj median {np.median(err):.2f} px", flush=True)

R = R_rc @ R_cs  # sfm -> desk rotation (R_cs: sfm -> camera, R_rc: camera -> desk)
rays_r = np.linalg.solve(K, np.c_[x2d[inl], np.ones(len(inl))].T).T @ R_rc.T
lam = -C_r[2] / rays_r[:, 2]
hit = C_r + lam[:, None] * rays_r
on_desk = (lam > 0) & (np.linalg.norm(hit[:, :2] - np.asarray(args.desk_center), axis=1) < args.desk_radius)
ratio = np.linalg.norm(hit - C_r, axis=1)[on_desk] / np.linalg.norm(X3d[inl] - C_s, axis=1)[on_desk]
s = float(np.median(ratio))
t = C_r - s * R @ C_s
z = (s * X3d[inl] @ R.T + t)[:, 2]
print(f"[align] scale {s:.5f} m/unit from {on_desk.sum()} desk inliers (ratio IQR {np.percentile(ratio, 25):.5f}.."
      f"{np.percentile(ratio, 75):.5f}); inlier z: median {1000 * np.median(z):+.1f} mm, "
      f"IQR {1000 * np.percentile(z, 25):+.1f}..{1000 * np.percentile(z, 75):+.1f} mm", flush=True)

X_all = np.array([p.xyz for p in rec.points3D.values()])
cen = np.array([-im.cam_from_world().rotation.matrix().T @ im.cam_from_world().translation for im in rec.images.values()])


def plane(Q):
    c = Q.mean(0)
    n = np.linalg.svd(Q - c)[2][2]
    return n * np.sign(n[2]), c


# diagnostics only: the desk inliers cover too small a patch to level by, and the plane of all SfM points near z = 0 is
# polluted by cables, papers and the robot
P = X_all * s @ R.T + t
Cd = s * cen @ R.T + t
Pi = X3d[inl][on_desk] * s @ R.T + t
near = P[(np.abs(P[:, 2]) < 0.02) & (np.linalg.norm(P[:, :2] - np.asarray(args.desk_center), axis=1) < args.desk_radius)]
tilt = np.degrees(np.arccos(plane(near)[0][2]))
print(f"[align] desk inliers span x {Pi[:, 0].min():.2f}..{Pi[:, 0].max():.2f} y {Pi[:, 1].min():.2f}..{Pi[:, 1].max():.2f} m; "
      f"{len(near)} SfM points within 2 cm of the desk, plane tilt {tilt:.2f} deg; video cameras z "
      f"{Cd[:, 2].min():.2f}..{Cd[:, 2].max():.2f} m (median {np.median(Cd[:, 2]):.2f})", flush=True)

out = dict(SPLAT_POS=[float(v) for v in t], SPLAT_QUAT=[float(v) for v in Rotation.from_matrix(R).as_quat()],
           SPLAT_SCALE=s, pnp_inliers=int(len(inl)), pnp_reproj_median_px=float(np.median(err)),
           video_cam_z_median=float(np.median(Cd[:, 2])), desk_points_tilt_deg=float(tilt), overhead_frames=ov_files)
json.dump(out, open(args.out, "w"), indent=1)

# overlay: SfM points projected into the overhead view (black background) next to the brightened real frame
cols = np.array([p.color for p in rec.points3D.values()])
pc = (P - C_r) @ R_rc
f = pc[:, 2] > 0.05
uv = (pc[f] / pc[f, 2:3]) @ K.T
order = np.argsort(-pc[f, 2])
dots = np.zeros((480, 640, 3), np.uint8)
for (u, v), col in zip(uv[order, :2].astype(int), cols[f][order][:, ::-1]):
    if 0 <= u < 640 and 0 <= v < 480:
        cv2.circle(dots, (int(u), int(v)), 2, tuple(int(x) for x in col), -1)
base = np.clip(cv2.imread(ov_files[0]).astype(float) * 2.5, 0, 255).astype(np.uint8)
cv2.imwrite(args.out.replace(".json", "_overlay.jpg"), np.concatenate([base, dots], 1))
print(f"[align] wrote {args.out}", flush=True)

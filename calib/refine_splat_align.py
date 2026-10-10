"""Refine align_splat.py's SfM -> desk transform with the trained splat itself, densely over the whole view.

align_splat.py anchors on the few video frames that match the overhead view (all near the robot base). Here the splat is
rendered (RGB + depth, the overhead camera's own boom culled) from the calibrated overhead pose under the current
transform, so render and real overhead frames differ only by the transform error: DISK + LightGlue matches spread over
the whole view, the render depth turns matched render pixels into SfM points, and PnP gives rotation + translation.
The overhead camera cannot see scale; it comes from the desk: the dominant mode of the rescale factors (about the camera
centre) that put the opaque, flat splat Gaussians seen on the desk top onto z = 0. (Sparse SfM points and the rendered
desk depth both lock onto other surfaces: the glossy wood is reconstructed partly below its surface.) Checks on the
result: cube top at z = 30 mm, cup footprint 75 mm. Iterates; writes the refined transform plus SPLAT_COLOR_GAIN (desk
colour, overhead camera vs splat) over --align (the input is kept as *_coarse.json) and OUT_IMG: real | render | blend.

  ~/gs/bin/python calib/refine_splat_align.py --ckpt RESULTS/ckpts/ckpt_29999_rank0.pt --scene SCENE \
      --gsplat_examples ~/splat/src/gsplat/examples --align SCENE/align.json --overhead 'overhead/*.png' \
      --calib calib/data/calib.json --out_img SCENE/refine.jpg --iters 6
"""
import argparse
import glob
import json
import shutil
import sys

import cv2
import kornia.feature as KF
import numpy as np
import torch
from gsplat import rasterization
from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--scene", required=True)
ap.add_argument("--gsplat_examples", required=True)
ap.add_argument("--align", required=True)
ap.add_argument("--overhead", required=True)
ap.add_argument("--calib", required=True)
ap.add_argument("--out_img", required=True)
ap.add_argument("--iters", type=int, default=3)
ap.add_argument("--n_overhead", type=int, default=10)
ap.add_argument("--desk_center", type=float, nargs=2, default=(0.25, 0.0))
ap.add_argument("--desk_radius", type=float, default=0.5)
ap.add_argument("--cull_depth", type=float, default=0.35,
                help="drop Gaussians this close to the overhead camera inside its view (its own mount)")
args = ap.parse_args()
dev = "cuda"
sys.path.insert(0, args.gsplat_examples)
from datasets.colmap import Parser  # noqa: E402  (same normalisation as training)

T = Parser(args.scene, factor=1, normalize=True, test_every=8).transform
A, b = T[:3, :3], T[:3, 3]
k = np.cbrt(np.linalg.det(A))
Q = A / k
cal = json.loads(open(args.calib).read())
K = np.asarray(cal["SCENE_K"], float).reshape(3, 3)
R_rc = Rotation.from_quat(cal["SCENE_CAM_QUAT"]).as_matrix()
C_r = np.asarray(cal["SCENE_CAM_POS"], float)
sp = torch.load(args.ckpt, map_location=dev, weights_only=False)["splats"]
means, quats = sp["means"], sp["quats"]
scales, opac = torch.exp(sp["scales"]), torch.sigmoid(sp["opacities"])
colors = torch.cat([sp["sh0"], sp["shN"]], 1)
sh_deg = int(np.sqrt(colors.shape[1])) - 1
disk = KF.DISK.from_pretrained("depth").to(dev).eval()
lg = KF.LightGlueMatcher("disk").to(dev).eval()
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
ov_files = sorted(glob.glob(args.overhead))[: args.n_overhead]
ovs = [cv2.imread(f) for f in ov_files]
H, W = ovs[0].shape[:2]


@torch.inference_mode()
def feats(img, n=4096):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = clahe.apply(lab[..., 0])
    rgb = cv2.cvtColor(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2RGB)
    f = disk(torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)[None].to(dev), n=n, pad_if_not_divisible=True)[0]
    laf = KF.laf_from_center_scale_ori(f.keypoints[None], torch.ones(1, len(f.keypoints), 1, 1, device=dev))
    return f.keypoints, f.descriptors, laf, img.shape[:2]


@torch.inference_mode()
def match(a, b_):
    _, idx = lg(a[1], b_[1], a[2], b_[2], hw1=torch.tensor(a[3], device=dev), hw2=torch.tensor(b_[3], device=dev))
    idx = idx.cpu().numpy()
    return a[0][idx[:, 0]].cpu().numpy(), b_[0][idx[:, 1]].cpu().numpy()


def norm_to_desk(al):
    """(s2, R2, t2): desk = s2 R2 x_ply + t2, from the SfM -> desk transform and gsplat's normalisation."""
    s, R, t = al["SPLAT_SCALE"], Rotation.from_quat(al["SPLAT_QUAT"]).as_matrix(), np.asarray(al["SPLAT_POS"])
    s2, R2 = s / k, R @ Q.T
    return s2, R2, t - s2 * R2 @ b


def near_overhead(xyz_desk, depth=None):
    """Gaussians in the overhead camera's view closer than --cull_depth: the camera's own boom/mount, which the real
    camera never sees but a virtual camera a few cm off sits behind."""
    pc = (xyz_desk - C_r) @ R_rc
    uv = pc[:, :2] / np.maximum(pc[:, 2:3], 1e-6) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    inside = (uv[:, 0] > -80) & (uv[:, 0] < W + 80) & (uv[:, 1] > -80) & (uv[:, 1] < H + 80)
    return (pc[:, 2] > -0.05) & (pc[:, 2] < (depth or args.cull_depth)) & inside


@torch.inference_mode()
def render(al):
    s2, R2, t2 = norm_to_desk(al)
    keep = torch.tensor(~near_overhead(means.cpu().numpy() @ (s2 * R2).T + t2), device=dev)
    vm = np.eye(4)
    vm[:3, :3] = R_rc.T @ R2
    vm[:3, 3] = R_rc.T @ (t2 - C_r) / s2  # camera frame scaled by 1/s2: same image, depth in PLY units
    out, alpha, _ = rasterization(means[keep], quats[keep], scales[keep], opac[keep], colors[keep],
                                  torch.tensor(vm, dtype=torch.float32, device=dev)[None],
                                  torch.tensor(K, dtype=torch.float32, device=dev)[None], W, H, sh_degree=sh_deg,
                                  render_mode="RGB+ED", near_plane=0.01)
    rgb = (out[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[..., ::-1].copy()
    return rgb, out[0, ..., 3].cpu().numpy() * s2, alpha[0, ..., 0].cpu().numpy()


G_MEANS = means.cpu().numpy()
_sc = scales.cpu().numpy()
G_OK = (opac.cpu().numpy() > 0.5) & (_sc.min(1) < 0.3 * _sc.max(1))  # opaque, flat (surface) Gaussians
al = json.loads(open(args.align).read())
shutil.copy(args.align, args.align.replace(".json", "_coarse.json"))
f_ov = [feats(o) for o in ovs]
for it in range(args.iters):
    s, R, t = al["SPLAT_SCALE"], Rotation.from_quat(al["SPLAT_QUAT"]).as_matrix(), np.asarray(al["SPLAT_POS"])
    rgb, depth, alpha = render(al)
    fr = feats(rgb)
    x2d, X3d, z = [], [], []
    for fo in f_ov:
        pr, po = match(fr, fo)
        if len(pr) < 15:
            continue
        # same viewpoint up to the transform error: keep matches that move with the bulk (PnP RANSAC does the rest)
        dd = po - pr
        m = np.linalg.norm(dd - np.median(dd, 0), axis=1) < 20.0
        u, v = pr[m, 0].astype(int).clip(0, W - 1), pr[m, 1].astype(int).clip(0, H - 1)
        ok = (alpha[v, u] > 0.9) & (depth[v, u] > 0.05)
        Xd = C_r + (np.c_[pr[m][ok], np.ones(ok.sum())] @ np.linalg.inv(K).T * depth[v, u][ok, None]) @ R_rc.T
        x2d += list(po[m][ok])
        X3d += list(((Xd - t) / s) @ R)  # desk (current estimate) -> SfM
        z += list(Xd[:, 2])
    x2d, X3d, z = np.array(x2d, float), np.array(X3d, float), np.array(z)
    okp, rvec, tvec, inl = cv2.solvePnPRansac(X3d, x2d, K, None, reprojectionError=3.0, iterationsCount=20000,
                                              confidence=0.9999, flags=cv2.SOLVEPNP_SQPNP)
    inl = inl.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(X3d[inl], x2d[inl], K, None, rvec, tvec)
    R_cs = cv2.Rodrigues(rvec)[0]
    C_s = (-R_cs.T @ tvec).ravel()
    err = np.linalg.norm(cv2.projectPoints(X3d[inl], rvec, tvec, K, None)[0].reshape(-1, 2) - x2d[inl], axis=1)
    Rn = R_rc @ R_cs
    # the scale is not observable by PnP: rescale about the overhead camera centre so the desk lands on z = 0. The desk
    # is the dominant mode of the rescale factors that would put each opaque, flat splat Gaussian seen on the desk top
    # onto z = 0 (sparse SfM points and the rendered depth both pick other surfaces / floaters too often)
    s2, R2, t2 = norm_to_desk(dict(al, SPLAT_POS=list(C_r - s * Rn @ C_s), SPLAT_QUAT=list(Rotation.from_matrix(Rn).as_quat())))
    Pg = G_MEANS @ (s2 * R2).T + t2
    dg = Pg - C_r
    pc = dg @ R_rc
    uv = pc[:, :2] / np.maximum(pc[:, 2:3], 1e-6) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    dirs = dg / np.linalg.norm(dg, axis=1, keepdims=True)
    hit = C_r + (-C_r[2] / np.minimum(dirs[:, 2], -1e-6))[:, None] * dirs
    m = G_OK & (pc[:, 2] > 0.2) & (uv[:, 0] > 0) & (uv[:, 0] < W) & (uv[:, 1] > 0) & (uv[:, 1] < H) & \
        (hit[:, 0] > 0.0) & (hit[:, 0] < 0.5) & (np.abs(hit[:, 1]) < 0.35) & (np.hypot(hit[:, 0], hit[:, 1]) > 0.15)
    fac = C_r[2] / (C_r[2] - Pg[m, 2])
    hh, ee = np.histogram(np.log(fac), bins=np.arange(-0.5, 0.4, 0.01))
    pk = ee[np.argmax(np.convolve(hh, np.ones(3), "same"))] + 0.005
    sel = np.abs(np.log(fac) - pk) < 0.015
    sn = s * float(np.median(fac[sel]))
    tn = C_r - sn * Rn @ C_s
    print(f"[refine]   desk mode from {sel.sum()}/{m.sum()} Gaussians: scale x{sn / s:.4f}", flush=True)
    dR = np.degrees(np.linalg.norm(Rotation.from_matrix(Rn @ R.T).as_rotvec()))
    probe = np.array([0.25, 0.0, 0.0])  # where the workspace centre of the old estimate moves to
    dt = np.linalg.norm(sn * Rn @ (((probe - t) / s) @ R) + tn - probe)
    print(f"[refine] iter {it}: {len(x2d)} matches, PnP inliers {len(inl)} (reproj median {np.median(err):.2f} px); "
          f"scale {s:.5f} -> {sn:.5f}; change: rotation {dR:.2f} deg, workspace centre {1000 * dt:.1f} mm", flush=True)
    al.update(SPLAT_POS=[float(x) for x in tn], SPLAT_QUAT=[float(x) for x in Rotation.from_matrix(Rn).as_quat()],
              SPLAT_SCALE=sn, refine_inliers=int(len(inl)), refine_reproj_median_px=float(np.median(err)),
              refine_desk_mode_gaussians=int(sel.sum()))
rgb, depth, alpha = render(al)
real = np.median(np.stack(ovs), 0).astype(np.uint8)
# colour: the splat carries the phone's exposure; per-channel gain that makes the desk look as the overhead camera sees it
v_, u_ = np.mgrid[0:H, 0:W]
Pd = C_r + (np.c_[u_.ravel(), v_.ravel(), np.ones(u_.size)] @ np.linalg.inv(K).T * depth.reshape(-1, 1)) @ R_rc.T
desk_px = ((alpha.ravel() > 0.9) & (np.abs(Pd[:, 2]) < 0.015) &
           (np.linalg.norm(Pd[:, :2] - np.asarray(args.desk_center), axis=1) < args.desk_radius)).reshape(H, W)
gain_c = np.median(real[desk_px].astype(float), 0) / np.maximum(np.median(rgb[desk_px].astype(float), 0), 1.0)
al["SPLAT_COLOR_GAIN"] = [float(g) for g in gain_c[::-1]]  # RGB
print(f"[refine] desk pixels {desk_px.sum()}: colour gain (RGB) {np.round(gain_c[::-1], 3).tolist()}", flush=True)
json.dump(al, open(args.align, "w"), indent=1)
gain = lambda im: np.clip(im.astype(float) * (100 / max(1.0, im.mean())), 0, 255).astype(np.uint8)
rr, rd = gain(real), gain(rgb)
cv2.imwrite(args.out_img, np.concatenate([rr, rd, ((rr.astype(int) + rd) // 2).astype(np.uint8)], 1))
print(f"[refine] wrote {args.align} and {args.out_img}", flush=True)

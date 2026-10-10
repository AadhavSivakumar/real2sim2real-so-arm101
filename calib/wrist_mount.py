"""Real SO-101 wrist-camera mount pose, fitted to the jaws seen in the real wrist image.

The Innomaker wrist camera is rigid to the gripper. In the 24 lens-calibration frames (calib/data/intr_wrist,
phone waved in front of the camera in a dark room) the jaws are static while the background changes, so a low
per-pixel percentile over the frames isolates the jaws (light printed plastic on a black background). The
frames are undistorted with cv2.undistort(img, K, DIST) (same K as output) and the fit renders the pinhole K,
i.e. the fitted pose is the physical mount. NOTE: Isaac's default camera spawn does NOT render K (see K_sim
below); the proposal JSON's 'apply_requirements' lists what has to change together with the pose.

Model: the URDF ``gripper_link`` meshes (wrist_roll_follower = gripper body + fixed jaw, the gripper STS3215)
plus ``moving_jaw_so101_v1_link`` at jaw angle q, full-resolution STL, numpy z-buffer rasteriser (helpers
copied from calib/silhouette.py, which is not imported so its ongoing edits cannot break this file).
Unknowns: camera pose in the gripper link frame (6 DoF) + q. Objective: blurred region agreement (a soft IoU,
TP - FP) between the rendered silhouette and the segmented jaws. Search: NVIDIA's pose + ~370 look-at starts around
the gripper (grid + random; both sides of the jaw plane, both image orientations) x 4 jaw angles, each pan/tilt-
aligned by FFT correlation; Nelder-Mead from the 10 best; 4 parallel basin-hopping chains from the best distinct
stage-1 results plus up to 2 'hypothesis' chains (best start with the camera on the opposite (-y) side of the jaw
plane, best start with the camera near NVIDIA's position) so that competing solutions get the same search effort;
full-res polish. Then a profile (camera centre displaced +-3 mm along each camera axis, rotation + jaw re-fitted)
shows how well the pose is determined. Deterministic (fixed seeds).

Frames / conventions
  * URDF ``gripper_link`` == USD ``gripper`` body/prim frame relative to the gripper geometry (T_USD_URDF below =
    identity). Verified in Isaac by scripts/check_wrist_mount.py: the URDF STLs placed in gripper_link / jaw link
    coincide with the USD visual meshes of the 'gripper' / 'moving_jaw_so101_v1' bodies (nearest point < 0.05 mm),
    the jaw joint is identical (0.000 mm / 0.000 deg over 10 configurations), and the camera prim's world pose equals
    gripper body pose x offset. (Kinematically, NVIDIA's USD lacks the 0.0486795 rad term of the URDF wrist_roll
    origin: at equal joint angles the sim gripper is rolled +2.79 deg about its z axis relative to URDF FK. That is a
    wrist_roll zero offset and does not affect a camera pose expressed relative to the gripper.)
  * Output WRIST_CAM_POS / WRIST_CAM_QUAT use exactly env_cfg's convention: CameraCfg.OffsetCfg(pos, rot,
    convention="opengl") under {ENV_REGEX_NS}/Robot/gripper/wrist_cam, i.e. the camera prim pose in the gripper
    prim frame, camera looking along -z with +y up, quaternion (x, y, z, w).
  * Internally a camera is (R, t): R = 3x3 with the OpenCV camera axes (x right, y down, z forward) as columns,
    t = camera centre, both in the gripper link frame. R_opengl = R @ diag(1, -1, -1).
  * Jaw angle q in sim radians (USD/URDF joint 'gripper'); LeRobot 0..100 via so101_r2s2r/units.py.
  * Isaac's default camera spawn (PinholeCameraCfg.from_intrinsic_matrix, as env_cfg uses it) ignores the
    principal point and non-square pixels (isaaclab.utils.sensors convert_camera_intrinsics_to_usd): it renders
    f = (fx + fy) / 2, cx = W/2, cy = H/2 (K_sim below). The fit uses the true K (physical mount);
    check_wrist_mount.py compares Isaac renders against the real median undistorted to K and to K_sim, and also
    renders a camera with spawn.distortion = OpenCvPinholeDistortionCfg(exact K, coefficients muted). The cmd_fit
    output also carries WRIST_CAM_QUAT_KSIM_COMPENSATED: the rotation that makes a K_sim render approximate the
    K image (fallback when neither the sim intrinsics nor the real-side undistortion target can change).

Usage (numpy + opencv + scipy; PY=~/TestingGrounds/robot_splat/.venv/bin/python):
  $PY calib/wrist_mount.py fit  --scratch DIR      # segment + fit -> calib/data/wrist_mount_proposal.json,
                                                   #   calib/data/wrist_mount_fit_overlay.jpg, DIR/real_*.png
  $PY calib/wrist_mount.py show --scratch DIR --pos X Y Z --quat X Y Z W --jaw Q   # score/overlay any pose
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

try:  # cv2 is only needed for the image I/O + fitting commands; scripts/check_wrist_mount.py (Isaac python) imports
    import cv2  # the geometry, renderer and metrics from this file without it
except ImportError:  # pragma: no cover
    cv2 = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
URDF = Path("/mnt/secondary/so101_r2s/assets/urdf/so101_new_calib.urdf")
FRAMES = HERE / "data" / "intr_wrist"
PROPOSAL = HERE / "data" / "wrist_mount_proposal.json"
GRIPPER_LINK, JAW_LINK, JAW_JOINT = "gripper_link", "moving_jaw_so101_v1_link", "gripper"
PRINTED, SERVO, JAW = 1, 2, 3  # render labels
GL_FROM_CV = np.diag([1.0, -1.0, -1.0])

# URDF gripper_link -> USD 'gripper' prim frame (x_usd = R @ x_urdf + p), relative to the gripper geometry. Identity:
# established with scripts/check_wrist_mount.py (proposal JSON 'frame_check': mesh match, jaw joint, camera prim pose).
T_USD_URDF = np.eye(4)


# ----------------------------------------------------------------------------- calibration constants
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sim_calib():
    """so101_r2s2r/calib.py loaded standalone (it reads calib/data/calib.json; no Isaac imports)."""
    return _load_module("so101_calib_standalone", ROOT / "so101_r2s2r" / "calib.py")


def units():
    return _load_module("so101_units_standalone", ROOT / "so101_r2s2r" / "units.py")


def camera_model():
    c = sim_calib()
    K = np.asarray(c.WRIST_K, float).reshape(3, 3)
    dist = np.asarray(c.WRIST_DIST, float)
    W, H = c.IMAGE_SIZE
    return K, dist, (int(W), int(H)), c


def k_sim(K, size):
    """The intrinsics Isaac actually renders for PinholeCameraCfg.from_intrinsic_matrix(K): mean focal length,
    centred principal point (isaaclab.utils.sensors.convert_camera_intrinsics_to_usd)."""
    f = (K[0, 0] + K[1, 1]) / 2
    return np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1.0]])


def ksim_compensation(K, size, step=8):
    """Rotation M (camera frame, OpenCV axes) such that a K_sim camera rotated by M sees, at every pixel, nearly
    the ray a K camera sees at the same pixel: argmin_M sum |M n(K_sim^-1 p) - n(K^-1 p)|^2 over a pixel grid
    (orthogonal Procrustes). R_compensated = R @ M. Returns (M, residual px over the image: mean, max)."""
    W, H = size
    u, v = np.meshgrid(np.linspace(0, W - 1, W // step + 1), np.linspace(0, H - 1, H // step + 1))
    P = np.stack([u.ravel(), v.ravel(), np.ones(u.size)], 0)
    Ks = k_sim(K, size)
    a = np.linalg.inv(Ks) @ P
    b = np.linalg.inv(K) @ P
    a /= np.linalg.norm(a, axis=0)
    b /= np.linalg.norm(b, axis=0)
    U, _, Vt = np.linalg.svd(b @ a.T)
    M = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    # residual: sim pixel p sees ray M K_sim^-1 p; where does the K camera see that ray?
    r = K @ (M @ np.linalg.inv(Ks) @ P)
    err = np.linalg.norm(r[:2] / r[2] - P[:2], axis=0)
    return M, float(err.mean()), float(err.max())


# NVIDIA's camera-mount model in the SO-101 USD (prim paths relative to the robot prim); not on the real arm.
NV_MOUNT_VISUAL = "gripper/visuals/camera_mount"
NV_MOUNT_COLLISION = "gripper/collisions/camera_mount"


def apply_requirements(prop=None):
    """What must change together with WRIST_CAM_POS / WRIST_CAM_QUAT when the proposal is applied (stored in the
    proposal JSON). Evidence numbers are filled from the proposal's 'isaac_validation' block when present."""
    prop = prop or {}
    res = prop.get("isaac_validation", {}).get("results", {})

    def iou(cam, tag):
        try:
            return round(res[cam][f"vs_real_undistorted_to_{tag}"]["iou"], 3)
        except KeyError:
            return None

    comp = prop.get("ksim_compensated", {})
    ex = res.get("fitted_exactK", {})
    exact_ok = ex.get("renderer_uses_exact_K")
    return [
        dict(id="remove_nvidia_camera_mount", required=True,
             what=f"Remove NVIDIA's camera-mount model from the robot in the env: make the visual prim "
                  f"<Robot>/{NV_MOUNT_VISUAL} invisible and disable collision on every collider under "
                  f"<Robot>/{NV_MOUNT_COLLISION}, with a mode='prestartup' EventTerm (de-instance the enclosing "
                  f"instanceable prim first). Ready-made function: remove_nvidia_mount() in scripts/check_wrist_mount.py. "
                  f"The collider part belongs in CubeInCupEnvCfg (all variants), the visual part at least in CubeInCupCamEnvCfg.",
             why="The mount occupies y = 24..84 mm, z = -56..-18 mm of the gripper frame, directly in front of the fitted "
                 "camera: with it visible the fitted camera images only the mount (all 307200 px at 2-3 cm depth, first "
                 "Isaac run). The real camera sees the jaws through that space, so the real arm has nothing there and the "
                 "collider is a phantom obstacle next to the jaws.",
             evidence=dict(isaac_check_with_mount_visible="scratchpad wrist_mount/isaac_check_1.log: fitted camera 'jaw pixels "
                                                          "307200 (depth [0.0218, 0.0330])'",
                           this_validation=prop.get("isaac_validation", {}).get("nvidia_mount"))),
        dict(id="wrist_intrinsics", required=True,
             what="Make the sim render the same pinhole the real frames are undistorted to. Pick ONE: "
                  "(b) env_cfg: scene.wrist_cam.spawn.distortion = sim_utils.OpenCvPinholeDistortionCfg(fx, fy, cx, cy from "
                  "WRIST_K, image_size=(640, 480), apply_lens_distortion=False) -> Isaac renders the exact K and the "
                  "documented real-side cv2.undistort(img, K, DIST) stays valid; "
                  "(a) keep the default spawn and undistort real frames with cv2.undistort(img, K, DIST, None, K_sim), "
                  "K_sim = intrinsics.K_isaac_render, and correct the comment in so101_r2s2r/calib.py; "
                  "(c) change nothing else and use WRIST_CAM_QUAT_KSIM_COMPENSATED instead of WRIST_CAM_QUAT "
                  "(approximation, see ksim_compensated).",
             why="PinholeCameraCfg.from_intrinsic_matrix drops the principal point and averages fx, fy "
                 "(isaaclab.utils.sensors.convert_camera_intrinsics_to_usd): Isaac renders K_sim (f 550.57, c (320, 240)) "
                 "while the real K has c (327.54, 246.31), a 7.5 / 6.3 px (0.78 / 0.66 deg) shift. With the pose alone, "
                 "the sim jaws sit ~7 px off the K-undistorted real jaws.",
             evidence=dict(
                 default_spawn_vs_real_undistorted_to_K=iou("fitted", "K"),
                 default_spawn_vs_real_undistorted_to_K_sim_option_a=iou("fitted", "Ksim"),
                 exact_K_spawn_vs_real_undistorted_to_K_option_b=iou("fitted_exactK", "K"),
                 exact_K_spawn_renderer_uses_exact_K=exact_ok,
                 ksim_compensated_numpy_iou_vs_real_undistorted_to_K_option_c=comp.get("iou_numpy_vs_real_undistorted_to_K"),
                 ksim_compensated_residual_px=dict(mean=comp.get("residual_px_mean"), max=comp.get("residual_px_max")))),
        dict(id="scene_cam_intrinsics", required=False,
             what="The overhead scene_cam is spawned the same way (from_intrinsic_matrix), so it too renders a centred "
                  "principal point and mean focal length; apply the same choice once SCENE_K is calibrated.",
             why="Same IsaacLab conversion; not measured here."),
    ]


# ----------------------------------------------------------------------------- conventions
def cv_to_env(R, t, T_usd_urdf=T_USD_URDF):
    """(R_cv, t) in the URDF gripper_link frame -> (WRIST_CAM_POS, WRIST_CAM_QUAT xyzw) as env_cfg uses them."""
    A, p = T_usd_urdf[:3, :3], T_usd_urdf[:3, 3]
    R_gl = A @ np.asarray(R, float) @ GL_FROM_CV
    pos = A @ np.asarray(t, float) + p
    q = Rotation.from_matrix(R_gl).as_quat()
    if q[3] < 0:
        q = -q
    return tuple(float(v) for v in pos), tuple(float(v) for v in q)


def env_to_cv(pos, quat, T_usd_urdf=T_USD_URDF):
    """Inverse of cv_to_env: env_cfg (pos, quat xyzw, OpenGL) -> (R_cv, t) in the URDF gripper_link frame."""
    A, p = T_usd_urdf[:3, :3], T_usd_urdf[:3, 3]
    R_gl = Rotation.from_quat(np.asarray(quat, float)).as_matrix()
    return A.T @ R_gl @ GL_FROM_CV, A.T @ (np.asarray(pos, float) - p)


# ----------------------------------------------------------------------------- geometry helpers
# (copied from calib/silhouette.py)
def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_angle(axis, a):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * Kx + (1 - np.cos(a)) * Kx @ Kx


def tf(R=np.eye(3), t=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def load_stl(path) -> np.ndarray:
    """STL (binary or ASCII) -> triangle soup (n, 3, 3) float64."""
    b = Path(path).read_bytes()
    if len(b) >= 84:
        n = int(np.frombuffer(b, np.uint32, 1, 80)[0])
        if len(b) == 84 + 50 * n:
            dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(b, dt, n, 84)["v"].astype(np.float64)
    tok = b.decode("ascii", "ignore").split()
    v = [[float(tok[i + 1]), float(tok[i + 2]), float(tok[i + 3])] for i, t in enumerate(tok) if t == "vertex"]
    return np.asarray(v, float).reshape(-1, 3, 3)


def cluster_mesh(soup: np.ndarray, cell: float):
    """Weld + decimate a triangle soup by vertex clustering on a `cell` [m] grid -> (V (n,3), F (m,3))."""
    P = soup.reshape(-1, 3)
    g = np.floor(P / cell).astype(np.int64)
    g -= g.min(0)
    dims = g.max(0) + 1
    key = (g[:, 0] * dims[1] + g[:, 1]) * dims[2] + g[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    V = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], 1) / cnt[:, None]
    F = inv.reshape(-1, 3)
    F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
    s = np.sort(F, 1)
    n = np.int64(len(V))
    _, first = np.unique((s[:, 0] * n + s[:, 1]) * n + s[:, 2], return_index=True)
    F = F[np.sort(first)]
    used, F = np.unique(F, return_inverse=True)
    return V[used], F.reshape(-1, 3).astype(np.int32)


def scale_K(K, s):
    """Intrinsics of the image resized by factor s (cv2.resize pixel-centre convention)."""
    Ks = np.array(K, float).copy()
    Ks[0, :] *= s
    Ks[1, :] *= s
    Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    Ks[2] = (0, 0, 1)
    return Ks


def _ramp(n):
    tot = int(n.sum())
    return np.arange(tot) - np.repeat(np.cumsum(n) - n, n)


def rasterize(x, y, z, lab, size, labels=True):
    """Triangles (per-vertex pixel coords x, y, depth z, all (m,3)) -> uint8 label image (z-buffered) or bool
    mask. A pixel is covered when its centre (integer coords, OpenCV convention) lies in a triangle."""
    W, H = size
    o = np.argsort(y, axis=1)
    x, y, z = np.take_along_axis(x, o, 1), np.take_along_axis(y, o, 1), np.take_along_axis(z, o, 1)
    xt, xm, xb, yt, ym, yb = x[:, 0], x[:, 1], x[:, 2], y[:, 0], y[:, 1], y[:, 2]
    area = (xm - xt) * (yb - yt) - (xb - xt) * (ym - yt)
    ylo = np.maximum(np.ceil(yt), 0)
    yhi = np.minimum(np.floor(yb), H - 1)
    keep = np.nonzero((yhi >= ylo) & (np.abs(area) > 1e-9) & (np.maximum(np.maximum(xt, xm), xb) >= 0)
                      & (np.minimum(np.minimum(xt, xm), xb) <= W - 1))[0]
    xt, xm, xb, yt, ym, yb, area, ylo, yhi = (a[keep] for a in (xt, xm, xb, yt, ym, yb, area, ylo, yhi))
    s_long = (xb - xt) / (yb - yt)
    s_up = np.where(ym > yt, (xm - xt) / np.where(ym > yt, ym - yt, 1.0), 0.0)
    s_low = np.where(yb > ym, (xb - xm) / np.where(yb > ym, yb - ym, 1.0), 0.0)
    h = (yhi - ylo + 1).astype(np.int64)
    tri = np.repeat(np.arange(len(h)), h)
    r = ylo[tri] + _ramp(h)
    ytt = yt[tri]
    x_long = xt[tri] + (r - ytt) * s_long[tri]
    ymt = ym[tri]
    x_short = np.where(r < ymt, xt[tri] + (r - ytt) * s_up[tri], xm[tri] + (r - ymt) * s_low[tri])
    c0 = np.maximum(np.ceil(np.minimum(x_long, x_short)), 0)
    c1 = np.minimum(np.floor(np.maximum(x_long, x_short)), W - 1)
    n = np.maximum(c1 - c0 + 1, 0).astype(np.int64)
    span = np.repeat(np.arange(len(n)), n)
    cc = c0[span] + _ramp(n)
    rr = r[span]
    pix = (rr * W + cc).astype(np.int64)
    if not labels:
        m = np.zeros(H * W, bool)
        m[pix] = True
        return m.reshape(H, W)
    zt, zm, zb = z[keep, 0], z[keep, 1], z[keep, 2]
    dx1, dx2, dy1, dy2, dz1, dz2 = xm - xt, xb - xt, ym - yt, yb - yt, zm - zt, zb - zt
    A = (dz1 * dy2 - dz2 * dy1) / area
    B = (dx1 * dz2 - dx2 * dz1) / area
    C = zt - A * xt - B * yt
    t = tri[span]
    zz = np.clip(C[t] + A[t] * cc + B[t] * rr, np.minimum(np.minimum(zt, zm), zb)[t], np.maximum(np.maximum(zt, zm), zb)[t])
    zbuf = np.full(H * W, np.inf)
    np.minimum.at(zbuf, pix, zz)
    win = zz <= zbuf[pix]
    out = np.zeros(H * W, np.uint8)
    out[pix[win]] = lab[keep][t[win]]
    return out.reshape(H, W)


# ----------------------------------------------------------------------------- gripper model
class GripperModel:
    """URDF gripper_link + moving jaw meshes in the gripper_link frame; `cell` > 0 decimates (vertex clustering)."""

    def __init__(self, urdf=URDF, cell: float = 0.0):
        urdf = Path(urdf)
        root = ET.parse(urdf).getroot()
        j = next(j for j in root.findall("joint") if j.get("name") == JAW_JOINT)
        o = j.find("origin")
        self.T_jaw0 = tf(rpy_matrix(*[float(v) for v in o.get("rpy").split()]), [float(v) for v in o.get("xyz").split()])
        self.jaw_axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
        lim = j.find("limit")
        self.jaw_limits = (float(lim.get("lower")), float(lim.get("upper")))
        assert j.find("parent").get("link") == GRIPPER_LINK and j.find("child").get("link") == JAW_LINK
        self.parts = {GRIPPER_LINK: [], JAW_LINK: []}
        for ln in root.findall("link"):
            name = ln.get("name")
            if name not in self.parts:
                continue
            for vis in ln.findall("visual"):
                m = vis.find("geometry/mesh")
                o = vis.find("origin")
                xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
                fn = urdf.parent / m.get("filename").replace("package://", "")
                soup = load_stl(fn) * np.array([float(v) for v in m.get("scale", "1 1 1").split()])
                V, F = cluster_mesh(soup, cell if cell > 0 else 1e-5)
                T = tf(rpy_matrix(*rpy), xyz)
                V = V @ T[:3, :3].T + T[:3, 3]
                lab = JAW if name == JAW_LINK else (SERVO if Path(fn).stem.startswith("sts3215") else PRINTED)
                self.parts[name].append((V, F, np.full(len(F), lab, np.uint8), Path(fn).stem))
        self.n_tris = sum(len(p[1]) for ps in self.parts.values() for p in ps)

    def jaw_transform(self, q):
        return self.T_jaw0 @ tf(axis_angle(self.jaw_axis, q))

    def mesh(self, q):
        """(V (n,3), F (m,3), label (m,)) in the gripper_link frame at jaw angle q [rad]."""
        Vs, Fs, Ls, off = [], [], [], 0
        TJ = self.jaw_transform(q)
        for link, parts in self.parts.items():
            for V, F, L, _ in parts:
                if link == JAW_LINK:
                    V = V @ TJ[:3, :3].T + TJ[:3, 3]
                Vs.append(V); Fs.append(F + off); Ls.append(L); off += len(V)
        return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ls)


def render(V, F, L, R, t, K, size, labels=True, near=0.002):
    """Label image (0 bg, PRINTED, SERVO, JAW) of a gripper-frame mesh for camera (R_cv, t) and pinhole K."""
    Xc = (V - np.asarray(t, float)) @ np.asarray(R, float)
    z = Xc[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    u = K[0, 0] * Xc[:, 0] / zs + K[0, 1] * Xc[:, 1] / zs + K[0, 2]
    v = K[1, 1] * Xc[:, 1] / zs + K[1, 2]
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    ok = (z[a] > near) & (z[b] > near) & (z[c] > near)
    ok &= (u[b] - u[a]) * (v[c] - v[a]) - (u[c] - u[a]) * (v[b] - v[a]) < 0  # back-face culling
    Ft = F[ok]
    return rasterize(u[Ft], v[Ft], z[Ft], L[ok], size, labels)


# ----------------------------------------------------------------------------- real image + segmentation
def real_views(frames_dir=FRAMES, K=None, dist=None, K_out=None, pct=25.0, weak_frac=0.35):
    """Median image and jaw segmentation of the static parts of the wrist frames, undistorted to K_out (default K).

    Segmentation: per-pixel `pct` percentile of the grey level over the frames (the waved phone and hands are
    transient; the jaws are static and light, the room is dark) -> hysteresis threshold (Otsu for the lit faces,
    weak_frac x Otsu for the shaded faces) -> soft probability, cleaned to the large connected components.
    """
    files = sorted(glob.glob(str(Path(frames_dir) / "*.jpg")))
    st = np.stack([cv2.imread(f) for f in files]).astype(np.float32)
    med = np.median(st, 0)
    low = np.percentile(st.mean(3), pct, axis=0).astype(np.float32)
    K_out = K if K_out is None else K_out
    und = lambda a: cv2.undistort(a, K, dist, None, K_out)  # noqa: E731
    med_u = np.clip(und(med), 0, 255).astype(np.uint8)
    low_u = und(low)
    v = np.clip(low_u, 0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # hysteresis: the lit jaw faces (~60 grey) are 'strong'; the shaded faces (~14-30) are kept when connected
    # to them; the room is ~2-6 next to the jaws (up to ~13 at the far image edges, not connected)
    weak_thr = weak_frac * thr
    prob = 1.0 / (1.0 + np.exp(-(low_u - weak_thr) / 1.5))
    m = (prob > 0.5).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    strong = low_u > thr
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros(n, bool)
    keep[1:] = (stats[1:, cv2.CC_STAT_AREA] > 0.004 * m.size) & (np.bincount(cc[strong], minlength=n)[1:] > 0.002 * m.size)
    m = keep[cc]
    inv = (~m).astype(np.uint8)  # fill holes (background components not touching the border)
    n2, cc2, st2, _ = cv2.connectedComponentsWithStats(inv, 4)
    H, W = m.shape
    for i in range(1, n2):
        x, y, w, h = st2[i, :4]
        if x > 0 and y > 0 and x + w < W and y + h < H:
            m[cc2 == i] = True
    near = cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    prob = np.where(near, prob, 0.0).astype(np.float32)
    prob[m] = np.maximum(prob[m], 0.5)
    return dict(median=med_u, low=np.clip(low_u, 0, 255).astype(np.uint8), mask=m, prob=prob, thr=float(thr),
                n_frames=len(files), K=K_out)


# ----------------------------------------------------------------------------- metrics + overlays
def _boundary(m):
    m = np.asarray(m, bool)
    b = m & ~ndimage.binary_erosion(m, np.ones((3, 3), bool), border_value=1)
    b[:1, :] = b[-1:, :] = False  # mask edges on the image border are not object boundaries
    b[:, :1] = b[:, -1:] = False
    return b


def mask_metrics(r, s):
    """IoU and symmetric boundary residuals [px] between rendered mask r and segmented mask s (bool HxW).
    Boundary pixels on the image border are ignored (the jaws leave the image there)."""
    r, s = np.asarray(r, bool), np.asarray(s, bool)
    inter, union = int((r & s).sum()), int((r | s).sum())
    rb, sb = _boundary(r), _boundary(s)
    out = dict(iou=inter / max(union, 1), n_rendered=int(r.sum()), n_segmented=int(s.sum()))
    if rb.any() and sb.any():
        dts = ndimage.distance_transform_edt(~sb)
        dtr = ndimage.distance_transform_edt(~rb)
        fwd, bwd = dts[rb], dtr[sb]
        both = np.r_[fwd, bwd]
        out.update(boundary_mean_px=float(both.mean()), boundary_median_px=float(np.median(both)),
                   boundary_p90_px=float(np.percentile(both, 90)), boundary_max_px=float(both.max()),
                   render_to_seg_mean_px=float(fwd.mean()), seg_to_render_mean_px=float(bwd.mean()))
    return out


def per_jaw_iou(r, s):
    """IoU of the left and right halves split at the gap between the two segmented jaws (column-wise)."""
    cols = np.nonzero(s.any(0))[0]
    if len(cols) == 0:
        return {}
    occ = s.sum(0)
    lo, hi = np.percentile(cols, 25), np.percentile(cols, 75)
    mid = int(lo + np.argmin(occ[int(lo):int(hi) + 1]))
    res = {}
    for name, sl in (("left", np.s_[:, :mid]), ("right", np.s_[:, mid:])):
        a, b = r[sl], s[sl]
        res[name] = float((a & b).sum() / max((a | b).sum(), 1))
    return res


def contour(img, m, color, thick=1):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cs, -1, color, thick)
    return img


def label(img, text, y=22, color=(255, 255, 255)):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (4, y - h - 4), (12 + w, y + base), (0, 0, 0), -1)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def brighten(img, gamma=0.5):
    return (255 * (img.astype(np.float32) / 255.0) ** gamma).astype(np.uint8)


# ----------------------------------------------------------------------------- fitter
class MountFitter:
    """Camera pose (R_cv, t in the gripper link frame) + jaw angle q from one segmented wrist image."""

    # x = [rotvec about the start camera's axes (deg) x3, translation in the start camera's axes (mm) x3, q (deg)]
    UNITS = np.array([1.0, 1.0, 1.0, 1e-3, 1e-3, 1e-3, np.pi / 180])

    def __init__(self, prob, K, size, verbose=True):
        self.prob, self.K, self.size = prob.astype(np.float32), np.asarray(K, float), size
        self.full = GripperModel(cell=0.0)
        self.coarse = GripperModel(cell=0.0008)
        self.qlim = self.full.jaw_limits
        self.verbose = verbose
        self._lv, self._mesh_cache = {}, {}
        self.n_eval = 0
        self.log(f"[fit] mesh: {self.full.n_tris} tris full, {self.coarse.n_tris} coarse")

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    def level(self, scale, sigma):
        key = (scale, sigma)
        if key not in self._lv:
            W, H = self.size
            size = (int(round(W * scale)), int(round(H * scale)))
            p = cv2.resize(self.prob, size, interpolation=cv2.INTER_AREA) if scale != 1 else self.prob
            A = 2 * p - 1
            if sigma > 0:
                A = cv2.GaussianBlur(A, (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
            self._lv[key] = (A, scale_K(self.K, scale), size, max(float(p.sum()), 1.0))
        return self._lv[key]

    def mesh(self, q, coarse):
        key = (round(float(q), 5), coarse)
        if key not in self._mesh_cache:
            if len(self._mesh_cache) > 64:
                self._mesh_cache.clear()
            self._mesh_cache[key] = (self.coarse if coarse else self.full).mesh(q)
        return self._mesh_cache[key]

    def score(self, R, t, q, scale, sigma, coarse=False):
        A, Ks, size, norm = self.level(scale, sigma)
        V, F, L = self.mesh(q, coarse)
        m = render(V, F, L, R, t, Ks, size, labels=False)
        self.n_eval += 1
        return float(A[m].sum()) / norm

    def pose(self, x, R0, t0, q0):
        p = np.asarray(x, float) * self.UNITS
        R = R0 @ Rotation.from_rotvec(np.radians(p[:3])).as_matrix()
        t = t0 + R0 @ p[3:6]
        q = float(np.clip(q0 + p[6], *self.qlim))
        return R, t, q

    def refine(self, R0, t0, q0, scale, sigma, step=1.0, maxfev=400, coarse=False, fit_q=True):
        def f(x):
            if not fit_q:
                x = np.r_[x, 0.0]
            R, t, q = self.pose(x, R0, t0, q0)
            pen = 0.0
            raw = q0 + x[6] * self.UNITS[6]
            if raw < self.qlim[0] or raw > self.qlim[1]:
                pen = 0.1 * abs(raw - np.clip(raw, *self.qlim))
            return -self.score(R, t, q, scale, sigma, coarse) + pen
        n = 7 if fit_q else 6
        steps = np.array([2.0, 2.0, 2.0, 4.0, 4.0, 4.0, 6.0])[:n] * step
        x0 = np.zeros(n)
        sim = np.vstack([x0, np.diag(steps)])
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=sim, xatol=0.02, fatol=1e-5, maxfev=maxfev, adaptive=True))
        x = res.x if fit_q else np.r_[res.x, 0.0]
        R, t, q = self.pose(x, R0, t0, q0)
        return R, t, q, -res.fun

    def diagnostics(self, R, t, q, K=None, size=None):
        K = self.K if K is None else K
        size = self.size if size is None else size
        V, F, L = self.full.mesh(q)
        lab = render(V, F, L, R, t, K, size)
        r = lab > 0
        s = self.prob > 0.5
        d = mask_metrics(r, s)
        d["per_jaw_iou"] = per_jaw_iou(r, s)
        d["visible_labels_px"] = {n: int((lab == k).sum()) for n, k in (("gripper_body", PRINTED), ("servo", SERVO), ("moving_jaw", JAW))}
        return d, lab


def look_at_cv(eye, target, right):
    """R_cv (columns x right, y down, z forward) of a camera at eye looking at target, `right` ~ image +x."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    x = np.asarray(right, float) - z * (np.asarray(right, float) @ z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def start_grid(n_random=240, seed=0):
    """Look-at starts around the gripper: a regular grid (camera on either side of the jaw plane, several heights
    along the jaws, looking at points along the jaws, image-right = +x or -x, i.e. fixed jaw left or right) plus
    random camera centres in a box around the gripper with random look-at points on the jaw axis."""
    out = []
    for side in (+1, -1):
        for y in (0.03, 0.045, 0.06, 0.075):
            for z in (0.01, -0.015, -0.04, -0.065):
                for tz in (-0.07, -0.10):
                    for right in (+1, -1):
                        eye = np.array([0.0, side * y, z])
                        out.append((f"grid y{side * y:+.3f} z{z:+.3f} tz{tz:+.2f} r{right:+d}",
                                    look_at_cv(eye, (0.0, 0.0, tz), (right, 0, 0)), eye))
    rng = np.random.default_rng(seed)
    for i in range(n_random):
        eye = np.array([rng.uniform(-0.03, 0.03), rng.choice([-1, 1]) * rng.uniform(0.02, 0.09), rng.uniform(-0.08, 0.03)])
        tgt = np.array([rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), rng.uniform(-0.11, -0.05)])
        right = np.array([rng.choice([-1, 1]), 0, 0]) + rng.normal(0, 0.2, 3)
        out.append((f"rand{i:03d}", look_at_cv(eye, tgt, right), eye))
    return out


def _rot_between(b, a):
    """Smallest rotation taking direction b onto direction a."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(b, a), float(b @ a)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def shift_align(fz, R, t, q, scale=0.25, sigma=3.0, max_frac=0.6):
    """Best image shift of the rendered silhouette against the agreement map (FFT cross-correlation), turned into
    a camera rotation about its centre (pan/tilt). Returns (score, R_aligned)."""
    A, Ks, size, norm = fz.level(scale, sigma)
    W, H = size
    V, F, L = fz.mesh(q, True)
    m = render(V, F, L, R, t, Ks, size, labels=False)
    if not m.any():
        return -np.inf, R
    Wp, Hp = 2 * W, 2 * H
    FA = np.fft.rfft2(np.pad(A, ((0, Hp - H), (0, Wp - W)), constant_values=-1.0))
    corr = np.fft.irfft2(np.conj(np.fft.rfft2(m.astype(np.float32), s=(Hp, Wp))) * FA, s=(Hp, Wp))
    mu, mv = int(max_frac * W), int(max_frac * H)
    dus, dvs = np.r_[0:mu + 1, -mu:0], np.r_[0:mv + 1, -mv:0]
    c = corr[np.ix_(dvs % Hp, dus % Wp)]
    iv, iu = np.unravel_index(np.argmax(c), c.shape)
    du, dv = dus[iu] / scale, dvs[iv] / scale
    ys, xs = np.nonzero(m)
    p0 = np.array([xs.mean(), ys.mean()]) / scale
    Kinv = np.linalg.inv(fz.K)
    a, b = Kinv @ np.r_[p0, 1], Kinv @ np.r_[p0 + (du, dv), 1]
    R2 = R @ _rot_between(b, a)
    return fz.score(R2, t, q, scale, sigma, coarse=True), R2


def _hop_chain(args):
    """One basin-hopping chain (own process): random pose kicks + local Nelder-Mead at 1/4 then 1/2 resolution;
    a kick is accepted when the 1/2-res score improves."""
    prob, K, size, R, t, q, seed, n_hops = args
    fz = MountFitter(prob, K, size, verbose=False)
    rng = np.random.default_rng(seed)
    R, t, q, s = fz.refine(R, t, q, 0.5, 1.5, step=0.7, maxfev=500)
    hist = [float(s)]
    for _ in range(n_hops):
        k = rng.choice([0.5, 1.0, 2.0])
        x = np.r_[rng.normal(0, 3.0 * k, 3), rng.normal(0, 6.0 * k, 3), rng.normal(0, 8.0 * k, 1)]
        Rp, tp, qp = fz.pose(x, R, t, q)
        Rp, tp, qp, _ = fz.refine(Rp, tp, qp, 0.25, 2.0, step=1.5, maxfev=300, coarse=True)
        Rp, tp, qp, sp = fz.refine(Rp, tp, qp, 0.5, 1.5, step=0.7, maxfev=400)
        if sp > s + 1e-5:
            R, t, q, s = Rp, tp, qp, sp
        hist.append(float(s))
    return R, t, q, s, hist, fz.n_eval


def fit(prob, K, size, R_nv, t_nv, q_starts=(0.0, 0.25, 0.5, 0.8), topk=10, n_chains=4, n_hops=30, verbose=True):
    """Multi-start + basin hopping. Stage 0 scores NVIDIA's pose and a grid of look-at starts (x 4 jaw angles) at
    1/4 resolution with a wide blur; stage 1 runs Nelder-Mead from the best `topk` (NVIDIA always included);
    stage 2 runs `n_chains` basin-hopping chains in parallel from the best distinct stage-1 results; stage 3
    refines the winner at full resolution. Besides the `n_chains` best distinct stage-1 results, the best stage-1
    result with the camera on the opposite (-y) side and the best one near NVIDIA's camera centre get a chain each
    (if not already among the seeds). The chains' spread is a measure of how well the pose is determined."""
    from concurrent.futures import ProcessPoolExecutor

    fz = MountFitter(prob, K, size, verbose)
    t0 = time.time()
    starts = [("nvidia", R_nv, t_nv)] + start_grid()
    cands = []
    for name, R, t in starts:
        for q in q_starts:
            s_raw = fz.score(R, t, q, 0.25, 3.0, coarse=True)
            s_al, R_al = shift_align(fz, R, t, q)
            cands.append((s_raw, name, R, t, q) if s_raw >= s_al else (s_al, name + " +shift", R_al, t, q))
    cands.sort(key=lambda c: -c[0])
    picked = []
    for c in cands:  # top-k distinct starts (camera centre > 8 mm apart or view direction > 8 deg apart)
        if all(np.linalg.norm(c[3] - d[3]) > 0.008 or np.degrees(np.arccos(np.clip(c[2][:, 2] @ d[2][:, 2], -1, 1))) > 8
               or abs(c[4] - d[4]) > 0.2 for d in picked):
            picked.append(c)
        if len(picked) >= topk:
            break
    if not any(c[1].startswith("nvidia") for c in picked):
        picked.append(max((c for c in cands if c[1].startswith("nvidia")), key=lambda c: c[0]))
    fz.log(f"[fit] stage 0: {len(cands)} starts scored in {time.time() - t0:.1f}s; best {cands[0][0]:.3f} ({cands[0][1]}, q={cands[0][4]:.2f})")
    res1 = []
    for s0, name, R, t, q in picked:
        R1, t1, q1, _ = fz.refine(R, t, q, 0.25, 3.0, step=2.0, maxfev=500, coarse=True)
        R1, t1, q1, s1 = fz.refine(R1, t1, q1, 0.5, 1.5, step=1.0, maxfev=500)
        res1.append((s1, name, R1, t1, q1))
        fz.log(f"[fit] stage 1: {name:44s} q0={q:.2f}: {s0:.3f} -> {s1:.4f}  t={np.round(t1 * 1000, 1)} mm q={q1:.3f}")
    res1.sort(key=lambda c: -c[0])
    seeds, roles = [res1[0]], ["best"]
    for c in res1[1:]:  # distinct basins first, then fill up with the best ones
        if len(seeds) < n_chains and all(pose_delta(c[2], c[3], d[2], d[3])["d_pos_norm_mm"] > 5 for d in seeds):
            seeds.append(c)
            roles.append("distinct basin")
    for c in res1:
        if len(seeds) >= n_chains:
            break
        if not any(c is d for d in seeds):
            seeds.append(c)
            roles.append("next best")
    # competing hypotheses get a basin-hopping chain of their own (appended: the seeds above keep their RNG seeds)
    hyps = (("hypothesis: camera on the opposite (-y) side", lambda c: c[3][1] < 0),
            ("hypothesis: camera near NVIDIA's position (< 20 mm)", lambda c: np.linalg.norm(c[3] - t_nv) < 0.020))
    for role, pred in hyps:
        if any(pred(d) for d in seeds):
            continue
        cand = [c for c in res1 if pred(c) and not any(c is d for d in seeds)]
        if cand:
            seeds.append(max(cand, key=lambda c: c[0]))
            roles.append(role)
    jobs = [(prob, K, size, c[2], c[3], c[4], 100 + i, n_hops) for i, c in enumerate(seeds)]
    with ProcessPoolExecutor(min(len(jobs), 6)) as ex:
        chains = list(ex.map(_hop_chain, jobs))
    for c, role, (R, t, q, s, hist, ne) in zip(seeds, roles, chains):
        fz.n_eval += ne
        fz.log(f"[fit] stage 2: chain from {c[1]:44s} [{role}]: {c[0]:.4f} -> {s:.4f}  t={np.round(t * 1000, 1)} mm q={q:.3f}")
    order = np.argsort([-ch[3] for ch in chains])
    R, t, q, s2 = chains[order[0]][:4]
    spread = [dict(score_half_res=float(chains[i][3]), start=seeds[i][1], role=roles[i],
                   start_centre_mm=[float(v) for v in seeds[i][3] * 1000], end_centre_mm=[float(v) for v in chains[i][1] * 1000],
                   **{k: v for k, v in pose_delta(R, t, chains[i][0], chains[i][1]).items()
                      if k in ("d_pos_mm", "d_pos_norm_mm", "d_rot_deg")},
                   d_q_rad=float(chains[i][2] - q)) for i in order]
    R, t, q, _ = fz.refine(R, t, q, 1.0, 1.0, step=0.3, maxfev=600)
    R, t, q, s = fz.refine(R, t, q, 1.0, 0.0, step=0.15, maxfev=400)  # polish on the unblurred agreement
    fz.log(f"[fit] done in {time.time() - t0:.1f}s, {fz.n_eval} renders: score {s:.4f} (start '{seeds[order[0]][1]}')")
    return dict(R=R, t=t, q=q, score=s, start=seeds[order[0]][1], runtime_s=time.time() - t0, n_eval=fz.n_eval,
                stage1=[(float(c[0]), c[1]) for c in res1], chains=spread, fitter=fz)


# ----------------------------------------------------------------------------- reporting helpers
def pose_delta(R_a, t_a, R_b, t_b):
    """Difference of two camera poses: centre offset [mm] (gripper frame), rotation angle [deg], optical-axis
    angle [deg], roll about the optical axis [deg]."""
    dR = R_a.T @ R_b
    ang = float(np.degrees(Rotation.from_matrix(dR).magnitude()))
    axis_ang = float(np.degrees(np.arccos(np.clip(R_a[:, 2] @ R_b[:, 2], -1, 1))))
    # roll: angle between the image-x axes after aligning the optical axes
    zb = R_b[:, 2]
    xa = R_a[:, 0] - zb * (R_a[:, 0] @ zb)
    roll = float(np.degrees(np.arctan2(np.cross(xa, R_b[:, 0]) @ zb, xa @ R_b[:, 0])))
    d = (np.asarray(t_b) - np.asarray(t_a)) * 1000
    return dict(d_pos_mm=[float(v) for v in d], d_pos_norm_mm=float(np.linalg.norm(d)), d_rot_deg=ang,
                d_optical_axis_deg=axis_ang, d_roll_deg=roll)


def describe(R, t):
    """Human-readable camera pose in the gripper frame (jaws along -z, moving jaw opens to +x)."""
    view = R[:, 2]
    tilt = float(np.degrees(np.arccos(np.clip(-view[2], -1, 1))))  # angle between view dir and the jaw direction (-z)
    return dict(centre_mm=[float(v) for v in np.asarray(t) * 1000], view_dir=[float(v) for v in view],
                angle_to_jaw_axis_deg=tilt, image_right=[float(v) for v in R[:, 0]], image_down=[float(v) for v in R[:, 1]])


def overlay_image(base, seg, renders, title=None):
    """base BGR + segmented contour (yellow) + rendered silhouettes [(mask, BGR color, text)]."""
    img = brighten(base)
    contour(img, seg, (0, 230, 255), 1)
    for i, (m, col, text) in enumerate(renders):
        contour(img, m, col, 2 if i == len(renders) - 1 else 1)
        label(img, text, 22 + 20 * (i + 1 if title else i), col)
    if title:
        label(img, title, 22)
    return img


# ----------------------------------------------------------------------------- CLI
def _real_both(K, dist, size, scratch):
    real = real_views(FRAMES, K, dist)
    real_sim = real_views(FRAMES, K, dist, K_out=k_sim(K, size))
    if scratch:
        scratch.mkdir(parents=True, exist_ok=True)
        for tag, r in (("K", real), ("Ksim", real_sim)):
            cv2.imwrite(str(scratch / f"real_median_und_{tag}.png"), r["median"])
            cv2.imwrite(str(scratch / f"real_mask_{tag}.png"), r["mask"].astype(np.uint8) * 255)
            cv2.imwrite(str(scratch / f"real_low_und_{tag}.png"), r["low"])
    return real, real_sim


def _profile_job(args):
    prob, K, size, R0, t0, q0, axis, d_mm = args
    fz = MountFitter(prob, K, size, verbose=False)
    t = t0 + R0[:, axis] * d_mm * 1e-3

    def f(x):
        return -fz.score(R0 @ Rotation.from_rotvec(np.radians(x[:3])).as_matrix(), t, q0 + np.radians(x[3]), 0.5, 1.0)
    x = np.zeros(4)
    for st in (np.array([1.5, 1.5, 1.5, 4.0]), np.array([0.3, 0.3, 0.3, 1.0])):
        x = minimize(f, x, method="Nelder-Mead", options=dict(initial_simplex=np.vstack([x, np.diag(st)]), xatol=0.005,
                                                              fatol=1e-6, maxfev=800, adaptive=True)).x
    d, _ = fz.diagnostics(R0 @ Rotation.from_rotvec(np.radians(x[:3])).as_matrix(), t, q0 + np.radians(x[3]))
    return dict(axis="xyz"[axis] + " (" + ("image right" if axis == 0 else "image down" if axis == 1 else "optical axis") + ")",
                axis_in_gripper_frame=[float(v) for v in R0[:, axis]], d_mm=float(d_mm), iou=d["iou"],
                boundary_mean_px=d.get("boundary_mean_px"), rot_refit_deg=float(np.linalg.norm(x[:3])), dq_deg=float(x[3]))


def cmd_fit(a):
    K, dist, size, c = camera_model()
    scratch = Path(a.scratch)
    real, real_sim = _real_both(K, dist, size, scratch)
    print(f"[seg] {real['n_frames']} frames, Otsu threshold {real['thr']:.1f}, jaw pixels {int(real['mask'].sum())}")
    R_nv, t_nv = env_to_cv(c.WRIST_CAM_POS, c.WRIST_CAM_QUAT)
    res = fit(real["prob"], K, size, R_nv, t_nv, verbose=True)
    fz = res["fitter"]
    R, t, q = res["R"], res["t"], res["q"]
    # NVIDIA reference: best jaw angle for NVIDIA's pose (pose fixed), so 'before' is not penalised for q
    qs = np.linspace(*fz.qlim, 40)
    q_nv = float(qs[int(np.argmax([fz.score(R_nv, t_nv, qq, 0.5, 1.0) for qq in qs]))])
    d_fit, lab_fit = fz.diagnostics(R, t, q)
    d_nv, lab_nv = fz.diagnostics(R_nv, t_nv, q_nv)
    d_nv_q, lab_nv_q = fz.diagnostics(R_nv, t_nv, q)
    pos, quat = cv_to_env(R, t)
    # sanity: round trip
    R_rt, t_rt = env_to_cv(pos, quat)
    assert np.allclose(R_rt, R, atol=1e-9) and np.allclose(t_rt, t, atol=1e-12)
    # how well is the pose determined? displace the camera centre along each camera axis by +-d mm, re-fit rotation +
    # jaw angle with the centre held, and see how much the match degrades (a profile of the objective)
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(real["prob"], K, size, R, t, q, ax, d) for ax in range(3) for d in a.profile_mm]
    with ProcessPoolExecutor(min(6, len(jobs))) as ex:
        prof = list(ex.map(_profile_job, jobs))
    for pr in prof:
        print(f"[profile] centre {pr['d_mm']:+.0f} mm along camera {pr['axis']}: IoU {pr['iou']:.4f}, boundary mean "
              f"{pr['boundary_mean_px']:.2f} px (rotation re-fit {pr['rot_refit_deg']:.2f} deg, jaw {pr['dq_deg']:+.2f} deg)")
    u = units()
    lr = u.sim_to_lerobot(np.r_[np.zeros(5), q])[5]
    delta = pose_delta(R_nv, t_nv, R, t)
    # option (c) of apply_requirements: rotation that makes Isaac's default K_sim render approximate the K image
    Ks = k_sim(K, size)
    M, comp_mean, comp_max = ksim_compensation(K, size)
    R_c = R @ M
    pos_c, quat_c = cv_to_env(R_c, t)
    d_comp, _ = fz.diagnostics(R_c, t, q, K=Ks)  # K_sim render vs the K-undistorted real segmentation
    d_plain_ksim, _ = fz.diagnostics(R, t, q, K=Ks)  # uncompensated, same comparison (the mismatch)
    print(f"[ksim] compensated quat {np.round(quat_c, 6).tolist()} (rotation {np.degrees(Rotation.from_matrix(M).magnitude()):.3f} deg): "
          f"K_sim render vs real(K) IoU {d_plain_ksim['iou']:.4f} -> {d_comp['iou']:.4f}; ray residual over the image "
          f"mean {comp_mean:.2f} px, max {comp_max:.2f} px")
    out = {
        "WRIST_CAM_POS": [round(v, 6) for v in pos],
        "WRIST_CAM_QUAT": [round(v, 6) for v in quat],
        "convention": "env_cfg CameraCfg.OffsetCfg(pos, rot, convention='opengl') under {ENV_REGEX_NS}/Robot/gripper/wrist_cam: "
                      "camera prim pose in the USD 'gripper' prim frame (== URDF gripper_link), camera looks along -z, +y up, quat xyzw",
        "jaw_angle_rad": round(q, 4),
        "jaw_angle_lerobot": round(float(lr), 2),
        "jaw_angle_note": "gripper angle while the 24 intr_wrist frames were taken (fitted); LeRobot 0..100 via so101_r2s2r/units.py (linear map, itself unverified)",
        "nvidia_pose": {"WRIST_CAM_POS": list(c.WRIST_CAM_POS), "WRIST_CAM_QUAT": list(c.WRIST_CAM_QUAT),
                        "best_jaw_angle_rad": round(q_nv, 4)},
        "difference_vs_nvidia": delta,
        "camera_description": {"fitted": describe(R, t), "nvidia": describe(R_nv, t_nv)},
        "fit_numpy": {"fitted": d_fit, "nvidia_at_its_best_q": d_nv, "nvidia_at_fitted_q": d_nv_q,
                      "score_soft_iou": res["score"], "start": res["start"], "runtime_s": res["runtime_s"],
                      "n_renders": res["n_eval"], "stage1_ranking": res["stage1"],
                      "basin_hopping_chains_vs_best": res["chains"], "profile_centre_displacement": prof},
        "segmentation": {"frames": str(FRAMES), "n_frames": real["n_frames"], "method": "25th-percentile grey over frames, undistorted, hysteresis threshold (Otsu / 0.35 x Otsu), largest components",
                         "otsu_threshold": real["thr"], "jaw_pixels": int(real["mask"].sum())},
        "intrinsics": {"K": K.tolist(), "dist": dist.tolist(), "K_isaac_render": Ks.tolist(),
                       "note": "K_isaac_render = what PinholeCameraCfg.from_intrinsic_matrix(K) renders in Isaac (mean f, "
                               "centred principal point); see apply_requirements 'wrist_intrinsics'"},
        "WRIST_CAM_QUAT_KSIM_COMPENSATED": [round(v, 6) for v in quat_c],
        "ksim_compensated": {
            "note": "Option (c) only: use with WRIST_CAM_POS when Isaac renders K_isaac_render (default spawn) but real frames "
                    "are undistorted to K. The K -> K_sim difference (principal point shift, mean focal length) is not a "
                    "rotation; this is its best rotation approximation over the whole image.",
            "WRIST_CAM_POS": [round(v, 6) for v in pos_c], "WRIST_CAM_QUAT": [round(v, 6) for v in quat_c],
            "rotation_deg": float(np.degrees(Rotation.from_matrix(M).magnitude())),
            "residual_px_mean": comp_mean, "residual_px_max": comp_max,
            "iou_numpy_vs_real_undistorted_to_K": d_comp["iou"],
            "boundary_mean_px_numpy_vs_real_undistorted_to_K": d_comp.get("boundary_mean_px"),
            "uncompensated_iou_numpy_vs_real_undistorted_to_K": d_plain_ksim["iou"]},
        "T_usd_urdf": T_USD_URDF.tolist(),
    }
    old = json.loads(PROPOSAL.read_text()) if PROPOSAL.exists() else {}
    if "frame_check" in old:  # written by scripts/check_wrist_mount.py; independent of the fitted pose
        out["frame_check"] = old["frame_check"]
    for key in ("isaac_validation", "isaac_validation_stale"):
        iv = old.get(key)
        if not iv:
            continue
        ev = iv.get("evaluated", {}).get("fitted", {})
        jaw = ev.get("jaw_angle_rad", (iv.get("arm_pose_rad") or [None] * 6)[5])
        same = (list(ev.get("pos", [])) == out["WRIST_CAM_POS"] and list(ev.get("quat", [])) == out["WRIST_CAM_QUAT"]
                and jaw is not None and abs(float(jaw) - out["jaw_angle_rad"]) < 1e-9)
        if same:  # Isaac results still describe this pose
            out["isaac_validation"] = iv
            break
        out["isaac_validation_stale"] = dict(iv, stale_note="Isaac results for a different pose / jaw angle than this "
                                                            "proposal; re-run scripts/check_wrist_mount.py")
        print("[out] the Isaac validation in the old proposal is for a different pose -> kept as 'isaac_validation_stale'")
        break
    out["apply_requirements"] = apply_requirements(out)
    PROPOSAL.write_text(json.dumps(out, indent=1))
    print(f"[out] wrote {PROPOSAL}")
    # overlays: real median (K-undistorted) + segmentation + NVIDIA + fitted silhouettes
    seg = real["mask"]
    nv = overlay_image(real["median"], seg, [(lab_nv > 0, (0, 0, 255), f"NVIDIA pose (q={q_nv:.2f}) IoU {d_nv['iou']:.3f}")],
                       title="real (yellow) vs NVIDIA mount")
    ft = overlay_image(real["median"], seg, [(lab_fit > 0, (0, 255, 0), f"fitted pose (q={q:.2f}) IoU {d_fit['iou']:.3f}")],
                       title="real (yellow) vs fitted mount")
    # colour-coded render of the fitted pose (body / servo / moving jaw) next to the real image
    col = np.zeros((*lab_fit.shape, 3), np.uint8)
    col[lab_fit == PRINTED] = (200, 200, 200)
    col[lab_fit == SERVO] = (80, 80, 80)
    col[lab_fit == JAW] = (230, 180, 120)
    label(col, "fitted render: body+fixed jaw / servo / moving jaw")
    sbs = np.vstack([np.hstack([nv, ft]), np.hstack([brighten(real["median"]), col])])
    cv2.imwrite(str(HERE / "data" / "wrist_mount_fit_overlay.jpg"), sbs, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(scratch / "fit_overlay.png"), sbs)
    print(f"[out] wrote {HERE / 'data' / 'wrist_mount_fit_overlay.jpg'}")
    print(json.dumps({k: out[k] for k in ("WRIST_CAM_POS", "WRIST_CAM_QUAT", "jaw_angle_rad", "jaw_angle_lerobot", "difference_vs_nvidia")}, indent=1))
    print(f"[out] numpy IoU: NVIDIA {d_nv['iou']:.3f} (q={q_nv:.2f}) -> fitted {d_fit['iou']:.3f}; "
          f"boundary mean {d_fit.get('boundary_mean_px', float('nan')):.2f} px, p90 {d_fit.get('boundary_p90_px', float('nan')):.2f} px")


def cmd_show(a):
    K, dist, size, c = camera_model()
    scratch = Path(a.scratch)
    real, _ = _real_both(K, dist, size, scratch)
    R, t = env_to_cv(a.pos, a.quat)
    fz = MountFitter(real["prob"], K, size, verbose=False)
    d, lab = fz.diagnostics(R, t, a.jaw)
    print(json.dumps(d, indent=1))
    img = overlay_image(real["median"], real["mask"], [(lab > 0, (0, 255, 0), f"pose IoU {d['iou']:.3f}")])
    cv2.imwrite(str(scratch / "show_overlay.png"), img)
    print("wrote", scratch / "show_overlay.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--scratch", default="/tmp/so101_wrist_mount", help="intermediate images (real median/mask PNGs)")
    f.add_argument("--profile_mm", type=float, nargs="*", default=[-3.0, 3.0],
                   help="camera-centre displacements [mm] for the determination profile")
    s = sub.add_parser("show")
    s.add_argument("--scratch", default="/tmp/so101_wrist_mount")
    s.add_argument("--pos", type=float, nargs=3, required=True)
    s.add_argument("--quat", type=float, nargs=4, required=True, help="xyzw, OpenGL camera (env_cfg convention)")
    s.add_argument("--jaw", type=float, default=0.0)
    a = ap.parse_args()
    {"fit": cmd_fit, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()

"""Marker-free pose of the fixed overhead camera, with the SO-101 itself as the calibration target.

For known joint angles the robot's silhouette is rendered from its URDF meshes for a candidate camera
pose, and the 6-DoF pose is optimised so the rendered silhouettes match the robot segmented in the
images -- jointly over several snapshots (different arm poses).

  RobotModel          full URDF tree (incl. the moving jaw) + binary-STL meshes welded/decimated by vertex
                      clustering (<= ~20k triangles), FK for every link, world-frame meshes
  project             numpy pinhole + OpenCV distortion model (k1 k2 p1 p2 k3 [k4 k5 k6]); == cv2.projectPoints
  rasterize           vectorised z-buffered triangle rasteriser -> label image (0 bg, 1 printed part, 2 servo)
  segment             robot-likeness maps from a BGR image: white-balance (the real room light is orange), then
                      robot = low saturation relative to the desk (works for the light-tan sim desk and the
                      dark-brown real one); among those, bright = printed part, dark = servo (Otsu split)
  solve_camera_pose   coarse search (nadir camera: yaw grid x height grid, image shift by FFT correlation, summed
                      over snapshots, scored by a local soft IoU on the parts that MOVE between snapshots, so static
                      white clutter cannot attract it; when the arm poses are too similar for that -- pose_diversity
                      < 0.3 -- the whole segmentation is searched as well) + Nelder-Mead over (rotation, position) on
                      a signed-distance agreement score minus unexplained robot-like pixels next to the render,
                      coarse-to-fine, candidates ranked by silhouette IoU; returns pose + per-snapshot IoU and
                      boundary residual [px]
  solve_pipeline      the above + a second pass with ROI-local segmentation statistics (what solve.py scene_robot
                      runs); verdict() turns its diagnostics into PASS / WARN
  fit_joint_offsets   pose + constant shoulder_lift / elbow_flex / wrist_flex reading offsets (LeRobot zero vs URDF
                      zero) jointly, Gaussian prior; joint_offset_check() flags readings that look offset. A
                      shoulder_pan offset is NOT observable (it equals a camera yaw about the base up to the base's
                      own silhouette)

Camera convention (same as solve.py cmd_scene and the sim, convention='ros'): R_world_cam has the camera axes
as columns (x right, y down, z = optical axis); pos = camera centre in the desk/world frame [m];
quat = Rotation.from_matrix(R_world_cam).as_quat() (xyzw).
World frame = desk frame: URDF base_link + fk.T_WORLD_URDF (rotation identity, fitted against Isaac).

numpy + opencv + scipy only (no Isaac). Run with ~/TestingGrounds/robot_splat/.venv/bin/python.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import fk  # noqa: E402

URDF = fk.URDF
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
T_WORLD_URDF = np.asarray(fk.T_WORLD_URDF, float)  # base_link -> desk frame: translation only
BG, PRINTED, SERVO = 0, 1, 2
IMAGE_SIZE = (640, 480)
PLACEHOLDER_K = np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1]])  # so101_r2s2r/calib.py SCENE_K


# ----------------------------------------------------------------------------- small math helpers
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


def nadir_rotation(yaw):
    """Camera looking straight down (-z), image-up = world direction (cos yaw, sin yaw, 0)."""
    up = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = -up
    return np.stack([np.cross(y, z), y, z], axis=1)


def look_at(eye, target, up=(1.0, 0.0, 0.0)):
    """R_world_cam (ROS/OpenCV axes) of a camera at eye looking at target, `up` at the top of the image."""
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    y = -np.asarray(up, float)
    y = y - z * (y @ z)
    y /= np.linalg.norm(y)
    return np.stack([np.cross(y, z), y, z], axis=1)


def yaw_of(R):
    """Yaw of the image-up direction (-camera y) projected on the desk plane [rad]."""
    return float(np.arctan2(-R[1, 1], -R[0, 1]))


def rot_angle_deg(Ra, Rb):
    return float(np.degrees(Rotation.from_matrix(Ra.T @ Rb).magnitude()))


def _sim_units():
    spec = importlib.util.spec_from_file_location("so101_units", HERE.parent / "so101_r2s2r" / "units.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def lerobot_to_sim(v):
    """LeRobot units (deg x5, gripper 0..100) -> sim radians (so101_r2s2r/units.py)."""
    return np.asarray(_sim_units().lerobot_to_sim(np.asarray(v, float)), float)


def q_from_follower(follower) -> np.ndarray:
    """state.json["follower"] (dict keyed 'shoulder_pan' or 'shoulder_pan.pos', or a list) -> 6 sim angles [rad]."""
    if isinstance(follower, dict):
        v = [follower[j] if j in follower else follower[f"{j}.pos"] for j in JOINTS]
    else:
        v = list(follower)[:6]
    return lerobot_to_sim(v)


# ----------------------------------------------------------------------------- meshes
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


class RobotModel:
    """SO-101 URDF: kinematic tree, decimated visual meshes per link (labelled printed / servo), FK."""

    def __init__(self, urdf=URDF, max_tris: int | None = 20000, t_world_urdf=T_WORLD_URDF):
        urdf = Path(urdf)
        root = ET.parse(urdf).getroot()
        self.t_world_urdf = np.asarray(t_world_urdf, float)
        self.joints = {}  # child link -> joint
        for j in root.findall("joint"):  # top level only (transmissions repeat joint names)
            o = j.find("origin")
            xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            ax = j.find("axis")
            axis = [float(v) for v in ax.get("xyz").split()] if ax is not None else [1, 0, 0]
            self.joints[j.find("child").get("link")] = dict(name=j.get("name"), type=j.get("type"), axis=axis,
                                                            parent=j.find("parent").get("link"), T0=tf(rpy_matrix(*rpy), xyz))
        links = [ln.get("name") for ln in root.findall("link")]
        self.root = next(ln for ln in links if ln not in self.joints)
        self.order = [self.root]
        while len(self.order) < len(links):
            grown = [c for c, j in self.joints.items() if j["parent"] in self.order and c not in self.order]
            if not grown:
                break
            self.order += grown
        visuals = []  # (link, file, T_link_mesh, scale, label)
        for ln in root.findall("link"):
            for vis in ln.findall("visual"):
                m = vis.find("geometry/mesh")
                if m is None:
                    continue
                o = vis.find("origin")
                xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0, 0, 0]
                rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
                scale = np.array([float(v) for v in m.get("scale", "1 1 1").split()])
                fn = m.get("filename").replace("package://", "")
                mat = vis.find("material")
                servo = Path(fn).stem.startswith("sts3215") or (mat is not None and mat.get("name") == "sts3215")
                visuals.append((ln.get("name"), urdf.parent / fn, tf(rpy_matrix(*rpy), xyz), scale, SERVO if servo else PRINTED))
        soups = {f: load_stl(f) for f in {v[1] for v in visuals}}
        if max_tris is None:  # full-resolution triangle soups (ground-truth renders in the tests)
            cell = 0.0
            dec = {f: (s.reshape(-1, 3), np.arange(3 * len(s), dtype=np.int32).reshape(-1, 3)) for f, s in soups.items()}
            total = sum(len(dec[v[1]][1]) for v in visuals)
        else:
            cell = 0.002
        while max_tris is not None:  # coarsen the clustering grid until the instanced triangle count fits the budget
            dec = {f: cluster_mesh(s, cell) for f, s in soups.items()}
            total = sum(len(dec[v[1]][1]) for v in visuals)
            if total <= max_tris:
                break
            cell *= 1.15
        self.cell, self.n_tris = cell, total
        self.n_tris_full = sum(len(soups[v[1]]) for v in visuals)
        per_link = {}
        for link, f, T, scale, lab in visuals:
            V, F = dec[f]
            per_link.setdefault(link, []).append(((V * scale) @ T[:3, :3].T + T[:3, 3], F, lab))
        self.link_meshes = {}
        for link, parts in per_link.items():
            Vs, Fs, Ls, off = [], [], [], 0
            for V, F, lab in parts:
                Vs.append(V); Fs.append(F + off); Ls.append(np.full(len(F), lab, np.uint8)); off += len(V)
            self.link_meshes[link] = (np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ls))

    def link_transforms(self, q) -> dict:
        """World (desk-frame) 4x4 pose of every link for 6 sim joint angles [rad] (JOINTS order) or a dict."""
        qd = dict(q) if isinstance(q, dict) else dict(zip(JOINTS, np.asarray(q, float).ravel()))
        T = {self.root: tf(np.eye(3), self.t_world_urdf)}
        for link in self.order[1:]:
            j = self.joints[link]
            A = T[j["parent"]] @ j["T0"]
            a = float(qd.get(j["name"], 0.0))
            if j["type"] in ("revolute", "continuous"):
                A = A @ tf(axis_angle(j["axis"], a))
            elif j["type"] == "prismatic":
                A = A @ tf(np.eye(3), a * np.asarray(j["axis"], float))
            T[link] = A
        return T

    def world_mesh(self, q):
        """(V (n,3) desk frame, F (m,3), label (m,)) of the whole robot at joint angles q."""
        T = self.link_transforms(q)
        Vs, Fs, Ls, off = [], [], [], 0
        for link, (V, F, L) in self.link_meshes.items():
            M = T[link]
            Vs.append(V @ M[:3, :3].T + M[:3, 3]); Fs.append(F + off); Ls.append(L); off += len(V)
        return np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ls)


@lru_cache(maxsize=4)
def get_model(urdf: str = str(URDF), max_tris: int = 20000) -> RobotModel:
    return RobotModel(urdf, max_tris)


# ----------------------------------------------------------------------------- projection + rasterisation
def project(X, R, pos, K, dist=None):
    """World points -> pixel coords (n,2) and camera depth (n,), OpenCV model (same as cv2.projectPoints).
    Supports 4, 5 or 8 distortion coefficients (k1 k2 p1 p2 [k3 [k4 k5 k6]]); the thin-prism (s1..s4) and tilt
    (tauX, tauY) terms of 12/14-coefficient models are not implemented -> ValueError if any of them is nonzero."""
    Xc = (np.asarray(X, float) - np.asarray(pos, float)) @ np.asarray(R, float)
    z = Xc[:, 2]
    zs = np.where(z > 1e-6, z, 1e-6)
    x, y = Xc[:, 0] / zs, Xc[:, 1] / zs
    if dist is not None and np.any(np.asarray(dist, float) != 0):
        dd = np.asarray(dist, float).ravel()
        if np.any(dd[8:] != 0):
            raise ValueError(f"project(): {len(dd)} distortion coefficients with nonzero thin-prism/tilt terms "
                             f"{dd[8:].tolist()} are not supported (calibrate with the default 5 or CALIB_RATIONAL_MODEL 8)")
        d = np.zeros(8)
        dd = dd[:8]
        d[:len(dd)] = dd
        k1, k2, p1, p2, k3, k4, k5, k6 = d
        r2 = x * x + y * y
        rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        x, y = (x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x),
                y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)
    u = K[0, 0] * x + K[0, 1] * y + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    return np.stack([u, v], 1), z


def _ramp(n):
    """concatenate(arange(k) for k in n) without a python loop."""
    tot = int(n.sum())
    return np.arange(tot) - np.repeat(np.cumsum(n) - n, n)


def rasterize(x, y, z, lab, size, labels=True):
    """Rasterise triangles given per-vertex pixel coords x, y and depth z (all (m,3)) and label (m,).

    Vectorised scanline spans (work ~ covered pixels). A pixel is covered when its centre (integer coords,
    OpenCV convention) lies in a triangle (edges inclusive). Returns a uint8 label image (nearest surface wins,
    per-pixel z-buffer) or, with labels=False, a bool mask.
    """
    W, H = size
    o = np.argsort(y, axis=1)  # vertices sorted top -> bottom
    x, y, z = np.take_along_axis(x, o, 1), np.take_along_axis(y, o, 1), np.take_along_axis(z, o, 1)
    xt, xm, xb, yt, ym, yb = x[:, 0], x[:, 1], x[:, 2], y[:, 0], y[:, 1], y[:, 2]
    area = (xm - xt) * (yb - yt) - (xb - xt) * (ym - yt)
    ylo = np.maximum(np.ceil(yt), 0)
    yhi = np.minimum(np.floor(yb), H - 1)
    keep = np.nonzero((yhi >= ylo) & (np.abs(area) > 1e-9) & (np.maximum(np.maximum(xt, xm), xb) >= 0)
                      & (np.minimum(np.minimum(xt, xm), xb) <= W - 1))[0]
    xt, xm, xb, yt, ym, yb, area, ylo, yhi = (a[keep] for a in (xt, xm, xb, yt, ym, yb, area, ylo, yhi))
    s_long = (xb - xt) / (yb - yt)  # yb > yt since area != 0
    s_up = np.where(ym > yt, (xm - xt) / np.where(ym > yt, ym - yt, 1.0), 0.0)
    s_low = np.where(yb > ym, (xb - xm) / np.where(yb > ym, yb - ym, 1.0), 0.0)
    h = (yhi - ylo + 1).astype(np.int64)
    tri = np.repeat(np.arange(len(h)), h)  # one entry per (triangle, pixel row)
    r = ylo[tri] + _ramp(h)
    ytt = yt[tri]
    x_long = xt[tri] + (r - ytt) * s_long[tri]
    ymt = ym[tri]
    x_short = np.where(r < ymt, xt[tri] + (r - ytt) * s_up[tri], xm[tri] + (r - ymt) * s_low[tri])
    # +-1e-6 px: a pixel centre on an edge shared by two triangles is computed from different vertex pairs in
    # each and could otherwise be missed by both (pinholes inside the silhouette)
    c0 = np.maximum(np.ceil(np.minimum(x_long, x_short) - 1e-6), 0)
    c1 = np.minimum(np.floor(np.maximum(x_long, x_short) + 1e-6), W - 1)
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
    A = (dz1 * dy2 - dz2 * dy1) / area  # triangle's screen-space depth plane z = C + A u + B v
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


def render(V, F, L, R, pos, K, dist, size, labels=True, near=0.02, cull=True):
    """Label image (0 bg, 1 printed, 2 servo) -- or a bool silhouette -- of a world-frame mesh.

    cull: drop back faces (the STL meshes are closed and consistently wound; front faces have negative signed
    area in image coordinates). The silhouette is identical either way; culling halves the work.
    """
    uv, z = project(V, R, pos, K, dist)
    u, v = uv[:, 0], uv[:, 1]
    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    ok = (z[a] > near) & (z[b] > near) & (z[c] > near)
    if cull:
        ok &= (u[b] - u[a]) * (v[c] - v[a]) - (u[c] - u[a]) * (v[b] - v[a]) < 0
    Ft = F[ok]
    return rasterize(u[Ft], v[Ft], z[Ft], L[ok], size, labels)


# ----------------------------------------------------------------------------- segmentation
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _otsu(v, bins=64):
    h, e = np.histogram(v, bins=bins, range=(0.0, 1.0))
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    return float(c[np.argmax(w0 * w1 * (m0 - m1) ** 2)])


def estimate_illuminant(img, roi=None, pct=99.0, e_min=0.35):
    """Colour of the light as BGR gains (max channel = 1): mean colour of the brightest unclipped pixels among
    those less saturated than the median pixel (white printed parts, cup rims, specular glints) -- a white-patch
    estimate restricted to near-neutral surfaces. Clamped at e_min so one bright coloured object cannot push a
    channel towards zero. The real room is lit by warm bulbs: white PLA reads ~(0.65, 0.92, 1.0) BGR there."""
    f = img.astype(np.float32) / 255.0
    V = f.max(2)
    S = (V - f.min(2)) / np.maximum(V, 1e-3)
    ok = (V < 0.97) & (V > 0.05)
    if roi is not None:
        ok &= roi
    if ok.sum() < 200:
        return np.ones(3, np.float32)
    ok &= S <= np.median(S[ok])
    sel = ok & (V >= np.percentile(V[ok], pct))
    e = f[sel].mean(0)
    return np.maximum(e / max(float(e.max()), 1e-6), e_min).astype(np.float32)


def segment(img, roi=None, s_rel=0.5, s_max=0.5, c_min=0.035, tau=0.25, white_balance=True):
    """Robot-likeness of every pixel of a BGR image (sim or real overhead view).

    The arm is achromatic (white printed parts, black/dark-grey servos) while the desk is coloured (light tan in
    the sim, dark brown wood in reality). The image is first white-balanced with estimate_illuminant() (the real
    room light is strongly orange: uncorrected, the white parts have a saturation of ~0.4-0.7). A pixel is then
    'robot' when its chroma (max-min) is well below the desk's saturation times its brightness, with an absolute
    floor for dark, noisy pixels. Among robot pixels an Otsu split on brightness separates printed parts (bright)
    from servos (dark). `roi` (bool HxW) restricts the desk/illuminant statistics. Returns dict(white, dark,
    robot (float HxW in 0..1), and the statistics used).
    """
    f = img.astype(np.float32) / 255.0
    illum = estimate_illuminant(img, roi) if white_balance else np.ones(3, np.float32)
    if white_balance:
        f = np.minimum(f / illum, 1.0)
    V = f.max(2)
    C = V - f.min(2)
    S = C / np.maximum(V, 1e-3)
    reg = np.ones(V.shape, bool) if roi is None else roi
    s_bg = float(np.median(S[reg]))
    s_thr = min(s_rel * s_bg, s_max)
    thr = np.maximum(s_thr * V, c_min)
    p_ach = _sigmoid((thr - C) / (tau * thr))
    ach = (C < thr) & reg
    v_split = _otsu(V[ach]) if ach.sum() > 100 else 0.5
    p_bright = _sigmoid((V - v_split) / 0.04)
    white = (p_ach * p_bright).astype(np.float32)
    dark = (p_ach * (1 - p_bright)).astype(np.float32)
    return dict(white=white, dark=dark, robot=white + dark, s_bg=s_bg, s_thr=s_thr, v_split=v_split,
                v_bg=float(np.median(V[reg])), illum=illum)


def signed_distance(mask):
    """Signed Euclidean distance [px] to the boundary of a bool mask: > 0 inside, < 0 outside, +-0.5 on the
    pixels next to the boundary."""
    m = mask.astype(np.uint8)
    if not m.any():
        return np.full(m.shape, -1e3, np.float32)
    if m.all():
        return np.full(m.shape, 1e3, np.float32)
    din = cv2.distanceTransform(m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dout = cv2.distanceTransform(1 - m, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return np.where(m > 0, din - 0.5, 0.5 - dout).astype(np.float32)


# ----------------------------------------------------------------------------- solver
@dataclass
class Snapshot:
    image: np.ndarray  # BGR uint8 (overhead camera)
    q: np.ndarray  # 6 sim joint angles [rad] (JOINTS order; gripper incl.)
    name: str = ""


@dataclass
class _Level:
    scale: float
    sigma: float
    K: np.ndarray
    size: tuple
    A_p: list = field(default_factory=list)  # agreement map for pixels rendered as printed part
    A_s: list = field(default_factory=list)  # ... rendered as servo
    B: list = field(default_factory=list)  # robot-like target mass that counts against a render in its band
    kern: np.ndarray = None  # band dilation kernel at this level
    norm: float = 1.0


MODES = ("class", "union", "white")


def scale_K(K, s):
    """Intrinsics of the image resized by factor s (cv2.resize pixel-centre convention)."""
    Ks = np.array(K, float).copy()
    Ks[0, :] *= s
    Ks[1, :] *= s
    Ks[0, 2] = (K[0, 2] + 0.5) * s - 0.5
    Ks[1, 2] = (K[1, 2] + 0.5) * s - 0.5
    return Ks


def _rot_between(b, a):
    """Smallest rotation taking direction b onto direction a."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(b, a), float(b @ a)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3)
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


class CameraSolver:
    """Joint silhouette alignment of the overhead camera over several snapshots.

    mode: 'class' -- printed parts must land on bright achromatic pixels and servos on dark achromatic ones
                     (uses the printed/servo internal edges; best when the servos contrast with the desk, e.g. sim)
          'union' -- whole robot vs achromatic pixels
          'white' -- only printed parts are matched (servos: 'not white'); for a dark desk where black servos
                     and shadows are indistinguishable
    roi:  bool HxW mask of where the robot may be (None = whole image); outside it the robot is penalised.
    """

    # pose parametrisation units: roll [deg], tilt x/y [deg], lateral [m], along the optical axis [m]
    UNITS = np.array([1.0, 2.0, 2.0, 0.005, 0.005, 0.03])

    def __init__(self, snaps, K=None, dist=None, mode="class", roi=None, model=None, coarse_model=None,
                 verbose=True, threads=3, agreement="sdf", band_px=8.0, band_weight=1.0, dynamic_coarse=True,
                 min_diversity=0.3):
        assert mode in MODES, mode
        assert agreement in ("sdf", "blur"), agreement
        self.agreement = agreement
        self.band_px, self.band_weight = band_px, band_weight  # [full-res px], weight of the band term
        self.dynamic_coarse, self.min_diversity = dynamic_coarse, min_diversity
        self.snaps = list(snaps)
        self.K = np.array(PLACEHOLDER_K if K is None else K, float).reshape(3, 3)
        self.dist = None if dist is None else np.asarray(dist, float).ravel()
        self.mode, self.verbose = mode, verbose
        H, W = self.snaps[0].image.shape[:2]
        self.size = (W, H)
        self.roi = roi
        self.model = model or get_model(max_tris=20000)
        self.coarse_model = coarse_model or get_model(max_tris=6000)
        self.meshes = [self.model.world_mesh(s.q) for s in self.snaps]
        self.coarse_meshes = [self.coarse_model.world_mesh(s.q) for s in self.snaps]
        self.seg = [segment(s.image, roi) for s in self.snaps]
        allV = np.concatenate([m[0] for m in self.meshes])
        self.pivot = allV.mean(0)  # robot centre (desk frame), the orbit point of the refinement
        self.q_off = np.zeros(6)  # joint-reading offsets [rad] applied to every snapshot (fit_joint_offsets)
        self._levels = {}
        self.diversity = None  # pose_diversity() seen by the last coarse search
        self.n_eval = 0
        self.pool = ThreadPoolExecutor(threads) if threads and threads > 1 and len(self.snaps) > 1 else None

    def log(self, *a):
        if self.verbose:
            print(*a, flush=True)

    # --- agreement maps
    def level(self, scale, sigma):
        key = (scale, sigma)
        if key in self._levels:
            return self._levels[key]
        W, H = self.size
        size = (int(round(W * scale)), int(round(H * scale)))
        lv = _Level(scale, sigma, scale_K(self.K, scale), size)
        roi = None if self.roi is None else cv2.resize(self.roi.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.5
        norm = 0.0
        for sg in self.seg:
            rs = lambda m: cv2.resize(m, size, interpolation=cv2.INTER_AREA) if scale != 1 else m  # noqa: E731
            white, dark = rs(sg["white"]), rs(sg["dark"])
            robot = white + dark
            if self.agreement == "sdf":
                # tanh of the signed distance [level px] to the segmented boundary, softness sigma: 0 exactly on the
                # boundary, so blurring never shrinks thin parts (a 2*blur(mask)-1 map goes negative inside limbs
                # thinner than the blur and then favours a too-small robot, i.e. a camera too high)
                soft = lambda m: np.tanh(rs(signed_distance(m > 0.5)) * scale / max(sigma, 0.5)).astype(np.float32)  # noqa: E731
                if self.mode == "class":
                    ap, as_ = soft(sg["white"]), soft(sg["dark"])
                elif self.mode == "union":
                    ap = as_ = soft(sg["robot"])
                else:
                    ap = soft(sg["white"])
                    as_ = -np.maximum(ap, 0.0)
            else:
                if self.mode == "class":
                    ap, as_ = 2 * white - 1, 2 * dark - 1
                elif self.mode == "union":
                    ap = as_ = 2 * robot - 1
                else:
                    ap, as_ = 2 * white - 1, -white
                blur = lambda m: cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma) if sigma > 0 else m.astype(np.float32)  # noqa: E731
                ap, as_ = blur(ap), blur(as_)
            # band term: robot-like evidence right next to the rendered robot is unexplained -> penalised
            b = np.maximum(ap, 0.0) if self.mode != "class" else np.maximum(np.maximum(ap, 0.0), np.maximum(as_, 0.0))
            if roi is not None:
                ap[~roi] = -0.5
                as_[~roi] = -0.5
                b[~roi] = 0.0
            lv.A_p.append(ap)
            lv.A_s.append(as_)
            lv.B.append(b.astype(np.float32))
            norm += float((white if self.mode == "white" else robot)[roi if roi is not None else slice(None)].sum())
        r = max(1, int(round(self.band_px * scale)))
        lv.kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        lv.norm = max(norm, 1.0)
        self._levels[key] = lv
        return lv

    def _score1(self, args):
        (V, F, L), ap, as_, b, R, pos, lv = args
        lab = render(V, F, L, R, pos, lv.K, self.dist, lv.size, labels=self.mode != "union")
        if self.mode == "union":
            s, robot = float(ap[lab].sum()), lab
        else:
            s, robot = float(ap[lab == PRINTED].sum() + as_[lab == SERVO].sum()), lab > 0
        if self.band_weight > 0:
            band = (cv2.dilate(robot.astype(np.uint8), lv.kern) > 0) & ~robot
            s -= self.band_weight * float(b[band].sum())
        return s

    def score(self, R, pos, lv, coarse_mesh=False, meshes=None):
        """Mean agreement (higher = better, ~1 for a perfect match) of the rendered robot over all snapshots:
        sum over rendered pixels of the agreement map, minus band_weight x robot-like evidence within band_px
        around the render (symmetric: a robot too small, or dropped into a large white clutter area, leaves
        segmented mass next to it). meshes: per-snapshot (V, F, L) to render instead of self.meshes."""
        if meshes is None:
            meshes = self.coarse_meshes if coarse_mesh else self.meshes
        jobs = [(m, ap, as_, b, R, pos, lv) for m, ap, as_, b in zip(meshes, lv.A_p, lv.A_s, lv.B)]
        parts = self.pool.map(self._score1, jobs) if self.pool else map(self._score1, jobs)
        self.n_eval += 1
        return sum(parts) / lv.norm

    # --- pose parametrisation around the pivot (orbit + camera-frame translation)
    def _pose(self, x, R0, pos0):
        d0 = R0.T @ (self.pivot - pos0)  # pivot in camera coordinates (kept fixed by the rotation part)
        p = np.asarray(x, float) * self.UNITS
        R = R0 @ Rotation.from_rotvec(np.radians([p[1], p[2], p[0]])).as_matrix()
        pos = self.pivot - R @ d0 + R @ p[3:]
        return R, pos

    def _penalty(self, R, pos):
        """0 for a plausible overhead camera; grows when it sinks to the robot's height or tilts towards the
        horizon (keeps Nelder-Mead from wandering off from a start with little silhouette overlap)."""
        return 10.0 * max(0.0, 0.25 - (pos[2] - self.pivot[2])) + 5.0 * max(0.0, 0.3 + R[2, 2])

    def refine(self, R0, pos0, scale, sigma, step=1.0, maxfev=300, coarse_mesh=False):
        lv = self.level(scale, sigma)

        def f(x):
            R, pos = self._pose(x, R0, pos0)
            pen = self._penalty(R, pos)
            return 1.0 + pen if pen > 0 else -self.score(R, pos, lv, coarse_mesh)  # scores are <= ~1

        x0 = np.zeros(6)
        sim = np.vstack([x0, np.eye(6) * step])
        res = minimize(f, x0, method="Nelder-Mead",
                       options=dict(initial_simplex=sim, xatol=0.03, fatol=2e-5, maxfev=maxfev, adaptive=True))
        R, pos = self._pose(res.x, R0, pos0)
        return R, pos, -res.fun

    # --- joint-reading offsets (LeRobot calibration zero vs URDF zero)
    def posed_meshes(self, off):
        """Per-snapshot world meshes with the joint readings shifted by `off` (6, rad, JOINTS order)."""
        fn = lambda s: self.model.world_mesh(s.q + off)  # noqa: E731
        return list(self.pool.map(fn, self.snaps) if self.pool else map(fn, self.snaps))

    def set_joint_offsets(self, off):
        """Apply constant joint-reading offsets [rad] (6, JOINTS order) to every snapshot (meshes re-posed)."""
        self.q_off = np.asarray(off, float).copy()
        self.meshes = self.posed_meshes(self.q_off)
        self.coarse_meshes = [self.coarse_model.world_mesh(s.q + self.q_off) for s in self.snaps]

    def fit_joint_offsets(self, R, pos, joints=(1, 2, 3), prior_deg=3.0, prior_weight=5e-4, t0=None, restarts=3,
                          tol=5e-4):
        """Refine the camera pose jointly with constant offsets [deg] of the given joints' readings (the same offset
        in every snapshot: what a LeRobot calibration zero that differs from the URDF zero does), from a solved pose.
        Each offset has a Gaussian prior N(0, prior_deg) (cost prior_weight per sigma^2, on the ~0..1 score scale).
        Default joints: shoulder_lift, elbow_flex, wrist_flex. shoulder_pan is left out by default: its offset is a
        rotation of everything but the base about the vertical base axis, i.e. nearly a camera yaw (only the base's
        silhouette tells them apart). Needs several clearly different arm poses (different pan angles and
        lift/elbow/wrist configurations), otherwise offsets and camera pose trade off against each other.
        Half resolution first, then full resolution with simplex restarts (as polish()). Applies the fitted offsets
        (set_joint_offsets) and returns the polish-style result dict + joint_offsets_deg (6, JOINTS order) and
        iou_mean_no_offsets / score_no_offsets (the given pose without offsets, for comparison)."""
        t0 = time.time() if t0 is None else t0
        joints = list(joints)
        lv_full = self.level(1.0, 1.0)
        base = self.q_off.copy()
        s_before = self.score(R, pos, lv_full)
        iou_before = self.mean_iou(R, pos)

        def objective(R0, p0, lv, off0):
            def f(x):
                Rx, px = self._pose(x[:6], R0, p0)
                pen = self._penalty(Rx, px)
                if pen > 0:
                    return 1.0 + pen
                o = off0.copy()
                o[joints] += np.radians(x[6:])
                d = np.degrees(o[joints] - base[joints])
                return -self.score(Rx, px, lv, meshes=self.posed_meshes(o)) + prior_weight * float(np.sum((d / prior_deg) ** 2))
            return f

        def run(R0, p0, off0, scale, sigma, pose_step, off_step, maxfev):
            f = objective(R0, p0, self.level(scale, sigma), off0)
            n = 6 + len(joints)
            x0 = np.zeros(n)
            sim = np.vstack([x0, np.diag(np.r_[np.full(6, pose_step), np.full(len(joints), off_step)])])
            r = minimize(f, x0, method="Nelder-Mead",
                         options=dict(initial_simplex=sim, xatol=0.03, fatol=2e-5, maxfev=maxfev, adaptive=True))
            Rn, pn = self._pose(r.x[:6], R0, p0)
            o = off0.copy()
            o[joints] += np.radians(r.x[6:])
            return Rn, pn, o, -r.fun

        Rj, pj, off, s = run(R, pos, base, 0.5, 1.0, 0.5, 1.5, 120 * (6 + len(joints)))
        Rj, pj, off, s = run(Rj, pj, off, 1.0, 1.0, 0.25, 0.5, 60 * (6 + len(joints)))
        for _ in range(restarts):
            R2, p2, o2, s2 = run(Rj, pj, off, 1.0, 1.0, 0.1, 0.25, 25 * (6 + len(joints)))
            gain = s2 - s
            if gain > 0:
                Rj, pj, off, s = R2, p2, o2, s2
            if gain < tol * max(abs(s), 1e-3):
                break
        self.set_joint_offsets(off)
        out = dict(R=Rj, pos=pj, quat=Rotation.from_matrix(Rj).as_quat(), score=s, start="joint_offsets",
                   runtime_s=time.time() - t0, n_eval=self.n_eval, pose_diversity=self.diversity,
                   joint_offsets_deg=np.degrees(off), offset_joints=[JOINTS[j] for j in joints],
                   score_no_offsets=s_before, iou_mean_no_offsets=iou_before, R_no_offsets=np.asarray(R, float),
                   pos_no_offsets=np.asarray(pos, float))
        out.update(self.diagnostics(Rj, pj))
        self.log(f"  joint offsets fitted in {out['runtime_s']:.1f}s: "
                 + ", ".join(f"{JOINTS[j]} {np.degrees(off[j]):+.2f}" for j in joints)
                 + f" deg; IoU {iou_before:.3f} -> {out['iou_mean']:.3f}")
        return out

    # --- coarse search: nadir camera, yaw x height grid, image shift by FFT correlation
    def pose_diversity(self, height=1.0):
        """Fraction of the rendered arm (nadir view from `height`, decimated mesh) that is NOT covered in every
        snapshot, mean over snapshots -- what the moving-parts coarse targets can keep: ~0.55-0.65 for well-spread
        arm poses, ~0.1 when the poses (nearly) repeat. Independent of clutter and segmentation."""
        if len(self.snaps) < 2:
            return 0.0
        R, pos = nadir_rotation(0.0), np.array([self.pivot[0], self.pivot[1], height])
        ren = [render(*m, R, pos, self.K, None, self.size, labels=False) for m in self.coarse_meshes]
        static = np.logical_and.reduce(ren)
        return float(np.mean([(r & ~static).sum() / max(int(r.sum()), 1) for r in ren]))

    def mean_iou(self, R, pos, near_px=20):
        """Mean silhouette IoU over the snapshots (as in diagnostics(), without the boundary distances)."""
        ious = []
        k = np.ones((2 * near_px + 1,) * 2, np.uint8)
        for (V, F, L), sg in zip(self.meshes, self.seg):
            lab = render(V, F, L, R, pos, self.K, self.dist, self.size, labels=self.mode == "white")
            r = (lab == PRINTED) if self.mode == "white" else lab > 0
            sm = (sg["white"] if self.mode == "white" else sg["robot"]) > 0.5
            if self.roi is not None:
                sm &= self.roi
            sm &= cv2.dilate(r.astype(np.uint8), k) > 0
            ious.append(float((r & sm).sum() / max(int((r | sm).sum()), 1)))
        return float(np.mean(ious))

    def _coarse_keys(self):
        return {"class": ("white", "dark"), "union": ("robot", None), "white": ("white", None)}[self.mode]

    def _coarse_maps(self, scale, sigma, dynamic=None):
        """Target maps the coarse search correlates with, per snapshot: (printed-target, servo-target or None,
        total segmented mass, unblurred matchable mass for the band term). With >= 3 snapshots the targets are
        'dynamic': robot-likeness minus its minimum over all snapshots, which removes everything static -- floor
        tiles, cup, paper, the other arm -- and keeps the moving links (the arm must be posed differently in each
        snapshot). The refinement then matches the full segmentation."""
        W, H = self.size
        size = (int(round(W * scale)), int(round(H * scale)))
        roi = None if self.roi is None else cv2.resize(self.roi.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.5
        keys = self._coarse_keys()
        src = [[sg[k] if k else None for k in keys] for sg in self.seg]
        dynamic = self.dynamic_coarse if dynamic is None else dynamic
        if dynamic and len(self.seg) >= 3:
            for j, k in enumerate(keys):
                if k:
                    static = np.min([sg[k] for sg in self.seg], axis=0)
                    for s_ in src:
                        s_[j] = np.clip(s_[j] - static, 0.0, 1.0)
        rs = lambda m: cv2.resize(m, size, interpolation=cv2.INTER_AREA).astype(np.float32)  # noqa: E731
        rmask = 1.0 if roi is None else roi
        out = []
        for fp, fs in src:
            tp, ts = rs(fp) * rmask, None if fs is None else rs(fs) * rmask
            mass = float(tp.sum() + (0 if ts is None else ts.sum()))
            if self.agreement == "sdf":  # 0.5 on the segmented boundary, ~1 inside even for thin parts, -> 0 outside
                soft = lambda m: (0.5 + 0.5 * np.tanh(rs(signed_distance(m > 0.5)) * scale / sigma)) * rmask  # noqa: E731
                sp, ss_ = soft(fp), None if fs is None else soft(fs)
            else:
                blur = lambda m: cv2.GaussianBlur(m, (0, 0), sigma)  # noqa: E731
                sp, ss_ = blur(tp), None if ts is None else blur(ts)
            out.append((sp.astype(np.float32), None if ss_ is None else ss_.astype(np.float32), mass,
                        tp if ts is None else tp + ts))
        return out

    def _seed_pose(self, h, yaw, pitch):
        """Coarse-grid camera: image-up = desk direction `yaw`, optical axis tilted by `pitch` [rad] from straight
        down towards image-up (0 = nadir), aimed at the robot centre, `h` above the desk -> (R, pos, depth of the
        robot centre along the optical axis)."""
        c = self.pivot
        R = nadir_rotation(yaw)
        if pitch:
            R = R @ axis_angle((1.0, 0.0, 0.0), pitch)
        D = (h - c[2]) / np.cos(pitch)
        return R, c - R[:, 2] * D, D

    def coarse_search(self, heights=np.geomspace(0.6, 2.0, 10), yaws=np.radians(np.arange(0, 360, 7.5)),
                      scale=0.25, sigma=2.0, topk=6, keep_frac=0.5, band_px=None, band_weight=None,
                      peaks=3, sep_px=40.0, per_place=2, dynamic=None, pitches=(0.0,)):
        """Rank nadir poses (height x yaw grid x image shift, the shift by FFT correlation) by a local soft IoU
        summed over snapshots: matched / (rendered + segmented-but-unexplained within band_px [full-res px] around
        the rendered robot). Unlike a plain agreement score it does not reward a too-small robot (camera too high)
        that fits inside the segmented robot, nor a robot dropped into a large achromatic clutter area (grey floor
        tiles, paper): both leave segmented mass right next to the render. A uniform clutter blob still scores
        well at every grid cell while the real robot only scores well at the cells closest to the true pose, so
        each cell contributes its `peaks` best image locations (>= sep_px apart) and the final list keeps at most
        `per_place` candidates per image location -- the refinement then decides.
        pitches [rad]: extra grid axis of cameras tilted towards image-up (default nadir only; the refinement
        reaches ~25-30 deg of tilt from a nadir seed)."""
        band_px = self.band_px if band_px is None else band_px
        band_weight = self.band_weight if band_weight is None else band_weight
        lv = self.level(scale, sigma)
        W, H = lv.size
        Wp, Hp = 2 * W, 2 * H
        maps = self._coarse_maps(scale, sigma, dynamic)
        FT_p = [np.fft.rfft2(mp_[0], s=(Hp, Wp)) for mp_ in maps]  # zero padding: nothing to match off-image
        FT_s = [None if mp_[1] is None else np.fft.rfft2(mp_[1], s=(Hp, Wp)) for mp_ in maps]
        FT_b = [np.fft.rfft2(mp_[3], s=(Hp, Wp)) for mp_ in maps]
        r = max(1, int(round(band_px * scale)))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        use_servo = self.mode == "class"
        c = self.pivot
        mu, mv = int(0.75 * W), int(0.75 * H)
        dus = np.arange(-mu, mu + 1)
        dvs = np.arange(-mv, mv + 1)
        rad = max(1, int(round(sep_px * scale / 2)))
        cands = []
        for h, yaw, pitch in ((h, y, p) for h in heights for y in yaws for p in pitches):
            R, pos, _ = self._seed_pose(h, yaw, pitch)
            tot, tot_b, n_ren = 0, 0, 0.0
            rend = lambda m: render(*m, R, pos, lv.K, self.dist, lv.size, labels=self.mode != "union")  # noqa: E731
            labs = self.pool.map(rend, self.coarse_meshes) if self.pool else map(rend, self.coarse_meshes)
            for lab, fp, fs, fb in zip(labs, FT_p, FT_s, FT_b):
                mp = (lab == PRINTED) if self.mode != "union" else lab
                matched = mp.copy()
                n_ren += float(mp.sum())
                tot = tot + np.conj(np.fft.rfft2(mp.astype(np.float32), s=(Hp, Wp))) * fp
                if use_servo:
                    ms = lab == SERVO
                    matched |= ms
                    n_ren += float(ms.sum())
                    tot = tot + np.conj(np.fft.rfft2(ms.astype(np.float32), s=(Hp, Wp))) * fs
                band = (cv2.dilate((lab > 0).astype(np.uint8), kern) > 0) & ~matched
                tot_b = tot_b + np.conj(np.fft.rfft2(band.astype(np.float32), s=(Hp, Wp))) * fb
            corr = np.fft.irfft2(tot, s=(Hp, Wp))[np.ix_(dvs % Hp, dus % Wp)]
            unexpl = np.fft.irfft2(tot_b, s=(Hp, Wp))[np.ix_(dvs % Hp, dus % Wp)]
            iou_map = corr / np.maximum(n_ren + band_weight * np.maximum(unexpl, 0.0), 1.0)
            for _ in range(peaks):  # greedy peaks, each suppressing a disc of radius 2*rad
                iv, iu = np.unravel_index(np.argmax(iou_map), iou_map.shape)
                if not np.isfinite(iou_map[iv, iu]):
                    break
                cands.append((float(iou_map[iv, iu]), h, yaw, dus[iu] / scale, dvs[iv] / scale, pitch))
                iou_map[max(0, iv - 2 * rad):iv + 2 * rad + 1, max(0, iu - 2 * rad):iu + 2 * rad + 1] = -np.inf
        cands.sort(key=lambda t: -t[0])
        picked = []
        for cd in cands:  # suppression: one candidate per (place, yaw, height, pitch) cell, <= per_place per image place
            if cd[0] < keep_frac * cands[0][0]:
                break
            near = [p for p in picked if np.hypot(cd[3] - p[3], cd[4] - p[4]) < sep_px]
            same = [p for p in near if abs((cd[2] - p[2] + np.pi) % (2 * np.pi) - np.pi) < np.radians(20)
                    and abs(np.log(cd[1] / p[1])) < 0.2 and abs(cd[5] - p[5]) < np.radians(10)]
            if not same and len(near) < per_place:
                picked.append(cd)
            if len(picked) >= topk:
                break
        out = []
        Kinv = np.linalg.inv(self.K)
        for sc, h, yaw, du, dv, pitch in picked:  # image shift -> camera pose (two interpretations)
            R, pos, D = self._seed_pose(h, yaw, pitch)
            out.append(("shift", sc, R, pos - R[:, 0] * du * D / self.K[0, 0] - R[:, 1] * dv * D / self.K[1, 1]))
            p0 = project(c[None], R, pos, self.K, None)[0][0]
            a, b = Kinv @ np.r_[p0, 1], Kinv @ np.r_[p0 + (du, dv), 1]
            out.append(("tilt", sc, R @ _rot_between(b, a), pos))
            self.log(f"  coarse cand: soft IoU {sc:.3f} height {h:.2f} m yaw {np.degrees(yaw):6.1f} deg"
                     + (f" pitch {np.degrees(pitch):+.0f} deg" if pitch else "") + f" shift ({du:.0f},{dv:.0f}) px")
        return out

    # --- full pipeline
    def robot_area_px(self, height):
        """Mean pixel area of the robot (coarse mesh) seen by a nadir camera `height` above the desk."""
        R, pos = nadir_rotation(0.0), np.array([self.pivot[0], self.pivot[1], height])
        return float(np.mean([render(*m, R, pos, self.K, self.dist, self.size, labels=False).sum() for m in self.coarse_meshes]))

    def solve(self, init_R=None, init_pos=None, coarse=True, heights=(0.6, 2.0), yaw_step=7.5, topk=5, yaw_range=None,
              coarse_scale=None, polish=True, pitches=(0.0,)):
        """Coarse search + 4-stage refinement. yaw_range = (centre, half width) [rad] restricts the coarse yaw
        grid (image-up direction on the desk). coarse_scale None: 1/4 resolution when the robot covers >= 3000 px
        at the reference height (init height, else the middle of `heights`), 1/2 when it is smaller.
        polish=False stops after the half-resolution stage (solve_pipeline polishes in its second pass).
        pitches [rad]: coarse seeds tilted towards image-up besides nadir (e.g. (0, +-25 deg) for a camera tilted
        by more than ~25 deg; each extra pitch adds one coarse-search pass)."""
        t0 = time.time()
        cands = []
        if init_R is not None and init_pos is not None:
            cands.append(("init", np.nan, np.asarray(init_R, float), np.asarray(init_pos, float)))
        h_ref = float(init_pos[2]) if init_pos is not None else float(np.sqrt(heights[0] * heights[1]))
        if coarse_scale is None:
            area = self.robot_area_px(h_ref)
            coarse_scale = 0.25 if area >= 3000 else 0.5
            self.log(f"  robot ~{area:.0f} px at {h_ref:.2f} m -> coarse search at 1/{1 / coarse_scale:.0f} resolution")
        if coarse:
            hs = np.geomspace(heights[0], heights[1], max(2, int(np.ceil(np.log(heights[1] / heights[0]) / np.log(1.14))) + 1))
            yaws = np.radians(np.arange(0, 360, yaw_step))
            if yaw_range is not None:
                yaws = yaws[np.abs((yaws - yaw_range[0] + np.pi) % (2 * np.pi) - np.pi) <= yaw_range[1] + 1e-9]
            self.diversity = self.pose_diversity(h_ref)
            dyn = self.dynamic_coarse and len(self.snaps) >= 3
            if dyn:
                self.log(f"  arm pose diversity: {100 * self.diversity:.0f}% of the rendered arm moves between snapshots")
                cands += self.coarse_search(hs, yaws, scale=coarse_scale, topk=topk, dynamic=True, pitches=pitches)
            if not dyn or self.diversity < self.min_diversity:
                # arm poses too similar (or < 3 snapshots): the moving-parts targets are mostly empty, so search the
                # full segmentation as well; static clutter competes there, the IoU ranking below has to reject it
                self.log(f"  {'only ' + str(len(self.snaps)) + ' snapshot(s)' if len(self.snaps) < 3 else 'arm poses too similar'}"
                         f" -> also searching the full segmentation")
                cands += self.coarse_search(hs, yaws, scale=coarse_scale, topk=topk, dynamic=False, pitches=pitches)
        self.log(f"  coarse search: {len(cands)} candidates, {time.time() - t0:.1f}s")
        # stage 1 (coarse resolution, softness 2 px there, decimated mesh): every candidate. Candidates are ranked by
        # silhouette IoU, not by the raw agreement score: the score grows with the area of segmented pixels covered,
        # so a robot rendered too large over an achromatic clutter area (floor tiles, paper) can out-score the truth
        res = []
        for name, sc, R, pos in cands:
            R1, p1, s1 = self.refine(R, pos, coarse_scale, 2.0, step=2.0, maxfev=110, coarse_mesh=True)
            iou = self.mean_iou(R1, p1)
            res.append((iou, s1, name, R1, p1))
            self.log(f"  cand {name:5s}: stage-1 IoU {iou:.3f} score {s1:.3f} pos {np.round(p1, 3)} yaw {np.degrees(yaw_of(R1)):6.1f}")
        res.sort(key=lambda t: -t[0])
        self.log(f"  stage 1 done, {time.time() - t0:.1f}s")
        # stage 2 (1/2 res, 2 px): best two; stage 3 (1/2 res, 1 px) and the polish (full res, 1 px): the winner
        res2 = []
        for _, _, n, R, p in res[:2]:
            R2, p2, s2 = self.refine(R, p, 0.5, 2.0, step=1.0, maxfev=200)
            res2.append((self.mean_iou(R2, p2), s2, n, R2, p2))
        res2.sort(key=lambda t: -t[0])
        _, s, name, R, pos = res2[0]
        R, pos, s = self.refine(R, pos, 0.5, 1.0, step=0.5, maxfev=250)
        if not polish:
            out = dict(R=R, pos=pos, quat=Rotation.from_matrix(R).as_quat(), score=s, start=name,
                       runtime_s=time.time() - t0, n_eval=self.n_eval, pose_diversity=self.diversity)
            out.update(self.diagnostics(R, pos))
            self.log(f"  half-resolution solution after {out['runtime_s']:.1f}s: IoU {out['iou_mean']:.3f}")
            return out
        return self.polish(R, pos, t0=t0, start=name)

    def polish(self, R, pos, t0=None, start="given", fine_maxfev=250, restarts=3, tol=5e-4):
        """Full-resolution refinement (1 px softness) from a nearby pose + diagnostics -> result dict. Nelder-Mead
        in 6-D stalls on this piecewise-constant score, so the simplex is restarted around the best pose until a
        restart gains less than `tol` (relative score)."""
        t0 = time.time() if t0 is None else t0
        R, pos, s = self.refine(R, pos, 1.0, 1.0, step=0.25, maxfev=fine_maxfev)
        for _ in range(restarts):
            R2, p2, s2 = self.refine(R, pos, 1.0, 1.0, step=0.1, maxfev=150)
            gain = s2 - s
            if gain > 0:
                R, pos, s = R2, p2, s2
            if gain < tol * max(abs(s), 1e-3):
                break
        out = dict(R=R, pos=pos, quat=Rotation.from_matrix(R).as_quat(), score=s, start=start,
                   runtime_s=time.time() - t0, n_eval=self.n_eval, pose_diversity=self.diversity)
        out.update(self.diagnostics(R, pos))
        self.log(f"  solved in {out['runtime_s']:.1f}s ({self.n_eval} renders x {len(self.snaps)} snapshots): score {s:.4f}, "
                 f"IoU {out['iou_mean']:.3f}, boundary {out['boundary_px_mean']:.2f} px")
        return out

    # --- diagnostics / overlays
    def diagnostics(self, R, pos, near_px=20):
        per = []
        for snap, (V, F, L), sg in zip(self.snaps, self.meshes, self.seg):
            lab = render(V, F, L, R, pos, self.K, self.dist, self.size)
            r = (lab == PRINTED) if self.mode == "white" else lab > 0
            s = (sg["white"] if self.mode == "white" else sg["robot"]) > 0.5
            if self.roi is not None:
                s &= self.roi
            near = cv2.dilate(r.astype(np.uint8), np.ones((2 * near_px + 1,) * 2, np.uint8)) > 0
            s &= near  # ignore unrelated achromatic blobs far from the robot (cup rim, clutter)
            iou = float((r & s).sum() / max((r | s).sum(), 1))
            k = np.ones((3, 3), np.uint8)
            rb = r & ~(cv2.erode(r.astype(np.uint8), k) > 0)
            sb = s & ~(cv2.erode(s.astype(np.uint8), k) > 0)
            dts = cv2.distanceTransform((~sb).astype(np.uint8), cv2.DIST_L2, 5)
            dtr = cv2.distanceTransform((~rb).astype(np.uint8), cv2.DIST_L2, 5)
            fwd, bwd = dts[rb], dtr[sb]
            per.append(dict(name=snap.name, iou=iou, arm_px=feret_px(lab > 0),
                            boundary_px=float(np.mean(fwd)) if len(fwd) else np.nan,
                            boundary_med_px=float(np.median(fwd)) if len(fwd) else np.nan,
                            boundary_sym_px=float((fwd.sum() + bwd.sum()) / max(len(fwd) + len(bwd), 1)),
                            n_rendered=int(r.sum()), n_segmented=int(s.sum())))
        return dict(per_snap=per, iou_mean=float(np.mean([p["iou"] for p in per])),
                    arm_px_mean=float(np.mean([p["arm_px"] for p in per])),
                    boundary_px_mean=float(np.mean([p["boundary_px"] for p in per])),
                    boundary_sym_px_mean=float(np.mean([p["boundary_sym_px"] for p in per])))

    def overlay(self, R, pos, cols=3, show_seg=True):
        """Tiled overlay: segmented robot boundary (yellow), rendered printed parts (green), servos (magenta)."""
        tiles = []
        for snap, (V, F, L), sg in zip(self.snaps, self.meshes, self.seg):
            img = snap.image.copy()
            if show_seg:
                cs, _ = cv2.findContours(((sg["white"] if self.mode == "white" else sg["robot"]) > 0.5).astype(np.uint8),
                                         cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, (0, 220, 255), 1)
            lab = render(V, F, L, R, pos, self.K, self.dist, self.size)
            for cls, col in ((PRINTED, (0, 255, 0)), (SERVO, (255, 0, 255))):
                cs, _ = cv2.findContours((lab == cls).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, col, 1)
            if self.roi is not None:
                cs, _ = cv2.findContours(self.roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, (255, 128, 0), 1)
            cv2.putText(img, snap.name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            tiles.append(img)
        return tile(tiles, cols)


def feret_px(mask):
    """Largest distance [px] between two pixels of a bool mask (its maximum Feret diameter; 0 if empty): how
    long the robot looks in the image."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return 0.0
    h = cv2.convexHull(np.c_[xs, ys].astype(np.int32)).reshape(-1, 2).astype(float)
    return float(np.sqrt(((h[:, None] - h[None]) ** 2).sum(-1)).max())


def tile(images, cols=3):
    h, w = images[0].shape[:2]
    rows = int(np.ceil(len(images) / cols))
    out = np.zeros((rows * h, min(cols, len(images)) * w, 3), np.uint8)
    for i, im in enumerate(images):
        out[(i // cols) * h:(i // cols + 1) * h, (i % cols) * w:(i % cols + 1) * w] = im
    return out


def roi_from_pose(snaps, R, pos, K, dist=None, grow=2.5, pad=40, model=None):
    """Bool mask: bounding box of the robot rendered at a (rough) pose over all snapshots, grown by `grow`
    about its centre plus `pad` px -- the region the real robot is searched in."""
    model = model or get_model(max_tris=6000)
    H, W = snaps[0].image.shape[:2]
    m = np.zeros((H, W), bool)
    for s in snaps:
        V, F, L = model.world_mesh(s.q)
        m |= render(V, F, L, R, pos, np.asarray(K, float), dist, (W, H), labels=False)
    roi = np.zeros((H, W), bool)
    if not m.any():
        roi[:] = True
        return roi
    ys, xs = np.nonzero(m)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    hw = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * grow + pad
    roi[max(0, int(cy - hw)):int(cy + hw) + 1, max(0, int(cx - hw)):int(cx + hw) + 1] = True
    return roi


def solve_camera_pose(snaps, K=None, dist=None, init_R=None, init_pos=None, mode="class", roi=None, coarse=True,
                      heights=(0.6, 2.0), yaw_step=7.5, verbose=True, min_diversity=0.3, **solve_kw):
    """Convenience wrapper -> (result dict with pos, quat (xyzw, ROS axes), R, per-snapshot IoU/boundary, solver).
    min_diversity: below this pose_diversity() the whole-image (static) targets are searched too (> 1: always).
    solve_kw: yaw_range, coarse_scale, topk, polish (CameraSolver.solve)."""
    solver = CameraSolver(snaps, K, dist, mode=mode, roi=roi, verbose=verbose, min_diversity=min_diversity)
    return solver.solve(init_R, init_pos, coarse=coarse, heights=heights, yaw_step=yaw_step, **solve_kw), solver


def solve_pipeline(snaps, K=None, dist=None, init_R=None, init_pos=None, mode="white", heights=(0.6, 2.5),
                   yaw_step=7.5, two_pass=True, roi_grow=1.6, roi_pad=30, verbose=True, min_diversity=0.3, **solve_kw):
    """What `solve.py scene_robot` runs. Pass 1: global search + refinement up to half resolution on the whole
    image. Pass 2 (two_pass): segment again with the desk/illuminant statistics taken only around the robot
    (ROI = box around the pass-1 render, grown), which ignores clutter elsewhere in the room, and polish at full
    resolution.
    Returns (result, solver); result['pass1'] keeps the pass-1 pose."""
    K = np.array(PLACEHOLDER_K if K is None else K, float).reshape(3, 3)
    t0 = time.time()
    res, solver = solve_camera_pose(snaps, K, dist, init_R, init_pos, mode=mode, heights=heights,
                                    yaw_step=yaw_step, verbose=verbose, polish=not two_pass,
                                    min_diversity=min_diversity, **solve_kw)
    if not two_pass:
        res["roi"] = None
        return res, solver
    roi = roi_from_pose(snaps, res["R"], res["pos"], K, dist, grow=roi_grow, pad=roi_pad)
    solver2 = CameraSolver(snaps, K, dist, mode=mode, roi=roi, verbose=verbose,
                           model=solver.model, coarse_model=solver.coarse_model)
    out = solver2.polish(res["R"], res["pos"], t0=t0, start=res["start"] + "+roi")
    out["n_eval"] += res["n_eval"]
    out["pass1"] = dict(pos=res["pos"], R=res["R"], iou_mean=res["iou_mean"], boundary_px_mean=res["boundary_px_mean"])
    out["pose_diversity"] = res.get("pose_diversity")
    out["roi"] = roi
    return out, solver2


# ----------------------------------------------------------------------------- I/O, checks, metrics
def load_intrinsics(path=HERE / "data" / "intrinsics_scene.json"):
    """(K, dist, source) of the overhead camera: calib/data/intrinsics_scene.json (solve.py intrinsics --cam
    scene) when it exists, else the sim placeholder fx=fy=525, cx=320, cy=240 without distortion."""
    p = Path(path)
    if p.exists():
        d = json.loads(p.read_text())
        return np.array(d["K"], float).reshape(3, 3), np.asarray(d["dist"], float).ravel(), str(p)
    return PLACEHOLDER_K.copy(), np.zeros(5), "placeholder (fx=fy=525, cx=320, cy=240, no distortion)"


def load_snapshot(snap_dir) -> Snapshot:
    """calib/capture.py snapshot dir (scene.jpg + state.json with the follower's LeRobot joint readings)."""
    d = Path(snap_dir)
    img = cv2.imread(str(d / "scene.jpg"))
    if img is None:
        raise FileNotFoundError(f"{d / 'scene.jpg'} missing or unreadable")
    state = json.loads((d / "state.json").read_text())
    if "follower" not in state or not state["follower"]:
        raise ValueError(f"{d / 'state.json'} has no follower joint readings")
    name = d.name[5:] if d.name.startswith("snap_") else d.name
    return Snapshot(img, q_from_follower(state["follower"]), name)


def initial_pose(height=1.4, yaw_deg=0.0, xy=None):
    """Nadir camera guess: `height` above the desk, image-up = desk direction at `yaw_deg` from +x (0 = the arm
    points up in the image, as in the real overhead view), over xy (default 0.3*height in front of the base)."""
    xy = (0.3 * height, 0.0) if xy is None else xy
    return nadir_rotation(np.radians(yaw_deg)), np.array([xy[0], xy[1], height], float)


def workspace_points(x=(-0.05, 0.45), y=(-0.30, 0.30), z=(0.0, 0.15), n=9):
    """Grid over the robot's desk workspace (desk level and 15 cm above it) [m]."""
    g = np.meshgrid(np.linspace(*x, n), np.linspace(*y, n), np.asarray(z, float), indexing="ij")
    return np.stack(g, -1).reshape(-1, 3)


def reprojection_discrepancy(R1, p1, R2, p2, K, dist=None, pts=None, size=IMAGE_SIZE):
    """Image-space effect of a pose error: pixel distance between where two camera poses put the same workspace
    points (only points inside the first camera's image) -> (mean, max) [px]. Tilt and sideways shift of a
    nadir camera nearly cancel here, so this is what a pose error costs in sim-vs-real image consistency."""
    pts = workspace_points() if pts is None else pts
    a, za = project(pts, R1, p1, K, dist)
    b, zb = project(pts, R2, p2, K, dist)
    ok = (za > 0) & (zb > 0) & (a[:, 0] >= 0) & (a[:, 0] < size[0]) & (a[:, 1] >= 0) & (a[:, 1] < size[1])
    if not ok.any():
        return float("nan"), float("nan")
    d = np.linalg.norm(a[ok] - b[ok], axis=1)
    return float(d.mean()), float(d.max())


def joint_offset_check(res_j, min_offset_deg=1.5, min_iou_gain=0.015):
    """Interpret CameraSolver.fit_joint_offsets() (res_j, fitted from a solve without offsets): flag when constant
    joint offsets of >= min_offset_deg explain the silhouettes clearly better (mean IoU gain >= min_iou_gain) --
    then the joint readings disagree with the URDF zero and the pose solved without offsets is off by roughly the
    reported camera move. -> dict(flag, message, offsets_deg, iou_gain, move_cm, move_deg, max_offset_deg)."""
    off = np.asarray(res_j["joint_offsets_deg"], float)
    idx = [JOINTS.index(j) for j in res_j["offset_joints"]]
    gain = float(res_j["iou_mean"] - res_j["iou_mean_no_offsets"])
    move_cm = float(np.linalg.norm(np.asarray(res_j["pos"]) - np.asarray(res_j["pos_no_offsets"])) * 100)
    move_deg = rot_angle_deg(np.asarray(res_j["R"]), np.asarray(res_j["R_no_offsets"]))
    big = float(np.abs(off[idx]).max())
    txt = ", ".join(f"{JOINTS[i]} {off[i]:+.1f}" for i in idx)
    flag = bool(big >= min_offset_deg and gain >= min_iou_gain)
    msg = (f"joint readings look offset from the URDF zero: adding {txt} deg to them fits the silhouettes better "
           f"(IoU {res_j['iou_mean_no_offsets']:.3f} -> {res_j['iou_mean']:.3f}) and moves the camera by {move_cm:.1f} cm / "
           f"{move_deg:.1f} deg -- check the LeRobot calibration (or rerun with --joint_offsets fit)")
    return dict(flag=flag, message=msg, offsets_deg=off, iou_gain=gain, move_cm=move_cm, move_deg=move_deg, max_offset_deg=big)


def verdict(res, iou_min=0.75, boundary_max=2.0, height=(0.5, 3.0), tilt_max_deg=35.0, min_arm_px=100.0,
            intrinsics_calibrated=True, offset_check=None):
    """('PASS' | 'WARN', [reasons]) for a solve: every snapshot's silhouette must agree (IoU, mean boundary
    distance), the pose must be a plausible overhead camera, the robot must be big enough in the image (below
    ~100 px the dark-JPEG edge bias puts the camera > ~1.5 cm too low, growing fast), the intrinsics must be a
    real calibration (with the placeholder K the camera height scales with the fx guess), and -- when
    offset_check (joint_offset_check()) is given -- the joint readings must not look offset from the URDF zero.
    What it cannot see: a shoulder_pan zero offset (indistinguishable from a camera yaw about the base)."""
    why = []
    if not intrinsics_calibrated:
        why.append("scene intrinsics not calibrated (placeholder fx=fy=525): the solved height/scale is only as good "
                   "as that guess -- calibrate first (capture.py intrinsics --cam scene; solve.py intrinsics --cam scene)")
    arm = res.get("arm_px_mean")
    if arm is not None and arm < min_arm_px:
        why.append(f"robot only ~{arm:.0f} px long in the image (< {min_arm_px:.0f}): the solved camera is biased low "
                   f"(>~1.5 cm, growing fast with distance); move the camera closer or capture at a higher resolution")
    if offset_check is not None and offset_check.get("flag"):
        why.append(offset_check["message"])
    for p in res["per_snap"]:
        if not p["iou"] >= iou_min:
            why.append(f"snapshot {p['name']}: IoU {p['iou']:.2f} < {iou_min}")
        if not p["boundary_px"] <= boundary_max:
            why.append(f"snapshot {p['name']}: boundary residual {p['boundary_px']:.2f} px > {boundary_max}")
    R, pos = np.asarray(res["R"]), np.asarray(res["pos"])
    if not height[0] <= pos[2] <= height[1]:
        why.append(f"camera height {pos[2]:.2f} m outside {height}")
    tilt = float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))
    if tilt > tilt_max_deg:
        why.append(f"optical axis {tilt:.0f} deg from straight down (> {tilt_max_deg})")
    if len(res["per_snap"]) < 3:
        why.append(f"only {len(res['per_snap'])} snapshot(s); use >= 4 distinct arm poses")
    return ("PASS" if not why else "WARN"), why

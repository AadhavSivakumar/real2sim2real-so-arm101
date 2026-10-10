"""Real-look appearance assets for the camera env, built from real teleop episodes (no Isaac).

  plate     robust per-pixel median of the overhead camera over frames of every episode; the arm (URDF
            silhouette at the recorded joints, calib/silhouette.py), the cube, the cup and the user's hands
            (bright outliers) are rejected; holes (robot base) are inpainted
  objects   cup + cube poses in each episode's first frame (cup: silhouette fit of the tapered cup model;
            cube: bright orange blob, top face at z = 30 mm) and their colours
  lamp      point-light position fitted to the cups' cast shadows (frame / plate ratio) over all episodes
  texture   the undistorted plate projected onto the desk plane z = 0 (1 mm/px) over the whole overhead view,
            divided by the fitted lamp's irradiance (de-lit: the sim lamp re-lights it); outside the view mirrored
  backdrop.usda   visual-only quads: desk top with UVs + UsdPreviewSurface/UsdUVTexture (z = +0.5 mm) and a plain
            dark back wall just beyond the overhead's view; no physics (the desk collider in env_cfg.py is unchanged)
  real_stats.json lamp estimate, colours, object poses per episode, per-camera image statistics (undistorted)

Run with the LeRobot venv (pyav, pandas, opencv, scipy), from the repo root:
  ~/lr/bin/python scripts/build_real2sim_assets.py [--dataset DIR] [--out DIR] [--step 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "calib"))
import silhouette as sil  # noqa: E402

DATASET = "/mnt/secondary/so101_r2s/lerobot_datasets/real_cube_cup_v1"
OUT = "/mnt/secondary/so101_r2s/assets/real2sim"
CALIB_JSON = REPO / "calib" / "data" / "calib.json"
W, H = 640, 480
CUP = dict(top_r=0.040, bot_r=0.030, height=0.095)  # env_cfg.CUP_DIMS (the overhead rim matches 80 mm)
CUBE = 0.030
DESK_Z = 0.0005  # textured desk quad just above the collider's top face (z = 0)
TEX_RES = 0.0005  # [m/px]; the overhead sees ~1.3 mm/px, the wrist camera up to ~0.3 mm/px -> procedural fine grain
DESK_EXTENT = dict(x=(-0.20, None), y=(-0.60, 0.60))  # x max: just beyond the overhead's view (computed)
WALL_HEIGHT = 0.80
WALL_COLOR = (0.035, 0.03, 0.028)  # linear albedo: the real wall is barely lit (dark in the wrist frames)


# ----------------------------------------------------------------------------- camera model
class Cam:
    """Calibrated camera (OpenCV model): K, dist, R (world <- camera, ROS axes as columns), centre C."""

    def __init__(self, K, dist, R, C):
        self.K, self.dist = np.asarray(K, float).reshape(3, 3), np.asarray(dist, float)
        self.R, self.C = np.asarray(R, float), np.asarray(C, float)

    def project(self, P):
        """World points -> UNDISTORTED pixel coords (n,2), depth (n,)."""
        Pc = (np.asarray(P, float).reshape(-1, 3) - self.C) @ self.R
        z = Pc[:, 2]
        uv = Pc[:, :2] / np.where(z > 1e-6, z, 1e-6)[:, None]
        return uv * [self.K[0, 0], self.K[1, 1]] + [self.K[0, 2], self.K[1, 2]], z

    def ray(self, uv):
        n = (np.asarray(uv, float).reshape(-1, 2) - [self.K[0, 2], self.K[1, 2]]) / [self.K[0, 0], self.K[1, 1]]
        return np.c_[n, np.ones(len(n))] @ self.R.T

    def on_plane_z(self, uv, z=0.0):
        """Undistorted pixels -> world points on the horizontal plane at height z."""
        d = self.ray(uv)
        return self.C + ((z - self.C[2]) / d[:, 2])[:, None] * d

    def undistort(self, img):
        return cv2.undistort(img, self.K, self.dist)


def scene_cam() -> Cam:
    c = json.loads(CALIB_JSON.read_text())
    return Cam(c["SCENE_K"], c["SCENE_DIST"], Rotation.from_quat(c["SCENE_CAM_QUAT"]).as_matrix(), c["SCENE_CAM_POS"])


def wrist_calib():
    c = json.loads(CALIB_JSON.read_text())
    return np.asarray(c["WRIST_K"], float).reshape(3, 3), np.asarray(c["WRIST_DIST"], float)


def srgb_to_lin(x):
    x = np.asarray(x, np.float32) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return 255.0 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def luma(img):
    return img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114


# ----------------------------------------------------------------------------- data
def n_episodes(dataset) -> int:
    return len(list((Path(dataset) / "data" / "chunk-000").glob("file-*.parquet")))


def episode_states(dataset, ep) -> np.ndarray:
    df = pd.read_parquet(Path(dataset) / "data" / "chunk-000" / f"file-{ep:03d}.parquet")
    return np.stack(df["observation.state"].to_numpy()).astype(np.float64)


def episode_frames(dataset, ep, cam, step=1, first_only=False, indices=None) -> list[np.ndarray]:
    path = Path(dataset) / "videos" / f"observation.images.{cam}" / "chunk-000" / f"file-{ep:03d}.mp4"
    out = []
    with av.open(str(path)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if (i in indices) if indices is not None else (i % step == 0):
                out.append(f.to_ndarray(format="rgb24"))
                if first_only or (indices is not None and len(out) == len(indices)):
                    break
    return out


def robot_mask(cam: Cam, q_lerobot, dilate=31) -> np.ndarray:
    """Arm silhouette in the DISTORTED overhead image (bool), dilated (calibration error, cables)."""
    V, F, L = sil.get_model().world_mesh(sil.lerobot_to_sim(q_lerobot))
    m = sil.render(V, F, L, cam.R, cam.C, cam.K, cam.dist, (W, H), labels=False).astype(np.uint8)
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate, dilate))) > 0


def _load_ep(job):
    dataset, ep, step = job
    cam = scene_cam()
    st = episode_states(dataset, ep)
    fr = episode_frames(dataset, ep, "scene", step)
    masks = [robot_mask(cam, st[min(i * step, len(st) - 1)]) for i in range(len(fr))]
    return np.stack(fr), np.stack(masks)


def clean_plate(frames, masks, bright_tol=25.0, dev_tol=18.0, min_count=5):
    """Robust background of a fixed camera: per pixel, drop samples under the arm, samples brighter than the
    pixel's 20th luminance percentile + bright_tol (cup, cube, white arm, the user's skin: all brighter than the
    dark desk), then samples far from that first median (cables, transient shadows); median of the rest."""
    lum = luma(frames.astype(np.float32))
    plate = np.zeros(frames.shape[1:], np.float32)
    count = np.zeros(frames.shape[1:3], np.int32)
    for r0 in range(0, frames.shape[1], 24):
        sl = slice(r0, r0 + 24)
        lm = np.where(masks[:, sl], np.nan, lum[:, sl])
        with np.errstate(all="ignore"):
            p20 = np.nanpercentile(lm, 20, axis=0)
            ok = ~masks[:, sl] & (lum[:, sl] <= p20[None] + bright_tol)
            a = np.where(ok[..., None], frames[:, sl].astype(np.float32), np.nan)
            ok &= np.abs(lum[:, sl] - luma(np.nanmedian(a, axis=0))[None]) < dev_tol
            a = np.where(ok[..., None], frames[:, sl].astype(np.float32), np.nan)
            plate[sl] = np.nanmedian(a, axis=0)
        count[sl] = ok.sum(0)
    hole = (count < min_count) | ~np.isfinite(plate).all(-1)
    p8 = np.nan_to_num(plate).clip(0, 255).astype(np.uint8)
    return cv2.inpaint(p8, hole.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA), hole


# ----------------------------------------------------------------------------- objects
def circle_pts(c, r, z, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a), np.full(n, z)]


def cup_silhouette(cam: Cam, xy) -> np.ndarray:
    """Undistorted-image silhouette (uint8 mask) of the tapered cup standing at desk point xy."""
    pts = np.r_[circle_pts(xy, CUP["bot_r"], 0.0), circle_pts(xy, CUP["top_r"], CUP["height"])]
    uv, _ = cam.project(pts)
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(uv * 4).astype(np.int32)), 1, shift=2)
    return m


def detect_objects(cam: Cam, frame_u, plate_u, arm_u):
    """Cup and cube in an undistorted first frame: (cup_xy, cube_xy, cube_yaw, masks) (None where not found)."""
    diff = np.abs(frame_u.astype(np.int16) - plate_u.astype(np.int16)).sum(-1)
    hsv = cv2.cvtColor(frame_u, cv2.COLOR_RGB2HSV)
    fg = ((diff > 60) & ~arm_u).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(fg)
    cup_xy, cup_m = None, None
    # cup: the largest bright, weakly saturated (white rim / light-blue inside) blob
    xy = cam.on_plane_z(cent, CUP["height"] * 0.6)[:, :2]
    cands = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 1500 and xy[i, 0] > 0.0 and np.hypot(*xy[i]) < 0.4
             and np.median(hsv[..., 2][lab == i]) > 120 and np.median(hsv[..., 1][lab == i]) < 110]
    if cands:
        i = max(cands, key=lambda k: stats[k, cv2.CC_STAT_AREA])
        blob = (lab == i).astype(np.uint8)
        xy0 = cam.on_plane_z(cent[i][None], CUP["height"] * 0.6)[0, :2]

        def cost(xy):
            s = cup_silhouette(cam, xy)
            return -float((s & blob).sum()) / max(float((s | blob).sum()), 1.0)
        r = minimize(cost, xy0, method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-4, initial_simplex=[xy0, xy0 + [0.01, 0], xy0 + [0, 0.01]]))
        if -r.fun > 0.6:
            cup_xy, cup_m = r.x, cup_silhouette(cam, r.x).astype(bool)
    # cube: bright orange blob (hue ~10-30 deg, brighter than the desk wood), ~20-30 px wide
    cube_xy, cube_yaw, cube_m = None, 0.0, None
    orange = (hsv[..., 0] >= 5) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 70) & (hsv[..., 2] > 110) & (fg > 0)
    if cup_m is not None:
        orange &= ~cv2.dilate(cup_m.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    orange = cv2.morphologyEx(orange.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(orange)
    xy = cam.on_plane_z(cent, CUBE)[:, :2]
    r = np.hypot(xy[:, 0], xy[:, 1])
    cands = [i for i in range(1, n) if 150 < stats[i, cv2.CC_STAT_AREA] < 2000 and 0.08 < r[i] < 0.34 and xy[i, 0] > 0.0]
    if cands:
        i = max(cands, key=lambda k: stats[k, cv2.CC_STAT_AREA])
        cube_m = lab == i
        cube_xy = cam.on_plane_z(cent[i][None], CUBE)[0, :2]
        (_, _), (_, _), ang = cv2.minAreaRect(np.argwhere(cube_m)[:, ::-1].astype(np.float32))
        cube_yaw = float(np.radians(ang) % (np.pi / 2))  # image-plane angle; good enough for a near-nadir view
    return cup_xy, cube_xy, cube_yaw, dict(cup=cup_m, cube=cube_m)


# ----------------------------------------------------------------------------- lamp
class Ortho:
    """Orthographic grid on the desk plane: rows = -x (top = far), cols = -y (left = +y), like the overhead view."""

    def __init__(self, x0, x1, y0, y1, res):
        self.x0, self.x1, self.y0, self.y1, self.res = x0, x1, y0, y1, res
        self.xs = x1 - (np.arange(int(round((x1 - x0) / res))) + 0.5) * res
        self.ys = y1 - (np.arange(int(round((y1 - y0) / res))) + 0.5) * res
        self.X, self.Y = np.meshgrid(self.xs, self.ys, indexing="ij")

    def points(self, z=0.0):
        return np.stack([self.X, self.Y, np.full_like(self.X, z)], -1)

    def sample(self, cam: Cam, img_u, z=0.0, interp=cv2.INTER_LINEAR):
        uv, _ = cam.project(self.points(z).reshape(-1, 3))
        mx, my = uv[:, 0].reshape(self.X.shape).astype(np.float32), uv[:, 1].reshape(self.X.shape).astype(np.float32)
        inside = (mx >= 0) & (mx <= W - 1) & (my >= 0) & (my <= H - 1)
        return cv2.remap(img_u, mx, my, interp, borderMode=cv2.BORDER_REPLICATE), inside

    def to_px(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.c_[(self.y1 - xy[:, 1]) / self.res, (self.x1 - xy[:, 0]) / self.res]  # (col, row)


def cup_shadow(L, xy, grid: Ortho) -> np.ndarray:
    """Cast shadow (on z = 0) of the cup at xy under a point light L, as a grid mask."""
    if L[2] <= CUP["height"] + 0.02:
        return np.zeros(grid.X.shape, bool)
    rim = circle_pts(xy, CUP["top_r"], CUP["height"])
    s = L[2] / (L[2] - rim[:, 2])
    proj = L[None, :2] + (rim[:, :2] - L[None, :2]) * s[:, None]
    pts = grid.to_px(np.r_[circle_pts(xy, CUP["bot_r"], 0)[:, :2], proj])
    m = np.zeros(grid.X.shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(pts * 4).astype(np.int32)), 1, shift=2)
    return m.astype(bool)


def fit_lamp(shots, min_contrast=0.15):
    """Point light that best explains the cups' cast shadows (episode first frames): maximise the mean IoU between
    predicted and observed shadow in each shot's window (desk top near the cup, minus what the cup, arm and cube
    cover in the image). Observed shadow = blurred frame/plate luminance ratio below halfway to the window's darkest
    (2nd percentile) value; cups whose shadow is faint (already in the lamp's shadow, e.g. behind the camera pole)
    are skipped. (The arm's own shadows were tried too: confounded by the user's arm and its shadow.)
    shots: list of (predict(L) -> grid mask, ratio grid, window grid). Returns (L, mean IoU, observed masks, used)."""
    obs, used = [], []
    for _, ratio, w in shots:
        rb = cv2.GaussianBlur(ratio, (0, 0), 2.0)
        contrast = 1.0 - float(np.percentile(rb[w], 2)) if w.sum() > 100 else 0.0
        obs.append((rb < 1.0 - 0.5 * contrast) & w)
        used.append(contrast > min_contrast)
    sel = [i for i, u in enumerate(used) if u]

    def score(L):
        L = np.asarray(L, float)
        if L[2] < 0.15:
            return 0.0
        tot = 0.0
        for i in sel:
            p = shots[i][0](L) & shots[i][2]
            tot += (p & obs[i]).sum() / max((p | obs[i]).sum(), 1)
        return tot / max(len(sel), 1)

    cands = sorted(((score(L), tuple(L)) for L in itertools.product(np.arange(-1.0, 0.61, 0.2), np.arange(-1.0, 1.01, 0.2),
                                                                      (0.4, 0.7, 1.0, 1.4))), reverse=True)[:3]
    best = None
    for _, L0 in cands:
        L0 = np.array(L0)
        r = minimize(lambda L: -score(L), L0, method="Nelder-Mead",
                     options=dict(xatol=2e-3, fatol=1e-4, maxiter=300, initial_simplex=[L0] + [L0 + d for d in np.eye(3) * 0.1]))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun, obs, used


def irradiance(L, P, normal):
    """Lambertian point-light irradiance (up to the intensity) at points P (..., 3) with a fixed normal."""
    d = L - P
    r = np.linalg.norm(d, axis=-1)
    return np.clip((d @ np.asarray(normal, float)) / r, 0, None) / r ** 2


def find_wall_line(cam: Cam, plate_u, cols=(40, 440), rows=(60, 150)):
    """Back edge of the desk (desk top meets the wall/shelf): strongest bright-above-dark edge per column of the
    undistorted plate -> desk-plane line x = a + b*y (robust fit). Returns (a, b)."""
    g = cv2.Sobel(cv2.GaussianBlur(luma(plate_u.astype(np.float32)), (0, 0), 1.5), cv2.CV_32F, 0, 1, ksize=5)
    cs = np.arange(cols[0], cols[1], 4)
    rs = rows[0] + np.argmin(g[rows[0]:rows[1], cs], axis=0)
    xy = cam.on_plane_z(np.c_[cs, rs], 0.0)[:, :2]
    keep = np.ones(len(xy), bool)
    for _ in range(3):
        b, a = np.polyfit(xy[keep, 1], xy[keep, 0], 1)
        res = np.abs(xy[:, 0] - (a + b * xy[:, 1]))
        keep = res < max(3 * np.median(res[keep]), 0.004)
    return float(a), float(b)


def _mirror_1d(line, ok):
    """Mirror the valid contiguous span of a 1-D (n, ...) array into its invalid ends."""
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return line, ok
    a, b = idx[0], idx[-1]
    n, span = len(line), b - a + 1
    j = np.arange(n)
    period = 2 * span
    k = np.mod(j - a, period)
    src = a + np.where(k < span, k, period - 1 - k)
    out = line.copy()
    out[:a], out[b + 1:] = line[src[:a]], line[src[b + 1:]]
    return out, np.ones_like(ok)


def mirror_fill(tex, valid):
    """Fill texels outside the view by mirroring each row's valid span, then each column's."""
    out, ok = tex.copy(), valid.copy()
    for r in range(out.shape[0]):
        if ok[r].any():
            out[r], ok[r] = _mirror_1d(out[r], ok[r])
    for c in range(out.shape[1]):
        out[:, c], ok[:, c] = _mirror_1d(out[:, c], ok[:, c])
    return out


# ----------------------------------------------------------------------------- USD
def _quad_usda(name, pts, normal, mat):
    pts = np.asarray(pts, float)
    lo, hi = pts.min(0), pts.max(0)
    f = lambda v: "(" + ", ".join(f"{c:.6f}" for c in v) + ")"
    return f'''
    def Mesh "{name}" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {{
        float3[] extent = [{f(lo)}, {f(hi)}]
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        rel material:binding = </Backdrop/Looks/{mat}>
        normal3f[] normals = [{", ".join([f(normal)] * 4)}] (
            interpolation = "vertex"
        )
        point3f[] points = [{", ".join(f(p) for p in pts)}]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        uniform token subdivisionScheme = "none"
    }}
'''


def _material_usda(name, texture, roughness):
    p = f"/Backdrop/Looks/{name}"
    return f'''
        def Material "{name}"
        {{
            token outputs:surface.connect = <{p}/Surface.outputs:surface>

            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = <{p}/Texture.outputs:rgb>
                float inputs:metallic = 0
                float inputs:roughness = {roughness}
                token outputs:surface
            }}

            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                float4 inputs:bias = (0, 0, 0, 0)
                asset inputs:file = @./{texture}@
                float4 inputs:scale = (1, 1, 1, 1)
                token inputs:sourceColorSpace = "sRGB"
                float2 inputs:st.connect = <{p}/Reader.outputs:result>
                token inputs:wrapS = "clamp"
                token inputs:wrapT = "clamp"
                float3 outputs:rgb
            }}

            def Shader "Reader"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}
        }}
'''


def _flat_material_usda(name, color, roughness):
    p = f"/Backdrop/Looks/{name}"
    return f'''
        def Material "{name}"
        {{
            token outputs:surface.connect = <{p}/Surface.outputs:surface>

            def Shader "Surface"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = ({color[0]}, {color[1]}, {color[2]})
                float inputs:metallic = 0
                float inputs:roughness = {roughness}
                token outputs:surface
            }}
        }}
'''


def write_backdrop_usda(path, desk_quad, wall_quad, wall_normal, wall_color):
    """desk_quad / wall_quad: 4 corners matching st (0,0) (1,0) (1,1) (0,1) (desk: texture bottom-left, bottom-right,
    top-right, top-left); the wall is a flat colour."""
    txt = f'''#usda 1.0
(
    defaultPrim = "Backdrop"
    doc = "Real-look desk top + back wall (visual only). Generated by scripts/build_real2sim_assets.py"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "Backdrop"
{{{_quad_usda("DeskTop", desk_quad, (0, 0, 1), "DeskMat")}{_quad_usda("Wall", wall_quad, wall_normal, "WallMat")}
    def Scope "Looks"
    {{{_material_usda("DeskMat", "desk_albedo.png", 0.6)}{_flat_material_usda("WallMat", wall_color, 0.9)}    }}
}}
'''
    Path(path).write_text(txt)


# ----------------------------------------------------------------------------- procedural woods (appearance DR)
def grain_detail(shape, seed, amp=0.15):
    """Multiplicative fine wood grain (streaks along the planks = texture rows), mean 1: the wrist camera sees the
    desk much closer than the overhead resolution."""
    rng = np.random.default_rng(seed)
    h, w = shape
    streaks = cv2.resize(rng.standard_normal((h, max(w // 40, 2))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), 0.8)
    g = 0.8 * streaks + 0.2 * fine
    return 1.0 + amp * g / g.std()


def procedural_wood(shape, base_rgb, seed, ring_scale=0.018, contrast=0.22):
    """Plank-style wood grain (stretched noise + growth-ring bands), sRGB uint8."""
    rng = np.random.default_rng(seed)
    h, w = shape
    noise = cv2.resize(rng.standard_normal((max(h // 64, 2), max(w // 2, 2))).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    fine = grain_detail(shape, seed + 100, 1.0) - 1.0
    yy = np.arange(h, dtype=np.float32)[:, None] * TEX_RES
    rings = np.sin(2 * np.pi * (yy + 0.01 * noise) / ring_scale) * 0.5 + 0.5
    planks = np.zeros((h, 1), np.float32)
    edges = np.cumsum(rng.uniform(0.12, 0.25, 16) / TEX_RES).astype(int)
    for k, (a, b) in enumerate(zip(np.r_[0, edges[:-1]], edges)):
        planks[a:b] = rng.uniform(-0.12, 0.12)
    v = 1 + contrast * (rings - 0.5) + 0.08 * fine + planks
    lin = srgb_to_lin(np.asarray(base_rgb, np.float32)) * v[..., None]
    return lin_to_srgb(lin).astype(np.uint8)


# ----------------------------------------------------------------------------- statistics
def image_stats(imgs):
    a = np.asarray(imgs, np.float32).reshape(-1, 3)
    return dict(mean=np.round(a.mean(0, dtype=np.float64), 2).tolist(), std=np.round(a.std(0, dtype=np.float64), 2).tolist())


def region_rgb(img, m):
    return np.round(np.median(img[m].reshape(-1, 3), axis=0), 1).tolist() if m is not None and m.sum() > 20 else None


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--step", type=int, default=4, help="use every N-th overhead frame for the plate")
    ap.add_argument("--wrist_step", type=int, default=10, help="every N-th wrist frame for the image statistics")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--reuse_plate", action="store_true", help="reuse debug/plate.npz from a previous run")
    args = ap.parse_args()
    out, dbg = Path(args.out), Path(args.out) / "debug"
    (dbg / "real_frames").mkdir(parents=True, exist_ok=True)
    save = lambda p, rgb: cv2.imwrite(str(p), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
    cam = scene_cam()
    wK, wD = wrist_calib()
    n_ep = n_episodes(args.dataset)
    t0 = time.time()

    # --- plate
    cache = dbg / "plate.npz"
    if args.reuse_plate and cache.exists():
        z = np.load(cache, allow_pickle=True)
        plate, hole, scene_all = z["plate"], z["hole"], z["scene_all"].item()
    else:
        with ProcessPoolExecutor(min(args.jobs, n_ep)) as ex:
            res = list(ex.map(_load_ep, [(args.dataset, e, args.step) for e in range(n_ep)]))
        frames, masks = np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])
        del res
        plate, hole = clean_plate(frames, masks)
        scene_all = image_stats([cam.undistort(f) for f in frames[:: max(1, 12 // args.step)]])
        del frames, masks
        np.savez_compressed(cache, plate=plate, hole=hole, scene_all=np.array(scene_all, dtype=object))
    plate_u = cam.undistort(plate)
    save(dbg / "plate.png", plate)
    save(dbg / "plate_undist.png", plate_u)
    print(f"[plate] {n_ep} episodes, every {args.step}th frame, {hole.sum()} px inpainted ({time.time() - t0:.0f}s)", flush=True)

    # --- back wall (desk-plane line x = a + b*y) and the analysis region of the real desk top
    a, b = find_wall_line(cam, plate_u)
    print(f"[wall] desk back edge: x = {a:.3f} {b:+.3f}*y  (x = {a + b * 0.3:.3f} at y=+0.3, {a - b * 0.3:.3f} at y=-0.3)")
    grid2 = Ortho(-0.10, 0.45, -0.30, 0.50, 0.002)
    roi2 = ((grid2.X > 0.0) & (grid2.X < a + b * grid2.Y - 0.02) & (grid2.Y > -0.16) & (grid2.Y < 0.30)
            & (np.hypot(grid2.X, grid2.Y) > 0.13))

    # --- objects + shadows in each episode's first frame
    lum_plate = luma(plate_u.astype(np.float32)) + 1.0
    to_grid = lambda m: grid2.sample(cam, m.astype(np.float32), interp=cv2.INTER_NEAREST)[0] > 0.5
    near = lambda xy, r: np.hypot(grid2.X - xy[0], grid2.Y - xy[1]) < r
    dilate = lambda m, k: (cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
                           if m is not None else np.zeros((H, W), bool))
    episodes, shots = [], []
    cube_rgb, cup_in_rgb, cup_out_rgb = [], [], []
    for e in range(n_ep):
        st = episode_states(args.dataset, e)
        f0 = episode_frames(args.dataset, e, "scene", first_only=True)[0]
        w0 = episode_frames(args.dataset, e, "wrist", first_only=True)[0]
        f0u, w0u = cam.undistort(f0), cv2.undistort(w0, wK, wD)
        save(dbg / "real_frames" / f"ep{e:03d}_scene.png", f0u)
        save(dbg / "real_frames" / f"ep{e:03d}_wrist.png", w0u)
        arm_u = cam.undistort(robot_mask(cam, st[0], dilate=41).astype(np.uint8) * 255) > 0
        cup_xy, cube_xy, cube_yaw, m = detect_objects(cam, f0u, plate_u, arm_u)
        ep = dict(episode=e, state0=np.round(st[0], 3).tolist(), n_frames=len(st),
                  cup_xy=None if cup_xy is None else np.round(cup_xy, 4).tolist(),
                  cube_xy=None if cube_xy is None else np.round(cube_xy, 4).tolist(), cube_yaw=round(cube_yaw, 3),
                  scene0=image_stats(f0u), wrist0=image_stats(w0u),
                  wrist_top_rgb=np.round(np.median(w0u[: H // 5].reshape(-1, 3), 0), 1).tolist())  # wall, at rest
        vis = f0u.copy()
        if m["cube"] is not None:
            cube_rgb.append(region_rgb(f0u, cv2.erode(m["cube"].astype(np.uint8), np.ones((5, 5), np.uint8)) > 0))
            vis[m["cube"]] = (0.5 * vis[m["cube"]] + [0, 127, 0]).astype(np.uint8)
        if cup_xy is not None:
            rim_uv, _ = cam.project(circle_pts(cup_xy, CUP["top_r"], CUP["height"]))
            ctr = rim_uv.mean(0)
            inner = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(inner, np.round(ctr + 0.8 * (rim_uv - ctr)).astype(np.int32), 1)
            rim = np.zeros((H, W), np.uint8)
            cv2.fillConvexPoly(rim, np.round(rim_uv).astype(np.int32), 1)
            outer = m["cup"] & ~(cv2.dilate(rim, np.ones((5, 5), np.uint8)) > 0)
            cup_in_rgb.append(region_rgb(f0u, inner > 0))
            cup_out_rgb.append(region_rgb(f0u, outer))
            ep["cup_inner_rgb"], ep["cup_outer_rgb"] = cup_in_rgb[-1], cup_out_rgb[-1]
            cv2.polylines(vis, [np.round(rim_uv).astype(np.int32)], True, (255, 0, 0), 1)
            ratio, _ = grid2.sample(cam, (luma(f0u.astype(np.float32)) + 1.0) / lum_plate)
            excl = to_grid(dilate(m["cup"], 25) | arm_u | dilate(m["cube"], 15))
            shots.append((lambda L, xy=cup_xy: cup_shadow(L, xy, grid2), ratio, roi2 & ~excl & near(cup_xy, 0.22)))
        save(dbg / f"objects_ep{e:03d}.png", vis)
        episodes.append(ep)
        print(f"[objects] ep {e}: cup {ep['cup_xy']} cube {ep['cube_xy']} yaw {cube_yaw:.2f}", flush=True)

    # --- lamp from the cast shadows
    t1 = time.time()
    L, iou, obs, used = fit_lamp(shots)
    spec = L + (L[2] / (L[2] + cam.C[2])) * (cam.C * [1, 1, -1] - L)  # mirror reflection of the lamp seen by the overhead
    print(f"[lamp] point light at {np.round(L, 3).tolist()} (mean shadow IoU {iou:.2f} over {sum(used)} of {len(shots)} "
          f"cups with a clear shadow; {time.time() - t1:.0f}s); its specular highlight seen by the overhead camera would sit "
          f"at desk ({spec[0]:.2f}, {spec[1]:.2f}), pixel {np.round(cam.project(spec)[0][0]).astype(int).tolist()}", flush=True)
    tiles = []
    for (pred, ratio, w), o, u in zip(shots, obs, used):
        t = cv2.cvtColor((np.clip(ratio, 0, 1.5) / 1.5 * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        t[~w] = t[~w] // 2
        p = pred(L)
        t[o] = (0.5 * t[o] + [127, 0, 0]).astype(np.uint8)
        cnt, _ = cv2.findContours(p.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(t, cnt, -1, (0, 255, 0) if u else (255, 255, 0), 1)
        tiles.append(t)
    save(dbg / "lamp_shadow_fit.png", sil.tile(tiles, cols=6))

    # --- desk texture: the whole overhead view on z = 0 (beyond the desk's back edge too: from the overhead's own
    # viewpoint that reproduces the shelf/wall strip at the image top; the steep view makes a vertical wall texture
    # useless), de-lit with the lamp (+ ambient, fitted on the real desk top); a plain dark wall stands just beyond
    valid_u = cv2.erode(cam.undistort(np.full((H, W), 255, np.uint8)), np.ones((5, 5), np.uint8)) > 250
    corners = cam.on_plane_z(cv2.undistortPoints(np.array([[[0, 0]], [[W - 1, 0]]], np.float64), cam.K, cam.dist, P=cam.K).reshape(-1, 2))
    wall_x = float(np.ceil((corners[:, 0].max() + 0.01) * 100) / 100)
    gd = Ortho(DESK_EXTENT["x"][0], wall_x, *DESK_EXTENT["y"], TEX_RES)
    tex, inside = gd.sample(cam, plate_u)
    valid = inside & (gd.sample(cam, valid_u.astype(np.float32), interp=cv2.INTER_NEAREST)[0] > 0.5)
    lin = srgb_to_lin(tex)
    E = irradiance(L, gd.points(), (0, 0, 1))
    roi = ((gd.X > 0.0) & (gd.X < a + b * gd.Y - 0.02) & (gd.Y > -0.16) & (gd.Y < 0.30) & (np.hypot(gd.X, gd.Y) > 0.13)) & valid
    Yl = cv2.blur(luma(lin), (41, 41))
    c1, c2 = np.linalg.lstsq(np.c_[E[roi], np.ones(roi.sum())], Yl[roi], rcond=None)[0]
    for _ in range(3):  # robust: drop shadowed / bright-grain outliers
        r = Yl[roi] - (c1 * E[roi] + c2)
        k = np.abs(r) < 2.5 * np.median(np.abs(r))
        c1, c2 = np.linalg.lstsq(np.c_[E[roi][k], np.ones(k.sum())], Yl[roi][k], rcond=None)[0]
    c2 = max(c2, 0.0)
    shade = c1 * E + c2
    ambient_frac = float(c2 / (c1 * E[roi].mean() + c2))
    norm = float(shade[roi].mean())
    print(f"[lamp] desk luminance ~ {c1:.4f}*E + {c2:.4f}: ambient share {ambient_frac:.2f} on the desk top; lamp "
          f"irradiance falls {E[roi].max() / E[roi].min():.1f}x across it", flush=True)
    desk_alb = mirror_fill(lin * (norm / shade)[..., None], valid) * grain_detail(lin.shape[:2], 0)[..., None]
    desk_png = lin_to_srgb(desk_alb).astype(np.uint8)
    save(out / "desk_albedo.png", desk_png)
    save(dbg / "desk_texture_lit.png", mirror_fill(tex, valid))

    # --- USD: st (0,0) = texture bottom-left
    desk_quad = [(gd.x0, gd.y1, DESK_Z), (gd.x0, gd.y0, DESK_Z), (gd.x1, gd.y0, DESK_Z), (gd.x1, gd.y1, DESK_Z)]
    wall_quad = [(wall_x, gd.y1, 0.0), (wall_x, gd.y0, 0.0), (wall_x, gd.y0, WALL_HEIGHT), (wall_x, gd.y1, WALL_HEIGHT)]
    write_backdrop_usda(out / "backdrop.usda", desk_quad, wall_quad, (-1, 0, 0), WALL_COLOR)
    print(f"[backdrop] desk quad x [{gd.x0}, {gd.x1}] y [{gd.y0}, {gd.y1}] ({desk_png.shape[1]}x{desk_png.shape[0]} px), "
          f"wall at x = {wall_x}", flush=True)

    # --- alternative woods (appearance DR)
    desk_med = np.median(desk_png[roi], axis=0)
    alts = []
    for k, (gain, seed) in enumerate([((0.75, 0.75, 0.75), 1), ((1.45, 1.3, 1.15), 2), ((1.1, 0.85, 0.75), 3)]):
        name = f"desk_alt_{k}.png"
        save(out / name, procedural_wood(desk_png.shape[:2], np.clip(desk_med * gain, 0, 255), seed))
        alts.append(name)

    # --- statistics
    wrist = []
    for e in range(n_ep):
        wrist += [cv2.undistort(f, wK, wD) for f in episode_frames(args.dataset, e, "wrist", args.wrist_step)]
    stats = dict(
        source=str(args.dataset), episodes=n_ep, built=time.strftime("%Y-%m-%d %H:%M"),
        lamp=dict(pos=np.round(L, 3).tolist(), shadow_iou=round(float(iou), 3), ambient_share=round(ambient_frac, 3),
                  shade_fit=[float(c1), float(c2)]),
        desk_back_edge=dict(a=a, b=b), desk_quad=np.round(desk_quad, 4).tolist(), wall_quad=np.round(wall_quad, 4).tolist(),
        colors=dict(cube=np.round(np.median([c for c in cube_rgb if c], 0), 1).tolist() if cube_rgb else None,
                    cup_inner=np.round(np.median([c for c in cup_in_rgb if c], 0), 1).tolist() if cup_in_rgb else None,
                    cup_outer=np.round(np.median([c for c in cup_out_rgb if c], 0), 1).tolist() if any(cup_out_rgb) else None,
                    desk_albedo=np.round(desk_med, 1).tolist()),
        textures=dict(desk="desk_albedo.png", alternatives=alts),
        image_stats=dict(scene=scene_all, wrist=image_stats(wrist)),
        per_episode=episodes,
    )
    (out / "real_stats.json").write_text(json.dumps(stats, indent=1))
    print(f"[stats] scene {stats['image_stats']['scene']}  wrist {stats['image_stats']['wrist']}")
    print(f"[stats] colours (sRGB, overhead): {stats['colors']}")
    print(f"wrote {out}/backdrop.usda, textures, real_stats.json; debug images in {dbg} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

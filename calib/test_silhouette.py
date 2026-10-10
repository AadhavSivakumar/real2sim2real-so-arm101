"""Tests for calib/silhouette.py (marker-free overhead-camera pose from the robot's silhouette). CPU only.

  a  unit checks: full-tree FK vs fk.py, LeRobot->sim units, projection vs cv2.projectPoints (incl. distortion;
     unsupported thin-prism/tilt terms raise), rasteriser (vs cv2.fillConvexPoly, z-buffer, area, back-face culling,
     scaled intrinsics, decimation), robot pixel length, verdict / joint-offset-check logic, segmentation
     (synthetic dark desk + the real dark overhead frame, which has no joint state)
  b  self-consistency: the solver's own renders (undecimated meshes, real-like camera 1.4 m up with non-placeholder
     K + distortion, dark-room colours, shadows, clutter incl. a grey-tile area bigger than the arm, noise, JPEG)
     from a poor initial guess -> truth (+ objective at truth vs solution: bias or search failure?); (full run)
     sensitivity to 2-3 deg joint-calibration offsets, the joint-offset fit on those readings (recovers the
     camera, check flags them) and on correct readings (no false alarm), the whole-image coarse search forced
     on (clutter candidates must lose), and a ~17 deg tilted camera with the arm at the bottom of the image
     solved from solve.py scene_robot's default start
  c  sim demos run0, several episodes, poor initial guess (1.4 m, yaw +20 deg, x/y +5 cm), mode 'class':
     error vs the placeholder pose, spread across episodes, mean pose, within-episode repeatability (two
     disjoint frame sets). NOTE: the demos jitter the camera per episode by N(0, 1 cm) per axis and N(0, 1 deg)
     per axis, so the error vs placeholder is bounded below by that jitter (~1.6 cm / ~1.6 deg on average).
  d  darkened-sim stress test mimicking the real Astra (dark-brown wood desk, warm light, gamma, noise, JPEG,
     mean brightness 33/255) in 'white' mode (what solve.py scene_robot uses), plus a 0.6x-downscaled
     variant (the robot's pixel footprint at the real camera's 1.4 m); each compared with the bright 'class'
     solve of the same episode (same camera -> a jitter-free measure of what the darkness costs)
  e  end-to-end: solve.py scene_robot --dry_run on synthetic snapshot dirs in the scratch dir (incl. the default
     joint-offset check); (full run) in-process with solve._update_calib replaced by a recorder: no intrinsics
     file (placeholder K -> WARN) + --init_yaw 180 (automatic full-yaw retry) + --force (pose written together
     with K/dist). calib.json must stay byte-identical

Run (from the repo root so101_r2s2r/):
  ~/TestingGrounds/robot_splat/.venv/bin/python calib/test_silhouette.py             # full
  ~/TestingGrounds/robot_splat/.venv/bin/python calib/test_silhouette.py --quick     # fewer episodes
Options: --parts abcde  --jobs N (parallel episode solves, default 3)  --out DIR  --episodes ...  --verbose
Writes only under --out (default: the session scratchpad); never writes calib/data/. Exit code 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fk  # noqa: E402
import silhouette as S  # noqa: E402

RUN0 = Path("/mnt/secondary/so101_r2s/sim_demos/run0")
SCRATCH = Path("/tmp/claude-1000/-home-asubuntudesktop-TestingGrounds/3f115a7c-614b-4117-a8ff-7da2d63a96e4/scratchpad")
REAL_FRAME = SCRATCH / "scene_now.jpg"
REAL_ROBOT_BOX = (315, 228, 410, 422)  # x0, y0, x1, y1 of the arm in REAL_FRAME (hand-annotated)
CALIB_JSON = HERE / "data" / "calib.json"
# placeholder overhead camera the demos were rendered with (so101_r2s2r/calib.py), before the per-episode jitter
POS_PH = np.array([0.22, 0.0, 0.85])
R_PH = S.look_at(POS_PH, (0.22, 0.0, 0.0), up=(1.0, 0.0, 0.0))
K_PH = S.PLACEHOLDER_K.copy()
JITTER_POS_STD, JITTER_ROT_STD_DEG = 0.01, 1.0
REAL_CAST = (0.65, 0.92, 1.0)  # BGR of the white PLA under the real room's warm light (measured on REAL_FRAME)

RESULTS: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{'  -- ' + detail if detail else ''}", flush=True)


def header(t):
    print(f"\n=== {t} ===", flush=True)


# ----------------------------------------------------------------------------- data helpers
def read_episode(ep, frames=None, run=RUN0):
    """(q (T,6) sim rad, {frame index: BGR}) for the requested frame indices of one demo episode."""
    d = run / f"ep_{ep:06d}"
    q = np.load(d / "data.npz")["state"].astype(float)
    if frames is None:
        return q, {}
    want, out = set(int(i) for i in frames), {}
    cap = cv2.VideoCapture(str(d / "scene.mp4"))
    i = 0
    while len(out) < len(want):
        ok, f = cap.read()
        if not ok:
            break
        if i in want:
            out[i] = f
        i += 1
    return q, out


def pick_frames(q, n=6, w=(1, 1, 1, 1, 1, 0.3)):
    """n arm poses far apart in joint space (farthest-point sampling) among the frames where the arm is
    (nearly) still, so image/state timing and motion blur do not matter."""
    v = np.r_[0, np.abs(np.diff(q[:, :5], axis=0)).sum(1)]
    v = np.convolve(v, np.ones(5) / 5, "same")
    cand = np.nonzero(v <= np.percentile(v, 40))[0]
    X = q[cand] * np.asarray(w)
    i = int(np.argmax(np.linalg.norm(X - X.mean(0), axis=1)))
    chosen, d = [i], np.linalg.norm(X - X[i], axis=1)
    while len(chosen) < n:
        i = int(np.argmax(d))
        chosen.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
    return sorted(int(cand[i]) for i in chosen)


def poor_init(pos_true, R_true, height=1.4, yaw_off_deg=20.0, xy_off=0.05):
    """Nadir camera at `height`, image-up rotated by yaw_off, x and y both off by xy_off."""
    R = S.nadir_rotation(S.yaw_of(R_true) + np.radians(yaw_off_deg))
    return R, np.array([pos_true[0] + xy_off, pos_true[1] + xy_off, height])


def wood(shape, rng, base_bgr=(4, 38, 69)):
    """Dark-brown wood texture (float BGR 0..1, the real desk's colour): grain along x + fine noise."""
    H, W = shape
    g = cv2.resize(rng.normal(0, 1, (max(4, H // 6), 6)).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
    fine = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 1.0)
    t = 1 + 0.18 * g / (g.std() + 1e-6) + 0.05 * fine / (fine.std() + 1e-6)
    return np.asarray(base_bgr, np.float32)[None, None] / 255 * t[..., None]


def camera_finish(f, rng, gamma=1.15, target_mean=33.0, noise=2.5, jpeg=75):
    """Gamma, global exposure to a mean of target_mean/255, sensor noise, JPEG (4:2:0 chroma) -> uint8 BGR."""
    out = np.clip(f, 0, 1) ** gamma
    out = out * (target_mean / 255 / max(float(out.mean()), 1e-6))
    out = out + rng.normal(0, noise / 255, out.shape).astype(np.float32)
    out = (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
    return cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)


def darken_like_real(img, rng, scale=1.0, gain=0.62, cast=REAL_CAST, **finish):
    """Sim overhead frame -> real-camera look: the light-tan desk (pixels of the desk's chromaticity) becomes
    dark-brown wood carrying the sim's shading (shadows); everything else is dimmed and tinted by the warm room
    light; then camera_finish (gamma, mean brightness 33, noise, JPEG). scale < 1 shrinks the frame and pastes it
    centred on more wood, mimicking the smaller robot footprint of a higher camera. Returns (image, its K)."""
    f = img.astype(np.float32) / 255
    V = f.max(2)
    chrom = f / np.maximum(f.sum(2, keepdims=True), 1e-3)
    dist = np.linalg.norm(chrom - np.median(chrom.reshape(-1, 3), 0), axis=2)
    a = (np.clip(1 - (dist - 0.015) / 0.03, 0, 1) * (V > 0.08))[..., None]  # soft desk mask
    vd = float(np.median(V[a[..., 0] > 0.9]))
    comp = a * wood(V.shape, rng) * (V / vd)[..., None] + (1 - a) * f * gain * np.asarray(cast, np.float32)
    K = K_PH.copy()
    if scale != 1.0:
        H, W = V.shape
        w, h = int(round(W * scale)), int(round(H * scale))
        ox, oy = (W - w) // 2, (H - h) // 2
        canvas = wood((H, W), rng)
        canvas[oy:oy + h, ox:ox + w] = cv2.resize(comp, (w, h), interpolation=cv2.INTER_AREA)
        comp = canvas
        K = S.scale_K(K_PH, w / W)
        K[0, 2] += ox
        K[1, 2] += oy
    return camera_finish(comp, rng, **finish), K


def pose_errors(res, R_t, pos_t, K, dist=None):
    dp = np.asarray(res["pos"]) - pos_t
    rm, rx = S.reprojection_discrepancy(R_t, pos_t, res["R"], res["pos"], K, dist)
    return dict(dpos_cm=dp * 100, pos_err_cm=float(np.linalg.norm(dp) * 100), rot_err_deg=S.rot_angle_deg(res["R"], R_t),
                reproj_mean_px=rm, reproj_max_px=rx, iou=res["iou_mean"], iou_min=min(p["iou"] for p in res["per_snap"]),
                boundary_px=res["boundary_px_mean"], runtime_s=res["runtime_s"], start=res["start"],
                diversity=res.get("pose_diversity"),
                pos=np.asarray(res["pos"]), R=np.asarray(res["R"]))


def fmt(e):
    return (f"dpos cm {np.array2string(e['dpos_cm'], precision=2, suppress_small=True)} |{e['pos_err_cm']:.2f}| "
            f"rot {e['rot_err_deg']:.2f} deg  img {e['reproj_mean_px']:.1f}/{e['reproj_max_px']:.1f} px  "
            f"IoU {e['iou']:.3f} (min {e['iou_min']:.3f})  bnd {e['boundary_px']:.2f} px  {e['runtime_s']:.0f}s  [{e['start']}"
            + (f", diversity {e['diversity']:.2f}" if e.get("diversity") is not None else "") + "]")


def summarize(errs, label):
    D = np.array([e["dpos_cm"] for e in errs])
    rot = np.array([e["rot_err_deg"] for e in errs])
    Rm = Rotation.from_matrix(np.array([e["R"] for e in errs])).mean().as_matrix()
    rot_dev = np.array([Rotation.from_matrix(Rm.T @ e["R"]).as_rotvec(degrees=True) for e in errs])  # camera axes
    mean_pos_err = float(np.linalg.norm(D.mean(0)))
    mean_rot_err = S.rot_angle_deg(Rm, R_PH)
    print(f"  {label}: {len(errs)} episodes")
    print(f"    per-episode |dpos| mean {np.linalg.norm(D, axis=1).mean():.2f} max {np.linalg.norm(D, axis=1).max():.2f} cm;  "
          f"rot err mean {rot.mean():.2f} max {rot.max():.2f} deg;  image discrepancy vs placeholder mean "
          f"{np.mean([e['reproj_mean_px'] for e in errs]):.1f} px")
    print(f"    spread (std over episodes): pos {np.round(D.std(0), 2).tolist()} cm (desk x,y,z), "
          f"rot {np.round(rot_dev.std(0), 2).tolist()} deg (camera x,y,z)  [demo jitter: 1 cm, 1 deg per axis]")
    print(f"    MEAN pose over episodes: offset {np.round(D.mean(0), 2).tolist()} cm (|{mean_pos_err:.2f}|), "
          f"rotation {mean_rot_err:.2f} deg from the placeholder")
    print(f"    IoU mean {np.mean([e['iou'] for e in errs]):.3f} (min snapshot {min(e['iou_min'] for e in errs):.3f}); "
          f"boundary mean {np.mean([e['boundary_px'] for e in errs]):.2f} px; solve time mean "
          f"{np.mean([e['runtime_s'] for e in errs]):.1f} s")
    return D, rot, mean_pos_err, mean_rot_err


# ----------------------------------------------------------------------------- episode worker (process pool)
def solve_episode(task):
    """task = dict(ep, n, frames, darken (None | kwargs), mode, overlay) -> pose errors (+ overlay image)."""
    ep = task["ep"]
    q, _ = read_episode(ep)
    idx = task.get("frames") or pick_frames(q, task.get("n", 6))
    _, imgs = read_episode(ep, idx)
    rng = np.random.default_rng(1000 + ep)
    K, snaps = K_PH, []
    for i in idx:
        img = imgs[i]
        if task.get("darken") is not None:
            img, K = darken_like_real(img, rng, **task["darken"])
        snaps.append(S.Snapshot(img, q[i], f"ep{ep} f{i}"))
    init_R, init_pos = poor_init(POS_PH, R_PH)
    res, solver = S.solve_pipeline(snaps, K, None, init_R, init_pos, mode=task.get("mode", "class"),
                                   verbose=task.get("verbose", False))
    e = pose_errors(res, R_PH, POS_PH, K)
    e["verdict"] = S.verdict(res)
    e["ep"], e["K"] = ep, K
    if "pass1" in res:
        e["pass2_move_cm"] = float(np.linalg.norm(res["pos"] - res["pass1"]["pos"]) * 100)
    if task.get("overlay"):
        e["overlay"] = solver.overlay(res["R"], res["pos"])
    return e


def run_tasks(tasks, jobs):
    if jobs > 1 and len(tasks) > 1:
        with mp.get_context("fork").Pool(min(jobs, len(tasks))) as pool:
            return pool.map(solve_episode, tasks)
    return [solve_episode(t) for t in tasks]


# ----------------------------------------------------------------------------- a: unit checks
def part_a(args):
    header("a: unit checks")
    rng = np.random.default_rng(0)
    model = S.get_model()
    print(f"  model: {model.n_tris_full} -> {model.n_tris} triangles (clustering cell {model.cell * 1000:.1f} mm)")
    names = ["shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link", "gripper_frame_link"]
    err = 0.0
    for _ in range(50):
        q = rng.uniform(-2.0, 2.0, 6)
        T = model.link_transforms(q)
        for n, P in zip(names, fk.link_poses(q[:5])):
            err = max(err, np.abs(T[n][:3, :3] - P[:3, :3]).max(), np.abs(T[n][:3, 3] - (P[:3, 3] + fk.T_WORLD_URDF)).max())
    check("full-tree FK == fk.py on the arm chain (50 random q, 6 frames)", err < 1e-9, f"max diff {err:.1e}")
    q = rng.uniform(-1, 1, 6)
    q2 = q.copy()
    q2[5] += 0.5
    T1, T2 = model.link_transforms(q), model.link_transforms(q2)
    moved = {n for n in T1 if not np.allclose(T1[n], T2[n])}
    jaw = "moving_jaw_so101_v1_link"
    ang = S.rot_angle_deg(T1[jaw][:3, :3], T2[jaw][:3, :3])
    check("gripper joint turns only the jaw link, by the joint angle", moved == {jaw} and abs(ang - np.degrees(0.5)) < 1e-6,
          f"moved {sorted(moved)}, {ang:.3f} deg")
    check("fingertip via full tree == fk.fingertip_in_world",
          np.allclose(T1["gripper_frame_link"][:3, 3], fk.fingertip_in_world(q[:5]), atol=1e-12))
    lr = {f"{j}.pos": v for j, v in zip(S.JOINTS, [10.0, -20.0, 30.0, -40.0, 50.0, 50.0])}
    qs = S.q_from_follower(lr)
    exp = np.r_[np.radians([10, -20, 30, -40, 50]), -0.175 + 0.5 * (1.745 + 0.175)]
    check("state.json follower (LeRobot deg, gripper 0..100) -> sim rad", np.allclose(qs, exp, atol=1e-5),
          f"{np.round(qs, 4).tolist()}")

    # projection vs OpenCV
    K = np.array([[545.0, 0.0, 318.0], [0.0, 543.0, 246.0], [0, 0, 1]])
    worst = {}
    for dist in (None, np.array([0.08, -0.2, 0.001, -0.002, 0.06]), np.array([0.08, -0.2, 0.001, -0.002, 0.06, 0.03, -0.02, 0.01])):
        e = 0.0
        for _ in range(10):
            pos = np.r_[rng.uniform(0.0, 0.5, 2), rng.uniform(0.6, 1.6)]
            R = S.look_at(pos, np.r_[rng.uniform(0.0, 0.4, 2), 0.0], up=(1.0, rng.uniform(-0.5, 0.5), 0.0))
            R = R @ Rotation.from_rotvec(rng.normal(0, 0.1, 3)).as_matrix()
            X = np.c_[rng.uniform(-0.1, 0.5, (200, 2)), rng.uniform(0.0, 0.3, 200)]
            uv, _ = S.project(X, R, pos, K, dist)
            ref, _ = cv2.projectPoints(X, cv2.Rodrigues(R.T)[0], -R.T @ pos, K, np.zeros(5) if dist is None else dist)
            e = max(e, float(np.abs(uv - ref[:, 0]).max()))
        worst[0 if dist is None else len(dist)] = float(f"{e:.1e}")
    check("project() == cv2.projectPoints (0 / 5 / 8 distortion coefs)", max(worst.values()) < 1e-6, f"max |diff| px {worst}")
    d8 = np.array([0.08, -0.2, 0.001, -0.002, 0.06, 0.03, -0.02, 0.01])
    same = np.abs(S.project(X, R, pos, K, np.r_[d8, np.zeros(6)])[0] - S.project(X, R, pos, K, d8)[0]).max()
    try:
        S.project(X, R, pos, K, np.r_[d8, 0, 0, 0, 0, 0.01, 0.0])  # 14 coefs with a nonzero tilt term
        raised = False
    except ValueError:
        raised = True
    check("project(): zero-padded 12/14 coefs accepted, nonzero thin-prism/tilt terms raise (not silently dropped)",
          same == 0 and raised, f"padded diff {same:.1e} px, ValueError {raised}")

    # rasteriser
    d = np.load(RUN0 / "ep_000000" / "data.npz")["state"]
    worst_frac, worst_far, n_hole = 0.0, 0, 0
    for t in (0, 200, 400):
        V, F, L = model.world_mesh(d[t])
        mine = S.render(V, F, L, R_PH, POS_PH, K_PH, None, S.IMAGE_SIZE, labels=False, cull=False)
        uv, _ = S.project(V, R_PH, POS_PH, K_PH, None)
        ref = np.zeros((480, 640), np.uint8)
        for f_ in F:
            cv2.fillConvexPoly(ref, np.round(uv[f_] * 16).astype(np.int32), 1, lineType=cv2.LINE_8, shift=4)
        ref = ref > 0
        x = mine ^ ref
        edge = cv2.morphologyEx(ref.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        # cv2 fills every pixel a triangle's outline touches, which closes the arm's small through-holes (screw /
        # vent holes); ours samples pixel centres and keeps them open -> differences inside our holes are expected
        n, cc = cv2.connectedComponents((~mine).astype(np.uint8), connectivity=4)
        hole = ~mine & ~np.isin(cc, np.unique(np.r_[cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
        worst_frac = max(worst_frac, x.sum() / ref.sum())
        worst_far = max(worst_far, int((x & ~edge & ~hole).sum()))
        n_hole += int((x & ~edge & hole).sum())
    check("rasteriser silhouette vs cv2.fillConvexPoly (3 robot poses)", worst_frac < 0.035 and worst_far == 0,
          f"xor {100 * worst_frac:.2f}% of the area (pixel-centre vs outline convention), 0 px expected away from the "
          f"boundary except inside mesh through-holes: {worst_far} (+{n_hole} in holes)")
    sq = lambda c, z, s: np.array([[c[0] - s, c[1] - s, z], [c[0] + s, c[1] - s, z], [c[0] + s, c[1] + s, z], [c[0] - s, c[1] + s, z]])  # noqa: E731
    Vz = np.r_[sq((0.0, 0.0), 0.10, 0.05), sq((0.03, 0.03), 0.20, 0.05)]  # world z up; camera 1 m above, looking down
    Fz = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])
    Lz = np.array([1, 1, 2, 2], np.uint8)
    Rz, pz = S.nadir_rotation(0.0), np.array([0.0, 0.0, 1.0])
    a1 = S.render(Vz, Fz, Lz, Rz, pz, K_PH, None, S.IMAGE_SIZE, cull=False)
    a2 = S.render(Vz, Fz[::-1], Lz[::-1], Rz, pz, K_PH, None, S.IMAGE_SIZE, cull=False)
    ctr = S.project(np.array([[0.04, 0.04, 0.2]]), Rz, pz, K_PH)[0][0].round().astype(int)
    check("z-buffer: the nearer surface wins, independent of triangle order",
          (a1 == a2).all() and a1[ctr[1], ctr[0]] == 2 and (a1 == 2).sum() > (a1 == 1).sum() > 0,
          f"label at the overlap {a1[ctr[1], ctr[0]]}")
    side = 525.0 * 0.1 / 0.9  # 0.1 m square seen from 0.9 m
    area = int(S.render(Vz[:4], Fz[:2], Lz[:2], Rz, pz, K_PH, None, S.IMAGE_SIZE, labels=False, cull=False).sum())
    check("rasterised area of a square == (f s / Z)^2 within one edge row", abs(area - side ** 2) < 2 * side + 1,
          f"{area} px vs {side ** 2:.0f}")
    V, F, L = model.world_mesh(d[200])
    c1 = S.render(V, F, L, R_PH, POS_PH, K_PH, None, S.IMAGE_SIZE, labels=False, cull=True)
    c0 = S.render(V, F, L, R_PH, POS_PH, K_PH, None, S.IMAGE_SIZE, labels=False, cull=False)
    check("back-face culling leaves the silhouette unchanged", (c1 ^ c0).sum() <= 0.002 * c0.sum(),
          f"{int((c1 ^ c0).sum())} px differ of {int(c0.sum())}")
    half = S.render(V, F, L, R_PH, POS_PH, S.scale_K(K_PH, 0.5), None, (320, 240), labels=False)
    down = cv2.resize(c1.astype(np.float32), (320, 240), interpolation=cv2.INTER_AREA) > 0.5
    iou = (half & down).sum() / (half | down).sum()
    check("scale_K: half-resolution render == downsampled full render", iou > 0.95, f"IoU {iou:.3f}")
    full = S.get_model(max_tris=None)
    Vf, Ff, Lf = full.world_mesh(d[200])
    cf = S.render(Vf, Ff, Lf, R_PH, POS_PH, K_PH, None, S.IMAGE_SIZE, labels=False)
    iou = (cf & c1).sum() / (cf | c1).sum()
    check(f"decimated mesh ({model.n_tris} tris) vs undecimated ({full.n_tris}) silhouette", iou > 0.97, f"IoU {iou:.3f}")
    t = time.perf_counter()
    for _ in range(30):
        S.render(V, F, L, R_PH, POS_PH, K_PH, None, S.IMAGE_SIZE)
    print(f"  render 640x480 with z-buffered labels: {(time.perf_counter() - t) / 30 * 1e3:.1f} ms")
    rect = np.zeros((480, 640), bool)
    rect[100:161, 200:281] = True  # 81 x 61 px -> corner-to-corner 80 x 60 px apart
    check("feret_px (robot length in the image) of a rectangle == its diagonal", abs(S.feret_px(rect) - 100.0) < 1e-9,
          f"{S.feret_px(rect):.2f} px")

    # verdict / joint-offset check logic (no solve: synthetic result dicts)
    good = dict(per_snap=[dict(name=f"s{i}", iou=0.9, boundary_px=0.8) for i in range(5)], R=R_PH, pos=POS_PH,
                arm_px_mean=180.0)
    rj = dict(joint_offsets_deg=np.array([0, -3.0, 2.0, 1.0, 0, 0]), offset_joints=["shoulder_lift", "elbow_flex", "wrist_flex"],
              iou_mean=0.93, iou_mean_no_offsets=0.86, R=R_PH, pos=POS_PH + 0.03, R_no_offsets=R_PH, pos_no_offsets=POS_PH)
    flagged = S.joint_offset_check(rj)
    clean = S.joint_offset_check(dict(rj, joint_offsets_deg=np.array([0, -0.4, 0.3, 0.5, 0, 0]), iou_mean=0.862))
    v = [S.verdict(good)[0], S.verdict(good, intrinsics_calibrated=False)[0], S.verdict(dict(good, arm_px_mean=80.0))[0],
         S.verdict(good, offset_check=flagged)[0], S.verdict(good, offset_check=clean)[0]]
    check("verdict: PASS; WARN for placeholder intrinsics, a robot < 100 px long, flagged joint offsets; clean check PASS",
          v == ["PASS", "WARN", "WARN", "WARN", "PASS"] and flagged["flag"] and not clean["flag"]
          and abs(flagged["move_cm"] - 100 * np.linalg.norm([0.03] * 3)) < 1e-9, f"{v}; {flagged['message']}")

    # segmentation, synthetic dark scene
    img = wood((480, 640), rng)
    img[200:260, 300:360] = np.array(REAL_CAST, np.float32) * 0.55  # warm-lit white PLA
    img[300:330, 100:130] = (0.02, 0.03, 0.035)  # black servo
    img[100:130, 450:480] = (0.10, 0.30, 0.55)  # orange cube
    sg = S.segment(camera_finish(img, rng))
    wmask = np.zeros((480, 640), bool)
    wmask[205:255, 305:355] = True
    desk = np.ones((480, 640), bool)
    desk[190:270, 290:370] = desk[290:340, 90:140] = desk[90:140, 440:490] = False
    check("segment(): warm-lit white part on dark wood is 'white'", (sg["white"][wmask] > 0.5).mean() > 0.97,
          f"{100 * (sg['white'][wmask] > 0.5).mean():.1f}% of it")
    check("segment(): wood desk and orange cube are not robot",
          (sg["robot"][desk] < 0.5).mean() > 0.98 and (sg["white"][105:125, 455:475] < 0.5).all(),
          f"desk {100 * (sg['robot'][desk] < 0.5).mean():.1f}% non-robot")
    if not REAL_FRAME.exists():
        print(f"  (real frame {REAL_FRAME} missing: skipped)")
        return
    real = cv2.imread(str(REAL_FRAME))
    x0, y0, x1, y1 = REAL_ROBOT_BOX
    box = np.zeros(real.shape[:2], bool)
    box[y0:y1, x0:x1] = True
    sg = S.segment(real)
    sg_old = S.segment(real, white_balance=False, s_max=0.3)  # the segmenter before white balancing
    w = sg["white"] > 0.5
    inside, inside_old = int((w & box).sum()), int(((sg_old["white"] > 0.5) & box).sum())
    check("real frame: illuminant estimated warm (blue gain < 0.8)", sg["illum"][0] < 0.8,
          f"illum BGR {np.round(sg['illum'], 2).tolist()}, mean brightness {real.mean():.1f}")
    check("real frame: white balance recovers the arm's printed parts (> 3x the unbalanced segmenter)",
          inside > 3 * inside_old, f"'white' px in the arm box {inside} vs {inside_old} before")
    # pass-2-like ROI (arm box grown x1.6 + 30 px, as roi_from_pose would give): the arm must dominate the
    # 'white' pixels there (cup rim, cube edges, cables, the paper scrap are the competing clutter)
    cxb, cyb, hw = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) / 2 * 1.6 + 30
    roi = np.zeros_like(box)
    roi[max(0, int(cyb - hw)):int(cyb + hw), max(0, int(cxb - hw)):int(cxb + hw)] = True
    wl = S.segment(real, roi=roi)["white"] > 0.5
    frac = (wl & box).sum() / max((wl & roi).sum(), 1)
    check("real frame: inside an ROI around the arm, >= 60% of 'white' px are on the arm", frac >= 0.6,
          f"{100 * frac:.0f}% ({int((wl & box).sum())} of {int((wl & roi).sum())})")
    n, _, st, cen = cv2.connectedComponentsWithStats(w.astype(np.uint8))
    top = sorted(range(1, n), key=lambda i: -st[i, cv2.CC_STAT_AREA])[:4]
    print("  real frame, largest 'white' blobs over the whole image (px @ centroid): "
          + ", ".join(f"{st[i, cv2.CC_STAT_AREA]} @ ({cen[i][0]:.0f},{cen[i][1]:.0f})" for i in top)
          + "  -- the grey floor tiles / keyboard at the top are bigger than the arm; they are static, so the coarse "
            "search's moving-parts target ignores them (part b has such a tile area)")
    args.real_seg = (real, sg)


# ----------------------------------------------------------------------------- b: self-consistency
def synth_frames(model, qs, R, pos, K, dist, rng, ss=2, size=S.IMAGE_SIZE):
    """Dark-room renders by our own rasteriser (2x supersampled, undecimated meshes): warm-lit white printed parts
    with smooth shading noise, black servos, offset shadow, dark wood, clutter, camera finish."""
    W, H = size
    tint = np.asarray(REAL_CAST, np.float32)
    out = []
    for i, q in enumerate(qs):
        V, F, L = model.world_mesh(q)
        lab = S.render(V, F, L, R, pos, S.scale_K(K, ss), dist, (W * ss, H * ss))
        img = wood((H * ss, W * ss), rng)
        img[np.roll(lab > 0, (14 * ss, 10 * ss), (0, 1))] *= 0.45  # shadow
        shade = cv2.GaussianBlur(rng.uniform(0.75, 1.0, (H * ss, W * ss)).astype(np.float32), (0, 0), 6 * ss)
        img[lab == S.PRINTED] = (0.62 * tint)[None] * shade[lab == S.PRINTED][:, None]
        img[lab == S.SERVO] = (0.025, 0.03, 0.035)
        cv2.circle(img, (110 * ss, 120 * ss), 22 * ss, tuple(float(c) for c in 0.6 * tint), 4 * ss)  # white cup rim
        cv2.ellipse(img, (80 * ss, 400 * ss), (40 * ss, 70 * ss), 20, 0, 360, (0.02, 0.025, 0.03), -1)  # black leader arm
        cv2.rectangle(img, (520 * ss, 60 * ss), (545 * ss, 85 * ss), (0.12, 0.33, 0.6), -1)  # orange cube
        # grey floor tiles behind the desk (bigger than the arm, achromatic, bright -- as in the real frame)
        cv2.rectangle(img, (120 * ss, 0), (400 * ss, 75 * ss), tuple(float(c) for c in 0.5 * tint), -1)
        for gx in range(120, 400, 45):
            cv2.line(img, (gx * ss, 0), (gx * ss, 75 * ss), (0.02, 0.03, 0.035), 2 * ss)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        out.append(S.Snapshot(camera_finish(img, rng, gamma=1.0), q, f"synth{i}"))
    return out


def part_b(args):
    header("b: self-consistency on the solver's own renders (real-like camera 1.4 m up, K != placeholder, distortion)")
    rng = np.random.default_rng(1)
    pos_t = np.array([0.45, 0.08, 1.40])
    R_t = S.look_at(pos_t, (0.32, 0.03, 0.0), up=(1.0, 0.08, 0.0))  # ~6 deg off nadir, ~5 deg roll
    K = np.array([[545.0, 0.0, 318.0], [0.0, 543.0, 246.0], [0, 0, 1]])
    dist = np.array([0.05, -0.12, 0.0008, -0.0012, 0.04])
    qall = np.concatenate([read_episode(e)[0] for e in (10, 11, 12)])
    qs = qall[pick_frames(qall, 5)]
    snaps = synth_frames(S.get_model(max_tris=None), qs, R_t, pos_t, K, dist, rng)
    init_R, init_pos = poor_init(pos_t, R_t, height=1.0)
    print(f"  truth pos {pos_t.tolist()}, {np.degrees(np.arccos(-R_t[2, 2])):.1f} deg off nadir; init pos {init_pos.tolist()} "
          f"(nadir, yaw +20 deg); 5 arm poses; mean brightness {np.mean([s.image.mean() for s in snaps]):.1f}")
    res, solver = S.solve_pipeline(snaps, K, dist, init_R, init_pos, mode="white", verbose=args.verbose)
    e = pose_errors(res, R_t, pos_t, K, dist)
    print("  " + fmt(e))
    # is the residual a search failure or a bias of the objective itself? (dark JPEG edges: mixed wood/white
    # pixels with < 50% white coverage still read 'white', so the segmented arm is ~0.2 px fatter all round)
    if e["pos_err_cm"] < 5:  # (the pass-2 ROI is built around the solution, so only meaningful near the truth)
        s_truth = solver.score(R_t, pos_t, solver.level(1.0, 1.0))
        print(f"  objective at the solution {res['score']:.4f} vs at the truth {s_truth:.4f} -> "
              + ("the remaining offset is a bias of the objective (segmentation), not a search failure"
                 if res["score"] >= s_truth else "the search stopped short of the optimum"))
    check("b: pose recovered (pos < 1.5 cm, rot < 0.5 deg, image discrepancy < 1 px)",
          e["pos_err_cm"] < 1.5 and e["rot_err_deg"] < 0.5 and e["reproj_mean_px"] < 1.0,
          f"{e['pos_err_cm']:.2f} cm, {e['rot_err_deg']:.2f} deg, {e['reproj_mean_px']:.2f} px")
    v, why = S.verdict(res)
    check("b: verdict PASS", v == "PASS", "; ".join(why))
    if args.quick:
        return
    # sensitivity: the solver is told joint angles with a systematic calibration error (real LeRobot zero vs URDF)
    off = np.radians([1.5, 3.0, -2.0, 2.0, 0.0, 0.0])
    bad = [S.Snapshot(s.image, s.q + off, s.name) for s in snaps]
    res2, sol2 = S.solve_pipeline(bad, K, dist, init_R, init_pos, mode="white", verbose=args.verbose)
    e2 = pose_errors(res2, R_t, pos_t, K, dist)
    print(f"  with joint offsets {np.round(np.degrees(off), 1).tolist()} deg in the joint readings: " + fmt(e2))
    print(f"    -> verdict without the joint-offset check {S.verdict(res2)[0]} {'; '.join(S.verdict(res2)[1])}")
    check("b: 2-3 deg joint calibration offsets keep the image discrepancy < 3 px (informational bound)",
          e2["reproj_mean_px"] < 3.0, f"{e2['reproj_mean_px']:.2f} px, {e2['pos_err_cm']:.2f} cm, {e2['rot_err_deg']:.2f} deg")
    # ... and what solve.py scene_robot's joint-offset stage makes of it (lift/elbow/wrist_flex fitted; the 1.5 deg
    # pan offset is not separable from a camera yaw about the base and stays in the rotation)
    rj = sol2.fit_joint_offsets(res2["R"], res2["pos"])
    ej, chk = pose_errors(rj, R_t, pos_t, K, dist), S.joint_offset_check(rj)
    fit_err = np.abs(rj["joint_offsets_deg"][1:4] + np.degrees(off[1:4]))
    print(f"    joint-offset fit: offsets {np.round(rj['joint_offsets_deg'][1:4], 2).tolist()} deg (truth "
          f"{np.round(-np.degrees(off[1:4]), 2).tolist()}): " + fmt(ej))
    print(f"    joint-offset check: {'FLAGGED' if chk['flag'] else 'not flagged'} -- {chk['message']}")
    check("b: joint-offset fit: camera < 1.5 cm (was ~3 cm), offsets within 1.5 deg, and the check flags the readings",
          ej["pos_err_cm"] < 1.5 and fit_err.max() < 1.5 and chk["flag"] and S.verdict(res2, offset_check=chk)[0] == "WARN",
          f"{e2['pos_err_cm']:.2f} -> {ej['pos_err_cm']:.2f} cm, rot {e2['rot_err_deg']:.2f} -> {ej['rot_err_deg']:.2f} deg; "
          f"offset errors {np.round(fit_err, 2).tolist()} deg")
    # no false alarm on correct joint readings (the main solve above)
    rj0 = solver.fit_joint_offsets(res["R"], res["pos"])
    ej0, chk0 = pose_errors(rj0, R_t, pos_t, K, dist), S.joint_offset_check(rj0)
    print(f"    joint-offset fit on CORRECT readings: offsets {np.round(rj0['joint_offsets_deg'][1:4], 2).tolist()} deg: "
          + fmt(ej0))
    check("b: joint-offset check on correct readings: not flagged, offsets < 1 deg, camera still < 1.5 cm",
          not chk0["flag"] and chk0["max_offset_deg"] < 1.0 and ej0["pos_err_cm"] < 1.5,
          f"max offset {chk0['max_offset_deg']:.2f} deg, IoU gain {chk0['iou_gain']:+.3f}, {ej0['pos_err_cm']:.2f} cm")
    # the same frames with the whole-image search forced on (what happens automatically when the arm poses are too
    # similar): the grey-tile area then yields coarse candidates too, and the IoU ranking has to reject them
    res4, _ = S.solve_pipeline(snaps, K, dist, init_R, init_pos, mode="white", verbose=args.verbose, min_diversity=2.0)
    e4 = pose_errors(res4, R_t, pos_t, K, dist)
    print("  whole-image coarse search forced on (clutter candidates compete): " + fmt(e4))
    check("b: forced whole-image search: tile clutter does not win (pos < 1.5 cm, rot < 0.5 deg, image < 1 px)",
          e4["pos_err_cm"] < 1.5 and e4["rot_err_deg"] < 0.5 and e4["reproj_mean_px"] < 1.0,
          f"{e4['pos_err_cm']:.2f} cm, {e4['rot_err_deg']:.2f} deg, {e4['reproj_mean_px']:.2f} px")
    # a strongly tilted camera (above the base, looking forward-down ~17 deg off nadir: the arm sits at the bottom
    # centre of the image pointing up, as in the real view), started exactly as `solve.py scene_robot` does by
    # default: nadir guess 1.4 m up, heights 0.7..2.24 m, yaw within +-90 deg of image-up = +x
    pos_t3 = np.array([0.02, -0.06, 1.35])
    R_t3 = S.look_at(pos_t3, (0.42, 0.02, 0.0), up=(1.0, 0.0, 0.0))
    snaps3 = synth_frames(S.get_model(max_tris=None), qs, R_t3, pos_t3, K, dist, rng)
    init_R3, init_pos3 = S.initial_pose(1.4, 0.0)
    res3, _ = S.solve_pipeline(snaps3, K, dist, init_R3, init_pos3, mode="white", heights=(0.7, 2.24),
                               yaw_range=(0.0, np.radians(90)), verbose=args.verbose)
    e3 = pose_errors(res3, R_t3, pos_t3, K, dist)
    print(f"  tilted camera ({np.degrees(np.arccos(-R_t3[2, 2])):.1f} deg off nadir, arm at the bottom of the image), "
          f"solve.py defaults: " + fmt(e3))
    check("b: tilted camera recovered from the CLI's default start (pos < 1.5 cm, rot < 0.5 deg, image < 1 px)",
          e3["pos_err_cm"] < 1.5 and e3["rot_err_deg"] < 0.5 and e3["reproj_mean_px"] < 1.0,
          f"{e3['pos_err_cm']:.2f} cm, {e3['rot_err_deg']:.2f} deg, {e3['reproj_mean_px']:.2f} px; verdict "
          f"{S.verdict(res3)[0]}")


# ----------------------------------------------------------------------------- c: sim demos (bright)
def part_c(args):
    header(f"c: sim demos run0, episodes {args.episodes}, poor init (1.4 m, yaw +20 deg, x/y +5 cm), mode 'class'")
    t0 = time.time()
    errs = run_tasks([dict(ep=ep, mode="class", verbose=args.verbose) for ep in args.episodes], args.jobs)
    for e in errs:
        print(f"  ep{e['ep']:3d}: {fmt(e)}")
    print(f"  wall time {time.time() - t0:.0f}s with {args.jobs} processes")
    D, rot, mpe, mre = summarize(errs, "run0 bright sim")
    n = np.linalg.norm(D, axis=1)
    check("c: every episode within 4 cm / 4 deg of the placeholder (the jitter alone gives ~1.6 cm / 1.6 deg)",
          n.max() < 4 and rot.max() < 4, f"max {n.max():.2f} cm, {rot.max():.2f} deg")
    check("c: silhouettes agree (every snapshot IoU > 0.85, mean boundary < 2 px = the verdict's limit)",
          min(e["iou_min"] for e in errs) > 0.85 and max(e["boundary_px"] for e in errs) < 2.0,
          f"min IoU {min(e['iou_min'] for e in errs):.3f}, max boundary {max(e['boundary_px'] for e in errs):.2f} px")
    lim = 2.5 * np.sqrt(3) * JITTER_POS_STD * 100 / np.sqrt(len(errs)) + 0.3
    limr = 2.5 * np.sqrt(3) * JITTER_ROT_STD_DEG / np.sqrt(len(errs)) + 0.3
    check(f"c: mean pose over episodes near the placeholder (< {lim:.1f} cm / {limr:.1f} deg, i.e. 2.5 sigma of the "
          "averaged jitter)", mpe < lim and mre < limr, f"{mpe:.2f} cm, {mre:.2f} deg")
    if args.quick:
        return
    print("  within-episode repeatability (same unknown camera, two disjoint sets of 5 arm poses):")
    tasks = []
    for ep in args.episodes[:3]:
        idx = pick_frames(read_episode(ep)[0], 10)
        tasks += [dict(ep=ep, mode="class", frames=idx[0::2]), dict(ep=ep, mode="class", frames=idx[1::2])]
    out = run_tasks(tasks, args.jobs)
    reps = []
    for a, b in zip(out[0::2], out[1::2]):
        dp = np.linalg.norm(a["pos"] - b["pos"]) * 100
        dr = S.rot_angle_deg(a["R"], b["R"])
        rm, rx = S.reprojection_discrepancy(a["R"], a["pos"], b["R"], b["pos"], K_PH)
        reps.append((dp, dr, rm))
        print(f"    ep{a['ep']}: |dpos| {dp:.2f} cm, drot {dr:.2f} deg, image discrepancy {rm:.2f} px mean / {rx:.2f} max")
    check("c: repeatability within an episode (image discrepancy < 1.5 px)", max(r[2] for r in reps) < 1.5,
          f"max {max(r[0] for r in reps):.2f} cm / {max(r[1] for r in reps):.2f} deg / {max(r[2] for r in reps):.2f} px")


# ----------------------------------------------------------------------------- d: darkened sim
def part_d(args):
    eps = args.dark_episodes
    header(f"d: darkened sim (real-camera look, mean brightness ~33), episodes {eps}, poor init, mode 'white'")
    small = eps[: max(1, len(eps) // 2)]
    tasks = [dict(ep=ep, mode="white", darken={}, overlay=(k == 0), verbose=args.verbose) for k, ep in enumerate(eps)]
    tasks += [dict(ep=ep, mode="white", darken=dict(scale=0.6), verbose=args.verbose) for ep in small]
    tasks += [dict(ep=ep, mode="class", verbose=args.verbose) for ep in eps]  # bright references, same cameras
    t0 = time.time()
    out = run_tasks(tasks, args.jobs)
    print(f"  wall time {time.time() - t0:.0f}s with {args.jobs} processes")
    errs, errs_s, refs = out[:len(eps)], out[len(eps):len(eps) + len(small)], {e["ep"]: e for e in out[len(eps) + len(small):]}
    diffs = {}
    for tag, es in (("", errs), (" 0.6x", errs_s)):
        for e in es:
            r = refs[e["ep"]]
            dm, dx = S.reprojection_discrepancy(r["R"], r["pos"], e["R"], e["pos"], K_PH)
            d = (float(np.linalg.norm(e["pos"] - r["pos"]) * 100), S.rot_angle_deg(e["R"], r["R"]), dm, dx)
            diffs.setdefault(tag, []).append(d)
            print(f"  ep{e['ep']:3d}{tag}: {fmt(e)}  pass-2 moved {e.get('pass2_move_cm', np.nan):.2f} cm  -> {e['verdict'][0]} "
                  f"{'; '.join(e['verdict'][1])}")
            print(f"         vs the bright 'class' solve of the same episode (same camera, no jitter ambiguity): "
                  f"|dpos| {d[0]:.2f} cm, rot {d[1]:.2f} deg, image {d[2]:.2f} px mean / {d[3]:.2f} max")
    for tag, ds in diffs.items():
        ds = np.array(ds)
        lim = 2.0 if tag == "" else 3.0
        check(f"d: dark{tag} vs bright solve of the same episodes: image discrepancy < {lim} px (pos/rot reported)",
              ds[:, 2].max() < lim, f"max |dpos| {ds[:, 0].max():.2f} cm, rot {ds[:, 1].max():.2f} deg, image "
              f"{ds[:, 2].max():.2f} px mean ({ds[:, 3].max():.2f} max)")
    D, rot, _, _ = summarize(errs, "darkened sim")
    n = np.linalg.norm(D, axis=1)
    check("d: dark: every episode within 4 cm / 4 deg of the placeholder", n.max() < 4 and rot.max() < 4,
          f"max {n.max():.2f} cm, {rot.max():.2f} deg")
    check("d: dark: verdict PASS on every episode", all(e["verdict"][0] == "PASS" for e in errs),
          "; ".join(f"ep{e['ep']}: {', '.join(e['verdict'][1])}" for e in errs if e["verdict"][0] != "PASS"))
    D2, rot2, _, _ = summarize(errs_s, "darkened sim, 0.6x footprint (robot pixel size as at 1.4 m)")
    n2 = np.linalg.norm(D2, axis=1)
    check("d: dark 0.6x: every episode within 5 cm / 4 deg of the placeholder", n2.max() < 5 and rot2.max() < 4,
          f"max {n2.max():.2f} cm, {rot2.max():.2f} deg")
    check("d: dark 0.6x: verdict PASS on every episode", all(e["verdict"][0] == "PASS" for e in errs_s),
          "; ".join(f"ep{e['ep']}: {', '.join(e['verdict'][1])}" for e in errs_s if e["verdict"][0] != "PASS"))
    # the one overlay PNG: darkened-sim tiles (+ the real frame's 'white' segmentation when part a ran)
    ov = errs[0]["overlay"]
    if args.real_seg is not None:
        real, sg = args.real_seg
        r = real.copy()
        cs, _ = cv2.findContours((sg["white"] > 0.5).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(r, cs, -1, (0, 220, 255), 1)
        cv2.putText(r, "real frame: 'white' segmentation (no joint state)", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        row = np.zeros((r.shape[0], ov.shape[1], 3), np.uint8)
        row[:, :r.shape[1]] = r
        ov = np.concatenate([ov, row], 0)
    p = args.out / "overlay_darkened_sim.png"
    cv2.imwrite(str(p), ov)
    print(f"  overlay (yellow: segmented 'white', green: rendered printed parts, magenta: rendered servos) -> {p}")


# ----------------------------------------------------------------------------- e: solve.py scene_robot end to end
def part_e(args):
    header("e: solve.py scene_robot --dry_run on synthetic snapshot dirs (darkened sim, LeRobot-unit state.json)")
    data = args.out / "cli_data"
    data.mkdir(parents=True, exist_ok=True)
    (data / "intrinsics_scene.json").write_text(json.dumps(dict(K=K_PH.tolist(), dist=[0.0] * 5, size=[640, 480])))
    units = S._sim_units()
    ep = 40
    q, _ = read_episode(ep)
    idx = pick_frames(q, 5)
    _, imgs = read_episode(ep, idx)
    rng = np.random.default_rng(7)
    names = []
    for k, i in enumerate(idx):
        d = data / f"snap_test{k}"
        d.mkdir(exist_ok=True)
        cv2.imwrite(str(d / "scene.jpg"), darken_like_real(imgs[i], rng)[0])
        fol = {f"{j}.pos": float(v) for j, v in zip(S.JOINTS, units.sim_to_lerobot(q[i]))}
        (d / "state.json").write_text(json.dumps(dict(follower=fol), indent=1))
        names.append(f"test{k}")
    before = hashlib.sha256(CALIB_JSON.read_bytes()).hexdigest() if CALIB_JSON.exists() else None
    t0 = time.time()
    cmd = [sys.executable, str(HERE / "solve.py"), "scene_robot", "--snaps", *names, "--data_dir", str(data), "--dry_run"]
    print("  $ " + " ".join(cmd), flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True)
    print("\n".join("  | " + ln for ln in p.stdout.strip().splitlines()))
    if p.returncode:
        print(p.stderr[-3000:])
    after = hashlib.sha256(CALIB_JSON.read_bytes()).hexdigest() if CALIB_JSON.exists() else None
    m = re.search(r"SCENE_CAM_POS\s*=\s*\[([^\]]+)\]", p.stdout)
    pos = np.array([float(v) for v in m.group(1).split(",")]) if m else np.full(3, np.nan)
    check("e: scene_robot exits 0 and prints a verdict",
          p.returncode == 0 and re.search(r"VERDICT: (PASS|WARN)", p.stdout) is not None, f"{time.time() - t0:.0f}s")
    check("e: --dry_run left calib.json byte-identical", before == after)
    check("e: overlay written to <data_dir>/scene_robot_overlay.jpg", (data / "scene_robot_overlay.jpg").exists())
    check("e: printed SCENE_CAM_POS within 4 cm of the placeholder", np.linalg.norm(pos - POS_PH) < 0.04,
          f"pos {np.round(pos, 4).tolist()}")
    check("e: joint-offset check ran on the (offset-free) sim readings and did not flag them",
          "joint-offset check:" in p.stdout and "joint readings look offset" not in p.stdout)
    if args.quick:
        return
    # the write path, in-process with solve._update_calib replaced by a recorder (nothing is written anywhere):
    # no intrinsics_scene.json (placeholder K -> WARN, written only with --force, K/dist stored with the pose) and
    # --init_yaw 180 (the true image-up, +x, is outside the default +-90 deg window -> poor match -> automatic retry)
    import contextlib
    import io
    import solve
    data2 = args.out / "cli_data_noK"
    data2.mkdir(parents=True, exist_ok=True)
    (data2 / "intrinsics_scene.json").unlink(missing_ok=True)
    for n in names:
        (data2 / f"snap_{n}").mkdir(exist_ok=True)
        for f in ("scene.jpg", "state.json"):
            (data2 / f"snap_{n}" / f).write_bytes((data / f"snap_{n}" / f).read_bytes())
    written = []
    real_update = solve._update_calib
    solve._update_calib = lambda **kv: written.append(kv)
    buf = io.StringIO()
    t0 = time.time()
    argv = sys.argv
    try:
        sys.argv = ["solve.py", "scene_robot", "--snaps", *names, "--data_dir", str(data2), "--init_yaw", "180",
                    "--force", "--joint_offsets", "off"]
        with contextlib.redirect_stdout(buf):
            solve.main()
    finally:
        solve._update_calib = real_update
        sys.argv = argv
    out = buf.getvalue()
    print("\n".join("  | " + ln for ln in out.strip().splitlines()))
    after2 = hashlib.sha256(CALIB_JSON.read_bytes()).hexdigest() if CALIB_JSON.exists() else None
    m = re.search(r"SCENE_CAM_POS\s*=\s*\[([^\]]+)\]", out)
    pos2 = np.array([float(v) for v in m.group(1).split(",")]) if m else np.full(3, np.nan)
    check("e: --init_yaw 180 (truth outside the yaw window): automatic full-yaw retry recovers the pose (< 4 cm)",
          "retrying with the full yaw range" in out and np.linalg.norm(pos2 - POS_PH) < 0.04,
          f"pos {np.round(pos2, 4).tolist()}, {time.time() - t0:.0f}s")
    check("e: placeholder intrinsics -> verdict WARN naming them", "VERDICT: WARN" in out and "intrinsics not calibrated" in out)
    ok = (len(written) == 1 and set(written[0]) == {"SCENE_CAM_POS", "SCENE_CAM_QUAT", "SCENE_K", "SCENE_DIST"}
          and np.allclose(written[0]["SCENE_K"], K_PH.ravel()) and np.allclose(written[0]["SCENE_DIST"], 0)
          and np.allclose(written[0]["SCENE_CAM_POS"], pos2, atol=1e-4))
    check("e: --force writes the pose together with the K/dist it was solved with (recorder; calib.json untouched)",
          ok and after2 == before, f"{[sorted(w) for w in written]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="fewer episodes, no repeatability test")
    ap.add_argument("--parts", default="abcde")
    ap.add_argument("--jobs", type=int, default=3, help="parallel episode solves (processes, ~3 threads each)")
    ap.add_argument("--episodes", type=int, nargs="+", default=None, help="run0 episodes for part c")
    ap.add_argument("--dark_episodes", type=int, nargs="+", default=None, help="run0 episodes for part d")
    ap.add_argument("--out", type=Path, default=SCRATCH / "silhouette")
    ap.add_argument("--verbose", action="store_true", help="solver progress")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.episodes is None:
        args.episodes = [0, 100, 175] if args.quick else [0, 25, 50, 75, 100, 125, 150, 175]
    if args.dark_episodes is None:
        args.dark_episodes = [10, 110] if args.quick else [10, 60, 110, 160]
    args.real_seg = None
    t0 = time.time()
    for k, fn in dict(a=part_a, b=part_b, c=part_c, d=part_d, e=part_e).items():
        if k in args.parts:
            t = time.time()
            fn(args)
            print(f"  (part {k}: {time.time() - t:.0f}s)")
    nfail = sum(not ok for _, ok, _ in RESULTS)
    print(f"\n{len(RESULTS) - nfail}/{len(RESULTS)} checks passed in {time.time() - t0:.0f}s -> {'PASS' if not nfail else 'FAIL'}")
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}  -- {detail}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())

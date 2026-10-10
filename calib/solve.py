"""Turn captures (calib/capture.py) into the calibration the sim uses (calib/data/calib.json).

  intrinsics --cam scene|wrist        ChArUco (paper or phone board, see boards.py) -> K, distortion (640x480)
  scene      --snap NAME              ArUco anchor seen by the overhead camera + jaw-tip touches
                                      (touch order: TL, TR, BR, BL as printed) -> overhead camera pose
  scene_robot --snaps A B C ...       no marker: fit the arm's URDF silhouette (known joint angles) to the
                                      overhead images of several snapshots -> overhead camera pose, stored
                                      with the K/dist it was solved with; checks the joint readings for
                                      constant zero offsets (--joint_offsets check|fit|off)
                                      (calib/silhouette.py; --dry_run to only report)
  splat      --scene DIR --ply PLY    triangulate the anchor in the phone video's COLMAP model ->
                                      similarity transform of the Gaussian splat into the desk frame,
                                      crop the robot out, convert to USD
Run with ~/TestingGrounds/robot_splat/.venv/bin/python (numpy, opencv, scipy, gsplat's COLMAP parser).
"""
import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
sys.path.insert(0, str(HERE))
import fk  # noqa: E402
from boards import BOARDS, detect  # noqa: E402

MARKER = 0.100  # black square side [m]
ARUCO = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
# marker corners in the marker frame (OpenCV order TL, TR, BR, BL; z out of the paper = up)
M_CORNERS = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * MARKER / 2
CALIB_JSON = DATA / "calib.json"


def _update_calib(**kv):
    cur = json.loads(CALIB_JSON.read_text()) if CALIB_JSON.exists() else {}
    cur.update(kv)
    CALIB_JSON.write_text(json.dumps(cur, indent=1))
    print(f"updated {CALIB_JSON}: {sorted(kv)}")


def _rigid(A, B, scale=False):
    """Least-squares B ~= s R A + t (Umeyama). Returns s, R, t, per-point residuals."""
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / (A0 ** 2).sum() if scale else 1.0
    t = cb - s * R @ ca
    return s, R, t, np.linalg.norm((s * A @ R.T + t) - B, axis=1)


def _detect_marker(img):
    corners, ids, _ = ARUCO.detectMarkers(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if ids is None or 0 not in ids.ravel():
        return None
    c = corners[list(ids.ravel()).index(0)][0]
    cv2.cornerSubPix(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), c, (5, 5), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3))
    return c


def cmd_intrinsics(args):
    d = DATA / f"intr_{args.cam}"
    files = sorted(d.glob("*.jpg"))
    BOARD = BOARDS[(d / "board.txt").read_text().strip() if (d / "board.txt").exists() else "paper"]
    obj_pts, img_pts, size = [], [], None
    for f in files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        size = img.shape[::-1]
        cc, ci = detect(img, BOARD)
        if ci is not None and len(ci) >= 6:
            o, i = BOARD.matchImagePoints(cc, ci)
            obj_pts.append(o); img_pts.append(i)
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)
    print(f"{args.cam}: {len(obj_pts)}/{len(files)} frames, RMS reprojection {rms:.3f} px")
    print("K =", np.round(K, 2).tolist(), "\ndist =", np.round(dist.ravel(), 5).tolist())
    (DATA / f"intrinsics_{args.cam}.json").write_text(json.dumps(dict(K=K.tolist(), dist=dist.ravel().tolist(), size=size, rms=rms), indent=1))
    _update_calib(**{f"{args.cam.upper()}_K": K.ravel().tolist(), f"{args.cam.upper()}_DIST": dist.ravel().tolist()})


def marker_in_world():
    """Pose of the anchor marker in the desk frame from the jaw-tip touches (ordered TL, TR, BR, BL)."""
    touches = json.loads((DATA / "touch_marker.json").read_text())
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    W = np.array([fk.fingertip_in_world(np.radians([t[j] for j in joints])) for t in touches])
    print("touched points (desk frame, mm):", np.round(W * 1000, 1).tolist())
    print(f"  heights above desk: {np.round(W[:, 2] * 1000, 1).tolist()} mm (should all be ~0 +- 3 mm)")
    n = len(W)
    best = None
    for perm in itertools.permutations(range(4), n):  # also report if another order fits better
        s, R, t, res = _rigid(M_CORNERS[list(perm)], W)
        if best is None or res.mean() < best[0]:
            best = (res.mean(), perm, R, t, res)
    ordered = _rigid(M_CORNERS[:n], W)
    print(f"  ordered (TL,TR,BR,BL) fit residual {ordered[3].mean() * 1000:.1f} mm; best order {best[1]} -> {best[0] * 1000:.1f} mm")
    if best[1] != tuple(range(n)):
        print("  WARNING: the touches fit a different corner order better -- check the touch order (TL,TR,BR,BL).")
    side = np.mean([np.linalg.norm(W[i] - W[(i + 1) % n]) for i in range(n)]) if n == 4 else float("nan")
    print(f"  touched square side {side * 1000:.1f} mm (printed: {MARKER * 1000:.0f} mm)")
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = ordered[1], ordered[2]
    return T


def cmd_scene(args):
    intr = json.loads((DATA / "intrinsics_scene.json").read_text())
    K, dist = np.array(intr["K"]), np.array(intr["dist"])
    img = cv2.imread(str(DATA / f"snap_{args.snap}" / "scene.jpg"))
    c = _detect_marker(img)
    if c is None:
        sys.exit("anchor marker (ArUco 4x4 id 0) not visible in the overhead image")
    ok, rvec, tvec = cv2.solvePnP(M_CORNERS, c, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    proj, _ = cv2.projectPoints(M_CORNERS, rvec, tvec, K, dist)
    print(f"marker PnP reprojection error {np.linalg.norm(proj[:, 0] - c, axis=1).mean():.2f} px, distance {np.linalg.norm(tvec):.3f} m")
    T_cam_marker = np.eye(4)
    T_cam_marker[:3, :3], T_cam_marker[:3, 3] = cv2.Rodrigues(rvec)[0], tvec.ravel()
    T_world_cam = marker_in_world() @ np.linalg.inv(T_cam_marker)
    pos, quat = T_world_cam[:3, 3], Rotation.from_matrix(T_world_cam[:3, :3]).as_quat()  # xyzw, ROS/OpenCV camera axes
    print(f"overhead camera in desk frame: pos {np.round(pos, 4).tolist()} m, looking along {np.round(T_world_cam[:3, 2], 3).tolist()}")
    _update_calib(SCENE_CAM_POS=pos.tolist(), SCENE_CAM_QUAT=quat.tolist(), MARKER_T_WORLD=marker_in_world().tolist())


def cmd_scene_robot(args):
    """Overhead camera pose without a marker: the arm itself is the target (calib/silhouette.py)."""
    import silhouette as sil

    data = Path(args.data_dir)
    K, dist, src = sil.load_intrinsics(data / "intrinsics_scene.json")
    calibrated = not src.startswith("placeholder")
    print(f"intrinsics: {src}: fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}, "
          f"dist {np.round(np.asarray(dist, float).ravel(), 4).tolist()}")
    if not calibrated:
        print("  WARNING: no scene intrinsics: with the placeholder fx=525 the solved camera height/scale is only as good as "
              "that guess (the verdict will say WARN; calibrate with capture.py/solve.py intrinsics --cam scene first)")
    snaps = [sil.load_snapshot(data / f"snap_{n}") for n in args.snaps]
    for s in snaps:
        if s.image.shape[:2] != (480, 640):
            print(f"  WARNING: snapshot {s.name} is {s.image.shape[1]}x{s.image.shape[0]}, intrinsics assume 640x480")
        print(f"  snapshot {s.name}: joints [rad] {np.round(s.q, 3).tolist()}, mean brightness {s.image.mean():.0f}")
    for i in range(len(snaps)):
        for j in range(i):
            if np.abs(snaps[i].q[:5] - snaps[j].q[:5]).max() < np.radians(10):
                print(f"  WARNING: snapshots {snaps[j].name} and {snaps[i].name} have nearly the same arm pose; the search "
                      f"relies on the arm MOVING between snapshots (static white clutter is ignored that way)")
    init_R, init_pos = sil.initial_pose(args.init_height, args.init_yaw)
    yaw_range = None if args.yaw_window >= 180 else (np.radians(args.init_yaw), np.radians(args.yaw_window))
    heights = (args.init_height / 2.5, args.init_height * 1.6)
    res, solver = sil.solve_pipeline(snaps, K, dist, init_R, init_pos, mode=args.match, yaw_range=yaw_range,
                                     heights=heights, verbose=args.verbose)
    if res["iou_mean"] < 0.7 and not args.no_retry:
        # a wrong basin (image-up outside the yaw window, a camera tilted beyond the nadir seeds' reach, ...): search
        # every yaw with seeds tilted +-25 deg as well, keep the better silhouette match
        print(f"  poor silhouette match (mean IoU {res['iou_mean']:.2f} < 0.7) -> retrying with the full yaw range and "
              f"coarse seeds tilted +-25 deg")
        res2, solver2 = sil.solve_pipeline(snaps, K, dist, init_R, init_pos, mode=args.match, yaw_range=None,
                                           heights=heights, pitches=np.radians([0.0, 25.0, -25.0]), verbose=args.verbose)
        print(f"  retry: mean IoU {res2['iou_mean']:.3f} (first search {res['iou_mean']:.3f}) -> keeping the "
              + ("retry" if res2["iou_mean"] > res["iou_mean"] else "first search"))
        if res2["iou_mean"] > res["iou_mean"]:
            res, solver = res2, solver2
    roi_move_cm = float(np.linalg.norm(res["pos"] - res["pass1"]["pos"]) * 100)
    div = res.get("pose_diversity")
    if div is not None:
        print(f"  arm pose diversity {100 * div:.0f}% (share of the arm's silhouette that moves between snapshots)"
              + (f" < {100 * solver.min_diversity:.0f}%: poses too similar, the whole image was searched too -- "
                 f"capture more varied poses" if div < solver.min_diversity else ""))
    # joint-reading offsets (LeRobot calibration zero vs URDF zero): refit with constant lift/elbow/wrist_flex offsets
    chk = None
    if args.joint_offsets != "off":
        if len(snaps) >= 5 and div is not None and div >= 0.4:
            res_j = solver.fit_joint_offsets(res["R"], res["pos"], t0=None)
            chk = sil.joint_offset_check(res_j)
            print(f"  joint-offset {'fit' if args.joint_offsets == 'fit' else 'check'}: constant offsets "
                  + ", ".join(f"{j} {res_j['joint_offsets_deg'][sil.JOINTS.index(j)]:+.2f}" for j in res_j["offset_joints"])
                  + f" deg -> mean IoU {res_j['iou_mean_no_offsets']:.3f} -> {res_j['iou_mean']:.3f}, camera moves "
                  f"{chk['move_cm']:.1f} cm / {chk['move_deg']:.1f} deg ({res_j['runtime_s']:.0f}s)")
            if args.joint_offsets == "fit":
                res_j["pass1"], res_j["pose_diversity"] = res["pass1"], div
                res_j["runtime_s"] += res["runtime_s"]
                res = res_j
            else:
                solver.set_joint_offsets(np.zeros(6))  # report / overlay the pose solved with the joint readings as given
        else:
            print(f"  joint-offset {args.joint_offsets} skipped: needs >= 5 snapshots and arm pose diversity >= 40% "
                  f"(have {len(snaps)}, {100 * (div or 0):.0f}%) -- joint-zero offsets are NOT checked")
    overlay = data / "scene_robot_overlay.jpg"
    cv2.imwrite(str(overlay), solver.overlay(res["R"], res["pos"]))
    for p in res["per_snap"]:
        print(f"  {p['name']:>16s}: IoU {p['iou']:.3f}  boundary residual {p['boundary_px']:.2f} px "
              f"(median {p['boundary_med_px']:.1f}, symmetric {p['boundary_sym_px']:.2f})  robot {p['arm_px']:.0f} px long")
    pos, quat, R = res["pos"], res["quat"], res["R"]
    tilt = np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1)))
    print(f"solved in {res['runtime_s']:.0f}s: camera {pos[2]:.3f} m above the desk, optical axis {tilt:.1f} deg off "
          f"straight down, image-up = desk yaw {np.degrees(sil.yaw_of(R)):.1f} deg; ROI pass moved it "
          f"{roi_move_cm:.1f} cm; robot ~{res['arm_px_mean']:.0f} px long in the image")
    if args.joint_offsets == "fit" and "joint_offsets_deg" in res:
        print("  pose solved WITH the fitted joint offsets above; fix the LeRobot calibration so the sim gets the same "
              "joint angles as the real arm (the camera pose only compensates in the image)")
    print(f"SCENE_CAM_POS = {np.round(pos, 5).tolist()}")
    print(f"SCENE_CAM_QUAT = {np.round(quat, 6).tolist()}  (xyzw, ROS camera axes)")
    v, why = sil.verdict(res, intrinsics_calibrated=calibrated, offset_check=chk if args.joint_offsets == "check" else None)
    print(f"VERDICT: {v}" + "".join(f"\n  - {w}" for w in why))
    if v == "PASS":
        print("  (PASS = the silhouettes agree. Not detectable here: a shoulder_pan zero offset, which looks exactly like a "
              "camera yaw about the robot base -- check the pan zero physically" + ("" if chk is not None else
              "; the other joint zeros were not checked either") + ")")
    print(f"overlay: {overlay} (yellow: segmented arm, green: rendered printed parts, magenta: rendered servos)")
    if args.dry_run:
        print("--dry_run: calib.json NOT updated")
    elif v != "PASS" and not args.force:
        print("calib.json NOT updated (verdict WARN); inspect the overlay, recapture, or rerun with --force")
    else:
        # the pose is only valid together with the intrinsics it was solved with: store them as a pair
        _update_calib(SCENE_CAM_POS=pos.tolist(), SCENE_CAM_QUAT=quat.tolist(), SCENE_K=K.ravel().tolist(),
                      SCENE_DIST=np.asarray(dist, float).ravel().tolist())
        print("  (if the scene intrinsics are recalibrated later, rerun scene_robot: this pose was solved with the K above)")


def _triangulate(rays):
    """Least-squares intersection of rays [(origin, unit dir)]."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in rays:
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    x = np.linalg.solve(A, b)
    return x, np.array([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (x - o)) for o, d in rays])


def cmd_splat(args):
    sys.path.insert(0, str(Path.home() / "TestingGrounds/robot_splat/gsplat/examples"))
    from datasets.colmap import Parser  # gsplat's COLMAP loader (applies the same normalisation as training)

    p = Parser(args.scene, factor=1, normalize=True, test_every=8)
    import pycolmap
    rec = pycolmap.Reconstruction(str(Path(args.scene) / "sparse" / "0"))
    rays = [[] for _ in range(4)]
    for im in rec.images.values():
        img = cv2.imread(str(Path(args.scene) / "images" / im.name))
        c = _detect_marker(img)
        if c is None:
            continue
        cam = rec.cameras[im.camera_id]
        Kc = np.array([[cam.params[0], 0, cam.params[2]], [0, cam.params[1], cam.params[3]], [0, 0, 1]])
        dist = np.array(cam.params[4:8]) if cam.model.name == "OPENCV" else np.zeros(4)
        n = cv2.undistortPoints(c.reshape(-1, 1, 2), Kc, dist).reshape(-1, 2)
        cfw = im.cam_from_world()
        R, t = cfw.rotation.matrix(), cfw.translation
        center = -R.T @ t
        for k in range(4):
            d = R.T @ np.r_[n[k], 1.0]
            rays[k].append((center, d / np.linalg.norm(d)))
    nviews = len(rays[0])
    if nviews < 3:
        sys.exit(f"anchor marker seen in only {nviews} phone frames; need >= 3")
    P_colmap, res = zip(*[_triangulate(r) for r in rays])
    P_colmap = np.array(P_colmap)
    P_norm = P_colmap @ p.transform[:3, :3].T + p.transform[:3, 3]  # into gsplat's normalised frame (= PLY frame)
    T_wm = np.array(json.loads(CALIB_JSON.read_text())["MARKER_T_WORLD"])
    W = M_CORNERS @ T_wm[:3, :3].T + T_wm[:3, 3]
    s, R, t, fit = _rigid(P_norm, W, scale=True)
    print(f"marker seen in {nviews} phone frames; ray residual {np.mean([r.mean() for r in res]):.4f} (COLMAP units)")
    print(f"splat -> desk: scale {s:.5f} m/unit, fit residual {fit.mean() * 1000:.2f} mm (max {fit.max() * 1000:.2f})")

    # crop the robot out (the sim renders the arm itself): drop Gaussians near the base / folded arm
    from plyfile import PlyData, PlyElement
    ply = PlyData.read(args.ply)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], -1) @ (s * R).T + t
    r = np.linalg.norm(xyz[:, :2] - np.array(args.base_xy), axis=1)
    robot = (r < args.crop_radius) & (xyz[:, 2] > -0.005) & (xyz[:, 2] < args.crop_height)
    below = xyz[:, 2] < -0.9  # nothing real is a metre below the desk top
    keep = ~(robot | below)
    out_ply = Path(args.out) / "background.ply"
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(v.data[keep], "vertex")], text=False).write(str(out_ply))
    print(f"cropped {int(robot.sum())} robot Gaussians (+{int(below.sum())} stray), kept {int(keep.sum())} -> {out_ply}")
    gs = sorted(Path.home().glob("isaacsim/extscache/omni.kit.converter.gsplat-*"))[-1]
    usd = Path(args.out) / "background.usdc"
    subprocess.run([str(Path.home() / "isaacsim/python.sh"), "-m", "usd_convert_gsplat", "-i", str(out_ply), "-o", str(usd),
                    "-n", "background", "--up-axis", "Z"], env={"PYTHONPATH": str(gs / "pip_prebundle"), "PATH": "/usr/bin:/bin",
                                                               "HOME": str(Path.home())}, check=True)
    _update_calib(SPLAT_USD=str(usd), SPLAT_POS=t.tolist(), SPLAT_QUAT=Rotation.from_matrix(R).as_quat().tolist(), SPLAT_SCALE=float(s))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("intrinsics"); a.add_argument("--cam", required=True)
    b = sub.add_parser("scene"); b.add_argument("--snap", default="anchor")
    c = sub.add_parser("splat"); c.add_argument("--scene", required=True); c.add_argument("--ply", required=True)
    c.add_argument("--out", default="/mnt/secondary/so101_r2s/assets/background")
    c.add_argument("--base_xy", type=float, nargs=2, default=[0.0, 0.0])
    c.add_argument("--crop_radius", type=float, default=0.13); c.add_argument("--crop_height", type=float, default=0.45)
    r = sub.add_parser("scene_robot", help="overhead camera pose from the arm's silhouette in several snapshots")
    r.add_argument("--snaps", nargs="+", required=True, help="NAMEs of data/snap_NAME dirs (capture.py snapshot)")
    r.add_argument("--init_height", type=float, default=1.4, help="rough camera height above the desk [m]")
    r.add_argument("--init_yaw", type=float, default=0.0, help="desk direction at the top of the image [deg from +x]")
    r.add_argument("--yaw_window", type=float, default=90.0, help="search that direction within +- this [deg]; 180 = all")
    r.add_argument("--match", default="white", choices=["white", "union", "class"],
                   help="white: match printed parts only (dark desk); class: printed + servos (bright desk)")
    r.add_argument("--data_dir", default=str(DATA), help="where snap_NAME/, intrinsics_scene.json and the overlay live")
    r.add_argument("--joint_offsets", default="check", choices=["check", "fit", "off"],
                   help="constant shoulder_lift/elbow_flex/wrist_flex reading offsets (LeRobot zero vs URDF zero): check = "
                        "fit them as a diagnostic and WARN when the readings look offset (default); fit = use the "
                        "offset-fitted pose; off = skip (needs >= 5 snapshots with diverse poses)")
    r.add_argument("--no_retry", action="store_true", help="no automatic full-yaw/tilted retry after a poor match")
    r.add_argument("--dry_run", action="store_true", help="solve and report, never write calib.json")
    r.add_argument("--force", action="store_true", help="write calib.json even when the verdict is WARN")
    r.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    {"intrinsics": cmd_intrinsics, "scene": cmd_scene, "splat": cmd_splat, "scene_robot": cmd_scene_robot}[args.mode](args)


if __name__ == "__main__":
    main()

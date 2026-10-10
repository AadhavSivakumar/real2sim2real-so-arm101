"""Replay the real teleop episodes (LeRobot v3) open-loop in sim and compare with the real follower.

  detect   cube + cup start poses from the first overhead frame of every episode (calibrated scene camera):
           cube = top face of the tan blob back-projected to z = 30 mm (minAreaRect -> centre, yaw mod 90);
           cup = circle fit to the far-side rim silhouette back-projected to z = cup height (the near side is
           the tapered outer wall). Writes <out>/poses.json + detect_debug.jpg. Needs pyav: ~/lr/bin/python.
  replay   every (episode x correction variant) in its own env: the robot starts at the real first
           observation.state, cube/cup at the detected poses, then the real `action` stream is applied at 30 Hz.
           Writes <out>/<tag>.npz (sim joints/TCP/objects per frame) and <tag>_summary.json; prints the table.
  analyze  metrics table + plots from <out>/<tag>.npz (numpy/matplotlib: ~/isl/bin/python).

A variant = corrections at the sim<->real boundary (units.py is the baseline):
  off   joint-zero offsets [deg], 5 arm joints, ADDED to the real readings: q_sim = rad(q_real + off)
  g0, span  gripper map q_sim = g0 + g/100 * span [rad] (units.py: -0.175, 1.92)
  kp, kd    arm stiffness / damping scale on the USD SysID gains;  delay  action delay [frames]
Spec: name:key=val/key=val, vectors comma-separated, e.g. "hyp:off=0,-0.2,-4.5,-3,0/span=2.337/delay=1".

Run:
  ~/lr/bin/python scripts/replay_real.py detect
  ./lab.sh -p scripts/replay_real.py replay --sweep zero --viz none
  ~/isl/bin/python scripts/replay_real.py analyze --tag zero --plot_eps 0 5
"""
import argparse
import sys

MODES = ("detect", "replay", "analyze")
parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=MODES)
parser.add_argument("--dataset", default="/mnt/secondary/so101_r2s/lerobot_datasets/real_cube_cup_v1")
parser.add_argument("--out", default="/mnt/secondary/so101_r2s/replay_real")
parser.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all")
parser.add_argument("--task", default="SO101-CubeInCup-v0")
parser.add_argument("--sweep", default="base", help="preset variant list (see PRESETS)")
parser.add_argument("--variants", nargs="*", default=[], help="extra variant specs (see module docstring)")
parser.add_argument("--tag", default=None, help="output name (default: the sweep name)")
parser.add_argument("--settle", type=int, default=45, help="frames to hold the last action after an episode")
parser.add_argument("--plot_eps", type=int, nargs="*", default=[0, 5])
parser.add_argument("--plot_variants", nargs="*", default=None, help="analyze: variants to plot (default first+best)")
parser.add_argument("--debug_gains", action="store_true", help="replay: print the written joint gains (data + PhysX view)")
if len(sys.argv) > 1 and sys.argv[1] == "replay":
    from isaaclab.app import add_launcher_args, launch_simulation
    from isaaclab_tasks.utils import setup_preset_cli

    add_launcher_args(parser)
    args, hydra_args = setup_preset_cli(parser)
    sys.argv = [sys.argv[0]] + hydra_args
else:
    args = parser.parse_args()

import glob
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
SHORT = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]
FPS = 30
CUBE_SIZE = 0.030
CUP_H, CUP_TOP_D = 0.095, 0.080  # env_cfg.CUP_DIMS (placeholder until measured)
SPAWN = dict(cube=((0.13, 0.26), 0.9), cup=((0.14, 0.25), 0.9))  # env_cfg reset_objects: r range [m], |phi| max
HYP_OFF = (0.0, -0.2, -4.5, -3.0, 0.0)  # calib/data/scene_cam_proposal.json joint_offsets_deg_suggested
GRIP_TICK_SPAN = (3476 - 1953) * 2 * np.pi / 4096  # follower.json gripper range_max - range_min [rad] = 2.337


def _module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


units = _module(REPO / "so101_r2s2r" / "units.py")  # no Isaac / gym imports
G0, SPAN = units.GRIPPER_SIM_CLOSED, units.GRIPPER_SIM_OPEN - units.GRIPPER_SIM_CLOSED


# ----------------------------------------------------------------------------- data
def load_episodes(root, which=None):
    """{episode: dict(state (T,6), action (T,6))} in LeRobot units, frame order."""
    rows = {}
    for f in sorted(glob.glob(f"{root}/data/chunk-*/file-*.parquet")):
        t = pq.read_table(f, columns=["observation.state", "action", "episode_index", "frame_index"]).to_pydict()
        for s, a, e, i in zip(t["observation.state"], t["action"], t["episode_index"], t["frame_index"]):
            rows.setdefault(e, []).append((i, s, a))
    eps = {}
    for e in sorted(rows):
        if which is not None and e not in which:
            continue
        r = sorted(rows[e], key=lambda x: x[0])
        eps[e] = dict(state=np.array([x[1] for x in r], float), action=np.array([x[2] for x in r], float))
    return eps


def scene_video(root, e):
    """(path, from_timestamp) of episode e's overhead video."""
    for f in sorted(glob.glob(f"{root}/meta/episodes/chunk-*/file-*.parquet")):
        t = pq.read_table(f).to_pydict()
        for k, ep in enumerate(t["episode_index"]):
            if ep == e:
                key = "videos/observation.images.scene"
                c, i = t[f"{key}/chunk_index"][k], t[f"{key}/file_index"][k]
                return f"{root}/videos/observation.images.scene/chunk-{c:03d}/file-{i:03d}.mp4", t[f"{key}/from_timestamp"][k]
    raise KeyError(e)


# ----------------------------------------------------------------------------- variants
def parse_variant(spec):
    name, _, body = spec.partition(":")
    v = dict(name=name, off=np.zeros(5), g0=G0, span=SPAN, kp=1.0, kd=1.0, delay=0, place="det", shift=np.zeros(2), slot=0.0)
    for kv in filter(None, body.split("/")):
        k, val = kv.split("=")
        if k in ("off", "shift") or "," in val:  # kp/kd may be per arm joint (5 values)
            v[k] = np.array([float(x) for x in val.split(",")])
        elif k == "delay":
            v["delay"] = int(val)
        elif k == "place":
            v["place"] = val
        else:
            v[k] = float(val)
    return v


def _off(o):
    return ",".join(f"{x:g}" for x in o)


_ZERO = [("base", ""), ("hyp+", f"off={_off(HYP_OFF)}"), ("hyp-", f"off={_off(-np.array(HYP_OFF))}"),
         ("elbow-4.5", "off=0,0,-4.5,0,0"), ("wflex-3", "off=0,0,0,-3,0"), ("wflex+3", "off=0,0,0,3,0"),
         ("lift+3", "off=0,3,0,0,0"), ("kinfit", "off=6.05,3.02,-5.05,-0.06,0")]
_BEST = f"off={_off(HYP_OFF)}/place=kin/slot=0.01/span={GRIP_TICK_SPAN:.4f}"
_DYN = "/delay=1/kp=4/kd=1,1,2,0.5,0.5"  # per-joint fit of the dyn sweep (analyze --tag dyn)
_STEPS = [("0-units.py", ""), ("1+offsets", f"off={_off(HYP_OFF)}"),
          ("2+grip", f"off={_off(HYP_OFF)}/span={GRIP_TICK_SPAN:.4f}"),
          ("3+dyn", f"off={_off(HYP_OFF)}/span={GRIP_TICK_SPAN:.4f}{_DYN}"),
          ("3b+dyn-kd2", f"off={_off(HYP_OFF)}/span={GRIP_TICK_SPAN:.4f}/delay=1/kp=4/kd=2"),
          ("3c+dyn-kd4", f"off={_off(HYP_OFF)}/span={GRIP_TICK_SPAN:.4f}/delay=1/kp=4/kd=4")]
PRESETS = {
    "base": ["base:"],
    # joint-zero offsets x object placement (det = camera detections, shift = detections moved by the mean
    # kinematic mismatch, kin = cube at the FK TCP at the real grasp frame, cup under the TCP at release)
    "zero": [f"{pl}/{n}:{b}/place={pl}".replace(":/", ":") for pl in ("det", "kin") for n, b in _ZERO]
            + [f"shift/{n}:{b}/shift=-0.028,-0.003".replace(":/", ":") for n, b in _ZERO[:2]],
    # gripper map: real jaw parks at 14.9 on the 30 mm cube, the sim jaw at 0.174 rad (zero sweep, kin placement);
    # follower.json gripper range = 1523 ticks = 2.337 rad for 0..100 (units.py assumes the sim's 1.92 rad joint range)
    "grip": [f"{pl}/{n}:off={_off(HYP_OFF)}/place={pl}{b}" for pl in ("kin", "det")
             for n, b in (("base", ""), ("tickspan", f"/span={GRIP_TICK_SPAN:.4f}"), ("anchor-off", f"/g0={0.174 - 0.149 * SPAN:.4f}"),
                          ("tickspan+0.03", f"/span={GRIP_TICK_SPAN:.4f}/g0={G0 + 0.03:.4f}"))]
            + [f"det/kinfit/tickspan:off=6.05,3.02,-5.05,-0.06,0/span={GRIP_TICK_SPAN:.4f}"],
    # kin placement margin (cube starts this far into the open jaw slot)
    "slot": [f"kin{1000 * sl:g}/{n}/{g}:off={_off(o)}/place=kin/slot={sl}{gb}" for sl in (0.0, 0.005, 0.010)
             for n, o in (("base", np.zeros(5)), ("hyp+", HYP_OFF), ("kinfit", (6.05, 3.02, -5.05, -0.06, 0)))
             for g, gb in (("grip", ""), ("tick", f"/span={GRIP_TICK_SPAN:.4f}"))],
    # actuator response: uniform arm stiffness/damping scales x action delay on the best kinematic setup; analyze
    # picks the per-joint best (per_joint_best)
    "dyn": [f"d{d}_kp{kp:g}_kd{kd:g}:{_BEST}/delay={d}/kp={kp}/kd={kd}" for d in (0, 1, 2)
            for kp in (0.25, 0.5, 1.0, 2.0, 4.0) for kd in (0.5, 1.0, 2.0)],
    # offset magnitude x dynamics on the kin placement; then the full pipeline on the camera placement
    "zero2": [f"kin/{a:g}hyp/{dn}:off={_off(a * np.array(HYP_OFF))}/place=kin/slot=0.01/span={GRIP_TICK_SPAN:.4f}{db}"
              for dn, db in (("usd", ""), ("fit", _DYN)) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
             + [f"kin/elbow{el:g}/{dn}:off=0,0,{el},0,0/place=kin/slot=0.01/span={GRIP_TICK_SPAN:.4f}{db}"
                for dn, db in (("usd", ""), ("fit", _DYN)) for el in (-2.25, -4.5)]
             + [f"kin/{a:g}hyp/fit-kp8:off={_off(a * np.array(HYP_OFF))}/place=kin/slot=0.01/span={GRIP_TICK_SPAN:.4f}"
                "/delay=1/kp=8/kd=1,1,2,0.5,0.5" for a in (0.5, 1.0)]
             + ["det/units.py/usd:"]
             + [f"{pl}/{a:g}hyp/fit/tick:off={_off(a * np.array(HYP_OFF))}/span={GRIP_TICK_SPAN:.4f}{_DYN}{sb}"
                for pl, sb in (("det", ""), ("det+shift", "/shift=-0.028,-0.003")) for a in (0.5, 1.0)],
    # report table: corrections applied cumulatively, camera vs kinematic placement, + placement jitter (slot)
    "final": [f"{pl}/{n}:{b}{pb}".replace(":/", ":") for pl, pb in (("det", ""), ("kin", "/place=kin/slot=0.01"))
              for n, b in _STEPS]
             + [f"kin-slot{1000 * sl:g}/{n}:{b}/place=kin/slot={sl}" for sl in (0.005, 0.015) for n, b in _STEPS[2:4]]
             + [f"det+shift/{n}:{b}/shift=-0.028,-0.003" for n, b in _STEPS[2:4]],
}


def to_sim(v, var):
    """LeRobot units (...,6) -> sim rad with a variant's corrections."""
    q = np.empty_like(np.asarray(v, float))
    q[..., :5] = np.radians(v[..., :5] + var["off"])
    q[..., 5] = var["g0"] + v[..., 5] / 100.0 * var["span"]
    return q


def to_real(q, var):
    v = np.empty_like(np.asarray(q, float))
    v[..., :5] = np.degrees(q[..., :5]) - var["off"]
    v[..., 5] = (q[..., 5] - var["g0"]) / var["span"] * 100.0
    return v


# ----------------------------------------------------------------------------- kinematics (numpy, calib/fk.py)
fk = _module(REPO / "calib" / "fk.py")  # URDF FK; base_link -> desk = fk.T_WORLD_URDF (validated vs Isaac)
TCP_OFFSET = np.array([-0.0088 + 0.015, 0.0, -0.080])  # env_cfg.TCP_OFFSET in the gripper link (= URDF gripper_link)


def tcp_fk(q5):
    """Grasp point (30 mm cube) and gripper x-axis (jaw-closing direction) in the desk frame, 5 arm angles [rad]."""
    T = fk.link_poses(q5)[4]
    return T[:3, :3] @ TCP_OFFSET + T[:3, 3] + fk.T_WORLD_URDF, T[:3, 0]


def hold_run(s):
    """(start, end) of the cube hold: the longest run (>= 15 frames) with the jaw parked at 13..19.5 right after
    closing from > 20 (empty-closed readings are ~10-12, the 30 mm cube stops the real jaw at ~15). None if none."""
    g = s[:, 5]
    m = (g >= 13) & (g <= 19.5) & (np.abs(np.r_[0, np.diff(g)]) < 0.3)
    runs, st = [], None
    for t, v in enumerate(np.r_[m, False]):
        if v and st is None:
            st = t
        elif not v and st is not None:
            if t - st >= 15 and st >= 5 and s[max(0, st - 25):st, 5].max() > 20:
                runs.append((st, t))
            st = None
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def place_objects(ep, var, det):
    """Cube (xy, yaw) and cup xy for one episode/variant (see PRESETS 'zero')."""
    if var["place"] == "kin":
        a, b = hold_run(ep["state"])
        q = np.radians(ep["state"][:, :5] + var["off"])
        x, ax = tcp_fk(q[a + 3])
        cup = tcp_fk(q[min(b + 2, len(q) - 1)])[0][:2]
        # slot: start the cube this far from the grasp point toward the moving jaw (the closing jaw pushes the cube
        # onto the fixed jaw; placed AT the grasp point the cube is flush with the descending fixed jaw)
        u = ax[:2] / np.linalg.norm(ax[:2])
        return x[:2] + var["slot"] * u, float(np.arctan2(ax[1], ax[0])), cup
    sh = var["shift"]
    return np.array(det["cube"]["xy"]) + sh, det["cube"]["yaw"], np.array(det["cup"]["xy"]) + sh


# ----------------------------------------------------------------------------- camera
def quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_scene_cam():
    d = json.loads((REPO / "calib" / "data" / "calib.json").read_text())
    return (np.array(d["SCENE_K"], float).reshape(3, 3), np.array(d["SCENE_DIST"], float),
            quat_to_R(d["SCENE_CAM_QUAT"]), np.array(d["SCENE_CAM_POS"], float))


def backproject(uv, h, cam):
    """Pixels (n,2) -> desk-frame points on the plane z = h."""
    import cv2
    K, dist, R, C = cam
    xn = cv2.undistortPoints(np.asarray(uv, np.float64).reshape(-1, 1, 2), K, dist).reshape(-1, 2)
    rays = np.c_[xn, np.ones(len(xn))] @ R.T
    return C + rays * ((h - C[2]) / rays[:, 2])[:, None]


def project(X, cam):
    import cv2
    K, dist, R, C = cam
    rvec, _ = cv2.Rodrigues(R.T)
    uv, _ = cv2.projectPoints(np.asarray(X, np.float64).reshape(-1, 1, 3), rvec, -R.T @ C, K, dist)
    return uv.reshape(-1, 2)


def circle_fit(p):
    """Algebraic (Kasa) circle fit -> centre (2,), radius."""
    A = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]
    a, b, c = np.linalg.lstsq(A, (p ** 2).sum(1), rcond=None)[0]
    return np.array([a, b]), float(np.sqrt(c + a * a + b * b))


# ----------------------------------------------------------------------------- detect
def first_frame(path, t0):
    import av
    with av.open(path) as c:
        for fr in c.decode(video=0):
            if fr.time is None or fr.time >= t0 - 1e-3:
                return fr.to_ndarray(format="bgr24")
    raise RuntimeError(path)


def robot_mask(q_sim, cam, size=(640, 480)):
    sys.path.insert(0, str(REPO / "calib"))
    import silhouette as sil
    K, dist, R, C = cam
    V, F, L = sil.get_model().world_mesh(q_sim)
    return np.asarray(sil.render(V, F, L, R, C, K, dist, size, labels=False), bool)


def detect_cube(hsv, roi, rmask, cam):
    import cv2
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    tan = (H >= 5) & (H <= 35) & (S >= 35) & (S <= 175) & (V >= 140) & roi & ~rmask
    tan = cv2.morphologyEx(tan.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(tan)
    best = None
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 60:
            continue
        ys, xs = np.nonzero(lab == k)
        P = backproject(np.c_[xs, ys], CUBE_SIZE, cam)[:, :2]
        (cx, cy), (w, h), ang = cv2.minAreaRect((P * 1000).astype(np.float32))
        score = abs(np.sqrt(w * h) - 32.0) + 0.3 * abs(w - h)  # side faces make the blob a bit larger than 30 mm
        if best is None or score < best[0]:
            best = (score, k, xs, ys)
    if best is None:
        return None
    _, k, xs, ys = best
    v = V[ys, xs]
    top = v >= np.percentile(v, 35)  # the lit top face is brighter than the side faces toward the camera
    if top.sum() >= 30:
        thr = cv2.threshold(v.astype(np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
        if (v > thr).sum() >= 30:
            top = v > thr
    P = backproject(np.c_[xs[top], ys[top]], CUBE_SIZE, cam)[:, :2]
    (cx, cy), (w, h), ang = cv2.minAreaRect((P * 1000).astype(np.float32))
    yaw = np.radians(ang) % (np.pi / 2)
    if yaw > np.pi / 4:
        yaw -= np.pi / 2
    return dict(xy=[cx / 1000, cy / 1000], yaw=float(yaw), side_mm=[float(w), float(h)], px=[float(xs.mean()), float(ys.mean())],
                n_px=int(len(xs)), n_top=int(top.sum()))


def cup_mask(hsv, roi, rmask):
    """Filled silhouette of the cup (cyan inside + white rim/wall) and its outer contour."""
    import cv2
    H, S, V = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    cyan = (H >= 70) & (H <= 105) & (S >= 15) & (S <= 150) & (V >= 120)
    white = (S < 45) & (V >= 170)
    m = ((cyan | white) & roi & ~rmask).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None, None
    k = max(range(1, n), key=lambda i: (cyan & (lab == i)).sum())
    cnts, _ = cv2.findContours((lab == k).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    filled = np.zeros(m.shape, np.uint8)
    cv2.drawContours(filled, [cnt], -1, 1, -1)
    return cnt.reshape(-1, 2), filled.astype(bool)


def rim_fit(cnt, h, cam):
    """Circle through the far-side silhouette (within +-100 deg of 'away from the camera nadir') back-projected to
    the rim plane z = h; the near side is the tapered outer wall. -> centre (2,), radius, rms [m]."""
    P = backproject(cnt, h, cam)[:, :2]
    nadir = cam[3][:2]
    c = P.mean(0)
    for _ in range(5):
        u = (c - nadir) / np.linalg.norm(c - nadir)
        d = P - c
        far = (d @ u) > np.cos(np.radians(100)) * np.linalg.norm(d, axis=1)
        c, r = circle_fit(P[far])
    return c, r, float(np.sqrt(np.mean((np.linalg.norm(P[far] - c, axis=1) - r) ** 2)))


def cup_render(c, rt, rb, h, cam, shape):
    """Silhouette of a tapered cup (rim radius rt at z=h, base radius rb at z=0) = hull of both projected circles."""
    import cv2
    ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    pts = np.r_[np.c_[c[0] + rt * np.cos(ang), c[1] + rt * np.sin(ang), np.full(72, h)],
                np.c_[c[0] + rb * np.cos(ang), c[1] + rb * np.sin(ang), np.zeros(72)]]
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(project(pts, cam).astype(np.int32)), 1)
    return m.astype(bool)


def fit_cup_dims(cups, cam, shape=(480, 640)):
    """Cup height and base radius shared by all episodes: for each (h, rb) the rim is refit at z = h and the
    rendered silhouette compared with the segmented one (mean IoU)."""
    hs, rbs = np.arange(0.070, 0.1301, 0.005), np.arange(0.020, 0.0401, 0.0025)
    iou = np.zeros((len(cups), len(hs), len(rbs)))
    for i, (cnt, mask) in enumerate(cups):
        for a, h in enumerate(hs):
            c, rt, _ = rim_fit(cnt, h, cam)
            for b, rb in enumerate(rbs):
                r = cup_render(c, rt, rb, h, cam, shape)
                iou[i, a, b] = (r & mask).sum() / max((r | mask).sum(), 1)
    a, b = np.unravel_index(np.argmax(iou.mean(0)), iou.shape[1:])
    return float(hs[a]), float(rbs[b]), iou, hs, rbs


def cmd_detect():
    import cv2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam = load_scene_cam()
    eps = load_episodes(args.dataset, args.episodes)
    ys, xs = np.mgrid[0:480, 0:640]
    D = backproject(np.c_[xs.ravel(), ys.ravel()], 0.0, cam)[:, :2].reshape(480, 640, 2)
    r, phi = np.linalg.norm(D, axis=-1), np.arctan2(D[..., 1], D[..., 0])
    roi_cube = (r > 0.06) & (r < 0.36) & (np.abs(phi) < 1.35)
    roi_cup = (r > 0.06) & (r < 0.48) & (np.abs(phi) < 1.45)  # z=0 back-projection of a 10 cm high rim lands ~13% further out
    frames, poses = {}, {}
    for e, ep in eps.items():  # pass 1: segment
        img = first_frame(*scene_video(args.dataset, e))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rm = robot_mask(units.lerobot_to_sim(ep["state"][0]), cam)
        rmd = cv2.dilate(rm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        cnt, mask = cup_mask(hsv, roi_cup, rmd)
        frames[e] = dict(img=img, rm=rm, cnt=cnt, mask=mask)
        poses[e] = dict(cube=detect_cube(hsv, roi_cube, rmd, cam))
    ok = [e for e in eps if frames[e]["cnt"] is not None]
    h_fit, rb_fit, iou, hs, rbs = fit_cup_dims([(frames[e]["cnt"], frames[e]["mask"]) for e in ok], cam)
    a0 = int(np.argmin(np.abs(hs - CUP_H)))
    rb_h = float(rbs[int(np.argmax(iou.mean(0)[a0]))])
    print(f"[detect] cup dims fit (silhouette IoU over {len(ok)} views): best height {1000 * h_fit:.0f} mm / base d {2000 * rb_fit:.0f} mm "
          f"(IoU {iou.mean(0).max():.3f}); at the env_cfg height {1000 * CUP_H:.0f} mm: base d {2000 * rb_h:.0f} mm (IoU "
          f"{iou.mean(0)[a0].max():.3f}). Height and base trade off along a ridge -> NOT determined; rim plane = env_cfg height")
    tiles = []
    for e in eps:  # pass 2: rim at the fitted height, overlays
        f, p = frames[e], poses[e]
        if f["cnt"] is not None:
            c, rt, rms = rim_fit(f["cnt"], CUP_H, cam)
            c2, rt2, _ = rim_fit(f["cnt"], h_fit, cam)
            m = cup_render(c, rt, rb_h, CUP_H, cam, f["mask"].shape)
            p["cup"] = dict(xy=c.tolist(), top_d_mm=2000 * rt, fit_rms_mm=1000 * rms, iou=float((m & f["mask"]).sum() / (m | f["mask"]).sum()),
                            xy_at_fit_h=c2.tolist(), shift_if_h_fit_mm=float(1000 * np.linalg.norm(c2 - c)))
        else:
            p["cup"] = None
        for name in ("cube", "cup"):
            d = p[name]
            if d is None:
                continue
            rr, ph = float(np.hypot(*d["xy"])), float(np.arctan2(d["xy"][1], d["xy"][0]))
            (lo, hi), pm = SPAWN[name]
            d.update(r=rr, phi=ph, in_spawn_range=bool(lo <= rr <= hi and abs(ph) < pm))
        cube, cup = p["cube"], p["cup"]
        if cube and cup:
            p["separation"] = float(np.hypot(cube["xy"][0] - cup["xy"][0], cube["xy"][1] - cup["xy"][1]))
        vis = f["img"].copy()
        cnts, _ = cv2.findContours(f["rm"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(vis, cnts, -1, (255, 0, 255), 1)
        if f["cnt"] is not None:
            cv2.polylines(vis, [f["cnt"].astype(np.int32)], True, (0, 255, 255), 1)
        if cup:
            ang = np.linspace(0, 2 * np.pi, 72)
            for z, rad, col in ((CUP_H, cup["top_d_mm"] / 2000, (0, 0, 255)), (0.0, rb_h, (0, 128, 255))):
                ring = np.c_[cup["xy"][0] + rad * np.cos(ang), cup["xy"][1] + rad * np.sin(ang), np.full(72, z)]
                cv2.polylines(vis, [project(ring, cam).astype(np.int32)], True, col, 1, cv2.LINE_AA)
            cv2.drawMarker(vis, tuple(project([[*cup["xy"], CUP_H]], cam)[0].astype(int)), (0, 0, 255), cv2.MARKER_CROSS, 8)
        if cube:
            c, s, h = np.array(cube["xy"]), CUBE_SIZE / 2, CUBE_SIZE
            Rz = np.array([[np.cos(cube["yaw"]), -np.sin(cube["yaw"])], [np.sin(cube["yaw"]), np.cos(cube["yaw"])]])
            sq = (np.array([[-s, -s], [s, -s], [s, s], [-s, s]]) @ Rz.T) + c
            for z, col in ((h, (0, 255, 0)), (0.0, (0, 160, 0))):
                cv2.polylines(vis, [project(np.c_[sq, np.full(4, z)], cam).astype(np.int32)], True, col, 1, cv2.LINE_AA)
        txt = [f"ep{e}"]
        if cube:
            txt.append(f"cube ({cube['xy'][0]:.3f},{cube['xy'][1]:.3f}) yaw {np.degrees(cube['yaw']):.0f}"
                       + ("" if cube["in_spawn_range"] else " OUT"))
        if cup:
            txt.append(f"cup ({cup['xy'][0]:.3f},{cup['xy'][1]:.3f}) d{cup['top_d_mm']:.0f}mm" + ("" if cup["in_spawn_range"] else " OUT"))
        for i, t in enumerate(txt):
            cv2.putText(vis, t, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, t, (6, 18 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(out / f"detect_ep{e}.jpg"), vis)
        tiles.append(vis[40:440, 40:520])
        print(f"[detect] ep{e}: " + " | ".join(txt[1:]) + (f" | cube side {cube['side_mm'][0]:.0f}x{cube['side_mm'][1]:.0f} mm"
              if cube else " | NO CUBE") + (f" | rim rms {cup['fit_rms_mm']:.1f} mm IoU {cup['iou']:.2f}" if cup else " | NO CUP"), flush=True)
    cols = 5
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)], 0)
    cv2.imwrite(str(out / "detect_debug.jpg"), grid)
    meta = dict(camera="calib/data/calib.json SCENE_*", cube_size=CUBE_SIZE, cup_height_assumed=CUP_H, cup_height_fit_unreliable=h_fit,
                cup_top_d_fit=float(np.median([poses[e]["cup"]["top_d_mm"] for e in ok]) / 1000),
                note="cube xy = top-face centre (z=30 mm), yaw mod 90 deg; cup xy = rim centre at z = env_cfg cup height")
    (out / "poses.json").write_text(json.dumps(dict(meta=meta, episodes={str(k): v for k, v in poses.items()}), indent=1))
    print(f"[detect] cup top d median {1000 * meta['cup_top_d_fit']:.0f} mm; wrote {out / 'poses.json'} and {out / 'detect_debug.jpg'}")


# ----------------------------------------------------------------------------- metrics (numpy)
def xcorr_lag(a, s, maxlag=12):
    """Lag [frames] of s behind a: peak of the normalised cross-correlation of their velocities (sub-frame)."""
    va, vs = np.diff(a), np.diff(s)
    cc = []
    for k in range(maxlag):
        x, y = va[:len(va) - k], vs[k:]
        cc.append(float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9)))
    k = int(np.argmax(cc))
    if 0 < k < maxlag - 1:
        y0, y1, y2 = cc[k - 1], cc[k], cc[k + 1]
        den = y0 - 2 * y1 + y2
        return k + (0.5 * (y0 - y2) / den if den < 0 else 0.0)
    return float(k)


def static_mask(a, win=15, tol=0.5):
    """Frames whose action has stayed within tol for the previous win frames (per joint)."""
    m = np.zeros(a.shape, bool)
    for t in range(win, len(a)):
        m[t] = np.ptp(a[t - win:t + 1], axis=0) < tol
    return m


def episode_metrics(real_s, real_a, sim_s):
    """Real-unit comparison over the real episode length."""
    T = len(real_s)
    sim_s = sim_s[:T]
    err = sim_s - real_s
    st = static_mask(real_a)
    hr = hold_run(real_s)
    gm = np.zeros(T, bool)
    if hr:
        gm[hr[0] + 5:hr[1]] = True
    tg = hr[0] + 3 if hr else None
    out = dict(rmse=np.sqrt((err ** 2).mean(0)), maxerr=np.abs(err).max(0),
               lag_real=np.array([xcorr_lag(real_a[:, j], real_s[:, j]) for j in range(6)]),
               lag_sim=np.array([xcorr_lag(real_a[:, j], sim_s[:, j]) for j in range(6)]),
               trk95_real=np.percentile(np.abs(real_a - real_s), 95, axis=0),
               trk95_sim=np.percentile(np.abs(real_a - sim_s), 95, axis=0),
               ss_real=np.array([(real_s - real_a)[st[:, j], j].mean() if st[:, j].any() else np.nan for j in range(6)]),
               ss_sim=np.array([(sim_s - real_a)[st[:, j], j].mean() if st[:, j].any() else np.nan for j in range(6)]),
               grip_hold_real=float(np.median(real_s[gm, 5])) if gm.any() else np.nan,
               grip_hold_sim=float(np.median(sim_s[gm, 5])) if gm.any() else np.nan, t_grasp=tg)
    return out


def summarize(npz, quiet=False):
    """Per-variant aggregate table from a replay npz."""
    d = np.load(npz, allow_pickle=True)
    variants = list(d["variants"])
    ep_ids, var_ids, lens = d["env_episode"], d["env_variant"], d["lengths"]
    rows = []
    for vi, var in enumerate(variants):
        envs = np.nonzero(var_ids == vi)[0]
        M, task = [], []
        for i in envs:
            e = int(ep_ids[i])
            T = int(lens[i])
            rs, ra = d[f"real_state_{e}"], d[f"real_action_{e}"]
            sim = to_real(d["q"][i], var)
            m = episode_metrics(rs, ra, sim)
            tend = min(T + int(d["settle"]), d["q"].shape[1] - 1)
            cube, cube0 = d["cube"][i], d["cube"][i, 0]
            tg = m["t_grasp"]
            tcp_err = (d["tcp"][i, tg] - cube0) if tg is not None else np.full(3, np.nan)
            # camera-free kinematics of the REAL readings under this variant's joint offsets
            q5 = np.radians(rs[:, :5] + var["off"])
            tip_z = np.array([fk.fingertip_in_world(q)[2] for q in q5[::2]])
            kin = tcp_fk(q5[tg])[0] if tg is not None else np.full(3, np.nan)
            det = d["det_cube"][i] if "det_cube" in d else np.full(2, np.nan)
            hr = hold_run(rs)
            held = bool(hr and cube[(hr[0] + hr[1]) // 2, 2] > CUBE_SIZE / 2 + 0.01)
            succ = bool(d["in_cup"][i, tend] and d["upright"][i, tend])
            task.append(dict(ep=e, success=succ, lifted=bool(cube[:, 2].max() > CUBE_SIZE / 2 + 0.02), held=held,
                             tipped=bool(not d["upright"][i, tend]),
                             fail="" if succ else ("cup tipped" if not d["upright"][i, tend] else "dropped/missed cup" if held else "grasp missed"),
                             tcp_xy_mm=float(1000 * np.hypot(*tcp_err[:2])), tcp_dz_mm=float(1000 * tcp_err[2]),
                             kin_xy_mm=float(1000 * np.hypot(*(kin[:2] - det))), tip_z_grasp_mm=float(1000 * fk.fingertip_in_world(q5[tg])[2]) if tg is not None else np.nan,
                             tip_z_min_mm=float(1000 * tip_z.min()), max_cube_z=float(cube[:, 2].max())))
            M.append((m, T))
        w = np.array([T for _, T in M], float)
        agg = lambda k: np.sqrt(np.sum([m[k] ** 2 * T for m, T in M], 0) / w.sum()) if k == "rmse" else np.nanmedian([m[k] for m, _ in M], 0)  # noqa: E731
        rows.append(dict(name=var["name"], var={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in var.items()},
                         rmse=agg("rmse").tolist(), rmse_arm=float(np.sqrt(np.mean(agg("rmse")[:5] ** 2))),
                         lag_real=agg("lag_real").tolist(), lag_sim=agg("lag_sim").tolist(),
                         trk95_real=agg("trk95_real").tolist(), trk95_sim=agg("trk95_sim").tolist(),
                         ss_real=agg("ss_real").tolist(), ss_sim=agg("ss_sim").tolist(),
                         grip_hold_real=float(np.nanmedian([m["grip_hold_real"] for m, _ in M])),
                         grip_hold_sim=float(np.nanmedian([m["grip_hold_sim"] for m, _ in M])),
                         success=int(sum(t["success"] for t in task)), lifted=int(sum(t["lifted"] for t in task)),
                         held=int(sum(t["held"] for t in task)), tipped=int(sum(t["tipped"] for t in task)), n=len(task),
                         kin_xy_mm=float(np.sqrt(np.nanmean([t["kin_xy_mm"] ** 2 for t in task]))),
                         tip_z_grasp_mm=float(np.nanmedian([t["tip_z_grasp_mm"] for t in task])),
                         tip_z_min_mm=float(np.nanmin([t["tip_z_min_mm"] for t in task])),
                         tcp_xy_mm=float(np.nanmedian([t["tcp_xy_mm"] for t in task])),
                         tcp_dz_mm=float(np.nanmedian([t["tcp_dz_mm"] for t in task])), per_episode=task))
    if not quiet:
        f = lambda x: " ".join(f"{v:5.2f}" for v in x)  # noqa: E731
        print(f"{'variant':18s} | RMSE deg (pan lift elbow wflex wroll | grip units) | armRMSE | lag sim (real {f(rows[0]['lag_real'][:5])}) "
              f"| trk95 sim (real {f(rows[0]['trk95_real'][:5])}) | grip hold sim/real | simTCP-cube xy/dz | kinTCP-det xy, tip z grasp/min mm "
              f"| held succ tipped")
        for r in rows:
            print(f"{r['name']:18s} | {f(r['rmse'])} | {r['rmse_arm']:5.2f} | {f(r['lag_sim'][:5])} | {f(r['trk95_sim'][:5])} | "
                  f"{r['grip_hold_sim']:5.1f}/{r['grip_hold_real']:5.1f} | {r['tcp_xy_mm']:5.1f}/{r['tcp_dz_mm']:+5.1f} | "
                  f"{r['kin_xy_mm']:5.1f} {r['tip_z_grasp_mm']:+5.1f}/{r['tip_z_min_mm']:+5.1f} | "
                  f"{r['held']:2d} {r['success']:2d}/{r['n']} {r['tipped']:2d}")
    return rows


def per_joint_best(rows):
    """For a uniform-scale gain grid (all variants with scalar kp/kd): per delay, the best (kp, kd) of every arm joint
    by that joint's RMSE; prints the joint-wise table and the per-joint variant spec to confirm."""
    by_delay = {}
    for r in rows:
        v = r["var"]
        if np.ndim(v["kp"]) or np.ndim(v["kd"]):
            continue
        by_delay.setdefault(v["delay"], []).append(r)
    out = {}
    for dly, rs in sorted(by_delay.items()):
        best = [min(rs, key=lambda r: r["rmse"][j]) for j in range(5)]
        rm = [b["rmse"][j] for j, b in enumerate(best)]
        out[dly] = dict(kp=[b["var"]["kp"] for b in best], kd=[b["var"]["kd"] for b in best], rmse=rm,
                        lag=[b["lag_sim"][j] for j, b in enumerate(best)], trk95=[b["trk95_sim"][j] for j, b in enumerate(best)],
                        rmse_arm=float(np.sqrt(np.mean(np.square(rm)))))
        o = out[dly]
        print(f"[per-joint] delay {dly}: armRMSE {o['rmse_arm']:.2f} | " + " | ".join(
            f"{SHORT[j]} kp{o['kp'][j]:g} kd{o['kd'][j]:g} rmse {o['rmse'][j]:.2f} lag {o['lag'][j]:.2f} trk95 {o['trk95'][j]:.2f}"
            for j in range(5)))
        print(f"[per-joint]   spec: kp={','.join(f'{x:g}' for x in o['kp'])}/kd={','.join(f'{x:g}' for x in o['kd'])}/delay={dly}")
    return out


def cmd_analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(args.out)
    tag = args.tag or args.sweep
    npz = out / f"{tag}.npz"
    rows = summarize(npz)
    (out / f"{tag}_summary.json").write_text(json.dumps(rows, indent=1, default=float))
    if any(np.ndim(r["var"]["kp"]) == 0 and r["var"]["kp"] != 1.0 for r in rows):
        per_joint_best(rows)
    d = np.load(npz, allow_pickle=True)
    variants = list(d["variants"])
    names = [v["name"] for v in variants]
    pick = args.plot_variants or [names[0]] + ([min(rows, key=lambda r: r["rmse_arm"])["name"]] if len(names) > 1 else [])
    pick = list(dict.fromkeys(pick))
    for e in args.plot_eps:
        rs, ra = d[f"real_state_{e}"], d[f"real_action_{e}"]
        t = np.arange(len(rs)) / FPS
        fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
        for j, ax in enumerate(axs.T.flat if False else axs.flat):
            ax.plot(t, ra[:, j], color="0.6", ls="--", lw=1, label="real action")
            ax.plot(t, rs[:, j], color="k", lw=1.6, label="real state")
            for k, name in enumerate(pick):
                vi = names.index(name)
                i = int(np.nonzero((d["env_episode"] == e) & (d["env_variant"] == vi))[0][0])
                sim = to_real(d["q"][i], variants[vi])[:len(rs)]
                ax.plot(t, sim[:, j], lw=1.2, color=["tab:red", "tab:blue", "tab:green", "tab:orange"][k % 4], label=f"sim {name}")
            ax.set_title(f"{JOINTS[j]} [{'0-100' if j == 5 else 'deg'}]")
            ax.grid(alpha=0.3)
        axs[0, 0].legend(fontsize=8)
        for ax in axs[-1]:
            ax.set_xlabel("t [s]")
        fig.suptitle(f"episode {e}: real follower vs open-loop sim replay ({tag})")
        fig.tight_layout()
        fig.savefig(out / f"{tag}_ep{e}.png", dpi=80)
        plt.close(fig)
    print(f"[analyze] wrote {out}/{tag}_summary.json and plots for episodes {args.plot_eps} (variants {pick})")


# ----------------------------------------------------------------------------- replay (Isaac)
def cmd_replay():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = load_episodes(args.dataset, args.episodes)
    poses = json.loads((out / "poses.json").read_text())["episodes"]
    variants = [parse_variant(s) for s in PRESETS[args.sweep]] + [parse_variant(s) for s in args.variants]
    tag = args.tag or args.sweep
    ep_list = list(eps)
    N = len(variants) * len(ep_list)
    env_variant = np.repeat(np.arange(len(variants)), len(ep_list))
    env_episode = np.tile(np.array(ep_list), len(variants))
    lengths = np.array([len(eps[e]["action"]) for e in env_episode])
    Tm = int(lengths.max()) + args.settle
    # per-env action tables (sim units, delayed, last action held)
    acts = np.zeros((N, Tm, 6))
    q0 = np.zeros((N, 6))
    place = np.zeros((N, 5))  # cube x y yaw, cup x y
    for i in range(N):
        var, ep = variants[env_variant[i]], eps[env_episode[i]]
        a = to_sim(ep["action"], var)
        idx = np.clip(np.arange(Tm) - var["delay"], 0, len(a) - 1)
        acts[i] = a[idx]
        q0[i] = to_sim(ep["state"][0], var)
        c, yaw, cup = place_objects(ep, var, poses[str(env_episode[i])])
        place[i] = [c[0], c[1], yaw, cup[0], cup[1]]
    print(f"[replay] {len(variants)} variants x {len(ep_list)} episodes = {N} envs, {Tm} steps", flush=True)

    import gymnasium as gym
    import torch

    import so101_r2s2r  # noqa: F401
    from isaaclab.utils import math as math_utils
    from isaaclab_tasks.utils import resolve_task_config
    from so101_r2s2r import mdp

    env_cfg, _ = resolve_task_config(args.task, "")
    env_cfg.scene.num_envs = N
    env_cfg.episode_length_s = Tm / FPS + 10.0
    for name in ("success", "cube_dropped", "cup_tipped", "physics_blowup"):
        setattr(env_cfg.terminations, name, None)
    env_cfg.events.reset_robot = None
    env_cfg.events.object_friction = None
    with launch_simulation(env_cfg, args):
        env = gym.make(args.task, cfg=env_cfg)
        env.reset(seed=0)
        u = env.unwrapped
        dev = u.device
        robot = u.scene["robot"]
        jid = [robot.joint_names.index(j) for j in JOINTS]
        arm = torch.tensor(jid[:5], device=dev)
        all_ids = torch.arange(N, device=dev)
        K0, D0 = robot.data.joint_stiffness.torch.clone(), robot.data.joint_damping.torch.clone()
        print("[replay] USD gains (env 0):", {j: (round(float(K0[0, k]), 3), round(float(D0[0, k]), 4)) for j, k in zip(JOINTS, jid)})
        print("[replay] effort limits:", [round(float(x), 2) for x in robot.data.joint_effort_limits.torch[0, jid]],
              "vel limits:", [round(float(x), 2) for x in robot.data.joint_vel_limits.torch[0, jid]],
              "armature:", [round(float(x), 4) for x in robot.data.joint_armature.torch[0, jid]])
        kp = torch.tensor(np.array([np.broadcast_to(variants[v]["kp"], 5) for v in env_variant]), device=dev, dtype=torch.float32)
        kd = torch.tensor(np.array([np.broadcast_to(variants[v]["kd"], 5) for v in env_variant]), device=dev, dtype=torch.float32)
        robot.write_joint_stiffness_to_sim_index(stiffness=(K0[:, arm] * kp).contiguous(), joint_ids=arm, env_ids=all_ids)
        robot.write_joint_damping_to_sim_index(damping=(D0[:, arm] * kd).contiguous(), joint_ids=arm, env_ids=all_ids)
        if args.debug_gains:
            for i in (0, N - 1):
                print(f"[gains] env {i} ({variants[env_variant[i]]['name']}): data K", np.round(robot.data.joint_stiffness.torch[i].cpu().numpy(), 2).tolist(),
                      "D", np.round(robot.data.joint_damping.torch[i].cpu().numpy(), 3).tolist(), "joint_names", robot.joint_names)
            print("[gains] physx view K", np.round(np.asarray(robot.root_view.get_dof_stiffnesses())[[0, N - 1]], 2).tolist(),
                  "D", np.round(np.asarray(robot.root_view.get_dof_dampings())[[0, N - 1]], 3).tolist(), "dof names", robot.root_view.shared_metatype.dof_names)
        # initial robot state (clamped to the sim joint limits) and object poses
        lim = robot.data.joint_pos_limits.torch[0, jid].cpu().numpy()
        q0c = np.clip(q0, lim[:, 0], lim[:, 1])
        print("[replay] q0 clamped by the sim joint limits (max |deg| per joint):", np.round(np.degrees(np.abs(q0c - q0).max(0)), 2).tolist())
        qfull = robot.data.joint_pos.torch.clone()
        qfull[:, jid] = torch.tensor(q0c, device=dev, dtype=torch.float32)
        robot.write_joint_state_to_sim_index(position=qfull, velocity=torch.zeros_like(qfull))
        k_cube = list(env_cfg.cube_sizes).index(CUBE_SIZE)
        mdp.active_cube_idx(u)[:] = k_cube
        origins = u.scene.env_origins
        zeros = torch.zeros(N, device=dev)
        pl = torch.tensor(place, device=dev, dtype=torch.float32)
        cube_xy, cube_yaw, cup_xy = pl[:, 0:2], pl[:, 2], pl[:, 3:5]
        for k, name in enumerate(mdp.CUBE_NAMES):
            if k != k_cube:
                continue
            pos = origins + torch.stack([cube_xy[:, 0], cube_xy[:, 1], zeros + CUBE_SIZE / 2 + 0.001], -1)
            quat = math_utils.quat_from_euler_xyz(zeros, zeros, cube_yaw)
            u.scene[name].write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1), env_ids=all_ids)
            u.scene[name].write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=dev), env_ids=all_ids)
        cup_pose = torch.cat([origins + torch.stack([cup_xy[:, 0], cup_xy[:, 1], zeros + 0.001], -1),
                              math_utils.quat_from_euler_xyz(zeros, zeros, zeros)], -1)
        u.scene["cup"].write_root_pose_to_sim_index(root_pose=cup_pose, env_ids=all_ids)
        u.scene["cup"].write_root_velocity_to_sim_index(root_velocity=torch.zeros(N, 6, device=dev), env_ids=all_ids)

        rec = dict(q=np.zeros((N, Tm + 1, 6), np.float32), tcp=np.zeros((N, Tm + 1, 3), np.float32),
                   cube=np.zeros((N, Tm + 1, 3), np.float32), cup=np.zeros((N, Tm + 1, 3), np.float32),
                   in_cup=np.zeros((N, Tm + 1), bool), upright=np.zeros((N, Tm + 1), bool))
        cube_name = mdp.CUBE_NAMES[k_cube]

        def record(t):
            rec["q"][:, t] = robot.data.joint_pos.torch[:, jid].cpu().numpy()
            rec["tcp"][:, t] = (mdp.tcp_pos_w(u) - origins).cpu().numpy()
            rec["cube"][:, t] = (u.scene[cube_name].data.root_pos_w.torch - origins).cpu().numpy()
            rec["cup"][:, t] = (u.scene["cup"].data.root_pos_w.torch - origins).cpu().numpy()
            rec["in_cup"][:, t] = mdp.cube_in_cup(u).cpu().numpy()
            rec["upright"][:, t] = mdp.cup_upright(u).cpu().numpy()

        acts_t = torch.tensor(acts, device=dev, dtype=torch.float32)
        resets = 0
        u.scene.update(u.physics_dt)
        for t in range(Tm):
            record(t)
            _, _, term, trunc, _ = env.step(acts_t[:, t])
            resets += int((term | trunc).sum())
            if t % 150 == 0:
                print(f"[replay] step {t}/{Tm}", flush=True)
        record(Tm)
        print("[replay] gains after replay (env 0, arm):", [round(float(x), 3) for x in robot.data.joint_stiffness.torch[0, arm]],
              f"env resets during replay: {resets}")
        save = dict(rec, variants=np.array(variants, dtype=object), env_variant=env_variant, env_episode=env_episode,
                    lengths=lengths, settle=args.settle, q0_sim=q0c, place=place,
                    det_cube=np.array([poses[str(e)]["cube"]["xy"] for e in env_episode]))
        for e in ep_list:
            save[f"real_state_{e}"], save[f"real_action_{e}"] = eps[e]["state"], eps[e]["action"]
        np.savez_compressed(out / f"{tag}.npz", **save)
        rows = summarize(out / f"{tag}.npz")  # inside the context: closing the app ends the process
        (out / f"{tag}_summary.json").write_text(json.dumps(rows, indent=1, default=float))
        print(f"[replay] wrote {out / tag}.npz and {tag}_summary.json", flush=True)
        env.close()


if __name__ == "__main__":
    {"detect": cmd_detect, "replay": cmd_replay, "analyze": cmd_analyze}[args.mode]()

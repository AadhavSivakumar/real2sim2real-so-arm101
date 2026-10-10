"""Validate the fitted wrist-camera mount (calib/data/wrist_mount_proposal.json, from calib/wrist_mount.py) in Isaac.

1. Frame check -- establishes the URDF gripper_link -> USD 'gripper' transform used by calib/wrist_mount.py:
   * robot.data.body_link_pos_w / quat_w of 'gripper' and 'moving_jaw_so101_v1' vs URDF FK (calib/fk.py chain +
     the URDF 'gripper' jaw joint) at several arm / jaw configurations (constant offset => frames agree),
   * the USD visual meshes of both bodies vs the URDF STLs placed in gripper_link (nearest-point distances),
   * every camera prim's world pose (Camera.data) vs gripper body pose x the configured offset,
   * the camera / mount prims NVIDIA's USD carries under the gripper (sensors variant).
2. Builds the Cam env with env_cfg.scene.wrist_cam.offset overridden in code by the proposal (calib.json is never
   written), a camera 'wrist_cam_nvidia' at the current (NVIDIA) offset and a camera 'wrist_cam_exactK' at the
   fitted offset whose spawn carries OpenCvPinholeDistortionCfg(exact K, coefficients muted) (--no_exact_k to skip);
   robot gravity off, arm at the desk-viewing home pose, jaw at the fitted angle; renders RGB + depth. Then the jaw
   is moved to NVIDIA's own best jaw angle (CPU fit) and the NVIDIA camera is rendered again (fair 'before').
   NVIDIA's camera-mount model (gripper/visuals/camera_mount + gripper/collisions/camera_mount, part of the USD)
   sits right in front of the real camera position: a prestartup event hides the visual and disables its colliders
   (--keep_nvidia_mount to keep it) -- the same function the env needs when the proposal is applied.
   update_latest_camera_pose=True so Camera.data reports the live pose.
3. Jaw mask = depth < --depth_max (only the gripper is that close). IoU / boundary residual vs the real median's
   segmentation, undistorted (a) to K (the documented pipeline) and (b) to K_sim (the centred-principal-point,
   mean-focal pinhole Isaac's default spawn renders); and a numpy re-render of the URDF meshes from the camera pose
   Isaac reports, with Isaac's reported K and with the other K (end-to-end convention check + which projection the
   renderer really used). Side-by-side + overlays -> --scratch and calib/data/wrist_mount_isaac_*.jpg; results ->
   'frame_check' / 'isaac_validation' / 'apply_requirements' of the proposal JSON (only if the proposal did not
   change during the run) and always --scratch/isaac_results.json.

Needs the real-image PNGs from `calib/wrist_mount.py fit --scratch DIR` in DIR.
Run (GPU lock): /mnt/secondary/so101_r2s/gpu_run.sh LOG -- ./lab.sh -p scripts/check_wrist_mount.py --scratch DIR --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--scratch", required=True, help="dir with real_*.png from calib/wrist_mount.py; outputs go here too")
parser.add_argument("--proposal", default=None, help="default: calib/data/wrist_mount_proposal.json")
parser.add_argument("--depth_max", type=float, default=0.14, help="jaw mask: depth to image plane below this [m]")
parser.add_argument("--settle_steps", type=int, default=90)
parser.add_argument("--no_write", action="store_true", help="do not update the proposal JSON / calib/data jpgs")
parser.add_argument("--keep_nvidia_mount", action="store_true", help="do not remove NVIDIA's camera-mount model")
parser.add_argument("--no_exact_k", action="store_true", help="skip the OpenCV exact-K camera")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import copy
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.spatial.transform import Rotation

import so101_r2s2r  # noqa: F401  (registers tasks)
from isaaclab.sensors import CameraCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import OpenCvPinholeDistortionCfg
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import calib as sim_calib
from so101_r2s2r.env_cfg import HOME_Q, JOINTS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "calib"))
import fk  # noqa: E402
import wrist_mount as wm  # noqa: E402  (numpy/scipy parts only; cv2 not needed here)

TASK = "SO101-CubeInCup-Cam-v0"
SCRATCH = Path(args.scratch)
SCRATCH.mkdir(parents=True, exist_ok=True)
PROPOSAL = Path(args.proposal) if args.proposal else wm.PROPOSAL
prop = json.loads(PROPOSAL.read_text())
FIT_POS, FIT_QUAT, Q_JAW = tuple(prop["WRIST_CAM_POS"]), tuple(prop["WRIST_CAM_QUAT"]), float(prop["jaw_angle_rad"])
NV_POS, NV_QUAT = tuple(sim_calib.WRIST_CAM_POS), tuple(sim_calib.WRIST_CAM_QUAT)
Q_NV = float(prop.get("nvidia_pose", {}).get("best_jaw_angle_rad", Q_JAW))  # NVIDIA pose's own best jaw angle (CPU fit)
K_EXACT = np.asarray(sim_calib.WRIST_K, float).reshape(3, 3)
IMG_W, IMG_H = sim_calib.IMAGE_SIZE
print(f"[check] fitted offset pos {FIT_POS} quat {FIT_QUAT}, jaw {Q_JAW:.4f} rad; NVIDIA pos {NV_POS} quat {NV_QUAT} "
      f"(its best jaw {Q_NV:.4f} rad)")

env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = 1
env_cfg.scene.robot.spawn.rigid_props.disable_gravity = True  # the arm holds the commanded pose exactly
env_cfg.events.scene_cam_jitter = None
for _t in ("success", "cube_dropped", "cup_tipped", "physics_blowup"):  # no auto-reset while the arm is posed
    setattr(env_cfg.terminations, _t, None)
env_cfg.scene.wrist_cam.data_types = ["rgb", "distance_to_image_plane"]
env_cfg.scene.wrist_cam.update_latest_camera_pose = True  # Camera.data.pos_w/quat_w are otherwise the spawn-time pose
env_cfg.scene.wrist_cam_nvidia = copy.deepcopy(env_cfg.scene.wrist_cam)
env_cfg.scene.wrist_cam_nvidia.prim_path = "{ENV_REGEX_NS}/Robot/gripper/wrist_cam_nvidia"
env_cfg.scene.wrist_cam_nvidia.offset = CameraCfg.OffsetCfg(pos=NV_POS, rot=NV_QUAT, convention="opengl")
env_cfg.scene.wrist_cam.offset = CameraCfg.OffsetCfg(pos=FIT_POS, rot=FIT_QUAT, convention="opengl")  # the proposal
CAMS = {"fitted": ("wrist_cam", FIT_POS, FIT_QUAT), "nvidia": ("wrist_cam_nvidia", NV_POS, NV_QUAT)}
if not args.no_exact_k:
    # option (b) of apply_requirements: same fitted offset, spawn with the exact OpenCV intrinsics, no distortion
    env_cfg.scene.wrist_cam_exactK = copy.deepcopy(env_cfg.scene.wrist_cam)
    env_cfg.scene.wrist_cam_exactK.prim_path = "{ENV_REGEX_NS}/Robot/gripper/wrist_cam_exactK"
    env_cfg.scene.wrist_cam_exactK.spawn.distortion = OpenCvPinholeDistortionCfg(
        fx=float(K_EXACT[0, 0]), fy=float(K_EXACT[1, 1]), cx=float(K_EXACT[0, 2]), cy=float(K_EXACT[1, 2]),
        image_size=(int(IMG_W), int(IMG_H)), apply_lens_distortion=False)
    CAMS["fitted_exactK"] = ("wrist_cam_exactK", FIT_POS, FIT_QUAT)
NV_MOUNT, NV_MOUNT_COL = wm.NV_MOUNT_VISUAL, wm.NV_MOUNT_COLLISION  # NVIDIA's camera-mount model; not on the real arm
MOUNT_LOG = {}


def _editable_prim(stage, path):
    """The prim at `path`, made editable by de-instancing its enclosing instanceable prim(s); None if missing."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    while prim.IsInstanceProxy():
        anc = prim.GetParent()
        while not anc.IsInstance():
            anc = anc.GetParent()
        anc.SetInstanceable(False)
        prim = stage.GetPrimAtPath(path)
    return prim


def remove_nvidia_mount(env, env_ids, robot_prim="Robot"):
    """mode='prestartup' event: remove NVIDIA's camera-mount model (USD 'sensors' variant of the SO-101 asset) from
    the robot -- the visual <robot>/gripper/visuals/camera_mount is made invisible and every collider under
    <robot>/gripper/collisions/camera_mount gets physics:collisionEnabled = False (before PhysX parses the stage).
    It occupies y = 24..84 mm, z = -56..-18 mm of the gripper frame, right in front of the real (fitted) camera,
    which would otherwise image only the mount; the real arm has nothing there. Use this event in env_cfg when the
    proposal is applied (apply_requirements 'remove_nvidia_camera_mount')."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = sim_utils.get_current_stage()
    n_vis, n_col, missing = 0, 0, []
    for i in range(env.num_envs):
        root = f"/World/envs/env_{i}/{robot_prim}"
        vis = _editable_prim(stage, f"{root}/{NV_MOUNT}")
        if vis is None:
            missing.append(f"{root}/{NV_MOUNT}")
        else:
            UsdGeom.Imageable(vis).MakeInvisible()
            n_vis += 1
        col = stage.GetPrimAtPath(f"{root}/{NV_MOUNT_COL}")
        if not col.IsValid():
            missing.append(f"{root}/{NV_MOUNT_COL}")
            continue
        paths = [p.GetPath() for p in Usd.PrimRange(col, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
        for path in paths:
            p = _editable_prim(stage, path)
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
            n_col += 1
    MOUNT_LOG.update(visuals_hidden=n_vis, colliders_disabled=n_col, missing=missing)
    print(f"[mount] NVIDIA camera mount: {n_vis} visual(s) hidden, {n_col} collider(s) disabled, missing {missing}")


if not args.keep_nvidia_mount:
    from isaaclab.managers import EventTermCfg

    env_cfg.events.remove_nvidia_mount = EventTermCfg(func=remove_nvidia_mount, mode="prestartup")


def T_from(pos, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, float)).as_matrix()
    T[:3, 3] = pos
    return T


def T_inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def T_err(T):
    """(translation [mm], rotation [deg]) of a 4x4 that should be identity."""
    return float(np.linalg.norm(T[:3, 3]) * 1000), float(np.degrees(Rotation.from_matrix(T[:3, :3]).magnitude()))


def mean_T(Ts):
    R = Rotation.from_matrix(np.stack([T[:3, :3] for T in Ts])).mean().as_matrix()
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.mean([T_[:3, 3] for T_ in Ts], 0)
    return T


def png(path, a):
    Image.fromarray(a).save(path)


def edge(m, width=2):
    m = np.asarray(m, bool)
    b = m & ~ndimage.binary_erosion(m, border_value=1)
    return ndimage.binary_dilation(b, iterations=width - 1) if width > 1 else b


def draw(img, m, color, width=2):
    out = img.copy()
    out[edge(m, width)] = color
    return out


def put_label(img, text):
    """Tiny label bar (PIL default font, '\\n' = new line) on top of an RGB uint8 image."""
    from PIL import ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for k, line in enumerate(text.split("\n")):
        d.rectangle([0, 14 * k, 8 + 6 * len(line), 14 * (k + 1)], fill=(0, 0, 0))
        d.text((4, 2 + 14 * k), line, fill=(255, 255, 255))
    return np.asarray(im)


def brighten(img, gamma=0.5):
    return (255 * (img.astype(np.float32) / 255.0) ** gamma).astype(np.uint8)


with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg)
    env.reset(seed=0)
    u = env.unwrapped
    dev = u.device
    robot = u.scene["robot"]
    origin = u.scene.env_origins[0].cpu().numpy()
    gi, ji = robot.body_names.index("gripper"), robot.body_names.index("moving_jaw_so101_v1")
    jid = {n: robot.joint_names.index(n) for n in JOINTS}
    model = wm.GripperModel()

    def body_T(idx):
        d = robot.data
        p = d.body_link_pos_w.torch[0, idx].cpu().numpy() - origin
        q = d.body_link_quat_w.torch[0, idx].cpu().numpy()
        # cross-check the quaternion convention (xyzw) against IsaacLab's own matrix conversion
        Rm = math_utils.matrix_from_quat(d.body_link_quat_w.torch[0:1, idx])[0].cpu().numpy()
        assert np.allclose(Rm, Rotation.from_quat(q).as_matrix(), atol=1e-5), "quaternion convention mismatch"
        return T_from(p, q)

    def set_q(qvec):
        jp = robot.data.joint_pos.torch.clone()
        for n, v in zip(JOINTS, qvec):
            jp[0, jid[n]] = float(v)
        robot.write_joint_position_to_sim_index(position=jp)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp))
        robot.set_joint_position_target_index(target=jp)
        u.sim.forward()
        u.scene.update(u.physics_dt)

    def urdf_T(q_act):
        """URDF gripper_link and moving-jaw link poses in the desk frame (fk.T_WORLD_URDF: base_link -> desk)."""
        Tb = np.eye(4)
        Tb[:3, 3] = fk.T_WORLD_URDF
        Tg = Tb @ fk.link_poses(q_act[:5])[4]  # chain index 4 = wrist_roll's child = gripper_link
        return Tg, Tg @ model.jaw_transform(q_act[5])

    # ------------------------------------------------------------------ 1. frame check
    rng = np.random.default_rng(1)
    configs = [np.zeros(6), np.r_[[HOME_Q[j] for j in JOINTS[:5]], Q_JAW]]
    configs += [np.r_[rng.uniform(-1.2, 1.2, 5), rng.uniform(-0.15, 1.7)] for _ in range(8)]
    X_g, X_j, X_jg = [], [], []
    for qv in configs:
        set_q(qv)
        q_act = np.array([float(robot.data.joint_pos.torch[0, jid[n]]) for n in JOINTS])
        Tg_usd, Tj_usd = body_T(gi), body_T(ji)
        Tg_urdf, Tj_urdf = urdf_T(q_act)
        X_g.append(T_inv(Tg_urdf) @ Tg_usd)  # URDF gripper_link -> USD gripper (should be constant)
        X_j.append(T_inv(Tj_urdf) @ Tj_usd)  # URDF jaw link -> USD jaw body
        X_jg.append(T_inv(T_inv(Tg_usd) @ Tj_usd) @ (T_inv(Tg_urdf) @ Tj_urdf))  # jaw rel. gripper, URDF vs USD
    Xg = mean_T(X_g)
    spread_g = [T_err(T_inv(Xg) @ X) for X in X_g]
    errs_j = [T_err(X) for X in X_j]
    errs_jg = [T_err(X) for X in X_jg]
    print("[frames] T(URDF gripper_link -> USD gripper), mean over", len(configs), "configs:")
    print("         R =", np.round(Xg[:3, :3], 5).tolist(), " t [mm] =", np.round(Xg[:3, 3] * 1000, 3).tolist())
    print(f"[frames]   |t| {T_err(Xg)[0]:.3f} mm, angle {T_err(Xg)[1]:.4f} deg; spread over configs max {max(s[0] for s in spread_g):.3f} mm "
          f"/ {max(s[1] for s in spread_g):.4f} deg")
    print(f"[frames] jaw link vs URDF FK: max {max(e[0] for e in errs_j):.3f} mm / {max(e[1] for e in errs_j):.4f} deg; "
          f"jaw relative to gripper (URDF vs USD): max {max(e[0] for e in errs_jg):.3f} mm / {max(e[1] for e in errs_jg):.4f} deg")

    # USD meshes of the gripper / jaw bodies vs the URDF STLs (in each body's own frame)
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    base = "/World/envs/env_0/Robot"
    set_q(np.r_[[HOME_Q[j] for j in JOINTS[:5]], 0.0])

    def prim_points(root_path, sub="visuals"):
        root = stage.GetPrimAtPath(root_path)
        w2r = np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse())
        out = {}
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh) and f"/{sub}" in str(prim.GetPath()):
                p = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
                m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())) @ w2r
                out[str(prim.GetPath())[len(root_path) + 1:]] = (np.c_[p, np.ones(len(p))] @ m)[:, :3]
        return out

    def nn_dist(A, B, chunk=4096):
        """For each point of A, distance to the nearest point of B (GPU)."""
        Bt = torch.tensor(B, dtype=torch.float32, device=dev)
        out = []
        for i in range(0, len(A), chunk):
            At = torch.tensor(A[i:i + chunk], dtype=torch.float32, device=dev)
            out.append(torch.cdist(At, Bt).min(1).values.cpu().numpy())
        return np.concatenate(out)

    mesh_check = {}
    usd_g = prim_points(f"{base}/gripper")
    usd_j = prim_points(f"{base}/moving_jaw_so101_v1")
    allg = np.concatenate(list(usd_g.values())) if usd_g else np.zeros((0, 3))
    allj = np.concatenate(list(usd_j.values())) if usd_j else np.zeros((0, 3))
    print("[mesh] USD visual meshes under gripper (gripper frame bbox, mm):")
    for k, P in usd_g.items():
        print(f"         {k:70s} n={len(P):6d} min {np.round(P.min(0) * 1000, 1).tolist()} max {np.round(P.max(0) * 1000, 1).tolist()}")
    for k, P in usd_j.items():
        print(f"       jaw {k:66s} n={len(P):6d} min {np.round(P.min(0) * 1000, 1).tolist()} max {np.round(P.max(0) * 1000, 1).tolist()}")
    TJ = model.jaw_transform(0.0)
    for link, parts in model.parts.items():
        for V, F, L, name in parts:
            if link == wm.JAW_LINK:  # URDF jaw mesh in the jaw link frame <-> USD jaw points in the jaw body frame
                P_urdf, P_usd = V, allj
            else:
                P_urdf, P_usd = V, allg
            sub = P_urdf[np.random.default_rng(0).choice(len(P_urdf), min(3000, len(P_urdf)), replace=False)]
            d_fwd = nn_dist(sub, P_usd) * 1000
            mesh_check[name] = dict(urdf_to_usd_nn_mm_mean=float(d_fwd.mean()), urdf_to_usd_nn_mm_p99=float(np.percentile(d_fwd, 99)),
                                    urdf_to_usd_nn_mm_max=float(d_fwd.max()))
            print(f"[mesh] URDF {name:32s} -> nearest USD visual point: mean {d_fwd.mean():.3f} mm, p99 {np.percentile(d_fwd, 99):.3f} mm, "
                  f"max {d_fwd.max():.3f} mm")
    # camera prims that come with NVIDIA's USD under the gripper
    usd_cams = {}
    g_prim = stage.GetPrimAtPath(f"{base}/gripper")
    w2g = np.array(UsdGeom.Xformable(g_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse())
    print("[usd] gripper children:", [c.GetName() for c in g_prim.GetChildren()])
    mp = stage.GetPrimAtPath(f"{base}/{NV_MOUNT}")
    mount_vis = str(UsdGeom.Imageable(mp).ComputeVisibility()) if mp.IsValid() else "missing"
    from pxr import UsdPhysics

    mc = stage.GetPrimAtPath(f"{base}/{NV_MOUNT_COL}")
    mount_cols = {}
    if mc.IsValid():
        for p in Usd.PrimRange(mc, Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                mount_cols[str(p.GetPath())[len(base) + 1:]] = bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get())
    mount_state = dict(visual_path=NV_MOUNT, visual_visibility_after_startup=mount_vis, collider_path=NV_MOUNT_COL,
                       collision_enabled_after_startup=mount_cols, removal_event=dict(MOUNT_LOG) or None)
    print(f"[usd] {NV_MOUNT} visibility: {mount_vis}; colliders under {NV_MOUNT_COL} (collisionEnabled): {mount_cols}")
    for prim in Usd.PrimRange(g_prim, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Camera) or "camera" in prim.GetName().lower():
            m = (np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())) @ w2g).T  # column conv.
            usd_cams[str(prim.GetPath())[len(base) + 1:]] = dict(type=prim.GetTypeName(), pos=m[:3, 3].tolist(),
                                                                   quat_xyzw=Rotation.from_matrix(m[:3, :3]).as_quat().tolist())
            print(f"[usd] {str(prim.GetPath())[len(base) + 1:]:50s} {prim.GetTypeName():8s} pos {np.round(m[:3, 3], 5).tolist()} "
                  f"quat(xyzw) {np.round(Rotation.from_matrix(m[:3, :3]).as_quat(), 5).tolist()}")

    # ------------------------------------------------------------------ 2. render at the desk-viewing pose
    real = {}
    for tag in ("K", "Ksim"):
        real[tag] = dict(img=np.asarray(Image.open(SCRATCH / f"real_median_und_{tag}.png").convert("RGB")),
                         mask=np.asarray(Image.open(SCRATCH / f"real_mask_{tag}.png")) > 127)
    K_SIM = wm.k_sim(K_EXACT, (IMG_W, IMG_H))

    def settle(qvec, steps):
        """Reach the joint target through the PD drive (gravity off -> exact), as the env normally moves."""
        action = torch.tensor([qvec], dtype=torch.float32, device=dev)
        for _ in range(steps):
            env.step(action)
        q_act = np.array([float(robot.data.joint_pos.torch[0, jid[n]]) for n in JOINTS])
        print("[view] joint angles after settling:", np.round(q_act, 4).tolist(), " target:", np.round(qvec, 4).tolist(),
              f" max err {np.degrees(np.abs(q_act - qvec).max()):.3f} deg")
        return q_act

    def evaluate(name, cam_name, pos, quat, q_act):
        Tg = body_T(gi)
        cam = u.scene[cam_name]
        rgb = cam.data.output["rgb"].torch[0, ..., :3].to(torch.uint8).cpu().numpy()
        depth = cam.data.output["distance_to_image_plane"].torch[0, ..., 0].cpu().numpy()
        Kc = cam.data.intrinsic_matrices.torch[0].cpu().numpy().astype(float)
        cpos = cam.data.pos_w.torch[0].cpu().numpy() - origin
        cquat_gl = cam.data.quat_w_opengl.torch[0].cpu().numpy()
        T_cam_w = T_from(cpos, cquat_gl)
        cam_err = T_err(T_inv(Tg @ T_from(pos, quat)) @ T_cam_w)
        T_cam_g = T_inv(Tg) @ T_cam_w  # camera (OpenGL) in the USD gripper frame, as Isaac reports it
        finite = np.isfinite(depth) & (depth > 0)
        jaw = finite & (depth < args.depth_max)
        dvals = depth[finite]
        # numpy re-render of the URDF meshes from the camera pose Isaac reports at the actual jaw angle, with Isaac's
        # reported intrinsics and with the other candidate K: which projection did the renderer really use?
        size = (rgb.shape[1], rgb.shape[0])
        R_cv, t_cv = wm.env_to_cv(T_cam_g[:3, 3], Rotation.from_matrix(T_cam_g[:3, :3]).as_quat())
        V, F, L = model.mesh(q_act[5])
        reports_exact = bool(np.allclose(Kc, K_EXACT, atol=1e-2))
        K_alt = K_SIM if reports_exact else K_EXACT
        lab_np = wm.render(V, F, L, R_cv, t_cv, Kc, size)
        lab_alt = wm.render(V, F, L, R_cv, t_cv, K_alt, size)
        np_vs_isaac = wm.mask_metrics(lab_np > 0, jaw)
        np_alt = wm.mask_metrics(lab_alt > 0, jaw)
        res = dict(camera_prim=cam_name, camera_world_vs_gripper_x_offset_err_mm=cam_err[0],
                   camera_world_vs_gripper_x_offset_err_deg=cam_err[1], isaac_intrinsics=Kc.tolist(),
                   isaac_reports_exact_K=reports_exact, jaw_angle_rad_actual=float(q_act[5]), jaw_pixels=int(jaw.sum()),
                   depth_jaw_range_m=[float(depth[jaw].min()), float(depth[jaw].max())] if jaw.any() else None,
                   depth_scene_min_m=float(dvals[dvals >= args.depth_max].min()) if (dvals >= args.depth_max).any() else None,
                   numpy_urdf_render_vs_isaac=np_vs_isaac,
                   numpy_urdf_render_with_other_K_vs_isaac=dict(K=K_alt.tolist(), **np_alt),
                   renderer_uses_reported_K=bool(np_vs_isaac["iou"] > np_alt["iou"]))
        if reports_exact:
            res["renderer_uses_exact_K"] = res["renderer_uses_reported_K"]
        for tag in ("K", "Ksim"):
            m = wm.mask_metrics(jaw, real[tag]["mask"])
            m["per_jaw_iou"] = wm.per_jaw_iou(jaw, real[tag]["mask"])
            res[f"vs_real_undistorted_to_{tag}"] = m
        png(SCRATCH / f"isaac_{name}_rgb.png", rgb)
        png(SCRATCH / f"isaac_{name}_jawmask.png", (jaw * 255).astype(np.uint8))
        np.save(SCRATCH / f"isaac_{name}_depth.npy", depth.astype(np.float32))
        print(f"[cam:{name}] camera world pose vs gripper x offset: {cam_err[0]:.4f} mm / {cam_err[1]:.4f} deg; K = "
              f"{np.round(Kc, 2).tolist()}; jaw {q_act[5]:.4f} rad")
        print(f"[cam:{name}] jaw pixels {int(jaw.sum())} (depth {res['depth_jaw_range_m']}), nearest scene depth {res['depth_scene_min_m']}")
        print(f"[cam:{name}] numpy URDF render vs Isaac jaw mask: reported K IoU {np_vs_isaac['iou']:.4f} "
              f"({np_vs_isaac.get('boundary_mean_px', float('nan')):.2f} px), other K IoU {np_alt['iou']:.4f} "
              f"({np_alt.get('boundary_mean_px', float('nan')):.2f} px) -> renderer uses reported K: {res['renderer_uses_reported_K']}")
        for tag in ("K", "Ksim"):
            m = res[f"vs_real_undistorted_to_{tag}"]
            print(f"[cam:{name}] vs real (undistorted to {tag:4s}): IoU {m['iou']:.4f}  boundary mean {m.get('boundary_mean_px', float('nan')):.2f} px "
                  f"p90 {m.get('boundary_p90_px', float('nan')):.2f} px  per-jaw {m['per_jaw_iou']}")
        return res, dict(rgb=rgb, jaw=jaw, np=lab_np > 0)

    q_view = np.r_[[HOME_Q[j] for j in JOINTS[:5]], Q_JAW]
    env.reset(seed=0)
    q_act = settle(q_view, args.settle_steps)
    results, frames = {}, {}
    for name, (cam_name, pos, quat) in CAMS.items():
        results[name], frames[name] = evaluate(name, cam_name, pos, quat, q_act)
    # fair 'before': the NVIDIA mount with the jaw at the angle that suits that mount best (CPU fit, pose held)
    q_view_nv = q_view.copy()
    q_view_nv[5] = Q_NV
    q_act_nv = settle(q_view_nv, 60)
    results["nvidia_at_its_best_q"], frames["nvidia_at_its_best_q"] = evaluate(
        "nvidia_at_its_best_q", "wrist_cam_nvidia", NV_POS, NV_QUAT, q_act_nv)

    # ------------------------------------------------------------------ 3. images
    YEL, RED, GRN, CYA = (255, 220, 0), (255, 40, 40), (40, 255, 40), (0, 200, 255)

    def iou_of(name, tag):
        return results[name][f"vs_real_undistorted_to_{tag}"]["iou"]

    def tile(tag, renders, text):
        img = draw(brighten(real[tag]["img"]), real[tag]["mask"], YEL, 1)
        for m, col in renders:
            img = draw(img, m, col, 2)
        return put_label(img, text)

    has_exact = "fitted_exactK" in frames
    row1 = [tile("Ksim", [(frames["nvidia_at_its_best_q"]["jaw"], RED), (frames["fitted"]["jaw"], GRN)],
                 f"default spawn (renders K_sim) vs real undistorted to K_sim (yellow)\n"
                 f"red: NVIDIA mount, jaw {Q_NV:.2f} (its best) IoU {iou_of('nvidia_at_its_best_q', 'Ksim'):.3f}\n"
                 f"green: fitted mount, jaw {Q_JAW:.2f} IoU {iou_of('fitted', 'Ksim'):.3f}")]
    if has_exact:
        row1.append(tile("K", [(frames["fitted_exactK"]["jaw"], GRN)],
                         f"exact-K spawn (OpenCvPinholeDistortionCfg) vs real undistorted to K (yellow)\n"
                         f"green: fitted mount IoU {iou_of('fitted_exactK', 'K'):.3f}"))
    row1.append(tile("K", [(frames["fitted"]["jaw"], GRN)],
                     f"MISMATCH: default spawn (K_sim) vs real undistorted to K (yellow)\n"
                     f"green: fitted mount IoU {iou_of('fitted', 'K'):.3f}"))
    row2 = [put_label(draw(frames["nvidia"]["rgb"], real["Ksim"]["mask"], YEL, 1),
                      f"Isaac, NVIDIA mount, jaw {Q_JAW:.2f} (as in the real frames)\nyellow: real jaws (K_sim)"),
            put_label(draw(frames["fitted"]["rgb"], real["Ksim"]["mask"], YEL, 1),
                      "Isaac, fitted mount, default spawn\nyellow: real jaws (K_sim)")]
    if has_exact:
        row2.append(put_label(draw(frames["fitted_exactK"]["rgb"], real["K"]["mask"], YEL, 1),
                              "Isaac, fitted mount, exact-K spawn\nyellow: real jaws (K)"))
    while len(row2) < len(row1):
        row2.append(np.zeros_like(row2[0]))
    sheet = np.vstack([np.hstack(row1), np.hstack(row2)])
    png(SCRATCH / "isaac_overlay_sheet.png", sheet)
    side = np.hstack([put_label(brighten(real["Ksim"]["img"]), "real wrist median (undistorted to K_sim)"),
                      put_label(frames["nvidia"]["rgb"], f"Isaac: NVIDIA mount (current), jaw {Q_JAW:.2f}"),
                      put_label(frames["fitted"]["rgb"], f"Isaac: fitted mount (proposal), jaw {Q_JAW:.2f}")])
    png(SCRATCH / "isaac_side_by_side.png", side)
    xc = np.hstack([put_label(draw(draw(f["rgb"], f["np"], CYA, 1), f["jaw"], GRN, 1),
                              f"{n}: Isaac depth mask (green)\nvs numpy URDF render, Isaac's K (cyan)") for n, f in frames.items()])
    png(SCRATCH / "isaac_vs_numpy.png", xc)
    print("[out] wrote", SCRATCH / "isaac_overlay_sheet.png", SCRATCH / "isaac_side_by_side.png", SCRATCH / "isaac_vs_numpy.png")

    # ------------------------------------------------------------------ 4. results
    rv = Rotation.from_matrix(Xg[:3, :3]).as_rotvec()
    ax = rv / max(np.linalg.norm(rv), 1e-12)
    local_identical = (max(v["urdf_to_usd_nn_mm_p99"] for v in mesh_check.values()) < 0.1
                       and max(e[0] for e in errs_jg) < 0.01 and max(e[1] for e in errs_jg) < 0.01)
    frame_check = dict(
        interpretation=dict(
            link_local_frames_identical=bool(local_identical),
            fk_rotation_axis_in_gripper_frame=ax.tolist(),
            equivalent_wrist_roll_offset_deg=float(np.degrees(np.linalg.norm(rv)) * np.sign(ax[2])),
            note="The URDF STLs placed in gripper_link / jaw link coincide with the USD visual meshes in the USD 'gripper' / "
                 "jaw bodies (nearest-point p99 < 0.1 mm) and the jaw joint is identical, so a pose relative to the gripper "
                 "geometry (the camera offset) needs no conversion: T_USD_URDF = identity. The kinematic (FK) difference is a "
                 "pure rotation about the gripper z = wrist_roll axis, i.e. a wrist_roll zero offset: the URDF wrist_roll "
                 "origin carries rpy pitch 0.0486795 rad (= Rz(-2.789 deg) on the child side) which NVIDIA's USD does not; "
                 "at equal joint angles the sim gripper is rolled +2.79 deg relative to URDF FK (calib/fk.py, silhouette.py)."),
        T_urdf_gripper_link_to_usd_gripper=dict(R=Xg[:3, :3].tolist(), t_m=Xg[:3, 3].tolist(), t_norm_mm=T_err(Xg)[0],
                                                angle_deg=T_err(Xg)[1]),
        spread_over_configs_max_mm=max(s[0] for s in spread_g), spread_over_configs_max_deg=max(s[1] for s in spread_g),
        n_configs=len(configs),
        jaw_link_vs_urdf_fk_max_mm=max(e[0] for e in errs_j), jaw_link_vs_urdf_fk_max_deg=max(e[1] for e in errs_j),
        jaw_relative_to_gripper_urdf_vs_usd_max_mm=max(e[0] for e in errs_jg),
        jaw_relative_to_gripper_urdf_vs_usd_max_deg=max(e[1] for e in errs_jg),
        urdf_stl_vs_usd_visual_mesh=mesh_check, usd_camera_prims_in_gripper_frame=usd_cams,
        note="URDF FK in the desk frame uses calib/fk.py T_WORLD_URDF (base_link -> desk, translation only)")
    summary = {
        "before_nvidia_at_fitted_jaw_vs_real_Ksim": iou_of("nvidia", "Ksim"),
        "before_nvidia_at_its_best_jaw_vs_real_Ksim": iou_of("nvidia_at_its_best_q", "Ksim"),
        "after_fitted_default_spawn_vs_real_Ksim (option a)": iou_of("fitted", "Ksim"),
        "after_fitted_default_spawn_vs_real_K (pose alone, intrinsics mismatch)": iou_of("fitted", "K"),
    }
    if has_exact:
        summary["after_fitted_exactK_spawn_vs_real_K (option b)"] = iou_of("fitted_exactK", "K")
        summary["exactK_spawn_renderer_uses_exact_K"] = results["fitted_exactK"].get("renderer_uses_exact_K")
    evaluated = dict(fitted=dict(pos=FIT_POS, quat=FIT_QUAT, jaw_angle_rad=Q_JAW, spawn="PinholeCameraCfg.from_intrinsic_matrix(WRIST_K) (env_cfg default)"),
                     nvidia=dict(pos=NV_POS, quat=NV_QUAT, jaw_angle_rad=Q_JAW),
                     nvidia_at_its_best_q=dict(pos=NV_POS, quat=NV_QUAT, jaw_angle_rad=Q_NV))
    if has_exact:
        evaluated["fitted_exactK"] = dict(pos=FIT_POS, quat=FIT_QUAT, jaw_angle_rad=Q_JAW, K=K_EXACT.tolist(),
                                          spawn="spawn.distortion = OpenCvPinholeDistortionCfg(fx, fy, cx, cy, image_size=(640, 480), apply_lens_distortion=False)")
    isaac_validation = dict(
        task=TASK, arm_pose_rad=q_view.tolist(), arm_pose_actual_rad=q_act.tolist(),
        arm_pose_nvidia_best_jaw_actual_rad=q_act_nv.tolist(), depth_max_m=args.depth_max,
        nvidia_mount=mount_state, nvidia_mount_removed=not args.keep_nvidia_mount,
        evaluated=evaluated, summary_iou=summary, results=results,
        images=["calib/data/wrist_mount_isaac_overlay.jpg", "calib/data/wrist_mount_isaac_side_by_side.jpg"])
    (SCRATCH / "isaac_results.json").write_text(json.dumps(dict(frame_check=frame_check, isaac_validation=isaac_validation), indent=1))
    print("[out] summary IoU:", json.dumps(summary, indent=1))
    print("[out] wrote", SCRATCH / "isaac_results.json")

    if not args.no_write:
        cur = json.loads(PROPOSAL.read_text())
        if (tuple(cur["WRIST_CAM_POS"]), tuple(cur["WRIST_CAM_QUAT"]), float(cur["jaw_angle_rad"])) != (FIT_POS, FIT_QUAT, Q_JAW):
            print("[out] ERROR: the proposal changed while this ran; NOT writing results for the old pose", FIT_POS, FIT_QUAT,
                  Q_JAW, "into it (they are in", SCRATCH / "isaac_results.json", ")")
        else:
            cal = PROPOSAL.parent
            Image.fromarray(sheet).save(cal / "wrist_mount_isaac_overlay.jpg", quality=92)
            Image.fromarray(side).save(cal / "wrist_mount_isaac_side_by_side.jpg", quality=92)
            cur.pop("isaac_validation_stale", None)
            cur["frame_check"] = frame_check
            cur["isaac_validation"] = isaac_validation
            cur["apply_requirements"] = wm.apply_requirements(cur)
            PROPOSAL.write_text(json.dumps(cur, indent=1))
            print("[out] updated", PROPOSAL)
    env.close()

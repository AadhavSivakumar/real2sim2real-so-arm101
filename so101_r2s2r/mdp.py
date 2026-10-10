"""MDP terms for the SO-101 cube-into-cup task.

Each episode one of the three printed cubes (20/30/40 mm) is "active": it is placed on the desk,
the other two are parked under the desk out of every camera's view. The active index lives on the
env as ``env.r2s_active_cube`` so observations/rewards/terminations agree on it.

Quaternions follow IsaacLab 3.0: (x, y, z, w).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

CUBE_NAMES = ("cube_s", "cube_m", "cube_l")


# ----------------------------------------------------------------------------- helpers
def active_cube_idx(env: ManagerBasedEnv) -> torch.Tensor:
    if not hasattr(env, "r2s_active_cube"):
        env.r2s_active_cube = torch.full((env.num_envs,), env.cfg.default_cube, dtype=torch.long, device=env.device)
    return env.r2s_active_cube


def cube_sizes(env: ManagerBasedEnv) -> torch.Tensor:
    return torch.tensor(env.cfg.cube_sizes, device=env.device)


def active_cube_size(env: ManagerBasedEnv) -> torch.Tensor:
    return cube_sizes(env)[active_cube_idx(env)]


def active_cube_pose_w(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.stack([env.scene[n].data.root_pos_w.torch for n in CUBE_NAMES], dim=1)
    quat = torch.stack([env.scene[n].data.root_quat_w.torch for n in CUBE_NAMES], dim=1)
    rows = torch.arange(env.num_envs, device=env.device)
    idx = active_cube_idx(env)
    return pos[rows, idx], quat[rows, idx]


def _to_base(env: ManagerBasedEnv, pos_w: torch.Tensor, quat_w: torch.Tensor | None = None):
    robot = env.scene["robot"]
    return math_utils.subtract_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, pos_w, quat_w
    )


def tcp_pos_w(env: ManagerBasedEnv) -> torch.Tensor:
    """Grasp point between the jaws (frame "tcp" of the ee_frame transformer)."""
    return env.scene["ee_frame"].data.target_pos_w.torch[:, 0, :]


def gripper_q(env: ManagerBasedEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.joint_pos.torch[:, robot.joint_names.index("gripper")]


def cube_in_cup(env: ManagerBasedEnv) -> torch.Tensor:
    """Cube centre inside the cup's inner volume (below the rim, within the inner radius)."""
    cube_pos, _ = active_cube_pose_w(env)
    cup = env.scene["cup"]
    cup_pos, cup_quat = cup.data.root_pos_w.torch, cup.data.root_quat_w.torch
    rel = math_utils.quat_apply_inverse(cup_quat, cube_pos - cup_pos)  # cube centre in cup frame
    d = env.cfg.cup_dims
    inner_r = d["top_d"] / 2 - d["wall"]
    radial = torch.linalg.vector_norm(rel[:, :2], dim=-1)
    return (radial < inner_r) & (rel[:, 2] > d["floor"]) & (rel[:, 2] < d["height"] - 0.005)


def cup_upright(env: ManagerBasedEnv) -> torch.Tensor:
    up = math_utils.quat_apply(env.scene["cup"].data.root_quat_w.torch,
                               torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, 3))
    return up[:, 2] > 0.8


# ----------------------------------------------------------------------------- events
def _inside_convex(p: torch.Tensor, poly) -> torch.Tensor:
    """Signed distance [m] of the points p[..., 2] inside the convex polygon ``poly`` ((x, y), ...; either winding):
    the smallest distance to an edge line, positive inside."""
    pts = [(float(x), float(y)) for x, y in poly]
    if sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1])) < 0:
        pts = pts[::-1]  # counter-clockwise, so the left normals point inward
    a = torch.tensor(pts, dtype=p.dtype, device=p.device)  # (E, 2) vertices
    d = a.roll(-1, 0) - a
    nrm = torch.stack([-d[:, 1], d[:, 0]], -1) / torch.linalg.vector_norm(d, dim=-1, keepdim=True)
    return ((p[..., None, :] - a) * nrm).sum(-1).amin(-1)


def reset_cube_and_cup(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cube_rphi: dict[str, tuple[float, float]],
    cup_rphi: dict[str, tuple[float, float]],
    min_separation: float,
    cube_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
    park_z: float = -0.70,
    cup_keepout: tuple | None = None,
    desk_poly: tuple | None = None,
    desk_margin: float = 0.015,
    num_candidates: int = 64,
):
    """Pick the active cube, place it and the cup (polar ranges around the base) with a minimum
    separation, park the others. ``cup_keepout`` ((x, y), r): no cup centre within r of that point (the hand
    of the robot's start pose). ``desk_poly`` ((x, y), ...): the real desk top's footprint in the env frame
    (convex, either winding); the cube centre must lie at least half its size + ``desk_margin`` inside it and the cup
    centre at least its rim radius (``cfg.cup_dims["top_d"] / 2``) + ``desk_margin``, so nothing rests on or hangs
    over an edge the real desk does not have."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, dev = len(env_ids), env.device
    idx = active_cube_idx(env)
    idx[env_ids] = torch.multinomial(torch.tensor(cube_probs, device=dev), n, replacement=True)
    sizes = cube_sizes(env)[idx[env_ids]]

    # vectorised rejection sampling: K candidate pairs per env, keep the first that meets every constraint
    # (with the desk ~35% of the pairs do -> P(no valid pair) ~ 0.65^64 ~ 1e-12)
    K = num_candidates
    def _uniform(r, shape):
        return torch.empty(shape, device=dev).uniform_(*r)
    def _polar(rng):
        r, phi = _uniform(rng["r"], (n, K)), _uniform(rng["phi"], (n, K))
        return torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim=-1)
    cube_c, cup_c = _polar(cube_rphi), _polar(cup_rphi)
    ok = torch.linalg.vector_norm(cube_c - cup_c, dim=-1) > min_separation
    if cup_keepout is not None:
        ok &= torch.linalg.vector_norm(cup_c - torch.tensor(cup_keepout[0], device=dev), dim=-1) > cup_keepout[1]
    on_desk = torch.ones_like(ok)
    if desk_poly is not None:
        cup_r = env.cfg.cup_dims["top_d"] / 2
        on_desk = (_inside_convex(cube_c, desk_poly) >= sizes[:, None] / 2 + desk_margin) \
            & (_inside_convex(cup_c, desk_poly) >= cup_r + desk_margin)
        ok &= on_desk
    # first valid pair; fallback (no valid pair): the first pair that is at least on the desk, else the first pair
    first = (2 * ok.long() + on_desk.long()).argmax(1)
    n_fallback = int((~ok.any(1)).sum())
    if not hasattr(env, "r2s_spawn_stats"):
        env.r2s_spawn_stats = {"resets": 0, "fallbacks": 0}
    stats = env.r2s_spawn_stats
    stats["resets"] += n
    stats["fallbacks"] += n_fallback
    if n_fallback:
        print(f"[reset_cube_and_cup] no candidate pair met every constraint in {n_fallback}/{n} envs "
              f"(total {stats['fallbacks']}/{stats['resets']} resets); widen the ranges or raise num_candidates")
    rows = torch.arange(n, device=dev)
    cube_p, cup_p = cube_c[rows, first], cup_c[rows, first]

    origins = env.scene.env_origins[env_ids]
    zeros = torch.zeros(n, device=dev)
    yaw = _uniform((-3.1416, 3.1416), (n,))
    zero_vel = torch.zeros(n, 6, device=dev)

    cup = env.scene["cup"]
    cup_pose = torch.cat([origins + torch.stack([cup_p[:, 0], cup_p[:, 1], zeros + 0.001], -1),
                          math_utils.quat_from_euler_xyz(zeros, zeros, _uniform((-3.1416, 3.1416), (n,)))], -1)
    cup.write_root_pose_to_sim_index(root_pose=cup_pose, env_ids=env_ids)
    cup.write_root_velocity_to_sim_index(root_velocity=zero_vel, env_ids=env_ids)

    for k, name in enumerate(CUBE_NAMES):
        cube = env.scene[name]
        active = idx[env_ids] == k
        on_desk = torch.stack([cube_p[:, 0], cube_p[:, 1], sizes / 2 + 0.001], -1)
        parked = torch.stack([zeros + 0.25 + 0.08 * k, zeros, zeros + park_z], -1)
        pos = origins + torch.where(active[:, None], on_desk, parked)
        quat = math_utils.quat_from_euler_xyz(zeros, zeros, torch.where(active, yaw, zeros))
        cube.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], -1), env_ids=env_ids)
        cube.write_root_velocity_to_sim_index(root_velocity=zero_vel, env_ids=env_ids)


# ----------------------------------------------------------------------------- observations
def cube_pos_b(env: ManagerBasedEnv) -> torch.Tensor:
    pos_w, _ = active_cube_pose_w(env)
    return _to_base(env, pos_w)[0]


def cube_quat_w(env: ManagerBasedEnv) -> torch.Tensor:
    return active_cube_pose_w(env)[1]


def cube_size_obs(env: ManagerBasedEnv) -> torch.Tensor:
    return active_cube_size(env).unsqueeze(-1)


def cup_pos_b(env: ManagerBasedEnv) -> torch.Tensor:
    return _to_base(env, env.scene["cup"].data.root_pos_w.torch)[0]


def tcp_pos_b(env: ManagerBasedEnv) -> torch.Tensor:
    return _to_base(env, tcp_pos_w(env))[0]


# ----------------------------------------------------------------------------- rewards
def _lifted(env: ManagerBasedRLEnv, height: float) -> torch.Tensor:
    pos, _ = active_cube_pose_w(env)
    bottom = pos[:, 2] - env.scene.env_origins[:, 2] - active_cube_size(env) / 2
    return bottom > height


def reach_cube(env: ManagerBasedRLEnv, std: float) -> torch.Tensor:
    pos, _ = active_cube_pose_w(env)
    return 1.0 - torch.tanh(torch.linalg.vector_norm(tcp_pos_w(env) - pos, dim=-1) / std)


def cube_lifted(env: ManagerBasedRLEnv, height: float) -> torch.Tensor:
    return _lifted(env, height).float()


def cube_over_cup(env: ManagerBasedRLEnv, std: float, lift_height: float) -> torch.Tensor:
    """Dense: pull the lifted cube horizontally toward the cup axis."""
    pos, _ = active_cube_pose_w(env)
    d = torch.linalg.vector_norm(pos[:, :2] - env.scene["cup"].data.root_pos_w.torch[:, :2], dim=-1)
    return _lifted(env, lift_height).float() * (1.0 - torch.tanh(d / std))


def cube_in_cup_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return cube_in_cup(env).float()


def placed_and_released(env: ManagerBasedRLEnv, open_q: float) -> torch.Tensor:
    return (cube_in_cup(env) & (gripper_q(env) > open_q)).float()


# ----------------------------------------------------------------------------- terminations
def task_success(env: ManagerBasedRLEnv, open_q: float) -> torch.Tensor:
    cube_vel = torch.stack([env.scene[n].data.root_lin_vel_w.torch for n in CUBE_NAMES], 1)
    still = torch.linalg.vector_norm(cube_vel[torch.arange(env.num_envs, device=env.device), active_cube_idx(env)], dim=-1) < 0.05
    return cube_in_cup(env) & (gripper_q(env) > open_q) & still & cup_upright(env)


def cube_dropped(env: ManagerBasedRLEnv, min_z: float = -0.05) -> torch.Tensor:
    pos, _ = active_cube_pose_w(env)
    return (pos[:, 2] - env.scene.env_origins[:, 2]) < min_z


def cup_tipped(env: ManagerBasedRLEnv) -> torch.Tensor:
    return ~cup_upright(env)


# ----------------------------------------------------------------------------- visual randomization
def recolor_printed_parts(env: ManagerBasedEnv, env_ids: torch.Tensor | None, base_color=(0.93, 0.93, 0.91),
                          jitter: float = 0.04, material_suffix: str = "material_a_d_printed", jaw_emissive: float = 0.0,
                          jaw_links: tuple = ("gripper", "moving_jaw_so101_v1")):
    """Rebind NVIDIA's yellow printed-plastic material to an off-white one (the real arm is white);
    the black STS3215 servo material is left alone. Run once at prestartup.
    ``jaw_emissive`` > 0: the printed parts of ``jaw_links`` (gripper body + fixed jaw, moving jaw) get their own material
    /World/Looks/r2s_jaw_<i> that also emits that fraction of its colour (thin white PLA is translucent: the real jaws
    glow white in the wrist view even where the gripper shadows them from the lamp; randomize_jaw_glow varies it)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.spawners.materials.visual_materials import spawn_preview_surface
    from pxr import Sdf, Usd, UsdShade

    stage = sim_utils.get_current_stage()
    gen = torch.Generator().manual_seed(0)
    for i in range(env.num_envs):
        color = (torch.tensor(base_color) + (torch.rand(3, generator=gen) * 2 - 1) * jitter).clamp(0, 1)
        mat_path = f"/World/Looks/r2s_printed_{i}"
        spawn_preview_surface(mat_path, sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color.tolist()), roughness=0.55))
        jaw_path = f"/World/Looks/r2s_jaw_{i}"
        if jaw_emissive > 0:
            spawn_preview_surface(jaw_path, sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(color.tolist()), roughness=0.55, emissive_color=tuple((jaw_emissive * color).tolist())))
        robot = f"/World/envs/env_{i}/Robot"
        n_jaw = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot)):
            rel = UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel()
            targets = rel.GetTargets() if rel else []
            if targets and str(targets[0]).endswith(material_suffix):
                rel_path = str(prim.GetPath())[len(robot) + 1:]
                on_jaw = jaw_emissive > 0 and rel_path.split("/")[0] in jaw_links
                rel.SetTargets([Sdf.Path(jaw_path if on_jaw else mat_path)])
                n_jaw += on_jaw
        if jaw_emissive > 0 and i == 0:
            print(f"[recolor_printed_parts] {n_jaw} printed jaw prim(s) of {jaw_links} -> {jaw_path} (emissive {jaw_emissive})")


def _editable_prim(stage, path):
    """The prim at ``path``, made editable by de-instancing its enclosing instanceable prim(s); None if missing."""
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


def remove_camera_mount(env: ManagerBasedEnv, env_ids: torch.Tensor | None,
                        visual: str = "gripper/visuals/camera_mount", collider: str = "gripper/collisions/camera_mount"):
    """Remove NVIDIA's wrist-camera mount (USD 'sensors' variant) from the robot: the user's rig has a different
    mount, and NVIDIA's model sits right in front of the fitted camera (calib/data/wrist_mount_proposal.json)
    and next to the jaws. Hides the visual and disables every collider under it. Run once at prestartup."""
    import isaaclab.sim as sim_utils
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = sim_utils.get_current_stage()
    for i in range(env.num_envs):
        root = f"/World/envs/env_{i}/Robot"
        vis = _editable_prim(stage, f"{root}/{visual}")
        if vis is not None:
            UsdGeom.Imageable(vis).MakeInvisible()
        col = stage.GetPrimAtPath(f"{root}/{collider}")
        if not col.IsValid():
            continue
        paths = [p.GetPath() for p in Usd.PrimRange(col, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
        for path in paths:
            UsdPhysics.CollisionAPI(_editable_prim(stage, path)).CreateCollisionEnabledAttr().Set(False)


def set_joint_limits(env: ManagerBasedEnv, env_ids: torch.Tensor | None, limits_deg: dict):
    """Overwrite revolute joint limits in the USD before PhysX parses it ({joint: (lower, upper)} in degrees,
    None keeps that side). Run once at prestartup: the articulation validates its default pose against these."""
    import isaaclab.sim as sim_utils
    from pxr import UsdPhysics

    stage = sim_utils.get_current_stage()
    for i in range(env.num_envs):
        for joint, (lo, hi) in limits_deg.items():
            prim = _editable_prim(stage, f"/World/envs/env_{i}/Robot/joints/{joint}")
            if prim is None:
                raise RuntimeError(f"set_joint_limits: no joint prim {joint} under env_{i}/Robot/joints")
            rev = UsdPhysics.RevoluteJoint(prim)
            if lo is not None:
                rev.GetLowerLimitAttr().Set(float(lo))
            if hi is not None:
                rev.GetUpperLimitAttr().Set(float(hi))


def randomize_light(env: ManagerBasedEnv, env_ids: torch.Tensor | None, intensity_range=(800.0, 2500.0),
                    warmth_range=(0.75, 1.0), light_path: str = "/World/Light"):
    """Dome light intensity and colour temperature (warm-white .. neutral)."""
    import isaaclab.sim as sim_utils
    from pxr import Gf

    prim = sim_utils.get_current_stage().GetPrimAtPath(light_path)
    w = float(torch.empty(1).uniform_(*warmth_range))
    prim.GetAttribute("inputs:intensity").Set(float(torch.empty(1).uniform_(*intensity_range)))
    prim.GetAttribute("inputs:color").Set(Gf.Vec3f(1.0, 0.82 + 0.18 * w, 0.62 + 0.38 * w))


def jitter_camera(env: ManagerBasedEnv, env_ids: torch.Tensor | None, sensor_name: str,
                  pos_std: float = 0.01, rot_std_deg: float = 1.0):
    """Perturb a fixed camera around its calibrated pose (robustness to calibration error)."""
    cam = env.scene[sensor_name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    if not hasattr(cam, "_r2s_nominal"):
        cam._r2s_nominal = (cam.data.pos_w.torch.clone(), cam.data.quat_w_ros.torch.clone())
    pos0, quat0 = cam._r2s_nominal
    n = len(env_ids)
    dpos = torch.randn(n, 3, device=env.device) * pos_std
    rpy = torch.randn(n, 3, device=env.device) * (rot_std_deg * 3.14159265 / 180.0)
    dq = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])
    cam.set_world_poses(pos0[env_ids] + dpos, math_utils.quat_mul(quat0[env_ids], dq), env_ids=env_ids, convention="ros")


def jitter_mounted_camera(env: ManagerBasedEnv, env_ids: torch.Tensor | None, prim_rel: str = "Robot/gripper/wrist_cam",
                          pos_std: float = 0.004, rot_std_deg: float = 2.0):
    """Perturb a camera rigidly mounted on a link (its local USD offset) around the calibrated mount pose: the wrist
    camera's mount fit and the arm's joint zeros leave a few mm / degrees of error (robustness)."""
    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom

    stage = sim_utils.get_current_stage()
    nominal = env.__dict__.setdefault("_r2s_mount_nominal", {})
    ids = range(env.num_envs) if env_ids is None else env_ids.tolist()
    for i in ids:
        path = f"/World/envs/env_{i}/{prim_rel}"
        ops = {op.GetOpName(): op for op in UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()}
        t_op, o_op = ops["xformOp:translate"], ops["xformOp:orient"]
        if path not in nominal:
            nominal[path] = (t_op.Get(), o_op.Get())
        t0, q0 = nominal[path]
        d = (torch.randn(3) * pos_std).tolist()
        t_op.Set(type(t0)(t0[0] + d[0], t0[1] + d[1], t0[2] + d[2]))
        r = (torch.randn(3) * rot_std_deg).tolist()
        dq = (Gf.Rotation(Gf.Vec3d(1, 0, 0), r[0]) * Gf.Rotation(Gf.Vec3d(0, 1, 0), r[1])
              * Gf.Rotation(Gf.Vec3d(0, 0, 1), r[2])).GetQuat()
        q = Gf.Quatd(q0.GetReal(), Gf.Vec3d(q0.GetImaginary())) * dq
        o_op.Set(type(q0)(q.GetReal(), type(q0.GetImaginary())(*q.GetImaginary())))


# ----------------------------------------------------------------------------- staged shaping (RL v2)
def _cube_bottom_above_desk(env: ManagerBasedRLEnv) -> torch.Tensor:
    pos, _ = active_cube_pose_w(env)
    return pos[:, 2] - env.scene.env_origins[:, 2] - active_cube_size(env) / 2


def _cube_cup_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    pos, _ = active_cube_pose_w(env)
    return torch.linalg.vector_norm(pos[:, :2] - env.scene["cup"].data.root_pos_w.torch[:, :2], dim=-1)


def raise_above_rim(env: ManagerBasedRLEnv, margin: float, std: float, lift_height: float) -> torch.Tensor:
    """Once lifted, pull the cube's bottom up to rim + margin (the cup is taller than a naive lift)."""
    gap = (env.cfg.cup_dims["height"] + margin - _cube_bottom_above_desk(env)).clamp(min=0.0)
    return _lifted(env, lift_height).float() * (1.0 - torch.tanh(gap / std))


def over_cup_clear(env: ManagerBasedRLEnv, std: float, margin: float) -> torch.Tensor:
    """Move toward the cup axis, but only while the cube clears the rim."""
    clear = _cube_bottom_above_desk(env) > env.cfg.cup_dims["height"] + margin
    return clear.float() * (1.0 - torch.tanh(_cube_cup_xy(env) / std))


def open_over_cup(env: ManagerBasedRLEnv, margin: float) -> torch.Tensor:
    """Reward opening the jaw when the cube is above the opening (it then drops in)."""
    inner_r = env.cfg.cup_dims["top_d"] / 2 - env.cfg.cup_dims["wall"]
    over = (_cube_cup_xy(env) < inner_r - 0.01) & (_cube_bottom_above_desk(env) > env.cfg.cup_dims["height"] - 0.01)
    # pay only for opening WIDER than the cube (jaw gap ~17 mm + 66 mm/rad, measured): a partly open
    # jaw that still holds the cube must not earn this
    q_release = (active_cube_size(env) - 0.017) / 0.066 + 0.15
    return over.float() * ((gripper_q(env) - q_release) / 0.3).clamp(0.0, 1.0)


def physics_blowup(env: ManagerBasedRLEnv, max_lin_vel: float = 20.0, max_joint_vel: float = 100.0) -> torch.Tensor:
    """Reset envs whose physics exploded (NaN/inf or absurd velocities) before they poison the batch."""
    robot = env.scene["robot"]
    bad = ~torch.isfinite(robot.data.joint_pos.torch).all(-1) | (robot.data.joint_vel.torch.abs() > max_joint_vel).any(-1)
    for name in (*CUBE_NAMES, "cup"):
        d = env.scene[name].data
        p, v = d.root_pos_w.torch, d.root_lin_vel_w.torch
        bad |= ~torch.isfinite(p).all(-1) | ~torch.isfinite(v).all(-1) | (torch.linalg.vector_norm(v, dim=-1) > max_lin_vel)
    return bad


# ----------------------------------------------------------------------------- anti-exploit shaping (RL v3)
def cube_grasped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube held between the jaws: its centre inside the jaw slot (gripper link frame) and the jaw
    stopped by it (angle near the cube-contact angle -- a jaw shut on nothing is NOT a grasp)."""
    pos, _ = active_cube_pose_w(env)
    size = active_cube_size(env)
    near = torch.linalg.vector_norm(tcp_pos_w(env) - pos, dim=-1) < 0.025
    robot = env.scene["robot"]
    gi = robot.body_names.index("gripper")
    c = math_utils.quat_apply_inverse(robot.data.body_link_quat_w.torch[:, gi],
                                      pos - robot.data.body_link_pos_w.torch[:, gi])
    fx, tz = env.cfg.jaw_fixed_x, env.cfg.jaw_tip_z
    # across the jaw width / past the fingertips the window grows with the cube: the expert holds the 40 mm
    # cube up to 13 mm off-centre; the jaws must still overlap >= 4 mm of it
    between = (((c[:, 0] - (fx + size / 2)).abs() < 0.010) & (c[:, 1].abs() < size / 2 + 0.005)
               & (c[:, 2] > tz - size / 2 + 0.004) & (c[:, 2] < tz + 0.040))
    q_contact = (size - 0.017) / 0.066  # jaw angle at which the jaws touch the cube (gap 17 mm + 66 mm/rad)
    q = gripper_q(env)
    return near & between & (q > q_contact - 0.06) & (q < q_contact + 0.15)


def grasped_lift(env: ManagerBasedRLEnv, height: float) -> torch.Tensor:
    return (cube_grasped(env) & _lifted(env, height)).float()


def grasped_raise_above_rim(env: ManagerBasedRLEnv, margin: float, std: float,
                            lift_height: float = 0.005) -> torch.Tensor:
    # paid only once the held cube is off the desk (with std 0.10 a cube still ON the desk earned 0.49)
    gap = (env.cfg.cup_dims["height"] + margin - _cube_bottom_above_desk(env)).clamp(min=0.0)
    return (cube_grasped(env) & _lifted(env, lift_height)).float() * (1.0 - torch.tanh(gap / std))


def grasped_over_cup(env: ManagerBasedRLEnv, std: float, margin: float) -> torch.Tensor:
    clear = _cube_bottom_above_desk(env) > env.cfg.cup_dims["height"] + margin
    return (cube_grasped(env) & clear).float() * (1.0 - torch.tanh(_cube_cup_xy(env) / std))


def cube_thrown(env: ManagerBasedRLEnv, max_speed: float = 2.0) -> torch.Tensor:
    v = torch.stack([env.scene[n].data.root_lin_vel_w.torch for n in CUBE_NAMES], 1)
    v = v[torch.arange(env.num_envs, device=env.device), active_cube_idx(env)]
    return torch.linalg.vector_norm(v, dim=-1) > max_speed


def cube_out_of_workspace(env: ManagerBasedRLEnv, max_r: float = 0.5) -> torch.Tensor:
    pos, _ = active_cube_pose_w(env)
    rel = pos[:, :2] - env.scene["robot"].data.root_pos_w.torch[:, :2]
    return torch.linalg.vector_norm(rel, dim=-1) > max_r


def grasp_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return cube_grasped(env).float()


def arm_drive_saturation(env: ManagerBasedRLEnv, deadband: float = 0.25, action_name: str = "joints") -> torch.Tensor:
    """Squared arm target error beyond a deadband, on the UNCLIPPED target: penalises targets past the
    joint limits and pressing into the desk at the torque cap (both give the policy no gradient otherwise)."""
    term = env.action_manager.get_term(action_name)
    target = term.raw_actions * term._scale + term._offset
    q = env.scene["robot"].data.joint_pos.torch[:, term._joint_ids]
    return ((target - q).abs() - deadband).clamp(min=0.0).square().sum(-1)


# ----------------------------------------------------------------------------- real-look appearance DR (camera env)
def _uniform(rng) -> float:
    return float(torch.empty(1).uniform_(*rng))


def _env_prim(env: ManagerBasedEnv, i: int, asset_name: str) -> str:
    asset = env.scene[asset_name]  # an asset, or the cfg itself for plain AssetBaseCfg entities
    return f"{env.scene.env_prim_paths[i]}/{getattr(asset, 'cfg', asset).prim_path.split('/')[-1]}"


def _preview_shaders(stage, root: str, material: str | None = None) -> list:
    """UsdPreviewSurface shaders under ``root`` (optionally only those of materials whose name contains ``material``)."""
    from pxr import Usd, UsdShade

    out = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if prim.GetTypeName() != "Shader" or UsdShade.Shader(prim).GetIdAttr().Get() != "UsdPreviewSurface":
            continue
        if material is None or material in prim.GetParent().GetName():
            out.append(prim)
    return out


def _tint(rgb, hue: float, sat: float, val: float) -> tuple[float, float, float]:
    """HSV jitter of a linear RGB colour: hue shift [turns], saturation and value scales."""
    import colorsys

    h, s, v = colorsys.rgb_to_hsv(*[float(c) for c in rgb])
    return tuple(min(max(c, 0.0), 1.0) for c in colorsys.hsv_to_rgb((h + hue) % 1.0, min(s * sat, 1.0), v * val))


def randomize_lamp(env: ManagerBasedEnv, env_ids: torch.Tensor | None, pos=(-0.3, -0.3, 0.7),
                   pos_range=((-0.15, 0.15), (-0.15, 0.15), (-0.15, 0.15)), intensity: float = 1.0e5,
                   intensity_range=(0.6, 1.5), color=(1.0, 1.0, 1.0), warmth_range=(-1.0, 1.0),
                   light_path: str = "/World/Lamp"):
    """Desk lamp (sphere light): position in a box around the estimate, intensity scale, colour temperature
    (warmth -1 cooler .. +1 warmer than ``color``)."""
    import isaaclab.sim as sim_utils
    from pxr import Gf

    prim = sim_utils.get_current_stage().GetPrimAtPath(light_path)
    tr = prim.GetAttribute("xformOp:translate")
    tr.Set(type(tr.Get())(*[c + _uniform(r) for c, r in zip(pos, pos_range)]))
    prim.GetAttribute("inputs:intensity").Set(intensity * _uniform(intensity_range))
    w = _uniform(warmth_range)
    prim.GetAttribute("inputs:color").Set(Gf.Vec3f(color[0], color[1] * (1 - 0.05 * w), color[2] * (1 - 0.15 * w)))


def _set_emissive(shader_prim, rgb):
    from pxr import Gf, Sdf, UsdShade

    UsdShade.Shader(shader_prim).CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in rgb]))


def randomize_object_color(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_names: tuple, base_color,
                           material: str | None = None, hue_range=(-0.02, 0.02), sat_range=(0.85, 1.15),
                           val_range=(0.8, 1.2), emissive: float = 0.0):
    """HSV jitter of the diffuse colour of the assets' UsdPreviewSurface material(s) around ``base_color``;
    ``emissive`` > 0 adds that fraction of the colour as emission (the thin translucent cup glows from inside)."""
    import isaaclab.sim as sim_utils
    from pxr import Gf

    stage = sim_utils.get_current_stage()
    ids = range(env.num_envs) if env_ids is None else env_ids.tolist()
    for i in ids:
        c = _tint(base_color, _uniform(hue_range), _uniform(sat_range), _uniform(val_range))
        for name in asset_names:
            for sh in _preview_shaders(stage, _env_prim(env, i, name), material):
                sh.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*c))
                if emissive > 0:
                    _set_emissive(sh, [emissive * v for v in c])


def randomize_jaw_glow(env: ManagerBasedEnv, env_ids: torch.Tensor | None, emissive_range=(0.5, 1.0),
                       material: str = "/World/Looks/r2s_jaw"):
    """Emission of the jaw material (recolor_printed_parts(jaw_emissive > 0)) as a fraction of its diffuse colour."""
    import isaaclab.sim as sim_utils
    from pxr import Gf

    stage = sim_utils.get_current_stage()
    ids = range(env.num_envs) if env_ids is None else env_ids.tolist()
    for i in ids:
        e = _uniform(emissive_range)
        for sh in _preview_shaders(stage, f"{material}_{i}"):
            c = sh.GetAttribute("inputs:diffuseColor").Get()
            if c is not None:
                _set_emissive(sh, [e * float(v) for v in c])


def randomize_desk_texture(env: ManagerBasedEnv, env_ids: torch.Tensor | None, asset_name: str = "backdrop",
                           gain_range=(0.75, 1.25), tint_std: float = 0.05, alt_prob: float = 0.0,
                           alternatives: tuple = (), shader: str = "Looks/DeskMat/Texture"):
    """Desk texture brightness and hue (UsdUVTexture scale) and, with probability ``alt_prob``, one of the
    ``alternatives`` (texture files next to the nominal one) instead of the real desk."""
    import os

    import isaaclab.sim as sim_utils
    from pxr import Gf, Sdf

    stage = sim_utils.get_current_stage()
    ids = range(env.num_envs) if env_ids is None else env_ids.tolist()
    for i in ids:
        prim = stage.GetPrimAtPath(f"{_env_prim(env, i, asset_name)}/{shader}")
        if not prim.IsValid():
            continue
        g = _uniform(gain_range) * (1.0 + tint_std * torch.randn(3)).clamp(0.5, 1.5)
        prim.GetAttribute("inputs:scale").Set(Gf.Vec4f(*g.tolist(), 1.0))
        f = prim.GetAttribute("inputs:file")
        if not hasattr(env, "_r2s_desk_tex"):
            env._r2s_desk_tex = f.Get().resolvedPath or f.Get().path
        path = env._r2s_desk_tex
        if alternatives and _uniform((0.0, 1.0)) < alt_prob:
            path = os.path.join(os.path.dirname(path), alternatives[int(torch.randint(len(alternatives), (1,)))])
        f.Set(Sdf.AssetPath(path))


def randomize_camera_gain(env: ManagerBasedEnv, env_ids: torch.Tensor | None, sensor_name: str, gain=(1.0, 1.0, 1.0),
                          gain_range=(0.85, 1.15), tint_std: float = 0.03, offset=None, offset_std: float = 0.0):
    """Per-camera colour gains (exposure x white balance) around the nominal ``gain``; applied by camera_rgb.
    ``offset`` (0..255 per channel, optional): the camera's black level / flare added after the gain, jittered by
    ``offset_std`` per channel (the cheap wrist camera lifts and tints its dark tones)."""
    if not hasattr(env, "r2s_cam_gain"):
        env.r2s_cam_gain = {}
    nominal = torch.tensor(gain, device=env.device)
    g = env.r2s_cam_gain.setdefault(sensor_name, nominal.repeat(env.num_envs, 1))
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    scale = torch.empty(n, 1, device=env.device).uniform_(*gain_range)
    g[env_ids] = nominal * scale * (1.0 + tint_std * torch.randn(n, 3, device=env.device))
    if offset is not None:
        if not hasattr(env, "r2s_cam_offset"):
            env.r2s_cam_offset = {}
        off0 = torch.tensor(offset, device=env.device, dtype=torch.float32)
        o = env.r2s_cam_offset.setdefault(sensor_name, off0.repeat(env.num_envs, 1))
        o[env_ids] = off0 + offset_std * torch.randn(n, 3, device=env.device)


def camera_rgb(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, gain=(1.0, 1.0, 1.0), offset=None,
               noise_std: float = 0.0) -> torch.Tensor:
    """Camera RGB (0..255) times its colour gains: the per-reset ones from randomize_camera_gain, else the nominal
    ``gain`` (the real camera's exposure / white balance relative to the sim render). Optional (splat config):
    ``offset`` = black level added after the gain (per-reset values from randomize_camera_gain(offset=...) win) and
    per-pixel Gaussian sensor noise of ``noise_std`` (0..255 units)."""
    import isaaclab.envs.mdp as base_mdp

    img = base_mdp.image(env, sensor_cfg=sensor_cfg, data_type="rgb", normalize=False)
    g = getattr(env, "r2s_cam_gain", {}).get(sensor_cfg.name)
    if offset is None and noise_std <= 0:  # the original (non-splat) path
        if g is None:
            if all(c == 1.0 for c in gain):
                return img
            g = torch.tensor(gain, device=img.device).expand(img.shape[0], 3)
        return (img[..., :3].float() * g[:, None, None, :]).clamp(0, 255).to(img.dtype)
    if g is None:
        g = torch.tensor(gain, device=img.device, dtype=torch.float32).expand(img.shape[0], 3)
    o = getattr(env, "r2s_cam_offset", {}).get(sensor_cfg.name)
    if o is None:
        o = torch.tensor(offset or (0.0, 0.0, 0.0), device=img.device, dtype=torch.float32).expand(img.shape[0], 3)
    out = img[..., :3].float() * g[:, None, None, :] + o[:, None, None, :]
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return out.clamp(0, 255).to(img.dtype)

"""Scripted cube-into-cup expert for the SO-101 (privileged state -> absolute joint targets).

Phases (per env): PREGRASP -> DESCEND -> CLOSE -> LIFT -> TRANSPORT -> LOWER -> RELEASE -> RETREAT -> DONE.

Geometry (measured with scripts/inspect_gripper.py, ``gripper`` link frame): the jaws point along -z,
the fixed jaw's inner face is at x = -8.8 mm and the moving jaw opens toward +x. A cube of size s is
therefore grasped with its centre at x = -8.8 mm + s/2 (+ a few mm clearance while approaching).

The SO-101 is a small 5-DOF arm: a straight-down gripper is only reachable near the base, so the
approach tilts outward with horizontal reach (0 deg at r <= 0.17 m, up to MAX_TILT at r >= 0.29 m).
The jaw-closing axis is the cube face normal projected onto the plane normal to the approach.
Targets are converted to joint targets by the SO-101 differential IK (position exact, orientation soft).

Re-planning: ``replan(env_ids)`` infers the phase from the CURRENT privileged state (cube in the cup ->
RELEASE/RETREAT/DONE; cube held -> LOWER/LIFT/TRANSPORT; cube between the open jaws -> CLOSE; cube lying on
the cup rim at the hand -> let go and back off; otherwise open the jaw and PREGRASP, climbing vertically
first when the hand is low, or DESCEND directly when the hand is already right above the cube) and
restarts the motion from the current TCP.
It lets the expert take over from any state (DAgger-style recovery data: scripts/record_takeover.py),
and the expert calls it by itself when the cube is lost from the hand during LIFT/TRANSPORT/LOWER
(missed or slipped grasp) or comes to rest outside the cup after the release (bounced out onto the desk
or the rim). From reset nothing changes unless one of those failures happens.
"""
from __future__ import annotations

import math

import torch

from isaaclab.utils import math as math_utils
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_controller import (
    SO101PoseIKController,
    SO101PoseIKControllerCfg,
)

from . import mdp
from .env_cfg import FIXED_JAW_X, GRIPPER_CLOSED, GRIPPER_OPEN, JOINTS, TCP_Z, TIP_Z

PREGRASP, DESCEND, CLOSE, LIFT, TRANSPORT, LOWER, RELEASE, RETREAT, DONE = range(9)
PHASE_NAMES = ["pregrasp", "descend", "close", "lift", "transport", "lower", "release", "retreat", "done"]
MAX_TILT = math.radians(40.0)
LOST_STEPS = 5      # control steps the cube must be clearly out of the hand (carry phases) before re-planning
FELL_OUT_STEPS = 8  # control steps the released cube must rest outside the cup before re-planning
NEAR_CUP = 0.16     # [m] cube centre this close to the cup axis -> re-planned grasp checks its clearance from the cup
CLEAR_Z = 0.05      # [m] replanned approach: lowest cruise height above the cube top (tilted hand: camera mount)
YAW_RATE, TILT_RATE = 3.0, 2.0  # [rad/s], [1/s] slew of the re-planned jaw yaw / tilt scale (no wrist whip)
ROLL_MARGIN = 0.25  # [rad] re-planned jaw yaw: keep the predicted wrist roll this far inside its limits
NEAR_CUP_OPEN = 0.35  # [rad] near-cup re-grasp: open the jaw only this far past the cube-contact angle
# Hand collision geometry in the gripper link frame (box corners [m]; from the robot's collision shapes, as printed by
# scripts/inspect_gripper.py) -- for the near-cup grasp choice in replan: (xmin, xmax, ymin, ymax, zmin, zmax)
HAND_BOXES = (
    (-0.024, 0.021, 0.024, 0.087, -0.064, -0.017),    # wrist-camera mount
    (-0.015, 0.030, -0.019, 0.020, -0.036, -0.011),   # servo
    (-0.017, 0.003, -0.024, 0.024, -0.036, -0.009),   # wrist-roll follower
    (-0.032, -0.012, -0.023, 0.023, -0.042, -0.004),  # fixed jaw, from the top ...
    (-0.036, -0.014, -0.013, 0.013, -0.064, -0.042),
    (-0.028, -0.013, -0.009, 0.009, -0.084, -0.063),
    (-0.021, -0.011, -0.008, 0.007, -0.093, -0.083),
    (-0.017, -0.009, -0.006, 0.005, -0.104, -0.093),  # ... to the tip
)
JAW_BOXES = (  # moving jaw at q = 0; it opens by rotating about the y axis through x = 21, z = -23 mm (tip toward +x, up)
    (0.011, 0.031, -0.025, 0.025, -0.045, -0.013),
    (0.012, 0.028, -0.011, 0.011, -0.075, -0.045),
    (0.013, 0.025, -0.008, 0.008, -0.085, -0.075),
    (0.011, 0.023, -0.007, 0.007, -0.095, -0.085),
    (0.008, 0.018, -0.006, 0.006, -0.105, -0.095),
)
JAW_HINGE = (0.021, -0.023)
CLEAR_MIN, CLEAR_W = 0.012, 400.0  # near-cup grasp choice: cost CLEAR_W [1/m] x (CLEAR_MIN - hand clearance from the cup)+
NEAR_TILTS = (1.0, 0.0)  # near-cup grasp choice: candidate scales of the reach-dependent outward tilt (1 = as in the demos)
TILT_OFF_COST = 0.3  # ... and the cost of each unit of deviation from 1 (prefer the demos' pose when the clearance is equal)
NEAR_CUP_HAND = 0.10  # [m] re-planned climb with the hand this close (xy) to the cup axis: jaw stays shut until clear
REST_STEPS = 30     # control steps a released cube must rest outside the cup, off the desk (on the rim), before re-planning
LOWER_DZ = 0.0      # [m] LOWER before RELEASE: 0 = drop from the carry height (cube bottom 2.5 cm above the rim), as the
                    # user does; lowering 2 cm put the tilted hand's fingertips inside far cups -> cube wedged, cup tipped
JOINT_RATE = 1.57   # [rad/s] arm joint command slew limit: the user's teleop moves at <= ~90 deg/s (99th pct, real episodes);
                    # without it the differential IK jumps -- from the folded rest pose by ~70 deg in one step


class CubeInCupExpert:
    def __init__(self, env, speed: float = 0.20, noise: float = 0.0, seed: int | None = None):
        """``speed``: max TCP speed [m/s]; ``noise``: std [m] of per-episode waypoint jitter (demo diversity)."""
        self.env = env
        robot = env.scene["robot"]
        self.robot = robot
        self.N, self.dev = env.num_envs, env.device
        self.dt = env.step_dt
        self.arm_ids = [robot.joint_names.index(j) for j in JOINTS[:5]]
        self.grip_id = robot.joint_names.index("gripper")
        self.body_idx = robot.body_names.index("gripper")
        self.jac_idx = self.body_idx - 1  # fixed-base articulation: jacobians skip the root body
        self.speed, self.noise = speed, noise
        self.gen = torch.Generator(device=self.dev)
        if seed is not None:
            self.gen.manual_seed(seed)
        self.ik = SO101PoseIKController(
            SO101PoseIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="adaptive_dls",
                ik_params={"lambda_min": 0.05, "lambda_max": 0.2, "sigma_thresh": 0.02},
                orientation_weight=0.5, orientation_joint_names=("wrist_flex", "wrist_roll"),
                joint_limit_avoidance_gain=0.5, joint_limit_avoidance_margin=0.3,
            ),
            num_envs=self.N, device=self.dev,
        )
        self.ik.set_orientation_joint_mask(torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], device=self.dev))
        lim = robot.data.joint_pos_limits.torch[:, self.arm_ids]
        self.ik.set_joint_pos_limits(lim[..., 0].clone(), lim[..., 1].clone())
        z = lambda *s: torch.zeros(*s, device=self.dev)
        self.tcp_off = z(self.N, 3)  # per-env grasp point in the gripper frame (depends on cube size)
        self.phase = torch.zeros(self.N, dtype=torch.long, device=self.dev)
        self.timer, self.face_yaw = z(self.N), z(self.N)
        self.cmd, self.jitter = z(self.N, 3), z(self.N, 3)
        self.hold_off_g = z(self.N, 3)  # cube centre - TCP, in the gripper frame, captured at grasp
        self.grip = torch.full((self.N,), GRIPPER_OPEN, device=self.dev)
        self.prev_tcp, self.tcp_speed = z(self.N, 3), z(self.N)  # for settle detection (EMA of TCP speed)
        self.rise = torch.zeros(self.N, dtype=torch.bool, device=self.dev)  # PREGRASP: climb vertically first
        self.rise_xy = z(self.N, 2)  # ... to above this point (replan: away from the cup when the hand is beside it)
        self.lost_steps = torch.zeros(self.N, dtype=torch.long, device=self.dev)  # debounce for self-recovery
        self.n_replans = torch.zeros(self.N, dtype=torch.long, device=self.dev)  # self-recoveries since reset
        self.tilt_scale = torch.ones(self.N, device=self.dev)  # near-cup grasps: less outward tilt (replan)
        self.pre_z = torch.full((self.N,), float("nan"), device=self.dev)  # replanned PREGRASP cruise height
        self.yaw_goal, self.tilt_goal = z(self.N), torch.ones(self.N, device=self.dev)  # face_yaw / tilt_scale slew to these
        self.open_q = torch.full((self.N,), GRIPPER_OPEN, device=self.dev)  # open-jaw command (replan: less near the cup)
        self.rest_steps = torch.zeros(self.N, dtype=torch.long, device=self.dev)  # debounce: released cube resting on the rim
        self.shut_q = torch.full((self.N,), float("nan"), device=self.dev)  # re-planned climb next to the cup: jaw held here
        self.q_cmd = robot.data.joint_pos.torch[:, self.arm_ids].clone()  # slew-limited arm command (JOINT_RATE)

    # ------------------------------------------------------------------ helpers
    def _body_pose(self):
        d = self.robot.data
        return d.body_link_pos_w.torch[:, self.body_idx], d.body_link_quat_w.torch[:, self.body_idx]

    def tcp_w(self):
        p, q = self._body_pose()
        return p + math_utils.quat_apply(q, self.tcp_off)

    def _target_rot(self, tcp_target_w: torch.Tensor, face_yaw: torch.Tensor, tilt_scale: torch.Tensor) -> torch.Tensor:
        """Gripper orientation (rotation matrix) for a TCP target: reach-dependent outward tilt, jaw axis ~ cube face."""
        base = self.robot.data.root_pos_w.torch
        rel = tcp_target_w - base
        r = torch.linalg.vector_norm(rel[:, :2], dim=-1)
        phi = torch.atan2(rel[:, 1], rel[:, 0])
        tilt = ((r - 0.17) / 0.12).clamp(0.0, 1.0) * MAX_TILT * tilt_scale
        approach = torch.stack([torch.sin(tilt) * torch.cos(phi), torch.sin(tilt) * torch.sin(phi), -torch.cos(tilt)], -1)
        z_g = -approach  # jaws extend along -z of the gripper link
        n = torch.stack([torch.cos(face_yaw), torch.sin(face_yaw), torch.zeros_like(r)], -1)
        x_g = n - (n * z_g).sum(-1, keepdim=True) * z_g
        x_g = x_g / torch.linalg.vector_norm(x_g, dim=-1, keepdim=True).clamp_min(1e-6)
        y_g = torch.cross(z_g, x_g, dim=-1)
        return torch.stack([x_g, y_g, z_g], dim=-1)

    def _target_quat(self, tcp_target_w: torch.Tensor) -> torch.Tensor:
        return math_utils.quat_from_matrix(self._target_rot(tcp_target_w, self.face_yaw, self.tilt_scale))

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = torch.arange(self.N, device=self.dev)
        size = mdp.active_cube_size(self.env)[env_ids]
        self.tcp_off[env_ids] = torch.stack(
            [FIXED_JAW_X + size / 2 + 0.004, torch.zeros_like(size), torch.full_like(size, TCP_Z)], -1)
        self.phase[env_ids] = PREGRASP
        self.timer[env_ids] = 0.0
        self.cmd[env_ids] = self.tcp_w()[env_ids]
        self.prev_tcp[env_ids] = self.cmd[env_ids]
        self.tcp_speed[env_ids] = 1.0  # assume moving until measured
        self.grip[env_ids] = GRIPPER_OPEN
        self.jitter[env_ids] = torch.randn(len(env_ids), 3, device=self.dev, generator=self.gen) * self.noise
        self.jitter[env_ids, 2] = self.jitter[env_ids, 2].abs()  # never jitter downward into the desk
        # jaw axis: the cube face normal closest to the current jaw axis (minimises wrist roll)
        _, cube_q = mdp.active_cube_pose_w(self.env)
        cube_yaw = math_utils.euler_xyz_from_quat(cube_q)[2][env_ids]
        _, bq = self._body_pose()
        x_axis = math_utils.quat_apply(bq, torch.tensor([1.0, 0.0, 0.0], device=self.dev).expand(self.N, 3))
        cur = torch.atan2(x_axis[env_ids, 1], x_axis[env_ids, 0])
        cands = cube_yaw[:, None] + torch.arange(4, device=self.dev)[None] * (math.pi / 2)
        diff = torch.remainder(cands - cur[:, None] + math.pi, 2 * math.pi) - math.pi
        self.face_yaw[env_ids] = cur + diff.gather(1, diff.abs().argmin(1, keepdim=True)).squeeze(1)
        # from the folded rest pose the hand starts below the cup rim: climb straight up to the cruise height first
        # (the diagonal path to the cube could sweep a cup that stands between them)
        desk_z = self.env.scene.env_origins[env_ids, 2]
        self.rise[env_ids] = self.cmd[env_ids, 2] < desk_z + self.env.cfg.cup_dims["height"] + 0.02
        self.rise_xy[env_ids] = self.cmd[env_ids, :2]
        self.lost_steps[env_ids] = 0
        self.n_replans[env_ids] = 0
        self.tilt_scale[env_ids] = 1.0
        self.pre_z[env_ids] = float("nan")
        self.yaw_goal[env_ids] = self.face_yaw[env_ids]
        self.tilt_goal[env_ids] = 1.0
        self.open_q[env_ids] = GRIPPER_OPEN
        self.rest_steps[env_ids] = 0
        self.shut_q[env_ids] = float("nan")
        self.q_cmd[env_ids] = self.robot.data.joint_pos.torch[env_ids][:, self.arm_ids]

    # ------------------------------------------------------------------ re-planning
    def _jaw_yaw(self) -> torch.Tensor:
        """World yaw of the gripper x axis (the jaw-closing axis)."""
        _, bq = self._body_pose()
        x_axis = math_utils.quat_apply(bq, torch.tensor([1.0, 0.0, 0.0], device=self.dev).expand(self.N, 3))
        return torch.atan2(x_axis[:, 1], x_axis[:, 0])

    def _cube_face_yaw(self) -> torch.Tensor:
        """Yaw of a horizontal face normal of the active cube, whichever face it rests on (a dropped
        cube may lie on any face; the Euler yaw is meaningless then)."""
        _, cube_q = mdp.active_cube_pose_w(self.env)
        rot = math_utils.matrix_from_quat(cube_q)  # columns = cube axes in the world
        side = (rot[:, 2, :].abs().argmax(-1) + 1) % 3  # an axis that is not the (most) vertical one
        axis = rot.gather(2, side[:, None, None].expand(-1, 3, 1)).squeeze(-1)
        return torch.atan2(axis[:, 1], axis[:, 0])

    def _cube_speed(self) -> torch.Tensor:
        v = torch.stack([self.env.scene[n].data.root_lin_vel_w.torch for n in mdp.CUBE_NAMES], 1)
        return torch.linalg.vector_norm(v[torch.arange(self.N, device=self.dev), mdp.active_cube_idx(self.env)], dim=-1)

    def _hand_points(self, open_q: torch.Tensor) -> torch.Tensor:
        """Corners of the hand's collision boxes in the gripper link frame [N, P, 3], moving jaw opened to ``open_q``."""
        def corners(boxes):
            b = torch.tensor(boxes, device=self.dev)
            ix = torch.tensor([[i, j, k] for i in (0, 1) for j in (2, 3) for k in (4, 5)], device=self.dev)
            return b[:, ix].reshape(-1, 3)  # [8 * boxes, 3]
        hand, jaw = corners(HAND_BOXES), corners(JAW_BOXES)
        dx, dz = jaw[None, :, 0] - JAW_HINGE[0], jaw[None, :, 2] - JAW_HINGE[1]
        c, s = torch.cos(open_q)[:, None], torch.sin(open_q)[:, None]
        xs, zs = JAW_HINGE[0] + dx * c - dz * s, JAW_HINGE[1] + dx * s + dz * c  # [N, 8 * boxes]
        jaw_q = torch.stack([xs, jaw[None, :, 1].expand_as(xs), zs], -1)
        return torch.cat([hand[None].expand(self.N, -1, -1), jaw_q], 1)

    def _grasp_clearance(self, tcp_t, face_yaw, tilt_scale, size, pts, cup_p, desk_z):
        """Horizontal clearance [m] of the hand from the cup rim when grasping with the TCP at ``tcp_t``, jaw yaw
        ``face_yaw`` and tilt scale ``tilt_scale``: every hand corner that ends up below the rim passes rim height on
        the vertical descent, so it should stay outside the rim radius (negative: it would hit the cup -- or hang
        over the mouth, which in the stress test went wrong more often than not: the hand lands ~1-2 cm off target)."""
        rot = self._target_rot(tcp_t, face_yaw, tilt_scale)
        off = torch.stack([FIXED_JAW_X + size / 2 + 0.004, torch.zeros_like(size), torch.full_like(size, TCP_Z)], -1)
        w = tcp_t[:, None] + torch.einsum("nij,npj->npi", rot, pts - off[:, None])
        d = torch.linalg.vector_norm(w[..., :2] - cup_p[:, None, :2], dim=-1) - self.env.cfg.cup_dims["top_d"] / 2
        low = (w[..., 2] - desk_z[:, None]) < self.env.cfg.cup_dims["height"] + 0.005
        return torch.where(low, d, torch.full_like(d, 1.0)).amin(-1)

    def replan(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Choose the phase that fits the CURRENT privileged state and restart from the current TCP.

        * cube in the cup: RELEASE if it is still in the hand, else RETREAT (DONE if already clear above);
        * cube held (``mdp.cube_grasped``): LOWER if already above the cup opening, else LIFT up to the
          carry height (vertically, never sweeping the rim), or TRANSPORT if already that high;
        * cube between the jaws, jaw not yet closed on it: CLOSE;
        * cube lying on / wedged in the cup rim at the hand: open the jaw (RELEASE) and back off up (RETREAT);
        * otherwise the jaw opens and the expert goes to PREGRASP at a cruise height (``self.pre_z``: the
          current height if the way over is clear, else the normal safe height), climbing straight up
          first if the hand is below it (``self.rise``; next to the cup with the jaw kept as it is until up);
          hand already right above the cube -> DESCEND.
          Near the cup the jaw axis is the cube face -- with the normal outward tilt or none -- whose grasp
          pose keeps the hand's collision boxes clearest of the cup, and the jaw opens only NEAR_CUP_OPEN
          past the cube.
        Returns the chosen phase of each env in ``env_ids``.
        """
        env, dev = self.env, self.dev
        if env_ids is None:
            env_ids = torch.arange(self.N, device=dev)
        env_ids = torch.as_tensor(env_ids, device=dev, dtype=torch.long).flatten()
        if env_ids.numel() == 0:
            return self.phase[env_ids]
        size = mdp.active_cube_size(env)
        self.tcp_off[env_ids] = torch.stack(
            [FIXED_JAW_X + size[env_ids] / 2 + 0.004, torch.zeros_like(size[env_ids]), torch.full_like(size[env_ids], TCP_Z)], -1)
        cube_p, _ = mdp.active_cube_pose_w(env)
        cup_p = env.scene["cup"].data.root_pos_w.torch
        desk_z = env.scene.env_origins[:, 2]
        cup_h = env.cfg.cup_dims["height"]
        inner_r = env.cfg.cup_dims["top_d"] / 2 - env.cfg.cup_dims["wall"]
        tcp = self.tcp_w()
        body_p, bq = self._body_pose()
        q_grip = self.robot.data.joint_pos.torch[:, self.grip_id]
        q_contact = (size - 0.017) / 0.066  # jaw angle touching the cube (as in mdp.cube_grasped)

        held = mdp.cube_grasped(env)
        in_cup = mdp.cube_in_cup(env)
        c_g = math_utils.quat_apply_inverse(bq, cube_p - body_p)  # cube centre in the gripper frame
        between = (((c_g[:, 0] - (FIXED_JAW_X + size / 2)).abs() < 0.010) & (c_g[:, 1].abs() < size / 2 + 0.005)
                   & (c_g[:, 2] > TIP_Z - size / 2 + 0.004) & (c_g[:, 2] < TIP_Z + 0.040))
        jaw_open = q_grip > q_contact + 0.15  # wider than the cube
        grasp_z = torch.maximum(cube_p[:, 2], desk_z + (TCP_Z - TIP_Z) + 0.004)
        safe_z = torch.maximum(grasp_z + size / 2 + 0.05, desk_z + cup_h + 0.03)
        xy_d = torch.linalg.vector_norm(tcp[:, :2] - cube_p[:, :2], dim=-1)
        cube_cup_d = torch.linalg.vector_norm(cube_p[:, :2] - cup_p[:, :2], dim=-1)

        # jaw axis: keep the current one while the cube is (being) held; otherwise the cube face normal
        # closest to it (least wrist roll). Near the cup (a dropped cube against its wall) the face -- and outward
        # tilt or none -- whose grasp pose keeps the hand's collision boxes (HAND_BOXES, JAW_BOXES opened to
        # NEAR_CUP_OPEN) clear of the cup: the hand is ~67 x 48 mm plus the wrist-camera mount, sits off-centre on
        # the fixed-jaw side, and its parts hang at cup height. (A fixed rule -- jaw axis tangent, mount away, no tilt
        # with the cup on the robot's side -- still hit the cup with the moving jaw in ~30% of the stress-test grasps.)
        near = cube_cup_d < NEAR_CUP
        cur = self._jaw_yaw()
        cands = self._cube_face_yaw()[:, None] + torch.arange(4, device=dev)[None] * (math.pi / 2)
        diff = torch.remainder(cands - cur[:, None] + math.pi, 2 * math.pi) - math.pi
        # wrist-roll range: for a hand pointing down, the world jaw yaw is ~ const + p * (pan - roll) (p = +-1: the
        # pan axis is the world +-z axis), and the roll follows the pan while the jaw keeps its world yaw. So a
        # candidate needs roll + p * (dpsi_reach - diff) at the grasp and p * dpsi_carry more over the cup (dpsi:
        # change of the arm's azimuth). Unchecked, a near-cup re-grasp picked a yaw ~157 deg away, the roll ran into
        # its limit during the carry, and the IK chattered and shoved the cup 3 cm.
        roll_id = self.arm_ids[4]
        roll = self.robot.data.joint_pos.torch[:, roll_id]
        roll_lim = self.robot.data.joint_pos_limits.torch[:, roll_id]
        roll_lo, roll_hi = roll_lim[:, 0] + ROLL_MARGIN, roll_lim[:, 1] - ROLL_MARGIN
        p = torch.sign(self.robot.data.body_link_jacobian_w.torch[:, self.jac_idx, 5, self.arm_ids[0]])
        p = torch.where(p == 0, torch.ones_like(p), p)
        base_xy = self.robot.data.root_pos_w.torch[:, :2]
        az = lambda xy: torch.atan2(xy[:, 1] - base_xy[:, 1], xy[:, 0] - base_xy[:, 0])
        wrap = lambda a: torch.remainder(a + math.pi, 2 * math.pi) - math.pi
        r_grasp = roll[:, None] + p[:, None] * (wrap(az(cube_p) - az(tcp))[:, None] - diff)
        r_cup = r_grasp + (p * wrap(az(cup_p) - az(cube_p)))[:, None]
        r_over = (torch.relu(roll_lo[:, None] - torch.minimum(r_grasp, r_cup))
                  + torch.relu(torch.maximum(r_grasp, r_cup) - roll_hi[:, None]))
        cost = diff.abs() + 50.0 * r_over  # [N, 4]
        # near the cup, also the hand's clearance from the cup for each candidate, with the normal outward tilt or none
        # (the 5-DOF arm can only tilt the hand toward / away from its base): [N, 4 yaws x 2 tilts]
        tcp_t = torch.stack([cube_p[:, 0], cube_p[:, 1], grasp_z], -1)
        # first with the demos' full jaw opening; only if no candidate clears the cup that way, with the jaw opened just
        # NEAR_CUP_OPEN past the cube (fully open, the moving jaw's tip reaches ~11 cm from the cube along +x, 2 cm above
        # the TCP: it hit cups up to ~13 cm away in the takeover test)
        tilts = torch.tensor(NEAR_TILTS, device=dev).repeat_interleave(4)
        def clearances(open_q_):
            pts = self._hand_points(open_q_)
            return torch.stack([self._grasp_clearance(tcp_t, cands[:, k], torch.full_like(size, t_), size, pts, cup_p, desk_z)
                                for t_ in NEAR_TILTS for k in range(4)], 1)
        clr_full = clearances(torch.full_like(size, GRIPPER_OPEN))
        narrow = (clr_full < CLEAR_MIN).all(1)
        clr = torch.where(narrow[:, None], clearances((q_contact + NEAR_CUP_OPEN).clamp(max=GRIPPER_OPEN)), clr_full)
        cost2 = cost.repeat(1, len(NEAR_TILTS)) + TILT_OFF_COST * (tilts - 1.0).abs() + CLEAR_W * torch.relu(CLEAR_MIN - clr)
        best = torch.where(near, cost2.argmin(1), cost.argmin(1))
        face_yaw = cur + diff.gather(1, (best % 4)[:, None]).squeeze(1)
        near_tilt = tilts[best]
        keep_axis = held | in_cup | (between & (q_grip > q_contact - 0.06))
        face_yaw = torch.where(keep_axis, cur, face_yaw)
        # cube held on the way to the cup: its jaw axis no longer matters -- if keeping it would run the roll into
        # a limit before the cup, turn it in flight by just enough (roll shift x <-> yaw change -p * x)
        r0, r1 = roll, roll + p * wrap(az(cup_p) - az(tcp))
        x_lo, x_hi = roll_lo - torch.minimum(r0, r1), roll_hi - torch.maximum(r0, r1)
        x = torch.where(x_lo <= x_hi, torch.maximum(x_lo, torch.minimum(torch.zeros_like(x_lo), x_hi)), 0.5 * (x_lo + x_hi))
        face_yaw = torch.where(held & ~in_cup, cur - p * x, face_yaw)

        # carry geometry as in act(): cube held -> its current offset in the hand; cube in the cup -> none
        hold_g = math_utils.quat_apply_inverse(bq, cube_p - tcp)
        hold_g = torch.where(held[:, None], hold_g, torch.zeros_like(hold_g))
        old_yaw, old_tilt = self.face_yaw.clone(), self.tilt_scale.clone()
        self.face_yaw[:], self.tilt_scale[:] = face_yaw, 1.0  # (held / in-cup envs carry with the normal tilt)
        hold_w = math_utils.quat_apply(self._target_quat(tcp), hold_g)
        self.face_yaw[:], self.tilt_scale[:] = old_yaw, old_tilt
        carry_tcp_z = desk_z + cup_h + size / 2 + 0.025 - hold_w[:, 2]
        bottom = cube_p[:, 2] - desk_z - size / 2

        ph = torch.full_like(self.phase, PREGRASP)
        # free cube: right above it with the jaws clearing its top -> straight down. Otherwise fly over at a
        # cruise height: the normal expert always uses safe_z (rim + 3 cm), but after a takeover the hand is
        # often already low near the cube and going up and down again is wasted motion (data that teaches
        # hesitation) -- so keep the current height, at least CLEAR_Z above the cube top, unless the
        # horizontal leg passes within 4.5 cm of the cup rim below safe_z. A hand lower than that climbs
        # vertically first (``rise``).
        # (the descent closes the xy offset and the height together: the fingertips must clear the cube top by more than
        # the offset, or the jaw lands on the cube's edge -- seen: a 2 cm offset 8 mm above a 40 mm cube, 5 failed grasps)
        tips_clear = tcp[:, 2] > cube_p[:, 2] + size / 2 + (TCP_Z - TIP_Z) + 0.008 + xy_d
        above = (xy_d < 0.02) & tips_clear & (tcp[:, 2] < safe_z + 0.02)
        ph = torch.where(above, DESCEND, ph)
        cruise = torch.minimum(torch.maximum(tcp[:, 2], cube_p[:, 2] + size / 2 + CLEAR_Z), safe_z)
        s_ = torch.linspace(0.0, 1.0, 17, device=dev)[None, :, None]
        leg = tcp[:, None, :2] + s_ * (cube_p[:, None, :2] - tcp[:, None, :2])
        cup_r = env.cfg.cup_dims["top_d"] / 2
        crosses = (torch.linalg.vector_norm(leg - cup_p[:, None, :2], dim=-1) < cup_r + 0.045).any(1)
        cruise = torch.where(crosses & (cruise < desk_z + cup_h + 0.03), safe_z, cruise)
        # a hand at the cup below its rim (e.g. pressed against it by the policy) first gets up and away from it: climb to
        # the safe height, backing off to NEAR_CUP_HAND from the cup axis if it is beside the cup (straight up if it is
        # over the mouth); moving on at its height swept a tilted cup over
        hand_cup_d = torch.linalg.vector_norm(tcp[:, :2] - cup_p[:, :2], dim=-1)
        at_cup = (hand_cup_d < NEAR_CUP_HAND) & (tcp[:, 2] < desk_z + cup_h + 0.03)
        cruise = torch.where(at_cup, safe_z, cruise)
        away = (tcp[:, :2] - cup_p[:, :2]) / hand_cup_d[:, None].clamp_min(1e-6)
        rise_xy = torch.where((at_cup & (hand_cup_d > cup_r))[:, None], cup_p[:, :2] + away * NEAR_CUP_HAND, tcp[:, :2])
        rise = ~above & (tcp[:, 2] < cruise - 0.01)
        closing = between & ~held & (q_grip > q_contact - 0.06)
        ph = torch.where(closing, CLOSE, ph)
        over_opening = (cube_cup_d < inner_r - size / 2) & (bottom > cup_h + 0.002)
        ph_held = torch.where(over_opening, LOWER, torch.where(tcp[:, 2] < carry_tcp_z - 0.01, LIFT, TRANSPORT))
        ph = torch.where(held, ph_held, ph)
        retreat_z = carry_tcp_z + 0.04
        up_done = torch.where(tcp[:, 2] > retreat_z - 0.012, DONE, RETREAT)
        ph_cup = torch.where(held | (between & ~jaw_open), RELEASE, up_done)
        ph = torch.where(in_cup, ph_cup, ph)
        # cube resting on / wedged in the cup rim with the hand at it (not held, not in the cup, supported at rim
        # height): closing on it and lifting drags the cup along (seen: 25 mm) -- open the jaw and back off vertically
        # instead; it drops into the cup, or onto the desk from where the self-recovery re-grasps it (or, if it stays on
        # the rim, the expert re-plans from above once it has rested there for REST_STEPS)
        on_rim = (~held & ~in_cup & (cube_cup_d < cup_r + size / 2) & (bottom > cup_h - size)
                  & (xy_d < 0.04) & ((tcp[:, 2] - cube_p[:, 2]).abs() < 0.04))
        ph = torch.where(on_rim, torch.where(jaw_open, up_done, RELEASE), ph)
        face_yaw = torch.where(on_rim, cur, face_yaw)  # (no wrist turn while letting go next to the cup)
        rise &= ph == PREGRASP

        free = near & ((ph == PREGRASP) | (ph == DESCEND))
        tilt_scale = torch.where(free, near_tilt, 1.0)
        pre_z = torch.where(ph == PREGRASP, cruise, torch.full_like(cruise, float("nan")))
        # right next to the cup (no candidate clears it with the full opening, see above) the jaw opens only NEAR_CUP_OPEN
        # past the cube: fully open (1.2 rad) the moving jaw's tip swings ~7 cm out and 5 cm up -- into the cup wall.
        # (Other re-grasps, and every release of a carried cube, keep the full opening of the demos; act() restores it
        # once the cube is grasped.)
        open_q = torch.where((free & narrow) | on_rim, (q_contact + NEAR_CUP_OPEN).clamp(max=GRIPPER_OPEN),
                             torch.full_like(q_contact, GRIPPER_OPEN))
        # a hand that climbs out from next to (or inside) the cup keeps its jaw as it is until it is up: opening it there
        # swings the moving jaw into the cup (stress test: most cup hits when re-planning from a hand at the cup)
        shut_q = torch.where(rise & (hand_cup_d < NEAR_CUP_HAND), q_grip.clamp(min=GRIPPER_CLOSED),
                             torch.full_like(q_grip, float("nan")))

        ids = env_ids
        self.phase[ids] = ph[ids]
        self.q_cmd[ids] = self.robot.data.joint_pos.torch[ids][:, self.arm_ids]  # (after a takeover: from where the arm is)
        self.tilt_goal[ids] = tilt_scale[ids].to(self.tilt_goal.dtype)  # tilt_scale / face_yaw slew there in act()
        self.pre_z[ids] = pre_z[ids]
        self.rise[ids] = rise[ids]
        self.rise_xy[ids] = rise_xy[ids]
        self.yaw_goal[ids] = face_yaw[ids]
        self.face_yaw[ids] = cur[ids]
        self.hold_off_g[ids] = torch.where((held | in_cup | on_rim)[ids, None], hold_g[ids], self.hold_off_g[ids])
        self.timer[ids] = 0.0
        self.cmd[ids] = tcp[ids]
        self.prev_tcp[ids] = tcp[ids]
        self.tcp_speed[ids] = 1.0  # assume moving until measured
        self.lost_steps[ids] = 0
        self.rest_steps[ids] = 0
        self.open_q[ids] = open_q[ids].to(self.open_q.dtype)
        self.shut_q[ids] = shut_q[ids].to(self.shut_q.dtype)
        closed = (ph >= CLOSE) & (ph < RELEASE)
        jaw = torch.where(torch.isnan(shut_q), open_q, shut_q)
        self.grip[ids] = torch.where(closed, torch.full_like(open_q, GRIPPER_CLOSED), jaw)[ids].to(self.grip.dtype)
        return ph[ids]

    def _auto_replan(self):
        """Self-recovery: re-plan when the cube is clearly out of the hand while carrying (missed or
        slipped grasp), or came to rest outside the cup after the release (bounced out onto the desk, or
        onto the rim)."""
        ph = self.phase
        carrying = (ph == LIFT) | (ph == TRANSPORT) | (ph == LOWER)
        after = (ph == RETREAT) | (ph == DONE)
        if not bool((carrying | after).any()):
            self.lost_steps.zero_()
            self.rest_steps.zero_()
            return
        env = self.env
        cube_p, _ = mdp.active_cube_pose_w(env)
        size = mdp.active_cube_size(env)
        q_grip = self.robot.data.joint_pos.torch[:, self.grip_id]
        far = torch.linalg.vector_norm(cube_p - self.tcp_w(), dim=-1) > 0.035
        shut = q_grip < (size - 0.017) / 0.066 - 0.10  # jaw closed past the cube: nothing between the jaws
        lost = carrying & ~mdp.cube_grasped(env) & (far | shut)
        on_desk = (cube_p[:, 2] - env.scene.env_origins[:, 2]) < 0.75 * size + 0.01
        out = after & ~mdp.cube_in_cup(env) & (self._cube_speed() < 0.03)
        fell = out & on_desk
        rests = out & ~on_desk & far  # resting off the desk (on the cup rim), hand away from it
        self.lost_steps = torch.where(lost | fell, self.lost_steps + 1, torch.zeros_like(self.lost_steps))
        self.rest_steps = torch.where(rests, self.rest_steps + 1, torch.zeros_like(self.rest_steps))
        trig = ((lost & (self.lost_steps >= LOST_STEPS)) | (fell & (self.lost_steps >= FELL_OUT_STEPS))
                | (rests & (self.rest_steps >= REST_STEPS)))
        if bool(trig.any()):
            ids = trig.nonzero().flatten()
            self.n_replans[ids] += 1
            self.replan(ids)

    # ------------------------------------------------------------------ policy
    def act(self) -> torch.Tensor:
        env, dev = self.env, self.dev
        self._auto_replan()
        # re-planned jaw yaw / tilt: slew instead of jumping (a jump commands a wrist flip in one step);
        # from reset goal == current, so this is a no-op
        ylim, tlim = YAW_RATE * self.dt, TILT_RATE * self.dt
        self.face_yaw = self.face_yaw + (self.yaw_goal - self.face_yaw).clamp(-ylim, ylim)
        self.tilt_scale = self.tilt_scale + (self.tilt_goal - self.tilt_scale).clamp(-tlim, tlim)
        aligned = ((self.yaw_goal - self.face_yaw).abs() < 1e-4) & ((self.tilt_goal - self.tilt_scale).abs() < 1e-4)
        cube_p, _ = mdp.active_cube_pose_w(env)
        size = mdp.active_cube_size(env)
        cup_p = env.scene["cup"].data.root_pos_w.torch
        desk_z = env.scene.env_origins[:, 2]
        tcp = self.tcp_w()
        ph = self.phase

        # grasp height: cube centre, but keep the fingertips >= 4 mm above the desk
        grasp_z = torch.maximum(cube_p[:, 2], desk_z + (TCP_Z - TIP_Z) + 0.004)
        # carrying: cube bottom 2.5 cm above the rim; TCP sits at cube_centre - (R @ hold_off_g)
        q_now = self._target_quat(self.cmd)
        hold_w = math_utils.quat_apply(q_now, self.hold_off_g)
        carry_tcp_z = desk_z + env.cfg.cup_dims["height"] + size / 2 + 0.025 - hold_w[:, 2]
        over_cup = torch.stack([cup_p[:, 0] - hold_w[:, 0], cup_p[:, 1] - hold_w[:, 1], carry_tcp_z], -1)

        wp = self.cmd.clone()
        # travel above the cup rim so the hand never sweeps through the cup, then descend vertically
        safe_z = torch.maximum(grasp_z + size / 2 + 0.05, desk_z + env.cfg.cup_dims["height"] + 0.03)
        pre_z = torch.where(torch.isnan(self.pre_z), safe_z, self.pre_z)  # (replan may set a lower cruise height)
        m = ph == PREGRASP; wp[m] = torch.stack([cube_p[:, 0], cube_p[:, 1], pre_z], -1)[m]
        m = m & self.rise;  wp[m] = torch.cat([self.rise_xy, pre_z[:, None]], -1)[m]  # climb first
        # (a re-planned DESCEND can start while the jaw yaw / tilt is still slewing: hold the height until aligned --
        # turning the open jaws at grasp height would sweep the cube; from reset ``aligned`` is always true)
        m = ph == DESCEND;  wp[m] = torch.stack([cube_p[:, 0], cube_p[:, 1],
                                                 torch.where(aligned, grasp_z, torch.maximum(grasp_z, self.cmd[:, 2]))], -1)[m]
        m = ph == LIFT;     wp[m] = torch.stack([self.cmd[:, 0], self.cmd[:, 1], carry_tcp_z], -1)[m]
        m = ph == TRANSPORT; wp[m] = over_cup[m]
        m = ph == LOWER;    wp[m] = (over_cup - torch.tensor([0.0, 0.0, LOWER_DZ], device=dev))[m]
        m = ph == RETREAT;  wp[m] = torch.stack([self.cmd[:, 0], self.cmd[:, 1], carry_tcp_z + 0.04], -1)[m]
        wp = wp + self.jitter * (((ph == PREGRASP) & ~self.rise) | (ph == TRANSPORT)).float()[:, None]

        # rate-limited command (half speed for the fine descents), then phase transitions
        lim = self.speed * self.dt * torch.where((ph == DESCEND) | (ph == LOWER), 0.6, 1.0)
        step = (wp - self.cmd).clamp(-lim[:, None], lim[:, None])
        # a re-planned approach (PREGRASP with its own cruise height) moves along the straight line to the waypoint --
        # replan's cup-crossing check assumes that path; the per-axis clamp would cut diagonally first, then along one axis
        line = (wp - self.cmd) * (lim / torch.linalg.vector_norm(wp - self.cmd, dim=-1).clamp_min(1e-9)).clamp(max=1.0)[:, None]
        step = torch.where(((ph == PREGRASP) & ~torch.isnan(self.pre_z))[:, None], line, step)
        self.cmd = self.cmd + step
        err = torch.linalg.vector_norm(tcp - wp, dim=-1)
        at_cmd = torch.linalg.vector_norm(self.cmd - wp, dim=-1) < 1e-4
        self.timer += self.dt
        tol = torch.where(ph == DESCEND, 0.006, 0.012)
        moving = (ph == PREGRASP) | (ph == DESCEND) | (ph == LIFT) | (ph == TRANSPORT) | (ph == LOWER) | (ph == RETREAT)
        # A move is also done once its command has arrived and the TCP has stopped, even short of the
        # target (IK / contact floor: DESCEND never met its 6 mm tolerance). Waiting out the 3 s time box
        # left ~8 s of every 13 s demo motionless, with the next action decided by a hidden timer -- an
        # imitation policy can't learn that (ACT v0 hesitated, and hovered over the cup without releasing).
        v = torch.linalg.vector_norm(tcp - self.prev_tcp, dim=-1) / self.dt
        self.prev_tcp = tcp.clone()
        self.tcp_speed = 0.7 * self.tcp_speed + 0.3 * v
        settled = at_cmd & (self.tcp_speed < 0.01) & (self.timer > 0.2)
        nxt = moving & ((at_cmd & (err < tol)) | settled | (self.timer > 3.0))  # time-box each move
        climbed = nxt & self.rise  # re-planned PREGRASP: the vertical climb is done, now go over the cube
        nxt &= aligned | ((ph != PREGRASP) & (ph != DESCEND))  # (re-planned) don't descend / close while turning
        # jaw moves: done once the jaw has stopped (closed on the cube / fully open)
        jaw_still = self.robot.data.joint_vel.torch[:, self.grip_id].abs() < 0.1
        nxt |= ((ph == CLOSE) | (ph == RELEASE)) & (((self.timer > 0.3) & jaw_still) | (self.timer > 0.7))
        if bool(climbed.any()):
            self.rise &= ~climbed
            nxt &= ~climbed
            self.timer[climbed] = 0.0
        grasping = nxt & (ph == CLOSE)
        if grasping.any():  # remember where the cube sits in the hand (gripper frame)
            _, bq = self._body_pose()
            self.hold_off_g[grasping] = math_utils.quat_apply_inverse(bq, cube_p - tcp)[grasping]
            self.open_q[grasping] = GRIPPER_OPEN  # (near-cup re-grasp) the release over the cup opens as in the demos
        self.phase = torch.where(nxt, ph + 1, ph).clamp(max=DONE)
        self.timer[nxt] = 0.0

        closed = (self.phase >= CLOSE) & (self.phase < RELEASE)
        jaw = torch.where(self.rise & ~torch.isnan(self.shut_q), self.shut_q, self.open_q)  # (from reset: open_q)
        self.grip = torch.where(closed, torch.full_like(self.grip, GRIPPER_CLOSED), jaw)
        step = JOINT_RATE * self.dt
        self.q_cmd = self.q_cmd + (self._ik(self.cmd) - self.q_cmd).clamp(-step, step)
        return torch.cat([self.q_cmd, self.grip[:, None]], dim=-1)

    def _ik(self, tcp_target_w: torch.Tensor) -> torch.Tensor:
        robot = self.robot
        q_des_w = self._target_quat(tcp_target_w)
        body_target_w = tcp_target_w - math_utils.quat_apply(q_des_w, self.tcp_off)
        root_p, root_q = robot.data.root_pos_w.torch, robot.data.root_quat_w.torch
        body_p, body_q = self._body_pose()
        ee_p_b, ee_q_b = math_utils.subtract_frame_transforms(root_p, root_q, body_p, body_q)
        tgt_p_b, tgt_q_b = math_utils.subtract_frame_transforms(root_p, root_q, body_target_w, q_des_w)
        jac = robot.data.body_link_jacobian_w.torch[:, self.jac_idx, :, :][:, :, self.arm_ids].clone()
        rot_b = math_utils.matrix_from_quat(math_utils.quat_inv(root_q))
        jac[:, :3] = torch.bmm(rot_b, jac[:, :3])
        jac[:, 3:] = torch.bmm(rot_b, jac[:, 3:])
        self.ik.set_command(torch.cat([tgt_p_b, tgt_q_b], -1), ee_p_b, ee_q_b)
        return self.ik.compute(ee_p_b, ee_q_b, jac, robot.data.joint_pos.torch[:, self.arm_ids])

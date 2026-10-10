"""SO-101 cube-into-cup environment configs (IsaacLab 3.0, PhysX).

World frame = desk frame: origin at the robot base on the desk surface, +x forward over the desk,
+z up, desk top at z = 0. Calibrated camera/splat poses (``calib.py``) are expressed in this frame.

* ``CubeInCupEnvCfg``     -- state-only (fast; RL and scripted-expert development).
* ``CubeInCupCamEnvCfg``  -- adds the wrist + overhead cameras (640x480, matching the real Jetson rig), the real
                             rig's look (desk texture from the overhead camera, desk lamp, measured object colours;
                             scripts/build_real2sim_assets.py) with appearance randomization around it, or the
                             Gaussian-splat background of the real room when available.

Action = 6 absolute joint-position targets [rad] (5 arm joints + gripper), the same interface LeRobot
uses on the real follower (there in degrees, gripper 0..100).
"""
from __future__ import annotations

import math
import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MassPropertiesCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.sensors.sensors_cfg import OpenCvPinholeDistortionCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.so101 import SO101_CFG  # isort: skip
from isaaclab_tasks.contrib.stack.stack_env_cfg import PhysicsCfg, make_ee_frame_cfg

from . import calib, mdp, units

ASSET_DIR = "/mnt/secondary/so101_r2s/assets"
CUP_USD = os.path.join(ASSET_DIR, "objects", "cup_two_tone.usda")  # so101_r2s2r/cup.py (white outside, blue inside)
REAL2SIM_DIR = os.path.join(ASSET_DIR, "real2sim")  # scripts/build_real2sim_assets.py
BACKDROP_USD = os.path.join(REAL2SIM_DIR, "backdrop.usda")  # visual-only desk-top texture + back wall

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOME_Q = {"shoulder_pan": 0.0, "shoulder_lift": -0.48, "elbow_flex": -0.18, "wrist_flex": 1.4, "wrist_roll": 0.0, "gripper": 0.0}  # TCP ~20 cm up, clear of the cup (RL action origin)
# The user's rest pose (real follower readings, LeRobot units; median of the real episodes' first frames): episodes
# start here, folded with the jaws open sideways, as the real ones do. shoulder_lift rests at -103 deg, past the
# USD's -100 deg limit -> SHOULDER_LIFT_MIN widens it to the follower's calibrated range.
REST_READING = (0.0, -103.0, 95.0, 52.0, 0.0, 20.0)
REST_Q = dict(zip(JOINTS, (float(q) for q in units.lerobot_to_sim(REST_READING))))
SHOULDER_LIFT_MIN = -1.85  # [rad]
GRIPPER_OPEN, GRIPPER_CLOSED = 1.2, -0.1  # [rad]; joint limits are [-0.175, 1.745]

CUBE_SIZES = (0.020, 0.030, 0.040)
CUBE_PROBS = (0.0, 1.0, 0.0)  # the user works with the 30 mm cube only (2026-10-08)
CUBE_MASSES = (0.004, 0.011, 0.024)  # [kg] PLA, ~20% infill (placeholder until weighed)
CUBE_COLOR = (0.82, 0.45, 0.125)  # linear albedo (wood cube), matched to the real frames by scripts/compare_real_sim.py
CUP_DIMS = dict(top_d=0.080, bottom_d=0.060, height=0.095, wall=0.002, floor=0.003)  # placeholder until measured
# The real desk top's footprint (desk frame, CCW; = calib/export_splat.py --desk_poly, the splat's desk-top quad): front
# edge x = -0.12 behind the base, right and back edges from the overhead view, left edge beyond the view. The physics
# desk is larger, so reset_objects keeps the cube and the cup inside this polygon (2026-10-09: ~15% of the cups had
# been spawned past the real right edge, over the splat's shelf clutter).
DESK_POLY = ((-0.12, 0.60), (-0.12, -0.226), (0.386, -0.164), (0.339, 0.60))

# Jaw geometry in the ``gripper`` link frame (measured by scripts/inspect_gripper.py): jaws point
# along -z, fingertips at z = -0.098, fixed jaw inner face at x = -8.8 mm, moving jaw opens toward +x
# (gap ~17 mm at q = 0, ~84 mm at q = 1.0 rad). TCP = grasp point for the 30 mm cube.
FIXED_JAW_X, TIP_Z, TCP_Z = -0.0088, -0.098, -0.080
TCP_OFFSET = (FIXED_JAW_X + 0.015, 0.0, TCP_Z)

_RIGID = RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1,
                                max_depenetration_velocity=5.0, disable_gravity=False)
_PLA = sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.7, restitution=0.0)


def _cube_cfg(name: str, k: int) -> RigidObjectCfg:
    s = CUBE_SIZES[k]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25 + 0.08 * k, 0.0, -0.70), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=sim_utils.CuboidCfg(
            size=(s, s, s),
            rigid_props=_RIGID,
            mass_props=MassPropertiesCfg(mass=CUBE_MASSES[k]),
            collision_props=CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
            physics_material=_PLA,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=CUBE_COLOR, roughness=0.6),
            semantic_tags=[("class", "cube")],
        ),
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=SO101_CFG.spawn.replace(variants={"Robot": "robot", "Sensor": "sensors", "Physics": "physx"}),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=calib.ROBOT_BASE_POS, rot=calib.ROBOT_BASE_QUAT, joint_pos=dict(REST_Q)
        ),
        # SysID (USD-authored) gains everywhere; cap the jaw torque like a real STS3215 grip so a
        # grasp squeezes instead of tunnelling through the cube.
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=JOINTS[:5], stiffness=None, damping=None),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper"], stiffness=None, damping=None,
                                           joint_effort_limit=2.0),
        },
    )
    desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.30, 0.0, -0.025)),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.2, 0.05),
            collision_props=CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.33, 0.19, 0.10), roughness=0.5),
            semantic_tags=[("class", "desk")],
        ),
    )
    floor = AssetBaseCfg(prim_path="/World/Floor", init_state=AssetBaseCfg.InitialStateCfg(pos=(0, 0, -0.75)),
                         spawn=GroundPlaneCfg())
    cube_s = _cube_cfg("CubeS", 0)
    cube_m = _cube_cfg("CubeM", 1)
    cube_l = _cube_cfg("CubeL", 2)
    cup = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cup",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.25, 0.15, 0.001), rot=(0.0, 0.0, 0.0, 1.0)),
        spawn=UsdFileCfg(usd_path=CUP_USD, rigid_props=_RIGID, mass_props=MassPropertiesCfg(mass=0.03),
                         semantic_tags=[("class", "cup")]),
    )
    ee_frame = make_ee_frame_cfg(
        base_prim_path="{ENV_REGEX_NS}/Robot/base",
        target_specs=[("{ENV_REGEX_NS}/Robot/gripper", "tcp", TCP_OFFSET)],
        marker_scale=(0.03, 0.03, 0.03),
    )
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(1.0, 0.92, 0.82), intensity=1500.0))


@configclass
class ActionsCfg:
    joints = base_mdp.JointPositionActionCfg(asset_name="robot", joint_names=JOINTS, scale=1.0,
                                             use_default_offset=False, preserve_order=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Privileged state (RL teacher / scripted expert)."""
        joint_pos = ObsTerm(func=base_mdp.joint_pos)
        joint_vel = ObsTerm(func=base_mdp.joint_vel)
        tcp_pos = ObsTerm(func=mdp.tcp_pos_b)
        cube_pos = ObsTerm(func=mdp.cube_pos_b)
        cube_quat = ObsTerm(func=mdp.cube_quat_w)
        cube_size = ObsTerm(func=mdp.cube_size_obs)
        cup_pos = ObsTerm(func=mdp.cup_pos_b)
        actions = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # NVIDIA's wrist-camera mount is not on the real arm (and sits in front of the real camera's view)
    remove_camera_mount = EventTerm(func=mdp.remove_camera_mount, mode="prestartup")
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_robot = EventTerm(
        func=base_mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )
    reset_objects = EventTerm(
        func=mdp.reset_cube_and_cup, mode="reset",
        # polar ranges around the robot base (r [m], phi [rad] from +x), covering where the user puts them in the
        # real episodes (cube r 0.13-0.27, |phi| < 1.06; cup r 0.19-0.32, |phi| < 1.15)
        params={"cube_rphi": {"r": (0.12, 0.27), "phi": (-1.1, 1.1)},
                "cup_rphi": {"r": (0.15, 0.32), "phi": (-1.2, 1.2)},
                "min_separation": 0.11, "cube_probs": CUBE_PROBS,
                # the hand rests ~7 cm above the desk at x ~0.18 m in REST_Q: no cup there (the user doesn't either)
                "cup_keepout": ((0.17, 0.0), 0.10),
                # both objects on the real desk: centres >= cube half size / cup rim radius + 1.5 cm inside DESK_POLY
                "desk_poly": DESK_POLY, "desk_margin": 0.015, "num_candidates": 64},
    )
    shoulder_lift_range = EventTerm(func=mdp.set_joint_limits, mode="prestartup",
                                    params={"limits_deg": {"shoulder_lift": (math.degrees(SHOULDER_LIFT_MIN), None)}})
    object_friction = EventTerm(
        func=base_mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("cube_m"), "static_friction_range": (0.5, 1.0),
                "dynamic_friction_range": (0.4, 0.9), "restitution_range": (0.0, 0.1), "num_buckets": 16},
    )


@configclass
class RewardsCfg:
    reach = RewTerm(func=mdp.reach_cube, weight=1.0, params={"std": 0.05})
    lifted = RewTerm(func=mdp.cube_lifted, weight=3.0, params={"height": 0.02})
    over_cup = RewTerm(func=mdp.cube_over_cup, weight=6.0, params={"std": 0.05, "lift_height": 0.02})
    in_cup = RewTerm(func=mdp.cube_in_cup_reward, weight=10.0)
    released = RewTerm(func=mdp.placed_and_released, weight=10.0, params={"open_q": 1.0})
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-1e-3)
    joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1e-4)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    success = DoneTerm(func=mdp.task_success, params={"open_q": 1.0})
    cube_dropped = DoneTerm(func=mdp.cube_dropped)
    cup_tipped = DoneTerm(func=mdp.cup_tipped)
    physics_blowup = DoneTerm(func=mdp.physics_blowup)


@configclass
class CubeInCupEnvCfg(ManagerBasedRLEnvCfg):
    scene: SceneCfg = SceneCfg(num_envs=16, env_spacing=1.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands = None
    curriculum = None

    # task constants read by mdp.py
    cube_sizes: tuple = CUBE_SIZES
    default_cube: int = 1
    cup_dims: dict = CUP_DIMS
    gripper_joint_names: list = ["gripper"]
    jaw_fixed_x: float = FIXED_JAW_X  # fixed-jaw inner face, gripper link frame [m]
    jaw_tip_z: float = TIP_Z          # fingertip plane, gripper link frame [m]

    def __post_init__(self):
        self.decimation = 4  # 120 Hz physics -> 30 Hz control (LeRobot records at 30 fps)
        self.episode_length_s = 15.0
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg().isaacsim_physx
        self.viewer.eye = (0.75, -0.55, 0.55)
        self.viewer.lookat = (0.2, 0.0, 0.0)


# The real rig's look (camera env only): one desk lamp (point light fitted to the cast shadows in the real overhead
# frames, build_real2sim_assets.py) + a weak fill; intensities, colours and camera gains matched to the real frames
# (scripts/compare_real_sim.py). CAM_GAIN = per-camera exposure x white balance applied to the rendered RGB.
LAMP_POS = (-1.13, -0.45, 1.78)  # fit: azimuth ~200 deg, elevation ~55 deg from the workspace (distance poorly constrained)
LAMP_RADIUS = 0.06
LAMP_INTENSITY = 7.0e5
LAMP_COLOR = (1.0, 1.0, 1.0)
DOME_INTENSITY = 400.0
CUP_COLORS = dict(outer=(0.80, 0.80, 0.78), inner=(0.35, 0.79, 0.70))  # linear albedo
CUP_INNER_EMISSIVE = 1.0  # x inner colour: the thin plastic cup is translucent and glows inside even in its own shadow
ROBOT_COLOR = (0.90, 0.90, 0.75)  # printed PLA as the overhead camera sees it under the lamp
# wrist: whole-image mean of the real rest-pose frames / the sim's (scripts/compare_real_sim.py, 4 episodes, after the
# wrist_roll fix of 2026-10-09 that turned the sim camera from the room to the desk): the real camera is dark and reddish
CAM_GAIN = dict(scene=(1.0, 1.0, 1.0), wrist=(0.46, 0.35, 0.52))
DESK_ALT_TEXTURES = ("desk_alt_0.png", "desk_alt_1.png", "desk_alt_2.png")

# Splat configuration (SO101_SPLAT=1 + calib/data/splat.json): the Gaussian splat of the real room replaces the plain
# backdrop, and the wrist view gets its own camera model, fitted to the real undistorted wrist frames with
# scripts/compare_real_sim.py (2026-10-09, track A1; /mnt/secondary/so101_r2s/brev/splat_check_v4):
#  * the real Innomaker wrist camera shows the white PLA jaws neutral white (~131) but the dark desk lifted and tinted
#    magenta (black level): CAM_GAIN["wrist"] alone (a pure gain fitted to the whole-image mean) turned the jaws dark
#    purple. Now gain x render + offset per channel, with the jaws as the white reference;
#  * the jaws glow (thin translucent PLA, lit from all sides in the real room): emissive jaw material;
#  * the desk seen from ~10 cm: desk_top_hr.usda (scripts/build_desk_hr.py) = the overhead-camera desk texture upsampled
#    with fine wood detail, plus per-pixel sensor noise in both cameras.
# The non-splat configuration keeps using CAM_GAIN / BACKDROP_USD unchanged.
SPLAT_ON = bool(calib.SPLAT_USD and os.path.exists(calib.SPLAT_USD))
# Fit (6 real episodes, nominal wrist mount): raw sim render (gain 1, jaw glow 0.4) jaw 219/221/208, desk 115/110/64;
# real jaw 128/131/132, desk 66/57/62, darkest 1% ~20/12/12 -> per-channel least squares through (dark, desk, jaw).
# Left over: the desk comes out ~72/66/53 (the overhead-camera texture is yellower than the wrist camera sees it);
# the offset jitter (std 6) and the gain tint cover part of it. The far field (far desk / wall, top half of the rest
# view) renders ~2x brighter than the real frames (the overhead-camera texture bakes in the lamp's glare on the glossy
# desk; the real room is dark during episodes), so the wrist exposure range is widened down to 0.6 (whole-image real
# means are ~0.65-0.7x the nominal sim) instead of the shared 0.8..1.2.
SPLAT_LOOK = dict(
    wrist_gain=(0.50, 0.55, 0.57), wrist_offset=(14.0, 5.0, 16.0), wrist_offset_std=6.0, wrist_noise=2.0,
    wrist_gain_range=(0.6, 1.2), scene_noise=1.5, jaw_emissive=0.4, jaw_emissive_range=(0.25, 0.6),
)


# Render the real lenses' distortion (calib *_DIST, OpenCV k1 k2 p1 p2 k3) in the sim? Off: on 2026-10-09 the RTX
# renderer's opencvPinhole model bent the wrist view the wrong way (jaws pushed outward instead of pulled in by the
# barrel lens), while the undistorted pinhole render matches the undistorted real frames. So real frames are
# cv2.undistort(img, K, DIST)'ed to the same K before a sim-trained policy sees them (dataset conversion + robot side).
RENDER_LENS_DISTORTION = False


def _pinhole(K, w: int, h: int, clipping_range, dist=None) -> sim_utils.PinholeCameraCfg:
    """Pinhole camera that renders exactly K (+ the OpenCV distortion ``dist`` if RENDER_LENS_DISTORTION).
    ``from_intrinsic_matrix`` alone averages fx/fy and drops the principal point (a ~7 px shift for the wrist
    camera); the OpenCV lens model keeps both."""
    spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(K, w, h, clipping_range=clipping_range)
    k1, k2, p1, p2, k3 = (float(c) for c in (dist if dist is not None else (0.0,) * 5))
    spawn.distortion = OpenCvPinholeDistortionCfg(fx=float(K[0]), fy=float(K[4]), cx=float(K[2]), cy=float(K[5]),
                                                  image_size=(w, h), k1=k1, k2=k2, k3=k3, p1=p1, p2=p2,
                                                  apply_lens_distortion=RENDER_LENS_DISTORTION and dist is not None)
    return spawn


@configclass
class CubeInCupCamEnvCfg(CubeInCupEnvCfg):
    """Adds the real rig's two 640x480 cameras (+ splat background if calibrated)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        w, h = calib.IMAGE_SIZE
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/gripper/wrist_cam",
            update_period=0.0, width=w, height=h, data_types=["rgb"],
            spawn=_pinhole(calib.WRIST_K, w, h, clipping_range=(0.005, 5.0), dist=calib.WRIST_DIST),
            offset=CameraCfg.OffsetCfg(pos=calib.WRIST_CAM_POS, rot=calib.WRIST_CAM_QUAT, convention="opengl"),
        )
        self.scene.scene_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/scene_cam",
            update_period=0.0, width=w, height=h, data_types=["rgb"],
            spawn=_pinhole(calib.SCENE_K, w, h, clipping_range=(0.05, 5.0), dist=calib.SCENE_DIST),
            offset=CameraCfg.OffsetCfg(pos=calib.SCENE_CAM_POS, rot=calib.SCENE_CAM_QUAT, convention="ros"),
        )
        if calib.SPLAT_USD and os.path.exists(calib.SPLAT_USD):
            self.scene.background = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Background",
                init_state=AssetBaseCfg.InitialStateCfg(pos=calib.SPLAT_POS, rot=calib.SPLAT_QUAT),
                spawn=UsdFileCfg(usd_path=calib.SPLAT_USD, scale=(calib.SPLAT_SCALE,) * 3),
            )
            # the splat is the room around the desk; its glossy desk top is reconstructed below the surface, so it is
            # cropped out (calib/export_splat.py) and the real desk texture is drawn on a quad over the desk footprint
            self.scene.desk.spawn.visible = False
            desk_top = os.path.join(os.path.dirname(calib.SPLAT_USD), "desk_top.usda")
            if os.path.exists(desk_top.replace(".usda", "_hr.usda")):  # scripts/build_desk_hr.py: detail for the wrist view
                desk_top = desk_top.replace(".usda", "_hr.usda")
            if os.path.exists(desk_top):
                self.scene.backdrop = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Backdrop", spawn=UsdFileCfg(usd_path=desk_top))
        elif os.path.exists(BACKDROP_USD):
            self.scene.backdrop = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Backdrop", spawn=UsdFileCfg(usd_path=BACKDROP_USD))
        # dark room lit by one desk lamp: the dome is only a weak fill and not seen as a background
        self.scene.light.spawn = self.scene.light.spawn.replace(intensity=DOME_INTENSITY, color=LAMP_COLOR,
                                                                visible_in_primary_ray=False)
        self.scene.lamp = AssetBaseCfg(
            prim_path="/World/Lamp", init_state=AssetBaseCfg.InitialStateCfg(pos=LAMP_POS),
            spawn=sim_utils.SphereLightCfg(intensity=LAMP_INTENSITY, radius=LAMP_RADIUS, color=LAMP_COLOR),
        )

        splat = getattr(self.scene, "background", None) is not None
        wrist_px = {"sensor_cfg": SceneEntityCfg("wrist_cam"), "gain": CAM_GAIN["wrist"]}
        scene_px = {"sensor_cfg": SceneEntityCfg("scene_cam"), "gain": CAM_GAIN["scene"]}
        if splat:
            wrist_px.update(gain=SPLAT_LOOK["wrist_gain"], offset=SPLAT_LOOK["wrist_offset"], noise_std=SPLAT_LOOK["wrist_noise"])
            scene_px.update(noise_std=SPLAT_LOOK["scene_noise"])

        @configclass
        class ImagesCfg(ObsGroup):
            wrist = ObsTerm(func=mdp.camera_rgb, params=wrist_px)
            scene = ObsTerm(func=mdp.camera_rgb, params=scene_px)

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        self.observations.images = ImagesCfg()

        # appearance randomization around the real look (the real arm is white; lamp, colours, exposure and the
        # overhead camera's calibration are never exact)
        self.events.recolor_robot = EventTerm(func=mdp.recolor_printed_parts, mode="prestartup", params={"base_color": ROBOT_COLOR})
        if splat:
            self.events.recolor_robot.params["jaw_emissive"] = SPLAT_LOOK["jaw_emissive"]
            self.events.jaw_glow = EventTerm(func=mdp.randomize_jaw_glow, mode="reset",
                                             params={"emissive_range": SPLAT_LOOK["jaw_emissive_range"]})
        self.events.light = EventTerm(func=mdp.randomize_light, mode="reset",
                                      params={"intensity_range": (0.5 * DOME_INTENSITY, 2.0 * DOME_INTENSITY), "warmth_range": (0.8, 1.0)})
        self.events.lamp = EventTerm(func=mdp.randomize_lamp, mode="reset", params={
            "pos": LAMP_POS, "pos_range": ((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2)), "intensity": LAMP_INTENSITY,
            "intensity_range": (0.6, 1.5), "color": LAMP_COLOR, "warmth_range": (-1.0, 1.0)})
        self.events.cube_color = EventTerm(func=mdp.randomize_object_color, mode="reset", params={
            "asset_names": ("cube_s", "cube_m", "cube_l"), "base_color": CUBE_COLOR,
            "hue_range": (-0.02, 0.02), "sat_range": (0.85, 1.15), "val_range": (0.75, 1.25)})
        self.events.cup_outer_color = EventTerm(func=mdp.randomize_object_color, mode="reset", params={
            "asset_names": ("cup",), "material": "cup_outer", "base_color": CUP_COLORS["outer"],
            "hue_range": (-0.05, 0.05), "sat_range": (0.5, 1.5), "val_range": (0.85, 1.1)})
        self.events.cup_inner_color = EventTerm(func=mdp.randomize_object_color, mode="reset", params={
            "asset_names": ("cup",), "material": "cup_inner", "base_color": CUP_COLORS["inner"],
            "hue_range": (-0.04, 0.04), "sat_range": (0.75, 1.25), "val_range": (0.85, 1.15), "emissive": CUP_INNER_EMISSIVE})
        if getattr(self.scene, "backdrop", None) is not None:
            self.events.desk_texture = EventTerm(func=mdp.randomize_desk_texture, mode="reset", params={
                "gain_range": (0.75, 1.25), "tint_std": 0.05, "alt_prob": 0.2, "alternatives": DESK_ALT_TEXTURES})
        for cam, tint in (("scene", 0.03), ("wrist", 0.06)):
            setattr(self.events, f"{cam}_cam_gain", EventTerm(func=mdp.randomize_camera_gain, mode="reset", params={
                "sensor_name": f"{cam}_cam", "gain": CAM_GAIN[cam], "gain_range": (0.8, 1.2), "tint_std": tint}))
        if splat:
            self.events.wrist_cam_gain.params.update(gain=SPLAT_LOOK["wrist_gain"], offset=SPLAT_LOOK["wrist_offset"],
                                                     offset_std=SPLAT_LOOK["wrist_offset_std"],
                                                     gain_range=SPLAT_LOOK["wrist_gain_range"])
        self.events.wrist_cam_jitter = EventTerm(func=mdp.jitter_mounted_camera, mode="reset",
                                                 params={"pos_std": 0.004, "rot_std_deg": 2.0})
        self.events.scene_cam_jitter = EventTerm(func=mdp.jitter_camera, mode="reset",
                                                 params={"sensor_name": "scene_cam", "pos_std": 0.01, "rot_std_deg": 1.0})
        self.num_rerenders_on_reset = 2


@configclass
class CubeInCupRLEnvCfg(CubeInCupEnvCfg):
    """RL variant: arm actions are joint offsets from the home pose (x0.6 rad, clipped to the joint
    limits), which keeps random exploration near a sensible posture; the gripper is binary (sign of the
    6th action: >= 0 open, < 0 close). Everything else matches the base task."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 512
        self.scene.robot.init_state.joint_pos = dict(HOME_Q)  # actions are offsets from this pose
        self.episode_length_s = 12.0
        # Shaping, learned the hard way (see scripts/diagnose_rl.py):
        # * every carry-stage reward requires the cube to be HELD (PPO otherwise learns to fling the
        #   cube upward to collect lift/rim rewards mid-air);
        # * no "open over cup" bonus (it was farmed with the cube resting on the open jaw) -- release is
        #   paid only by the cube actually sitting in the cup with the jaw open;
        # * no success termination (with dense shaping, hovering forever out-earned a terminal success);
        # * throwing the cube or knocking it out of reach ends the episode.
        self.rewards.lifted = RewTerm(func=mdp.grasped_lift, weight=3.0, params={"height": 0.02})
        self.rewards.over_cup = None
        self.rewards.grasp = RewTerm(func=mdp.grasp_reward, weight=1.5)  # bridges reach -> lift
        self.rewards.raise_above_rim = RewTerm(func=mdp.grasped_raise_above_rim, weight=4.0, params={"margin": 0.02, "std": 0.10})
        self.rewards.over_cup_clear = RewTerm(func=mdp.grasped_over_cup, weight=8.0, params={"std": 0.05, "margin": 0.01})
        self.rewards.cup_tipped = RewTerm(func=mdp.cup_tipped, weight=-10.0)
        self.rewards.in_cup.weight = 20.0
        self.rewards.released.weight = 20.0
        self.terminations.success = None
        self.terminations.cube_thrown = DoneTerm(func=mdp.cube_thrown, params={"max_speed": 2.0})
        self.terminations.cube_out = DoneTerm(func=mdp.cube_out_of_workspace, params={"max_r": 0.5})
        # * a continuous gripper offset and unclipped arm targets let the policy jam itself (gripper
        #   commanded far past closed, hand pressed into the desk at the torque cap) where changing the
        #   action changes nothing -> binary gripper (like the expert), clipped arm, penalty on saturation.
        self.actions.joints = base_mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=JOINTS[:5], preserve_order=True, use_default_offset=True, scale=0.6,
            clip={"shoulder_pan": (-1.920, 1.920), "shoulder_lift": (-1.745, 1.745), "elbow_flex": (-1.690, 1.690),
                  "wrist_flex": (-1.658, 1.658), "wrist_roll": (-2.744, 2.841)},
        )
        self.actions.gripper = base_mdp.BinaryJointPositionActionCfg(
            asset_name="robot", joint_names=["gripper"],
            open_command_expr={"gripper": GRIPPER_OPEN}, close_command_expr={"gripper": GRIPPER_CLOSED},
        )
        self.rewards.arm_saturation = RewTerm(func=mdp.arm_drive_saturation, weight=-0.1, params={"deadband": 0.25})

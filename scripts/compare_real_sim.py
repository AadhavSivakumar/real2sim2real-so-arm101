"""Real vs sim look at matched poses (camera env).

Per real episode the sim robot is set to the episode's first joint reading and the cup / cube to the poses detected
in its first overhead frame (assets/real2sim/real_stats.json from scripts/build_real2sim_assets.py); both cameras are
rendered with the nominal appearance (env_cfg constants, DR values overridden) and put side by side with the real
undistorted first frames, with per-channel mean/std of the whole image and of a few regions (desk top, cube top, cup
inside). --dr N adds N resets with the env's appearance randomization as a grid.

  ./lab.sh -p scripts/compare_real_sim.py --episodes 0,3,7,9 --dr 6 --out /tmp/real2sim --viz none

--sweep FILE: JSON list of appearance overrides, each rendered for every episode and summarised (tuning). Keys:
  lamp_pos, lamp_intensity, lamp_color, dome_intensity, cube_color, cup_outer, cup_inner, cup_inner_emissive,
  wall_color, desk_gain, robot_color, gain_scene, gain_wrist, q_offset, offset_wrist (black level, splat config),
  jaw_emissive (splat config), desk_tex (texture file for the desk quad, e.g. ".../desk_albedo.png" vs "..._hr.png").
With SO101_SPLAT=1 the nominal look is env_cfg.SPLAT_LOOK (wrist gain + offset, jaw glow). w_desk_hp = std of the
high-passed (image - Gaussian(6 px)) wrist desk region: fine texture / noise.
--q_offset: added to the real joint readings (LeRobot units: deg, gripper 0..100) before units.lerobot_to_sim, for
  joint-zero offsets between the real arm's calibration and the sim (e.g. "0,0,0,0,90,0").
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SO101-CubeInCup-Cam-v0")
parser.add_argument("--episodes", default="all", help="comma-separated real episode indices, or 'all'")
parser.add_argument("--out", default="/tmp/real2sim")
parser.add_argument("--dr", type=int, default=0, help="number of randomized renders for the DR grid")
parser.add_argument("--sweep", default=None, help="JSON file: list of appearance overrides")
parser.add_argument("--settle", type=int, default=8, help="env steps before capturing")
parser.add_argument("--stats", default=None, help="real_stats.json (default: <REAL2SIM_DIR>/real_stats.json)")
parser.add_argument("--q_offset", default="0,0,0,0,0,0", help="LeRobot-unit offsets added to the real joint readings")
parser.add_argument("--seed", type=int, default=0)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import json
import os

import cv2
import gymnasium as gym
import numpy as np
import torch
from scipy.spatial.transform import Rotation

import so101_r2s2r  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import calib, mdp
from so101_r2s2r import env_cfg as C
from so101_r2s2r.units import JOINTS, lerobot_to_sim

STATS = json.load(open(args.stats or os.path.join(C.REAL2SIM_DIR, "real_stats.json")))
REAL_FRAMES = os.path.join(os.path.dirname(args.stats or os.path.join(C.REAL2SIM_DIR, "x")), "debug", "real_frames")
EPS = [e for e in STATS["per_episode"] if args.episodes == "all" or e["episode"] in {int(v) for v in args.episodes.split(",")}]
NOMINAL = dict(lamp_pos=C.LAMP_POS, lamp_intensity=C.LAMP_INTENSITY, lamp_color=C.LAMP_COLOR, dome_intensity=C.DOME_INTENSITY,
               cube_color=C.CUBE_COLOR, cup_outer=C.CUP_COLORS["outer"], cup_inner=C.CUP_COLORS["inner"], wall_color=None,
               cup_inner_emissive=C.CUP_INNER_EMISSIVE, desk_gain=1.0, robot_color=C.ROBOT_COLOR, gain_scene=C.CAM_GAIN["scene"],
               gain_wrist=C.CAM_GAIN["wrist"], q_offset=[float(v) for v in args.q_offset.split(",")],
               offset_wrist=None, jaw_emissive=None, desk_tex=None)
if C.SPLAT_ON:
    NOMINAL.update(gain_wrist=C.SPLAT_LOOK["wrist_gain"], offset_wrist=C.SPLAT_LOOK["wrist_offset"],
                   jaw_emissive=C.SPLAT_LOOK["jaw_emissive"])
SWEEP = [dict(NOMINAL, **o) for o in (json.load(open(args.sweep)) if args.sweep else [{}])]
PARK = dict(cup=(0.0, -0.58), cube=(0.05, -0.55))  # outside both cameras' views

# overhead camera (pinhole with K: sim renders and real frames are both undistorted to it)
K = np.asarray(calib.SCENE_K, float).reshape(3, 3)
R_CAM = Rotation.from_quat(calib.SCENE_CAM_QUAT).as_matrix()
C_CAM = np.asarray(calib.SCENE_CAM_POS, float)


def project(P):
    Pc = (np.asarray(P, float).reshape(-1, 3) - C_CAM) @ R_CAM
    return (Pc[:, :2] / Pc[:, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def desk_region(ep):
    """Overhead pixels that see the real desk top away from the arm, cup and cube."""
    v, u = np.mgrid[0:480, 0:640]
    d = np.c_[(u.ravel() - K[0, 2]) / K[0, 0], (v.ravel() - K[1, 2]) / K[1, 1], np.ones(u.size)] @ R_CAM.T
    P = C_CAM + (-C_CAM[2] / d[:, 2])[:, None] * d
    x, y = P[:, 0].reshape(480, 640), P[:, 1].reshape(480, 640)
    m = (x > 0.0) & (x < 0.33) & (y > -0.16) & (y < 0.30) & (np.hypot(x, y) > 0.15)
    for key, r in (("cup_xy", 0.10), ("cube_xy", 0.05)):
        if ep.get(key):
            m &= np.hypot(x - ep[key][0], y - ep[key][1]) > r
    return m


def regions(ep):
    out = dict(desk=desk_region(ep))
    if ep.get("cube_xy"):
        u, v = project([[*ep["cube_xy"], C.CUBE_SIZES[1]]])[0].round().astype(int)
        m = np.zeros((480, 640), bool)
        m[v - 4:v + 5, u - 4:u + 5] = True
        out["cube"] = m
    if ep.get("cup_xy"):
        a = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        rim = project(np.c_[ep["cup_xy"][0] + 0.04 * np.cos(a), ep["cup_xy"][1] + 0.04 * np.sin(a), np.full(48, 0.095)])
        c = rim.mean(0)
        m = np.zeros((480, 640), np.uint8)
        cv2.fillConvexPoly(m, np.round(c + 0.6 * (rim - c)).astype(np.int32), 1)
        out["cup_in"] = m > 0
    return out


WRIST_DESK = np.zeros((480, 640), bool)
WRIST_DESK[300:420, 270:410] = True  # between the jaws: the desk right in front of the gripper at rest
WRIST_JAW = np.zeros((480, 640), bool)
WRIST_JAW[440:475, 160:220] = True  # on the fixed jaw (white PLA: a white-balance reference)


def stats(img, m=None):
    a = img.reshape(-1, 3) if m is None else img[m]
    return np.r_[a.mean(0), a.std(0)] if len(a) else np.full(6, np.nan)


def hp_stats(img, m, sigma=6.0):
    """mean / std of the high-passed region (fine texture + noise)."""
    f = img.astype(np.float32)
    return stats(f - cv2.GaussianBlur(f, (0, 0), sigma) + 128.0, m)


def label(img, lines, color=(255, 255, 255)):
    img = np.ascontiguousarray(img)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (6, 18 + 18 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def fmt(s):
    return f"mean {s[0]:.0f}/{s[1]:.0f}/{s[2]:.0f} std {s[3]:.0f}/{s[4]:.0f}/{s[5]:.0f}"


C.RENDER_LENS_DISTORTION = False  # compared with undistorted real frames (same K)
env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = 1
env_cfg.seed = args.seed
env_cfg.episode_length_s = 1000.0
for t in ("success", "cube_dropped", "cup_tipped"):
    setattr(env_cfg.terminations, t, None)
os.makedirs(args.out, exist_ok=True)

with launch_simulation(env_cfg, args):
    import isaaclab.sim as sim_utils
    from pxr import Gf

    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset(seed=args.seed)
    u = env.unwrapped
    dev = u.device
    stage = sim_utils.get_current_stage()
    robot = u.scene["robot"]
    jidx = [robot.joint_names.index(j) for j in JOINTS]
    env0 = u.scene.env_prim_paths[0]

    def set_color(root, material, rgb):
        for sh in mdp._preview_shaders(stage, root, material):
            sh.GetAttribute("inputs:diffuseColor").Set(Gf.Vec3f(*[float(c) for c in rgb]))

    def apply_look(cfg):
        lamp = stage.GetPrimAtPath("/World/Lamp")
        tr = lamp.GetAttribute("xformOp:translate")
        tr.Set(type(tr.Get())(*[float(c) for c in cfg["lamp_pos"]]))
        lamp.GetAttribute("inputs:intensity").Set(float(cfg["lamp_intensity"]))
        lamp.GetAttribute("inputs:color").Set(Gf.Vec3f(*cfg["lamp_color"]))
        dome = stage.GetPrimAtPath("/World/Light")
        dome.GetAttribute("inputs:intensity").Set(float(cfg["dome_intensity"]))
        dome.GetAttribute("inputs:color").Set(Gf.Vec3f(*cfg["lamp_color"]))
        for n in mdp.CUBE_NAMES:
            set_color(mdp._env_prim(u, 0, n), None, cfg["cube_color"])
        set_color(f"{env0}/Cup", "cup_outer", cfg["cup_outer"])
        set_color(f"{env0}/Cup", "cup_inner", cfg["cup_inner"])
        for sh in mdp._preview_shaders(stage, f"{env0}/Cup", "cup_inner"):
            mdp._set_emissive(sh, [cfg["cup_inner_emissive"] * c for c in cfg["cup_inner"]])
        if cfg["wall_color"] is not None:
            set_color(f"{env0}/Backdrop", "WallMat", cfg["wall_color"])
        if cfg["robot_color"] is not None:
            set_color("/World/Looks", "r2s_printed_0", cfg["robot_color"])
        tex = stage.GetPrimAtPath(f"{env0}/Backdrop/Looks/DeskMat/Texture")
        if tex.IsValid():
            from pxr import Sdf
            tex.GetAttribute("inputs:scale").Set(Gf.Vec4f(*([float(cfg["desk_gain"])] * 3), 1.0))
            if cfg["desk_tex"]:
                tex.GetAttribute("inputs:file").Set(Sdf.AssetPath(cfg["desk_tex"]))
            elif hasattr(u, "_r2s_desk_tex"):
                tex.GetAttribute("inputs:file").Set(Sdf.AssetPath(u._r2s_desk_tex))
        for cam in ("scene", "wrist"):
            if hasattr(u, "r2s_cam_gain"):
                u.r2s_cam_gain[f"{cam}_cam"][:] = torch.tensor(cfg[f"gain_{cam}"], device=dev)
        if cfg["offset_wrist"] is not None and "wrist_cam" in getattr(u, "r2s_cam_offset", {}):
            u.r2s_cam_offset["wrist_cam"][:] = torch.tensor(cfg["offset_wrist"], device=dev, dtype=torch.float32)
        if cfg["jaw_emissive"] is not None:
            for sh in mdp._preview_shaders(stage, "/World/Looks/r2s_jaw_0"):
                c = sh.GetAttribute("inputs:diffuseColor").Get()
                mdp._set_emissive(sh, [float(cfg["jaw_emissive"]) * float(v) for v in c])
        cam = u.scene["scene_cam"]
        if hasattr(cam, "_r2s_nominal"):
            cam.set_world_poses(*cam._r2s_nominal, convention="ros")
        # undo mdp.jitter_mounted_camera (the wrist camera's mount DR): without this every comparison rendered the
        # wrist view from a randomly displaced camera (4 mm / 2 deg std), which moved and scaled the jaws and the cube
        from pxr import UsdGeom
        for path, (t0, q0) in getattr(u, "_r2s_mount_nominal", {}).items():
            ops = {op.GetOpName(): op for op in UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()}
            ops["xformOp:translate"].Set(t0)
            ops["xformOp:orient"].Set(q0)

    def set_episode(ep, q_offset):
        """Robot at the episode's first joint reading, cup/cube at the detected poses; returns the hold action."""
        q = torch.tensor(lerobot_to_sim(np.asarray(ep["state0"]) + np.asarray(q_offset)), device=dev)
        jp = robot.data.joint_pos.torch.clone()
        jp[0, jidx] = q
        robot.write_joint_position_to_sim_index(position=jp)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp))
        robot.set_joint_position_target_index(target=jp)
        mdp.active_cube_idx(u)[:] = 1
        o = u.scene.env_origins[0]
        zero = torch.zeros(1, 6, device=dev)
        cup_xy = ep.get("cup_xy") or PARK["cup"]
        pose = torch.tensor([[cup_xy[0], cup_xy[1], 0.001, 0, 0, 0, 1]], device=dev, dtype=torch.float32)
        pose[:, :3] += o
        u.scene["cup"].write_root_pose_to_sim_index(root_pose=pose)
        u.scene["cup"].write_root_velocity_to_sim_index(root_velocity=zero)
        for k, n in enumerate(mdp.CUBE_NAMES):
            if k == 1:
                xy, yaw = ep.get("cube_xy") or PARK["cube"], -float(ep.get("cube_yaw") or 0.0)
                p = [xy[0], xy[1], C.CUBE_SIZES[1] / 2 + 0.001]
            else:
                p, yaw = [0.25 + 0.08 * k, 0.0, -0.70], 0.0
            pose = torch.tensor([[*p, 0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]], device=dev, dtype=torch.float32)
            pose[:, :3] += o
            u.scene[n].write_root_pose_to_sim_index(root_pose=pose)
            u.scene[n].write_root_velocity_to_sim_index(root_velocity=zero)
        return q[None]

    def capture(action):
        o = None
        for _ in range(args.settle):
            o, *_ = env.step(action)
        return {c: o["images"][c][0, ..., :3].to(torch.uint8).cpu().numpy() for c in ("scene", "wrist")}

    real = {e["episode"]: {c: cv2.cvtColor(cv2.imread(os.path.join(REAL_FRAMES, f"ep{e['episode']:03d}_{c}.png")), cv2.COLOR_BGR2RGB)
                           for c in ("scene", "wrist")} for e in EPS}
    keys = ["scene", "wrist", "desk", "cube", "cup_in", "w_desk", "w_jaw", "w_desk_hp"]
    summary = []
    for k, cfg in enumerate(SWEEP):
        acc = {key: [] for key in keys}
        for ep in EPS:
            e = ep["episode"]
            env.reset()
            apply_look(cfg)
            q_cmd = set_episode(ep, cfg["q_offset"])
            sim = capture(q_cmd)
            q_err = float((robot.data.joint_pos.torch[0, jidx] - q_cmd[0]).abs().max())
            if q_err > 0.05:
                print(f"[cmp] WARNING ep{e}: arm off its target by {q_err:.3f} rad: target "
                      f"{np.round(q_cmd[0].cpu().numpy(), 3).tolist()} actual "
                      f"{np.round(robot.data.joint_pos.torch[0, jidx].cpu().numpy(), 3).tolist()}", flush=True)
            reg = regions(ep)
            rows = []
            for cam in ("scene", "wrist"):
                sr, ss = stats(real[e][cam]), stats(sim[cam])
                acc[cam].append((sr, ss))
                rows.append(np.concatenate([label(real[e][cam], [f"REAL ep{e} {cam}", fmt(sr)]),
                                            label(sim[cam], [f"SIM {cam}", fmt(ss)])], 1))
            for key in ("desk", "cube", "cup_in"):
                if key in reg:
                    acc[key].append((stats(real[e]["scene"], reg[key]), stats(sim["scene"], reg[key])))
            acc["w_desk"].append((stats(real[e]["wrist"], WRIST_DESK), stats(sim["wrist"], WRIST_DESK)))
            acc["w_jaw"].append((stats(real[e]["wrist"], WRIST_JAW), stats(sim["wrist"], WRIST_JAW)))
            acc["w_desk_hp"].append((hp_stats(real[e]["wrist"], WRIST_DESK), hp_stats(sim["wrist"], WRIST_DESK)))
            if args.sweep:
                cv2.imwrite(os.path.join(args.out, f"sim_ep{e:03d}_c{k}_wrist.png"), cv2.cvtColor(sim["wrist"], cv2.COLOR_RGB2BGR))
            if k == 0 or args.sweep:
                tag = f"_c{k}" if args.sweep else ""
                cv2.imwrite(os.path.join(args.out, f"cmp_ep{e:03d}{tag}.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
        line = {key: (np.nanmean([a for a, _ in v], 0), np.nanmean([b for _, b in v], 0)) for key, v in acc.items() if v}
        summary.append(dict(config=k, overrides={kk: vv for kk, vv in cfg.items() if NOMINAL.get(kk) != vv},
                            **{key: dict(real=np.round(r, 1).tolist(), sim=np.round(s, 1).tolist()) for key, (r, s) in line.items()}))
        print(f"[cmp] config {k}: {summary[-1]['overrides']}", flush=True)
        for key, (r, s) in line.items():
            print(f"[cmp]   {key:6s} real {fmt(r)} | sim {fmt(s)} | sim/real mean {np.round(s[:3] / np.maximum(r[:3], 1), 2).tolist()}",
                  flush=True)
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)

    if args.dr:
        tiles = []
        for i in range(args.dr):
            obs, _ = env.reset()
            q = robot.data.joint_pos.torch[:, jidx].clone()
            img = capture(q)
            tiles.append(np.concatenate([label(img["scene"], [f"DR {i} scene"]), label(img["wrist"], [f"DR {i} wrist"])], 1))
        tiles = [cv2.resize(t, (640, 240), interpolation=cv2.INTER_AREA) for t in tiles]
        cols = 2
        rows = [np.concatenate(tiles[r:r + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[r:r + cols])), 1)
                for r in range(0, len(tiles), cols)]
        cv2.imwrite(os.path.join(args.out, "dr_grid.png"), cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR))
        print(f"[cmp] DR grid of {args.dr} -> {args.out}/dr_grid.png", flush=True)
    print(f"[cmp] done -> {args.out}", flush=True)
    env.close()

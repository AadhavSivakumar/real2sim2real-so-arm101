"""Measure the SO-101 jaw geometry in the ``gripper`` link frame at several jaw openings.

Reports, in the fingertip band, the fixed jaw's inner face, the moving jaw's inner face and the gap.
Run: ./lab.sh -p scripts/inspect_gripper.py --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import numpy as np
import torch

import so101_r2s2r  # noqa: F401
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import resolve_task_config

TASK = "SO101-CubeInCup-v0"
env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = 1


def mesh_points(stage, root_path):
    """All collision-mesh vertices under ``root_path``, expressed in ``root_path``'s own frame."""
    from pxr import Usd, UsdGeom

    root = stage.GetPrimAtPath(root_path)
    w2r = np.array(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse())
    pts = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.IsA(UsdGeom.Mesh) and "/collisions" in str(prim.GetPath()):
            p = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
            m = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())) @ w2r
            pts.append(np.c_[p, np.ones(len(p))] @ m)  # USD matrices are row-vector convention
    return np.concatenate(pts)[:, :3]


with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    base = "/World/envs/env_0/Robot"
    fixed = mesh_points(stage, f"{base}/gripper")
    jaw = mesh_points(stage, f"{base}/moving_jaw_so101_v1")
    print(f"[grip] fixed-jaw pts {len(fixed)}, moving-jaw pts {len(jaw)}")
    band = lambda p: p[(p[:, 2] < -0.065) & (p[:, 2] > -0.100)]
    fb = band(fixed)
    print(f"[grip] fixed jaw (gripper frame, tip band z in [-0.10,-0.065]): x[{fb[:, 0].min():.4f},{fb[:, 0].max():.4f}] "
          f"y[{fb[:, 1].min():.4f},{fb[:, 1].max():.4f}]")

    robot = u.scene["robot"]
    gi, ji = robot.body_names.index("gripper"), robot.body_names.index("moving_jaw_so101_v1")
    gid = robot.joint_names.index("gripper")
    for q in (-0.17, 0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.745):
        jp = robot.data.joint_pos.torch.clone()
        jp[:, gid] = q
        robot.write_joint_position_to_sim_index(position=jp)
        robot.set_joint_position_target_index(target=jp)
        for _ in range(3):
            u.sim.step(render=False)
            u.scene.update(u.physics_dt)
        d = robot.data
        p_rel, q_rel = math_utils.subtract_frame_transforms(
            d.body_link_pos_w.torch[:, gi], d.body_link_quat_w.torch[:, gi],
            d.body_link_pos_w.torch[:, ji], d.body_link_quat_w.torch[:, ji])
        R = math_utils.matrix_from_quat(q_rel)[0].cpu().numpy()
        jw = jaw @ R.T + p_rel[0].cpu().numpy()
        jb = band(jw)
        if len(jb) == 0:
            print(f"[grip] q={q:+.2f}: moving jaw has no points in the tip band (tip z range {jw[:, 2].min():.3f}..{jw[:, 2].max():.3f})")
            continue
        print(f"[grip] q={q:+.2f} (actual {float(d.joint_pos.torch[0, gid]):+.2f}): moving jaw tip band x[{jb[:, 0].min():.4f},{jb[:, 0].max():.4f}] "
              f"z[{jb[:, 2].min():.4f},{jb[:, 2].max():.4f}]  gap(x) = {jb[:, 0].min() - fb[:, 0].max():.4f}")
    env.close()

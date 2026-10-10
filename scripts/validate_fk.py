"""Check calib/fk.py against Isaac's physics: fingertip ('gripperframe') position in the base frame
for random joint configurations.

Run: ./lab.sh -p scripts/validate_fk.py --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import os

import gymnasium as gym
import numpy as np
import torch

import so101_r2s2r  # noqa: F401
from isaaclab.utils import math as math_utils
from isaaclab_tasks.utils import resolve_task_config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "calib"))
import fk  # noqa: E402

TASK = "SO101-CubeInCup-v0"
env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = 1
with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    robot = u.scene["robot"]
    gi, bi = robot.body_names.index("gripper"), robot.body_names.index("base")
    tip_in_gripper = torch.tensor([[-0.0079, -0.000218, -0.0981]], device=u.device)
    rng = np.random.default_rng(0)
    errs = []
    tests = [np.zeros(6)] + [np.r_[np.eye(5)[i] * 0.5, 0.0] for i in range(5)] + [np.r_[rng.uniform(-1.2, 1.2, 5), 0.0] for _ in range(24)]
    P_fk, P_w = [], []
    for q in tests:
        jp = torch.tensor([q], dtype=torch.float32, device=u.device)
        robot.write_joint_position_to_sim_index(position=jp)
        robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp))
        robot.set_joint_position_target_index(target=jp)
        u.sim.forward()
        u.scene.update(u.physics_dt)
        d = robot.data
        tip_w = d.body_link_pos_w.torch[:, gi] + math_utils.quat_apply(d.body_link_quat_w.torch[:, gi], tip_in_gripper)
        tip_b = math_utils.subtract_frame_transforms(d.body_link_pos_w.torch[:, bi], d.body_link_quat_w.torch[:, bi], tip_w)[0][0]
        q_act = d.joint_pos.torch[0, :5].cpu().numpy()
        f = fk.fingertip_in_base(q_act)
        P_fk.append(f)
        P_w.append((tip_w[0] - u.scene.env_origins[0]).cpu().numpy())
    # rigid fit  P_w ~= R @ P_fk + t  (URDF base_link -> desk/world frame)
    A, B = np.array(P_fk), np.array(P_w)
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    res = np.linalg.norm(A @ R.T + t - B, axis=1)
    print("[fk] R_world_urdf =", np.round(R, 5).tolist())
    print("[fk] t_world_urdf =", np.round(t, 5).tolist())
    print(f"[fk] residual over {len(A)} poses: mean {res.mean() * 1000:.3f} mm, max {res.max() * 1000:.3f} mm")
    env.close()

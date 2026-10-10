"""Run an exported RL policy (TorchScript) and trace what happens around the cup.

Run: ./lab.sh -p scripts/diagnose_rl.py --policy <run>/exported/policy.pt --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=360)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import so101_r2s2r  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import mdp

TASK = "SO101-CubeInCup-RL-v0"
env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = args.num_envs
with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg)
    obs, _ = env.reset(seed=11)
    u = env.unwrapped
    policy = torch.jit.load(args.policy, map_location=u.device).eval()
    stats = dict(over_open=torch.zeros(u.num_envs, device=u.device), in_cup=torch.zeros(u.num_envs, device=u.device))
    h = u.cfg.cup_dims["height"]
    for t in range(args.steps):
        with torch.inference_mode():
            a = policy(obs["policy"])
        cube, _ = mdp.active_cube_pose_w(u)
        size = mdp.active_cube_size(u)
        cup = u.scene["cup"].data.root_pos_w.torch
        bottom = cube[:, 2] - u.scene.env_origins[:, 2] - size / 2
        dxy = torch.linalg.vector_norm(cube[:, :2] - cup[:, :2], dim=-1)
        tcp_d = torch.linalg.vector_norm(mdp.tcp_pos_w(u) - cube, dim=-1)
        gq = mdp.gripper_q(u)
        oo = mdp.open_over_cup(u, margin=0.01) > 0
        stats["over_open"] += oo.float()
        stats["in_cup"] += mdp.cube_in_cup(u).float()
        if t % 10 == 0 or (oo[0] and t % 2 == 0):
            print(f"[diag] t={t:3d} env0: cube{int(size[0] * 1000)}mm bottom={bottom[0] * 1000:6.1f}mm (rim {h * 1000:.0f}) "
                  f"xy_to_cup={dxy[0] * 1000:5.1f}mm grip_q={gq[0]:+.2f} tcp-cube={tcp_d[0] * 1000:5.1f}mm "
                  f"cube_vz={u.scene[mdp.CUBE_NAMES[int(mdp.active_cube_idx(u)[0])]].data.root_lin_vel_w.torch[0, 2]:+.2f} "
                  f"over_open={bool(oo[0])} in_cup={bool(mdp.cube_in_cup(u)[0])}")
        obs, _, term, trunc, _ = env.step(a)
    print("[diag] per-env steps with (over cup & opened):", stats["over_open"].int().tolist())
    print("[diag] per-env steps with cube in cup:        ", stats["in_cup"].int().tolist())
    env.close()

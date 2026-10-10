"""Sanity-check RL reward terms along scripted-expert episodes: each stage term should fire on a good
trajectory, and the anti-exploit terminations (throw / out-of-reach) should not.

Run: ./lab.sh -p scripts/check_rewards.py --num_envs 32 --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=420)
parser.add_argument("--force_closed", action="store_true", help="keep the jaw shut: no real grasp may register")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import so101_r2s2r  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import mdp
from so101_r2s2r.env_cfg import GRIPPER_CLOSED
from so101_r2s2r.expert import CubeInCupExpert

TASK = "SO101-CubeInCup-v0"
env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = args.num_envs
env_cfg.terminations.success = None
env_cfg.episode_length_s = 30.0
with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg)
    env.reset(seed=3)
    u = env.unwrapped
    ex = CubeInCupExpert(u)
    ex.reset()
    terms = {
        "grasped": lambda: mdp.cube_grasped(u).float(),
        "grasped_lift": lambda: mdp.grasped_lift(u, 0.02),
        "grasped_raise_above_rim": lambda: mdp.grasped_raise_above_rim(u, 0.02, 0.05),
        "grasped_over_cup": lambda: mdp.grasped_over_cup(u, 0.05, 0.01),
        "in_cup": lambda: mdp.cube_in_cup(u).float(),
        "released": lambda: mdp.placed_and_released(u, 1.0),
        "cube_thrown": lambda: mdp.cube_thrown(u, 2.0).float(),
        "cube_out": lambda: mdp.cube_out_of_workspace(u, 0.5).float(),
    }
    fired = {k: torch.zeros(u.num_envs, device=u.device) for k in terms}
    for t in range(args.steps):
        a = ex.act()
        if args.force_closed:
            a[:, 5] = GRIPPER_CLOSED
        env.step(a)
        for k, f in terms.items():
            fired[k] = torch.maximum(fired[k], (f() > 0.5).float())
    for k in terms:
        print(f"[rew] {k:24s} fired in {int(fired[k].sum())}/{u.num_envs} episodes")
    env.close()

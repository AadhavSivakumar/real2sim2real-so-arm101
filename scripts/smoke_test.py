"""Build the cube-in-cup env, hold the home pose for a few seconds, report rewards/terminations,
and (camera variant) save the first frames of both cameras.

Run: ./lab.sh -p scripts/smoke_test.py --task SO101-CubeInCup-v0 --num_envs 4 --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SO101-CubeInCup-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=90)
parser.add_argument("--out", default="/mnt/secondary/so101_r2s/smoke")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import os

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import so101_r2s2r  # noqa: F401  (registers tasks)
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import mdp
from so101_r2s2r.env_cfg import HOME_Q, JOINTS

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = args.num_envs

with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    obs, _ = env.reset()
    u = env.unwrapped
    print("[smoke] obs groups:", {k: (tuple(v.shape) if hasattr(v, "shape") else {kk: tuple(vv.shape) for kk, vv in v.items()}) for k, v in obs.items()})
    print("[smoke] action space:", env.action_space.shape)
    print("[smoke] active cube idx:", mdp.active_cube_idx(u).tolist())
    print("[smoke] cube pos (base):", [[round(x, 3) for x in r] for r in mdp.cube_pos_b(u).tolist()])
    print("[smoke] cup pos (base):", [[round(x, 3) for x in r] for r in mdp.cup_pos_b(u).tolist()])
    print("[smoke] tcp pos (base):", [[round(x, 3) for x in r] for r in mdp.tcp_pos_b(u).tolist()])
    hold = torch.tensor([[HOME_Q[j] for j in JOINTS]], device=u.device).repeat(u.num_envs, 1)
    rew_sum = torch.zeros(u.num_envs, device=u.device)
    n_term = 0
    for t in range(args.steps):
        obs, rew, term, trunc, info = env.step(hold)
        rew_sum += rew
        n_term += int(term.sum())
    print("[smoke] mean return:", round(float(rew_sum.mean()), 3), "terminations:", n_term)
    print("[smoke] reward terms (last step):", {k: round(float(v), 4) for k, v in info.get("log", {}).items() if "Episode_Reward" in k})
    print("[smoke] cube z after settle:", [round(float(z), 4) for z in mdp.active_cube_pose_w(u)[0][:, 2]])
    print("[smoke] cup upright:", mdp.cup_upright(u).tolist())
    if "images" in obs:
        os.makedirs(args.out, exist_ok=True)
        for k, v in obs["images"].items():
            img = v[0].cpu().numpy().astype(np.uint8)
            Image.fromarray(img[..., :3]).save(f"{args.out}/{k}.png")
            print("[smoke] saved", f"{args.out}/{k}.png", img.shape)
    env.close()

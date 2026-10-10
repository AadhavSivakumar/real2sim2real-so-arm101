"""RL env (512 envs, state only): many resets on the GPU; every active cube / cup inside the desk polygon?

Run (r2s-sim): cd ~/so101_r2s2r && ./lab.sh -p ~/r2s/desk_check/rl_reset_check.py --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SO101-CubeInCup-RL-v0")
parser.add_argument("--resets", type=int, default=40)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import numpy as np
import torch

import so101_r2s2r  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import env_cfg as ecfg, mdp

cfg, _ = resolve_task_config(args.task, "")
with launch_simulation(cfg, args):
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    print(f"[rl] {args.task} num_envs={u.num_envs} params={cfg.events.reset_objects.params}", flush=True)
    cube_d, cup_d, xs = [], [], []
    for k in range(args.resets):
        env.reset(seed=k)
        o = u.scene.env_origins
        cube = mdp.active_cube_pose_w(u)[0] - o
        cup = u.scene["cup"].data.root_pos_w.torch - o
        cube_d.append((mdp._inside_convex(cube[:, :2], ecfg.DESK_POLY) - mdp.active_cube_size(u) / 2).cpu().numpy())
        cup_d.append(mdp._inside_convex(cup[:, :2], ecfg.DESK_POLY).cpu().numpy())
        xs.append(cup[:, :2].cpu().numpy())
    cube_d, cup_d, xs = np.concatenate(cube_d), np.concatenate(cup_d), np.concatenate(xs)
    print(f"[rl] {len(cube_d)} spawns: cube (centre - half size) min {cube_d.min()*100:.2f} cm inside (need 1.5); "
          f"cup centre min {cup_d.min()*100:.2f} cm inside (need 5.5); cup y<0 {np.mean(xs[:, 1] < 0):.3f}; "
          f"spawn stats {getattr(u, 'r2s_spawn_stats', None)}", flush=True)
    print("[rl] done", flush=True)
    env.close()

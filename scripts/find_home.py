"""Find a joint-space home pose that puts the TCP at a given point with the jaws pointing down.

Run: ./lab.sh -p scripts/find_home.py --target 0.12 0.0 0.20 --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--target", type=float, nargs=3, default=[0.12, 0.0, 0.20])
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import so101_r2s2r  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r.env_cfg import JOINTS
from so101_r2s2r.expert import CubeInCupExpert

TASK = "SO101-CubeInCup-v0"
env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = 1
env_cfg.events.reset_objects.params["cube_rphi"] = {"r": (0.40, 0.40), "phi": (1.4, 1.4)}  # keep objects out of the way
env_cfg.events.reset_objects.params["cup_rphi"] = {"r": (0.40, 0.40), "phi": (-1.4, -1.4)}
env_cfg.events.reset_objects.params["desk_poly"] = None  # the cup is parked past the real desk on purpose

with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    u = env.unwrapped
    ex = CubeInCupExpert(u)
    ex.reset()
    target = torch.tensor([args.target], device=u.device) + u.scene.env_origins[:, :3]
    for t in range(120):
        q_arm = ex._ik(target)
        env.step(torch.cat([q_arm, torch.zeros(1, 1, device=u.device)], -1))
    q = u.scene["robot"].data.joint_pos.torch[0].tolist()
    print("[home] tcp:", [round(v, 3) for v in (ex.tcp_w() - u.scene.env_origins[:, :3])[0].tolist()])
    print("[home] HOME_Q = {" + ", ".join(f'"{j}": {v:.3f}' for j, v in zip(JOINTS[:5], q[:5])) + ', "gripper": 0.0}')
    env.close()

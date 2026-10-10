"""Closed-loop evaluation of a LeRobot policy (served by scripts/policy_server.py) in the camera env.

The policy sees exactly what the real robot would give it: LeRobot-unit joint state + 640x480 wrist and
overhead RGB frames; its actions are converted back to sim joint targets.
Run: ./lab.sh -p scripts/eval_policy.py --episodes 50 --video 4 --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SO101-CubeInCup-Cam-v0")
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--port", type=int, default=6100)
parser.add_argument("--video", type=int, default=0, help="save the first N episodes to eval.mp4")
parser.add_argument("--out", default="/mnt/secondary/so101_r2s/eval")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--task_text", default="Put the cube in the cup")
parser.add_argument("--settle_steps", type=int, default=1,
                    help="hold steps after reset: the first post-reset camera frame is a stale pre-reset render "
                         "(record_demos.py drops it from the data too)")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import math
import os
import subprocess
from collections import Counter
from multiprocessing.connection import Client

import gymnasium as gym
import numpy as np
import torch

import so101_r2s2r  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import mdp, units

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = 1
env_cfg.seed = args.seed
env_cfg.episode_length_s = 20.0
os.makedirs(args.out, exist_ok=True)

with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    u = env.unwrapped
    conn = Client(("127.0.0.1", args.port), authkey=b"so101")
    writer = None
    if args.video:
        writer = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480",
                                   "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                   "-pix_fmt", "yuv420p", os.path.join(args.out, "eval.mp4")], stdin=subprocess.PIPE)
    results, by_size, ok_size = Counter(), Counter(), Counter()
    by_r, ok_r = Counter(), Counter()  # success by the cup's distance from the base
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        for _ in range(args.settle_steps):
            obs, *_ = env.step(u.scene["robot"].data.joint_pos.torch.clone())
        conn.send({"cmd": "reset"}); conn.recv()
        size = int(round(float(mdp.active_cube_size(u)[0]) * 1000))
        cup0 = u.scene["cup"].data.root_pos_w.torch[0] - u.scene.env_origins[0]
        r_bin = min(int(float(cup0[:2].norm()) / 0.05), 6) * 5  # 5 cm bins (cm)
        while True:
            q = u.scene["robot"].data.joint_pos.torch[0].cpu().numpy()
            wrist = obs["images"]["wrist"][0, ..., :3].to(torch.uint8).cpu().numpy()
            scene = obs["images"]["scene"][0, ..., :3].to(torch.uint8).cpu().numpy()
            conn.send({"cmd": "act", "task": args.task_text, "obs": {
                "observation.state": units.sim_to_lerobot(q),
                "observation.images.wrist": wrist, "observation.images.scene": scene}})
            a = units.lerobot_to_sim(conn.recv())
            if writer is not None and ep < args.video:
                writer.stdin.write(np.concatenate([scene, wrist], axis=1).tobytes())
            # the state the episode ends in (env.step resets an ended episode before returning)
            cup_p = u.scene["cup"].data.root_pos_w.torch[0] - u.scene.env_origins[0]
            cube_p = mdp.active_cube_pose_w(u)[0][0] - u.scene.env_origins[0]
            pre = (float(cup_p[:2].norm()), math.degrees(float(torch.atan2(cup_p[1], cup_p[0]))), float(cube_p[:2].norm()),
                   float(cube_p[2]), float((cube_p[:2] - cup_p[:2]).norm()), bool(mdp.cube_grasped(u)[0]),
                   float(mdp.gripper_q(u)[0]))
            obs, rew, term, trunc, info = env.step(torch.tensor(a, device=u.device)[None])
            if bool(term[0] | trunc[0]):
                tm = u.termination_manager
                outcome = next((n for n in ("success", "cube_dropped", "cup_tipped") if bool(tm.get_term(n)[0])), "time_out")
                break
        results[outcome] += 1
        by_size[size] += 1
        ok_size[size] += outcome == "success"
        by_r[r_bin] += 1
        ok_r[r_bin] += outcome == "success"
        print(f"[eval] episode {ep + 1}/{args.episodes}: {outcome} (cube {size} mm)", flush=True)
        if outcome != "success":
            print(f"[why] {outcome}: cup r {pre[0]:.3f} phi {pre[1]:+.0f} deg | cube r {pre[2]:.3f} z {pre[3]:.3f}, "
                  f"{1000 * pre[4]:.0f} mm from the cup axis, grasped {int(pre[5])}, gripper {pre[6]:.2f} rad", flush=True)
    if writer is not None:
        writer.stdin.close(); writer.wait()
    print(f"[eval] success {results['success']}/{args.episodes} = {results['success'] / args.episodes:.0%}; outcomes {dict(results)}")
    print("[eval] by cube size:", {f"{k}mm": f"{ok_size[k]}/{by_size[k]}" for k in sorted(by_size)})
    print("[eval] by cup distance:", {f"{k}-{k + 5}cm": f"{ok_r[k]}/{by_r[k]}" for k in sorted(by_r)})
    env.close()

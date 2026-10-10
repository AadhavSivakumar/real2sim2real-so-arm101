"""Does the RTX renderer's OpenCV lens model match OpenCV? (env_cfg.RENDER_LENS_DISTORTION)

Renders the wrist and overhead views twice -- the env's distorted cameras and undistorted twins at the same poses --
then distorts the undistorted renders with cv2 (K, DIST from calib) and compares them with the renderer's distorted
ones: mean absolute difference and the edge offset (phase correlation) in the centre and the corners.

  ./lab.sh -p scripts/check_lens_distortion.py --out /tmp/lens --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SO101-CubeInCup-Cam-v0")
parser.add_argument("--out", default="/tmp/lens")
parser.add_argument("--steps", type=int, default=20)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import copy
import os

import cv2
import gymnasium as gym
import numpy as np
import torch

with launch_simulation(args):
    import so101_r2s2r  # noqa: F401  (registers the tasks)
    from isaaclab_tasks.utils import parse_env_cfg
    from so101_r2s2r import calib

    env_cfg = parse_env_cfg(args.task, num_envs=1)
    for cam in ("wrist_cam", "scene_cam"):
        twin = copy.deepcopy(getattr(env_cfg.scene, cam))
        twin.prim_path = twin.prim_path + "_u"
        twin.spawn.distortion.apply_lens_distortion = False
        setattr(env_cfg.scene, cam + "_u", twin)
    env_cfg.events.scene_cam_jitter = None  # the twin isn't jittered
    env = gym.make(args.task, cfg=env_cfg)
    env.reset(seed=0)
    u = env.unwrapped
    act = torch.tensor([[u.scene["robot"].data.joint_pos.torch[0, j].item() for j in range(6)]], device=u.device)
    for _ in range(args.steps):
        env.step(act)
    os.makedirs(args.out, exist_ok=True)
    for cam, K, D in (("wrist_cam", calib.WRIST_K, calib.WRIST_DIST), ("scene_cam", calib.SCENE_K, calib.SCENE_DIST)):
        K = np.asarray(K, float).reshape(3, 3)
        D = np.asarray(D, float)
        dist = u.scene[cam].data.output["rgb"].torch[0, ..., :3].cpu().numpy()[..., ::-1].copy()
        und = u.scene[cam + "_u"].data.output["rgb"].torch[0, ..., :3].cpu().numpy()[..., ::-1].copy()
        h, w = dist.shape[:2]
        # cv2 distortion of the undistorted render: for each distorted pixel, where it comes from in the pinhole image
        v, uu = np.mgrid[0:h, 0:w].astype(np.float32)
        pts = cv2.undistortPoints(np.stack([uu.ravel(), v.ravel()], -1)[:, None, :], K, D, P=K).reshape(h, w, 2)
        ref = cv2.remap(und, pts[..., 0], pts[..., 1], cv2.INTER_LINEAR)
        g1, g2 = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY).astype(np.float32) for x in (dist, ref))
        out = [f"[lens] {cam}: mean |isaac - cv2| {np.abs(g1 - g2).mean():.1f} (undistorted vs distorted {np.abs(cv2.cvtColor(und, cv2.COLOR_BGR2GRAY).astype(np.float32) - g1).mean():.1f})"]
        for name, (y0, x0) in dict(centre=(h // 2 - 64, w // 2 - 64), top_left=(8, 8), bottom_right=(h - 136, w - 136)).items():
            a, b = g1[y0:y0 + 128, x0:x0 + 128], g2[y0:y0 + 128, x0:x0 + 128]
            (dx, dy), resp = cv2.phaseCorrelate(a, b)
            out.append(f"{name} shift ({dx:+.1f}, {dy:+.1f}) px")
        print("; ".join(out), flush=True)
        cv2.imwrite(os.path.join(args.out, f"{cam}.jpg"), np.concatenate([dist, ref, und], 1))
    env.close()

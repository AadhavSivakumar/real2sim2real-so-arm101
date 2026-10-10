"""Reset-only check of the desk-constrained spawn: N resets of the camera env, overhead frame after the objects settle,
4x4 grids (after = desk_poly on, before = desk_poly off) + per-reset object positions.

Run (r2s-sim):  cd ~/so101_r2s2r && SO101_SPLAT=1 ./lab.sh -p ~/r2s/desk_check/desk_grid.py --viz none --out ~/r2s/desk_check
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SO101-CubeInCup-Cam-v0")
parser.add_argument("--resets", type=int, default=16)
parser.add_argument("--settle_steps", type=int, default=15, help="hold steps after each reset (0.5 s) before the frame")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--before", type=int, default=16, help="also this many resets with desk_poly=None (old sampler)")
parser.add_argument("--out", required=True)
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import json
import os

import cv2
import gymnasium as gym
import numpy as np
import torch

import so101_r2s2r  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import calib, env_cfg as ecfg, mdp

cfg, _ = resolve_task_config(args.task, "")
cfg.scene.num_envs = 1
cfg.seed = args.seed
cfg.episode_length_s = 20.0
os.makedirs(args.out, exist_ok=True)
print(f"[desk] SPLAT_USD={calib.SPLAT_USD} background={getattr(cfg.scene, 'background', None) is not None}", flush=True)
print(f"[desk] reset_objects params: {cfg.events.reset_objects.params}", flush=True)


def inside(p):
    a = np.asarray(ecfg.DESK_POLY, float)
    b = np.roll(a, -1, 0)
    d = b - a
    n = np.stack([-d[:, 1], d[:, 0]], -1) / np.linalg.norm(d, axis=-1, keepdims=True)
    return float(np.min(((np.asarray(p)[None, :2] - a) * n).sum(-1)))


def grid(tiles, path):
    rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles), 4)]
    img = np.concatenate(rows, 0)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[desk] wrote {path}", flush=True)


with launch_simulation(cfg, args):
    env = gym.make(args.task, cfg=cfg)
    u = env.unwrapped
    term = u.event_manager.get_term_cfg("reset_objects")
    records = {}
    for tag, n, poly in (("after", args.resets, term.params["desk_poly"]), ("before", args.before, None)):
        if n <= 0:
            continue
        term.params["desk_poly"] = poly
        tiles, rec = [], []
        for k in range(n):
            obs, _ = env.reset(seed=args.seed + k)
            p0 = {nm: (u.scene[nm].data.root_pos_w.torch[0] - u.scene.env_origins[0]).cpu().numpy().copy()
                  for nm in ("cup",)}
            p0["cube"] = (mdp.active_cube_pose_w(u)[0][0] - u.scene.env_origins[0]).cpu().numpy().copy()
            for _ in range(args.settle_steps):
                obs, *_ = env.step(u.scene["robot"].data.joint_pos.torch.clone())
            cup = (u.scene["cup"].data.root_pos_w.torch[0] - u.scene.env_origins[0]).cpu().numpy()
            cube = (mdp.active_cube_pose_w(u)[0][0] - u.scene.env_origins[0]).cpu().numpy()
            up = float(torch.as_tensor(mdp.cup_upright(u))[0])
            r = dict(k=k, cube_spawn=p0["cube"].round(4).tolist(), cup_spawn=p0["cup"].round(4).tolist(),
                     cube=cube.round(4).tolist(), cup=cup.round(4).tolist(),
                     cube_in_cm=round(100 * inside(cube), 1), cup_in_cm=round(100 * inside(cup), 1), cup_upright=bool(up))
            rec.append(r)
            print(f"[desk] {tag} {k:2d} cube ({cube[0]:+.3f},{cube[1]:+.3f},{cube[2]:+.3f}) in {r['cube_in_cm']:5.1f} cm | "
                  f"cup ({cup[0]:+.3f},{cup[1]:+.3f},{cup[2]:+.3f}) in {r['cup_in_cm']:5.1f} cm upright {up}", flush=True)
            img = obs["images"]["scene"][0, ..., :3].to(torch.uint8).cpu().numpy()
            img = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
            col = (255, 255, 255) if r["cup_in_cm"] >= 5.5 and r["cube_in_cm"] >= 3.0 else (255, 60, 60)
            cv2.putText(img, f"{tag} {k}: cube {r['cube_in_cm']:.1f} cup {r['cup_in_cm']:.1f} cm in", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        grid(tiles, os.path.join(args.out, f"desk_grid_{tag}.jpg"))
        records[tag] = rec
    records["spawn_stats"] = getattr(u, "r2s_spawn_stats", None)
    with open(os.path.join(args.out, "desk_grid.json"), "w") as f:
        json.dump(records, f, indent=1)
    print(f"[desk] spawn stats {records['spawn_stats']}", flush=True)
    a = records.get("after", [])
    print(f"[desk] after: min cube inside {min(r['cube_in_cm'] for r in a):.1f} cm, min cup inside "
          f"{min(r['cup_in_cm'] for r in a):.1f} cm, cups upright {sum(r['cup_upright'] for r in a)}/{len(a)}", flush=True)
    print("[desk] done", flush=True)
    env.close()

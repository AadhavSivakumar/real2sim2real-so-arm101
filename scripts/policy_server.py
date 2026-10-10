"""Serve a trained LeRobot policy on a local socket so Isaac (which can't import LeRobot) can run it
closed-loop. Observations/actions use LeRobot units and keys, exactly as on the real robot.

Run with the LeRobot venv:
  $LR scripts/policy_server.py --ckpt <run>/checkpoints/last/pretrained_model
"""
import argparse
from multiprocessing.connection import Listener

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--port", type=int, default=6100)
ap.add_argument("--device", default="cuda")
ap.add_argument("--n_action_steps", type=int, default=None, help="ACT: actions executed per chunk (default: as trained)")
ap.add_argument("--temporal_ensemble", type=float, default=None,
                help="ACT: temporal-ensembling coefficient (e.g. 0.01; forces n_action_steps=1)")
args = ap.parse_args()

cfg = PreTrainedConfig.from_pretrained(args.ckpt)
cfg.pretrained_path = args.ckpt
cfg.device = args.device
if args.temporal_ensemble is not None:
    cfg.temporal_ensemble_coeff, cfg.n_action_steps = args.temporal_ensemble, 1
elif args.n_action_steps is not None:
    cfg.n_action_steps = args.n_action_steps
policy = get_policy_class(cfg.type).from_pretrained(args.ckpt, config=cfg).to(args.device).eval()
pre, post = make_pre_post_processors(cfg, pretrained_path=args.ckpt,
                                     preprocessor_overrides={"device_processor": {"device": args.device}})
device = torch.device(args.device)
print(f"[server] {cfg.type} policy from {args.ckpt} on {args.device} (n_action_steps="
      f"{getattr(cfg, 'n_action_steps', None)}, temporal_ensemble={getattr(cfg, 'temporal_ensemble_coeff', None)}); "
      f"listening on :{args.port}", flush=True)

with Listener(("127.0.0.1", args.port), authkey=b"so101") as listener:
    while True:
        with listener.accept() as conn:
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    break
                if msg["cmd"] == "reset":
                    policy.reset()
                    conn.send("ok")
                elif msg["cmd"] == "act":
                    a = predict_action(msg["obs"], policy, device, pre, post, use_amp=False,
                                       task=msg.get("task"), robot_type="so_follower")
                    conn.send(np.asarray(a.cpu(), dtype=np.float32).reshape(-1))
                elif msg["cmd"] == "quit":
                    conn.send("bye")
                    raise SystemExit(0)

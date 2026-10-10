"""Evaluate the scripted expert: success rate, failure phases, episode length.

Run: ./lab.sh -p scripts/run_expert.py --num_envs 32 --episodes 200 --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="SO101-CubeInCup-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--speed", type=float, default=0.20)
parser.add_argument("--noise", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--debug", action="store_true", help="print a trace of env 0")
parser.add_argument("--why", action="store_true", help="print the state just before every failure (+ by cup radius)")
parser.add_argument("--once", action="store_true",
                    help="count exactly the first episode of every env (episodes = num_envs); unbiased, whereas the "
                         "first --episodes to END over-sample quick successes")
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

from collections import Counter

import gymnasium as gym
import torch

import so101_r2s2r  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config
from so101_r2s2r import mdp
from so101_r2s2r.expert import PHASE_NAMES, CubeInCupExpert

env_cfg, _ = resolve_task_config(args.task, "")
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed

with launch_simulation(env_cfg, args):
    env = gym.make(args.task, cfg=env_cfg)
    env.reset(seed=args.seed)
    u = env.unwrapped
    expert = CubeInCupExpert(u, speed=args.speed, noise=args.noise, seed=args.seed)
    expert.reset()
    done_eps, successes, steps = 0, 0, 0
    fail_phase, fail_reason, by_size = Counter(), Counter(), Counter()
    size_ok = Counter()
    ep_len = torch.zeros(u.num_envs, device=u.device)
    lens = []
    counted = torch.zeros(u.num_envs, dtype=torch.bool, device=u.device)
    replans_ok, replans_eps = 0, 0
    by_r = {}  # --why: success by cup radius bin  # self-recoveries (expert.replan from act) in counted episodes
    target = u.num_envs if args.once else args.episodes
    while done_eps < target:
        cube_idx = mdp.active_cube_idx(u).clone()
        phase_before = expert.phase.clone()
        action = expert.act()
        if args.debug and steps % 10 == 0:
            tcp = expert.tcp_w()[0]
            q = u.scene["robot"].data.joint_pos.torch[0]
            print(f"[trace] t={steps * u.step_dt:5.2f} phase={PHASE_NAMES[int(expert.phase[0])]:9s} "
                  f"cmd={[round(v, 3) for v in expert.cmd[0].tolist()]} tcp={[round(v, 3) for v in tcp.tolist()]} "
                  f"err={float((tcp - expert.cmd[0]).norm()):.4f} q={[round(v, 2) for v in q.tolist()]} "
                  f"q_tgt={[round(v, 2) for v in action[0].tolist()]}")
        if args.why:  # the state the episode ends in (env.step resets the ended envs before returning)
            cup_p = u.scene["cup"].data.root_pos_w.torch - u.scene.env_origins
            cube_p = mdp.active_cube_pose_w(u)[0] - u.scene.env_origins
            pre = dict(in_cup=mdp.cube_in_cup(u).clone(), upright=mdp.cup_upright(u).clone(), grip=mdp.gripper_q(u).clone(),
                       cup_r=torch.linalg.vector_norm(cup_p[:, :2], dim=-1), cup_phi=torch.atan2(cup_p[:, 1], cup_p[:, 0]),
                       cube_r=torch.linalg.vector_norm(cube_p[:, :2], dim=-1), cube_z=cube_p[:, 2].clone(),
                       cube_cup=torch.linalg.vector_norm(cube_p[:, :2] - cup_p[:, :2], dim=-1))
        obs, rew, term, trunc, info = env.step(action)
        steps += 1
        ep_len += 1
        ended = (term | trunc).nonzero().flatten()
        if len(ended):
            tm = u.termination_manager
            succ = tm.get_term("success")[ended]
            n_rep = getattr(expert, "n_replans", torch.zeros(u.num_envs, dtype=torch.long, device=u.device))
            for k, e in enumerate(ended.tolist()):
                if args.once and bool(counted[e]):
                    continue
                counted[e] = True
                done_eps += 1
                if int(n_rep[e]) > 0:
                    replans_eps += 1
                    replans_ok += int(bool(succ[k]))
                by_size[int(cube_idx[e])] += 1
                lens.append(float(ep_len[e]) * u.step_dt)
                if bool(succ[k]):
                    successes += 1
                    size_ok[int(cube_idx[e])] += 1
                else:
                    fail_phase[PHASE_NAMES[int(phase_before[e])]] += 1
                    if args.why:
                        print(f"[why] {PHASE_NAMES[int(phase_before[e])]:8s} in_cup={int(pre['in_cup'][e])} upright={int(pre['upright'][e])} "
                              f"grip={float(pre['grip'][e]):.2f} cube z={float(pre['cube_z'][e]) * 1000:.0f}mm "
                              f"cube-cup={float(pre['cube_cup'][e]) * 1000:.0f}mm | cup r={float(pre['cup_r'][e]):.3f} "
                              f"phi={float(pre['cup_phi'][e]):+.2f} cube r={float(pre['cube_r'][e]):.3f}", flush=True)
                if args.why:
                    b = min(int((float(pre["cup_r"][e]) - 0.15) / 0.035), 4)
                    by_r[b] = by_r.get(b, (0, 0)); by_r[b] = (by_r[b][0] + int(bool(succ[k])), by_r[b][1] + 1)
                    for name in ("time_out", "cube_dropped", "cup_tipped"):
                        if bool(tm.get_term(name)[e]):
                            fail_reason[name] += 1
            ep_len[ended] = 0
            expert.reset(ended)
    print(f"[expert] episodes={done_eps} success={successes / done_eps:.1%} mean_len={sum(lens) / len(lens):.1f}s")
    print("[expert] success by cube size:", {f"{int(env_cfg.cube_sizes[k] * 1000)}mm": f"{size_ok[k]}/{by_size[k]}" for k in sorted(by_size)})
    print("[expert] failures by phase:", dict(fail_phase))
    print("[expert] failure terminations:", dict(fail_reason))
    print(f"[expert] episodes with a self-recovery re-plan: {replans_eps} ({replans_ok} of them succeeded)")
    if args.why:
        print("[expert] success by cup radius:", {f"{0.15 + 0.035 * b:.3f}-{0.15 + 0.035 * (b + 1):.3f}": f"{v[0]}/{v[1]}"
                                                  for b, v in sorted(by_r.items())})
    env.close()

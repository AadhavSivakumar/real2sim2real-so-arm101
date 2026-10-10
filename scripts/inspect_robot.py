"""Print SO-101 bodies/joints and jaw geometry (in the gripper link frame) to locate the grasp point.

Run: ~/IsaacLab/isaaclab.sh -p scripts/inspect_robot.py --viz none
"""
import argparse
import sys

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab_tasks.utils import setup_preset_cli

parser = argparse.ArgumentParser()
add_launcher_args(parser)
args, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + hydra_args

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import resolve_task_config

TASK = "IsaacContrib-Stack-Cube-SO101-v0"
env_cfg, _ = resolve_task_config(TASK, "")
env_cfg.scene.num_envs = 1

with launch_simulation(env_cfg, args):
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    print("[inspect] joints:", robot.joint_names)
    print("[inspect] default q:", [round(v, 3) for v in robot.data.default_joint_pos.torch[0].tolist()])
    lim = robot.data.joint_pos_limits.torch[0]
    print("[inspect] limits:", {n: (round(a, 3), round(b, 3)) for n, (a, b) in zip(robot.joint_names, lim.tolist())})
    print("[inspect] bodies:", robot.body_names)
    print("[inspect] root pose w:", robot.data.root_pos_w.torch[0].tolist(), robot.data.root_quat_w.torch[0].tolist())

    from pxr import Usd, UsdGeom, Gf
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    gripper = stage.GetPrimAtPath("/World/envs/env_0/Robot/gripper")
    w2g = UsdGeom.Xformable(gripper).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    for q_grip in (0.0, 1.745):
        robot.write_joint_position_to_sim_index(
            position=torch.tensor([[q_grip]], device=env.unwrapped.device),
            joint_ids=[robot.joint_names.index("gripper")],
        )
        env.unwrapped.sim.render()
        cache.Clear()
        for name in ["gripper/visuals", "moving_jaw_so101_v1/visuals"]:
            prim = stage.GetPrimAtPath(f"/World/envs/env_0/Robot/{name}")
            if not prim.IsValid():
                print("[inspect] missing", name)
                continue
            # world-space corners of the (aligned) bbox, expressed in the gripper link frame
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            corners = [Gf.Vec3d(x, y, z) for x in (rng.GetMin()[0], rng.GetMax()[0])
                       for y in (rng.GetMin()[1], rng.GetMax()[1]) for z in (rng.GetMin()[2], rng.GetMax()[2])]
            loc = [w2g.Transform(c) for c in corners]
            mn = [round(min(p[i] for p in loc), 4) for i in range(3)]
            mx = [round(max(p[i] for p in loc), 4) for i in range(3)]
            print(f"[inspect] gripper_q={q_grip} {name}: bbox in gripper frame min={mn} max={mx}")
    # children of gripper link (camera mount etc.)
    print("[inspect] gripper children:", [c.GetName() for c in gripper.GetChildren()])
    env.close()

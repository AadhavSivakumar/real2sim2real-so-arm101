"""Sim <-> LeRobot unit conversion for the SO-101 (no Isaac imports; usable from any venv).

LeRobot (use_degrees=True): arm joints in degrees, 0 deg = midpoint of the calibrated range; gripper
0..100 over the calibrated range. NVIDIA's SO-101 USD ("so101_new_calib") uses the same signs (IsaacTeleop's
so101_leader plugin maps ticks -> radians as (ticks - range_mid) * 2*pi/4096 with sign +1 for drive_mode 0).

The zeros do not all coincide on the user's follower (fitted 2026-10-09 from the real episodes: open-loop replay of
the recorded actions in sim and overhead-camera silhouettes, scripts/replay_real.py and
/mnt/secondary/so101_r2s/brev/roll_sign.py): sim angle = rad(reading + JOINT_ZERO_OFFSET_DEG). wrist_roll is a
full-turn motor that LeRobot zeroes wherever the wrist was turned during the "middle pose" calibration step; on this
arm that is 90 deg from the USD zero (the jaws open sideways, seen from above, at a reading of 0).

The gripper is mapped linearly onto [GRIPPER_SIM_CLOSED, GRIPPER_SIM_OPEN]; the span is the follower's calibrated
range (1953..3476 ticks = 133.9 deg), which makes the sim jaw stop on the 30 mm cube at the real reading (~15).
"""
from __future__ import annotations

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LEROBOT_NAMES = [f"{j}.pos" for j in JOINTS]
JOINT_ZERO_OFFSET_DEG = np.array([0.0, -0.2, -4.5, -3.0, -90.0])  # sim = reading + offset (arm joints)
GRIPPER_SIM_CLOSED, GRIPPER_SIM_OPEN = -0.175, 2.161  # [rad] <-> LeRobot 0 / 100 (sim jaw limit 1.745 = reading 82)


def sim_to_lerobot(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    out = np.empty_like(q)
    out[..., :5] = np.degrees(q[..., :5]) - JOINT_ZERO_OFFSET_DEG
    out[..., 5] = (q[..., 5] - GRIPPER_SIM_CLOSED) / (GRIPPER_SIM_OPEN - GRIPPER_SIM_CLOSED) * 100.0
    return out.astype(np.float32)


def lerobot_to_sim(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    out[..., :5] = np.radians(v[..., :5] + JOINT_ZERO_OFFSET_DEG)
    out[..., 5] = GRIPPER_SIM_CLOSED + v[..., 5] / 100.0 * (GRIPPER_SIM_OPEN - GRIPPER_SIM_CLOSED)
    return out.astype(np.float32)

"""Standalone SO-101 forward kinematics from TheRobotStudio's so101_new_calib.urdf (numpy only).

fingertip_in_base(q) -> position of ``gripper_frame_link`` (between the jaw tips) in the ``base_link``
frame, for q = 5 arm joint angles [rad] (shoulder_pan .. wrist_roll). The gripper joint does not move
this frame. Validated against Isaac Sim by scripts/validate_fk.py.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path("/mnt/secondary/so101_r2s/assets/urdf/so101_new_calib.urdf")
CHAIN = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper_frame_joint"]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _axis_angle(axis, a):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _load(urdf=URDF):
    joints = {}
    for j in ET.parse(urdf).getroot().findall("joint"):  # top level only (transmissions repeat names)
        o = j.find("origin")
        xyz = np.array([float(v) for v in o.get("xyz").split()]) if o is not None else np.zeros(3)
        rpy = [float(v) for v in o.get("rpy").split()] if o is not None else [0, 0, 0]
        ax = j.find("axis")
        joints[j.get("name")] = dict(xyz=xyz, R=_rpy(*rpy), axis=[float(v) for v in ax.get("xyz").split()] if ax is not None else [0, 0, 1],
                                     type=j.get("type"))
    return joints


_J = None


def link_poses(q):
    """4x4 poses (in base_link) of each joint's child frame along the chain, for 5 arm angles q."""
    global _J
    if _J is None:
        _J = _load()
    T = np.eye(4)
    out = []
    for name, qi in zip(CHAIN, list(q) + [0.0]):
        j = _J[name]
        A = np.eye(4)
        A[:3, :3] = j["R"]
        A[:3, 3] = j["xyz"]
        if j["type"] == "revolute":
            M = np.eye(4)
            M[:3, :3] = _axis_angle(j["axis"], qi)
            A = A @ M
        T = T @ A
        out.append(T.copy())
    return out


def fingertip_in_base(q) -> np.ndarray:
    return link_poses(q)[-1][:3, 3]


# URDF base_link -> sim desk/world frame (robot seated as in so101_r2s2r/calib.py). Fitted against Isaac
# physics over 30 random poses by scripts/validate_fk.py: rotation = identity (< 0.03 deg),
# residual 0.2 mm mean / 0.4 mm max.
T_WORLD_URDF = np.array([-0.01578, 0.02053, 0.00234])


def fingertip_in_world(q) -> np.ndarray:
    """Fingertip ('gripper_frame_link') in the sim desk frame for 5 arm angles [rad]."""
    return fingertip_in_base(q) + T_WORLD_URDF

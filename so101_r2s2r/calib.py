"""Real-world calibration of the desk scene, expressed in the desk frame (see env_cfg.py).

Everything here is a PLACEHOLDER until the calibration step (ArUco anchor + ChArUco intrinsics +
splat alignment) overwrites it. Quaternions are (x, y, z, w).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def look_at_ros(eye, target, up=(1.0, 0.0, 0.0)) -> tuple[float, float, float, float]:
    """Quaternion (xyzw) of a ROS-convention camera (+z forward, -y image-up) at ``eye`` looking at ``target``.

    ``up`` is the world direction that should appear at the top of the image.
    """
    z = np.asarray(target, float) - np.asarray(eye, float)
    z /= np.linalg.norm(z)
    y = -np.asarray(up, float)
    y -= z * (y @ z)
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return quat_from_matrix(np.stack([x, y, z], axis=1))


def quat_from_matrix(m: np.ndarray) -> tuple[float, float, float, float]:
    """Rotation matrix -> quaternion (xyzw), Shepperd's method (stable for 180-degree rotations)."""
    tr = np.trace(m)
    if tr > 0:
        s = 2 * np.sqrt(1 + tr)
        q = ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, s / 4)
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
        v = [0.0, 0.0, 0.0]
        v[i] = s / 4
        v[j] = (m[j, i] + m[i, j]) / s
        v[k] = (m[k, i] + m[i, k]) / s
        q = (*v, (m[k, j] - m[j, k]) / s)
    q = np.asarray(q, float)
    q /= np.linalg.norm(q)
    return tuple(float(c) for c in q)


IMAGE_SIZE = (640, 480)  # both Jetson cameras record 640x480 MJPG @ 30 fps

# Robot base on the desk (same seat as IsaacLab's SO-101 stack task: arm faces +x).
ROBOT_BASE_POS = (0.0, 0.0, -0.03008)
ROBOT_BASE_QUAT = (0.0, 0.0, 0.70710678, 0.70710678)

# Wrist camera: NVIDIA's SO-101 camera-mount pose (USD camera prim under the gripper link, OpenGL
# convention) and the wowrobo 2MP intrinsics. calib.json replaces both with the user's rig: K from the
# Innomaker's ChArUco calibration (2026-10-07: fx ~551, i.e. 60 deg HFOV, not the wowrobo's 87 deg) and
# the pose fitted to the jaws in real wrist frames (calib/wrist_mount.py, 2026-10-08).
# The sim renders exactly *_K as an undistorted pinhole (env_cfg._pinhole; RENDER_LENS_DISTORTION is off), so real frames
# must be undistorted with cv2.undistort(img, K, DIST) (same K as output) before a sim-trained policy sees them.
WRIST_K = [339.27, 0.0, 323.56, 0.0, 338.82, 250.27, 0.0, 0.0, 1.0]
WRIST_DIST = (0.0, 0.0, 0.0, 0.0, 0.0)
WRIST_CAM_POS = (-0.00189, 0.05226, -0.05853)
WRIST_CAM_QUAT = (-0.30071, 0.0, 0.0, 0.95372)

# Overhead Astra Pro HD (RGB). Placeholder: 0.85 m above the desk looking straight down, robot at the
# bottom of the image (+x desk = image up), as in the real overhead frame.
SCENE_K = [525.0, 0.0, 320.0, 0.0, 525.0, 240.0, 0.0, 0.0, 1.0]
SCENE_DIST = (0.0, 0.0, 0.0, 0.0, 0.0)
SCENE_CAM_POS = (0.22, 0.0, 0.85)
SCENE_CAM_QUAT = look_at_ros(SCENE_CAM_POS, (0.22, 0.0, 0.0), up=(1.0, 0.0, 0.0))

# Gaussian-splat background (USD ParticleField) and its similarity transform into the desk frame.
SPLAT_USD: str | None = None
SPLAT_POS = (0.0, 0.0, 0.0)
SPLAT_QUAT = (0.0, 0.0, 0.0, 1.0)
SPLAT_SCALE = 1.0

# Measured values written by calib/solve.py override the placeholders above.
_MEASURED = Path(__file__).resolve().parents[1] / "calib" / "data" / "calib.json"
if _MEASURED.exists():
    for _k, _v in json.loads(_MEASURED.read_text()).items():
        if _k in globals() and _k.isupper():
            globals()[_k] = tuple(_v) if isinstance(_v, list) else _v

# The splat background (calib/export_splat.py -> splat.json: SPLAT_USD/POS/QUAT/SCALE) replaces the textured desk and
# backdrop, so it is opt-in: SO101_SPLAT=1.
_SPLAT = _MEASURED.with_name("splat.json")
if os.environ.get("SO101_SPLAT") == "1" and _SPLAT.exists():
    for _k, _v in json.loads(_SPLAT.read_text()).items():
        if _k.startswith("SPLAT_"):
            globals()[_k] = tuple(_v) if isinstance(_v, list) else _v

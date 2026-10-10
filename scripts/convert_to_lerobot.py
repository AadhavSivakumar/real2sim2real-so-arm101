"""Convert recorded sim episodes (record_demos.py) into a LeRobot v3 dataset with exactly the schema of
the Jetson web-console recorder (so sim and real data can be mixed for co-training).

Run with the LeRobot venv:
  /mnt/secondary/so101_r2s/.venv-lerobot/bin/python scripts/convert_to_lerobot.py \
      --src /mnt/secondary/so101_r2s/sim_demos/run0 --name sim_cube_in_cup_v0
"""
import argparse
import importlib.util
import subprocess
from pathlib import Path

import numpy as np

from lerobot.configs import RGBEncoderConfig
from lerobot.datasets import LeRobotDataset

_spec = importlib.util.spec_from_file_location("units", Path(__file__).resolve().parents[1] / "so101_r2s2r" / "units.py")
units = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(units)

FPS = 30
TASK = "Put the cube in the cup"
CAMERAS = ("wrist", "scene")
IMG = {"dtype": "video", "shape": (480, 640, 3), "names": ["height", "width", "channels"]}
FEATURES = {  # identical to ~/lerobot/so101_web/recorder.py on the Jetson
    "observation.state": {"dtype": "float32", "shape": (6,), "names": units.LEROBOT_NAMES},
    "action": {"dtype": "float32", "shape": (6,), "names": units.LEROBOT_NAMES},
    **{f"observation.images.{c}": dict(IMG) for c in CAMERAS},
}


def read_video(path: Path, w: int = 640, h: int = 480) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="one or more record_demos.py output dirs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--root", default="/mnt/secondary/so101_r2s/lerobot_datasets")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--max_episodes", type=int, default=None)
    args = ap.parse_args()

    eps = sorted(p for s in args.src for p in Path(s).glob("ep_*") if (p / "data.npz").exists())[: args.max_episodes]
    ds = LeRobotDataset.create(
        f"local/{args.name}", FPS, FEATURES, root=Path(args.root) / args.name, robot_type="so_follower",
        use_videos=True, rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="fast", crf=23, g=2),
        streaming_encoding=True, encoder_threads=4, encoder_queue_maxsize=120,  # encode on the fly, no PNG round-trip
    )
    for k, ep in enumerate(eps):
        d = np.load(ep / "data.npz")
        state, action = units.sim_to_lerobot(d["state"]), units.sim_to_lerobot(d["action"])
        frames = {c: read_video(ep / f"{c}.mp4") for c in CAMERAS}
        T = min(len(state), *(len(f) for f in frames.values()))
        for t in range(T):
            ds.add_frame({"observation.state": state[t], "action": action[t],
                          **{f"observation.images.{c}": frames[c][t] for c in CAMERAS}, "task": args.task})
        ds.save_episode()
        print(f"[convert] {k + 1}/{len(eps)} {ep.name}: {T} frames, cube {d['cube_size'] * 1000:.0f} mm", flush=True)
    ds.finalize()
    print(f"[convert] wrote {len(eps)} episodes to {Path(args.root) / args.name}")


if __name__ == "__main__":
    main()

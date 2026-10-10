"""Calibration capture through the Jetson web console's read-only HTTP endpoints
(start it on the Jetson with:  cd ~/lerobot && ./so101.sh web). Never commands the arm.

  intrinsics --cam scene|wrist [--board paper|phone]   grab frames while you wave the ChArUco board
                                 (page 2 of the PDF, or so101_phone_board.png full-screen on a phone)
  snapshot   --name NAME         one frame from each camera + current joint angles
  touch      --name NAME         press Enter each time the closed jaw tips touch a marker corner

Run with any python that has numpy + opencv, e.g. ~/TestingGrounds/robot_splat/.venv/bin/python.
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from boards import BOARDS, MIN_CORNERS, detect

CONSOLE = os.environ.get("SO101_CONSOLE", "http://jetson:8080")  # the Jetson web console
DATA = Path(__file__).resolve().parent / "data"


def get_state() -> dict:
    return json.loads(urllib.request.urlopen(f"{CONSOLE}/api/state", timeout=3).read())


def get_frame(cam: str, skip: int = 2) -> np.ndarray:
    """Newest JPEG from the console's MJPEG stream (skips a couple of possibly stale frames)."""
    with urllib.request.urlopen(f"{CONSOLE}/stream/{cam}.mjpg", timeout=5) as r:
        buf, got = b"", 0
        while True:
            buf += r.read(8192)
            a = buf.find(b"\xff\xd8")
            b = buf.find(b"\xff\xd9", a + 2) if a >= 0 else -1
            if a >= 0 and b >= 0:
                jpg, buf = buf[a:b + 2], buf[b + 2:]
                got += 1
                if got > skip:
                    return cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)


def detect_charuco(img, board):
    corners, ids = detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), board)
    return ({} if ids is None else {int(i): c for i, c in zip(ids.ravel(), corners.reshape(-1, 2))})


def too_similar(det: dict, kept: list, px: float = 15.0) -> bool:
    """Same view as an already-kept frame (board barely moved): adds nothing to the calibration."""
    for k in kept:
        common = det.keys() & k.keys()
        if len(common) >= 4 and np.mean([np.linalg.norm(det[i] - k[i]) for i in common]) < px:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("intrinsics"); a.add_argument("--cam", required=True); a.add_argument("--n", type=int, default=40)
    a.add_argument("--period", type=float, default=0.5); a.add_argument("--board", choices=BOARDS, default="paper")
    a.add_argument("--timeout", type=float, default=1800, help="give up after this many seconds")
    a.add_argument("--debug_every", type=int, default=0, help="also save every Nth frame (any detection) to debug/")
    s = sub.add_parser("snapshot"); s.add_argument("--name", required=True)
    t = sub.add_parser("touch"); t.add_argument("--name", default="marker")
    args = ap.parse_args()

    state = get_state()
    print(f"console OK: mode={state.get('mode')} torque={state.get('torque')} hz={state.get('hz')}")
    if args.mode == "intrinsics":
        out = DATA / f"intr_{args.cam}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "board.txt").write_text(args.board)
        board, need = BOARDS[args.board], MIN_CORNERS[args.board]
        kept_det = [detect_charuco(cv2.imread(str(f)), board) for f in sorted(out.glob("*.jpg"))]
        kept = len(kept_det)
        print(f"Move the {args.board} ChArUco board slowly in front of the {args.cam} camera: tilt it, fill the corners "
              f"and edges of the view, vary distance. Keeping distinct frames with >= {need} corners.")
        t0, seen = time.time(), 0
        while kept < args.n and time.time() - t0 < args.timeout:
            try:
                img = get_frame(args.cam)
            except OSError as e:  # console restarting / network blip: wait instead of dying
                print(f"  console unreachable ({e}); retrying...", flush=True)
                time.sleep(3)
                continue
            det = detect_charuco(img, board)
            seen += 1
            if args.debug_every and seen % args.debug_every == 0:
                (out / "debug").mkdir(exist_ok=True)
                cv2.imwrite(str(out / "debug" / f"{seen:04d}_{len(det)}c.jpg"), img)
            if len(det) >= need and not too_similar(det, kept_det):
                cv2.imwrite(str(out / f"{kept:03d}.jpg"), img)
                kept_det.append(det)
                kept += 1
                print(f"  kept {kept}/{args.n} ({len(det)} corners) at {time.time() - t0:.0f}s", flush=True)
            time.sleep(args.period)
        print(f"saved {kept} frames to {out}")
    elif args.mode == "snapshot":
        out = DATA / f"snap_{args.name}"
        out.mkdir(parents=True, exist_ok=True)
        for cam in ("scene", "wrist"):
            cv2.imwrite(str(out / f"{cam}.jpg"), get_frame(cam))
        (out / "state.json").write_text(json.dumps(get_state(), indent=1))
        print(f"saved {out}")
    elif args.mode == "touch":
        out = DATA / f"touch_{args.name}.json"
        pts = []
        print("Close the gripper fully. With the leader arm, touch the closed jaw tips to a corner of the black\n"
              "square (tips as vertical as you can), hold still, press Enter. Repeat for all 4 corners in any\n"
              "order. Type q + Enter when done.")
        while True:
            if input(f"[{len(pts)} recorded] Enter = record, q = finish: ").strip().lower() == "q":
                break
            st = get_state()
            pts.append(st["follower"])
            print("  follower:", {k: round(v, 1) for k, v in st["follower"].items()})
        out.write_text(json.dumps(pts, indent=1))
        print(f"saved {len(pts)} touches to {out}")


if __name__ == "__main__":
    sys.exit(main())

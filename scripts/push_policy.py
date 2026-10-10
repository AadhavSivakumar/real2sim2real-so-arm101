"""Package a trained LeRobot checkpoint for the SO-101 web console and put it where the console can run it.

  push   copy a pretrained_model dir (local path or brev-host:path) to the Jetson's ~/so101_policies/<name>/ with an
         so101_deploy.json sidecar (start pose, speed cap, time limit, task, and -- for policies trained on
         sim + undistorted real frames -- the camera undistortion from calib/data/calib.json).
  serve  for models the Jetson can't run (SmolVLA, pi0, GR00T: no transformers there): copy the checkpoint and
         policy_worker.py to a GPU box, start the worker there, and print the tunnel + remotes.json entry that let
         the console reach it.

  python scripts/push_policy.py push --src r2s-train:r2s/train_runs/act_v3a/checkpoints/080000/pretrained_model \
      --name act_v3a_80k --undistort --notes "ACT co-trained: 1000 sim v3 + 10 real (x6)"
  python scripts/push_policy.py serve --src r2s-real:r2s/train_runs/smolvla_v3a/checkpoints/020000/pretrained_model \
      --name smolvla_v3a --host r2s-real --port 7100 --undistort
"""
import argparse
import json
import re
import os
import secrets
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CALIB = REPO / "calib" / "data" / "calib.json"
STAGE = Path("/mnt/secondary/so101_r2s/policies")
WORKER = Path("/mnt/secondary/so101_r2s/so101_web_dev/web/policy_worker.py")
JETSON = os.environ.get("SO101_JETSON", "asnano@jetson")  # user@host of the Jetson
SSH = ["-o", "ConnectTimeout=30"]
REST = [-2.4, -103.1, 95.6, 55.0, 0.5, 22.0]  # median first frame of the real demos (cube_cup_real_v1)


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def sidecar(args):
    d = {"undistort": None, "start_pose": args.start_pose or REST, "task": args.task, "max_speed": args.max_speed,
         "time_limit_s": args.time_limit, "n_action_steps": args.n_action_steps, "notes": args.notes or ""}
    m = re.search(r"checkpoints/0*(\d+)/pretrained_model", args.src)
    if m:  # the console shows this instead of the run's planned total (train_config.json)
        d["train_step"] = int(m.group(1))
    if args.undistort:
        cal = json.loads(CALIB.read_text())
        d["undistort"] = {cam: {"K": cal[f"{key}_K"], "dist": cal[f"{key}_DIST"]}
                          for cam, key in (("wrist", "WRIST"), ("scene", "SCENE"))}
    return d


def stage(args):
    dst = STAGE / args.name
    dst.mkdir(parents=True, exist_ok=True)
    run(["rsync", "-a", "--delete", "--exclude", "so101_deploy.json", args.src.rstrip("/") + "/", str(dst) + "/"])
    if not (dst / "config.json").exists():
        raise SystemExit(f"{args.src} has no config.json: pass the pretrained_model directory")
    (dst / "so101_deploy.json").write_text(json.dumps(sidecar(args), indent=1))
    return dst


def push(args):
    dst = stage(args)
    run(["ssh", *SSH, JETSON, f"mkdir -p ~/so101_policies/{args.name}"])
    for attempt in range(5):  # the Jetson link drops now and then; rsync --partial resumes
        r = subprocess.run(["rsync", "-a", "--partial", "--timeout=120", "-e", "ssh " + " ".join(SSH),
                            str(dst) + "/", f"{JETSON}:so101_policies/{args.name}/"])
        if r.returncode == 0:
            break
        print(f"rsync failed (attempt {attempt + 1}), retrying")
    else:
        raise SystemExit("copy to the Jetson failed")
    print(f"pushed {args.name}: select it in the console's Policy card (Refresh)")


def serve(args):
    dst = stage(args)
    key = args.authkey or secrets.token_hex(16)
    rdir = f"so101_policies/{args.name}"
    run(["ssh", args.host, f"mkdir -p ~/{rdir} ~/so101_policies/bin"])
    run(["rsync", "-a", str(dst) + "/", f"{args.host}:{rdir}/"])
    run(["rsync", "-a", str(WORKER), str(WORKER.with_name("jetson_accel.py")), f"{args.host}:so101_policies/bin/"])
    log = f"~/so101_policies/{args.name}.worker.log"
    run(["ssh", args.host, f"(nohup {args.python} ~/so101_policies/bin/policy_worker.py --ckpt ~/{rdir} "
                           f"--address 127.0.0.1:{args.port} --authkey {key} > {log} 2>&1 < /dev/null &); "
                           f"sleep 60; grep -m1 -E '^(READY|ERROR)' {log} || tail -5 {log}"])
    print(f"""
Worker on {args.host}:127.0.0.1:{args.port}. Make it reachable from the Jetson with two tunnels from this desktop:
  ssh -N -L {args.port}:127.0.0.1:{args.port} {args.host} &
  ssh -N -R {args.port}:127.0.0.1:{args.port} {" ".join(SSH)} {JETSON} &
then add to the Jetson's ~/so101_policies/remotes.json:
  {{"name": "{args.name}@{args.host}", "address": "127.0.0.1:{args.port}", "authkey": "{key}"}}""")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("push", "serve"):
        p = sub.add_parser(name)
        p.add_argument("--src", required=True, help="pretrained_model dir: local path or host:path (rsync syntax)")
        p.add_argument("--name", required=True)
        p.add_argument("--undistort", action="store_true",
                       help="policy saw undistorted frames (sim co-training): the console undistorts with calib.json")
        p.add_argument("--start_pose", type=float, nargs=6, default=None)
        p.add_argument("--task", default="Put the cube in the cup")
        p.add_argument("--max_speed", type=float, default=120.0)
        p.add_argument("--time_limit", type=float, default=40.0)
        p.add_argument("--n_action_steps", type=int, default=None)
        p.add_argument("--notes", default="")
        if name == "serve":
            p.add_argument("--host", required=True, help="GPU box with LeRobot (ssh alias)")
            p.add_argument("--port", type=int, default=7100)
            p.add_argument("--python", default="~/lr/bin/python")
            p.add_argument("--authkey", default=None)
    args = ap.parse_args()
    {"push": push, "serve": serve}[args.cmd](args)


if __name__ == "__main__":
    main()

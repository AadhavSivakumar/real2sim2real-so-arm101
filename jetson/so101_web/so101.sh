#!/usr/bin/env bash
# SO-101 helper:
#   ./so101.sh calibrate-leader | calibrate-follower | teleop
#   ./so101.sh web [port]      (web console on 0.0.0.0:8080, set SO101_TOKEN=... to require a token)
#   ./so101.sh record "task description" [num_episodes] [dataset_name]
#   ./so101.sh rollout <policy> [seconds]   (run a trained policy with lerobot-rollout; console must be stopped)
#   ./so101.sh prep-pi05 <dataset_name>   (true quantile stats for pi0.5, after the last recording session)
#   ./so101.sh policy-server <ckpt_dir> [port]   (serve a policy to the web console from this machine's GPU,
#              on 0.0.0.0:port (default 7100); authkey from SO101_POLICY_KEY, default "so101". Requests are
#              pickled: keep the port on a private network (Tailscale) and set a non-default key. List the
#              server in ~/so101_policies/remotes.json: [{"name": "...", "address": "HOST:PORT", "authkey": "..."}])
# Web console datasets go to ~/so101_datasets/<name> (override with SO101_DATA_ROOT), policies are loaded from
# ~/so101_policies/<name> (override with SO101_POLICY_ROOT).
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

# The Rerun viewer needs a screen; default to the Jetson's desktop session so it also works from tmux/SSH.
export DISPLAY=${DISPLAY:-:1}

LEADER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6053876-if00    # 5V arm
FOLLOWER=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6080770-if00  # 12V arm
WRIST_CAM=/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0
SCENE_CAM=/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0
CAMERAS="{wrist: {type: opencv, index_or_path: $WRIST_CAM, width: 640, height: 480, fps: 30},
          scene: {type: opencv, index_or_path: $SCENE_CAM, width: 640, height: 480, fps: 30}}"

# Live Rerun viewer is OFF by default: on the Jetson it can't keep up with two camera streams and
# back-pressures the control loop (arm moves in bursts). Enable with e.g.  VIEW=1 ./so101.sh teleop
if [ "${VIEW:-0}" = "1" ]; then
  VIEW_ARGS="--display_data=true --display_compressed_images=true"
else
  VIEW_ARGS="--display_data=false"
fi

# The Innomaker wrist camera has been found stuck in manual exposure (5 fps, washed out); reset it.
fix_cams() {
  v4l2-ctl -d "$WRIST_CAM" --set-ctrl=auto_exposure=3,exposure_dynamic_framerate=0,brightness=0,gain=0 2>/dev/null || true
}

# One owner of the follower bus at a time: the web console locks /tmp/so101_follower.lock itself; the CLI commands
# hold the same lock (fd 9, inherited by the lerobot process) for as long as they run.
claim_arm() {
  if pgrep -f "[s]o101_web/app.py" > /dev/null; then
    echo "The web console is running and owns the arm: stop it first (Ctrl-C in its terminal)."; exit 1
  fi
  exec 9> /tmp/so101_follower.lock
  flock -n 9 || { echo "The arm is in use by another so101.sh command."; exit 1; }
}

case "$1" in
  calibrate-leader)
    lerobot-calibrate --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=leader ;;
  calibrate-follower)
    lerobot-calibrate --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower ;;
  teleop)
    claim_arm
    fix_cams
    lerobot-teleoperate \
      --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower \
      --robot.cameras="$CAMERAS" \
      --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=leader \
      --fps=30 $VIEW_ARGS ;;
  web)
    fix_cams
    python so101_web/app.py --port "${2:-8080}" --real ;;
  record)
    TASK="${2:?usage: $0 record \"task description\" [num_episodes] [dataset_name]}"
    claim_arm
    fix_cams
    lerobot-record \
      --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower \
      --robot.cameras="$CAMERAS" \
      --teleop.type=so101_leader --teleop.port=$LEADER --teleop.id=leader \
      --dataset.repo_id=local/${4:-so101_data} \
      --dataset.single_task="$TASK" \
      --dataset.num_episodes=${3:-10} \
      --dataset.episode_time_s=30 --dataset.reset_time_s=10 \
      --dataset.push_to_hub=false \
      $VIEW_ARGS ;;
  rollout)
    # Run a trained policy with LeRobot's own deployment engine (lerobot-rollout; the same command HF's LeLab web UI runs).
    # Stop the web console first: it owns the arm and the cameras. Ctrl-C stops the run and the arm holds where it is.
    #   ./so101.sh rollout act_real_v0 [seconds]      (policy dir, or a name under ~/so101_policies)
    #   MAX_STEP=3 ./so101.sh rollout act_real_v0 20  (slower: max degrees per 1/30 s step, default 5 = 150 deg/s)
    POLICY="${2:?usage: $0 rollout <policy dir or name in ~/so101_policies> [seconds]}"
    [ -d "$POLICY" ] || POLICY="$HOME/so101_policies/$POLICY"
    [ -f "$POLICY/config.json" ] || { echo "no config.json in $POLICY"; exit 1; }
    claim_arm
    fix_cams
    lerobot-rollout --strategy.type=base \
      --policy.path="$POLICY" --policy.use_amp=true \
      --robot.type=so101_follower --robot.port=$FOLLOWER --robot.id=follower \
      --robot.cameras="$CAMERAS" \
      --robot.disable_torque_on_disconnect=false --robot.max_relative_target=${MAX_STEP:-5.0} \
      --task="${TASK:-Put the cube in the cup}" --duration=${3:-40} --fps=30 \
      --return_to_initial_position=false --play_sounds=false --device=cuda ;;
  prep-pi05)
    python so101_web/prep_pi05.py "${2:?usage: $0 prep-pi05 <dataset_name>}" ;;
  policy-server)
    CKPT="${2:?usage: $0 policy-server <ckpt_dir> [port]}"
    SO101_POLICY_KEY="${SO101_POLICY_KEY:-so101}" \
      python so101_web/policy_worker.py --ckpt "$CKPT" --address "0.0.0.0:${3:-7100}" ;;
  *)
    echo "usage: $0 calibrate-leader | calibrate-follower | teleop | web [port] | record \"task\" [episodes] [name] | rollout <policy> [seconds] | prep-pi05 <name> | policy-server <ckpt_dir> [port]"; exit 1 ;;
esac

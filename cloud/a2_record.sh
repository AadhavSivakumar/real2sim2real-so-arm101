#!/bin/bash
# Track A2 recovery-data recorders, run ON r2s-sim (Isaac via ~/so101_r2s2r/lab.sh), in a retry loop: the Isaac camera
# renderer occasionally crashes; both recorders resume in OUT (episodes are renamed into place only when complete).
#
#   a2_record.sh takeover OUT N [record_takeover.py args...]
#       N successful expert takeovers of the policy served on :6210 (scripts/policy_server.py, ACT v3a 80k, n_action_steps 30;
#       start it first -- see the server line below). Resume continues the attempt / seed sequence and the statistics.
#       The server takes ONE client at a time (ACT keeps per-episode action-queue state): parallel takeover recorders each
#       need their own server (PORT=6211 ..., started like the one below) and their own OUT / --seed range.
#   a2_record.sh dart OUT N [record_demos.py args...]
#       N successful scripted-expert demos with DART action noise (default STD=0.08 rad --noise_tau 0.4: smoke test 10/11
#       expert successes; STD=0.05 was 7/7 -- the conservative choice).
#       Each retry uses a new --seed (SEED + 7919 * try) so a resumed run does not repeat the scenes.
#
# Server (once; leave it running):
#   cd ~/so101_r2s2r && (nohup ~/lr/bin/python scripts/policy_server.py \
#     --ckpt /mnt/secondary/so101_r2s/train_runs/act_v3a/checkpoints/080000/pretrained_model --port 6210 --n_action_steps 30 \
#     > ~/logs/server_act_v3a_80k_a2.log 2>&1 < /dev/null & echo $! > ~/logs/server_act_v3a_80k_a2.pid)
# Launch in the background, e.g.:
#   (nohup bash /mnt/secondary/so101_r2s/brev/a2_record.sh takeover /mnt/secondary/so101_r2s/sim_demos/takeover_v3a_s1 300 --seed 10000 \
#     > ~/logs/takeover_v3a_s1.log 2>&1 < /dev/null & echo $! > ~/logs/takeover_v3a_s1.pid)
MODE=$1; OUT=$2; N=$3; shift 3
SEED=${SEED:-0}
cd ~/so101_r2s2r || exit 1
mkdir -p "$OUT"
for i in $(seq 0 29); do
  if [ "$MODE" = takeover ]; then
    ./lab.sh -p scripts/record_takeover.py --episodes "$N" --ckpt "" --port "${PORT:-6210}" --preview 12 --fail_preview 12 \
      --out "$OUT" --viz none "$@" && exit 0
  elif [ "$MODE" = dart ]; then
    ./lab.sh -p scripts/record_demos.py --episodes "$N" --action_noise "${STD:-0.08}" --noise_tau "${TAU:-0.4}" --preview 4 \
      --seed $((SEED + 7919 * i)) --out "$OUT" --viz none "$@" && exit 0
  else
    echo "usage: $0 takeover|dart OUT N [args]"; exit 2
  fi
  echo "[retry] $MODE run $i failed $(date +%T)"
  sleep 10
done
exit 1

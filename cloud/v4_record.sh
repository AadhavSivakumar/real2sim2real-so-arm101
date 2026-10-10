#!/bin/bash
# v4 recorders, run ON r2s-sim (Isaac via ~/so101_r2s2r/lab.sh). a2_record.sh's retry/resume loop plus a wall-clock
# deadline: every run is wrapped in `timeout` (kills its whole process group: lab.sh, Isaac, ffmpeg, the policy server
# it started), so all recorders stop by DEADLINE (epoch s). Episodes are renamed into place only when complete, so a
# run killed at the deadline leaves no half episode (left-over .tmp_ep_* dirs are removed at the end).
#
#   v4_record.sh takeover OUT N DEADLINE [record_takeover.py args...]
#       expert takeovers of ACT v3a 80k. PORT=<port>; CKPT='' uses a server already running on PORT, CKPT=<dir> makes
#       record_takeover.py start its own server (n_action_steps 30). Resume continues the attempt / seed sequence.
#   v4_record.sh dart OUT N DEADLINE [record_demos.py args...]
#       scripted-expert demos with DART noise (STD, default 0.08 rad; TAU 0.4 s); run i uses --seed SEED + 7919 * i.
# SO101_SPLAT is passed through from the caller's environment (SO101_SPLAT=1 = Gaussian-splat room).
MODE=$1; OUT=$2; N=$3; DL=$4; shift 4
SEED=${SEED:-0}
cd ~/so101_r2s2r || exit 1
mkdir -p "$OUT"
echo "[v4rec] $(date +%T) start $MODE $OUT N=$N SO101_SPLAT=${SO101_SPLAT:-unset} deadline $(date -d @"$DL" +%T)"
for i in $(seq 0 29); do
  left=$((DL - $(date +%s)))
  if [ $left -lt 120 ]; then echo "[v4rec] $(date +%T) deadline reached"; break; fi
  if [ "$MODE" = takeover ]; then
    timeout -k 60 $left ./lab.sh -p scripts/record_takeover.py --episodes "$N" --ckpt "${CKPT-}" --port "${PORT:?}" \
      --n_action_steps 30 --preview 12 --fail_preview 12 --out "$OUT" --viz none "$@"
  elif [ "$MODE" = dart ]; then
    timeout -k 60 $left ./lab.sh -p scripts/record_demos.py --episodes "$N" --action_noise "${STD:-0.08}" \
      --noise_tau "${TAU:-0.4}" --preview 4 --seed $((SEED + 7919 * i)) --out "$OUT" --viz none "$@"
  else
    echo "usage: $0 takeover|dart OUT N DEADLINE [args]"; exit 2
  fi
  rc=$?
  if [ $rc = 0 ]; then echo "[v4rec] $(date +%T) finished"; break; fi
  if [ $rc = 124 ] || [ $rc = 137 ]; then echo "[v4rec] $(date +%T) stopped at the deadline (rc $rc)"; break; fi
  echo "[retry] $MODE run $i failed rc=$rc $(date +%T)"
  sleep 10
done
rm -rf "$OUT"/.tmp_ep_*
echo "[v4rec] $(date +%T) end: $(ls -d "$OUT"/ep_* 2>/dev/null | wc -l) episodes in $OUT"

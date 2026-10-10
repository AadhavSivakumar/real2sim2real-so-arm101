#!/bin/bash
# Usage: train_retry.sh RUN LOG MAX [extra lerobot-train args...]
# Resumes RUN from its last checkpoint after each crash (RAM corruption), up to MAX attempts.
RUN=$1; LOG=$2; MAX=$3; shift 3
cd /mnt/secondary/so101_r2s
for i in $(seq 1 $MAX); do
  echo "[retry] attempt $i at $(date +%T)" >> "$LOG"
  ./guard.sh "$LOG.part$i" -- .venv-lerobot/bin/lerobot-train --config_path=$RUN/checkpoints/last/pretrained_model/train_config.json --resume=true "$@"
  rc=$?; cat "$LOG.part$i" >> "$LOG"
  grep -qiE "loss:(nan|inf)" "$LOG.part$i" && { echo "[retry] non-finite loss, stopping" >> "$LOG"; exit 2; }
  [ $rc -eq 0 ] && { echo "[retry] finished at $(date +%T)" >> "$LOG"; exit 0; }
  echo "[retry] attempt $i failed rc=$rc" >> "$LOG"; sleep 15
done; exit 1

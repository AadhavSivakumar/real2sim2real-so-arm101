#!/bin/bash
# Push act_real40 checkpoints (raw frames, no undistortion; chunk steps 30 = trained) to the Jetson as they appear.
B=/mnt/secondary/so101_r2s/brev; R=/home/asubuntudesktop/TestingGrounds/so101_r2s2r
exec >> $B/push_real40.log 2>&1
for s in 040000 080000; do
  until timeout 30 ssh r2s-h100 "test -f ~/r2s/train_runs/act_real40/checkpoints/$s/pretrained_model/policy_postprocessor.json" 2>/dev/null; do sleep 60; done
  sleep 30
  k=$((10#$s / 1000))k
  python3 $R/scripts/push_policy.py push --src r2s-h100:r2s/train_runs/act_real40/checkpoints/$s/pretrained_model \
    --name act_real40_$k --n_action_steps 30 \
    --notes "ACT trained only on the 40 real demos (cube_cup_real_v1 + v2, cube on both sides), raw frames, $k steps (bf16, H100). Chunk steps 30 = trained. Start with a dry run." \
    && echo "[push] $(date +%T) act_real40_$k on the Jetson"
done

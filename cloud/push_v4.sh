#!/bin/bash
# Push act_v4 checkpoints (raw frames, no undistortion; chunk steps 30 = trained) to the Jetson as they appear.
B=/mnt/secondary/so101_r2s/brev; R=/home/asubuntudesktop/TestingGrounds/so101_r2s2r
exec >> $B/push_real40.log 2>&1
for s in 040000 080000; do
  until timeout 30 ssh r2s-h100 "test -f ~/r2s/train_runs/act_v4/checkpoints/$s/pretrained_model/policy_postprocessor.json" 2>/dev/null; do sleep 60; done
  sleep 30
  k=$((10#$s / 1000))k
  python3 $R/scripts/push_policy.py push --src r2s-h100:r2s/train_runs/act_v4/checkpoints/$s/pretrained_model \
    --name act_v4_$k --undistort --n_action_steps 30 \
    --notes "ACT v4 co-trained: v3 sim + v4 splat DART demos + ACT-v3a takeover (recovery) episodes + the 40 real demos (~25% of frames), undistorted frames, $k steps (bf16, H100). Chunk steps 30 = trained. Start with a dry run." \
    && echo "[push] $(date +%T) act_v4_$k on the Jetson"
done

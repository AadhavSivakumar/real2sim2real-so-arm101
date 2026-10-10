#!/bin/bash
# Push act_v4b checkpoints (trained on undistorted frames -> --undistort; chunk steps 30 = trained) to the Jetson's policy
# folder as they appear on r2s-h100. Only copies files (scripts/push_policy.py); nothing is run on the arm.
B=/mnt/secondary/so101_r2s/brev; R=/home/asubuntudesktop/TestingGrounds/so101_r2s2r
exec >> $B/push_v4b.log 2>&1
echo "[push] $(date +%T) waiting for act_v4b checkpoints 040000 / 080000 on r2s-h100"
for s in 040000 080000; do
  until timeout 30 ssh r2s-h100 "test -f ~/r2s/train_runs/act_v4b/checkpoints/$s/pretrained_model/policy_postprocessor.json" 2>/dev/null; do sleep 60; done
  sleep 30
  k=$((10#$s / 1000))k
  python3 $R/scripts/push_policy.py push --src r2s-h100:r2s/train_runs/act_v4b/checkpoints/$s/pretrained_model \
    --name act_v4b_$k --undistort --n_action_steps 30 \
    --notes "ACT v4b co-trained = ACT v4's data minus the sim episodes whose cup/cube sat off (or within 5 cm of the edge of) the real desk (1509 of 4013 dropped; v3 sim + v4 splat DART + ACT-v3a takeover), + the 40 real demos x11 (~24% of frames), undistorted frames, $k steps (bf16, H100). Chunk steps 30 = trained. Start with a dry run." \
    && echo "[push] $(date +%T) act_v4b_$k on the Jetson" || echo "[push] $(date +%T) act_v4b_$k push FAILED"
done
echo "[push] $(date +%T) done"

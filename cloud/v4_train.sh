#!/bin/bash
# ON r2s-sim: train ACT v4 on ~/r2s/lerobot_datasets/cotrain_v4 with the v3 ACT recipe (brev/v3_train.sh: chunk_size 50,
# n_action_steps 30, 80k steps, batch 16, image transforms on, torchcodec, save_freq 10000).
# Output ~/r2s/train_runs/act_v4 (same root as v3 on r2s-train), log ~/r2s/train_act_v4.log, PID ~/r2s/train_act_v4.pid.
D=${D:-cotrain_v4}; RUN=${RUN:-act_v4}; STEPS=${STEPS:-80000}
cd ~/r2s || exit 1
(nohup ~/lr/bin/lerobot-train --dataset.repo_id=local/$D --dataset.root=$HOME/r2s/lerobot_datasets/$D \
  --dataset.video_backend=torchcodec --dataset.image_transforms.enable=true --policy.device=cuda \
  --policy.push_to_hub=false --wandb.enable=false --num_workers=12 --log_freq=200 \
  --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=30 \
  --output_dir=train_runs/$RUN --job_name=$RUN --steps=$STEPS --batch_size=16 --save_freq=10000 \
  > ~/r2s/train_$RUN.log 2>&1 < /dev/null & echo $! > ~/r2s/train_$RUN.pid)
echo "[train] $(date +%T) $RUN started, pid $(cat ~/r2s/train_$RUN.pid), log ~/r2s/train_$RUN.log"

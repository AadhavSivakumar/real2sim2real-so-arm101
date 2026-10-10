#!/bin/bash
# Launch a GR00T N1.7 post-train on r2s-h100b (LeRobot 0.6.2 policy "groot"), SO-101 recipe from lerobot docs/source/groot.mdx:
# new_embodiment, chunk 16, relative arm actions (gripper absolute), bf16, image transforms on, lr 1e-4 cosine with 5% warmup
# (warmup = policy.max_steps * 0.05, so policy.max_steps is set to the run length).
# Usage: groot_train.sh <name> <dataset> <steps> <batch> [save_freq] [extra lerobot-train args...]
#   e.g. groot_train.sh groot_real40 real_all_v1 20000 64 5000
H=${H:-r2s-h100b}; NAME=$1; DS=$2; STEPS=$3; BS=$4; SAVE=${5:-$((STEPS / 4))}; shift 5 2>/dev/null; EXTRA="$*"
[ -n "$NAME" ] && [ -n "$DS" ] && [ -n "$STEPS" ] && [ -n "$BS" ] || { echo "usage: $0 name dataset steps batch [save_freq]"; exit 2; }
ssh $H "cd ~/r2s && (HF_HUB_OFFLINE=1 nohup ~/lr/bin/lerobot-train --dataset.repo_id=local/$DS --dataset.root=/home/\$USER/r2s/lerobot_datasets/$DS \
  --dataset.video_backend=torchcodec --dataset.image_transforms.enable=true \
  --policy.type=groot --policy.base_model_path=nvidia/GR00T-N1.7-3B --policy.embodiment_tag=new_embodiment \
  --policy.chunk_size=16 --policy.n_action_steps=16 --policy.use_relative_actions=true --policy.relative_exclude_joints='[\"gripper\"]' \
  --policy.use_bf16=true --policy.max_steps=$STEPS --policy.push_to_hub=false --policy.device=cuda \
  --wandb.enable=false --num_workers=16 --log_freq=50 --seed=42 \
  --output_dir=train_runs/$NAME --job_name=$NAME --steps=$STEPS --batch_size=$BS --save_freq=$SAVE $EXTRA \
  > ~/r2s/train_$NAME.log 2>&1 < /dev/null & echo \$! > ~/r2s/train_$NAME.pid)"
echo "launched $NAME on $H: pid $(ssh $H cat ~/r2s/train_$NAME.pid), log ~/r2s/train_$NAME.log"

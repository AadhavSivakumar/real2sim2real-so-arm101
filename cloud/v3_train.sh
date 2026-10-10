#!/bin/bash
# Usage: v3_train.sh TAG   -- wait for v3_data.sh TAG to finish, then train on ~/r2s/lerobot_datasets/cotrain_v3$TAG:
# ACT on r2s-train, SmolVLA on r2s-real (dataset copied over via this machine). Log: brev/v3_train_$TAG.log
TAG=$1; ACT_STEPS=${ACT_STEPS:-80000}; VLA_STEPS=${VLA_STEPS:-20000}
B=/mnt/secondary/so101_r2s/brev
D=cotrain_v3$TAG
exec >> $B/v3_train_$TAG.log 2>&1
until grep -q "DONE $D" $B/v3_data_$TAG.log; do
  grep -qE "\[v3\] .*(failed|missing)" $B/v3_data_$TAG.log && { echo "[train] data step failed"; exit 1; }
  sleep 30
done
echo "[train] $(date +%T) $D ready"
COMMON="--dataset.repo_id=local/$D --dataset.root=/home/shadeform/r2s/lerobot_datasets/$D --dataset.video_backend=torchcodec \
--dataset.image_transforms.enable=true --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false \
--num_workers=12 --log_freq=200"
ssh r2s-train "cd ~/r2s && (nohup ~/lr/bin/lerobot-train $COMMON --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=30 \
  --output_dir=train_runs/act_v3$TAG --job_name=act_v3$TAG --steps=$ACT_STEPS --batch_size=16 --save_freq=10000 \
  > ~/r2s/train_act_v3$TAG.log 2>&1 < /dev/null &); echo act started"
rsync -a r2s-train:r2s/lerobot_datasets/$D /mnt/secondary/so101_r2s/lerobot_datasets/ && \
  rsync -a /mnt/secondary/so101_r2s/lerobot_datasets/$D r2s-real:r2s/lerobot_datasets/ || { echo "[train] dataset copy failed"; exit 1; }
echo "[train] $(date +%T) dataset on r2s-real"
ssh r2s-real "cd ~/r2s && (nohup ~/lr/bin/lerobot-train $COMMON --policy.path=lerobot/smolvla_base \
  --rename_map='{\"observation.images.scene\": \"observation.images.camera1\", \"observation.images.wrist\": \"observation.images.camera2\"}' \
  --output_dir=train_runs/smolvla_v3$TAG --job_name=smolvla_v3$TAG --steps=$VLA_STEPS --batch_size=64 --save_freq=5000 \
  > ~/r2s/train_smolvla_v3$TAG.log 2>&1 < /dev/null &); echo smolvla started"
sleep 240
for h in r2s-train r2s-real; do
  ssh $h 'for f in ~/r2s/train_*_v3*.log; do echo "$f: $(grep -acE "Traceback|Error" $f) errors; $(tail -c 300 $f | tr "\r" "\n" | grep -aoE "[0-9]+/[0-9]+ \[[^]]*\]" | tail -1)"; done'
done
echo "[train] $(date +%T) launched"

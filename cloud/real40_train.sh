#!/bin/bash
# Real-only ACT on all 40 real demos (cube_cup_real_v1 + merged cube_cup_real_v2), raw frames (no undistortion),
# on a fresh Brev A6000 "r2s-real2". Log: brev/real40.log. Checkpoints every 10k -> ~/r2s/train_runs/act_real40.
H=${H:-r2s-real2}; B=/mnt/secondary/so101_r2s/brev; STEPS=${STEPS:-80000}
exec >> $B/real40.log 2>&1
echo "[r40] $(date +%T) create $H"
brev ls 2>/dev/null | grep -q "^ *$H " || brev create $H --type ${TYPES:-hyperstack_A6000,massedcompute_A6000_plus,massedcompute_A6000} --detached
for i in $(seq 1 90); do
  timeout 20 ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=no $H true 2>/dev/null && break
  [ $((i % 6)) = 0 ] && brev refresh > /dev/null 2>&1; sleep 20
done
timeout 20 ssh $H true || { echo "[r40] box not reachable"; exit 1; }
echo "[r40] $(date +%T) box up; installing LeRobot (same freeze as r2s-sim)"
scp -q $B/lr_freeze.txt $H:/tmp/lr_freeze.txt
ssh $H 'set -e; for t in $(seq 1 40); do command -v ffmpeg > /dev/null && break; sudo apt-get -o DPkg::Lock::Timeout=60 update -qq && sudo apt-get -o DPkg::Lock::Timeout=300 install -y -qq ffmpeg > /dev/null && break; sleep 15; done; command -v ffmpeg > /dev/null
  export PATH=$HOME/.local/bin:$PATH; command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
  [ -x ~/lr/bin/python ] || uv venv --python 3.12 --seed ~/lr -q
  grep -v "^-e " /tmp/lr_freeze.txt > /tmp/req.txt
  uv pip install --python ~/lr/bin/python -q -r /tmp/req.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  ~/lr/bin/python -c "import lerobot, torch, torchcodec; print(\"env ok\", lerobot.__version__, torch.__version__, torch.cuda.is_available())"
  mkdir -p ~/r2s/lerobot_datasets ~/r2s/train_runs' || { echo "[r40] setup failed"; exit 1; }
echo "[r40] $(date +%T) uploading data"
rsync -a --delete /mnt/secondary/so101_r2s/real_datasets/cube_cup_real_v2_merged/ $H:r2s/lerobot_datasets/cube_cup_real_v2/ && \
rsync -a /mnt/secondary/so101_r2s/lerobot_datasets/real_cube_cup_v1/ $H:r2s/lerobot_datasets/real_cube_cup_v1/ || { echo "[r40] upload failed"; exit 1; }
ssh $H 'cd ~/r2s && rm -rf lerobot_datasets/real_all_v1 && ~/lr/bin/python - <<PY
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset
R = Path.home() / "r2s/lerobot_datasets"
aggregate_datasets(["local/real_cube_cup_v1", "local/cube_cup_real_v2"], "local/real_all_v1",
                   roots=[R / "real_cube_cup_v1", R / "cube_cup_real_v2"], aggr_root=R / "real_all_v1")
ds = LeRobotDataset("local/real_all_v1", root=R / "real_all_v1")
print("real_all_v1:", ds.meta.total_episodes, "episodes", ds.meta.total_frames, "frames", list(ds.meta.tasks.index))
PY' > /tmp/r40_aggr.log 2>&1; grep -E "real_all_v1|Error|Traceback" /tmp/r40_aggr.log
grep -q "^real_all_v1: 40 episodes" /tmp/r40_aggr.log || { echo "[r40] aggregate failed"; exit 1; }
echo "[r40] $(date +%T) training $STEPS steps"
ssh $H "cd ~/r2s && (nohup ~/lr/bin/lerobot-train --dataset.repo_id=local/real_all_v1 --dataset.root=/home/\$USER/r2s/lerobot_datasets/real_all_v1 \
  --dataset.video_backend=torchcodec --dataset.image_transforms.enable=true --policy.device=cuda --policy.push_to_hub=false \
  --wandb.enable=false --num_workers=12 --log_freq=200 --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=30 \
  --output_dir=train_runs/act_real40 --job_name=act_real40 --steps=$STEPS --batch_size=16 --save_freq=10000 --accelerator.mixed_precision=bf16 \
  > ~/r2s/train_act_real40.log 2>&1 < /dev/null & echo \$! > ~/r2s/train_act_real40.pid)"
sleep 300
ssh $H 'grep -acE "Traceback|Error" ~/r2s/train_act_real40.log; grep -a "step:" ~/r2s/train_act_real40.log | tail -2'
echo "[r40] $(date +%T) launched"

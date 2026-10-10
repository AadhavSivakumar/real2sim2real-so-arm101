#!/bin/bash
# ACT v2: train on run1 + takeover1 (sim_cube_in_cup_v2), then the same 50-episode eval as v1 (seed 123).
# Launch as: flock /mnt/secondary/so101_r2s/gpu.lock bash train_eval_v2.sh   (holds the GPU lock for the whole chain)
R=/mnt/secondary/so101_r2s
cd $R
grep -q "wrote 700 episodes" convert_v2.log || { echo "[chain] convert_v2 incomplete, stopping"; exit 1; }

RUN=train_runs/act_sim_v2
LOG=$R/train_runs_act_sim_v2.log
for i in 1 2 3; do  # fresh starts until a first checkpoint exists
  [ -d $RUN/checkpoints/last ] && break
  rm -rf $RUN
  echo "[chain] fresh train attempt $i at $(date +%T)"
  ./guard.sh $LOG -- .venv-lerobot/bin/lerobot-train --dataset.repo_id=local/sim_cube_in_cup_v2 \
    --dataset.root=$R/lerobot_datasets/sim_cube_in_cup_v2 --dataset.video_backend=torchcodec \
    --dataset.image_transforms.enable=true --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=30 \
    --policy.device=cuda --policy.push_to_hub=false --output_dir=$RUN --job_name=act_sim_v2 --steps=40000 \
    --batch_size=16 --num_workers=3 --log_freq=500 --save_freq=1000 --wandb.enable=false && break
done
if [ ! -d $RUN/checkpoints/040000 ]; then
  [ -d $RUN/checkpoints/last ] || { echo "[chain] no checkpoint, stopping"; exit 1; }
  echo "[chain] resuming at $(date +%T)"
  ./train_retry.sh $RUN $R/train_runs_act_sim_v2_resume.log 30 || { echo "[chain] training failed"; exit 1; }
fi
echo "[chain] training done at $(date +%T)"

CK=$R/$RUN/checkpoints/040000/pretrained_model
cd /home/asubuntudesktop/TestingGrounds/so101_r2s2r
for i in 1 2 3 4 5; do
  $R/guard.sh $R/policy_server_act_v2.log -- $R/.venv-lerobot/bin/python -u scripts/policy_server.py --ckpt $CK &
  for t in $(seq 60); do grep -q listening $R/policy_server_act_v2.log 2>/dev/null && break; sleep 5; done
  $R/guard.sh $R/eval_act_v2.log -- ./lab.sh -p scripts/eval_policy.py --episodes 50 --video 4 --viz none \
    --out $R/eval/act_sim_v2
  rc=$?
  pkill -TERM -f "scripts/policy_server.py --ckpt $CK"; sleep 5
  [ $rc -eq 0 ] && grep -q "^\[eval\] success" $R/eval_act_v2.log && { echo "[chain] eval done at $(date +%T)"; exit 0; }
  echo "[chain] eval attempt $i failed rc=$rc at $(date +%T)"
  python3 -c "
import os, sys
for root in sys.argv[1:]:
    for d, _, fs in os.walk(root):
        for f in fs:
            try:
                fd = os.open(os.path.join(d, f), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
            except OSError:
                pass
" /mnt/secondary/isaacsim/kit/cache /mnt/secondary/isaacsim/extscache
  sleep 10
done
exit 1

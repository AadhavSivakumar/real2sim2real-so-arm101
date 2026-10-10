#!/bin/bash
set -x
rsync -a r2s-real:r2s/train_runs/smolvla_v3a/checkpoints/020000/pretrained_model/ /mnt/secondary/so101_r2s/checkpoints/smolvla_v3a_20k/ && \
rsync -a r2s-train:r2s/train_runs/act_v3a/checkpoints/060000/pretrained_model/ /mnt/secondary/so101_r2s/checkpoints/act_v3a_60k/ && \
brev delete r2s-real r2s-train
brev ls
echo DEL_DONE

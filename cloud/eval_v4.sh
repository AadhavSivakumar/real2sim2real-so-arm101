#!/bin/bash
# Sim evals of ACT v4 as checkpoints land on r2s-h100: plain backdrop and Gaussian-splat room, 50 episodes each (seed 123).
B=/mnt/secondary/so101_r2s/brev
( bash $B/eval_ckpt3.sh act_v4_40k r2s-h100 act_v4 040000 6301 50 ) &
sleep 900
( EVAL_ENV="SO101_SPLAT=1" bash $B/eval_ckpt3.sh act_v4_40k_splat r2s-h100 act_v4 040000 6302 50 ) &
wait

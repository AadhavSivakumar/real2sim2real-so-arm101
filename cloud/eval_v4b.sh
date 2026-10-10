#!/bin/bash
# Sim evals on the FIXED sampler (all placements on the real desk), 50 episodes each:
#   act_v4_80k now (baseline: trained on the uncleaned data), then act_v4b 40k and 80k as they land; splat for act_v4b_80k.
B=/mnt/secondary/so101_r2s/brev
( bash $B/eval_ckpt3.sh act_v4_80k_fixed r2s-h100 act_v4 080000 6321 50 ) &
( bash $B/eval_ckpt3.sh act_v4b_40k r2s-h100 act_v4b 040000 6322 50 ) &
wait
( bash $B/eval_ckpt3.sh act_v4b_80k r2s-h100 act_v4b 080000 6323 50 ) &
( EVAL_ENV="SO101_SPLAT=1" bash $B/eval_ckpt3.sh act_v4b_80k_splat r2s-h100 act_v4b 080000 6324 50 ) &
wait

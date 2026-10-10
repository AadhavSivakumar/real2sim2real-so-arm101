#!/bin/bash
# ON r2s-sim, after the v4 recorders: convert the sim_v4_* recordings (v4_convert.sh), wait for the v3 shard conversions,
# aggregate every converted part + the undistorted real datasets (real ~REAL_FRAC of the frames; v4_aggregate.py,
# hard-linked videos) into ~/r2s/lerobot_datasets/cotrain_v4, delete the intermediate parts, then start ACT v4
# (v4_train.sh) unless NOTRAIN=1. (Adapted from brev/v3_data.sh, whose separate training box no longer exists.)
#   (nohup bash v4_data.sh > ~/logs/v4_data.log 2>&1 < /dev/null &)
REAL_FRAC=${REAL_FRAC:-0.25}; J=${J:-14}
B=/mnt/secondary/so101_r2s/brev
S=/mnt/secondary/so101_r2s/sim_demos
R=$HOME/r2s
set -o pipefail
echo "[v4data] $(date +%T) waiting for the recorders"
for f in ~/logs/v4_takeover_*.pid ~/logs/v4_dart_*.pid; do
  while kill -0 "$(cat "$f")" 2>/dev/null; do sleep 15; done
done
echo "[v4data] $(date +%T) recorders done:"
for d in $S/sim_v4_*; do echo "  $(basename $d): $(ls -d $d/ep_* 2>/dev/null | wc -l) episodes"; done
bash $B/v4_convert.sh "$J" 40 $S/sim_v4_* || { echo "[v4data] convert failed"; exit 1; }
for f in ~/logs/v4_convert_v3.pid ~/logs/v4_convert_v3b.pid; do
  [ -f "$f" ] && while kill -0 "$(cat "$f")" 2>/dev/null; do sleep 15; done
done
# anything left over (a failed / interrupted part) gets one more try
bash $B/v4_convert.sh "$J" 84 $S/v3_expert_s1 $S/v3_expert_s2 $S/v3_expert_s3 $S/v3_expert_s4 > /dev/null
bash $B/v4_convert.sh "$J" 63 $(for k in 5 6 7 8 9 10 11 12; do echo $S/v3_expert_s$k; done) > /dev/null
bash $B/v4_convert.sh "$J" 40 $S/sim_v4_* > /dev/null
parts=$(cd $R/logs && ls convert_v4p_*.done | sed 's/^convert_//; s/\.done$//' | sort)
missing=$(cd $R/sim_demos/parts && ls | sort | comm -23 - <(echo "$parts"))
[ -n "$missing" ] && { echo "[v4data] parts not converted: $missing"; exit 1; }
echo "[v4data] $(date +%T) $(echo $parts | wc -w) parts converted; disk $(df -h / | awk 'NR==2{print $4}') free"
cd ~/so101_r2s2r || exit 1
~/lr/bin/python $B/v4_aggregate.py --out cotrain_v4 --real_frac "$REAL_FRAC" --sims $parts \
  --reals real_cube_cup_v1_undist cube_cup_real_v2_undist 2>&1 | grep -av "Auto-inserting\|Starting second pass" \
  || { echo "[v4data] aggregate failed"; exit 1; }
grep -q '"real_dup"' $R/lerobot_datasets/cotrain_v4/meta/v4_sources.json || { echo "[v4data] aggregate incomplete"; exit 1; }
rm -rf $(for p in $parts; do echo $R/lerobot_datasets/$p; done) $R/sim_demos/parts
echo "[v4data] $(date +%T) DONE cotrain_v4 ($(du -sh $R/lerobot_datasets/cotrain_v4 | cut -f1)); parts deleted; disk $(df -h / | awk 'NR==2{print $4}') free"
[ "${NOTRAIN:-0}" = 1 ] || bash $B/v4_train.sh

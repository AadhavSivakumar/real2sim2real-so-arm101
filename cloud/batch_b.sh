#!/bin/bash
# Wait for the r2s-sim2/sim3 recorders (shards 5-12), convert + aggregate all 12 shards into cotrain_v3b, then delete
# the two sim boxes once their shards are pulled. Log: brev/batch_b.log
B=/mnt/secondary/so101_r2s/brev
exec >> $B/batch_b.log 2>&1
echo "[b] $(date +%T) waiting for recorders"
until [ "$(for H in r2s-sim2 r2s-sim3; do timeout 30 ssh $H 'grep -l "\[rec\] done" ~/r2s_work/logs/rec_v3_s*.log 2>/dev/null' 2>/dev/null; done | wc -l)" -ge 8 ]; do sleep 120; done
echo "[b] $(date +%T) recorders done"
for H in r2s-sim2 r2s-sim3; do timeout 30 ssh $H 'grep -h "\[rec\] done" ~/r2s_work/logs/rec_v3_s*.log' 2>/dev/null; done
$B/v3_data.sh b "1 2 3 4 5 6 7 8 9 10 11 12" &
until grep -q "pulled to\|failed" $B/v3_data_b.log 2>/dev/null; do sleep 20; done
if grep -q "pulled to" $B/v3_data_b.log && [ "$(ls -d /mnt/secondary/so101_r2s/sim_demos/v3_expert_s{5,6,7,8,9,10,11,12}/ep_000124 2>/dev/null | wc -l)" = 8 ]; then
  echo "[b] $(date +%T) shards pulled; deleting r2s-sim2 r2s-sim3"
  brev delete r2s-sim2 r2s-sim3
else
  echo "[b] pull incomplete; boxes kept"
fi
wait
echo "[b] $(date +%T) finished"

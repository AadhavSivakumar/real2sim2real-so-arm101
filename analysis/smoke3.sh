#!/bin/bash
# run ON r2s-sim: 1 episode each of record_demos / eval_policy / record_takeover with the desk-constrained sampler
D=~/r2s/desk_check; cd ~/so101_r2s2r || exit 1
export SO101_SPLAT=1
rm -rf $D/rd $D/ev $D/tk
( timeout -k 30 1500 ./lab.sh -p scripts/record_demos.py --episodes 1 --preview 1 --out $D/rd --viz none > $D/rd.log 2>&1; echo "[smoke] record_demos rc=$?" >> $D/smoke.log ) &
( timeout -k 30 1500 ./lab.sh -p scripts/eval_policy.py --episodes 1 --port 6311 --video 1 --out $D/ev --viz none > $D/ev.log 2>&1; echo "[smoke] eval_policy rc=$?" >> $D/smoke.log
  timeout -k 30 1800 ./lab.sh -p scripts/record_takeover.py --episodes 1 --ckpt "" --port 6311 --preview 1 --max_attempts 4 --out $D/tk --viz none > $D/tk.log 2>&1; echo "[smoke] record_takeover rc=$?" >> $D/smoke.log ) &
wait
echo "[smoke] all done" >> $D/smoke.log

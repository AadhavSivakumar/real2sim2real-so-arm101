#!/bin/bash
# Usage: [SERVER_ARGS="--n_action_steps 10"] eval_ckpt2.sh NAME SRC_HOST RUN STEP PORT [EPISODES]
#   e.g. eval_ckpt.sh act_v3a_40k r2s-train act_v3a 040000 6201
# Wait for SRC_HOST:~/r2s/train_runs/RUN/checkpoints/STEP, copy it via this machine to r2s-sim, serve it on PORT and run
# scripts/eval_policy.py (current repo on r2s-sim = the env the v3 data came from). Log: brev/eval_NAME.log;
# results: r2s-sim + local /mnt/secondary/so101_r2s/eval/NAME (eval.log, eval.mp4).
NAME=$1; H=$2; RUN=$3; STEP=$4; PORT=$5; N=${6:-50}; SERVER_ARGS=${SERVER_ARGS:-}
B=/mnt/secondary/so101_r2s/brev; M=/mnt/secondary/so101_r2s/train_runs; E=/mnt/secondary/so101_r2s/eval
CK=$RUN/checkpoints/$STEP/pretrained_model
exec >> $B/eval_$NAME.log 2>&1
echo "[evalck] $(date +%T) waiting for $H:$CK"
until timeout 30 ssh $H "test -f ~/r2s/train_runs/$CK/policy_postprocessor.json" 2>/dev/null; do sleep 120; done
sleep 60
mkdir -p $M/$RUN/checkpoints/$STEP
rsync -a $H:r2s/train_runs/$CK $M/$RUN/checkpoints/$STEP/ || { echo "[evalck] pull failed"; exit 1; }
ssh r2s-sim "mkdir -p /mnt/secondary/so101_r2s/train_runs/$RUN/checkpoints/$STEP"
rsync -a $M/$CK r2s-sim:/mnt/secondary/so101_r2s/train_runs/$RUN/checkpoints/$STEP/ || { echo "[evalck] push failed"; exit 1; }
echo "[evalck] $(date +%T) checkpoint on r2s-sim"
ssh r2s-sim "cd ~/so101_r2s2r && (nohup ~/lr/bin/python scripts/policy_server.py --ckpt /mnt/secondary/so101_r2s/train_runs/$CK \
  --port $PORT $SERVER_ARGS > ~/logs/server_$NAME.log 2>&1 < /dev/null & echo \$! > ~/logs/server_$NAME.pid)"
until ssh r2s-sim "grep -q 'listening\|Error' ~/logs/server_$NAME.log"; do sleep 10; done
ssh r2s-sim "tail -n 1 ~/logs/server_$NAME.log"
ssh r2s-sim "cd ~/so101_r2s2r && (nohup ./lab.sh -p scripts/eval_policy.py --episodes $N --port $PORT --video 6 --out $E/$NAME \
  --viz none > ~/logs/eval_$NAME.log 2>&1 < /dev/null &)"
until ssh r2s-sim "grep -aq '\[eval\] by cube size\|Traceback' ~/logs/eval_$NAME.log" 2>/dev/null; do sleep 60; done
sleep 20
ssh r2s-sim "kill \$(cat ~/logs/server_$NAME.pid); cp ~/logs/eval_$NAME.log $E/$NAME/eval.log"
mkdir -p $E/$NAME && rsync -a r2s-sim:$E/$NAME/ $E/$NAME/
grep -a "\[eval\] success\|by cube\|Traceback" $E/$NAME/eval.log
echo "[evalck] $(date +%T) DONE $NAME"

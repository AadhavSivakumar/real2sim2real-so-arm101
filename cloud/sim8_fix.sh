#!/bin/bash
# r2s-sim8 was bootstrapped with the old script (checkpoint dir never created): once its recorders have started, give it the
# ACT v3a checkpoint and restart the 4 takeover recorders (same args as bootstrap_v4.sh, IDX 5).
H=r2s-sim8; B=/mnt/secondary/so101_r2s/brev; CK=/mnt/secondary/so101_r2s/train_runs/act_v3a/checkpoints/080000/pretrained_model
S=/mnt/secondary/so101_r2s/sim_demos; DL=1791588900; IDX=5
exec >> $B/$H.log 2>&1
until grep -q "v4boot.*recorders started\|v4boot.*failed" $B/$H.log; do sleep 15; done
echo "[fix] $(date +%T) restarting takeover recorders with the checkpoint"
ssh $H 'for f in ~/logs/v4_takeover_*.pid; do pkill -P $(cat $f) 2>/dev/null; kill $(cat $f) 2>/dev/null; done; sleep 3; pkill -f "[r]ecord_takeover.py"; pkill -f "[p]olicy_server.py"; mkdir -p '$CK
rsync -a $CK/ $H:$CK/ || { echo "[fix] ckpt copy failed"; exit 1; }
for k in 1 2 3 4; do
  SPL=$([ $k -le 2 ] && echo "SO101_SPLAT=1" || echo "SO101_SPLAT=")
  ssh $H "cd ~/so101_r2s2r && ($SPL PORT=$((6300 + k)) CKPT=$CK nohup bash $B/v4_record.sh takeover $S/sim_v4_takeover_${H}_$k 60 $DL --seed $((450000 + IDX * 10000 + k * 1000)) > ~/logs/v4_takeover_$k.log 2>&1 < /dev/null & echo \$! > ~/logs/v4_takeover_$k.pid)"
  sleep 20
done
echo "[fix] $(date +%T) takeover recorders restarted"

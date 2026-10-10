#!/bin/bash
# Usage: bootstrap_v4.sh HOST ROLE IDX  -- new render box for the v4 data: Isaac (bootstrap_sim.sh), LeRobot venv ~/lr (r2s-sim's
# freeze; policy servers + conversion), the ACT v3a 80k checkpoint, then 4 recorders until the shared deadline.
#   ROLE dart: 4 x DART expert demos with the splat; ROLE takeover: 4 x takeovers of ACT v3a (2 with the splat), own servers.
H=$1; ROLE=$2; IDX=$3; DL=1791588900
B=/mnt/secondary/so101_r2s/brev; CK=/mnt/secondary/so101_r2s/train_runs/act_v3a/checkpoints/080000/pretrained_model
bash $B/bootstrap_sim.sh $H
exec >> $B/$H.log 2>&1
tail -n 4 $B/$H.log | grep -q "^\[rec\] done" || { echo "[v4boot] bootstrap failed"; exit 1; }
scp -q $B/lr_freeze.txt $H:/tmp/lr_freeze.txt
ssh $H 'export PATH=$HOME/.local/bin:$PATH; [ -x ~/lr/bin/python ] || uv venv --python 3.12 --seed ~/lr -q; grep -v "^-e " /tmp/lr_freeze.txt > /tmp/req.txt
  uv pip install --python ~/lr/bin/python -q -r /tmp/req.txt --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  ~/lr/bin/python -c "import lerobot; print(\"lr ok\", lerobot.__version__)"; mkdir -p ~/logs ~/r2s/logs /mnt/secondary/so101_r2s/brev /mnt/secondary/so101_r2s/train_runs/act_v3a/checkpoints/080000/pretrained_model' || { echo "[v4boot] lr failed"; exit 1; }
rsync -a $B/v4_record.sh $B/v4_convert.sh $H:/mnt/secondary/so101_r2s/brev/
[ "$ROLE" = takeover ] && rsync -a $CK/ $H:$CK/
S=/mnt/secondary/so101_r2s/sim_demos
for k in 1 2 3 4; do
  if [ "$ROLE" = dart ]; then
    ssh $H "cd ~/so101_r2s2r && (SO101_SPLAT=1 SEED=$((600000 + IDX * 10000 + k * 1000)) STD=0.08 nohup bash /mnt/secondary/so101_r2s/brev/v4_record.sh dart $S/sim_v4_dart_${H}_$k 100 $DL > ~/logs/v4_dart_$k.log 2>&1 < /dev/null & echo \$! > ~/logs/v4_dart_$k.pid)"
  else
    SPL=$([ $k -le 2 ] && echo "SO101_SPLAT=1" || echo "SO101_SPLAT=")
    ssh $H "cd ~/so101_r2s2r && ($SPL PORT=$((6300 + k)) CKPT=$CK nohup bash /mnt/secondary/so101_r2s/brev/v4_record.sh takeover $S/sim_v4_takeover_${H}_$k 60 $DL --seed $((450000 + IDX * 10000 + k * 1000)) > ~/logs/v4_takeover_$k.log 2>&1 < /dev/null & echo \$! > ~/logs/v4_takeover_$k.pid)"
  fi
  sleep 20
done
echo "[v4boot] $(date +%T) $ROLE recorders started (deadline $(date -d @$DL +%T))"

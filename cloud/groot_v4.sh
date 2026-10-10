#!/bin/bash
# GR00T N1.7 co-trained on cotrain_v4 on a second H100 (r2s-h100c): set it up like r2s-h100b (groot_setup.sh + the Cosmos-Reason2-2B
# tokenizer/processor files, no token on the box), wait for cotrain_v4 on r2s-h100 (v4_final.sh), copy it over via this desktop,
# then train 9000 steps x batch 64 (~3 h at 1.14 s/step). Log: brev/groot_v4.log
B=/mnt/secondary/so101_r2s/brev; H=r2s-h100c; L=/mnt/secondary/so101_r2s/lerobot_datasets/cotrain_v4
exec >> $B/groot_v4.log 2>&1
echo "[gv4] $(date +%T) create $H"
brev ls 2>/dev/null | grep -q "^ *$H " || brev create $H --type hyperstack_H100,massedcompute_H100,scaleway_H100 --detached
H=$H bash $B/groot_setup.sh
grep -q "\[groot\].*setup done" $B/groot.log || { echo "[gv4] setup failed"; exit 1; }
ssh $H 'mkdir -p ~/.cache/huggingface/hub' && rsync -a /mnt/secondary/so101_r2s/hf_cosmos/hub/models--nvidia--Cosmos-Reason2-2B $H:.cache/huggingface/hub/ \
  || { echo "[gv4] cosmos upload failed"; exit 1; }
echo "[gv4] $(date +%T) $H ready; waiting for cotrain_v4 on r2s-h100"
until timeout 30 ssh r2s-h100 'test -f ~/r2s/lerobot_datasets/cotrain_v4/meta/v4_sources.json' 2>/dev/null; do sleep 60; done
sleep 30
echo "[gv4] $(date +%T) copying cotrain_v4"
mkdir -p $L && rsync -aH r2s-h100:r2s/lerobot_datasets/cotrain_v4/ $L/ && rsync -aH $L/ $H:r2s/lerobot_datasets/cotrain_v4/ \
  || { echo "[gv4] copy failed"; exit 1; }
echo "[gv4] $(date +%T) cotrain_v4 on $H ($(du -sh $L | cut -f1)); launching"
H=$H bash $B/groot_train.sh groot_v4 cotrain_v4 9000 64 2250
sleep 600
ssh $H 'grep -ac Traceback ~/r2s/train_groot_v4.log; grep -a -o "INFO.*step:[0-9K]* .*" ~/r2s/train_groot_v4.log | tail -1'
echo "[gv4] $(date +%T) GR00T v4 training"

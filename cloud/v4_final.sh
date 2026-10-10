#!/bin/bash
# After the 19:35 deadline: convert every box's v4 recordings on that box (v4_convert.sh), finish r2s-sim's v3 parts, gather all
# parts + the undistorted real datasets on r2s-h100 (via this desktop: archive copy in /mnt/secondary/so101_r2s/v4parts),
# aggregate cotrain_v4 there (v4_aggregate.py, real ~25% of frames) and start ACT v4 (bf16). Log: brev/v4_final.log
B=/mnt/secondary/so101_r2s/brev; P=/mnt/secondary/so101_r2s/v4parts; H=r2s-h100; STEPS=${STEPS:-80000}
BOXES="r2s-sim r2s-sim4 r2s-sim5 r2s-sim6 r2s-sim7 r2s-sim8"
exec >> $B/v4_final.log 2>&1
echo "[final] $(date +%T) waiting for the 19:35 deadline"
while [ $(date +%s) -lt 1791589020 ]; do sleep 60; done
echo "[final] $(date +%T) waiting for the recorders on $BOXES"
for h in $BOXES; do
  until timeout 30 ssh -o ConnectTimeout=15 $h '! pgrep -f "[v]4_record.sh" > /dev/null && ! pgrep -f "[e]xtend_sim" > /dev/null' 2>/dev/null; do sleep 30; done
  echo "[final] $(date +%T) $h: recorders done: $(timeout 30 ssh $h 'for d in /mnt/secondary/so101_r2s/sim_demos/sim_v4_*; do echo -n "$(basename $d)=$(ls -d $d/ep_* 2>/dev/null | wc -l) "; done' 2>/dev/null)"
done
echo "[final] $(date +%T) converting on every box"
for h in $BOXES; do
  (timeout 5400 ssh $h 'S=/mnt/secondary/so101_r2s/sim_demos; B=/mnt/secondary/so101_r2s/brev
     pkill -f "[p]olicy_server.py"; bash $B/v4_convert.sh 14 40 $S/sim_v4_* > ~/logs/v4_final_convert.log 2>&1
     if [ -d $S/v3_expert_s1 ]; then
       for f in ~/logs/v4_convert_v3.pid ~/logs/v4_convert_v3b.pid; do [ -f $f ] && while kill -0 $(cat $f) 2>/dev/null; do sleep 15; done; done
       bash $B/v4_convert.sh 14 84 $S/v3_expert_s1 $S/v3_expert_s2 $S/v3_expert_s3 $S/v3_expert_s4 >> ~/logs/v4_final_convert.log 2>&1
       bash $B/v4_convert.sh 14 63 $(for k in 5 6 7 8 9 10 11 12; do echo $S/v3_expert_s$k; done) >> ~/logs/v4_final_convert.log 2>&1
     fi
     cd ~/r2s/sim_demos/parts && for p in *; do [ -f ~/r2s/logs/convert_$p.done ] || echo "MISSING $p"; done; echo "converted $(ls | wc -l) parts"' 2>&1 | sed "s/^/[final] $h: /") &
done
wait
mkdir -p $P
for h in $BOXES; do
  timeout 3600 ssh $h 'cd ~/r2s/lerobot_datasets && for d in v4p_*; do [ -f ~/r2s/logs/convert_$d.done ] && echo $d; done' > $P/.list_$h 2>/dev/null
  rsync -a --files-from=<(sed 's|$|/|' $P/.list_$h) -r $h:r2s/lerobot_datasets/ $P/ && echo "[final] $(date +%T) pulled $(wc -l < $P/.list_$h) parts from $h"
done
rsync -a r2s-sim:r2s/lerobot_datasets/cube_cup_real_v2_undist $P/ && rsync -a /mnt/secondary/so101_r2s/lerobot_datasets/real_cube_cup_v1_undist $P/
echo "[final] $(date +%T) $(ls -d $P/v4p_* | wc -l) parts, $(du -sh $P | cut -f1) -> $H"
ssh $H 'mkdir -p ~/r2s/lerobot_datasets'
rsync -a $P/ $H:r2s/lerobot_datasets/ && rsync -a $B/v4_aggregate.py $H:r2s/ || { echo "[final] push failed"; exit 1; }
parts=$(cd $P && ls -d v4p_* | tr '\n' ' ')
ssh $H "cd ~/r2s && ~/lr/bin/python v4_aggregate.py --out cotrain_v4 --real_frac 0.25 --sims $parts --reals real_cube_cup_v1_undist cube_cup_real_v2_undist 2>&1 | grep -av 'Auto-inserting\|Starting second pass' | tail -12"
ssh $H 'test -f ~/r2s/lerobot_datasets/cotrain_v4/meta/v4_sources.json' || { echo "[final] aggregate failed"; exit 1; }
ssh $H "cd ~/r2s && (nohup ~/lr/bin/lerobot-train --dataset.repo_id=local/cotrain_v4 --dataset.root=/home/\$USER/r2s/lerobot_datasets/cotrain_v4 \
  --dataset.video_backend=torchcodec --dataset.image_transforms.enable=true --policy.device=cuda --policy.push_to_hub=false \
  --wandb.enable=false --num_workers=16 --log_freq=200 --policy.type=act --policy.chunk_size=50 --policy.n_action_steps=30 \
  --accelerator.mixed_precision=bf16 --output_dir=train_runs/act_v4 --job_name=act_v4 --steps=$STEPS --batch_size=16 --save_freq=10000 \
  > ~/r2s/train_act_v4.log 2>&1 < /dev/null & echo \$! > ~/r2s/train_act_v4.pid)"
sleep 240
ssh $H 'grep -ac Traceback ~/r2s/train_act_v4.log; grep -a -o "INFO.*step:[0-9K]* .*" ~/r2s/train_act_v4.log | tail -1'
echo "[final] $(date +%T) ACT v4 training started"

#!/bin/bash
# Usage: bootstrap_sim.sh HOST   -- wait for a new Brev box, install Isaac (setup_sim.sh), copy the repo (remote lab.sh from
# r2s-sim) + assets + the real dataset, then run a 1-episode recording check. Log: /mnt/secondary/so101_r2s/brev/HOST.log
H=$1
B=/mnt/secondary/so101_r2s/brev
R=/home/asubuntudesktop/TestingGrounds/so101_r2s2r
exec >> $B/$H.log 2>&1
echo "[boot] $(date +%T) waiting for $H"
brev refresh > /dev/null 2>&1
until timeout 20 ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 $H true 2>/dev/null; do sleep 20; brev refresh > /dev/null 2>&1; done
echo "[boot] $(date +%T) reachable"
scp -q -o StrictHostKeyChecking=no $B/setup_sim.sh $H:setup_sim.sh
timeout 3600 ssh $H 'bash ~/setup_sim.sh > ~/setup_sim.log 2>&1; tail -1 ~/setup_sim.log' || { echo "[boot] setup failed"; exit 1; }
echo "[boot] $(date +%T) isaac installed"
ssh $H 'mkdir -p ~/so101_r2s2r ~/r2s_work/logs /mnt/secondary/so101_r2s/sim_demos /mnt/secondary/so101_r2s/lerobot_datasets'
rsync -aL --exclude lab.sh --exclude lab.sh.desktop --exclude __pycache__ --exclude 'calib/data/*/' --exclude 'calib/data/pre_calib_patch_backup' $R/ $H:so101_r2s2r/
scp -q -3 r2s-sim:so101_r2s2r/lab.sh $H:so101_r2s2r/lab.sh
ssh $H 'chmod +x ~/so101_r2s2r/lab.sh'
rsync -a --exclude debug /mnt/secondary/so101_r2s/assets/ $H:/mnt/secondary/so101_r2s/assets/
ssh $H 'cd ~/so101_r2s2r && ./lab.sh -p scripts/record_demos.py --episodes 1 --out /tmp/rd_check --viz none > ~/r2s_work/logs/rd_check.log 2>&1; grep -E "^\[rec\] done|Traceback" ~/r2s_work/logs/rd_check.log'
echo "[boot] $(date +%T) READY"

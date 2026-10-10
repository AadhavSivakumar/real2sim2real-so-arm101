#!/bin/bash
# Usage: v3_data.sh TAG SHARDS...   e.g.  v3_data.sh a "1 2 3 4"
# Pull sim v3 expert shards (record_demos.py dirs) from the sim boxes via this machine to r2s-train, convert each shard
# to LeRobot (new units.py, in parallel), then aggregate them with the undistorted real dataset (real duplicated so it is
# ~REAL_FRAC of the frames) into ~/r2s/lerobot_datasets/cotrain_v3$TAG. Log: /mnt/secondary/so101_r2s/brev/v3_data_$TAG.log
TAG=$1; SHARDS=$2; REAL_FRAC=${REAL_FRAC:-0.10}; SPLIT=${SPLIT:-2}
B=/mnt/secondary/so101_r2s/brev
R=/home/asubuntudesktop/TestingGrounds/so101_r2s2r
L=/mnt/secondary/so101_r2s/sim_demos
T=r2s-train
exec >> $B/v3_data_$TAG.log 2>&1
set -o pipefail
echo "[v3] $(date +%T) start: shards $SHARDS"
host_of() { if [ $1 -le 4 ]; then echo r2s-sim; elif [ $1 -le 8 ]; then echo r2s-sim2; else echo r2s-sim3; fi; }
for k in $SHARDS; do
  rsync -a $(host_of $k):/mnt/secondary/so101_r2s/sim_demos/v3_expert_s$k $L/ || { echo "[v3] pull s$k failed"; exit 1; }
done
echo "[v3] $(date +%T) pulled to $L"
rsync -aL --exclude lab.sh --exclude lab.sh.desktop --exclude __pycache__ --exclude 'calib/data/*/' --exclude 'calib/data/pre_calib_patch_backup' $R/ $T:so101_r2s2r/
ssh $T 'mkdir -p ~/r2s/sim_demos ~/r2s/logs'
for k in $SHARDS; do rsync -a $L/v3_expert_s$k $T:r2s/sim_demos/; done
if ! ssh $T 'test -f ~/r2s/lerobot_datasets/real_cube_cup_v1_undist/meta/info.json'; then
  rsync -a r2s-real:r2s/lerobot_datasets/real_cube_cup_v1_undist /mnt/secondary/so101_r2s/lerobot_datasets/
  rsync -a /mnt/secondary/so101_r2s/lerobot_datasets/real_cube_cup_v1_undist $T:r2s/lerobot_datasets/
fi
echo "[v3] $(date +%T) pushed to $T"
# convert each shard in SPLIT parts in parallel (symlinked subsets of its episodes; skip parts already converted)
ssh $T "cd ~/so101_r2s2r && for k in $SHARDS; do for p in \$(seq 0 $((SPLIT - 1))); do
  n=sim_v3_s\${k}_\$p; [ -f ~/r2s/lerobot_datasets/\$n/meta/info.json ] && continue
  rm -rf ~/r2s/lerobot_datasets/\$n ~/r2s/sim_demos/parts/\$n; mkdir -p ~/r2s/sim_demos/parts/\$n
  i=0; for e in ~/r2s/sim_demos/v3_expert_s\$k/ep_*; do [ \$((i % $SPLIT)) = \$p ] && ln -s \$e ~/r2s/sim_demos/parts/\$n/; i=\$((i + 1)); done
  ~/lr/bin/python scripts/convert_to_lerobot.py --src ~/r2s/sim_demos/parts/\$n --name \$n \
    --root ~/r2s/lerobot_datasets > ~/r2s/logs/convert_\$n.log 2>&1 &
done; done; wait; tail -qn 1 ~/r2s/logs/convert_sim_v3_s*.log" || { echo "[v3] convert failed"; exit 1; }
for k in $SHARDS; do for p in $(seq 0 $((SPLIT - 1))); do
  ssh $T "test -f ~/r2s/lerobot_datasets/sim_v3_s${k}_$p/meta/info.json" || { echo "[v3] sim_v3_s${k}_$p missing"; exit 1; }
done; done
echo "[v3] $(date +%T) converted"
ssh $T "~/lr/bin/python - <<EOF
import json, shutil
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
root = Path.home() / 'r2s/lerobot_datasets'
sims = [root / f'sim_v3_s{k}_{p}' for k in '$SHARDS'.split() for p in range($SPLIT)]
real = root / 'real_cube_cup_v1_undist'
frames = lambda p: json.loads((p / 'meta/info.json').read_text())['total_frames']
fs, fr = sum(frames(p) for p in sims), frames(real)
dup = max(1, round($REAL_FRAC * fs / ((1 - $REAL_FRAC) * fr)))
print(f'[v3] sim frames {fs}, real frames {fr}, real x{dup} -> real share {dup * fr / (fs + dup * fr):.3f}', flush=True)
reals = []
for i in range(dup):  # physical copies: each aggregated source needs its own root
    p = root / f'_real_undist_dup{i}'
    if not p.exists():
        shutil.copytree(real, p)
    reals.append(p)
out = root / 'cotrain_v3$TAG'
if out.exists():
    shutil.rmtree(out)
srcs = sims + reals
aggregate_datasets([f'local/{p.name}' for p in srcs], 'local/cotrain_v3$TAG', roots=srcs, aggr_root=out)
print('[v3] aggregated', {k: v for k, v in json.loads((out / 'meta/info.json').read_text()).items() if k.startswith('total')})
EOF" || { echo "[v3] aggregate failed"; exit 1; }
echo "[v3] $(date +%T) DONE cotrain_v3$TAG"

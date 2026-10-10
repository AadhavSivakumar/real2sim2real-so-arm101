#!/bin/bash
# ON r2s-sim: convert record_demos.py / record_takeover.py output dirs into LeRobot datasets ("parts"): each SRC is split
# into contiguous symlinked subsets of <= PART episodes, J conversions run in parallel (niced, so running Isaac recorders
# keep priority). A part is skipped when ~/r2s/logs/convert_<name>.done exists (written only after a clean finish;
# LeRobotDataset.create writes meta/info.json up front, so that file is no completion marker).
#   v4_convert.sh J PART SRC_DIR...    ->  ~/r2s/lerobot_datasets/v4p_<basename SRC>_<k>  (log ~/r2s/logs/convert_<name>.log)
J=$1; PART=$2; shift 2
R=$HOME/r2s
mkdir -p "$R/sim_demos/parts" "$R/lerobot_datasets" "$R/logs"
todo=$(mktemp)
for src in "$@"; do
  b=$(basename "$src")
  mapfile -t eps < <(ls -d "$src"/ep_* | sort)
  n=${#eps[@]}
  for p in $(seq 0 $(( (n + PART - 1) / PART - 1 ))); do
    name=v4p_${b}_$p
    [ -f "$R/logs/convert_$name.done" ] && continue
    rm -rf "$R/lerobot_datasets/$name" "$R/sim_demos/parts/$name"
    mkdir -p "$R/sim_demos/parts/$name"
    for e in "${eps[@]:$((p * PART)):$PART}"; do ln -s "$e" "$R/sim_demos/parts/$name/"; done
    echo "$name" >> "$todo"
  done
done
echo "[conv] $(date +%T) $(wc -l < "$todo") parts to convert, $J in parallel"
cd ~/so101_r2s2r || exit 1
xargs -a "$todo" -P "$J" -I{} nice -n 15 bash -c '~/lr/bin/python scripts/convert_to_lerobot.py --src ~/r2s/sim_demos/parts/{} \
  --name {} --root ~/r2s/lerobot_datasets > ~/r2s/logs/convert_{}.log 2>&1 && touch ~/r2s/logs/convert_{}.done \
  && echo "[conv] $(date +%T) {}: $(tail -n 1 ~/r2s/logs/convert_{}.log)" || echo "[conv] $(date +%T) {} FAILED"'
rm -f "$todo"
echo "[conv] $(date +%T) done"

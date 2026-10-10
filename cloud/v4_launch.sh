#!/bin/bash
# Start the v4 recorders on r2s-sim (staggered; each in v4_record.sh's retry loop with the shared deadline).
#   v4_launch.sh DEADLINE_EPOCH NAME...      NAME in: takeover_ns1 takeover_ns2 takeover_sp1 takeover_sp2 dart_sp1 dart_sp2 dart_sp3
# Logs ~/logs/v4_<NAME>.log, PID files ~/logs/v4_<NAME>.pid (the wrapper's bash), outputs sim_demos/sim_v4_<NAME>.
# ns = no splat (plain backdrop, as v3), sp = SO101_SPLAT=1. Takeovers: ACT v3a 80k; takeover_ns1 uses the server already
# running on :6210 (track A2), the others start their own server (ports 6222-6224) that dies with the recorder.
DL=$1; shift
S=/mnt/secondary/so101_r2s/sim_demos
B=/mnt/secondary/so101_r2s/brev
CK=/mnt/secondary/so101_r2s/train_runs/act_v3a/checkpoints/080000/pretrained_model
L=$HOME/logs
go() {
  local name=$1; shift
  (nohup env "$@" > "$L/v4_$name.log" 2>&1 < /dev/null & echo $! > "$L/v4_$name.pid")
  echo "[launch] $(date +%T) v4_$name pid $(cat "$L/v4_$name.pid")"
}
first=1
for n in "$@"; do
  [ $first = 1 ] || sleep 45  # stagger the Isaac start-ups (host-memory peak)
  first=0
  case $n in
    takeover_ns1) go $n -u SO101_SPLAT PORT=6210 CKPT= bash $B/v4_record.sh takeover $S/sim_v4_$n 125 $DL --seed 410000 ;;
    takeover_ns2) go $n -u SO101_SPLAT PORT=6222 CKPT=$CK bash $B/v4_record.sh takeover $S/sim_v4_$n 125 $DL --seed 420000 ;;
    takeover_sp1) go $n SO101_SPLAT=1 PORT=6223 CKPT=$CK bash $B/v4_record.sh takeover $S/sim_v4_$n 125 $DL --seed 430000 ;;
    takeover_sp2) go $n SO101_SPLAT=1 PORT=6224 CKPT=$CK bash $B/v4_record.sh takeover $S/sim_v4_$n 125 $DL --seed 440000 ;;
    dart_sp1) go $n SO101_SPLAT=1 SEED=510000 STD=0.08 bash $B/v4_record.sh dart $S/sim_v4_$n 500 $DL ;;
    dart_sp2) go $n SO101_SPLAT=1 SEED=520000 STD=0.08 bash $B/v4_record.sh dart $S/sim_v4_$n 500 $DL ;;
    dart_sp3) go $n SO101_SPLAT=1 SEED=530000 STD=0.08 bash $B/v4_record.sh dart $S/sim_v4_$n 500 $DL ;;
    *) echo "[launch] unknown $n" ;;
  esac
done

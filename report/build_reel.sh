#!/bin/bash
# Highlight reel: Manim title card, a few seconds of each step video, end card (CPU only, niced).
set -euo pipefail
R=/mnt/secondary/so101_r2s/report; O=$R/out; TXT=$R/txt; mkdir -p "$TXT"
BOLD=/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf; REG=/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf
AMBER=0xF5A524; VIOLET=0xB7A6F2; BG=0x101216
t() { printf '%s' "$2" > "$TXT/$1.txt"; echo "$TXT/$1.txt"; }
dt() { echo "drawtext=expansion=none:fontfile=$1:textfile=$(t $2 "$3"):fontsize=$4:fontcolor=$5:x=$6:y=$7"; }

# step file, start s, duration s
SEGS=("step01_capture 1 3" "step02_splat 1.5 4" "step03_real2sim 0.5 4" "step04_sim_demos 0.9 4" "step05_recovery 5 4"
      "step06_data_audit 5.5 3" "step07_real_demos 1 4" "step08_iterations 2 5" "step09_sim_eval 4 4"
      "step10_real_arm 3 6")
IN=(-i "$O/intro_manim.mp4")   # title card rendered with Manim (manim/intro.py, see manim/README)
for s in "${SEGS[@]}"; do set -- $s; IN+=(-ss "$2" -t "$3" -i "$O/$1.mp4"); done
IN+=(-f lavfi -t 4.5 -i "color=c=$BG:s=1280x720:r=30")
N=$(( ${#SEGS[@]} + 2 )); LAST=$(( N - 1 ))

F="[0:v]fps=30,format=yuv420p,setsar=1[c0];"
for i in $(seq 1 ${#SEGS[@]}); do F+="[$i:v]fps=30,format=yuv420p,setsar=1,fade=t=in:st=0:d=0.25[c$i];"; done
F+="[$LAST:v]$(dt BOLD r9a '88%' 60 $AMBER 120 170),$(dt REG r9b 'sim success on 50 held-out randomized scenes' 28 white 330 196),\
$(dt BOLD r9c '5 / 6' 60 $AMBER 120 270),$(dt REG r9d 'runs on the real arm  (real-only baselines: 6 / 21)' 28 white 330 296),\
$(dt BOLD r9e '40 ms' 60 $AMBER 120 370),$(dt REG r9f 'policy inference on a Jetson Orin Nano' 28 white 330 396),\
$(dt REG r9g '4,013 sim demos + 40 real demos  ·  trained on one H100' 22 white@0.7 120 500),\
fade=t=in:st=0:d=0.4,fade=t=out:st=3.9:d=0.6,format=yuv420p,setsar=1[c$LAST];"
for i in $(seq 0 $LAST); do F+="[c$i]"; done
F+="concat=n=$N:v=1:a=0[v]"
F=${F//fontfile=ARROW/fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf}; F=${F//fontfile=BOLD/fontfile=$BOLD}; F=${F//fontfile=REG/fontfile=$REG}
nice -n 19 ffmpeg -v error -y -threads 8 "${IN[@]}" -filter_complex "$F" -map "[v]" \
  -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p -r 30 -movflags +faststart -an "$O/step00_reel.mp4"
ffprobe -v error -show_entries format=duration,size -of csv=p=0 "$O/step00_reel.mp4"

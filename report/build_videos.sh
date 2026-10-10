#!/bin/bash
# Builds one short video per pipeline step for the project report (CPU only, niced).
# Usage: build_videos.sh [step ...]   (default: all)
set -euo pipefail
R=/mnt/secondary/so101_r2s
OUT=$R/report/out; TMP=$R/report/clips; TXT=$R/report/txt
mkdir -p "$OUT" "$TMP" "$TXT"
BOLD=/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf
REG=/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf
MONO=/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf
AMBER=0xF5A524; VIOLET=0xB7A6F2; BG=0x101216
ENC=(-c:v libx264 -preset slow -crf 25 -pix_fmt yuv420p -r 30 -movflags +faststart -an)
FF=(nice -n 19 ffmpeg -v error -y -threads 8)
TM="zscale=t=linear:npl=203,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"

# t KEY TEXT -> path of a text file (drawtext reads it verbatim, so no escaping)
t() { printf '%s' "$2" > "$TXT/$1.txt"; echo "$TXT/$1.txt"; }

# hdr STEP TITLE CAPTION KEY -> drawtext chain for a 1280x720 frame: top band (step + title), bottom caption
hdr() {
  local s; s=$(t "$4_s" "STEP $1"); local ti; ti=$(t "$4_t" "$2"); local c; c=$(t "$4_c" "$3")
  echo "drawbox=y=0:w=iw:h=60:color=black@0.62:t=fill,drawbox=y=ih-52:w=iw:h=52:color=black@0.62:t=fill,\
drawtext=expansion=none:fontfile=$BOLD:textfile=$s:fontsize=20:fontcolor=$AMBER:x=28:y=21,\
drawtext=expansion=none:fontfile=$BOLD:textfile=$ti:fontsize=27:fontcolor=white:x=124:y=14,\
drawtext=expansion=none:fontfile=$REG:textfile=$c:fontsize=19:fontcolor=white@0.92:x=28:y=h-36"
}
# tag KEY TEXT X Y COLOR -> small label with a dark box
tag() { local f; f=$(t "$1" "$2"); echo "drawtext=expansion=none:fontfile=$BOLD:textfile=$f:fontsize=19:fontcolor=$5:x=$3:y=$4:box=1:boxcolor=black@0.6:boxborderw=7"; }
# cat_clips OUT IN... -> concatenate same-format clips without re-encoding
cat_clips() { local o=$1; shift; local l=$TMP/list_$$.txt; : > "$l"; for f in "$@"; do echo "file '$f'" >> "$l"; done
  "${FF[@]}" -f concat -safe 0 -i "$l" -c copy -movflags +faststart "$o"; rm -f "$l"; }

step1() {  # phone capture (first 29 s only: later frames show packages with shipping labels)
  "${FF[@]}" -ss 0 -t 29 -i ~/Downloads/IMG_3527.mp4 -filter_complex "\
[0:v]fps=12,$TM,setpts=PTS/2.5,scale=-2:720,split[a][b];\
[a]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,boxblur=30:3,eq=brightness=-0.18[bg];\
[bg][b]overlay=(W-w)/2:0,$(hdr 1 'Capture the real workspace' 'A 45 s phone walk-around (1080x1920, 60 fps HDR, tone-mapped) is the only input to the 3D reconstruction.  Shown at 2.5x.' s1),\
$(tag s1a 'iPhone video  IMG_3527' 28 80 white)[v]" -map "[v]" "${ENC[@]}" "$OUT/step01_capture.mp4"
}

step2() {  # splat orbit (rendered on CPU by splat_render/)
  "${FF[@]}" -i $R/report/splat_render/splat_orbit.mp4 -filter_complex "\
[0:v]fps=30,format=yuv420p,$(hdr 2 'Reconstruct a Gaussian splat' 'The room around the desk as 346,881 Gaussians, in the robot frame.  Desk top, arm, cube and cup are simulated objects.' s2)[v]" \
    -map "[v]" "${ENC[@]}" "$OUT/step02_splat.mp4"
}

wipe() {  # wipe IMG KEY DUR CROPY LABEL_L LABEL_R -> 1280x720 clip with an oscillating real/sim wipe
  local img=$1 key=$2 dur=$3 cy=$4
  "${FF[@]}" -loop 1 -framerate 30 -t "$dur" -i "$img" -filter_complex "\
[0]crop=640:440:0:$cy,scale=818:562,format=gbrp[r];[0]crop=640:440:640:$cy,scale=818:562,format=gbrp[s];\
[r][s]blend=all_expr='if(lt(abs(X-W*(0.5+0.44*sin(2*PI*T/$dur))),2),255,if(lt(X,W*(0.5+0.44*sin(2*PI*T/$dur))),A,B))',format=yuv420p[w];\
color=c=$BG:s=1280x720:r=30:d=$dur[bg];[bg][w]overlay=231:83:shortest=1,\
$(tag ${key}L "$5" 241 93 $AMBER),$(tag ${key}R "$6" 'w-241-tw-4' 93 $VIOLET),\
$(hdr 3 'Real-to-sim alignment' "$7" $key)[v]" -map "[v]" "${ENC[@]}" "$TMP/$key.mp4"
}
step3() {
  local C=$R/brev/splat_check cap='Calibrated cameras + robot: the real frame (left of the line) vs. the Isaac Sim render with the splat background (right).'
  wipe $C/cmp_ep000.png w0 6 40 'REAL overhead camera' 'SIM  Isaac Sim + splat' "$cap"
  wipe $C/cmp_ep003.png w1 5 40 'REAL overhead camera' 'SIM  Isaac Sim + splat' "$cap"
  wipe $C/cmp_ep007.png w2 5 40 'REAL overhead camera' 'SIM  Isaac Sim + splat' "$cap"
  wipe $C/cmp_ep003.png w3 5 520 'REAL wrist camera' 'SIM wrist camera' 'Wrist camera: lens distortion measured with a ChArUco board, mount pose fitted, real frames undistorted before the policy.'
  "${FF[@]}" -loop 1 -framerate 30 -t 4 -i $C/refine_overhead.jpg -filter_complex "\
[0]scale=1280:-2[i];color=c=$BG:s=1280x720:r=30:d=4[bg];[bg][i]overlay=0:(H-h)/2:shortest=1,\
$(tag w4a 'real' 12 200 $AMBER),$(tag w4b 'splat render' 440 200 $VIOLET),$(tag w4c 'overlay' 867 200 white),\
$(hdr 3 'Real-to-sim alignment' 'Splat pose refined by matching renders to the real overhead frame (scale, rotation, colour gain).' w4)[v]" \
    -map "[v]" "${ENC[@]}" "$TMP/w4.mp4"
  cat_clips "$OUT/step03_real2sim.mp4" $TMP/w0.mp4 $TMP/w1.mp4 $TMP/w2.mp4 $TMP/w3.mp4 $TMP/w4.mp4
}

step4() {  # expert demos + randomization grid (only episodes whose cube and cup spawned on the real desk)
  "${FF[@]}" -i $R/videos_v4/1_splat_expert_demos.mp4 -filter_complex "\
[0:v]split[x][y];[x]trim=start=8.05:end=14.65,setpts=PTS-STARTPTS[p];[y]trim=start=23.95:end=32.0,setpts=PTS-STARTPTS[q];\
[p][q]concat=n=2:v=1:a=0,setpts=PTS/1.33,fps=30[d];color=c=$BG:s=1280x720:r=30[bg];[bg][d]overlay=0:120:shortest=1,\
$(tag e4a 'overhead camera' 10 130 white),$(tag e4b 'wrist camera' 650 130 white),\
$(hdr 4 'Generate demonstrations in sim' 'Scripted expert in Isaac Lab with domain randomization and DART action noise (labels stay clean).  1.33x.' e4)[v]" \
    -map "[v]" "${ENC[@]}" "$TMP/e4.mp4"
  # spawn grid after the desk fix; the per-tile debug labels are covered
  "${FF[@]}" -loop 1 -framerate 30 -t 4 -i $R/videos_v4/desk_fix_grid.jpg -filter_complex "\
[0]drawbox=y=0:w=iw:h=24:color=0x1b2016:t=fill,drawbox=y=240:w=iw:h=24:color=0x1b2016:t=fill,\
drawbox=y=480:w=iw:h=24:color=0x1b2016:t=fill,drawbox=y=720:w=iw:h=24:color=0x1b2016:t=fill,scale=810:608,format=yuv420p[g];\
color=c=$BG:s=1280x720:r=30:d=4[bg];[bg][g]overlay=235:60:shortest=1,\
$(hdr 4 'Generate demonstrations in sim' 'Randomized per episode: cube + cup pose on the desk, desk texture, lighting, splat colour, camera pose, friction.' e4g)[v]" \
    -map "[v]" "${ENC[@]}" "$TMP/e4g.mp4"
  cat_clips "$OUT/step04_sim_demos.mp4" $TMP/e4.mp4 $TMP/e4g.mp4
}

step5() {  # DAgger-style takeovers (only episodes whose cube and cup spawned on the real desk)
  local segs=("16.85 24.88" "24.92 39.58" "47.25 56.0" "73.55 80.78" "94.5 102.65") f="" c="" i=0
  for sg in "${segs[@]}"; do set -- $sg; f+="[0:v]trim=start=$1:end=$2,setpts=PTS-STARTPTS[t$i];"; c+="[t$i]"; i=$((i+1)); done
  "${FF[@]}" -i $R/videos_v4/2_recovery_takeovers_splat.mp4 -filter_complex "\
${f}${c}concat=n=$i:v=1:a=0,setpts=PTS/2.5,fps=30[d];color=c=$BG:s=1280x720:r=30[bg];[bg][d]overlay=0:120:shortest=1,\
$(hdr 5 'Teach recovery (DAgger-style takeovers)' 'An earlier policy drives (blue bar); when it stalls or fails, the expert takes over (orange) and the recovery is recorded.  2.5x.' e5)[v]" \
    -map "[v]" "${ENC[@]}" "$OUT/step05_recovery.mp4"
}

step6() {  # data bug: spawns off the desk
  "${FF[@]}" -loop 1 -framerate 30 -t 5 -i $R/videos_v4/desk_fix_grid_before.jpg -loop 1 -framerate 30 -t 5 -i $R/videos_v4/desk_fix_grid.jpg -filter_complex "\
[0]scale=800:600,format=yuv420p[a];[1]scale=800:600,format=yuv420p[b];[a][b]xfade=transition=wipeleft:duration=1:offset=4[x];\
color=c=$BG:s=1280x720:r=30:d=9[bg];[bg][x]overlay=240:60:shortest=1,\
$(tag e6a 'BEFORE' 1060 80 0xFF6B5B):enable='lt(t,4.5)',$(tag e6b 'AFTER' 1060 80 0x7BD88F):enable='gte(t,4.5)',\
$(hdr 6 'Audit the data' 'Bug found in review: cups spawned off the real desk.  Spawns now clipped to the measured desk polygon; 1,509 of 4,013 episodes dropped.' e6)[v]" \
    -map "[v]" "${ENC[@]}" "$OUT/step06_data_audit.mp4"
}

step7() {  # 9 real teleop demos in a grid + 3 wrist views
  local V=$R/real_datasets/cube_cup_real_v2_merged/videos CSV=$R/report/cube_cup_real_v2_merged_eps.csv
  local eps=(0 3 7 10 13 17 20 24 28) wr=(3 13 24) D=11 inputs=() f="" i=0
  for e in "${eps[@]}"; do
    local row; row=$(awk -F, -v e=$e 'NR>1 && $1==e' $CSV)
    local fi_; fi_=$(echo "$row" | cut -d, -f8); local st; st=$(echo "$row" | cut -d, -f9)
    inputs+=(-ss "$st" -t $D -i "$V/observation.images.scene/chunk-000/file-$(printf %03d $fi_).mp4")
    f+="[$i:v]setpts=PTS/1.25,scale=266:200[t$i];"; i=$((i+1))
  done
  for e in "${wr[@]}"; do
    local row; row=$(awk -F, -v e=$e 'NR>1 && $1==e' $CSV)
    local fi_; fi_=$(echo "$row" | cut -d, -f4); local st; st=$(echo "$row" | cut -d, -f5)
    inputs+=(-ss "$st" -t $D -i "$V/observation.images.wrist/chunk-000/file-$(printf %03d $fi_).mp4")
    f+="[$i:v]setpts=PTS/1.25,scale=266:200[t$i];"; i=$((i+1))
  done
  f+="[t0][t1][t2][t3][t4][t5][t6][t7][t8]xstack=inputs=9:layout=0_0|w0_0|w0+w1_0|0_h0|w0_h0|w0+w1_h0|0_h0+h3|w0_h0+h3|w0+w1_h0+h3[g];"
  f+="[t9][t10][t11]vstack=inputs=3[wr];color=c=$BG:s=1280x720:r=30[bg];[bg][g]overlay=100:60:shortest=1[b1];[b1][wr]overlay=910:60:shortest=1,fps=30,"
  f+="$(tag e7a 'overhead camera, 9 of 40 demos' 108 68 white),$(tag e7b 'wrist camera' 918 68 white),"
  f+="$(hdr 7 'Collect real demonstrations' '40 teleoperated demos (leader-follower arm, Jetson web console), 14,704 frames.  Cube and cup placed all over the desk.  1.25x.' e7)[v]"
  "${FF[@]}" "${inputs[@]}" -filter_complex "$f" -map "[v]" -t 8.8 "${ENC[@]}" "$OUT/step07_real_demos.mp4"
}

step8() {  # policy generations, 2x2 grid of overhead views
  local E=$R/eval D=20
  "${FF[@]}" -t $D -i $E/act_sim_v0/eval.mp4 -t $D -i $E/act_sim_v1/eval.mp4 -t $D -i $E/act_sim_v2/eval.mp4 -t $D -i $E/act_v4_80k_fixed/eval.mp4 -filter_complex "\
[0:v]crop=640:480:0:0,scale=400:300[a];[1:v]crop=640:480:0:0,scale=400:300[b];[2:v]crop=640:480:0:0,scale=400:300[c];[3:v]crop=640:480:0:0,scale=400:300[d];\
[a][b][c][d]xstack=inputs=4:layout=0_0|w0+6_0|0_h0+6|w0+6_h0+6:fill=$BG[g];color=c=$BG:s=1280x720:r=30[bg];[bg][g]overlay=237:61:shortest=1,\
$(tag e8a 'v0   200 expert demos          28%' 245 69 white),$(tag e8b 'v1   400 demos, no idle waits   62%' 651 69 white),\
$(tag e8c 'v2   + 300 recovery takeovers   84%' 245 375 white),$(tag e8d 'v4   + splat, DART, 40 real   88%' 651 375 $AMBER),\
$(hdr 8 'Train and iterate' 'ACT policies, 50 randomized sim episodes each.  v0-v2 on the first sim build; v4 on the recalibrated, wider-spawn build.' e8)[v]" \
    -map "[v]" -t $D "${ENC[@]}" "$OUT/step08_iterations.mp4"
}

step9() {  # final sim eval
  "${FF[@]}" -i $R/eval/act_v4_80k_fixed/eval.mp4 -filter_complex "\
[0:v]setpts=PTS/1.25,fps=30[d];color=c=$BG:s=1280x720:r=30[bg];[bg][d]overlay=0:120:shortest=1,\
$(tag e9a 'ACT v4  80k steps    44 / 50 = 88%' 10 130 $AMBER),\
$(hdr 9 'Evaluate in sim' 'Held-out seeds, randomized scenes.  Failures: 4 time-outs (cube knocked out of reach, no retry), 2 tipped cups.  1.25x.' e9)[v]" \
    -map "[v]" "${ENC[@]}" "$OUT/step09_sim_eval.mp4"
}

step10() {  # real-arm rollouts recorded by the Jetson console: real-only vs co-trained, then three co-trained successes
  local RO=$R/report/rollouts
  "${FF[@]}" -i $RO/rollouts_act_real40_80k/videos/observation.images.scene/chunk-000/file-001.mp4 \
    -i $RO/rollouts_act_v4_80k/videos/observation.images.scene/chunk-000/file-003.mp4 -filter_complex "\
[0:v]trim=end=11,setpts=PTS-STARTPTS[a];[1:v]tpad=stop_mode=clone:stop_duration=2,trim=end=11,setpts=PTS-STARTPTS[b];\
[a][b]hstack=inputs=2[d];color=c=$BG:s=1280x720:r=30[bg];[bg][d]overlay=0:124:shortest=1,\
$(tag e10ca 'REAL DATA ONLY   40 real demos' 12 136 white),$(tag e10cb 'REAL + SIM   40 real + 4,013 sim demos' 652 136 white),\
$(tag e10cc 'cube on the right, near the base:  hovers, never grasps   FAIL' 12 564 0xFF6B5B),\
$(tag e10cd 'cube right of center:  grasps and places   SUCCESS' 652 564 $AMBER),\
$(hdr 10 'Deploy on the real arm' 'Same ACT model and 80k training steps.  Right: the same 40 real demos plus the sim demos.  Overhead camera, real time.' e10c)[v]" \
    -map "[v]" "${ENC[@]}" "$TMP/e10_cmp.mp4"
  local V=$RO/rollouts_act_v4_80k/videos/observation.images run i=0 parts=("$TMP/e10_cmp.mp4")
  for run in "0 1 15.2" "1 2 9.6" "3 4 10.8"; do set -- $run; i=$((i+1))
    local f; f=$(printf 'chunk-000/file-%03d.mp4' $1)
    "${FF[@]}" -i $V.scene/$f -i $V.wrist/$f -filter_complex "\
[0:v][1:v]hstack=inputs=2,fade=t=in:st=0:d=0.3[d];color=c=$BG:s=1280x720:r=30[bg];[bg][d]overlay=0:124:shortest=1,\
$(tag e10o$i 'OVERHEAD CAMERA' 12 136 white),$(tag e10w$i 'WRIST CAMERA' 652 136 white),\
$(tag e10r$i "Run $2   SUCCESS   $3 s" 12 564 $AMBER),\
$(hdr 10 'Deploy on the real arm' 'Co-trained ACT v4 running on the Jetson, real time.  This session: 3 of 4 succeeded (the miss bounced off the cup rim).' e10)[v]" \
      -map "[v]" "${ENC[@]}" "$TMP/e10_$i.mp4"
    parts+=("$TMP/e10_$i.mp4")
  done
  cat_clips "$OUT/step10_real_arm.mp4" "${parts[@]}"
}

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(1 2 3 4 5 6 7 8 9 10)
for s in "${steps[@]}"; do echo "[build] step$s"; "step$s"; done
ls -la "$OUT"

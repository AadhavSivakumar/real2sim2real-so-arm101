#!/bin/bash
# encode the orbit + contact sheet (CPU, nice 19)
set -e
D=/mnt/secondary/so101_r2s/report/splat_render
PY=/home/asubuntudesktop/TestingGrounds/robot_splat/.venv/bin/python
CRF=${CRF:-19}
nice -n 19 ffmpeg -nostdin -loglevel error -y -framerate 30 -i $D/work/frames/f_%04d.png \
  -c:v libx264 -preset slow -crf $CRF -maxrate 5M -bufsize 10M -pix_fmt yuv420p -profile:v high \
  -movflags +faststart -threads 6 $D/splat_orbit.mp4
OMP_NUM_THREADS=2 nice -n 19 $PY $D/scripts/sheet.py $D/work/frames $D/sheet.jpg 640 frame
ls -la $D/splat_orbit.mp4 $D/sheet.jpg
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_frames,bit_rate -show_entries format=duration,size -of default=nw=1 $D/splat_orbit.mp4

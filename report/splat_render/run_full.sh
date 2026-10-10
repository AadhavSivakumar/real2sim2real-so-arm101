#!/bin/bash
# full orbit render: 2 processes x 5 threads (10 total), CPU only, nice 19
cd /mnt/secondary/so101_r2s/report/splat_render
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=5 MKL_NUM_THREADS=5
PY=/home/asubuntudesktop/TestingGrounds/robot_splat/.venv/bin/python
nice -n 19 $PY scripts/render_orbit.py work/frames 1280 720 0 330 2 > work/render_0.log 2>&1 &
nice -n 19 $PY scripts/render_orbit.py work/frames 1280 720 1 330 2 > work/render_1.log 2>&1 &
wait
echo ALLDONE

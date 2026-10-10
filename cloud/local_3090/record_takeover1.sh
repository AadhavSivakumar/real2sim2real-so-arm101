#!/bin/bash
# 300 expert-takeover (DAgger-style) episodes from ACT v1, resuming after crashes.
# Run via: gpu_run.sh /mnt/secondary/so101_r2s/sim_demos/takeover1.log -- bash record_takeover1.sh
cd /home/asubuntudesktop/TestingGrounds/so101_r2s2r
for i in $(seq 1 30); do
  ./lab.sh -p scripts/record_takeover.py --episodes 300 --preview 12 --fail_preview 12 --out /mnt/secondary/so101_r2s/sim_demos/takeover1 --viz none && exit 0
  echo "[retry] run $i failed $(date +%T)"
  # bad RAM corrupts page-cached Isaac shader caches: drop the cached copies (disk unchanged)
  python3 -c "import os,sys
for r in sys.argv[1:]:
 for d,_,fs in os.walk(r):
  for f in fs:
   try: fd=os.open(os.path.join(d,f),os.O_RDONLY)
   except OSError: continue
   os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)" /mnt/secondary/isaacsim/kit/cache /mnt/secondary/isaacsim/extscache
  sleep 5
done
exit 1

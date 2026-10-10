#!/bin/bash
# Screenshot every stub scenario at desktop and phone width (headless Chrome, own profile dir).
# usage: ui_stub/shoot.sh [scenario ...]   (stub must be running on 127.0.0.1:8199)
cd "$(dirname "$0")"
PROF=$(mktemp -d /tmp/so101_ui_chrome.XXXX)
SC=${@:-none loading ready homing running running_dry intervention pending error fault rec_session limp}
for s in $SC; do
  for w in 1400,1800 390,1600 390,4200; do
    tag=${w%%,*}; [ "$w" = 390,4200 ] && tag=390tall
    timeout 60 google-chrome --headless=new --disable-gpu --no-first-run --no-default-browser-check \
      --user-data-dir="$PROF" --hide-scrollbars --window-size=$w --virtual-time-budget=2500 \
      --screenshot="shots/${s}_${tag}.png" "http://127.0.0.1:8199/?phase=$s" >/dev/null 2>&1
    echo "shots/${s}_${tag}.png $(stat -c %s shots/${s}_${tag}.png 2>/dev/null || echo MISSING)"
  done
done
rm -rf "$PROF"

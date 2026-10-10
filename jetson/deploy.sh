#!/bin/bash
# Install the committed dev console into the Jetson's production copy (~/lerobot/so101_web + ~/lerobot/so101.sh),
# keeping a dated backup, then smoke-test the production copy in MOCK mode on port 8098 (never touches the arm).
#   deploy.sh            install + smoke test
#   deploy.sh rollback   restore the newest backup
set -e
J=${SO101_JETSON:-asnano@jetson}  # user@host of the Jetson
SSH="ssh -o ConnectTimeout=30 $J"
DEV=/mnt/secondary/so101_r2s/so101_web_dev/web
STAMP=$(date +%Y%m%d_%H%M)
if [ "$1" = rollback ]; then
  $SSH 'b=$(ls -d ~/lerobot/so101_web_backup_* | tail -1); [ -n "$b" ] || exit 1; rm -rf ~/lerobot/so101_web &&
        cp -a "$b/so101_web" ~/lerobot/so101_web && cp -a "$b/so101.sh" ~/lerobot/so101.sh && echo "restored $b"'
  exit
fi
[ -z "$(git -C $DEV status --porcelain)" ] || { echo "uncommitted changes in $DEV"; exit 1; }
if $SSH 'pgrep -f "[s]o101_web/app.py" > /dev/null'; then
  echo "NOTE: the console is running; files are replaced, restart it (Ctrl-C, ./so101.sh web) to use the new version"
fi
$SSH "mkdir -p ~/lerobot/so101_web_backup_$STAMP && cp -a ~/lerobot/so101_web ~/lerobot/so101.sh ~/lerobot/so101_web_backup_$STAMP/"
rsync -a -e "ssh -o ConnectTimeout=30" --exclude .git --exclude __pycache__ --exclude test_datasets --exclude logs \
  --exclude so101.sh $DEV/ $J:lerobot/so101_web/
rsync -a -e "ssh -o ConnectTimeout=30" $DEV/so101.sh $J:lerobot/so101.sh
echo "installed $(git -C $DEV log --oneline -1) (backup: ~/lerobot/so101_web_backup_$STAMP)"
# smoke test of the installed copy, mock hardware only
$SSH 'cd ~/lerobot && (SO101_MOCK=1 SO101_DEPTH=0 timeout 90 .venv/bin/python so101_web/app.py --port 8098 \
        --data-root /tmp/so101_smoke_data > /tmp/so101_smoke.log 2>&1 & echo $! > /tmp/so101_smoke.pid) ; for i in $(seq 1 40); do
        curl -s -m 2 localhost:8098/api/policies > /tmp/so101_smoke.json && break; sleep 1; done;
        python3 -c "import json; d=json.load(open(\"/tmp/so101_smoke.json\")); print(\"smoke: policies\", [(p[\"name\"], p[\"runnable\"]) for p in d[\"policies\"]])";
        curl -s -m 2 localhost:8098/api/state | python3 -c "import json,sys; s=json.load(sys.stdin); print(\"smoke: mock\", s.get(\"mock\"), \"policy phase\", s[\"policy\"][\"phase\"])";
        kill $(cat /tmp/so101_smoke.pid) 2>/dev/null; echo smoke done'

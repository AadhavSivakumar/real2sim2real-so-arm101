#!/bin/bash
# Keep one SSH port forward alive (reconnects after drops). Usage: tunnel.sh -L|-R PORT HOST
DIR=$1; PORT=$2; HOST=$3
while true; do
  ssh -N -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -o ConnectTimeout=20 \
      $DIR $PORT:127.0.0.1:$PORT $HOST
  echo "$(date +%T) tunnel $DIR $PORT $HOST dropped, reconnecting"; sleep 5
done

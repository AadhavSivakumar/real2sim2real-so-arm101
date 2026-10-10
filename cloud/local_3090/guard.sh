#!/bin/bash
# Usage: guard.sh LOG -- cmd args...   Runs cmd (stdout/stderr -> LOG); kills it if the kernel logs a memory-corruption sign.
LOG=$1; shift; [ "$1" = "--" ] && shift
"$@" > "$LOG" 2>&1 &
PID=$!
( journalctl -k -f -n0 --no-pager 2>/dev/null | grep --line-buffered -iE "general protection|BUG:|Oops|machine check|mce:|EDAC.*(CE|UE)|page dumped because|Bad page" | while read -r l; do
    echo "[guard] $(date +%T) kernel: $l" >> "$LOG"
    echo "[guard] stopping job $PID" >> "$LOG"
    pkill -TERM -P $PID; kill -TERM $PID; sleep 20; pkill -KILL -P $PID; kill -KILL $PID
    break
  done ) &
WPID=$!
wait $PID; RC=$?
pkill -P $WPID 2>/dev/null; kill $WPID 2>/dev/null
echo "[guard] job exited rc=$RC at $(date +%T)" >> "$LOG"
exit $RC

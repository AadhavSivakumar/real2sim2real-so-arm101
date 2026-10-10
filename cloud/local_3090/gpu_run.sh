#!/bin/bash
# Usage: gpu_run.sh LOG -- cmd args...
# Runs an Isaac/GPU job under guard.sh, serialized with every other gpu_run.sh job (one heavy job at a time).
exec flock /mnt/secondary/so101_r2s/gpu.lock /mnt/secondary/so101_r2s/guard.sh "$@"

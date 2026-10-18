#!/bin/bash
# Memory and throughput grid for chunked prefill: one fresh process per
# configuration so allocator and cache state never carry between runs.
# CHUNK=0 is the one-shot reference, rerun only at 512 as a calibration
# of the measurement against the frozen QWEN-METAL-QUAL-0 value (12.7 GB);
# the 1,024 / 2,048 / 4,096 one-shot peaks are reused from that receipt.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/qmetal/chunked/grid.log
mkdir -p logs/qmetal/chunked
for cfg in "512 0" "512 64" "512 128" "512 256" "512 512" \
           "1024 64" "1024 128" "1024 256" "1024 512" \
           "2048 128" "2048 256" "2048 512" \
           "4096 256" "4096 512" "4096 128"; do
  set -- $cfg
  echo "== ctx $1 chunk $2 $(date -u +%H:%M:%S)" >> "$LOG"
  MODE=mem CTX=$1 CHUNK=$2 .venv/bin/python scratch/qwen_metal_chunked.py 2>&1 | grep -v Warning | tail -1 >> "$LOG"
done
echo "== grid done $(date -u +%H:%M:%S)" >> "$LOG"

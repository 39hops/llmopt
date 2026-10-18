#!/bin/bash
# Memory and throughput grid for chunked prefill: one fresh process per
# configuration so allocator and cache state never carry between runs.
# CHUNK=0 is the one-shot reference, rerun only at 512 as a calibration
# of the measurement against the frozen QWEN-METAL-QUAL-0 value (12.7 GB);
# the 1,024 / 2,048 / 4,096 one-shot peaks are reused from that receipt.
# A failed configuration stops the grid with its exit code and a FAILED
# line in the log; the "grid done" marker is written only when every
# configuration exited 0.
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
  TMP=$(mktemp)
  MODE=mem CTX=$1 CHUNK=$2 .venv/bin/python scratch/qwen_metal_chunked.py > "$TMP" 2>&1
  rc=$?
  grep -v Warning "$TMP" | tail -1 >> "$LOG"
  rm -f "$TMP"
  if [ "$rc" -ne 0 ]; then
    echo "== grid FAILED ctx $1 chunk $2 rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
    exit "$rc"
  fi
done
echo "== grid done $(date -u +%H:%M:%S)" >> "$LOG"

#!/usr/bin/env bash
# STREAM-WDISTILL-CENSUS-0 overnight chain (3080): four MoE layers
# through the v2 apparatus, sequential, one receipt per layer.
# L22 already measured (logs/streamwd/v2proto.jsonl); together these
# give five depth points {2, 12, 22, 33, 42} of 43.
set -eo pipefail
cd ~/code/llmopt
for L in 2 12 33 42; do
  echo "=== census layer $L ==="
  TORCH_DISABLE_NATIVE_JIT=1 SHARD_CACHE=~/shards LAYER=$L \
    .venv/bin/python -u scratch/streamwd_v2.py
done
echo "census chain complete"

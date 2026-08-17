#!/usr/bin/env bash
# QWEN FFN depth census (GPT bridge item 1): layers {8,48,56} + the
# existing 32 = 4 depth points of 64. Each writes its own receipt.
set -eo pipefail
cd ~/code/llmopt
for L in 8 48 56; do
  echo "=== qwen census layer $L ==="
  LAYER=$L SHARD_CACHE=~/shards TORCH_DISABLE_NATIVE_JIT=1 \
    .venv/bin/python -u scratch/qwen_stream_probe.py
done
echo "qwen census complete"

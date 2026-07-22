#!/bin/bash
# 2026-07-21 day chain: late arm finishes -> shaped GRPO (the b-lever).
cd ~/code/llmopt
P=~/code/llmopt/.venv/bin/python
until grep -q "HOT_CHAIN_DONE" gen7_gates.log 2>/dev/null; do sleep 300; done
$P scratch/grpo_shaped.py > grpo_shaped.log 2>&1
$P scratch/gate_ckpt.py checkpoints/grpo_shaped.pt 512 12 2304 8 "GRPO-SHAPED" >> gen7_gates.log 2>&1
$P scratch/l9_probe.py checkpoints/grpo_shaped.pt 512 12 2304 8 GRPO-SHAPED >> gen7_gates.log 2>&1
echo DAY_CHAIN_DONE >> gen7_gates.log

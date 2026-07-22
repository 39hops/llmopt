#!/bin/bash
# Boundary-or-bulk grid, Mac half: fp32 @ d256 and d768, gen4
# corpus, 3ep, serial. Gates are KV-accelerated now.
cd ~/code/llmopt
P=~/code/llmopt/.venv/bin/python
$P scripts/train_mathnative.py --gen4 --epochs 3 --d 256 --layers 8 --ffn 1024 --heads 4 --out checkpoints/grid_fp32_256.pt > grid_fp32_256.log 2>&1
$P scratch/gate_ckpt.py checkpoints/grid_fp32_256.pt 256 8 1024 4 "GRID-fp32-d256" >> grid_gates.log 2>&1
$P scripts/train_mathnative.py --gen4 --epochs 3 --d 768 --layers 8 --ffn 3072 --heads 12 --out checkpoints/grid_fp32_768.pt > grid_fp32_768.log 2>&1
$P scratch/gate_ckpt.py checkpoints/grid_fp32_768.pt 768 8 3072 12 "GRID-fp32-d768" >> grid_gates.log 2>&1
echo MAC_GRID_DONE >> grid_gates.log

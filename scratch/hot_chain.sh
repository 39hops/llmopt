#!/bin/bash
cd ~/code/llmopt
P=~/code/llmopt/.venv/bin/python
$P scratch/metabolic_hot.py > metabolic_hot.log 2>&1
$P scratch/gate_ckpt.py checkpoints/metabolic_hot.pt 512 12 2304 8 "METABOLIC-HOT" >> gen7_gates.log 2>&1
$P scratch/l9_probe.py checkpoints/metabolic_hot.pt 512 12 2304 8 METABOLIC-HOT >> gen7_gates.log 2>&1
$P scratch/metabolic_hot.py --late > metabolic_late.log 2>&1
$P scratch/gate_ckpt.py checkpoints/metabolic_late.pt 512 12 2304 8 "METABOLIC-LATE" >> gen7_gates.log 2>&1
$P scratch/l9_probe.py checkpoints/metabolic_late.pt 512 12 2304 8 METABOLIC-LATE >> gen7_gates.log 2>&1
echo HOT_CHAIN_DONE >> gen7_gates.log

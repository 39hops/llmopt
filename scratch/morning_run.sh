#!/bin/bash
cd ~/code/llmopt
P=~/code/llmopt/.venv/bin/python
$P scratch/gate_ckpt.py checkpoints/mathnative_gen7.pt 512 12 2368 8 "GEN-7" >> gen7_gates.log 2>&1
$P scratch/l9_probe.py checkpoints/mathnative_gen7.pt 512 12 2368 8 GEN-7 >> gen7_gates.log 2>&1
$P scratch/gate_ckpt.py checkpoints/metabolic_live.pt 512 12 2304 8 "METABOLIC-final" >> gen7_gates.log 2>&1
$P scratch/l9_probe.py checkpoints/metabolic_live.pt 512 12 2304 8 METABOLIC >> gen7_gates.log 2>&1
$P scratch/vrm_ab.py > vrm_ab.log 2>&1
echo MORNING_RUN_DONE >> gen7_gates.log

#!/bin/bash
set -e
cd ~/code/llmopt
while pgrep -f "mac_day_chain" > /dev/null; do sleep 120; done
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_infixtwin_z1.pt 384 8 1536 6 pro_z1 > logs/prologue_z1_gate.log 2>&1
for ARM in b1 z1c; do
  .venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_infixtwin_$ARM.pt 384 8 1536 6 pro_$ARM > logs/prologue_${ARM}_gate.log 2>&1
done
echo Z1_DONE >> logs/prologue_z1_gate.log

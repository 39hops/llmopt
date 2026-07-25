#!/bin/bash
set -e
cd ~/code/llmopt
while pgrep -f "mac_day_chain|ptq4_gates.sh|prologue_gates" > /dev/null; do sleep 120; done
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_infixtwin_z1.pt 384 8 1536 6 pro_z1 > logs/prologue_z1_gate.log 2>&1
echo Z1_DONE >> logs/prologue_z1_gate.log

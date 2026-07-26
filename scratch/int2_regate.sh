#!/bin/bash
set -e
cd ~/code/llmopt
while pgrep -f "z1_gate|ptq4_gates|prologue_gates|gate_ckpt" > /dev/null; do sleep 120; done
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_infixtwin_int2.pt 384 8 1536 6 race_int2fix > logs/lmrace_int2fix_gate.log 2>&1
echo INT2FIX_DONE >> logs/lmrace_int2fix_gate.log

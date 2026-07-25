#!/bin/bash
set -e
cd ~/code/llmopt
while pgrep -f "mac_day_chain|ptq4_gates.sh" > /dev/null; do sleep 120; done
for ARM in s4 sparse tern m4 gflip_m4 gflip_tern; do
  .venv/bin/python scratch/gate_ckpt.py \
    checkpoints/mathnative_19m_infixtwin_$ARM.pt 384 8 1536 6 pro_$ARM \
    > logs/prologue_${ARM}_gate.log 2>&1
done
echo PROLOGUE_DONE >> logs/prologue_gflip_tern_gate.log
grep -h "gate" logs/prologue_*_gate.log

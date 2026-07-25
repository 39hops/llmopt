#!/bin/bash
# 4-bit PTQ arm gates — waits for the day chain to free the Mac.
set -e
cd ~/code/llmopt
while pgrep -f mac_day_chain > /dev/null; do sleep 120; done
for ARM in p4 lm16z nf4; do
  .venv/bin/python scratch/gate_ckpt.py \
    checkpoints/mathnative_19m_infixtwin_$ARM.pt 384 8 1536 6 race_$ARM \
    > logs/lmrace_${ARM}_gate.log 2>&1
done
echo PTQ4_DONE >> logs/lmrace_nf4_gate.log
grep -h "gate" logs/lmrace_p4_gate.log logs/lmrace_lm16z_gate.log logs/lmrace_nf4_gate.log

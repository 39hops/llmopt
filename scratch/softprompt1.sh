#!/bin/bash
# SOFT-PROMPT-1 driver, 3080. Pre-reg: RESULTS PRE-REG SOFT-PROMPT-1.
# CHAINED: waits for METALLICITY-1's DONE marker, then runs the
# prefix rung on the freshly-born metal_z3_d64 checkpoint.
set -eo pipefail
cd ~/code/llmopt
while [ ! -f logs/metallicity1.DONE ]; do sleep 300; done
mkdir -p logs/softprompt1
.venv/bin/python scratch/softprompt1.py checkpoints/metal_z3_d64.pt 64 \
  2>&1 | tee logs/softprompt1/run.log

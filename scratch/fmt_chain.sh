#!/bin/bash
cd /Users/artin/code/llmopt
export BIRTH_SEED=1
.venv/bin/python scratch/format_delta_prep.py > logs/fmt_delta_prep.log 2>&1
for F in traces skip dechain oneshot randpack delta; do
  FORMAT=$F SCHED=1p .venv/bin/python scratch/format_ladder.py > logs/fmt_${F}_1p.log 2>&1
  .venv/bin/python scratch/gate_ckpt.py checkpoints/fmt_${F}_1p.pt 256 8 1024 4 fmt_${F}_1p >> logs/fmt_${F}_1p.log 2>&1
  .venv/bin/python scratch/ce400.py checkpoints/fmt_${F}_1p.pt fmt_${F}_1p >> logs/fmt_${F}_1p.log 2>&1
done
echo FMT1P_DONE > logs/fmt_1p_done.marker

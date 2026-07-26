#!/bin/bash
cd /Users/artin/code/llmopt
export BIRTH_SEED=1
until [ -f logs/fmt_1p_done.marker ]; do sleep 60; done
for F in traces delta randpack; do
  FORMAT=$F SCHED=3e .venv/bin/python scratch/format_ladder.py > logs/fmt_${F}_3e.log 2>&1
  .venv/bin/python scratch/gate_ckpt.py checkpoints/fmt_${F}_3e.pt 256 8 1024 4 fmt_${F}_3e >> logs/fmt_${F}_3e.log 2>&1
  .venv/bin/python scratch/ce400.py checkpoints/fmt_${F}_3e.pt fmt_${F}_3e >> logs/fmt_${F}_3e.log 2>&1
done
FORMAT=revpairs SCHED=1p .venv/bin/python scratch/format_ladder.py > logs/fmt_revpairs_1p.log 2>&1
.venv/bin/python scratch/gate_ckpt.py checkpoints/fmt_revpairs_1p.pt 256 8 1024 4 fmt_revpairs_1p >> logs/fmt_revpairs_1p.log 2>&1
.venv/bin/python scratch/ce400.py checkpoints/fmt_revpairs_1p.pt fmt_revpairs_1p >> logs/fmt_revpairs_1p.log 2>&1
echo FMT2_DONE > logs/fmt_2_done.marker

#!/bin/bash
cd /Users/artin/code/llmopt
until [ -f logs/fmt_2_done.marker ]; do sleep 120; done
for C in fmt_oneshot_1p fmt_skip_1p fmt_dechain_1p fmt_randpack_1p fmt_delta_1p fmt_traces_1p fmt_revpairs_1p; do
  .venv/bin/python scratch/gate_pp.py checkpoints/${C}.pt 256 8 1024 4 ${C} > logs/pp_${C}.log 2>&1
done
.venv/bin/python scratch/gate_pp.py checkpoints/mathnative_wfloor_d256_stream4.pt 256 8 1024 4 pairs_1p > logs/pp_pairs_1p.log 2>&1
.venv/bin/python scratch/gate_pp.py checkpoints/mathnative_wfloor_d256.pt 256 8 1024 4 pairs_3e > logs/pp_pairs_3e.log 2>&1
echo PP_DONE > logs/fmt_pp_done.marker

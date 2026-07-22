#!/bin/bash
cd ~/code/llmopt
P=~/code/llmopt/.venv/bin/python
until grep -q "CALIB_NIGHT_DONE" holdout.log 2>/dev/null; do sleep 600; done
$P scratch/gate_batched.py checkpoints/mathnative_gen6_grown.pt 512 12 2304 8 "CHAMPION" 12 >> holdout.log 2>&1
$P scratch/gate_batched.py checkpoints/mathnative_gen6_grown.pt 512 12 2304 8 "CHAMPION" 24 >> holdout.log 2>&1
echo GATE_V2_BENCH_DONE >> holdout.log

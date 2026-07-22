#!/bin/bash
# Tuesday night 2026-07-21: SERIAL. holdout-v2 -> gate-v2 idle
# timing -> ternary session 2 (compounding + absorption) -> done.
cd ~/code/llmopt
P=~/code/llmopt/.venv/bin/python
$P scratch/holdout_v2.py checkpoints/mathnative_gen6_grown.pt 512 12 2304 8 "CHAMPION" >> holdout.log 2>&1
$P scratch/holdout_v2.py checkpoints/mathnative_gen6_ternary.pt 512 12 2048 8 "TERNARY-73" >> holdout.log 2>&1
( time $P scratch/gate_ckpt.py checkpoints/mathnative_gen6_grown.pt 512 12 2304 8 "CHAMPION-unbatched-idle" )  >> holdout.log 2>&1
$P scratch/gate_batched.py checkpoints/mathnative_gen6_grown.pt 512 12 2304 8 "CHAMPION-idle" 12 >> holdout.log 2>&1
$P scratch/ternary_session2.py > ternary_s2.log 2>&1
echo TUESDAY_NIGHT_DONE >> holdout.log

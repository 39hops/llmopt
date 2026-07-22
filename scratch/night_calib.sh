#!/bin/bash
# Calibration night 2026-07-21: holdout battery + seed-variance.
cd ~/code/llmopt
P=~/code/llmopt/.venv/bin/python
until grep -q "DAY_CHAIN_DONE" gen7_gates.log 2>/dev/null; do sleep 600; done
# 1) frozen holdout on the models of record
$P scratch/holdout_gate.py checkpoints/mathnative_gen6_grown.pt 512 12 2304 8 "CHAMPION" >> holdout.log 2>&1
$P scratch/holdout_gate.py checkpoints/mathnative_gen6_ternary.pt 512 12 2048 8 "TERNARY-73" >> holdout.log 2>&1
# 2) seed-variance: three 19M births, identical config, seeds differ
for s in 1 2 3; do
  BIRTH_SEED=$s $P scripts/train_mathnative.py --gen4 --epochs 3 \
    --d 384 --layers 8 --ffn 1536 --heads 6 \
    --out checkpoints/seedvar_$s.pt > seedvar_$s.log 2>&1
  $P scratch/gate_ckpt.py checkpoints/seedvar_$s.pt 384 8 1536 6 "SEEDVAR-$s" >> holdout.log 2>&1
done
echo CALIB_NIGHT_DONE >> holdout.log

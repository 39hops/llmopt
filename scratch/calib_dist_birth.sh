#!/bin/bash
# Rung-3 paired births (spec 2026-07-28, 3-arm design): fresh gen4
# control (same-day-control doctrine: trainer drifted since wfloor)
# + dose-control (pick x4 on treated states) + dist (distribution
# x4 on treated states). Same recipe as wfloor: gen-4 diet class,
# 3ep, BIRTH_SEED=1, d256/L8/ffn1024/h4, MPS, no --fast.
# gate_ckpt args: ckpt d layers ffn heads label.
set -e
cd ~/code/llmopt
PY=.venv/bin/python

BIRTH_SEED=1 $PY scripts/train_mathnative.py --gen4 --epochs 3 \
    --d 256 --layers 8 --ffn 1024 --heads 4 \
    --out checkpoints/calib_d256_ctl.pt \
    > logs/calib_ctl_birth.log 2>&1
$PY scratch/gate_ckpt.py checkpoints/calib_d256_ctl.pt \
    256 8 1024 4 calib_ctl 2>&1 | tee logs/calib_ctl_gate.log

BIRTH_SEED=1 $PY scripts/train_mathnative.py \
    --diet data/diet_dosectl_d256.jsonl --epochs 3 \
    --d 256 --layers 8 --ffn 1024 --heads 4 \
    --out checkpoints/calib_d256_dosectl.pt \
    > logs/calib_dosectl_birth.log 2>&1
$PY scratch/gate_ckpt.py checkpoints/calib_d256_dosectl.pt \
    256 8 1024 4 calib_dosectl 2>&1 | tee logs/calib_dosectl_gate.log

BIRTH_SEED=1 $PY scripts/train_mathnative.py \
    --diet data/diet_dist_d256.jsonl --epochs 3 \
    --d 256 --layers 8 --ffn 1024 --heads 4 \
    --out checkpoints/calib_d256_dist.pt \
    > logs/calib_dist_birth.log 2>&1
$PY scratch/gate_ckpt.py checkpoints/calib_d256_dist.pt \
    256 8 1024 4 calib_dist 2>&1 | tee logs/calib_dist_gate.log

# probe delta (rung-1 instrument) on all three arms
for arm in ctl dosectl dist; do
  $PY scratch/calib_probe.py checkpoints/calib_d256_${arm}.pt \
      256 8 1024 4 16 | tee -a logs/calib_r3_probe.log
done
touch logs/calib_r3_done.marker

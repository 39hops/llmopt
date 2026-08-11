#!/bin/bash
# GROW-DECOMP-1 cell A driver (Mac). Pre-reg: RESULTS PRE-REG
# GROW-DECOMP-1. One fresh birth at the crown's exact arch, then
# the standard gate.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs/grow_decomp1
BIRTH_SEED=0 .venv/bin/python scripts/train_mathnative.py \
  --v22 --gen4 --l8 --fp32 --epochs 3 \
  --d 512 --layers 12 --ffn 2304 --heads 8 \
  --out checkpoints/growdecomp_fresh512.pt \
  2>&1 | tee logs/grow_decomp1/birth.log
.venv/bin/python scratch/gate_ckpt.py checkpoints/growdecomp_fresh512.pt \
  512 12 2304 8 growdecomp_fresh512 2>&1 | tee logs/grow_decomp1/gate.log
echo GROW-DECOMP-1-A done

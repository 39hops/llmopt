#!/bin/bash
# GROW-DECOMP-1 cell A driver (Mac). Pre-reg: RESULTS PRE-REG
# GROW-DECOMP-1. One fresh birth at the crown's exact arch, then
# the standard gate.
# pipefail is load-bearing: these steps pipe through `tee`, and a
# pipeline reports the LAST command's status. Without it a training
# process that dies mid-epoch leaves tee returning 0, set -e never
# fires, the gate step runs against a checkpoint that was never
# written, and the driver prints 'done'. That happened on
# 2026-08-11 (GROW-DECOMP-1 cell A) and the job recorded rc=0.
set -eo pipefail
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

#!/bin/bash
# FLOOR-HK-1 (R3) driver: three fresh births on the sat_s2 warm diet
# (--v22 --gen4 --l8 = load_rows(True,True,True,True,True,False,None)),
# same family shape as sat_s2's d256 (L8 H4 ffn=4d), fp32 on mps.
# Floors read from the "epoch N: loss" lines; gates via the standard
# 120 (gate script invoked by the runner that owns each checkpoint).
# Pre-reg: RESULTS PRE-REG FLOOR-HK-1. Streams per-width, smallest
# first so partial results book if the wall hits.
# pipefail is load-bearing: these steps pipe through `tee`, and a
# pipeline reports the LAST command's status. Without it a training
# process that dies mid-epoch leaves tee returning 0, set -e never
# fires, the gate step runs against a checkpoint that was never
# written, and the driver prints 'done'. That happened on
# 2026-08-11 (GROW-DECOMP-1 cell A) and the job recorded rc=0.
set -eo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/floor_hk1
for D in 64 128 512; do
  FFN=$((4 * D))
  BIRTH_SEED=0 .venv/bin/python scripts/train_mathnative.py \
    --v22 --gen4 --l8 --fp32 --epochs 3 \
    --d "$D" --layers 8 --ffn "$FFN" --heads 4 \
    --out "checkpoints/floorhk_d${D}.pt" \
    2>&1 | tee "logs/floor_hk1/birth_d${D}.log"
done
echo FLOOR-HK-1 births done

#!/bin/bash
# FLOOR-HK-1 (R3) driver: three fresh births on the sat_s2 warm diet
# (--v22 --gen4 --l8 = load_rows(True,True,True,True,True,False,None)),
# same family shape as sat_s2's d256 (L8 H4 ffn=4d), fp32 on mps.
# Floors read from the "epoch N: loss" lines; gates via the standard
# 120 (gate script invoked by the runner that owns each checkpoint).
# Pre-reg: RESULTS PRE-REG FLOOR-HK-1. Streams per-width, smallest
# first so partial results book if the wall hits.
set -e
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

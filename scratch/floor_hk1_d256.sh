#!/bin/bash
# FLOOR-HK-1 fresh d256 cell (AMENDMENT LOSS-FLOOR-1-ARCH: the 0.348
# reference is d512-grown; the ladder needs its own fresh d256).
# Same recipe as floor_hk1.sh, then gates for ALL FOUR ladder widths
# as the separate post-birth step the pre-reg registered.
set -e
cd "$(dirname "$0")/.."
mkdir -p logs/floor_hk1
BIRTH_SEED=0 .venv/bin/python scripts/train_mathnative.py \
  --v22 --gen4 --l8 --fp32 --epochs 3 \
  --d 256 --layers 8 --ffn 1024 --heads 4 \
  --out checkpoints/floorhk_d256.pt \
  2>&1 | tee logs/floor_hk1/birth_d256.log
for D in 64 128 256 512; do
  FFN=$((4 * D))
  .venv/bin/python scratch/gate_ckpt.py "checkpoints/floorhk_d${D}.pt" \
    "$D" 8 "$FFN" 4 "floorhk_d${D}" 2>&1 | tee "logs/floor_hk1/gate_d${D}.log"
done
echo FLOOR-HK-1 d256+gates done

#!/bin/bash
# THE STABILITY ATLAS (pre-reg 2026-07-28): 8x4 (LR x BS) d64 birth
# grid, gate-colored — is capability-over-hyperparams smooth or
# fractal? + 4 seed-2 cells for the noise floor. cuda/TF32.
cd ~/code/llmopt
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
for LR in 0.0001 0.0002 0.0004 0.0008 0.0015 0.003 0.006 0.012; do
  for BS in 8 16 32 64; do
    tag="lr${LR}_bs${BS}"
    BIRTH_SEED=1 BIRTH_BS=$BS $PY scripts/train_mathnative.py \
      --gen4 --epochs 3 --d 64 --ffn 256 --heads 4 --lr $LR \
      --out checkpoints/atlas_${tag}.pt \
      > logs/atlas_${tag}.log 2>&1
    $PY scratch/gate_ckpt_cuda.py checkpoints/atlas_${tag}.pt \
      64 8 256 4 "ATLAS_${tag}" >> logs/atlas_grid.log 2>&1
    rm -f checkpoints/atlas_${tag}.pt checkpoints/atlas_${tag}.pt.ep
  done
done
for cell in "0.0004 32" "0.0015 16" "0.003 32" "0.006 8"; do
  set -- $cell; LR=$1; BS=$2
  tag="lr${LR}_bs${BS}_s2"
  BIRTH_SEED=2 BIRTH_BS=$BS $PY scripts/train_mathnative.py \
    --gen4 --epochs 3 --d 64 --ffn 256 --heads 4 --lr $LR \
    --out checkpoints/atlas_${tag}.pt > logs/atlas_${tag}.log 2>&1
  $PY scratch/gate_ckpt_cuda.py checkpoints/atlas_${tag}.pt \
    64 8 256 4 "ATLAS_${tag}" >> logs/atlas_grid.log 2>&1
  rm -f checkpoints/atlas_${tag}.pt checkpoints/atlas_${tag}.pt.ep
done
touch logs/atlas_done.marker

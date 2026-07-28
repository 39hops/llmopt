#!/bin/bash
# ATLAS-2: the Lyapunov leg (pre-reg 2026-07-28). Twin births at
# the atlas peak cell (lr 0.0015, bs 8, d64/3ep, seed 1): epsilon
# init perturbations {1e-6,1e-4,1e-2} + energy arms (INIT_SCALE
# 4 and 0.25) + an independent-seed reference (the saturation
# distance). Observables: gate (coarse) + teacher-forced argmax
# disagreement (fine, lyap_compare.py). Checkpoints KEPT.
# Waits for the atlas marker so the GPU is never double-booked.
cd ~/code/llmopt
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
until [ -f logs/atlas_done.marker ]; do sleep 60; done

birth () {  # tag eps scale — explicit args, no env leakage
  tag=$1; eps=${2:-0}; scale=${3:-1}
  INIT_PERTURB=$eps INIT_SCALE=$scale PERTURB_SEED=1 \
  BIRTH_SEED=1 BIRTH_BS=8 $PY scripts/train_mathnative.py \
    --gen4 --epochs 3 --d 64 --ffn 256 --heads 4 --lr 0.0015 \
    --out checkpoints/lyap_${tag}.pt > logs/lyap_${tag}.log 2>&1
  $PY scratch/gate_ckpt_cuda.py checkpoints/lyap_${tag}.pt \
    64 8 256 4 "LYAP_${tag}" >> logs/lyap_grid.log 2>&1
}

birth base
birth e6 1e-6
birth e4 1e-4
birth e2 1e-2
birth hi_base 0 4
birth hi_e4 1e-4 4
birth lo_base 0 0.25
birth lo_e4 1e-4 0.25
BIRTH_SEED=7 $PY scripts/train_mathnative.py \
  --gen4 --epochs 3 --d 64 --ffn 256 --heads 4 --lr 0.0015 \
  --out checkpoints/lyap_seed7.pt > logs/lyap_seed7.log 2>&1
$PY scratch/gate_ckpt_cuda.py checkpoints/lyap_seed7.pt \
  64 8 256 4 "LYAP_seed7" >> logs/lyap_grid.log 2>&1

for pair in "base e6" "base e4" "base e2" "base seed7" \
            "hi_base hi_e4" "lo_base lo_e4"; do
  set -- $pair
  $PY scratch/lyap_compare.py checkpoints/lyap_$1.pt \
    checkpoints/lyap_$2.pt "$1-vs-$2" >> logs/lyap_grid.log 2>&1
done
touch logs/lyap_done.marker

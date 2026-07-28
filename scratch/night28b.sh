#!/bin/bash
# NIGHT-28b (pre-reg 2026-07-28 late): taming-the-chaos battery.
# Chained behind the 45M retrofit marker. Cells: (A) twin soups,
# (D) hyperparam soup, (B) EMA twins, (C) symmetry twins + the
# twin-disagreement reads. Marker fires on success only.
set -e
cd ~/code/llmopt
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
until [ -f logs/sym45_done.marker ]; do sleep 120; done

LOG=logs/night28b.log
# --- (A) twin soups (desk: lyap checkpoints are kept) ---
for pair in e6 e4 e2; do
  $PY scratch/soup_gate.py "twin_base+$pair" 64 8 256 4 \
    checkpoints/lyap_base.pt checkpoints/lyap_$pair.pt >> $LOG 2>&1
done
$PY scratch/soup_gate.py "twin_all4" 64 8 256 4 \
  checkpoints/lyap_base.pt checkpoints/lyap_e6.pt \
  checkpoints/lyap_e4.pt checkpoints/lyap_e2.pt >> $LOG 2>&1

# --- (D) hyperparam soup: rebirth 4 plateau cells, then soup ---
for LR in 0.0008 0.0015 0.003 0.006; do
  BIRTH_SEED=1 BIRTH_BS=8 $PY scripts/train_mathnative.py \
    --gen4 --epochs 3 --d 64 --ffn 256 --heads 4 --lr $LR \
    --out checkpoints/hsoup_lr${LR}.pt > logs/hsoup_${LR}.log 2>&1
  $PY scratch/gate_ckpt_cuda.py checkpoints/hsoup_lr${LR}.pt \
    64 8 256 4 "HSOUP_lr${LR}" >> $LOG 2>&1
done
$PY scratch/soup_gate.py "hyperparam4" 64 8 256 4 \
  checkpoints/hsoup_lr0.0008.pt checkpoints/hsoup_lr0.0015.pt \
  checkpoints/hsoup_lr0.003.pt checkpoints/hsoup_lr0.006.pt \
  >> $LOG 2>&1

# --- (B) EMA twins (dense, eps 1e-6 on the twin) ---
ARM=dense TAG=_emaA EMA=0.999 $PY scratch/sym_birth.py >> $LOG 2>&1
ARM=dense TAG=_emaB EMA=0.999 EPS=1e-6 $PY scratch/sym_birth.py \
  >> $LOG 2>&1
$PY scratch/lyap_compare.py checkpoints/sym_birth_dense_emaA.pt \
  checkpoints/sym_birth_dense_emaB.pt "raw-emaA-vs-emaB" >> $LOG 2>&1
$PY scratch/lyap_compare.py \
  checkpoints/sym_birth_dense_emaA_ema.pt \
  checkpoints/sym_birth_dense_emaB_ema.pt "EMA-vs-EMA" >> $LOG 2>&1

# --- (C) symmetry twins (C8 commutant, eps 1e-6 on the twin) ---
ARM=c8 TAG=_twA $PY scratch/sym_birth.py >> $LOG 2>&1
ARM=c8 TAG=_twB EPS=1e-6 $PY scratch/sym_birth.py >> $LOG 2>&1
$PY scratch/lyap_compare.py checkpoints/sym_birth_c8_twA.pt \
  checkpoints/sym_birth_c8_twB.pt "c8-twA-vs-twB" >> $LOG 2>&1

touch logs/night28b_done.marker

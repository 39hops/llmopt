#!/bin/bash
# night-29 (pre-reg 2026-07-29): Leg A hardening on the 3080.
# Battery 1: seed replication x3 of the width floor, d64 v d56
#   paired ON CUDA (its own comparator line — never read against
#   MPS numbers; cross-device doctrine).
# Battery 2: finer floor along the CONTINUOUS axis — d56 with
#   FFN {192,160,128} (width quantizes to multiples of 8 at
#   heads=4 + RoPE-half; FFN steps by 1).
set -e  # marker fires on success ONLY
cd ~/code/llmopt
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export TORCH_DISABLE_NATIVE_JIT=1

for s in 1 2 3; do
  ARM=dense TAG=_cu64_s$s SEED=$s D=64 FFN=256 HEADS=4 EMA=0.999 \
    $PY scratch/sym_birth.py
  ARM=dense TAG=_cu56_s$s SEED=$s D=56 FFN=224 HEADS=4 EMA=0.999 \
    $PY scratch/sym_birth.py
done
for f in 192 160 128; do
  ARM=dense TAG=_cu56_f$f D=56 FFN=$f HEADS=4 EMA=0.999 \
    $PY scratch/sym_birth.py
done
echo NIGHT29-BATTERY-COMPLETE

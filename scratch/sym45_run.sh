#!/bin/bash
# C8-retrofit at 45M (pre-reg 2026-07-28): waits for the lyapunov
# marker, gates the projected init (math), warm-trains, gates
# math + ZX. 3080 tail window ~2-3h.
cd ~/code/llmopt
PY=.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
until [ -f logs/lyap_done.marker ]; do sleep 60; done

$PY scratch/sym45.py > logs/sym45_train.log 2>&1
$PY scratch/gate_ckpt_cuda.py checkpoints/union_45m_c8_projinit.pt \
  512 12 2048 8 "SYM45_projinit_math" >> logs/sym45_gates.log 2>&1
$PY scratch/gate_ckpt_cuda.py checkpoints/union_45m_c8.pt \
  512 12 2048 8 "SYM45_healed_math" >> logs/sym45_gates.log 2>&1
ZX_D=512 ZX_LAYERS=12 ZX_HEADS=8 ZX_FFN=2048 \
VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:" \
  $PY scratch/gate_zx.py checkpoints/union_45m_c8.pt real \
  "SYM45_healed_zx" >> logs/sym45_gates.log 2>&1
touch logs/sym45_done.marker

#!/bin/bash
cd ~/code/llmopt
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
export SEQ_CAP=1536
export BIRTH_SEED=1
export TF32=1
export GRAD_CKPT=1
export BIRTH_BS=8
.venv/bin/python scripts/tournament_birth.py --alpha M5 --epochs 3 --tag _zx --diet data/zx_farm1_train.jsonl > logs/zx_M5_birth.log 2>&1 && \
CPLX_ALPHA=G5 .venv/bin/python scratch/complex_birth.py --epochs 3 --tag _zx --diet data/zx_farm1_train.jsonl > logs/zx_G5_birth.log 2>&1 && \
CPLX_ALPHA=none .venv/bin/python scratch/complex_birth.py --epochs 3 --tag _zx --diet data/zx_farm1_train.jsonl > logs/zx_none_birth.log 2>&1 || exit 1
.venv/bin/python scratch/gate_zx.py checkpoints/tourn_M5_zx.pt real M5_zx > logs/zx_M5_gate.log 2>&1
.venv/bin/python scratch/gate_zx.py checkpoints/cplx_G5_zx.pt cplx G5_zx_latent > logs/zx_G5_latent_gate.log 2>&1
.venv/bin/python scratch/gate_zx.py checkpoints/cplx_G5_zx_dep.pt cplx G5_zx_dep > logs/zx_G5_dep_gate.log 2>&1
.venv/bin/python scratch/gate_zx.py checkpoints/cplx_none_zx.pt cplx none_zx > logs/zx_none_gate.log 2>&1
echo ZXCUDA_DONE > logs/zx_cuda_done.marker

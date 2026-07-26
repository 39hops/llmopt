#!/bin/bash
cd /Users/artin/code/llmopt
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
export SEQ_CAP=1536
export BIRTH_SEED=1
.venv/bin/python scripts/tournament_birth.py --alpha M5 --epochs 3 --tag _zx --diet data/zx_farm1_train.jsonl > logs/zx_M5_birth.log 2>&1 && \
CPLX_ALPHA=G5 .venv/bin/python scratch/complex_birth.py --epochs 3 --tag _zx --diet data/zx_farm1_train.jsonl > logs/zx_G5_birth.log 2>&1 && \
CPLX_ALPHA=none .venv/bin/python scratch/complex_birth.py --epochs 3 --tag _zx --diet data/zx_farm1_train.jsonl > logs/zx_none_birth.log 2>&1 && \
echo ZX_DONE > logs/zx_chain_done.marker

#!/bin/bash
cd /Users/artin/code/llmopt
CPLX_ALPHA=none BIRTH_SEED=1 .venv/bin/python scratch/complex_birth.py --epochs 3 > logs/cplx_none_birth.log 2>&1 && \
CPLX_ALPHA=G5 BIRTH_SEED=1 .venv/bin/python scratch/complex_birth.py --epochs 3 > logs/cplx_G5_birth.log 2>&1 && \
echo CPLX_DONE > logs/cplx_chain_done.marker

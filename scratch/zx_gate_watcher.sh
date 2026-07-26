#!/bin/bash
cd /Users/artin/code/llmopt
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
until [ -f logs/zx_chain_done.marker ]; do sleep 120; done
.venv/bin/python scratch/gate_zx.py checkpoints/tourn_M5_zx.pt real M5_zx > logs/zx_M5_gate.log 2>&1
.venv/bin/python scratch/gate_zx.py checkpoints/cplx_G5_zx.pt cplx G5_zx_latent > logs/zx_G5_latent_gate.log 2>&1
.venv/bin/python scratch/gate_zx.py checkpoints/cplx_G5_zx_dep.pt cplx G5_zx_dep > logs/zx_G5_dep_gate.log 2>&1
.venv/bin/python scratch/gate_zx.py checkpoints/cplx_none_zx.pt cplx none_zx > logs/zx_none_gate.log 2>&1
echo ZXGATES_DONE > logs/zx_gates_done.marker

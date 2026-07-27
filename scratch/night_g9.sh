#!/bin/bash
# G9-ZX roots-of-unity cell — TEMPLATE ONLY, DO NOT LAUNCH AS-IS
# (conditional pre-reg 2026-07-27: the rotation wing is CLOSED
# 07-26 and the 19M ZX seed fence drowns 19M cells; this fires
# only re-shaped to 45M-class, and only if the 45M union's ZX
# gate clears its bar — see RESULTS conditional pre-reg).
set -e
cd ~/code/llmopt
dead=0
while [ ! -f logs/night_45m_done.marker ]; do
    if pgrep -f 'train_mathnativ[e]|gate_ckpt_cud[a]|gate_z[x]|night_45[m]' > /dev/null; then
        dead=0
    else
        dead=$((dead+1))
        [ "$dead" -ge 3 ] && { echo "45M job died without marker"; exit 1; }
    fi
    sleep 120
done
[ -s data/zx_farm1_train.jsonl ] || { echo "missing diet"; exit 1; }
export VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:"
export SEQ_CAP=1536
export TF32=1
export GRAD_CKPT=1
export BIRTH_BS=8

CPLX_ALPHA=G9 BIRTH_SEED=1 .venv/bin/python scratch/complex_birth.py \
    --epochs 3 --tag _zx --diet data/zx_farm1_train.jsonl \
    > logs/zx_g9_birth.log 2>&1
# latent + deployed gates: alpha=none (no forward quantizer; the
# deployed ckpt's weights are already snapped).
CPLX_ALPHA=none .venv/bin/python scratch/gate_zx.py \
    checkpoints/cplx_G9_zx.pt cplx g9_zx_latent \
    > logs/zx_g9_latent_gate.log 2>&1
CPLX_ALPHA=none .venv/bin/python scratch/gate_zx.py \
    checkpoints/cplx_G9_zx_dep.pt cplx g9_zx_dep \
    > logs/zx_g9_dep_gate.log 2>&1
touch logs/night_g9_done.marker

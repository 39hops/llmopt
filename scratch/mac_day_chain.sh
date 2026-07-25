#!/bin/bash
# Day chain 2026-07-25 (spec: 2026-07-25-day-spec.md), Mac leg.
# Order: sigma births (T2) -> Lloyd-Max race gates (rider) ->
# d256 sigma pair. Waits for the in-flight s2 birth first.
set -e
cd ~/code/llmopt
while pgrep -f "mathnative_19m_gen9B_mps_s2" > /dev/null; do sleep 60; done
BIRTH_SEED=3 VOCAB_EXTRA=t .venv/bin/python scripts/train_mathnative.py \
  --diet data/gen9_diet_B.jsonl --epochs 3 \
  --out checkpoints/mathnative_19m_gen9B_mps_s3.pt \
  > logs/gen9_19m_mps_s3_birth.log 2>&1
for ARM in lm2 lm2z lm3 lm3z lmt int2; do
  .venv/bin/python scratch/gate_ckpt.py \
    checkpoints/mathnative_19m_infixtwin_$ARM.pt 384 8 1536 6 race_$ARM \
    > logs/lmrace_${ARM}_gate.log 2>&1
done
for S in 2 3; do
  BIRTH_SEED=$S .venv/bin/python scripts/train_mathnative.py \
    --gen4 --epochs 3 --d 256 --layers 8 --ffn 1024 --heads 4 \
    --out checkpoints/mathnative_wfloor_d256_s$S.pt \
    > logs/wfloor_d256_s${S}_birth.log 2>&1
  .venv/bin/python scratch/gate_ckpt.py \
    checkpoints/mathnative_wfloor_d256_s$S.pt 256 8 1024 4 wfloor_d256_s$S \
    > logs/wfloor_d256_s${S}_gate.log 2>&1
done
echo MAC_CHAIN_DONE >> logs/mac_day_chain.log
grep -h "gate" logs/lmrace_*_gate.log logs/wfloor_d256_s*_gate.log

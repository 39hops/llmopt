#!/bin/bash
# Rung-1 birth pair (spec 2026-07-25-native-transformer): prefix vs
# infix twin, 19M, materialized paired diets, same seed, Mac.
# Pre-reg in the spec; (ii) length bar already measured FAILED
# (-4.5% median vs -20% bar) — pair still fires for (i)/(iii)/(iv).
set -e
cd ~/code/llmopt
for f in data/gen4_diet_infix.jsonl data/gen4_diet_prefix.jsonl; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
done
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
  --diet data/gen4_diet_infix.jsonl --epochs 3 \
  --out checkpoints/mathnative_19m_infixtwin.pt \
  > logs/prefix_pair_infix_birth.log 2>&1
.venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_19m_infixtwin.pt \
  384 8 1536 6 infixtwin > logs/prefix_pair_infix_gate.log 2>&1
BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
  --diet data/gen4_diet_prefix.jsonl --epochs 3 \
  --out checkpoints/mathnative_19m_prefix.pt \
  > logs/prefix_pair_prefix_birth.log 2>&1
.venv/bin/python scratch/gate_prefix.py checkpoints/mathnative_19m_prefix.pt \
  384 8 1536 6 prefix > logs/prefix_pair_prefix_gate.log 2>&1
echo PREFIX_PAIR_DONE >> logs/prefix_pair_prefix_gate.log
grep -h "gate" logs/prefix_pair_*_gate.log

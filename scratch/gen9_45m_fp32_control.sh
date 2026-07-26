#!/bin/bash
set -e
cd ~/code/llmopt
for f in data/gen9_diet_B.jsonl data/series_probe_1e.jsonl data/phys_energy_probe.jsonl data/poly3_probe.jsonl; do
  [ -s "$f" ] || { echo "MISSING $f"; exit 1; }
done
BIRTH_SEED=1 VOCAB_EXTRA=t .venv/bin/python scripts/train_mathnative.py \
  --diet data/gen9_diet_B.jsonl --epochs 3 \
  --d 512 --layers 12 --ffn 2048 --heads 8 \
  --out checkpoints/mathnative_45m_gen9B_fp32.pt \
  > logs/gen9_45m_fp32_birth.log 2>&1
for P in series_probe_1e phys_energy_probe poly3_probe; do
  sed -e "s/series_probe.jsonl/$P.jsonl/" \
      -e "s/build_model(len(tok.vocab))/build_model(len(tok.vocab), d=512, layers=12, ffn=2048, heads=8)/" \
      -e "s/\"mps\" if torch.backends.mps.is_available() else \"cpu\"/\"cuda\" if torch.cuda.is_available() else \"cpu\"/" \
      scratch/series_probe.py > /tmp/g9f_probe.py
  VOCAB_EXTRA=t .venv/bin/python /tmp/g9f_probe.py checkpoints/mathnative_45m_gen9B_fp32.pt \
    > logs/mathnative_45m_gen9B_fp32_$P.log 2>&1
done
VOCAB_EXTRA=t .venv/bin/python scratch/gate_rarity.py \
  checkpoints/mathnative_45m_gen9B_fp32.pt 512 12 2048 8 45m_fp32_r \
  > logs/mathnative_45m_gen9B_fp32_rarity.log 2>&1
echo GEN9_45M_FP32_DONE >> logs/gen9_45m_fp32_birth.log

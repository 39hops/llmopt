#!/bin/bash
set -e
cd ~/code/llmopt
[ -s data/gen9_diet_B.jsonl ] || { echo "MISSING diet"; exit 1; }
[ -s data/series_probe_1e.jsonl ] || { echo "MISSING probe"; exit 1; }
BIRTH_SEED=1 VOCAB_EXTRA=t .venv/bin/python scripts/train_mathnative.py \
  --diet data/gen9_diet_B.jsonl --epochs 3 \
  --out checkpoints/mathnative_19m_gen9B_cuda_fp32.pt \
  > logs/gen9_19m_cuda_fp32_birth.log 2>&1
sed -e "s/series_probe.jsonl/series_probe_1e.jsonl/" \
    -e "s/build_model(len(tok.vocab))/build_model(len(tok.vocab), d=384, layers=8, ffn=1536, heads=6)/" \
    -e "s/\"mps\" if torch.backends.mps.is_available() else \"cpu\"/\"cuda\" if torch.cuda.is_available() else \"cpu\"/" \
    scratch/series_probe.py > /tmp/g19f_probe.py
VOCAB_EXTRA=t .venv/bin/python /tmp/g19f_probe.py checkpoints/mathnative_19m_gen9B_cuda_fp32.pt \
  > logs/mathnative_19m_gen9B_cuda_fp32_series.log 2>&1
VOCAB_EXTRA=t .venv/bin/python scratch/gate_rarity.py \
  checkpoints/mathnative_19m_gen9B_cuda_fp32.pt 384 8 1536 6 19m_cuda_fp32_r \
  > logs/mathnative_19m_gen9B_cuda_fp32_rarity.log 2>&1
echo GEN9_19M_CUDA_FP32_DONE >> logs/gen9_19m_cuda_fp32_birth.log

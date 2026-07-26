#!/bin/bash
set -e
cd ~/code/llmopt
while pgrep -f g19_sigma_cuda > /dev/null; do sleep 120; done
BIRTH_SEED=1 VOCAB_EXTRA=t .venv/bin/python scripts/train_mathnative.py \
  --diet data/gen9_diet_B.jsonl --epochs 3 --fast --nopack \
  --out checkpoints/mathnative_19m_gen9B_cuda_bf16np.pt \
  > logs/gen9_19m_cuda_bf16np_birth.log 2>&1
sed -e "s/series_probe.jsonl/series_probe_1e.jsonl/" \
    -e "s/build_model(len(tok.vocab))/build_model(len(tok.vocab), d=384, layers=8, ffn=1536, heads=6)/" \
    -e "s/\"mps\" if torch.backends.mps.is_available() else \"cpu\"/\"cuda\" if torch.cuda.is_available() else \"cpu\"/" \
    scratch/series_probe.py > /tmp/g19bn_probe.py
VOCAB_EXTRA=t .venv/bin/python /tmp/g19bn_probe.py checkpoints/mathnative_19m_gen9B_cuda_bf16np.pt \
  > logs/mathnative_19m_gen9B_cuda_bf16np_series.log 2>&1
echo BF16_ISOLATION_DONE >> logs/gen9_19m_cuda_bf16np_birth.log

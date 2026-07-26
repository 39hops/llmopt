#!/bin/bash
set -e
cd ~/code/llmopt
while pgrep -f tournament_birth > /dev/null; do sleep 60; done
for S in 2 3; do
  sed -e "s/series_probe.jsonl/series_probe_1e.jsonl/" \
      -e "s/build_model(len(tok.vocab))/build_model(len(tok.vocab), d=384, layers=8, ffn=1536, heads=6)/" \
      -e "s/\"mps\" if torch.backends.mps.is_available() else \"cpu\"/\"cuda\" if torch.cuda.is_available() else \"cpu\"/" \
      scratch/series_probe.py > /tmp/mps_s_probe.py
  VOCAB_EXTRA=t .venv/bin/python /tmp/mps_s_probe.py checkpoints/mathnative_19m_gen9B_mps_s$S.pt \
    > logs/mathnative_19m_gen9B_mps_s${S}_cuda_series.log 2>&1
done
echo MPS_SIGMA_GATES_DONE >> logs/mathnative_19m_gen9B_mps_s3_cuda_series.log

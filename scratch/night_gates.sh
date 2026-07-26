#!/bin/bash
set -e
cd ~/code/llmopt
.venv/bin/python /tmp/z1_gate_probe.py checkpoints/tourn_Z1.pt > logs/tourn_Z1_gate.log 2>&1
for S in 2 3; do
  VOCAB_EXTRA=t .venv/bin/python /tmp/mps_s_probe.py checkpoints/mathnative_19m_gen9B_mps_s$S.pt > logs/mathnative_19m_gen9B_mps_s${S}_cuda_series.log 2>&1
done
echo ALL_GATES_DONE >> logs/tourn_Z1_gate.log

#!/bin/bash
cd ~/code/llmopt
until ! pgrep -f metabolic_v5.py > /dev/null; do sleep 300; done
grep -q FINAL logs/v5_s1.log || { echo V5_DID_NOT_FINISH_CLEAN >> logs/b768_rerun.log; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 GRAD_CKPT=1
.venv/bin/python scripts/tournament_birth.py --alpha B --epochs 6   --d 768 --ffn 3072 --heads 12 > logs/b768_rerun.log 2>&1
.venv/bin/python scratch/gate_ckpt.py checkpoints/tourn_B.pt 768 8 3072 12 B768-6ep >> logs/b768_rerun.log 2>&1
echo B768_RERUN_DONE >> logs/b768_rerun.log

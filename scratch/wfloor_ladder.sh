#!/bin/bash
# Downward width ladder: W_min mapping + boundary-or-bulk grid cells.
# gen-4 corpus (matches the five-point ladder's existing points), 3ep,
# production chain gate per birth. Pre-reg: floor set by GRAMMAR
# structure not vocab count; L4+ (long dependencies) dies first.
set -e
cd ~/code/llmopt
run_one () {  # d ffn heads
  local D=$1 F=$2 H=$3
  BIRTH_SEED=1 .venv/bin/python scripts/train_mathnative.py \
    --gen4 --epochs 3 --d $D --layers 8 --ffn $F --heads $H \
    --out checkpoints/mathnative_wfloor_d$D.pt \
    > logs/wfloor_d${D}_birth.log 2>&1
  .venv/bin/python scratch/gate_ckpt.py checkpoints/mathnative_wfloor_d$D.pt \
    $D 8 $F $H wfloor_d$D > logs/wfloor_d${D}_gate.log 2>&1
}
run_one 256 1024 4
run_one 128 512 4
run_one 64 256 4
echo WFLOOR_DONE >> logs/wfloor_d64_gate.log
grep -h "gate" logs/wfloor_d*_gate.log

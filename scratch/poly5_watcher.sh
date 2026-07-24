#!/bin/bash
# fires poly5 after gen-9 A/B completes AND chain5 is fully written
source "$(dirname "$0")/remote.env.sh"
cd ~/code/llmopt
F="/mnt/c/Users/a/Documents/code/axiom/data/qual/poly_chain5.jsonl"
until grep -q GEN9_AB_DONE logs/gen9B_rarity.log 2>/dev/null; do sleep 600; done
while true; do
  s1=$(ssh -i "$WSL_KEY" -o BatchMode=yes "$WSL_REMOTE" "stat -c %s $F 2>/dev/null" || echo 0)
  sleep 90
  s2=$(ssh -i "$WSL_KEY" -o BatchMode=yes "$WSL_REMOTE" "stat -c %s $F 2>/dev/null" || echo 0)
  [ "$s1" != "0" ] && [ "$s1" = "$s2" ] && break
  sleep 300
done
bash scratch/poly5_pipeline.sh > logs/poly5_pipeline.log 2>&1 \
  && echo POLY5_PIPELINE_OK >> logs/poly5_pipeline.log

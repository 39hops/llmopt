#!/bin/bash
# Waits for (1) gen-8 pipeline completion and (2) poly_chain4 fully
# written on the axiom box (size stable across 90s — the file is
# actively streamed), then runs the bridge-law pipeline.
cd ~/code/llmopt
R="a@10.0.0.184"; K="-i $HOME/.ssh/winwsl -o BatchMode=yes"
F="/mnt/c/Users/a/Documents/code/axiom/data/qual/poly_chain4.jsonl"
until grep -q GEN8_DONE logs/gen8_rarity.log 2>/dev/null; do sleep 300; done
echo "[watch] gen-8 done, waiting on poly_chain4 stability"
while true; do
  s1=$(ssh $K $R "stat -c %s $F 2>/dev/null" || echo 0)
  sleep 90
  s2=$(ssh $K $R "stat -c %s $F 2>/dev/null" || echo 0)
  [ "$s1" != "0" ] && [ "$s1" = "$s2" ] && break
  sleep 120
done
echo "[watch] poly_chain4 stable at $s2 bytes; launching pipeline"
bash scratch/poly4_pipeline.sh > logs/poly4_pipeline.log 2>&1 \
  && echo POLY4_PIPELINE_OK >> logs/poly4_pipeline.log

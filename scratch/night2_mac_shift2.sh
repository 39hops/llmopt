#!/bin/bash
cd ~/code/llmopt
P=~/code/llmopt/.venv/bin/python
until grep -q "MAC_GRID_DONE" grid_gates.log 2>/dev/null; do sleep 300; done
$P scratch/synonym_test.py > synonym.log 2>&1
for s in 1 2 3; do
  $P scratch/holdout_v2.py checkpoints/seedvar_$s.pt 384 8 1536 6 "SEEDVAR-$s" >> holdout.log 2>&1
done
$P scratch/chain_carry.py chains > cc_chains.log 2>&1
$P scratch/chain_carry.py oneshot > cc_oneshot.log 2>&1
echo MAC_SHIFT2_DONE >> grid_gates.log

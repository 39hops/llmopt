#!/bin/bash
cd ~/code/llmopt
until grep -q "TUESDAY_NIGHT_DONE" holdout.log 2>/dev/null; do sleep 300; done
~/code/llmopt/.venv/bin/python scratch/kv_equiv.py mps > kv_equiv_mps.log 2>&1

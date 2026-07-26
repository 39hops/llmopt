#!/bin/bash
cd ~/code/llmopt
until [ -f logs/closers_done.marker ]; do sleep 120; done
CENSUS_SEC=300 .venv/bin/python scratch/metabolic_v5.py checkpoints/metab_v5_s1.pt data/stuck_states_v5_head.jsonl 120 > logs/v5_s2_retention.log 2>&1 && echo V5S2_DONE > logs/v5s2_done.marker

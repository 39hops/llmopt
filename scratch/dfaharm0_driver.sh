#!/bin/bash
# DFA-LOWER-HARM-DESK-0 driver: transplant gates (mps, 6 gates) then the
# ACT / alignment trajectory (CPU) on the seed-23 frontier arms. Zero
# training. Under the liverun interlock; marker on success only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
mkdir -p logs/dfaharm0
rc=0
.venv/bin/python scratch/dfa_harm_desk.py 2>&1 | tee logs/dfaharm0/harm.log || rc=$?
[ "$rc" -eq 0 ] || { mark_done logs/dfaharm0.DONE "$rc"; exit "$rc"; }
.venv/bin/python scratch/caf_actpost.py 2>&1 | tee logs/dfaharm0/actpost.log || rc=$?
mark_done logs/dfaharm0.DONE "$rc"

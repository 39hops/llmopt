#!/bin/bash
# WHY-DFA-FAILED-DESCRIPTIVE-0 driver: the zero-training postmortem of
# the four WRITER-DFA-1 qualification arms (scratch/dfa_postmortem.py),
# CPU only, under the liverun interlock; marker on success only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
mkdir -p logs/dfapost
rc=0
.venv/bin/python scratch/dfa_postmortem.py 2>&1 | tee logs/dfapost/postmortem.log || rc=$?
mark_done logs/dfapost.DONE "$rc"

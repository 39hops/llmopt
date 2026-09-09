#!/bin/bash
# WRITER-DFA-1 discovery driver (pre-reg RESULTS L68321 item 12): the
# paired same-W_0 BP control at seed 2 (stock recipe, MODE=bp) then the
# DFA arm at seed 2 with the (S, LR) frozen in
# logs/writerdfa1/qual_selection.json; nothing is read until both
# finish. Under the liverun interlock; marker on success only.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd
[ -z "$(git status --porcelain)" ] || { echo "driver: dirty tree"; git status --porcelain; exit 2; }
[ -f logs/writerdfa1/qual_selection.json ] || { echo "driver: qual_selection.json missing"; exit 3; }
[ -f logs/writerdfa1/act_envelope.json ] || { echo "driver: act_envelope.json missing (ACT observation must be booked first)"; exit 3; }
S=$(.venv/bin/python -c 'import json;s=json.load(open("logs/writerdfa1/qual_selection.json"))["selected"];print(s["s"])')
LR=$(.venv/bin/python -c 'import json;s=json.load(open("logs/writerdfa1/qual_selection.json"))["selected"];print(s["lr"])')
[ -n "$S" ] && [ -n "$LR" ] || { echo "driver: no selection"; exit 3; }
rc=0
echo "=== control MODE=bp seed 2 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
DISCOVERY=1 MODE=bp SEED=2 LR=3e-4 .venv/bin/python scratch/birth19m_dfa.py 2>&1 | tee logs/writerdfa1/train_disc_bp.log || rc=$?
[ "$rc" -eq 0 ] || { mark_done logs/writerdfa1_disc.DONE "$rc"; exit "$rc"; }
echo "=== DFA MODE=dfa seed 2 S=$S LR=$LR $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
DISCOVERY=1 MODE=dfa SEED=2 S=$S LR=$LR .venv/bin/python scratch/birth19m_dfa.py 2>&1 | tee logs/writerdfa1/train_disc_dfa.log || rc=$?
mark_done logs/writerdfa1_disc.DONE "$rc"

#!/bin/bash
# EX6-MED-0 full run: outcome-blind capture pass, then cells pass
# with token-identity qualification. Receipts under logs/ex6med/.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd

for f in logs/ex6med/zcap.jsonl logs/ex6med/cells.jsonl; do
  [ -e "$f" ] && { echo "REFUSING: $f exists"; exit 2; }
done

PASS=cap .venv/bin/python scratch/ex6med.py 2>&1 | tee logs/ex6med/cap.log
test "${PIPESTATUS[0]}" -eq 0 || exit 1
PASS=cells .venv/bin/python scratch/ex6med.py 2>&1 | tee logs/ex6med/cells.log
test "${PIPESTATUS[0]}" -eq 0 || exit 1
mark_done logs/ex6med/ex6med.DONE

#!/bin/bash
# EX6-PHASE-0 treatment driver (Mac): 4 modes x 3 fresh seeds
# (7001/8002/9003), rows stream into logs/ex6/ex6.jsonl.
# Pre-reg docs/preregs/ex6-phase-0.json + AMENDMENT -WRAPPER.
#
# FAIL-CLOSED launch gate (review-adopted): treatment fires only if
# the v2 qualification receipt holds EXACTLY the two expected
# cell-exact rows and no treatment receipt exists. Anything else
# exits nonzero and launches nothing.
. "$(dirname "$0")/lib/driver.sh"
llmopt_cd

.venv/bin/python - << 'EOF'
import json, sys
rows = [r for r in map(json.loads, open("logs/ex6/qual.jsonl"))
        if r["arm"].endswith("_v2")]
want = {"ex6_qual_none_v2": (59, {"1": 22, "2": 19, "3": 18}),
        "ex6_qual_all_v2": (78, {"1": 28, "2": 26, "3": 24})}
if len(rows) != 2:
    sys.exit(f"GATE: expected exactly 2 v2 qual rows, got {len(rows)}")
for r in rows:
    exp = want.get(r["arm"])
    if exp is None:
        sys.exit(f"GATE: unexpected v2 arm {r['arm']}")
    if (r["gate_ok"], r["gate_per_level"]) != exp:
        sys.exit(f"GATE: {r['arm']} = {r['gate_ok']} "
                 f"{r['gate_per_level']} != booked {exp}")
import os
if os.path.exists("logs/ex6/ex6.jsonl"):
    sys.exit("GATE: logs/ex6/ex6.jsonl already exists")
print("GATE: qualification cell-exact 2/2, treatment path clear")
EOF

for SEED in 7001 8002 9003; do
    echo "[ex6] === seed $SEED ===" >&2
    for MODE in NONE ALL PROMPT DECODE; do
        ARM="ex6_$(echo "$MODE" | tr '[:upper:]' '[:lower:]')"
        MODE="$MODE" ARM="$ARM" SEED="$SEED" N_EVAL=120 PERPROB=1 \
            LOG=logs/ex6/ex6.jsonl \
            PERPROB_LOG=logs/ex6/ex6_perprob.jsonl \
            .venv/bin/python scratch/ex6_phase.py
    done
done
mark_done logs/ex6/ex6.DONE $?

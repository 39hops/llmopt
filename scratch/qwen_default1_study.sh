#!/bin/bash
# QWEN-LOW-DEFAULT-1 chain (GO 2026-10-23): one stage, the integration smoke
# driver scratch/qwen_default1_smoke.py, which itself runs the release test
# suites in fresh subprocesses (the live ones on the tower) and then the
# fixture, the bounded multi-turn smoke and the contract checks; receipt
# logs/qwen_release/default1_smoke_2.json (refuses to overwrite; the first
# smoke's receipt default1_smoke.json is superseded by AMENDMENT
# QWEN-LOW-DEFAULT-1-SMOKE and kept as the record). Run under
# scripts/liverun.py (clean tree, no commits while live).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs/qwen_release
COMMIT=$(git rev-parse --short HEAD)
LOG=logs/qwen_release/default1_$COMMIT.log
PY=.venv/bin/python
echo "== smoke $(date -u +%H:%M:%S) $COMMIT" >> "$LOG"
TMP=$(mktemp)
$PY scratch/qwen_default1_smoke.py > "$TMP" 2>&1
rc=$?
grep -v Warning "$TMP" | grep -E "^\[|passed|failed" >> "$LOG"
if [ "$rc" -ne 0 ]; then echo "== FAILED smoke rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"; tail -30 "$TMP" >> "$LOG"; rm -f "$TMP"; exit "$rc"; fi
rm -f "$TMP"
echo "== DONE $(date -u +%H:%M:%S)" >> "$LOG"

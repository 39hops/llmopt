#!/bin/bash
# ATOM-DIET-TRAJECTORY-1 provenance-only REPAIR driver (registration
# RESULTS 2026-09-07, AMENDMENT ATOM-DIET-TRAJECTORY-1-REPAIR-0).
# Runs INSIDE the detached worktree /Users/artin/code/llmopt-repair,
# pinned to the original launch commit ec6de1ae, using that commit's
# instrument sources unchanged. The worktree already holds the two
# valid seed-5 births (snapshot dirs and their two original receipt
# rows, byte-identical). This driver reruns only the four invalidated
# births in the frozen original order with their original order
# indices, then ALL 24 post-hoc gates, the census and the independent
# verifier, all at ec6de1ae. The interpreter is the main checkout's
# venv by absolute path (the worktree carries no .venv so its tree
# stays clean). Marker fires on success only with the real rc.
set -euo pipefail
WT=/Users/artin/code/llmopt-repair
PY=/Users/artin/code/llmopt/.venv/bin/python
cd "$WT"
[ "$(git rev-parse --short HEAD)" = "ec6de1ae" ] || { echo "repair: worktree HEAD is not ec6de1ae"; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "repair: worktree dirty"; git status --porcelain; exit 2; }
[ "$(wc -l < logs/atomtraj1/births.jsonl | tr -d ' ')" = "2" ] || { echo "repair: evidence set must hold exactly the two seed-5 rows"; exit 2; }
mark() { if [ "$2" -eq 0 ]; then printf 'rc=0 ts=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$1"; else echo "repair: rc=$2, no marker"; fi; }
rc=0
i=2
for cell in stock_s6 atoms_s6 atoms_s7 stock_s7; do
  ARM="${cell%_s*}"; SEED="${cell#*_s}"
  echo "=== repair birth $i: arm=$ARM seed=$SEED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ARM=$ARM SEED=$SEED ORDER_INDEX=$i "$PY" scratch/birth19m_atoms_traj.py 2>&1 | tee "logs/atomtraj1/train_${cell}.log" || rc=$?
  [ "$rc" -eq 0 ] || { echo "repair: birth $cell failed rc=$rc"; mark logs/atomtraj1.DONE "$rc"; exit "$rc"; }
  i=$((i + 1))
done
echo "=== repair: all four births complete $(date -u +%Y-%m-%dT%H:%M:%SZ); 24 post-hoc gates ==="
"$PY" scratch/birth19m_atoms_trajgate.py 2>&1 | tee logs/atomtraj1/gates.log || rc=$?
[ "$rc" -eq 0 ] || { mark logs/atomtraj1.DONE "$rc"; exit "$rc"; }
echo "=== repair: census $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PY" scratch/atomtraj_census.py 2>&1 | tee logs/atomtraj1/census.log || rc=$?
[ "$rc" -eq 0 ] || { mark logs/atomtraj1.DONE "$rc"; exit "$rc"; }
echo "=== repair: verifier $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PY" scratch/atomtraj_verify.py 2>&1 | tee logs/atomtraj1/verify.log || rc=$?
mark logs/atomtraj1.DONE "$rc"
exit "$rc"

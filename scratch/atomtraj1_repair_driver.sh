#!/bin/bash
# ATOM-DIET-TRAJECTORY-1 provenance-only REPAIR driver (registration
# RESULTS 2026-09-07, AMENDMENT ATOM-DIET-TRAJECTORY-1-REPAIR-0 and its
# -FOLDS entry). Runs INSIDE the detached worktree
# /Users/artin/code/llmopt-repair, pinned to the original launch commit
# ec6de1ae, using that commit's instrument sources unchanged. The
# worktree already holds the two valid seed-5 births (snapshot dirs and
# their two original receipt rows, byte-identical to the first two rows
# of the main checkout's first-run receipt). This driver reruns only the
# four invalidated births in the frozen original order with their
# original order indices, then ALL 24 post-hoc gates, the census and the
# independent verifier, all at ec6de1ae. The interpreter is the main
# checkout's venv by absolute path (the worktree carries no .venv so its
# tree stays clean). HEAD and cleanliness are re-asserted before every
# process that writes a receipt. Marker fires on success only with the
# real rc.
set -euo pipefail
WT=/Users/artin/code/llmopt-repair
MAIN=/Users/artin/code/llmopt
PY=$MAIN/.venv/bin/python
LAUNCH=ec6de1ae
cd "$WT"
mkdir -p logs/atomtraj1 checkpoints/atomtraj1
guard() {
  [ "$(git rev-parse --short HEAD)" = "$LAUNCH" ] || { echo "repair: worktree HEAD is not $LAUNCH"; exit 2; }
  [ -z "$(git status --porcelain)" ] || { echo "repair: worktree dirty"; git status --porcelain; exit 2; }
}
guard
RECEIPT=logs/atomtraj1/births.jsonl
[ "$(wc -l < "$RECEIPT" | tr -d ' ')" = "2" ] || { echo "repair: evidence set must hold exactly the two seed-5 rows"; exit 2; }
head -n 2 "$MAIN/$RECEIPT" | cmp -s - "$RECEIPT" || { echo "repair: evidence rows differ from the first-run seed-5 rows"; exit 2; }
mark() { if [ "$2" -eq 0 ]; then printf 'rc=0 ts=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$1"; else echo "repair: rc=$2, no marker"; fi; }
rc=0
i=2
for cell in stock_s6 atoms_s6 atoms_s7 stock_s7; do
  ARM="${cell%_s*}"; SEED="${cell#*_s}"
  guard
  echo "=== repair birth $i: arm=$ARM seed=$SEED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  ARM=$ARM SEED=$SEED ORDER_INDEX=$i "$PY" scratch/birth19m_atoms_traj.py 2>&1 | tee "logs/atomtraj1/train_${cell}.log" || rc=$?
  [ "$rc" -eq 0 ] || { echo "repair: birth $cell failed rc=$rc"; mark logs/atomtraj1.DONE "$rc"; exit "$rc"; }
  i=$((i + 1))
done
guard
echo "=== repair: all four births complete $(date -u +%Y-%m-%dT%H:%M:%SZ); 24 post-hoc gates ==="
"$PY" scratch/birth19m_atoms_trajgate.py 2>&1 | tee logs/atomtraj1/gates.log || rc=$?
[ "$rc" -eq 0 ] || { mark logs/atomtraj1.DONE "$rc"; exit "$rc"; }
guard
echo "=== repair: census $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PY" scratch/atomtraj_census.py 2>&1 | tee logs/atomtraj1/census.log || rc=$?
[ "$rc" -eq 0 ] || { mark logs/atomtraj1.DONE "$rc"; exit "$rc"; }
guard
echo "=== repair: verifier $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
"$PY" scratch/atomtraj_verify.py 2>&1 | tee logs/atomtraj1/verify.log || rc=$?
mark logs/atomtraj1.DONE "$rc"
exit "$rc"

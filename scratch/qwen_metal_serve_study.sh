#!/bin/bash
# Research round 2026-10-19: hardened prefill entry point A/B and E0
# recount (Phase A), E0 disagreement census (Phase B), SwiGLU permutation
# study (Phase C) and the Strassen bound test (Phase D), one stage per
# process, in order; a failed stage stops the chain with a FAILED line.
# START=<stage> resumes from a stage; ONLY="a b" runs a subset.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/qmetal/prefill_serve/study.log
mkdir -p logs/qmetal/prefill_serve
PY=.venv/bin/python
START=${START:-}
ONLY=${ONLY:-}
started=0
run() {
  local name=$1; shift
  if [ -n "$START" ] && [ "$started" = 0 ]; then
    if [ "$name" = "$START" ]; then started=1; else echo "== skip $name (START=$START)" >> "$LOG"; return 0; fi
  fi
  if [ -n "$ONLY" ] && ! echo " $ONLY " | grep -q " $name "; then echo "== skip $name (ONLY=$ONLY)" >> "$LOG"; return 0; fi
  echo "== $name $(date -u +%H:%M:%S)" >> "$LOG"
  TMP=$(mktemp)
  "$@" > "$TMP" 2>&1
  rc=$?
  grep -v Warning "$TMP" | grep -E "^\[" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    echo "== FAILED $name rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
    tail -20 "$TMP" >> "$LOG"
    rm -f "$TMP"
    exit "$rc"
  fi
  rm -f "$TMP"
}
COMMIT=$(git rev-parse --short HEAD)
run ab_fused          env MODE=ab ARM=fused $PY scratch/qwen_metal_prefill_serve_ab.py
run ab_gemm           env MODE=ab ARM=gemm  $PY scratch/qwen_metal_prefill_serve_ab.py
run ab_gemm_nolimit   env MODE=ab ARM=gemm TAG=_nolimit CACHE_LIMIT=none $PY scratch/qwen_metal_prefill_serve_ab.py
run compare           env MODE=compare      $PY scratch/qwen_metal_prefill_serve_ab.py
run census            env MODE=census       $PY scratch/qwen_metal_prefill_serve_ab.py
run disagree          env CENSUS_RECEIPT=logs/qmetal/prefill_serve/census_$COMMIT.json $PY scratch/qwen_e0_disagreement_census.py
run perm              $PY scratch/mlp_permutation_study.py
run strassen          $PY scratch/strassen_bench.py
echo "== study done $(date -u +%H:%M:%S)" >> "$LOG"

#!/bin/bash
# Decode-once GEMM prefill: measurement chain, one process per stage.
#   bench   crossover per shape, decode and GEMM parts, numerics (fused vs GEMM vs float64)
#   floor   256-token chunk with the hybrid path (full, no_head, no_w4_mlp, no_fused)
#   ab      fused control against the hybrid at 1,024 real tokens (full and final head)
#   mem     one process per row: fused and hybrid, final head, 512 / 1,024 / 2,048 / 4,096
#   qual    the E0 census (24 prompts, 4,722 positions) through the hybrid path
# A failed stage stops the chain with its exit code and a FAILED line; the
# "study done" marker is written only when every stage exited 0.
# START=<prefix> resumes after a failure; ONLY="<prefix> ..." selects stages.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=logs/qmetal/prefill_gemm/study.log
mkdir -p logs/qmetal/prefill_gemm
run_stage() {
  local label=$1; shift
  echo "== $label $(date -u +%H:%M:%S)" >> "$LOG"
  TMP=$(mktemp)
  "$@" > "$TMP" 2>&1
  local rc=$?
  grep -v Warning "$TMP" | grep "^\[" >> "$LOG"
  if [ "$rc" -ne 0 ]; then
    grep -v Warning "$TMP" | tail -12 >> "$LOG"
    echo "== study FAILED at $label rc=$rc $(date -u +%H:%M:%S)" >> "$LOG"
    rm -f "$TMP"
    exit "$rc"
  fi
  rm -f "$TMP"
}
START=${START:-bench}
ONLY=${ONLY:-}
started=0
run() {
  if [ "$started" -eq 0 ]; then
    case "$1" in "$START"*) started=1;; *) echo "== skip $1 (START=$START)" >> "$LOG"; return 0;; esac
  fi
  if [ -n "$ONLY" ]; then
    local hit=0 pfx
    for pfx in $ONLY; do case "$1" in "$pfx"*) hit=1;; esac; done
    if [ "$hit" -eq 0 ]; then echo "== skip $1 (ONLY=$ONLY)" >> "$LOG"; return 0; fi
  fi
  run_stage "$@"
}
PY=.venv/bin/python
G=QMETAL_PREFILL_GEMM
run bench env $PY scratch/bench_prefill_gemm.py
run floor env $G=1 MODE=floor CHUNK=256 ROUNDS=3 OUT=logs/qmetal/prefill_gemm/floor_hybrid_$(git rev-parse --short HEAD).json $PY scratch/qmetal_prefill_kernel_study.py
run ab env MODE=ab CTX=1024 CHUNK=256 OUT=logs/qmetal/prefill_gemm/ab_$(git rev-parse --short HEAD).json $PY scratch/qwen_metal_prefill_head.py
for cfg in "512 0" "512 1" "1024 0" "1024 1" "2048 0" "2048 1" "4096 0" "4096 1"; do
  set -- $cfg
  run "mem ctx $1 gemm $2" env $G=$2 MODE=mem CTX=$1 CHUNK=256 HEAD=final OUT=logs/qmetal/prefill_gemm/mem_ctx$1_chunk256_final_gemm$2.json $PY scratch/qwen_metal_prefill_head.py
done
run qual env $G=1 LONG_CTX=512,1024 OUT=logs/qwen_metal_qual/C_gemm $PY scratch/qwen_metal_qual.py
echo "== study done $(date -u +%H:%M:%S)" >> "$LOG"

#!/usr/bin/env bash
# SOURCE-TREE green, the single executable definition (producer-
# consumer rule applied to CI): ci.yml's tests job calls THIS and
# /qualify's ritual calls THIS. Scope is honest: the wheel and
# core-deps CI jobs are SEPARATE checks this script does not cover
# — "source green" is not "pipeline green". GPU/MLX/toolchain
# tests skip cleanly by design.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python \
    || echo python)}
COV=$($PY -c "import pytest_cov" 2>/dev/null \
    && echo "--cov=llmopt --cov-report=" || true)

echo "== generated docs current =="
# --check never writes: this script is a CHECK and must leave the
# tree byte-identical (read-only inspection invariant, 2026-09-29).
# On drift each generator prints what is stale; regenerate by running
# it without --check and commit the result.
$PY scripts/gen_index.py --check
$PY scripts/gen_codemap.py --check
$PY scripts/gen_results_index.py --check
$PY scripts/gen_receipt_lock.py --check

echo "== pytest =="
$PY -m pytest tests/ -q $COV

echo "== ruff (enforced tier) =="
$PY -m ruff check llmopt tests scripts

echo "== ruff (scratch, report only) =="
$PY -m ruff check scratch --exit-zero

echo "== generated README in sync =="
$PY scripts/gen_readme.py --check

echo "== SOURCE TREE GREEN (wheel/core-deps are separate CI jobs) =="

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
$PY scripts/gen_index.py >/dev/null
$PY scripts/gen_codemap.py >/dev/null
$PY scripts/gen_results_index.py >/dev/null
git diff --quiet -- scripts/INDEX.md docs/CODEMAP.md \
    docs/results-index.jsonl \
    || { echo "STALE generated docs (diff above)"; git --no-pager \
         diff --stat -- scripts/INDEX.md docs/CODEMAP.md \
         docs/results-index.jsonl; exit 1; }

echo "== pytest =="
$PY -m pytest tests/ -q $COV

echo "== ruff (enforced tier) =="
$PY -m ruff check llmopt tests scripts

echo "== ruff (scratch, report only) =="
$PY -m ruff check scratch --exit-zero

echo "== generated README in sync =="
$PY scripts/gen_readme.py --check

echo "== SOURCE TREE GREEN (wheel/core-deps are separate CI jobs) =="

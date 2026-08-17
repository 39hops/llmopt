#!/usr/bin/env bash
# The single executable definition of "green" (producer-consumer
# rule applied to CI itself, 2026-08-17): .github/workflows/ci.yml
# calls THIS, and /qualify's clean-worktree ritual calls THIS. If a
# check matters, it lives here; a ritual that runs a subset is half
# the pipeline. GPU/MLX/toolchain tests skip cleanly by design.
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

echo "== ALL GREEN =="

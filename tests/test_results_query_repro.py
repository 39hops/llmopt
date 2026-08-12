"""tests/test_results_query_repro.py"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repro_prints_worktree_command():
    row = next(json.loads(l) for l in
               (ROOT / "docs" / "results-index.jsonl").open()
               if json.loads(l).get("code_commit"))
    out = subprocess.run(
        [sys.executable, "scripts/results_query.py", "--repro", row["id"]],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0
    assert f"git worktree add" in out.stdout
    assert row["code_commit"][:12] in out.stdout

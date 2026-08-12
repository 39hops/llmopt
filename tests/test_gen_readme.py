"""tests/test_gen_readme.py"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readme_ledger_counts_match_findings():
    out = subprocess.run(
        [sys.executable, "scripts/gen_readme.py", "--check"],
        cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr

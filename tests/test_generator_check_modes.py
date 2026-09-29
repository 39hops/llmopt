"""Every generator behind the CI "generated docs current" gate must
offer a non-mutating --check (exit 1 on drift, zero writes) and an
explicit --out destination, so freshness can be verified without
touching the tree. The docs-marked test asserts the committed
generated files are current."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

GENERATORS = {
    "scripts/gen_index.py": "scripts/INDEX.md",
    "scripts/gen_codemap.py": "docs/CODEMAP.md",
}


def _run(*args, cwd=ROOT):
    return subprocess.run([PY, *args], cwd=cwd, capture_output=True,
                          text=True, timeout=600)


@pytest.mark.parametrize("gen", sorted(GENERATORS))
def test_out_then_check_roundtrip_and_check_never_writes(gen, tmp_path):
    out = tmp_path / "out.md"
    r = _run(gen, "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    assert out.exists()
    r = _run(gen, "--check", "--out", str(out))
    assert r.returncode == 0, r.stdout + r.stderr
    out.write_text(out.read_text() + "\ndrift\n")
    before = out.read_bytes()
    r = _run(gen, "--check", "--out", str(out))
    assert r.returncode == 1
    assert out.read_bytes() == before  # --check wrote nothing


@pytest.mark.docs
@pytest.mark.parametrize("gen", sorted(GENERATORS) + [
    "scripts/gen_results_index.py", "scripts/gen_receipt_lock.py"])
def test_committed_generated_docs_are_current(gen):
    r = _run(gen, "--check")
    assert r.returncode == 0, (
        f"{gen} --check reports drift; regenerate and commit:\n"
        + r.stdout + r.stderr)

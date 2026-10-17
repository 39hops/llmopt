"""The speculative-decoding theorems compile under the pinned Lean
toolchain when it and the Mathlib build cache are present; skipped
otherwise (CI has no Lean). The file must contain no `sorry`."""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEAN_DIR = ROOT / "scratch" / "leancheck"
FILE = LEAN_DIR / "SpecDecoding.lean"
LAKE = pathlib.Path.home() / ".elan" / "bin" / "lake"


def test_spec_decoding_lean_has_no_sorry_outside_the_header():
    body = FILE.read_text().split("namespace SpecDecoding", 1)[1]
    assert "sorry" not in body


@pytest.mark.skipif(not LAKE.exists() or shutil.which("git") is None
                    or not (LEAN_DIR / ".lake" / "packages" / "mathlib" / ".lake" / "build").exists(),
                    reason="Lean toolchain or Mathlib build cache absent")
def test_spec_decoding_lean_compiles():
    r = subprocess.run([str(LAKE), "env", "lean", str(FILE)], cwd=LEAN_DIR,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "error" not in r.stdout + r.stderr

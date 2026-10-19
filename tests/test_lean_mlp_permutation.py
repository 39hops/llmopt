"""The SwiGLU neuron-permutation theorems (scratch/leancheck/
MlpPermutation.lean) compile under the pinned Lean toolchain when it and
the Mathlib build cache are present; skipped otherwise (CI has no Lean).
The file must contain no `sorry`, no `axiom` and no `opaque`."""
import pathlib
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEAN_DIR = ROOT / "scratch" / "leancheck"
FILE = LEAN_DIR / "MlpPermutation.lean"
LAKE = pathlib.Path.home() / ".elan" / "bin" / "lake"


def test_mlp_permutation_lean_has_no_sorry_axiom_or_opaque():
    body = FILE.read_text()
    assert "sorry" not in body
    assert "\naxiom " not in body and "\nopaque " not in body
    for name in ("gatedMlp_perm_invariant", "gatedMlp_perm_invariant_fun", "swiglu_perm_invariant",
                 "permMatrix_mul_eq_submatrix", "mul_permMatrix_transpose_eq_submatrix"):
        assert f"theorem {name}" in body


@pytest.mark.skipif(not LAKE.exists() or shutil.which("git") is None
                    or not (LEAN_DIR / ".lake" / "packages" / "mathlib" / ".lake" / "build").exists(),
                    reason="Lean toolchain or Mathlib build cache absent")
def test_mlp_permutation_lean_compiles_on_standard_axioms(tmp_path):
    probe = tmp_path / "Probe.lean"
    probe.write_text(FILE.read_text() + "\n#print axioms MlpPermutation.swiglu_perm_invariant\n"
                     "#print axioms MlpPermutation.gatedMlp_perm_invariant\n")
    r = subprocess.run([str(LAKE), "env", "lean", str(probe)], cwd=LEAN_DIR,
                       capture_output=True, text=True, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "error" not in out
    for line in out.splitlines():
        if "depends on axioms" in line:
            used = line.split("depends on axioms:", 1)[1].strip(" []")
            assert set(a.strip() for a in used.split(",")) <= {"propext", "Classical.choice", "Quot.sound"}, line

"""Adoption guards for llmopt.lab (spec 2026-08-05-llmopt-lab-extraction).

Phase 3 complete (2026-08-12): all five pairs are shims — the frozen
scripts/scratch originals re-export the canonical llmopt.lab bodies.
Source-identity guards are gone; behavior is pinned by batteries:
WAVE_CASES here, tests/test_lab_verify_gen_battery.py (Phase D
167/167 replay + gen pins), tests/test_gate_battery.py (booked
sampler-amendment replay + gate problem-grid pins),
tests/test_lab_keepsets.py, tests/test_lab_oracle.py.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a scripts/ module without polluting sys.path for others."""
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bvf():
    pytest.importorskip("sympy")
    return _load_script("bench_verify_fast")


@pytest.fixture(scope="module")
def bst():
    pytest.importorskip("torch")  # script imports torch at module top
    return _load_script("bench_step_tokens")


def test_verify_shim_binds_lab_bodies(bvf):
    from llmopt.lab import verify
    assert bvf._wave_worker is verify._wave_worker
    assert bvf.verify_wave is verify.verify_wave
    assert bvf._WAVE_CACHE is verify._WAVE_CACHE


def test_gen_shim_binds_lab_body(bst):
    from llmopt.lab import gen
    assert bst._gen_isolated is gen._gen_isolated


# Behavior battery: true step, solved step, perturbed, sign flip,
# garbage, unresolved-Integral candidate, constant-only pair.
WAVE_CASES = [
    ("Integral(2*x + 3, x)",
     ["Integral(2*x, x) + Integral(3, x)",   # true, unsolved
      "x**2 + 3*x",                          # true, solved
      "x**2 + 4*x",                          # perturbed
      "x**2 - 3*x",                          # sign flip
      "x***bogus((",                         # garbage
      "Integral(sin(x)/x, x) + 1"]),         # unresolved carrier
    ("Integral(x*cos(x), x)",
     ["x*sin(x) - Integral(sin(x), x)",      # true (by parts)
      "x*sin(x) + cos(x)",                   # true, solved
      "x*cos(x) + sin(x)"]),                 # wrong
    ("3", ["3", "2 + 1", "4"]),              # constant-only branch
]


# expected ok-verdicts per WAVE_CASES row, same order as the cands
WAVE_EXPECT = [
    [True, True, False, False, False, False],
    [True, True, False],
    [True, True, False],
]


def test_verify_wave_expected_verdicts():
    pytest.importorskip("sympy")
    from llmopt.lab.verify import verify_wave as lab_vw
    for (prev, cands), want in zip(WAVE_CASES, WAVE_EXPECT):
        got = lab_vw(prev, cands)
        assert [got[c][0] for c in cands] == want, prev


@pytest.fixture(scope="module")
def sgm():
    pytest.importorskip("torch")
    return _load_script("step_grpo_micro")


def test_gate_shim_binds_lab_bodies(sgm):
    from llmopt.lab import gate
    assert sgm.sample_wave_lp is gate.sample_wave_lp
    assert sgm.gate_eval is gate.gate_eval


def test_gate_shim_line_anchors():
    # RESULTS cites step_grpo_micro.py:65 and :184 (inside the old
    # bodies); the shim is line-count-preserving and must keep the
    # quoted fragments on their booked lines.
    text = (ROOT / "scripts" / "step_grpo_micro.py").read_text() \
        .splitlines()
    assert "torch.multinomial(probs[b], 1, generator=gens[b])" \
        in text[64]
    assert "[gate] weights sha" in text[183]
    assert text[76].startswith("from llmopt.lab.gate import")
    assert text[223].startswith("from llmopt.lab.gate import")
    assert text[79].startswith("def collect(")
    assert text[226].startswith("def main(")


def test_gate_constants_match_lineage(sgm):
    from llmopt.lab import gate
    assert gate.GRPO_MICRO.wave == sgm.B
    assert gate.GRPO_MICRO.levels == sgm.GATE_LEVELS
    assert gate.GRPO_MICRO.n == sgm.GATE_N
    assert gate.GRPO_MICRO.band == sgm.GATE_BAND
    # module defaults start at the standard lineage
    assert (gate.B, gate.GATE_LEVELS, gate.GATE_N, gate.GATE_BAND) == \
        (sgm.B, sgm.GATE_LEVELS, sgm.GATE_N, sgm.GATE_BAND)

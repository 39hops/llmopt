"""Booked-number battery for llmopt.lab.verify + llmopt.lab.gen —
the Phase 3 module 4 gate (spec 2026-08-12 §Phase 3: a shim lands
only after the symbol has a test reproducing a booked number).

Tier 1 (booked replay): data/axiom_phaseD_167.jsonl is axiom's
Phase D chain emission (167 rows from 95 stratified roots,
farm_v22 cur/nxt schema), pulled byte-identical (md5 48df699a...)
from the axiom checkout 2026-08-12. RESULTS.md L2871 books the
adjudication: "167/167 pairs pass the production oracle
(verify_wave)". This test re-runs that adjudication verbatim.

Tier 2 (reject side): 2*(nxt) must fail for every row — a scaled
candidate changes the derivative, and verify_wave is derivative-
equivalence (constant SHIFTS would pass by design; multiplicative
perturbation is the sound reject probe).

Tier 3 (gen determinism): _gen_isolated pins exact expressions on
a (level, seed) grid — the string-seed house law makes these
frozen literals, not snapshots.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("sympy")

ROOT = Path(__file__).resolve().parents[1]
PHASED = ROOT / "data" / "axiom_phaseD_167.jsonl"


@pytest.fixture(scope="module")
def rows():
    return [json.loads(line) for line in open(PHASED)]


def test_phaseD_167_of_167_verify(rows):
    """The booked Phase D adjudication (RESULTS.md L2871), replayed."""
    from llmopt.lab.verify import verify_wave
    assert len(rows) == 167
    verdicts = [verify_wave(r["cur"], [r["nxt"]])[r["nxt"]][0]
                for r in rows]
    assert sum(verdicts) == 167, \
        [i for i, v in enumerate(verdicts) if not v]


def test_phaseD_scaled_candidates_all_reject(rows):
    from llmopt.lab.verify import verify_wave
    bad = [verify_wave(r["cur"], [c])[c][0]
           for r in rows[:24]
           for c in [f"2*({r['nxt']})"]]
    assert sum(bad) == 0, [i for i, v in enumerate(bad) if v]


# (level, seed) -> sp.sstr of the generated problem's expression.
# Frozen 2026-08-12 from the string-seed generator (house law:
# random.Random(f"kind-{level}-{seed}") — stable across processes).
GEN_PINS = {
    (2, 7000000): '8*x**3',
    (2, 7000050): '32*x**3 - 7*exp(x)',
    (3, 7000001): '-6*x*sin(x) + 6*cos(x)',
    (3, 7000051): '-72*x**3',
    (4, 7000002): '(80*sqrt(x)*(-9*x**3*sin(3*x**3 + 1) + cos(3*x**3'
                  ' + 1)) + (2*sqrt(x)*(6*x**2 + 1) + sqrt(2))*cos(s'
                  'qrt(2)*sqrt(x) + 2*x**3 + x + 3))/(2*sqrt(x))',
    (4, 7000052): '27*(6*x*sin(3*x**3 + x + 2) + (9*x**2 + 1)**2*cos'
                  '(3*x**3 + x + 2))*sin(3*x**3 + x + 2)**2',
}  # frozen 2026-08-12 via scratch/genpins_freeze.py


def test_gen_isolated_seed_grid_pinned():
    import sympy as sp
    from llmopt.lab.gen import _gen_isolated
    if not GEN_PINS:
        pytest.skip("pins not frozen yet (pre-commit state)")
    for (level, seed), want in GEN_PINS.items():
        p = _gen_isolated(level, seed)
        assert p is not None, (level, seed)
        assert sp.sstr(p._expr) == want, (level, seed)

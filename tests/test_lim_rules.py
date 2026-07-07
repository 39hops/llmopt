"""Limit rules — the origin-story rung: limits resisted LoRA training
(<=21%), motivating the engine; these moves close the loop."""

import sympy as sp

from llmopt.search.derivation import beam_search
from llmopt.search.rules import LIM_RULES

x = sp.Symbol("x")
RULES = dict(LIM_RULES)


def test_l_direct_continuity():
    (v,) = RULES["l_direct"](sp.Limit(x**2 + 1, x, 2))
    assert v == 5
    assert RULES["l_direct"](sp.Limit(sp.sin(x) / x, x, 0)) == []  # 0/0


def test_l_factor_cancel_planted_zero():
    node = sp.Limit((x**2 - 4) / (x - 2), x, 2)
    (rw,) = RULES["l_factor_cancel"](node)
    assert isinstance(rw, sp.Limit)
    assert sp.limit((x**2 - 4) / (x - 2), x, 2) == rw.doit()


def test_l_hopital_emits_unevaluated_derivatives():
    node = sp.Limit(sp.sin(x) / x, x, 0)
    (rw,) = RULES["l_hopital"](node)
    assert rw.has(sp.Derivative)  # chains into the diff rules
    assert rw.doit() == 1


def test_beam_solves_mathgen_style_limit():
    # the make_limit family shape: planted (x - a) common factor
    r = beam_search(sp.Limit((x**2 - 1) / (x - 1), x, 1))
    assert r.solved
    assert sp.simplify(r.state.expr - 2) == 0
    assert r.state.plies >= 2  # cancel then substitute: visible steps


def test_beam_solves_sinx_over_x_via_hopital_chain():
    r = beam_search(sp.Limit(sp.sin(x) / x, x, 0), max_plies=16)
    assert r.solved
    assert sp.simplify(r.state.expr - 1) == 0
    assert any(h.startswith("l_hopital@") for h in r.state.history)
    assert any(h.startswith("d_") or "d_chain" in h for h in r.state.history)

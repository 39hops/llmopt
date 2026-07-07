"""Taylor polynomials + continuity kinds."""

import sympy as sp

from llmopt.mathgen.problems import make_continuity, make_taylor

x = sp.Symbol("x")


def test_taylor_valid_and_equivalent_forms():
    for seed in range(15):
        p = make_taylor(level=1, seed=seed)
        assert p.check(p.answer), p.prompt
        ans = sp.sympify(p.answer)
        assert p.check(sp.sstr(sp.factor(ans))) or p.check(sp.sstr(ans))
        assert not p.check(p.answer + " + x**9")


def test_taylor_shifted_center():
    for seed in range(10):
        p = make_taylor(level=2, seed=seed)
        assert p.check(p.answer), p.prompt


def test_continuity_value_correct():
    for seed in range(15):
        p = make_continuity(level=1, seed=seed)
        assert p.check(p.answer), p.prompt
        assert not p.check(p.answer + " + 1")


def test_determinism():
    assert [make_taylor(1, s).prompt for s in range(20)] == \
           [make_taylor(1, s).prompt for s in range(20)]
    assert [make_continuity(1, s).prompt for s in range(20)] == \
           [make_continuity(1, s).prompt for s in range(20)]

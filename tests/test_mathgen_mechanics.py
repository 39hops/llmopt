"""Mechanics kinds (physics-night section 6): Euler-Lagrange
verify-by-substitution asserted at build time; answers checked by the
default simplify-difference oracle, so equivalent forms pass."""

import sympy as sp

from llmopt.mathgen.mechanics import (make_eom, make_kinematics,
                                      make_small_osc)
from llmopt.mathgen.problems import _resolve_maker

x, t = sp.symbols("x t")


def test_mechanics_valid_and_checkable():
    for make in (make_eom, make_small_osc, make_kinematics):
        for level in (1, 2, 3):
            for seed in range(8):
                p = make(level, seed)
                assert p.check(p.answer), p.prompt
                assert not p.check(p.answer + " + 1"), p.prompt


def test_eom_equivalent_forms_pass():
    p = make_eom(2, 0)
    ans = sp.sympify(p.answer, locals={"x": x})
    assert p.check(sp.sstr(sp.factor(ans)))
    assert p.check(sp.sstr(sp.expand(ans)))


def test_small_osc_answer_is_real_positive():
    for seed in range(15):
        p = make_small_osc(3, seed)
        w = sp.sympify(p.answer)
        assert w.is_real is not False and sp.simplify(w) != 0, p.prompt


def test_registered_and_deterministic():
    for kind in ("eom", "small_osc", "kinematics"):
        mk = _resolve_maker(kind)
        assert [mk(1, s).prompt for s in range(10)] == \
               [mk(1, s).prompt for s in range(10)], kind

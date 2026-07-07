"""Discrete-math kinds: closed-form sums (symbolic equality) and
recurrences (verify-by-substitution, the ODE-check pattern)."""

import sympy as sp

from llmopt.mathgen.problems import make_recurrence, make_sum

n = sp.Symbol("n")


def test_sum_valid_and_checkable():
    for seed in range(15):
        p = make_sum(level=1, seed=seed)
        assert p.check(p.answer), p.prompt
        assert not p.check(p.answer + " + n")


def test_sum_equivalent_forms_pass():
    p = make_sum(level=1, seed=0)
    ans = sp.sympify(p.answer, locals={"n": n})
    assert p.check(sp.sstr(sp.expand(ans)))
    assert p.check(sp.sstr(sp.factor(ans)))


def test_recurrence_verify_not_compare():
    for seed in range(10):
        p = make_recurrence(level=1, seed=seed)
        assert p.check(p.answer), p.prompt
        assert not p.check("n**7")


def test_determinism():
    assert [make_sum(1, s).prompt for s in range(20)] == \
           [make_sum(1, s).prompt for s in range(20)]
    assert [make_recurrence(1, s).prompt for s in range(20)] == \
           [make_recurrence(1, s).prompt for s in range(20)]

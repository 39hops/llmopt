"""Complex-coefficient calculus kinds (spec: mathgen-expansion Part A)."""

import random

import sympy as sp

from llmopt.mathgen.problems import make_cdiff, make_cint, parse_answer

X = sp.Symbol("x")


def test_cdiff_valid_and_checkable():
    for seed in range(20):
        p = make_cdiff(level=2, seed=seed)
        assert p.check(p.answer), p.prompt
        assert not p.check(p.answer + " + 1")


def test_cdiff_actually_complex():
    hits = 0
    for seed in range(20):
        p = make_cdiff(level=2, seed=seed)
        hits += sp.sympify(p.answer).has(sp.I)
    assert hits > 10, "most level-2 cdiff answers should involve I"


def test_cint_any_antiderivative_passes():
    p = make_cint(level=2, seed=3)
    F = sp.sympify(p.answer)
    assert p.check(sp.sstr(F + 7)), "constant offset must pass"
    assert not p.check(sp.sstr(F * 2))


def test_parse_answer_understands_I():
    assert parse_answer("(2 + 3*I)*x") == (2 + 3 * sp.I) * X


def test_determinism_and_space():
    a = [make_cdiff(2, s).prompt for s in range(50)]
    b = [make_cdiff(2, s).prompt for s in range(50)]
    assert a == b
    assert len(set(a)) > 45  # collision guard, small-n version

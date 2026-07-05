"""Problem generator + symbolic checker (pure sympy, no model)."""

import pytest

sp = pytest.importorskip("sympy")

from llmopt.mathgen.evaluate import extract_expression
from llmopt.mathgen.problems import (
    ALL_KINDS,
    X,
    make_dataset,
    make_differentiate,
    make_integrate,
    make_limit,
    parse_answer,
)


def test_deterministic_per_seed():
    assert make_differentiate(2, 7) == make_differentiate(2, 7)
    assert make_differentiate(2, 7) != make_differentiate(2, 8)


def test_own_answer_passes_check():
    for p in make_dataset(120, kinds=ALL_KINDS, seed=1):
        assert p.check(p.answer), (p.kind, p.level, p.prompt, p.answer)


def test_all_kinds_generate():
    ds = make_dataset(60, kinds=ALL_KINDS, seed=5)
    assert {p.kind for p in ds} == set(ALL_KINDS)


def test_definite_integral_matches_sympy():
    from llmopt.mathgen.problems import make_definite_integral

    for s in range(5):
        p = make_definite_integral(2, s)
        body = p.prompt.split("integral of ")[1]
        integrand = parse_answer(body.split(" with respect")[0])
        a, b = body.split("from ")[1].split(" to ")
        ref = sp.integrate(integrand, (X, int(a), int(b)))
        assert sp.simplify(ref - parse_answer(p.answer)) == 0


def test_wrong_answer_fails_check():
    for p in make_dataset(30, seed=2):
        assert not p.check(f"({p.answer}) + x**7")


def test_equivalent_forms_accepted():
    p = make_differentiate(1, 3)
    expanded = sp.sstr(sp.expand(sp.sympify(p.answer, locals={"x": X})))
    factored = sp.sstr(sp.factor(sp.sympify(p.answer, locals={"x": X})))
    assert p.check(expanded) and p.check(factored)


def test_integrate_accepts_any_constant():
    p = make_integrate(2, 4)
    assert p.check(f"({p.answer}) + 17")  # +C differs, still an antiderivative
    assert p.check(f"({p.answer}) + C")


def test_limit_matches_sympy_limit():
    for s in range(10):
        p = make_limit(2, s)
        num_str, den_str = p.prompt.split(" of (")[1].rstrip(")").split(") / (")
        a = int(p.prompt.split("approaches ")[1].split(" of")[0])
        f = parse_answer(num_str) / parse_answer(den_str)
        assert sp.simplify(sp.limit(f, X, a) - parse_answer(p.answer)) == 0


def test_parse_answer_robust():
    assert parse_answer("2*x + 3") == 2 * X + 3
    assert parse_answer("f(x) = 2*x") == 2 * X  # tolerates lhs
    assert parse_answer("sin(x)**2") == sp.sin(X) ** 2
    assert parse_answer("!!garbage((") is None


def test_train_eval_split_disjoint():
    ev = {p.prompt for p in make_dataset(100, seed=99)}
    train = {p.prompt for p in make_dataset(200, seed=0, exclude=frozenset(ev))}
    assert not (train & ev)  # exclusion is a guarantee, not a probability
    assert len(train) == 200  # dedup still fills the requested count


def test_extract_expression_takes_first_line():
    assert extract_expression("2*x + 1\nBecause the derivative...") == "2*x + 1"
    assert extract_expression("\n  cos(x)  \n") == "cos(x)"

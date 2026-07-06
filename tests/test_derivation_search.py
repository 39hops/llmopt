"""Derivation search foundations: legality, HCE ordering, beam solves."""

import sympy as sp

from llmopt.search.derivation import (
    State,
    beam_search,
    hce,
    is_solved,
    successors,
)

x = sp.Symbol("x")


def test_successors_are_equal_to_parent():
    s = State(sp.Derivative(x**3 + sp.sin(x), x))
    for name, child in successors(s):
        assert sp.simplify(child.expr - s.expr.doit()) == 0 or sp.simplify(
            child.expr.doit() - s.expr.doit()
        ) == 0, name


def test_hce_prefers_solved_states():
    unsolved = State(sp.Derivative(x**2, x))
    solved = State(2 * x)
    assert hce(solved) < hce(unsolved)


def test_solved_detection():
    assert is_solved(State(2 * x))
    assert not is_solved(State(sp.Integral(x, x)))


def test_beam_solves_derivative():
    r = beam_search(sp.Derivative(x**3 + sp.sin(x), x))
    assert r.solved
    assert sp.simplify(r.state.expr - (3 * x**2 + sp.cos(x))) == 0


def test_beam_solves_nested_derivative():
    r = beam_search(sp.Derivative(sp.Derivative(x**4, x), x))
    assert r.solved
    assert sp.simplify(r.state.expr - 12 * x**2) == 0


def test_beam_solves_integral():
    r = beam_search(sp.Integral(3 * x**2 + 2 * x, x))
    assert r.solved
    assert sp.simplify(sp.diff(r.state.expr, x) - (3 * x**2 + 2 * x)) == 0


def test_beam_records_history():
    r = beam_search(sp.Derivative(x**2, x))
    assert r.solved
    assert len(r.state.history) == r.state.plies >= 1

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


def test_higher_order_unsolved_at_rung1():
    # sympy collapses Derivative(Derivative(f,x),x) into a single
    # second-order node at construction — there is no unevaluated
    # "peeled" form to rewrite to. Rung-1 rules are first-order only
    # (spec), so this is an honest miss, not a bug.
    r = beam_search(sp.Derivative(sp.Derivative(x**4, x), x))
    assert not r.solved


def test_integral_unsolved_at_rung1():
    # rung-1 scope is differentiation only (spec): no rule fires on
    # Integral, and doit is a verifier now, not a move. Honest miss.
    r = beam_search(sp.Integral(3 * x**2 + 2 * x, x))
    assert not r.solved


def test_search_is_not_degenerate():
    """Rung-0 regression: doit solved everything in ~1 ply. Rung-1
    derivations must be genuine multi-ply descents."""
    r = beam_search(sp.Derivative(x**3 + sp.sin(x), x))
    assert r.solved
    assert r.state.plies > 1
    assert not any("doit" in h or h == "simplify" for h in r.state.history)


def test_history_is_a_legible_step_chain():
    r = beam_search(sp.Derivative(x**2 * sp.sin(x), x))
    assert r.solved
    rule_steps = [h for h in r.state.history if "@" in h]
    assert rule_steps, "expected at least one rule@node entry"
    names = {h.split("@")[0] for h in rule_steps}
    assert names <= {
        "d_const", "d_x", "d_sum", "d_product", "d_power",
        "d_chain_table", "d_quotient",
    }


def test_beam_matches_sympy_on_mathgen_set():
    import random

    from llmopt.mathgen.problems import _expression

    rng = random.Random("rung1-e2e-2-0")
    for _ in range(10):
        f = _expression(rng, 2)
        r = beam_search(sp.Derivative(f, x), max_plies=20)
        assert r.solved, f
        assert sp.simplify(r.state.expr - sp.diff(f, x)) == 0, f


def test_macros_off_by_default():
    s = State(sp.Derivative(sp.sin(x) / (x**2 + 1), x))
    assert not any("d_quotient" in name for name, _ in successors(s))
    assert any(
        "d_quotient" in name for name, _ in successors(s, use_macros=True)
    )


def test_max_nodes_budget():
    r = beam_search(sp.Derivative(x**3 + sp.sin(x), x), max_nodes=2)
    assert r.nodes <= 2


def test_trace_collects_equivalent_states():
    trace = []
    root = sp.Derivative(x**2 * sp.sin(x), x)
    r = beam_search(root, trace=trace)
    assert r.solved
    assert len(trace) >= r.state.plies  # at least the winning path
    for s in trace:
        assert sp.simplify(s.expr.doit() - root.doit()) == 0


def test_beam_records_history():
    r = beam_search(sp.Derivative(x**2, x))
    assert r.solved
    assert len(r.state.history) == r.state.plies >= 1

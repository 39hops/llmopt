"""The packaged engine: measured-best config as one import."""

import sympy as sp

from llmopt.search.engine import MarkovPrior, solve

x = sp.Symbol("x")


def test_solve_diff_int_limit_and_ceiling():
    for root, truth in [
        (sp.Derivative(x**2 * sp.sin(x), x), sp.diff(x**2 * sp.sin(x), x)),
        (sp.Integral(x * sp.cos(x), x), None),
        (sp.Limit(sp.sin(x) / x, x, 0), sp.Integer(1)),
        (sp.Integral(sp.sin(x) ** 2, x), None),  # the moved ceiling
    ]:
        r = solve(root, budget=300)
        assert r.solved, root
        if truth is not None:
            assert sp.simplify(r.state.expr - truth) == 0
        elif isinstance(root, sp.Integral):
            assert sp.simplify(
                sp.diff(r.state.expr, x) - root.function) == 0


def test_prior_roundtrip():
    p = MarkovPrior.load()
    assert p.unigram and p.bigram
    rows = [{"state": "s", "moves": ["a@1", "b@2"], "answer": 0},
            {"state": "t", "moves": ["b@2", "a@1"], "answer": 0}]
    q = MarkovPrior.from_rows(rows)
    assert q.bigram["a"]["b"] == 1


def test_magic_pruning_in_engine():
    """engine.solve prunes Risch-certified dead states by default and
    still solves the standard cases; the detector itself certifies
    the canonical non-elementary integrand."""
    import sympy as sp

    from llmopt.search.derivation import State
    from llmopt.search.engine import solve
    from llmopt.search.magic import is_dead

    x = sp.Symbol("x")
    assert is_dead(State(sp.Integral(sp.exp(x**2), x)))
    assert not is_dead(State(sp.Integral(sp.exp(x), x)))
    r = solve(sp.Integral(x * sp.cos(x), x), budget=150)
    assert r.solved
    r2 = solve(sp.Integral(x * sp.cos(x), x), budget=150, magic=False)
    assert r2.solved

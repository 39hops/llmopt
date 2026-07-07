"""Integration-rule property tests. Antiderivatives are equivalence
classes modulo constants, so equivalence is checked by differentiating
the difference, not by exact equality."""

import random

import pytest
import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.rules import INT_RULES, U

x = sp.Symbol("x")
RULES = dict(INT_RULES)


def _equiv_mod_const(node: sp.Integral, rewrite: sp.Expr) -> bool:
    d = node.doit() - rewrite.doit()
    return sp.simplify(sp.diff(d, x)) == 0


def test_i_const():
    assert RULES["i_const"](sp.Integral(sp.Integer(5), x)) == [5 * x]
    assert RULES["i_const"](sp.Integral(x, x)) == []


def test_i_power():
    (rw,) = RULES["i_power"](sp.Integral(x**3, x))
    assert sp.simplify(rw - x**4 / 4) == 0
    (rw,) = RULES["i_power"](sp.Integral(1 / x, x))
    assert rw == sp.log(x)
    (rw,) = RULES["i_power"](sp.Integral(x, x))
    assert sp.simplify(rw - x**2 / 2) == 0


def test_i_sum_stays_unevaluated():
    node = sp.Integral(x + sp.sin(x), x)
    (rw,) = RULES["i_sum"](node)
    assert rw.has(sp.Integral)
    assert _equiv_mod_const(node, rw)


def test_i_const_factor():
    node = sp.Integral(3 * sp.sin(x), x)
    (rw,) = RULES["i_const_factor"](node)
    assert rw.has(sp.Integral(sp.sin(x), x))
    assert _equiv_mod_const(node, rw)
    assert RULES["i_const_factor"](sp.Integral(x * sp.sin(x), x)) == []


def test_i_table():
    for f, F in [(sp.sin(x), -sp.cos(x)), (sp.cos(x), sp.sin(x)),
                 (sp.exp(x), sp.exp(x))]:
        assert RULES["i_table"](sp.Integral(f, x)) == [F]
    assert RULES["i_table"](sp.Integral(sp.sin(2 * x), x)) == []


def test_i_usub_fires_on_composition():
    node = sp.Integral(2 * x * sp.cos(x**2), x)
    rewrites = RULES["i_usub"](node)
    assert any(
        isinstance(rw, sp.Subs) and rw.has(sp.Integral(sp.cos(U), U))
        for rw in rewrites
    ), rewrites
    for rw in rewrites:
        assert _equiv_mod_const(node, rw)


def test_i_usub_no_candidate_no_fire():
    assert RULES["i_usub"](sp.Integral(x**2, x)) == []


def test_i_parts_branches_and_stays_stepwise():
    node = sp.Integral(x * sp.cos(x), x)
    rewrites = RULES["i_parts"](node)
    assert rewrites, "by-parts should fire on x*cos(x)"
    for rw in rewrites:
        assert rw.has(sp.Integral)
        assert _equiv_mod_const(node, rw)


def test_rules_ignore_definite_and_multilimit():
    for name, rule in INT_RULES:
        assert rule(sp.Integral(x, (x, 0, 1))) == [], name
        assert rule(sp.Integral(sp.cos(x), x, x)) == [], name


@pytest.mark.parametrize("level", [1, 2, 3])
def test_property_on_generated_integrands(level):
    rng = random.Random(f"int-rules-prop-{level}-0")
    for _ in range(15):
        f = sp.simplify(sp.diff(_expression(rng, level), x))
        if f == 0:
            continue
        node = sp.Integral(f, x)
        for name, rule in INT_RULES:
            for rw in rule(node):
                assert _equiv_mod_const(node, rw), f"{name} broke on {f}"


def test_euler_rewrite_moves_the_ceiling():
    """sin^2 has NO derivation in the real-form rule set (pre-registered
    ceiling); the euler move opens an all-existing-rules chain."""
    from llmopt.search.derivation import beam_search

    node = sp.Integral(sp.sin(x) ** 2, x)
    r = beam_search(node, max_plies=24, max_nodes=300)
    assert r.solved, "ceiling did not move"
    assert sp.simplify(sp.diff(r.state.expr, x) - sp.sin(x) ** 2) == 0
    assert any(h == "euler" for h in r.state.history)


def test_cyclic_moves_the_third_ceiling():
    """exp(x)*sin(x) had NO derivation pre-i_cyclic (2026-07-07
    autopsy top family): by-parts twice returns the original integral
    and the winning step is algebra on the equation, outside the
    rewrite space. The table macro emits the solved form directly."""
    from llmopt.search.engine import solve

    for f in (sp.exp(x) * sp.sin(x + sp.pi / 4),
              sp.exp(2 * x) * sp.cos(3 * x - 1)):
        assert len(RULES["i_cyclic"](sp.Integral(f, x))) == 1
        got = RULES["i_cyclic"](sp.Integral(f, x))[0]
        assert sp.simplify(sp.diff(got, x) - f) == 0
    # non-matches stay silent
    assert RULES["i_cyclic"](sp.Integral(sp.exp(x**2) * sp.sin(x), x)) == []
    assert RULES["i_cyclic"](sp.Integral(sp.sin(x) * sp.cos(x), x)) == []

    node = sp.Integral(27 * sp.sqrt(2) * sp.exp(x) * sp.sin(x + sp.pi / 4), x)
    r = solve(node, budget=100)
    assert r.solved, "third ceiling did not move"
    movers = ("i_cyclic@", "i_unprod@", "i_linear_basis@")
    assert any(h.startswith(movers) for h in r.state.history)


def test_unprod_reverse_product_rule():
    """Expanded d/dx[f*G(u)] sums (dominant L4 autopsy family): no
    single Mul node holds the product-rule pair, so i_parts never
    sees it. Autopsy failure #3, now one ply via engine.solve."""
    from llmopt.search.engine import solve

    g = (162 * x * sp.cos(3 * x**3 + 3 * x + 2)
         - 81 * (3 * x**2 + 1) ** 2 * sp.sin(3 * x**3 + 3 * x + 2))
    out = RULES["i_unprod"](sp.Integral(g, x))
    assert any(sp.simplify(sp.diff(o, x) - g) == 0 for o in out)
    r = solve(sp.Integral(g, x), budget=200)
    assert r.solved
    assert any(h.startswith(("i_unprod@", "i_linear_basis@"))
               for h in r.state.history)


def test_ansatz_exp_undetermined_coefficients():
    """P(x)*exp(w) with f*w' spanning several expanded terms: per-term
    cofactor guessing can't reassemble it; solve Q'+Q*w'=P instead.
    Autopsy failure #19. Also guards the exp(w1+w2) auto-split."""
    from llmopt.search.engine import solve

    g = sp.expand((72 * x * (2 * x + 1) + 24) * sp.exp(3 * x**2 + 3 * x + 1))
    out = RULES["i_ansatz_exp"](sp.Integral(g, x))
    assert len(out) == 1 and sp.simplify(sp.diff(out[0], x) - g) == 0
    # non-elementary: no polynomial Q exists, rule must stay silent
    assert RULES["i_ansatz_exp"](sp.Integral(sp.exp(x**2), x)) == []
    r = solve(sp.Integral(g, x), budget=200)
    assert r.solved
    assert any(h.startswith(("i_ansatz_exp@", "i_linear_basis@"))
               for h in r.state.history)


def test_linear_basis_bidirectional_v0():
    """Bidirectional search collapsed into linear algebra: d/dx is
    linear, so meet-in-the-middle over answer shapes = one matrix
    solve. Must subsume the cyclic/unprod/ansatz families AND reach
    mixed exp*trig products none of them can; must stay silent on
    non-elementary integrands."""
    solvable = [
        sp.exp(x) * sp.sin(x),
        sp.expand(sp.diff((x**2 + 1) * sp.exp(2 * x) * sp.sin(3 * x), x)),
        sp.expand(sp.diff((2 * x - 1) * sp.sin(4 * x**2 + 3 * x + 5), x)),
        sp.expand(sp.diff(x**3 * sp.exp(2 * x) * sp.cos(5 * x) + x**2, x)),
    ]
    for g in solvable:
        out = RULES["i_linear_basis"](sp.Integral(g, x))
        assert out and sp.simplify(sp.diff(out[0], x) - g) == 0, g
    assert RULES["i_linear_basis"](sp.Integral(sp.exp(x**2), x)) == []


def test_markov_prior_smoothing_reaches_unseen_rules():
    """Rules absent from the mined prior must not score 0 (they'd be
    guillotined by the top-3 cut until the prior is re-mined) —
    unseen rules get the median unigram mass."""
    from llmopt.search.engine import MarkovPrior

    prior = MarkovPrior({"i_sum": 100, "i_power": 50, "expand": 10}, {})
    prop = prior.proposer()
    from llmopt.search.derivation import State
    s = State(sp.Integral(sp.sin(x), x))
    kids = [("i_sum@node", s), ("brand_new_rule@node", s),
            ("expand@node", s)]
    ranked = prop(s, kids)
    names = [k[0] for k in ranked]
    # median (50) puts the unseen rule ahead of the rare seen one
    assert names.index("brand_new_rule@node") < names.index("expand@node")


def test_apart_moves_the_second_ceiling():
    """1/(x**2-1) had NO derivation pre-i_apart (measured); the move
    opens i_sum -> i_const_factor -> i_usub -> i_power(log) chains."""
    from llmopt.search.engine import solve

    node = sp.Integral(1 / (x**2 - 1), x)
    r = solve(node, budget=300)
    assert r.solved, "second ceiling did not move"
    assert sp.simplify(sp.diff(r.state.expr, x) - 1 / (x**2 - 1)) == 0
    assert any(h.startswith("i_apart@") for h in r.state.history)

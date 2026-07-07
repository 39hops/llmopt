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


def test_apart_moves_the_second_ceiling():
    """1/(x**2-1) had NO derivation pre-i_apart (measured); the move
    opens i_sum -> i_const_factor -> i_usub -> i_power(log) chains."""
    from llmopt.search.engine import solve

    node = sp.Integral(1 / (x**2 - 1), x)
    r = solve(node, budget=300)
    assert r.solved, "second ceiling did not move"
    assert sp.simplify(sp.diff(r.state.expr, x) - 1 / (x**2 - 1)) == 0
    assert any(h.startswith("i_apart@") for h in r.state.history)

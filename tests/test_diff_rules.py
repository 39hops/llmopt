"""Per-rule property tests: every rewrite a rule emits must be
sympy-equivalent to the Derivative node it replaces."""

import random

import pytest
import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.rules import CORE_RULES, MACRO_RULES

x = sp.Symbol("x")
RULES = dict(CORE_RULES + MACRO_RULES)


def _equiv(node: sp.Derivative, rewrite: sp.Expr) -> bool:
    return sp.simplify(node.doit() - rewrite.doit()) == 0


def test_d_const():
    assert RULES["d_const"](sp.Derivative(sp.Integer(7), x)) == [0]
    assert RULES["d_const"](sp.Derivative(sp.pi, x)) == [0]
    assert RULES["d_const"](sp.Derivative(x**2, x)) == []


def test_d_x():
    assert RULES["d_x"](sp.Derivative(x, x)) == [1]
    assert RULES["d_x"](sp.Derivative(x**2, x)) == []


def test_d_sum_emits_unevaluated_terms():
    node = sp.Derivative(x**2 + sp.sin(x), x)
    (rw,) = RULES["d_sum"](node)
    # linearity costs a visible ply: children stay unevaluated
    assert rw.has(sp.Derivative)
    assert _equiv(node, rw)


def test_d_product_branches_over_splits():
    node = sp.Derivative(x**2 * sp.sin(x) * sp.exp(x), x)
    rewrites = RULES["d_product"](node)
    assert len(rewrites) == 3  # one (head, rest) split per factor
    for rw in rewrites:
        assert rw.has(sp.Derivative)
        assert _equiv(node, rw)


def test_d_power_builds_chain():
    node = sp.Derivative((x**2 + 1) ** 3, x)
    (rw,) = RULES["d_power"](node)
    assert rw.has(sp.Derivative(x**2 + 1, x))
    assert _equiv(node, rw)


def test_d_power_rejects_x_in_exponent():
    assert RULES["d_power"](sp.Derivative(x**x, x)) == []


def test_d_chain_table():
    node = sp.Derivative(sp.sin(x**2), x)
    (rw,) = RULES["d_chain_table"](node)
    assert rw.has(sp.Derivative(x**2, x))
    assert _equiv(node, rw)


def test_d_quotient_macro():
    node = sp.Derivative(sp.sin(x) / (x**2 + 1), x)
    rewrites = RULES["d_quotient"](node)
    assert rewrites, "quotient macro should fire on u/v"
    for rw in rewrites:
        assert _equiv(node, rw)
    # no x in the denominator: macro must not fire
    assert RULES["d_quotient"](sp.Derivative(sp.sin(x) / 3, x)) == []


def test_d_const_factor_macro():
    node = sp.Derivative(6 * x**2 * sp.sin(x), x)
    (rw,) = RULES["d_const_factor"](node)
    assert rw == 6 * sp.Derivative(x**2 * sp.sin(x), x)
    assert _equiv(node, rw)
    assert RULES["d_const_factor"](sp.Derivative(x * sp.sin(x), x)) == []
    assert RULES["d_const_factor"](sp.Derivative(sp.Integer(6), x)) == []


@pytest.mark.parametrize("level", [1, 2, 3])
def test_property_all_rules_equivalent_on_generated_exprs(level):
    """Every rule, applied to Derivatives of mathgen-generated
    expressions and their subexpressions, only emits equivalent
    rewrites. String seeds per repo convention."""
    rng = random.Random(f"rules-prop-{level}-0")
    for _ in range(20):
        f = _expression(rng, level)
        node = sp.Derivative(f, x)
        for name, rule in CORE_RULES + MACRO_RULES:
            for rw in rule(node):
                assert _equiv(node, rw), f"{name} broke on {f}"


def test_rules_ignore_higher_order_and_multivar():
    y = sp.Symbol("y")
    for name, rule in CORE_RULES + MACRO_RULES:
        assert rule(sp.Derivative(x**3, x, 2)) == [], name
        assert rule(sp.Derivative(x * y, x, y)) == [], name

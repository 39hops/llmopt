"""mathgen level 4: validity + the collision-rate guard (repo scar
tissue: widen the generator space before trusting any split)."""

import random

import sympy as sp

from llmopt.mathgen.problems import _expression

X = sp.Symbol("x")


def test_l4_valid_and_differentiable():
    rng = random.Random("l4-valid-0")
    for _ in range(200):
        f = _expression(rng, 4)
        assert f.has(X), f
        d = sp.diff(f, X)
        assert d is not None


def test_l4_deeper_than_l3():
    """L4 must actually add structure: mean count_ops well above L3."""
    r3, r4 = random.Random("l4-cmp-3"), random.Random("l4-cmp-4")
    ops3 = sum(sp.count_ops(_expression(r3, 3)) for _ in range(100)) / 100
    ops4 = sum(sp.count_ops(_expression(r4, 4)) for _ in range(100)) / 100
    assert ops4 > ops3 * 1.5, (ops3, ops4)


def test_l4_collision_guard():
    """<1% duplicate sreprs over 1000 draws (contamination guard)."""
    rng = random.Random("l4-collide-0")
    seen = [sp.srepr(_expression(rng, 4)) for _ in range(1000)]
    dupes = len(seen) - len(set(seen))
    assert dupes < 10, f"collision rate {dupes / 10:.1f}% >= 1%"

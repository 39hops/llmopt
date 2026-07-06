"""pmap: order preservation, serial bypass, serial==parallel results."""

import sympy as sp

from llmopt.search.parallel import pmap

x = sp.Symbol("x")


def _square(n: int) -> int:
    return n * n


def _solve(seed: int):
    # a real sympy workload so fork-context sharing is exercised
    from llmopt.search.derivation import beam_search

    f = x ** (seed % 3 + 2) + seed * x
    r = beam_search(sp.Derivative(f, x), max_nodes=100)
    return (r.solved, sp.srepr(r.state.expr))


def test_order_preserved():
    assert pmap(_square, list(range(20)), jobs=4) == [n * n for n in range(20)]


def test_serial_bypass_identical():
    items = list(range(6))
    assert pmap(_solve, items, jobs=1) == pmap(_solve, items, jobs=3)

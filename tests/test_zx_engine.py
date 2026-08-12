"""Oracle-verified tests for llmopt.search.zx_engine (spec Phase 6.1).

The engine backs booked verdicts (ZX rungs 5-6) and had no tests.
Everything here scores by RUNNING the graphs against the boundary
oracle (tensor equality via verify_equal), never by structure match.
"""
import pytest

zx = pytest.importorskip("pyzx")

from pyzx.circuit import Circuit  # noqa: E402

from llmopt.search.zx_engine import (  # noqa: E402
    ZXState, best_first_zx, moves, tcount, verify_equal,
)


def _known_circuit():
    """3 qubits, exactly 2 T gates: known tcount, nontrivial graph."""
    c = Circuit(3)
    c.add_gate("HAD", 0)
    c.add_gate("CNOT", 0, 1)
    c.add_gate("T", 1)
    c.add_gate("CNOT", 1, 2)
    c.add_gate("T", 2)
    c.add_gate("HAD", 2)
    return c


def test_tcount_known_circuit():
    g = _known_circuit().to_graph()
    assert tcount(g) == 2


def test_best_first_reduces_and_verifies():
    c = _known_circuit()
    g = c.to_graph()
    t0 = tcount(g)
    best, nodes = best_first_zx(g.copy(), budget=100)
    assert nodes >= 1
    assert tcount(best.g) <= t0
    assert verify_equal(c, best.g, qubits=3)


def test_moves_preserve_equality():
    c = _known_circuit()
    g = c.to_graph()
    checked = 0
    for _label, child in moves(ZXState(g)):
        assert verify_equal(c, child.g, qubits=3)
        checked += 1
        if checked >= 3:
            break
    assert checked >= 1, "no moves fired on a fusable circuit"

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from expert_loop import gate_verdict


def _sb(solves, validity):
    return {"solves": solves, "validity": validity}


def test_gate_promotes_on_frontier_gain():
    prev = _sb({2: 30, 3: 20}, 10.0)
    new = _sb({2: 30, 3: 24}, 10.0)
    ok, reason = gate_verdict(prev, new, frontier=3)
    assert ok


def test_gate_promotes_on_validity_gain_alone():
    prev = _sb({2: 30, 3: 20}, 10.0)
    new = _sb({2: 30, 3: 20}, 12.5)
    ok, _ = gate_verdict(prev, new, frontier=3)
    assert ok


def test_gate_rejects_regression_even_with_frontier_gain():
    prev = _sb({2: 30, 3: 20}, 10.0)
    new = _sb({2: 27, 3: 25}, 15.0)   # L2 lost 3 > 2
    ok, reason = gate_verdict(prev, new, frontier=3)
    assert not ok and "regress" in reason


def test_gate_rejects_no_improvement():
    prev = _sb({2: 30, 3: 20}, 10.0)
    new = _sb({2: 30, 3: 20}, 11.0)   # validity +1 < 2 points
    ok, _ = gate_verdict(prev, new, frontier=3)
    assert not ok

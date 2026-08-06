"""TENET D3 budget accountant — the R8-defect regression tests.

The property under test is the one R8 violated: no sequence of
charges can push spent past total, and every refusal is visible.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scratch"))
from tenet_d3_budget import BudgetAccountant, charge  # noqa: E402


def test_hard_ceiling_never_exceeded():
    acct = BudgetAccountant(total=50)
    for _ in range(100):
        acct.debit(7, "arm")
    assert acct.spent <= 50
    assert acct.spent == 49  # 7 fits 7 times
    assert acct.refusals["arm"] == 93


def test_refusal_charges_nothing():
    acct = BudgetAccountant(total=10)
    assert acct.debit(8, "a")
    assert not acct.debit(3, "b")
    assert acct.spent == 8
    assert "b" not in acct.ledger


def test_census_ledger_sums():
    acct = BudgetAccountant(total=100)
    acct.debit(30, "fwd")
    acct.debit(20, "peel")
    c = acct.census()
    assert c["spent"] == 50 == sum(c["ledger"].values())
    assert c["remaining"] == 50


def test_charge_wave_refused_returns_empty():
    acct = BudgetAccountant(total=10)
    assert charge(acct, "w", ["abc"], tok_counts=[4]) == ["abc"]
    assert charge(acct, "w", ["defgh"], tok_counts=[7]) == []
    assert acct.spent == 4


def test_r8_shape_peel_cannot_ride_on_top():
    """The R8 anatomy: peeler spent 5.7x the forward budget because
    it was never debited. Under one accountant, the same sequence
    hits the ceiling and the peeler starves instead."""
    acct = BudgetAccountant(total=27_053 * 2)  # A's spend, both arms
    assert acct.debit(27_053, "forward")
    assert not acct.debit(154_641, "peeler")  # the R8 overdraft
    assert acct.spent == 27_053
    assert acct.refusals == {"peeler": 1}


def test_negative_and_zero_guards():
    with pytest.raises(ValueError):
        BudgetAccountant(total=0)
    acct = BudgetAccountant(total=5)
    with pytest.raises(ValueError):
        acct.debit(-1, "x")

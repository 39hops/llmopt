"""CONTROL-ADEQUATE precondition of FROZEN-BACKBONE-1 (AMENDMENT
FROZEN-BACKBONE-1-CONTROL-ADEQUATE): the pure law in scratch/fb_gate.py
adjudicate(). A FULL control below the house floor (24) makes the rung
NOT-RESOLVABLE-CONTROL; in particular FULL = 0 / FROZEN = 0 on every seed
(delta 0 >= -7 arithmetically) must never score REPLICATES. With adequate
controls the sealed law is unchanged."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adjudicate():
    for p in (ROOT, ROOT / "scripts", ROOT / "scratch"):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("fb_gate", ROOT / "scratch" / "fb_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.FLOOR == 24 and mod.LAW == -7 and mod.SEEDS == (24, 25, 26)
    return mod.adjudicate


def test_zero_zero_cannot_replicate(adjudicate):
    r = adjudicate({24: (0, 0), 25: (0, 0), 26: (0, 0)})
    assert r["deltas"] == [0, 0, 0]
    assert r["CONTROL_ADEQUATE"] is False
    assert r["full_below_floor"] == [24, 25, 26]
    assert r["verdict"] == "NOT-RESOLVABLE-CONTROL"
    assert r["REPLICATES"] is None


def test_one_full_below_floor_is_not_resolvable(adjudicate):
    r = adjudicate({24: (62, 60), 25: (23, 60), 26: (61, 59)})
    assert r["verdict"] == "NOT-RESOLVABLE-CONTROL" and r["REPLICATES"] is None
    assert r["full_below_floor"] == [25]


def test_floor_is_inclusive(adjudicate):
    r = adjudicate({24: (24, 17), 25: (24, 24), 26: (24, 30)})
    assert r["CONTROL_ADEQUATE"] is True and r["verdict"] == "REPLICATES"


def test_sealed_law_unchanged_when_adequate(adjudicate):
    assert adjudicate({24: (62, 59), 25: (60, 56), 26: (64, 57)})["verdict"] == "REPLICATES"
    k = adjudicate({24: (62, 59), 25: (60, 52), 26: (64, 57)})
    assert k["verdict"] == "KILLED" and k["REPLICATES"] is False and k["deltas"] == [-3, -8, -7]
    assert k["mean_delta_descriptive"] == -6


def test_frozen_zero_with_adequate_control_is_killed(adjudicate):
    assert adjudicate({24: (62, 0), 25: (60, 0), 26: (64, 0)})["verdict"] == "KILLED"

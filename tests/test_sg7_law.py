"""SG-BOUNDARY-BLOCK7-1 qualification law (scratch/sg7_qualgate.adjudicate):
control first; CONTROL-ADEQUATE iff finite and c >= 24 (else
NOT-RESOLVABLE-CONTROL, the cell never adjudicated); the one SG7 cell is
FUNCTION-MATCH iff finite and c - 7 <= g <= c + 7 (inclusive), SG7-MISS
otherwise, SG7-UNSTABLE when it did not finish finite; in-progress states
name the next arm."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adj():
    for p in (ROOT, ROOT / "scripts", ROOT / "scratch"):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("sg7_qualgate", ROOT / "scratch" / "sg7_qualgate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.FLOOR == 24 and m.BAND == 7 and m.SEED == 28
    return m.adjudicate


def test_control_first(adj):
    st = adj(None, None)
    assert st["verdict"] is None and not st["stop"] and st["next"] == "control"


def test_inadequate_control(adj):
    for c in ({"finite": True, "gate": 23}, {"finite": False, "gate": None}):
        st = adj(c, {"finite": True, "gate": 23})
        assert st["verdict"] == "NOT-RESOLVABLE-CONTROL" and st["stop"] and st["cell"] is None and not st["control_adequate"]


def test_floor_and_band_inclusive(adj):
    st = adj({"finite": True, "gate": 24}, None)
    assert st["control_adequate"] and st["verdict"] is None and st["next"] == "sg7" and not st["stop"]
    for g, v in ((17, "FUNCTION-MATCH"), (31, "FUNCTION-MATCH"), (16, "SG7-MISS"), (32, "SG7-MISS"), (0, "SG7-MISS")):
        st = adj({"finite": True, "gate": 24}, {"finite": True, "gate": g})
        assert st["verdict"] == v and st["stop"], (g, v)
        assert st["cell"]["delta_v_control"] == g - 24 and st["cell"]["function_match"] == (v == "FUNCTION-MATCH")


def test_unstable_cell(adj):
    st = adj({"finite": True, "gate": 59}, {"finite": False, "gate": None})
    assert st["verdict"] == "SG7-UNSTABLE" and st["stop"] and st["cell"]["gate"] is None and not st["cell"]["stable"]

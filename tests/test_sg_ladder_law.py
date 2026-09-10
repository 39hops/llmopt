"""SYNTHETIC-GRADIENT-WRITER-1 ladder law (scratch/sg_qualgate.py
adjudicate(), pure): adequate control (finite, c >= 24) precedes any SG
adjudication; the frozen order LINEAR 3e-4, LINEAR 3e-5, MLP-256 3e-4,
MLP-256 3e-5; FIRST FUNCTION-MATCH (c - 7 <= g <= c + 7) stops the ladder
and is the selection regardless of later (unborn) cells; no match after
all four = ACCESSIBILITY-ONLY; a control at 0 / 0 cannot yield a match."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def adj():
    for p in (ROOT, ROOT / "scripts", ROOT / "scratch"):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("sg_qualgate", ROOT / "scratch" / "sg_qualgate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.FLOOR == 24 and m.BAND == 7 and m.SEED == 27
    assert m.LADDER == (("linear", 3e-4), ("linear", 3e-5), ("mlp256", 3e-4), ("mlp256", 3e-5))
    return m.adjudicate


def ok(g):
    return {"finite": True, "gate": g}


def test_control_first(adj):
    st = adj(None, {})
    assert st["verdict"] is None and st["stop"] is False and st["next"] == "control"


def test_inadequate_control_blocks_everything(adj):
    for ctrl in ({"finite": False, "gate": None}, ok(23), ok(0)):
        st = adj(ctrl, {("linear", 3e-4): ok(60)})
        assert st["verdict"] == "NOT-RESOLVABLE-CONTROL" and st["stop"] is True and st["selected"] is None
    # FULL 0 / SG 0 cannot match
    st = adj(ok(0), {("linear", 3e-4): ok(0)})
    assert st["verdict"] == "NOT-RESOLVABLE-CONTROL"


def test_floor_inclusive_and_band_inclusive(adj):
    st = adj(ok(24), {("linear", 3e-4): ok(17)})
    assert st["control_adequate"] is True and st["verdict"] == "FUNCTION-MATCH"
    st = adj(ok(60), {("linear", 3e-4): ok(67)})
    assert st["verdict"] == "FUNCTION-MATCH"
    st = adj(ok(60), {("linear", 3e-4): ok(68)})
    assert st["verdict"] is None and st["next"] == "linear:3e-05"


def test_first_match_stops_in_frozen_order(adj):
    st = adj(ok(62), {("linear", 3e-4): ok(40), ("linear", 3e-5): ok(58)})
    assert st["verdict"] == "FUNCTION-MATCH" and st["selected"]["family"] == "linear" and st["selected"]["plr"] == 3e-5
    assert st["stop"] is True and "mlp256:0.0003" not in st["cells"]
    # a later, higher cell never overrides an earlier match
    st = adj(ok(62), {("linear", 3e-4): ok(56), ("linear", 3e-5): ok(64), ("mlp256", 3e-4): ok(66)})
    assert st["selected"]["plr"] == 3e-4 and st["cells"].get("mlp256:0.0003") is None


def test_in_progress_and_accessibility_only(adj):
    st = adj(ok(62), {("linear", 3e-4): ok(30)})
    assert st["verdict"] is None and st["stop"] is False and st["next"] == "linear:3e-05"
    st = adj(ok(62), {("linear", 3e-4): ok(30), ("linear", 3e-5): {"finite": False, "gate": None}, ("mlp256", 3e-4): ok(0), ("mlp256", 3e-5): ok(50)})
    assert st["verdict"] == "ACCESSIBILITY-ONLY" and st["selected"] is None
    cells = st["cells"]
    assert cells["linear:0.0003"]["stable"] and cells["linear:0.0003"]["floor"] and not cells["linear:0.0003"]["function_match"]
    assert cells["linear:3e-05"]["finite"] is False and cells["mlp256:0.0003"]["stable"] is False
    assert cells["mlp256:3e-05"]["delta_v_control"] == -12


def test_all_unstable_is_ladder_unstable(adj):
    dead = {"finite": False, "gate": None}
    st = adj(ok(62), {("linear", 3e-4): dead, ("linear", 3e-5): dead, ("mlp256", 3e-4): dead, ("mlp256", 3e-5): dead})
    assert st["verdict"] == "LADDER-UNSTABLE" and st["stop"] is True and st["selected"] is None
    st = adj(ok(62), {("linear", 3e-4): dead, ("linear", 3e-5): dead, ("mlp256", 3e-4): dead, ("mlp256", 3e-5): ok(3)})
    assert st["verdict"] == "ACCESSIBILITY-ONLY"

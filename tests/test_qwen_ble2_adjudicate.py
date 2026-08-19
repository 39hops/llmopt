"""Fixtures for the BLE-FREEGEN-2 row gate + bars (committed before
the screen's receipts exist)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ba():
    spec = importlib.util.spec_from_file_location(
        "qwen_ble2_adjudicate",
        ROOT / "scratch" / "qwen_ble2_adjudicate.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen_ble2_adjudicate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _rows(n=30, term_x=0, correct_total=0):
    rows = []
    k = 0
    for cell in ("nothink", "xhigh"):
        for i in range(n):
            rows.append({"cell": cell, "id": i, "arm": "BLe",
                         "runtime": "qcuda_tower", "greedy": True,
                         "device_actual": "NVIDIA GeForce RTX 3080",
                         "think_terminated": (cell == "xhigh"
                                              and i < term_x),
                         "correct": k < correct_total,
                         "truncated": False})
            k += 1
    return rows


def test_gate_passes_clean(ba):
    assert ba.gate_rows(_rows()) == []


def test_gate_catches_count_dup_id_arm_runtime_device(ba):
    rows = _rows()
    assert ba.gate_rows(rows[:59])          # missing row
    d = _rows(); d[5] = dict(d[4])          # duplicate + id-set hole
    assert ba.gate_rows(d)
    a = _rows(); a[0]["arm"] = "B"
    assert any("arm" in v for v in ba.gate_rows(a))
    r = _rows(); r[0]["runtime"] = "rung4"
    assert any("runtime" in v for v in ba.gate_rows(r))
    g = _rows(); g[0]["greedy"] = False
    assert any("greedy" in v for v in ba.gate_rows(g))
    dv = _rows(); dv[0]["device_actual"] = "other"
    assert any("devices" in v for v in ba.gate_rows(dv))
    c = _rows(); c[0]["cell"] = "high"
    assert any("cell" in v for v in ba.gate_rows(c))


def test_bars_fire_semantics(ba):
    r = ba.adjudicate(_rows(term_x=0, correct_total=0))
    assert (r["bar1_termination"], r["bar2_competence"]) == \
        ("NO-FIRE", "NO-FIRE")
    r = ba.adjudicate(_rows(term_x=1, correct_total=0))
    assert r["bar1_termination"] == "FIRE" and r["xhigh_terminated"] == 1
    r = ba.adjudicate(_rows(term_x=0, correct_total=1))
    assert r["bar2_competence"] == "FIRE" and r["total_correct"] == 1


def test_counts_recomputed_from_rows_not_summary(ba):
    rows = _rows(term_x=3, correct_total=7)
    r = ba.adjudicate(rows)
    assert r["xhigh_terminated"] == 3
    assert r["total_correct"] == 7
    assert r["per_cell"]["nothink"]["correct"] == 7  # first 7 rows


def test_provenance_helper_shape():
    from llmopt.lab.provenance import start_provenance
    p = start_provenance(["llmopt/lab/provenance.py"])
    assert set(p) == {"start_commit", "start_status_porcelain",
                      "interpreter", "file_sha256"}
    assert len(p["file_sha256"]["llmopt/lab/provenance.py"]) == 64
    with pytest.raises(FileNotFoundError):
        start_provenance(["no/such/file.py"])

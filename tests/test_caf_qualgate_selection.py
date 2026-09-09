"""CREDIT-ANCHOR-FRONTIER-1 selector law (Artin fold 2026-09-09, before any
seed-23 birth): scratch/caf_qualgate.py writes qual_selection.json only when
a hybrid cell clears the floor (candidate = smallest such k, best cell) or
when every k is complete without one (FRONTIER-CLOSED); an incomplete
frontier exits 3 and creates no artifact; zero-credit controls never
enter the selection. Runs on synthetic receipt rows (no gate_eval)."""
import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scratch" / "caf_qualgate.py"


def _birth(cell, mode, k, s, lr):
    return {"kind": "birth", "phase": "qual", "cell": cell, "mode": mode, "k_bp": k, "s": s, "peak_lr": lr,
            "final": {"state_digest": "x"}, "outdir": "nowhere"}


def _gate(cell, mode, k, s, lr, total):
    return {"kind": "gate", "phase": "qual", "cell": cell, "mode": mode, "k_bp": k, "s": s, "peak_lr": lr, "trained": True,
            "total": total, "solves": {"3": total, "4": 0, "5": 0, "6": 0, "7": 0}}


def _rows_for(k, g1, g2, gz):
    return [_birth(f"h{k}a", "hybrid", k, 1.0, 3e-4), _gate(f"h{k}a", "hybrid", k, 1.0, 3e-4, g1),
            _birth(f"h{k}b", "hybrid", k, 1.0, 1e-4), _gate(f"h{k}b", "hybrid", k, 1.0, 1e-4, g2),
            _birth(f"z{k}", "zero", k, None, 3e-4), _gate(f"z{k}", "zero", k, None, 3e-4, gz)]


def _run(tmp_path, monkeypatch, rows):
    q = tmp_path / "qual.jsonl"
    q.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sel = tmp_path / "qual_selection.json"
    monkeypatch.setenv("GATE_ONLY", "0")
    monkeypatch.setenv("CAF_QUAL_PATH", str(q))
    monkeypatch.setenv("CAF_SEL_PATH", str(sel))
    sys.path.insert(0, str(ROOT / "scratch"))
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT))
    g = runpy.run_path(str(SCRIPT), run_name="not_main")
    rc = None
    try:
        g["main"]()
    except SystemExit as e:
        rc = e.code
    return rc, sel


def test_incomplete_frontier_writes_nothing(tmp_path, monkeypatch):
    rc, sel = _run(tmp_path, monkeypatch, _rows_for(1, 3, 0, 40))     # k=1 below floor, zero control clears, k=2/4 unborn
    assert rc == 3 and not sel.exists()


def test_first_k_clears_selects_and_ignores_zero(tmp_path, monkeypatch):
    rc, sel = _run(tmp_path, monkeypatch, _rows_for(1, 10, 2, 60) + _rows_for(2, 30, 35, 0))
    assert rc is None and sel.exists()
    d = json.loads(sel.read_text())
    assert d["selected"] == {"k": 2, "s": 1.0, "lr": 1e-4, "gate": 35, "cell": "h2b"} and d["frontier_closed"] is False
    assert d["per_k"]["1"]["zero_clears_floor"] is True and d["per_k"]["1"]["hybrid_clears_floor"] is False


def test_tie_prefers_lr_3e4(tmp_path, monkeypatch):
    rc, sel = _run(tmp_path, monkeypatch, _rows_for(1, 30, 30, 0))
    assert json.loads(sel.read_text())["selected"]["lr"] == 3e-4


def test_frontier_closed_only_when_all_three_complete(tmp_path, monkeypatch):
    rc, sel = _run(tmp_path, monkeypatch, _rows_for(1, 0, 0, 0) + _rows_for(2, 5, 1, 0))
    assert rc == 3 and not sel.exists()
    rc, sel = _run(tmp_path, monkeypatch, _rows_for(1, 0, 0, 0) + _rows_for(2, 5, 1, 0) + _rows_for(4, 23, 0, 50))
    d = json.loads(sel.read_text())
    assert d["selected"] is None and d["frontier_closed"] is True

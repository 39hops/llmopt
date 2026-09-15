"""FIRST-MOMENT-ERASURE-1 guards: the source pins match the sealed shared
sources byte for byte (a change in any pinned causal-path file turns this
red: amend / requalify before any treatment); the control pin verifies
exact state digests and refuses on missing or drifted snapshots; the
locked-receipt sha law; the single-writer label carries its fence and
never a two-writer clause; the sealed BAR-1 literal and thresholds; Z / E
touch every tensor (59 on the house model); the leg digest law and the
scheduler continuity reuse OMA's tested mechanics; refuse-if-exists on
the receipt path; no writer-B code path."""
import ast
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    prev = os.environ.get("SMOKE")
    os.environ["SMOKE"] = "1"
    for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scratch")):
        if p not in sys.path:
            sys.path.insert(0, p)
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        m = importlib.import_module("first_moment_erasure")
    finally:
        os.chdir(cwd)
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


def test_source_pins_match_disk(mod):
    bad = mod.check_pins(root=ROOT)
    assert bad == {}, f"pinned causal-path source changed since the seal: {list(bad)} -> amend / requalify before any treatment"
    assert set(mod.PINS) >= {"scratch/optimizer_memory_ablation.py", "scratch/optimizer_geometry_desk.py", "scratch/update_geometry_census.py",
                             "scratch/onecycle_component_audit.py", "scratch/birth19m_curric.py", "scratch/atomtraj_pins.py", "scripts/train_mathnative.py", "llmopt/train/mathnative.py"}


def test_pin_mismatch_is_detected(mod, tmp_path):
    (tmp_path / "x.py").write_text("print(1)\n")
    good = hashlib.sha256((tmp_path / "x.py").read_bytes()).hexdigest()
    assert mod.check_pins({"x.py": good}, root=tmp_path) == {}
    assert list(mod.check_pins({"x.py": "0" * 64}, root=tmp_path)) == ["x.py"]
    assert mod.check_pins({"missing.py": good}, root=tmp_path)["missing.py"][1] is None


def _fake_stage0(tmp_path, digests, drop=None, drift=None):
    from atomtraj_pins import state_digest
    snaps = {}
    for h, sd in digests.items():
        p = tmp_path / "ck" / "A" / "C" / f"h{h:04d}.pt"
        p.parent.mkdir(parents=True, exist_ok=True)
        if h != drop:
            torch.save({"model": sd if h != drift else {k: v + 1 for k, v in sd.items()}, "step": 7200 + h}, p)
        snaps[str(h)] = {"path": str(p.relative_to(tmp_path)), "state_digest": state_digest(sd),
                         "sha256": (hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() and h != drift else "0" * 64)}
    return {"cells": {"A": {"C": {"snapshots": snaps}}}}


def test_control_pin_verifies_and_refuses(mod, tmp_path):
    sds = {h: {"w": torch.full((3,), float(h))} for h in (1, 2, 3)}
    ok = mod.verify_control(_fake_stage0(tmp_path, sds), root=tmp_path, horizons=[1, 2, 3])
    assert mod.control_ok(ok) and all(x["file_digest"] == x["receipt_digest"] for x in ok.values())
    missing = mod.verify_control(_fake_stage0(tmp_path / "m", sds, drop=2), root=tmp_path / "m", horizons=[1, 2, 3])
    assert missing[2]["status"] == "MISSING" and not mod.control_ok(missing)
    drifted = mod.verify_control(_fake_stage0(tmp_path / "d", sds, drift=3), root=tmp_path / "d", horizons=[1, 2, 3])
    assert drifted[3]["status"] == "DRIFTED" and drifted[3]["file_digest"] != drifted[3]["receipt_digest"] and not mod.control_ok(drifted)


def test_real_receipt_provenance_law(mod, monkeypatch):
    """The registered-mode provenance block against the REAL locked receipts (structure cells.A.C.bind, OMA sha at Stage 0, the
    source literals v the lock, the sealed n_pred on both receipts, the thread / version law)."""
    monkeypatch.setattr(mod, "STAGE0", ROOT / "logs" / "oma1" / "stage0.json")
    monkeypatch.setattr(mod, "DESK", ROOT / "logs" / "oma1" / "desk_bar1.json")
    stage0 = json.loads((ROOT / "logs" / "oma1" / "stage0.json").read_text())
    desk = json.loads((ROOT / "logs" / "oma1" / "desk_bar1.json").read_text())
    lock = json.loads((ROOT / "docs" / "receipts.lock.json").read_text())
    lock["receipts"] = {str(mod.STAGE0): {"sha256": mod.STAGE0_SHA}, str(mod.DESK): {"sha256": mod.DESK_SHA}, **lock["receipts"]}
    out = mod.assert_provenance(stage0, desk, lock, mod.ANCHOR_SHA, 8, stage0["torch_version"], stage0["numpy_version"])
    assert out["stage0_lock_verified"] and out["oma_sha_at_stage0"] == stage0["source_sha256"] == desk["source_sha256"]
    with pytest.raises(AssertionError):
        mod.assert_provenance(stage0, desk, lock, "0" * 64, 8, stage0["torch_version"], stage0["numpy_version"])
    with pytest.raises(AssertionError):
        mod.assert_provenance(stage0, desk, lock, mod.ANCHOR_SHA, 4, stage0["torch_version"], stage0["numpy_version"])


def test_leg_path_symbol_sha_pinned(mod):
    assert mod.leg_path_sha() == mod.LEG_PATH_SHA
    assert "one_leg" in mod.LEG_PATH_SYMBOLS and "apply_arm" in mod.LEG_PATH_SYMBOLS and "resume_sched" in mod.LEG_PATH_SYMBOLS
    h = hashlib.sha256()
    for s in ["apply_arm"]:
        h.update("changed".encode())
    assert mod.leg_path_sha(symbols=["apply_arm"]) != mod.LEG_PATH_SHA


def test_locked_receipt_law(mod):
    lock = json.loads((ROOT / "docs" / "receipts.lock.json").read_text())
    for rel in ("logs/oma1/stage0.json", "logs/oma1/desk_bar1.json"):
        assert mod.locked_sha(lock, rel) == mod.sha256_file(ROOT / rel)      # the locked receipts this rung pins are intact on disk


def test_sealed_literals(mod):
    assert mod.N_PRED == 0.894284652673395 and mod.WRITER == "A"
    desk = json.loads((ROOT / "logs" / "oma1" / "desk_bar1.json").read_text())
    assert desk["cells"]["A"]["law"]["n_pred"] == mod.N_PRED
    assert mod.ANCHOR_SHA.startswith("a0cdf244fcf44f05")
    O = mod.OMA
    assert (O.FORGOTTEN, O.PERSISTENT, O.SENSITIVE, O.CE_FLOOR, O.CE_MULT, O.BAR1_TOL, O.BAR1_BAND, O.EPS_TWIN) == (0.05, 0.25, 0.10, 0.005, 3.0, 0.02, (0.80, 1.00), 0.01)
    assert mod.HORIZONS == [1, 2, 3] and mod.LEG == 3        # SMOKE constants under the fixture; real: [1, 5, 20, 100, 900] / 900


def test_single_writer_label_and_no_b_path(mod):
    adj = {"bar1": "PASS", "path": "FORGOTTEN", "sensitivity": "SPECIFIC", "function": "NEUTRAL"}
    lab = mod.single_writer_label(adj)
    assert lab.startswith("FORGOTTEN-SPECIFIC+FUNCTION-NEUTRAL") and "writer A only" in lab and "MIXED" not in lab
    assert mod.single_writer_label(dict(adj, bar1="INSTRUMENT-FAULT")) == "INSTRUMENT-FAULT"
    src = (ROOT / "scratch" / "first_moment_erasure.py").read_text()
    assert "program_label" not in src and "backsched" not in src and 'WRITER = "A"' in src      # substring tripwire
    # structural check: no subscript of OMA.WRITERS (or WRITERS) by a constant other than WRITER, and no string constant "B"
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "WRITERS":
            assert isinstance(node.slice, ast.Name) and node.slice.id == "WRITER", ast.dump(node)
        if isinstance(node, ast.Constant) and node.value == "B":
            raise AssertionError("string constant 'B' in the single-writer instrument")
    # an out-of-band n_Z(1) reaches INSTRUMENT-FAULT through OMA.adjudicate
    w = {"n_Z": {1: 0.70, 2: 0.3, 3: 0.04}, "n_E": {1: 0.009, 2: 0.003, 3: 0.001}, "dCE_Z": {1: 0, 2: 0, 3: 0}, "dCE_E": {1: 0, 2: 0, 3: 0}, "n_pred": mod.N_PRED, "h_end": 3}
    assert mod.single_writer_label(mod.OMA.adjudicate(w)) == "INSTRUMENT-FAULT"
    w["n_Z"][1] = 0.8943
    assert mod.single_writer_label(mod.OMA.adjudicate(w)).startswith("FORGOTTEN-SPECIFIC+FUNCTION-NEUTRAL")


def test_arms_touch_every_tensor(mod):
    from optimizer_memory_ablation import apply_arm
    m = mod.UG.build(mod.TM.MathTokenizer(), "cpu")
    o = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    x = torch.zeros(2, 8, dtype=torch.long); mk = torch.ones(2, 8, dtype=torch.long)
    m(x, mk).sum().backward(); o.step(); o.zero_grad()
    assert apply_arm(o, "E") == 59 and apply_arm(o, "Z") == 59 == len(mod.UG.KEYS)
    assert all(float(o.state[p]["exp_avg"].abs().sum()) == 0.0 and float(o.state[p]["exp_avg_sq"].abs().sum()) > 0.0 for p in o.param_groups[0]["params"])


def test_stream_and_scheduler_continuity_reuse_oma(mod):
    O = mod.OMA
    starts = [(i, i + 32) for i in range(0, 1000 - 32, 32)]
    a = O.leg_slices(starts, 1000, 7, 5); b = O.leg_slices(starts, 1000, 7, 5)
    assert hashlib.sha256(json.dumps(a).encode()).hexdigest() == hashlib.sha256(json.dumps(b).encode()).hexdigest()
    rows, _ = O.OA.table("stock")
    p = torch.nn.Parameter(torch.zeros(3)); opt = torch.optim.AdamW([p], lr=O.LR, weight_decay=O.WD)
    r = rows[7199]; opt.param_groups[0]["lr"] = r[1]; opt.param_groups[0]["betas"] = (r[2], r[3])
    _s, nxt = O.resume_sched("stock", opt, 7200, {"lr": r[1], "beta1": r[2], "beta2": r[3], "wd": r[4]})
    assert nxt["lr"] == rows[7200][1] and nxt["beta1"] == rows[7200][2]


def test_refuse_if_exists(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "RECEIPT", tmp_path / "treat.json")
    monkeypatch.setattr(mod, "STREAM", tmp_path / "treat.jsonl")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "CK_DIR", tmp_path / "ck")
    (tmp_path / "treat.json").write_text("{}")
    with pytest.raises(SystemExit):
        mod.main()

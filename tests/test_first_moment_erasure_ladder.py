"""FIRST-MOMENT-ERASURE-LADDER-1 guards: the source pins (the FME1 set plus
the FME1 instrument) match disk; the locked FME1 / Stage-0 receipt law and
the reference-digest law against the REAL locked receipts; BAR 0
qualification refuses a tampered digest; the eps law (e1 / e1e-2 are the
FME1 float operations, every tensor touched); the readouts with the
zero-norm law (never NaN); the preflight law; the regime order with the
two-sided band and REGIME-UNRESOLVED; the absolute function bar; the
locus; refuse-if-exists; no writer-B code path; the arena literals."""
import ast
import importlib
import json
import math
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
        m = importlib.import_module("first_moment_erasure_ladder")
    finally:
        os.chdir(cwd)
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


def test_source_pins_match_disk(mod):
    bad = mod.FME.check_pins(mod.PINS, root=ROOT)
    assert bad == {}, f"pinned causal-path source changed since the seal: {list(bad)} -> amend / requalify"
    assert set(mod.PINS) > set(mod.FME.PINS) and "scratch/first_moment_erasure.py" in mod.PINS
    assert mod.FME.leg_path_sha() == mod.FME.LEG_PATH_SHA


def test_locked_receipts_and_reference_digests(mod, monkeypatch):
    monkeypatch.chdir(ROOT)
    lock = json.loads((ROOT / "docs" / "receipts.lock.json").read_text())
    for rel, lit in (("logs/fme1/treat.json", mod.FME1_SHA), ("logs/oma1/stage0.json", mod.FME.STAGE0_SHA)):
        assert mod.FME.locked_sha(lock, rel) == lit == mod.sha256_file(ROOT / rel)
    stage0 = json.loads((ROOT / "logs" / "oma1" / "stage0.json").read_text())
    fme1 = json.loads((ROOT / "logs" / "fme1" / "treat.json").read_text())
    real_q = [1, 5, 20, 100, 900]
    refs = mod.reference_digests(stage0, fme1, horizons=real_q)
    assert set(refs) == {"C", "e1", "e1e-2"} and all(set(refs[a]) == set(real_q) for a in refs)
    assert all(len(v) == 64 for a in refs for v in refs[a].values())
    assert refs["e1"][1] == fme1["arms"]["Z"]["snapshots"]["1"]["state_digest"] and refs["C"][900] == stage0["cells"]["A"]["C"]["snapshots"]["900"]["state_digest"]
    # the FME1 provenance block against the real receipts (patched to the real paths under the SMOKE fixture)
    old = mod.FME1, mod.STAGE0
    mod.FME1, mod.STAGE0 = Path("logs/fme1/treat.json"), Path("logs/oma1/stage0.json")      # the lock is keyed by repo-relative paths
    try:
        out = mod.assert_fme1_provenance(fme1, lock, stage0, horizons=real_q)
        assert out["fme1_label"].startswith("PERSISTENT-SPECIFIC+FUNCTION-NEUTRAL")
        with pytest.raises(AssertionError):
            mod.assert_fme1_provenance(dict(fme1, n_pred_literal=0.5), lock, stage0, horizons=real_q)
    finally:
        mod.FME1, mod.STAGE0 = old


def test_qualification_refuses_tampered_digest(mod):
    refs = {"C": {1: "a" * 64, 5: "b" * 64}, "e1": {1: "c" * 64}}
    assert mod.qualify("C", 1, "a" * 64, refs) == (True, "a" * 64)
    assert mod.qualify("C", 5, "x" * 64, refs) == (False, "b" * 64)
    assert mod.qualify("e1e-1", 1, "a" * 64, refs) is None and mod.qualify("C", 20, "a" * 64, refs) is None


def test_apply_eps_is_the_fme1_operation_and_touches_every_tensor(mod):
    import copy
    m0 = mod.UG.build(mod.TM.MathTokenizer(), "cpu")
    x = torch.zeros(2, 8, dtype=torch.long); mk = torch.ones(2, 8, dtype=torch.long)

    def fresh():
        m = copy.deepcopy(m0)                       # identical model + optimizer state for every arm
        o = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
        m(x, mk).sum().backward(); o.step(); o.zero_grad()
        return o

    o = fresh(); ref = {id(p): o.state[p]["exp_avg"].clone() for p in o.param_groups[0]["params"]}
    assert mod.apply_eps(o, 0.0) == 0
    assert mod.apply_eps(o, 1e-3) == 59 == len(mod.UG.KEYS)
    for p in o.param_groups[0]["params"]:
        assert torch.equal(o.state[p]["exp_avg"], ref[id(p)] * (1.0 - 1e-3))
    o2 = fresh(); o3 = fresh()
    assert mod.apply_eps(o2, 0.01) == 59 and mod.OMA.apply_arm(o3, "E") == 59
    for p2, p3 in zip(o2.param_groups[0]["params"], o3.param_groups[0]["params"]):
        assert torch.equal(o2.state[p2]["exp_avg"], o3.state[p3]["exp_avg"])
    o4 = fresh(); assert mod.apply_eps(o4, 1.0) == 59
    assert all(float(o4.state[p]["exp_avg"].abs().sum()) == 0.0 and float(o4.state[p]["exp_avg_sq"].abs().sum()) > 0.0 for p in o4.param_groups[0]["params"])


def _linear_world(rng, d=50, n_target=None):
    W0 = rng.normal(size=d); WC = W0 + rng.normal(size=d)
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    scale = 1.0 if n_target is None else n_target * float(np.linalg.norm(WC - W0))
    W = {a: WC + e * scale * u for a, e in mod_arms().items()}
    return W0, WC, W


def mod_arms():
    return {"e1e-3": 1e-3, "e1e-2": 1e-2, "e1e-1": 1e-1, "e1": 1.0}


def test_horizon_metrics_linear_and_zero_norm_law(mod):
    rng = np.random.default_rng(0)
    W0, WC, W = _linear_world(rng)
    m = mod.horizon_metrics(W0, WC, W)
    assert abs(m["alpha"] - 1.0) < 1e-9 and m["cosmin"] > 1 - 1e-9 and all(abs(m["R"][a] - 1.0) < 1e-9 for a in W) and m["undefined"] == {}
    assert len(m["pair_cos"]) == 6
    # a dead arm: n = 0, R = 0, ln / cosine UNDEFINED, alpha and cosmin UNDEFINED, no NaN anywhere
    W["e1e-3"] = WC.copy()
    z = mod.horizon_metrics(W0, WC, W)
    assert z["n"]["e1e-3"] == 0.0 and z["R"]["e1e-3"] == 0.0 and z["ln"]["e1e-3"] is None and z["alpha"] is None and z["cosmin"] is None
    assert "alpha" in z["undefined"] and "cosmin" in z["undefined"]
    assert "nan" not in json.dumps(z).lower()
    # eps = 1 dead: R UNDEFINED for every arm
    W["e1"] = WC.copy()
    z2 = mod.horizon_metrics(W0, WC, W)
    assert all(z2["R"][a] is None for a in W) and z2["R"]["e1e-2"] is None
    assert mod.cosine(np.zeros(3), np.ones(3)) is None and abs(mod.cosine(np.ones(3), np.ones(3)) - 1) < 1e-12
    # alpha reads the slope: a sublinear ladder ||dW|| ~ eps^0.5
    W3 = {a: WC + math.sqrt(e) * np.ones(50) for a, e in mod_arms().items()}
    assert abs(mod.horizon_metrics(W0, WC, W3)["alpha"] - 0.5) < 1e-9


def test_preflight_law(mod):
    rng = np.random.default_rng(1)
    W0, WC, W = _linear_world(rng, n_target=0.8942846)
    m1 = mod.horizon_metrics(W0, WC, W)
    ok, det = mod.preflight_law(m1, n_pred=0.894284652673395)
    assert ok and det["fails"] == [] and all(det["cos_to_e1"][a] >= 0.999 for a in W)
    bad = dict(m1); bad["R"] = dict(m1["R"], **{"e1e-3": 1.01})
    assert not mod.preflight_law(bad, n_pred=0.894284652673395)[0]
    bad2 = dict(m1); bad2["pair_cos"] = dict(m1["pair_cos"], **{"e1e-3|e1": 0.9})
    assert not mod.preflight_law(bad2, n_pred=0.894284652673395)[0]
    assert not mod.preflight_law(m1, n_pred=0.5)[0]
    dead = dict(m1); dead["R"] = dict(m1["R"], **{"e1e-3": None})
    assert not mod.preflight_law(dead, n_pred=0.894284652673395)[0]


def _mH(alpha, cosmin, n1, n_all=None):
    n = {"e1e-3": 0.3, "e1e-2": 0.3, "e1e-1": 0.3, "e1": n1} if n_all is None else n_all
    return {"alpha": alpha, "cosmin": cosmin, "n": n}


def test_regime_order_two_sided_band_and_unresolved(mod):
    R = mod.regime
    assert R(_mH(1.0, 0.95, 0.5), True) == "MAGNITUDE-SCALED PERSISTENT"
    assert R(_mH(1.19, 0.95, 0.5), True) == "MAGNITUDE-SCALED PERSISTENT" and R(_mH(0.81, 0.95, 0.5), True) == "MAGNITUDE-SCALED PERSISTENT"
    assert R(_mH(1.3, 0.95, 0.5), True) == "NONLINEAR DIRECTION-SHARED"          # above the band is nonlinear too (two-sided)
    assert R(_mH(0.6, 0.95, 0.5), True) == "NONLINEAR DIRECTION-SHARED"
    assert R(_mH(1.0, 0.95, 0.1), True) == "INTERMEDIATE"                        # in band, shared, but n_1(H) < 0.25
    assert R(_mH(0.1, 0.2, 0.5), True) == "TRAJECTORY-SENSITIVE"
    assert R(_mH(0.5, 0.2, 0.5), True) == "INTERMEDIATE" and R(_mH(0.1, 0.7, 0.5), True) == "INTERMEDIATE"
    assert R(_mH(None, 0.95, 0.5), True) == "REGIME-UNRESOLVED" and R(_mH(1.0, None, 0.5), True) == "REGIME-UNRESOLVED"
    assert R(_mH(1.0, 0.95, 0.5), False) == "REGIME-UNRESOLVED"                  # preflight failure never becomes a three-point ladder
    assert R(_mH(None, None, 0.01, {"e1e-3": 0.0, "e1e-2": 0.01, "e1e-1": 0.02, "e1": 0.04}), True) == "FORGOTTEN"    # FORGOTTEN first, n only
    assert R(_mH(1.0, 0.95, 0.5, {"e1e-3": None, "e1e-2": 0.3, "e1e-1": 0.3, "e1": 0.5}), True) == "REGIME-UNRESOLVED"
    assert mod.ALPHA_BAND == (0.80, 1.20) and mod.COS_SHARED == 0.90 and mod.ALPHA_FLAT == 0.20 and mod.COS_DECOR == 0.50


def test_function_bar_absolute_and_locus(mod):
    per, lab = mod.function_bar({"e1e-3": 0.004, "e1e-2": -0.005, "e1e-1": 0.0, "e1": 0.0049})
    assert all(v == "NEUTRAL" for v in per.values()) and lab == "FUNCTION-NEUTRAL"
    per, lab = mod.function_bar({"e1e-3": 0.02, "e1e-2": -0.0051, "e1e-1": 0.0, "e1": 0.0})
    assert per["e1e-3"] == "HARMED" and per["e1e-2"] == "HELPED" and lab == "FUNCTION-e1e-3:HARMED,e1e-2:HELPED"
    # the twin's own dCE never enters another arm's bar
    per2, _ = mod.function_bar({"e1e-3": 0.1, "e1e-2": 0.0, "e1e-1": 0.0, "e1": 0.004})
    assert per2["e1"] == "NEUTRAL"
    ms = {1: {"alpha": 1.0, "cosmin": 0.99}, 5: {"alpha": 1.1, "cosmin": 0.95}, 20: {"alpha": 0.9, "cosmin": 0.8}, 100: {"alpha": None, "cosmin": None},
          900: {"alpha": 0.3, "cosmin": 0.4}, 8220: {"alpha": 0.1, "cosmin": 0.1}}
    assert mod.locus(ms) == {"h_lin": 5, "h_dec": 900}
    assert mod.locus({1: {"alpha": 1.3, "cosmin": 0.99}}) == {"h_lin": None, "h_dec": None}
    assert mod.label("INTERMEDIATE", "FUNCTION-NEUTRAL").startswith("INTERMEDIATE+FUNCTION-NEUTRAL [writer A only")


def test_arena_literals_and_no_b_path(mod):
    assert mod.LADDER == [1e-3, 1e-2, 1e-1, 1.0] and mod.ARM_ORDER == ["C", "e1", "e1e-2", "e1e-1", "e1e-3"]
    assert mod.ARENA["grid"] == [1, 2, 3] and mod.ARENA["leg"] == 8220 and mod.ARENA["qual"] == [1, 5, 20, 100, 900]   # SMOKE grid under the fixture
    assert mod.N_PRED == 0.894284652673395 and mod.WRITER == "A" and mod.CE_ABS == 0.005 and mod.COS1_MIN == 0.999 and mod.R1_TOL == 1e-3
    assert mod.MIN_FREE_BYTES == 16 * (1 << 30)
    src = (ROOT / "scratch" / "first_moment_erasure_ladder.py").read_text()
    assert "program_label" not in src and "backsched" not in src
    real_grid = [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220]
    assert str(real_grid) in src and "gallery19m_phase_s2.pt" in src and "15300" in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "WRITERS":
            assert isinstance(node.slice, ast.Name) and node.slice.id == "WRITER", ast.dump(node)
        if isinstance(node, ast.Constant) and node.value == "B":
            raise AssertionError("string constant 'B' in the single-writer instrument")


def test_disk_preflight_and_refuse_if_exists(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None)      # main() sets global torch state before refusing
    monkeypatch.setattr(torch, "set_num_threads", lambda *a, **k: None)
    ok, free = mod.disk_preflight(tmp_path, min_free=1)
    assert ok and free > 0
    assert not mod.disk_preflight(tmp_path, min_free=free + (1 << 40))[0]
    monkeypatch.setattr(mod, "RECEIPT", tmp_path / "ladder.json")
    monkeypatch.setattr(mod, "STREAM", tmp_path / "ladder.jsonl")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path)
    monkeypatch.setattr(mod, "CK_DIR", tmp_path / "ck")
    (tmp_path / "ladder.json").write_text("{}")
    with pytest.raises(SystemExit):
        mod.main()
    (tmp_path / "ladder.json").unlink()
    (tmp_path / "ck" / "A" / "C").mkdir(parents=True)
    (tmp_path / "ck" / "A" / "C" / "h0001.pt").write_bytes(b"x")
    with pytest.raises(SystemExit):
        mod.main()


def test_preflight_bypass_is_smoke_only(mod):
    src = (ROOT / "scratch" / "first_moment_erasure_ladder.py").read_text()
    assert 'PREFLIGHT_BYPASS = SMOKE and os.environ.get("SMOKE_PREFLIGHT_BYPASS", "0") == "1"' in src
    assert mod.PREFLIGHT_BYPASS is False                      # not set under the fixture
    assert mod.regime(_mH(1.0, 0.95, 0.5), False) == "REGIME-UNRESOLVED"      # a bypassed preflight still books REGIME-UNRESOLVED


def test_real_mode_arena_constants():
    """The module's real-mode constants (SMOKE unset), evaluated in a subprocess: 8220-step leg to 15420, the 12-point grid, the
    five-point qualification set, the substrate pins, and no preflight bypass."""
    import subprocess
    code = ("import os,sys,json; os.environ.pop('SMOKE', None); os.environ.pop('SMOKE_PREFLIGHT_BYPASS', None); sys.path[:0]=['.','scripts','scratch']; "
            "import first_moment_erasure_ladder as L; print(json.dumps([L.LEG_FULL, L.ANCHOR + L.LEG_FULL, L.GRID, L.QUAL, L.H_END, L.TAIL_H, str(L.END_MODEL), "
            "L.END_DIGEST, L.END_SHA, L.MID_STEP, str(L.MID_MODEL), L.MID_DIGEST, L.MID_SHA, L.PREFLIGHT_BYPASS, str(L.CK_DIR), str(L.RECEIPT), str(L.FME1)]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    v = json.loads(out.stdout.strip().splitlines()[-1])
    assert v[:6] == [8220, 15420, [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220], [1, 5, 20, 100, 900], 8220, 6000]
    assert v[6:13] == ["checkpoints/gallery19m_phase_s2.pt", "4633efe5d376f911", "e7207b3bd4df541a", 15300, "checkpoints/phase19m/m015300.pt", "d97b19e0ff3c84e5", "6a715beb6394e3b3"]
    assert v[13] is False and v[14:] == ["checkpoints/fmel1", "logs/fmel1/ladder.json", "logs/fme1/treat.json"]


def test_abort_and_preflight_failure_paths(mod, tmp_path, monkeypatch):
    """The in-line abort (a BAR 0 mismatch or a preflight-digest mismatch inside on_step) raises Abort after filling the cell;
    mode_ladder's caller books NOT-RUN and main()'s finally still writes the receipt; the PREFLIGHT-FAILED early return carries the
    registered label. Driven through a fake run_leg so no model is bound."""
    import types
    calls = {}

    class FakeModel:
        pass

    def fake_bind(writer, tok, dev):
        return FakeModel(), types.SimpleNamespace(param_groups=[{"lr": 1.0, "betas": (0.9, 0.999), "params": []}]), None, {"serialized": {}}

    def fake_run_leg(model, opt, sched, tok, enc, slices, dev, horizons, on_step=None):
        for i in range(1, len(slices) + 1):
            on_step(i, 0.0)
        return {}, [0.0] * len(slices), "x"

    monkeypatch.setattr(mod.OMA, "bind", fake_bind)
    monkeypatch.setattr(mod.OMA, "resume_sched", lambda kind, opt, anchor, ser: (None, {}))
    monkeypatch.setattr(mod, "apply_eps", lambda opt, eps: 0 if eps == 0.0 else len(mod.UG.KEYS))
    monkeypatch.setattr(mod.OMA, "run_leg", fake_run_leg)
    monkeypatch.setattr(mod.OMA, "sd_cpu", lambda m: {"w": torch.zeros(1)})
    monkeypatch.setattr(mod, "state_digest", lambda sd: "d" * 64)
    refs = {"C": {1: "d" * 64, 2: "0" * 64, 3: "d" * 64}}
    cell = {}
    with pytest.raises(mod.Abort):
        mod.long_leg("C", 0.0, None, None, [None] * 3, None, None, None, refs, "d" * 64, lambda row: calls.setdefault("rows", []).append(row), cell, None)
    assert cell["abort"].startswith("BAR 0 mismatch at h = 2") and cell["qualification"]["1"]["ok"] and not cell["qualification"]["2"]["ok"]
    assert "snapshots" not in cell                                 # the aborting arm writes no snapshot
    cell2 = {}
    with pytest.raises(mod.Abort):
        mod.long_leg("e1", 1.0, None, None, [None] * 3, None, None, None, refs, "p" * 64, lambda row: None, cell2, None)
    assert cell2["preflight_digest_match"] is False and "preflight" in cell2["abort"]
    # main()'s finally writes the receipt even when mode_ladder raises
    monkeypatch.setattr(mod, "RECEIPT", tmp_path / "ladder.json"); monkeypatch.setattr(mod, "STREAM", tmp_path / "ladder.jsonl")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path); monkeypatch.setattr(mod, "CK_DIR", tmp_path / "ck")
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None); monkeypatch.setattr(torch, "set_num_threads", lambda *a, **k: None)
    monkeypatch.setattr(mod.TM, "MathTokenizer", lambda: types.SimpleNamespace(vocab=[0] * 40))
    monkeypatch.setattr(mod.OMA.OA, "assert_verbatim", lambda: None)
    monkeypatch.setattr(mod.UG, "build", lambda tok, dev: None)
    monkeypatch.setattr(mod.UG, "flatten_law", lambda m: ([], 18_911_616, "f"))
    monkeypatch.setattr(mod.OMA, "future_stream", lambda tok: ([], [0] * 5140, {"n_enc": 164_490, "steps_per_epoch": 5_140, "probe64_digest": "p"}))
    monkeypatch.setattr(mod.OMA, "UGC0", tmp_path / "ugc0.json"); (tmp_path / "ugc0.json").write_text(json.dumps({"probe": {"digest": "p"}}))
    monkeypatch.setattr(mod.UG, "probe_batches", lambda tok: ([], {"held": [], "digest": "p"}))
    monkeypatch.setattr(mod, "SMOKE", True); monkeypatch.setattr(mod, "MODE", "smoke")
    monkeypatch.setattr(mod, "FME1", tmp_path / "fme1.json"); (tmp_path / "fme1.json").write_text("{}")
    monkeypatch.setattr(mod, "STAGE0", tmp_path / "s0.json"); (tmp_path / "s0.json").write_text("{}")
    monkeypatch.setenv("SMOKE_ARENA_TAG", "x")
    monkeypatch.setattr(mod.OMA, "base_record", lambda *a: {"source_sha256": "s", "writers": {"A": {}}, "eps_twin": 0.01})
    monkeypatch.setattr(mod, "sha256_file", lambda p: "s")

    def boom(*a, **k):
        raise RuntimeError("mid-run failure")

    monkeypatch.setattr(mod, "mode_ladder", boom)
    with pytest.raises(RuntimeError):
        mod.main()
    rec = json.loads((tmp_path / "ladder.json").read_text())
    assert "wall_s" in rec and rec["arms"] == {} and rec["oma_source_sha256"] == "s" and list(rec["writers"]) == ["A"]
    # the PREFLIGHT-FAILED label
    assert mod.label("REGIME-UNRESOLVED", "FUNCTION-NOT-MEASURED").startswith("REGIME-UNRESOLVED+FUNCTION-NOT-MEASURED")

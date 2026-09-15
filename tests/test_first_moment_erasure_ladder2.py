"""FIRST-MOMENT-ERASURE-LADDER-2 guards: pins (L1's set + L1 itself + the locked FMEL1 receipt); reference digests read from the
locked receipts WITHOUT opening any historical checkpoint; the three-point ladder with independent alpha_global and the local decade
slopes, the log-uniform identity alpha_global == (alpha_low + alpha_high) / 2, the zero-norm / one-missing-decade / non-finite law
and classification refusal (never NaN, never fall-through); the preflight receipt separating identities from informative checks;
the regime order requiring BOTH local slopes; the locus incl. h_curve; the absolute function bar; real-mode constants in a
subprocess; refuse-if-exists; no writer-B path."""
import ast
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
        m = importlib.import_module("first_moment_erasure_ladder2")
    finally:
        os.chdir(cwd)
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


ARMS = {"e1e-2": 1e-2, "e1e-1": 1e-1, "e1": 1.0}


def test_source_pins_and_locked_receipts(mod):
    assert mod.FME.check_pins(mod.PINS, root=ROOT) == {}
    assert set(mod.PINS) > set(mod.L1.PINS) and "scratch/first_moment_erasure_ladder.py" in mod.PINS and "scratch/first_moment_erasure.py" in mod.PINS
    lock = json.loads((ROOT / "docs" / "receipts.lock.json").read_text())
    for rel, lit in (("logs/fmel1/ladder.json", mod.FMEL1_SHA), ("logs/fme1/treat.json", mod.L1.FME1_SHA), ("logs/oma1/stage0.json", mod.FME.STAGE0_SHA)):
        assert mod.FME.locked_sha(lock, rel) == lit == mod.sha256_file(ROOT / rel)
    assert mod.FME.leg_path_sha() == mod.FME.LEG_PATH_SHA


def test_reference_digests_read_receipts_not_checkpoints(mod, monkeypatch):
    """BAR 0's references come from the locked receipts only: torch.load is forbidden for the whole call, and any path under
    checkpoints/fme1 or checkpoints/oma1 raises."""
    stage0 = json.loads((ROOT / "logs" / "oma1" / "stage0.json").read_text())
    fme1 = json.loads((ROOT / "logs" / "fme1" / "treat.json").read_text())
    fmel1 = json.loads((ROOT / "logs" / "fmel1" / "ladder.json").read_text())

    def forbidden(*a, **k):
        raise AssertionError(f"torch.load called during reference_digests: {a[:1]}")

    monkeypatch.setattr(torch, "load", forbidden)
    real_q = [1, 5, 20, 100, 900]
    refs = mod.reference_digests(stage0, fme1, fmel1, horizons=real_q)
    assert set(refs) == {"C", "e1", "e1e-2", "e1e-1"}
    assert set(refs["C"]) == set(refs["e1"]) == set(refs["e1e-2"]) == set(real_q) and set(refs["e1e-1"]) == {1}
    assert refs["e1e-1"][1] == fmel1["preflight"]["arms"]["e1e-1"]["digest"]
    assert refs["C"][1] == fmel1["preflight"]["arms"]["C"]["digest"] == stage0["cells"]["A"]["C"]["snapshots"]["1"]["state_digest"]
    assert refs["e1"][1] == fme1["arms"]["Z"]["snapshots"]["1"]["state_digest"] and all(len(v) == 64 for a in refs for v in refs[a].values())
    bad = json.loads(json.dumps(fmel1)); bad["preflight"]["arms"]["e1"]["digest"] = "0" * 64
    with pytest.raises(AssertionError):
        mod.reference_digests(stage0, fme1, bad, horizons=real_q)
    # the FMEL1 provenance block against the real receipts (repo-relative paths; the lock is keyed that way)
    monkeypatch.chdir(ROOT)
    old = mod.FMEL1, mod.FME1, mod.STAGE0
    mod.FMEL1, mod.FME1, mod.STAGE0 = Path("logs/fmel1/ladder.json"), Path("logs/fme1/treat.json"), Path("logs/oma1/stage0.json")
    try:
        out = mod.assert_fmel1_provenance(fmel1, lock := json.loads((ROOT / "docs" / "receipts.lock.json").read_text()), stage0, fme1)
        assert out["fmel1_status"] == "PREFLIGHT-FAILED" and out["fmel1_commit"] == "f9872a27"
        with pytest.raises(AssertionError):
            mod.assert_fmel1_provenance(dict(fmel1, status="DONE"), lock, stage0, fme1)
    finally:
        mod.FMEL1, mod.FME1, mod.STAGE0 = old
    # no checkpoint under the historical trees is ever named as a state input by this instrument
    src = (ROOT / "scratch" / "first_moment_erasure_ladder2.py").read_text()
    tree = ast.parse(src)
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)][1:]      # [0] = the module docstring
    assert not any(("checkpoints/fme1/" in c) or ("checkpoints/oma1" in c) for c in consts), "a historical checkpoint tree is named as a path in the instrument"


def _world(rng, d=60, slopes=None, n_target=None):
    """Linear (slopes None) or per-decade power law: ||dW_eps|| = s * eps^a with a piecewise exponent."""
    W0 = rng.normal(size=d); WC = W0 + rng.normal(size=d)
    u = rng.normal(size=d); u /= np.linalg.norm(u)
    scale = 1.0 if n_target is None else n_target * float(np.linalg.norm(WC - W0))
    if slopes is None:
        mag = {a: e * scale for a, e in ARMS.items()}
    else:
        lo, hi = slopes
        mag = {"e1": scale, "e1e-1": scale * 10 ** (-hi), "e1e-2": scale * 10 ** (-hi - lo)}
    return W0, WC, {a: WC + mag[a] * u for a in ARMS}


def test_alpha_identity_and_local_slopes(mod):
    rng = np.random.default_rng(0)
    W0, WC, W = _world(rng)
    m = mod.horizon_metrics(W0, WC, W)
    assert abs(m["alpha_global"] - 1) < 1e-9 and abs(m["alpha_low"] - 1) < 1e-9 and abs(m["alpha_high"] - 1) < 1e-9
    assert m["alpha_identity_gap"] <= mod.ALPHA_IDENTITY_TOL and m["cosmin"] > 1 - 1e-9 and len(m["pair_cos"]) == 3 and m["undefined"] == {}
    assert all(abs(m["R"][a] - 1) < 1e-9 for a in ARMS)
    # curvature: low decade slope 0.5, high decade slope 1.1 -> global 0.8 by the identity, computed independently
    W0, WC, W = _world(rng, slopes=(0.5, 1.1))
    m = mod.horizon_metrics(W0, WC, W)
    assert abs(m["alpha_low"] - 0.5) < 1e-9 and abs(m["alpha_high"] - 1.1) < 1e-9 and abs(m["alpha_global"] - 0.8) < 1e-9
    # the regression really is independent: feeding it a non-log-uniform ladder breaks the identity
    ln = {"e1e-2": 0.0, "e1e-1": 1.0, "e1": 3.0}
    assert abs(mod.alpha_regression({"e1e-2": 1e-2, "e1e-1": 3e-2, "e1": 1.0}, ln) - (mod.alpha_regression(ARMS, ln))) > 1e-3
    assert mod.alpha_regression(ARMS, {"e1": 0.0}) is None


def test_zero_norm_missing_decade_nonfinite_and_refusal(mod):
    rng = np.random.default_rng(1)
    W0, WC, W = _world(rng)
    # one dead arm (a missing decade): the decade slopes touching it are UNDEFINED, alpha_global UNDEFINED, the other decade defined
    Wz = dict(W); Wz["e1e-2"] = WC.copy()
    z = mod.horizon_metrics(W0, WC, Wz)
    assert z["n"]["e1e-2"] == 0.0 and z["R"]["e1e-2"] == 0.0 and z["ln"]["e1e-2"] is None
    assert z["alpha_low"] is None and abs(z["alpha_high"] - 1) < 1e-9 and z["alpha_global"] is None and z["cosmin"] is None
    assert {"alpha_global", "alpha_low", "cosmin", "ln:e1e-2"} <= set(z["undefined"]) and "alpha_high" not in z["undefined"]
    assert "nan" not in json.dumps(z).lower()
    assert mod.regime(z, True) == "REGIME-UNRESOLVED"
    # eps = 1 dead: every R UNDEFINED
    W0b, WCb, Wb = _world(rng, n_target=2.0)                   # large n so FORGOTTEN (a legal first match) is not the answer
    Wz1 = dict(Wb); Wz1["e1"] = WCb.copy()
    z1 = mod.horizon_metrics(W0b, WCb, Wz1)
    assert all(z1["R"][a] is None for a in ARMS) and z1["alpha_high"] is None and mod.regime(z1, True) == "REGIME-UNRESOLVED"
    # non-finite weights: every derived value UNDEFINED with a reason, no NaN in the JSON, classification refuses
    Wn = dict(W); Wn["e1e-1"] = W["e1e-1"].copy(); Wn["e1e-1"][0] = float("nan")
    nf = mod.horizon_metrics(W0, WC, Wn)
    assert nf["alpha_global"] is None and nf["alpha_low"] is None and nf["alpha_high"] is None and nf["cosmin"] is None and nf["pair_cos"] == {}
    assert all(nf["n"][a] is None for a in ARMS) and "nonfinite" in nf["undefined"] and "nan" not in json.dumps(nf).lower()
    assert mod.regime(nf, True) == "REGIME-UNRESOLVED"
    Wi = dict(W); Wi["e1"] = W["e1"].copy(); Wi["e1"][0] = float("inf")
    assert mod.regime(mod.horizon_metrics(W0, WC, Wi), True) == "REGIME-UNRESOLVED"
    # a zero control leg: n UNDEFINED -> refusal (not FORGOTTEN, not fall-through)
    zc = mod.horizon_metrics(W0, W0, {a: W0 + (W[a] - WC) for a in ARMS})
    assert all(zc["n"][a] is None for a in ARMS) and mod.regime(zc, True) == "REGIME-UNRESOLVED"
    assert mod.cosine(np.zeros(3), np.ones(3)) is None and mod.cosine(np.array([float("nan"), 1.0]), np.ones(2)) is None


def test_preflight_identities_and_informative_checks(mod):
    rng = np.random.default_rng(2)
    W0, WC, W = _world(rng, n_target=0.8942846)
    m1 = mod.horizon_metrics(W0, WC, W)
    ok, det = mod.preflight_law(m1, n_pred=0.894284652673395)
    assert ok and det["fails"] == []
    assert det["identities"] == ["R:e1", "cos:e1"] and set(det["informative"]) == {"n_1", "R:e1e-2", "cos:e1e-2", "R:e1e-1", "cos:e1e-1"}
    assert det["cos_to_e1"]["e1"] == 1.0 and m1["R"]["e1"] == 1.0
    bad = dict(m1); bad["R"] = dict(m1["R"], **{"e1e-2": 1.0011})
    assert not mod.preflight_law(bad, n_pred=0.894284652673395)[0]
    bad2 = dict(m1); bad2["pair_cos"] = dict(m1["pair_cos"], **{"e1e-2|e1": 0.9989})
    ok2, det2 = mod.preflight_law(bad2, n_pred=0.894284652673395)
    assert not ok2 and det2["fails"] == ["cos(dW_e1e-2(1), dW_1(1)) 0.9989 < 0.999"]
    assert not mod.preflight_law(m1, n_pred=0.5)[0]
    dead = dict(m1); dead["R"] = dict(m1["R"], **{"e1e-1": None})
    assert not mod.preflight_law(dead, n_pred=0.894284652673395)[0]
    assert (mod.R1_TOL, mod.COS1_MIN, mod.BAR1_TOL) == (1e-3, 0.999, 0.02)


def _mH(ag, al, ah, cm, n1, n_all=None):
    n = {"e1e-2": 0.3, "e1e-1": 0.3, "e1": n1} if n_all is None else n_all
    return {"alpha_global": ag, "alpha_low": al, "alpha_high": ah, "cosmin": cm, "n": n}


def test_regime_order_requires_both_local_slopes(mod):
    R = mod.regime
    assert R(_mH(1.0, 1.0, 1.0, 0.95, 0.5), True) == "MAGNITUDE-SCALED PERSISTENT"
    assert R(_mH(1.0, 0.81, 1.19, 0.95, 0.5), True) == "MAGNITUDE-SCALED PERSISTENT"
    assert R(_mH(1.0, 0.7, 1.3, 0.95, 0.5), True) == "NONLINEAR DIRECTION-SHARED"        # global slope 1.0 hides the curvature; locals do not
    assert R(_mH(1.0, 1.0, 1.25, 0.95, 0.5), True) == "NONLINEAR DIRECTION-SHARED"
    assert R(_mH(0.6, 0.6, 0.6, 0.95, 0.5), True) == "NONLINEAR DIRECTION-SHARED"
    assert R(_mH(1.0, 1.0, 1.0, 0.95, 0.1), True) == "INTERMEDIATE"                      # not persistent
    assert R(_mH(0.1, 0.1, 0.1, 0.2, 0.5), True) == "TRAJECTORY-SENSITIVE"
    assert R(_mH(0.1, 0.0, 0.45, 0.2, 0.5), True) == "INTERMEDIATE"                      # a local slope above LOCAL_FLAT
    assert R(_mH(0.5, 0.5, 0.5, 0.2, 0.5), True) == "INTERMEDIATE" and R(_mH(0.1, 0.1, 0.1, 0.7, 0.5), True) == "INTERMEDIATE"
    assert R(_mH(None, 1.0, 1.0, 0.95, 0.5), True) == "REGIME-UNRESOLVED" and R(_mH(1.0, None, 1.0, 0.95, 0.5), True) == "REGIME-UNRESOLVED"
    assert R(_mH(1.0, 1.0, None, 0.95, 0.5), True) == "REGIME-UNRESOLVED" and R(_mH(1.0, 1.0, 1.0, None, 0.5), True) == "REGIME-UNRESOLVED"
    assert R(_mH(1.0, 1.0, 1.0, 0.95, 0.5), False) == "REGIME-UNRESOLVED"
    assert R(_mH(None, None, None, None, 0.01, {"e1e-2": 0.0, "e1e-1": 0.02, "e1": 0.04}), True) == "FORGOTTEN"
    assert R(_mH(1.0, 1.0, 1.0, 0.95, 0.5, {"e1e-2": None, "e1e-1": 0.3, "e1": 0.5}), True) == "REGIME-UNRESOLVED"
    assert R(_mH(1.0, 1.0, 1.0, 0.95, float("nan")), True) == "REGIME-UNRESOLVED"
    assert (mod.ALPHA_BAND, mod.COS_SHARED, mod.ALPHA_FLAT, mod.LOCAL_FLAT, mod.COS_DECOR, mod.CURVE_GAP) == ((0.80, 1.20), 0.90, 0.20, 0.40, 0.50, 0.20)


def test_locus_function_bar_and_label(mod):
    ms = {1: {"alpha_low": 1.0, "alpha_high": 1.0, "cosmin": 0.99}, 5: {"alpha_low": 1.1, "alpha_high": 0.95, "cosmin": 0.95},
          20: {"alpha_low": 0.9, "alpha_high": 1.15, "cosmin": 0.8}, 100: {"alpha_low": None, "alpha_high": 0.7, "cosmin": None},
          300: {"alpha_low": 0.6, "alpha_high": 0.85, "cosmin": 0.6}, 900: {"alpha_low": 0.3, "alpha_high": 0.7, "cosmin": 0.4}, 8220: {"alpha_low": 0.1, "alpha_high": 0.1, "cosmin": 0.1}}
    assert mod.locus(ms) == {"h_lin": 5, "h_dec": 900, "h_curve": 20}
    assert mod.locus({1: {"alpha_low": 1.3, "alpha_high": 1.0, "cosmin": 0.99}}) == {"h_lin": None, "h_dec": None, "h_curve": 1}
    per, lab = mod.function_bar({"e1e-2": 0.004, "e1e-1": -0.005, "e1": 0.0049})
    assert all(v == "NEUTRAL" for v in per.values()) and lab == "FUNCTION-NEUTRAL"
    per, lab = mod.function_bar({"e1e-2": 0.02, "e1e-1": -0.0051, "e1": 0.0})
    assert per["e1e-2"] == "HARMED" and per["e1e-1"] == "HELPED" and lab == "FUNCTION-e1e-2:HARMED,e1e-1:HELPED"
    assert mod.label("INTERMEDIATE", "FUNCTION-NEUTRAL").endswith("eps >= 1e-2; gate descriptive]")


def test_arena_literals_no_b_path_and_l1_sharing(mod):
    assert mod.LADDER == [1e-2, 1e-1, 1.0] and mod.ARM_ORDER == ["C", "e1", "e1e-2", "e1e-1"] and set(mod.ARMS) == {"C", "e1e-2", "e1e-1", "e1"}
    assert mod.ARENA["grid"] == [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220] and mod.ARENA["leg"] == 8220
    assert mod.GRID is mod.L1.GRID and mod.QUAL is mod.L1.QUAL and mod.long_leg is mod.L1.long_leg and mod.apply_eps is mod.L1.apply_eps
    assert mod.N_PRED == 0.894284652673395 and mod.WRITER == "A" and mod.CE_ABS == 0.005 and mod.MIN_FREE_BYTES == 16 * (1 << 30)
    src = (ROOT / "scratch" / "first_moment_erasure_ladder2.py").read_text()
    assert "program_label" not in src and "backsched" not in src and "e1e-3" not in src and "1e-3]" not in src      # no fourth rung anywhere
    assert "float64" not in src.lower()                                      # no float64-master work under this rung
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "WRITERS":
            assert isinstance(node.slice, ast.Name) and node.slice.id == "WRITER", ast.dump(node)
        if isinstance(node, ast.Constant) and node.value == "B":
            raise AssertionError("string constant 'B' in the single-writer instrument")


def test_real_mode_arena_constants():
    import subprocess
    code = ("import os,sys,json; os.environ.pop('SMOKE', None); os.environ.pop('SMOKE_PREFLIGHT_BYPASS', None); sys.path[:0]=['.','scripts','scratch']; "
            "import first_moment_erasure_ladder2 as L; print(json.dumps([L.LEG_FULL, L.ANCHOR + L.LEG_FULL, L.GRID, L.QUAL, L.H_END, L.TAIL_H, L.LADDER, L.PREFLIGHT_BYPASS, "
            "str(L.CK_DIR), str(L.RECEIPT), str(L.FMEL1), str(L.FME1), str(L.STAGE0), str(L.END_MODEL), L.END_SHA, L.MID_STEP, L.MID_SHA]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    v = json.loads(out.stdout.strip().splitlines()[-1])
    assert v[:7] == [8220, 15420, [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220], [1, 5, 20, 100, 900], 8220, 6000, [1e-2, 1e-1, 1.0]]
    assert v[7] is False and v[8:13] == ["checkpoints/fmel2", "logs/fmel2/ladder.json", "logs/fmel1/ladder.json", "logs/fme1/treat.json", "logs/oma1/stage0.json"]
    assert v[13:] == ["checkpoints/gallery19m_phase_s2.pt", "e7207b3bd4df541a", 15300, "6a715beb6394e3b3"]


def test_refuse_if_exists(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *a, **k: None)
    monkeypatch.setattr(mod, "RECEIPT", tmp_path / "ladder.json"); monkeypatch.setattr(mod, "STREAM", tmp_path / "ladder.jsonl")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path); monkeypatch.setattr(mod, "CK_DIR", tmp_path / "ck")
    (tmp_path / "ladder.json").write_text("{}")
    with pytest.raises(SystemExit):
        mod.main()
    (tmp_path / "ladder.json").unlink()
    (tmp_path / "ck" / "A" / "C").mkdir(parents=True); (tmp_path / "ck" / "A" / "C" / "h0001.pt").write_bytes(b"x")
    with pytest.raises(SystemExit):
        mod.main()


def test_preflight_disposition_identity_flag_and_bypass_is_smoke_only(mod):
    D = mod.preflight_disposition
    assert D(True, False, True) is None
    st, reg, lab = D(False, False, True)
    assert (st, reg) == ("PREFLIGHT-FAILED", "REGIME-UNRESOLVED") and lab.startswith("REGIME-UNRESOLVED+FUNCTION-NOT-MEASURED [writer A only")
    assert D(False, True, True) is None                                       # SMOKE-only bypass proceeds to the legs ...
    assert mod.regime(_mH(1.0, 1.0, 1.0, 0.95, 0.5), False) == "REGIME-UNRESOLVED"     # ... but a failed preflight never becomes a regime
    st, reg, lab = D(True, False, False)
    assert (st, reg) == ("NOT-RUN", "NOT-RUN") and "substrate change" in lab
    assert D(False, False, False)[0] == "PREFLIGHT-FAILED"                    # the preflight law is judged before the digest law
    src = (ROOT / "scratch" / "first_moment_erasure_ladder2.py").read_text()
    assert 'PREFLIGHT_BYPASS = SMOKE and os.environ.get("SMOKE_PREFLIGHT_BYPASS", "0") == "1"' in src and mod.PREFLIGHT_BYPASS is False
    # the log-uniform identity: recorded at every horizon, refused (REGIME-UNRESOLVED) after the legs, raised only in the strict preflight call
    rng = np.random.default_rng(3)
    W0, WC, W = _world(rng)
    m = mod.horizon_metrics(W0, WC, W)
    assert m["alpha_identity_ok"] is True and m["alpha_identity_gap"] <= mod.ALPHA_IDENTITY_TOL
    forged = dict(m, alpha_identity_ok=False)
    assert mod.regime(forged, True) == "REGIME-UNRESOLVED"
    real = mod.alpha_regression
    try:
        mod.alpha_regression = lambda eps_of, ln: real(eps_of, ln) + 1e-6
        loose = mod.horizon_metrics(W0, WC, W)
        assert loose["alpha_identity_ok"] is False and "alpha_identity" in loose["undefined"] and mod.regime(loose, True) == "REGIME-UNRESOLVED"
        with pytest.raises(AssertionError):
            mod.horizon_metrics(W0, WC, W, strict_identity=True)
    finally:
        mod.alpha_regression = real
    # BAR 5 tail and the substrate rho never divide a None / non-finite value
    assert mod.finite(None) is False and mod.finite(float("nan")) is False and mod.finite(1.0) is True

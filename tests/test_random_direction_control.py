"""RANDOM-DIRECTION-CONTROL-1 guards: pins (L2's set + L2 itself + the locked FMEL2 receipt); BAR 0 references and the comparison-
vector records read from the locked receipt WITHOUT opening any checkpoint; the K map against a real AdamW step; the construction
law (per-group write-norm match, group shares, orthogonality, no division by K, seed reproducibility); the preflight law and its
refusals; growth / A_r with the zero-norm law; BAR 2 / 3 / 4 bands and boundaries; the leg-path copy's source identity with L1;
the wall cap; real-mode constants in a subprocess; refuse-if-exists; no writer-B path."""
import ast
import importlib
import inspect
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
        m = importlib.import_module("random_direction_control")
    finally:
        os.chdir(cwd)
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


SEGS = [("a", 0, 1200, "BLOCK0"), ("b", 1200, 1250, "OUTSIDE")]
D = 1250
GRP = {"lr": 1.7e-4, "beta1": 0.892, "beta2": 0.999, "wd": 0.01}


class _M(torch.nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b


def _state(seed, dtype):
    torch.manual_seed(seed)
    q1 = torch.nn.Parameter(torch.randn(40, 30, dtype=dtype)); q2 = torch.nn.Parameter(torch.randn(50, dtype=dtype))
    opt = torch.optim.AdamW([q1, q2], lr=GRP["lr"], betas=(GRP["beta1"], GRP["beta2"]), weight_decay=GRP["wd"])
    for q in (q1, q2):
        opt.state[q] = {"step": torch.tensor(7200.0), "exp_avg": torch.randn_like(q) * 1e-3, "exp_avg_sq": torch.rand_like(q) * 1e-6}
    return _M(q1, q2), opt


def _flat(m):
    return np.concatenate([m.a.detach().double().reshape(-1).numpy(), m.b.detach().double().numpy()])


def test_source_pins_and_locked_receipts(mod):
    assert mod.FME.check_pins(mod.PINS, root=ROOT) == {}
    assert set(mod.PINS) > set(mod.L2.PINS) and "scratch/first_moment_erasure_ladder2.py" in mod.PINS and "scratch/first_moment_erasure_ladder.py" in mod.PINS
    lock = json.loads((ROOT / "docs" / "receipts.lock.json").read_text())
    for rel, lit in (("logs/fmel2/ladder.json", mod.FMEL2_SHA), ("logs/fmel1/ladder.json", mod.L2.FMEL1_SHA), ("logs/fme1/treat.json", mod.L1.FME1_SHA), ("logs/oma1/stage0.json", mod.FME.STAGE0_SHA)):
        assert mod.FME.locked_sha(lock, rel) == lit == mod.sha256_file(ROOT / rel)
    assert mod.FME.leg_path_sha() == mod.FME.LEG_PATH_SHA
    assert mod.SEEDS == {"R1": 2026091501, "R2": 2026091502, "R3": 2026091503} and mod.ARM_ORDER == ["C", "R1", "R2", "R3"]


def test_reference_digests_and_comparison_vectors_read_receipt_not_checkpoints(mod, monkeypatch):
    fmel2 = json.loads((ROOT / "logs" / "fmel2" / "ladder.json").read_text())
    real_grid = [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220]

    def forbidden(*a, **k):
        raise AssertionError(f"torch.load called during reference_digests: {a[:1]}")

    monkeypatch.setattr(torch, "load", forbidden)
    old = mod.M_TREE; mod.M_TREE = "checkpoints/fmel2/A/e1e-1/"
    try:
        refs, mrefs = mod.reference_digests(fmel2, horizons=real_grid)
    finally:
        mod.M_TREE = old
    assert set(refs) == {"C"} and set(refs["C"]) == set(real_grid) and all(len(v) == 64 for v in refs["C"].values())
    assert set(mrefs) == set(real_grid) and all(mrefs[h]["path"] == f"checkpoints/fmel2/A/e1e-1/h{h:04d}.pt" for h in real_grid)
    assert refs["C"][1] == fmel2["preflight"]["arms"]["C"]["digest"] and mrefs[1]["state_digest"] == fmel2["preflight"]["arms"]["e1e-1"]["digest"]
    bad = json.loads(json.dumps(fmel2)); bad["arms"]["e1e-1"]["snapshots"]["1"]["state_digest"] = "0" * 64
    mod.M_TREE = "checkpoints/fmel2/A/e1e-1/"
    try:
        with pytest.raises(AssertionError):
            mod.reference_digests(bad, horizons=real_grid)
        # the provenance block against the real receipts (repo-relative paths; the lock is keyed that way)
        monkeypatch.chdir(ROOT)
        stage0 = json.loads(Path("logs/oma1/stage0.json").read_text()); fme1 = json.loads(Path("logs/fme1/treat.json").read_text()); fmel1 = json.loads(Path("logs/fmel1/ladder.json").read_text())
        lock = json.loads(Path("docs/receipts.lock.json").read_text())
        paths = mod.FMEL2, mod.FMEL1, mod.FME1, mod.STAGE0
        mod.FMEL2, mod.FMEL1, mod.FME1, mod.STAGE0 = Path("logs/fmel2/ladder.json"), Path("logs/fmel1/ladder.json"), Path("logs/fme1/treat.json"), Path("logs/oma1/stage0.json")
        try:
            out = mod.assert_fmel2_provenance(fmel2, lock, stage0, fme1, fmel1, horizons=real_grid)
            assert out["fmel2_commit"] == "1a5ccb7d" and out["fmel2_regime"] == "INTERMEDIATE"
            with pytest.raises(AssertionError):
                mod.assert_fmel2_provenance(dict(fmel2, status="NOT-RUN"), lock, stage0, fme1, fmel1, horizons=real_grid)
        finally:
            mod.FMEL2, mod.FMEL1, mod.FME1, mod.STAGE0 = paths
    finally:
        mod.M_TREE = old
    # the comparison-vector file check is a pure sha read (MISSING / DRIFTED / OK), never a torch.load
    v = mod.verify_m_files({1: {"path": "checkpoints/nowhere/h0001.pt", "sha256": "0" * 64, "state_digest": "0" * 64}})
    assert v["1"]["status"] == "MISSING"
    v = mod.verify_m_files({1: {"path": "logs/fmel2/ladder.json", "sha256": mod.FMEL2_SHA, "state_digest": "x"}})
    assert v["1"]["status"] == "OK"
    v = mod.verify_m_files({1: {"path": "logs/fmel2/ladder.json", "sha256": "0" * 64, "state_digest": "x"}})
    assert v["1"]["status"] == "DRIFTED"
    src = (ROOT / "scratch" / "random_direction_control.py").read_text()
    consts = [n.value for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and isinstance(n.value, str)][1:]
    assert not any(("checkpoints/fme1/" in c) or ("checkpoints/oma1" in c) or ("checkpoints/fmel1" in c) for c in consts)
    assert not any(c.startswith("checkpoints/fmel2/A/") and c != "checkpoints/fmel2/A/e1e-1/" for c in consts), "only the e1e-1 comparison tree is ever named"


def test_k_map_is_the_exact_first_step_map(mod):
    """dw = K dm exactly (float64) for a real torch AdamW step from an identically seeded state; K independent of exp_avg."""
    torch.manual_seed(5); g1 = torch.randn(40, 30, dtype=torch.float64); g2 = torch.randn(50, dtype=torch.float64)
    mA, oA = _state(0, torch.float64); mB, oB = _state(0, torch.float64)
    c = np.concatenate([g1.reshape(-1).numpy(), g2.numpy()])
    K, m, zero_v = mod.k_map(oA, mA, c, SEGS, D, 7201, GRP)
    assert K.shape == (D,) and np.all(K < 0) and zero_v.sum() == 0
    q = np.random.default_rng(1).normal(size=D)
    dm, crec = mod.construct(q, K, m, SEGS)
    n, delta = mod.apply_dm(oB, mB, dm, SEGS, D)
    assert n == 2 and np.allclose(delta, dm)
    # K does not depend on exp_avg: recomputed on the perturbed state it is identical
    K2, m2, _ = mod.k_map(oB, mB, c, SEGS, D, 7201, GRP)
    assert np.array_equal(K, K2) and not np.array_equal(m, m2)
    for opt, mm in ((oA, mA), (oB, mB)):
        mm.a.grad = g1.clone(); mm.b.grad = g2.clone(); opt.step()
    dw = _flat(mB) - _flat(mA)
    assert np.linalg.norm(dw - K * dm) / np.linalg.norm(K * dm) < 1e-9
    # the untouched-state digest ignores exp_avg and changes with exp_avg_sq
    mC, oC = _state(0, torch.float64); mD, oD = _state(0, torch.float64)
    assert mod.untouched_digest(oC, mC, SEGS) == mod.untouched_digest(oD, mD, SEGS)
    mod.apply_dm(oD, mD, dm, SEGS, D)
    assert mod.untouched_digest(oC, mC, SEGS) == mod.untouched_digest(oD, mD, SEGS)
    with torch.no_grad():
        oD.state[mD.a]["exp_avg_sq"].mul_(2.0)
    assert mod.untouched_digest(oC, mC, SEGS) != mod.untouched_digest(oD, mD, SEGS)


def test_apply_dm_float32_one_rounding_and_realized_residual(mod):
    """The registered law exp_avg <- float32(float64(exp_avg) + dm_R): on a float32 state the realized delta is the float32 rounding
    of the intended vector (single rounding, |delta - dm| <= half an ulp of the new value), the residual is non-zero, and the
    realized first-step write still follows K delta to float32 write rounding."""
    torch.manual_seed(5); g1 = torch.randn(40, 30, dtype=torch.float32); g2 = torch.randn(50, dtype=torch.float32)
    mA, oA = _state(0, torch.float32); mB, oB = _state(0, torch.float32)
    c = np.concatenate([g1.double().reshape(-1).numpy(), g2.double().numpy()])
    K, m, _ = mod.k_map(oA, mA, c, SEGS, D, 7201, GRP)
    q = np.random.default_rng(1).normal(size=D)
    dm, _ = mod.construct(q, K, m, SEGS)
    old = np.concatenate([oB.state[mB.a]["exp_avg"].double().reshape(-1).numpy(), oB.state[mB.b]["exp_avg"].double().numpy()])
    n, delta = mod.apply_dm(oB, mB, dm, SEGS, D)
    new = np.concatenate([oB.state[mB.a]["exp_avg"].double().reshape(-1).numpy(), oB.state[mB.b]["exp_avg"].double().numpy()])
    assert n == 2 and oB.state[mB.a]["exp_avg"].dtype == torch.float32
    assert np.array_equal(new, (old + dm).astype(np.float32).astype(np.float64))            # exactly one float32 rounding of the float64 sum
    assert np.array_equal(delta, new - old) and not np.array_equal(delta, dm) and np.linalg.norm(delta - dm) / np.linalg.norm(dm) < 1e-6
    assert np.all(np.abs(delta - dm) <= np.spacing(np.abs(new).astype(np.float32)).astype(np.float64))
    for opt, mm in ((oA, mA), (oB, mB)):
        mm.a.grad = g1.clone(); mm.b.grad = g2.clone(); opt.step()
    dw = _flat(mB) - _flat(mA)
    assert np.linalg.norm(dw - K * delta) / np.linalg.norm(K * delta) < 0.05                 # float32 weight-write rounding on O(1) toy weights
    assert abs(mod.cosine(dw, K * delta)) > 0.999


def test_construction_law_matches_norm_shares_orthogonality_without_division(mod):
    rng = np.random.default_rng(2)
    K = -np.exp(rng.normal(size=D) * 3.0)                  # strongly coordinate-dependent, never zero
    m = rng.normal(size=D) * 1e-3
    q = rng.normal(size=D)
    dm, rec = mod.construct(q, K, m, SEGS)
    vM = K * (-0.1 * m)
    assert rec["undefined"] == {} and set(rec["groups"]) == {"BLOCK0", "OUTSIDE"}
    assert abs(np.linalg.norm(K * dm) / np.linalg.norm(vM) - 1) < 1e-12
    for g in rec["groups"]:
        assert abs(rec["groups"][g]["write_cos_to_M"]) < 1e-12
    assert abs(rec["analytic_write_cos_to_M_global"]) < 1e-12
    sh, shM = mod.group_shares(K * dm, SEGS), mod.group_shares(vM, SEGS)
    assert all(abs(sh[g] - shM[g]) < 1e-12 for g in shM)
    assert rec["dmR_over_dmM_total"] > 0 and set(rec["dmR_over_dmM_group"]) == {"BLOCK0", "OUTSIDE"}
    # division-free: zeros in K are harmless (those coordinates simply carry no write)
    K0 = K.copy(); K0[:200] = 0.0
    dm0, rec0 = mod.construct(q, K0, m, SEGS)
    assert dm0 is not None and np.all(np.isfinite(dm0)) and rec0["undefined"] == {} and abs(np.linalg.norm(K0 * dm0) / np.linalg.norm(K0 * (-0.1 * m)) - 1) < 1e-12
    # a group with a zero moment-axis write cannot be matched: UNDEFINED with a reason, never NaN, never a silently dropped group
    mz = m.copy(); mz[1200:] = 0.0
    dmz, recz = mod.construct(q, K, mz, SEGS)
    assert dmz is None and "OUTSIDE" in recz["undefined"] and "nan" not in json.dumps(recz).lower()
    # non-finite group target norm (K non-finite in a group) and non-finite projected write (q non-finite) both fail closed
    Kn = K.copy(); Kn[1200] = float("inf")
    dmn, recn = mod.construct(q, Kn, m, SEGS)
    assert dmn is None and "OUTSIDE" in recn["undefined"] and "nan" not in json.dumps(recn).lower()
    qn = q.copy(); qn[3] = float("nan")
    dmq, recq = mod.construct(qn, K, m, SEGS)
    assert dmq is None and "BLOCK0" in recq["undefined"] and "nan" not in json.dumps(recq).lower()
    # a random draw whose projected write is exactly zero in a group (q parallel to dm_M there) fails closed
    qz = q.copy(); qz[1200:] = (-0.1 * m)[1200:]
    dmp, recp = mod.construct(qz, K, m, SEGS)
    assert dmp is None and "OUTSIDE" in recp["undefined"]
    # seed reproducibility and independence of the frozen draws
    a, b, c2 = mod.draw(2026091501, 5000), mod.draw(2026091501, 5000), mod.draw(2026091502, 5000)
    assert np.array_equal(a, b) and not np.array_equal(a, c2) and a.dtype == np.float64 and mod.flat_digest(a) == mod.flat_digest(b)
    src = inspect.getsource(mod.construct)
    assert "/ K" not in src and "/ Kg" not in src and "1.0 / K" not in src
    qs = mod.dm_quantiles(dm)
    assert qs["max"] >= qs["q999"] >= qs["q99"] >= qs["q90"] >= qs["q50"] >= 0 and abs(qs["norm"] - np.linalg.norm(dm)) < 1e-12
    st = mod.k_stats(K0, K0 == 0.0, SEGS)
    assert st["BLOCK0"]["zero_v"] == 200 and 1 < st["BLOCK0"]["n_eff"] <= 1200 and st["OUTSIDE"]["n"] == 50


def test_preflight_law_and_refusals(mod):
    rng = np.random.default_rng(3)
    devM = rng.normal(size=D); devM[1200:] *= 0.05
    sharesM = mod.group_shares(devM, SEGS)

    def ortho(v):
        u = devM / np.linalg.norm(devM)
        v = v - (v @ u) * u
        return v * (np.linalg.norm(devM) / np.linalg.norm(v))

    def shaped():
        v = rng.normal(size=D); v[1200:] *= 0.05
        return ortho(v)

    dev = {"R1": shaped(), "R2": shaped(), "R3": shaped()}
    # shapes are only approximately matched by this toy: relax the share check by building shares from devM itself
    shares = {a: sharesM for a in dev}
    ok, det = mod.preflight_law(dev, devM, shares, sharesM)
    assert ok, det["fails"]
    assert all(abs(det["magnitude"][a] - 1) < 1e-12 for a in dev) and all(abs(det["cos_to_M"][a]) < 1e-12 for a in dev)
    assert det["target_norm"] == np.linalg.norm(devM) and len(det["pair_cos"]) == 3
    bad = dict(dev); bad["R2"] = dev["R2"] * 1.002
    assert not mod.preflight_law(bad, devM, shares, sharesM)[0]
    bad = dict(dev); bad["R3"] = (dev["R3"] + 0.2 * devM) * (np.linalg.norm(devM) / np.linalg.norm(dev["R3"] + 0.2 * devM))
    ok3, det3 = mod.preflight_law(bad, devM, shares, sharesM)
    assert not ok3 and any(f.startswith("orthogonality R3") for f in det3["fails"])
    sh = dict(shares); sh["R1"] = {"BLOCK0": sharesM["BLOCK0"] - 0.011, "OUTSIDE": sharesM["OUTSIDE"] + 0.011}
    ok4, det4 = mod.preflight_law(dev, devM, sh, sharesM)
    assert not ok4 and any(f.startswith("locus R1") for f in det4["fails"])
    dup = dict(dev); dup["R2"] = dev["R1"]
    ok5, det5 = mod.preflight_law(dup, devM, shares, sharesM)
    assert not ok5 and any(f.startswith("independence R1|R2") for f in det5["fails"])
    assert not mod.preflight_law(dev, np.zeros(D), shares, sharesM)[0]
    dead = dict(dev); dead["R1"] = np.zeros(D)
    assert not mod.preflight_law(dead, devM, shares, sharesM)[0]
    assert (mod.MAG_TOL, mod.COS_M_MAX, mod.SHARE_TOL, mod.COS_PAIR_MAX) == (1e-3, 0.05, 0.01, 0.05)
    D_ = mod.preflight_disposition
    assert D_(True, False, True) is None
    assert D_(True, False, False)[0] == "NOT-RUN" and "substrate change" in D_(True, False, False)[2]
    st, reg, lab = D_(False, False, True)
    assert (st, reg) == ("PREFLIGHT-FAILED", "CONSTRUCTION-UNRESOLVED") and lab.startswith("CONSTRUCTION-UNRESOLVED+DIRECTION-NOT-MEASURED+FUNCTION-NOT-MEASURED [writer A only")
    assert D_(False, True, True) is None and D_(False, True, False)[0] == "NOT-RUN"      # the SMOKE-only bypass never overrides the digest law
    assert mod.PREFLIGHT_BYPASS is False


def test_growth_and_zero_norm_law(mod):
    rng = np.random.default_rng(4)
    W0 = rng.normal(size=60); WC = W0 + rng.normal(size=60)
    u = rng.normal(size=60); u /= np.linalg.norm(u); v = rng.normal(size=60); v -= (v @ u) * u; v /= np.linalg.norm(v)
    ms = {}
    for h, (gr, gm) in {1: (1.0, 1.0), 5: (4.0, 4.0), 900: (150.0, 157.0), 8220: (600.0, 517.0)}.items():
        W = {"R1": WC + 0.01 * gr * v, "R2": WC + 0.01 * gr * (0.6 * v + 0.8 * u), "R3": WC + 0.01 * gr * u}
        ms[h] = mod.horizon_metrics(W0, WC, W, WC + 0.01 * gm * u)
    mod.growth(ms)
    assert abs(ms[8220]["G_M"] - 517.0) < 1e-9 and abs(ms[8220]["G"]["R1"] - 600.0) < 1e-9 and abs(ms[8220]["A"]["R1"] - 600.0 / 517.0) < 1e-9
    assert abs(ms[8220]["cos_M"]["R1"]) < 1e-12 and abs(ms[8220]["cos_M"]["R3"] - 1) < 1e-12 and abs(ms[8220]["cos_M"]["R2"] - 0.8) < 1e-12
    assert ms[8220]["undefined"] == {} and ms[900]["n_M"] > 0 and len(ms[1]["pair_cos"]) == 3
    assert mod.h_amp({h: ms[h]["G"]["R1"] for h in ms}) == 900 and mod.h_amp({h: ms[h]["G_M"] for h in ms}) == 900 and mod.h_amp({1: 1.0, 5: 4.0}) is None
    # zero h = 1 deviation: G, A UNDEFINED with reasons; non-finite: everything UNDEFINED; never NaN
    z = dict(ms); z[1] = mod.horizon_metrics(W0, WC, {"R1": WC.copy(), "R2": WC + 0.01 * u, "R3": WC + 0.01 * v}, WC + 0.01 * u)
    mod.growth(z)
    assert z[8220]["G"]["R1"] is None and z[8220]["A"]["R1"] is None and "G:R1" in z[8220]["undefined"] and "A:R1" in z[8220]["undefined"]
    assert "nan" not in json.dumps(z[8220]).lower() and mod.regime(z[8220]["A"], True) == "REGIME-UNRESOLVED"
    Wn = {"R1": WC + 0.01 * v, "R2": WC + 0.01 * u, "R3": WC + 0.01 * v}; Wn["R2"] = Wn["R2"].copy(); Wn["R2"][0] = float("nan")
    nf = mod.horizon_metrics(W0, WC, Wn, WC + 0.01 * u)
    assert "nonfinite" in nf["undefined"] and all(nf["n"][a] is None for a in nf["n"]) and "nan" not in json.dumps(nf).lower()
    zc = mod.horizon_metrics(W0, W0, {"R1": W0 + v, "R2": W0 + u, "R3": W0 + v}, W0 + u)
    assert all(zc["n"][a] is None for a in zc["n"]) and zc["n_M"] is None


def _A(r1, r2, r3):
    return {"R1": r1, "R2": r2, "R3": r3}


def test_bar2_bar3_bar4_bands_and_boundaries(mod):
    R = mod.regime
    assert R(_A(1.0, 0.5, 2.0), True) == "DIRECTION-GENERIC"
    assert R(_A(0.25, 0.1, 0.2), True) == "MOMENT-SPECIFIC"
    assert R(_A(4.0, 10.0, 5.0), True) == "RANDOM-DOMINANT"
    assert R(_A(1.0, 0.49, 1.0), True) == "MIXED" and R(_A(1.0, 2.01, 1.0), True) == "MIXED" and R(_A(0.25, 0.26, 0.1), True) == "MIXED"
    assert R(_A(3.99, 4.0, 4.0), True) == "MIXED" and R(_A(0.3, 0.3, 0.3), True) == "MIXED"
    assert R(_A(1.0, 1.0, 1.0), False) == "REGIME-UNRESOLVED" and R(_A(1.0, None, 1.0), True) == "REGIME-UNRESOLVED" and R(_A(1.0, float("nan"), 1.0), True) == "REGIME-UNRESOLVED"
    assert R({"R1": 1.0, "R2": 1.0}, True) == "REGIME-UNRESOLVED"                       # a missing arm never passes silently
    assert (mod.GENERIC_BAND, mod.SPECIFIC_MAX, mod.DOMINANT_MIN) == ((0.5, 2.0), 0.25, 4.0)
    Dn = mod.direction
    assert Dn(_A(0.5, -0.9, 0.7)) == "DIRECTION-SHARED-LATE" and Dn(_A(0.25, -0.1, 0.0)) == "DIRECTION-INDEPENDENT-LATE"
    assert Dn(_A(0.3, 0.1, 0.0)) == "MIXED-DIRECTION" and Dn(_A(0.5, 0.2, 0.6)) == "MIXED-DIRECTION" and Dn(_A(0.5, None, 0.6)) == "DIRECTION-LATE-UNRESOLVED" and Dn(_A(0.5, float("nan"), 0.6)) == "DIRECTION-LATE-UNRESOLVED"
    assert (mod.SHARED_LATE, mod.INDEP_LATE, mod.H_AMP_G) == (0.50, 0.25, 10.0)
    per, lab = mod.function_bar({"R1": 0.004, "R2": -0.005, "R3": 0.0})
    assert lab == "FUNCTION-NEUTRAL" and all(v == "NEUTRAL" for v in per.values())
    per, lab = mod.function_bar({"R1": 0.0051, "R2": -0.02, "R3": 0.0})
    assert per["R1"] == "HARMED" and per["R2"] == "HELPED" and lab == "FUNCTION-R1:HARMED,R2:HELPED" and mod.CE_ABS == 0.005
    assert mod.label("MIXED", "MIXED-DIRECTION", "FUNCTION-NEUTRAL").startswith("MIXED+MIXED-DIRECTION+FUNCTION-NEUTRAL [writer A only")


def test_leg_path_copy_is_l1_long_leg_with_the_intervention_replaced(mod):
    """long_leg_r is L1.long_leg verbatim except the declared substitutions (intervention call, C qualification at every grid
    horizon, the stream / cell fields); one_step_r shares the bind / resume / run_leg mechanics."""
    src_l1 = inspect.getsource(mod.L1.long_leg)
    src_r = inspect.getsource(mod.long_leg_r)
    subs = [
        ("def long_leg(arm, eps, tok, enc, slices, segs, d, held, refs, preflight_digest, stream, cell, mid):", "def long_leg_r(arm, dm, tok, enc, slices, segs, d, held, refs, preflight_digest, stream, cell, mid):"),
        ("    touched = apply_eps(opt, eps)\n", "    touched, delta = apply_dm(opt, model, dm, segs, d)\n"),
        ("    assert touched == (0 if eps == 0.0 else len(UG.KEYS)), (arm, touched)\n", "    assert touched == (0 if dm is None else len(UG.KEYS)), (arm, touched)\n"),
        ('    cell.update({"arm": arm, "eps": eps, "bind": binfo,', '    cell.update({"arm": arm, "seed": SEEDS.get(arm), "delta_digest": (None if delta is None else flat_digest(delta)), "bind": binfo,'),
        ("    check_at = set(QUAL) | {1} |", "    check_at = set(GRID) | {1} |"),
        ('        stream({"writer": WRITER, "arm": arm, "eps": eps, "device": "cpu",', '        stream({"writer": WRITER, "arm": arm, "seed": SEEDS.get(arm), "device": "cpu",'),
        ('                dm = Wm - mid["W"]\n', '                dmid = Wm - mid["W"]\n'),
        ('"rho": float(np.linalg.norm(dm) / max(', '"rho": float(np.linalg.norm(dmid) / max('),
        ('"abs": float(np.linalg.norm(dm)), "ce_fresh"', '"abs": float(np.linalg.norm(dmid)), "ce_fresh"'),
    ]
    body_l1 = src_l1.split('"""')[2]; body_r = src_r.split('"""')[2]          # strip the docstrings; the signature line precedes them
    head_l1 = src_l1.split('"""')[0]; head_r = src_r.split('"""')[0]
    mapped_head, mapped_body = head_l1, body_l1
    for old, new in subs:
        n = (mapped_head + mapped_body).count(old)
        assert n == 1, (old, n)
        mapped_head = mapped_head.replace(old, new); mapped_body = mapped_body.replace(old, new)
    assert mapped_head == head_r, "long_leg_r signature drifted from L1.long_leg"
    assert mapped_body == body_r, "long_leg_r body drifted from L1.long_leg beyond the declared substitutions"
    # the wall-cap mechanism is L2's with this rung's names only
    for fn in ("install_wall_limit", "wall_record"):
        assert inspect.getsource(getattr(mod, fn)) == inspect.getsource(getattr(mod.L2, fn))
    assert inspect.getsource(mod.wall_label) == inspect.getsource(mod.L2.wall_label).replace("L75924", "L76553")
    assert inspect.getsource(mod.set_phase) == inspect.getsource(mod.L2.set_phase).replace("[fmel2]", "[rdc1]")
    assert mod.qualify is mod.L1.qualify and mod.GRID is mod.L1.GRID and mod.function_bar is mod.L2.function_bar and mod.cosine is mod.L1.cosine


def test_arena_literals_no_b_path_and_receipt_reads(mod):
    assert mod.ARENA["grid"] == [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220] and mod.ARENA["leg"] == 8220
    assert mod.N_PRED == 0.894284652673395 and mod.WRITER == "A" and mod.MIN_FREE_BYTES == 16 * (1 << 30) and mod.M_ARM == "e1e-1" and mod.M_EPS == 0.1
    src = (ROOT / "scratch" / "random_direction_control.py").read_text()
    assert "program_label" not in src and "backsched" not in src and "float64_master" not in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "WRITERS":
            assert isinstance(node.slice, ast.Name) and node.slice.id == "WRITER", ast.dump(node)
        if isinstance(node, ast.Constant) and node.value == "B":
            raise AssertionError("string constant 'B' in the single-writer instrument")
    # the display literal appears once, in the preflight record under its display name, never in a comparison
    assert src.count("0.012408559997086078") == 1 and '"norm_M_display_literal": 0.012408559997086078' in src
    for lit in ("0.1117", "0.1277", "0.1339", "0.0016"):
        assert lit not in src, f"rounded share literal {lit} must not appear in the instrument"


def test_real_mode_arena_constants():
    import subprocess
    code = ("import os,sys,json; os.environ.pop('SMOKE', None); os.environ.pop('SMOKE_PREFLIGHT_BYPASS', None); sys.path[:0]=['.','scripts','scratch']; "
            "import random_direction_control as R; print(json.dumps([R.LEG_FULL, R.ANCHOR + R.LEG_FULL, R.GRID, R.H_END, R.TAIL_H, R.PREFLIGHT_BYPASS, "
            "str(R.CK_DIR), str(R.RECEIPT), str(R.STREAM), str(R.FMEL2), str(R.FMEL1), str(R.FME1), str(R.STAGE0), R.M_TREE, str(R.END_MODEL), R.END_SHA, R.MID_STEP, R.MID_SHA, "
            "R.MAX_WALL_S, R.MAX_WALL_S_REAL, R.SEEDS, R.ARM_ORDER]))")
    env = dict(os.environ, SMOKE_MAX_WALL_S="5", RDC1_MAX_WALL_S="5", MAX_WALL_S="5", SMOKE_TAMPER_REF="1", SMOKE_TAMPER_M="1", SMOKE_TAMPER_CONSTRUCTION="1")
    env.pop("SMOKE", None)
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    v = json.loads(out.stdout.strip().splitlines()[-1])
    assert v[:5] == [8220, 15420, [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220], 8220, 6000] and v[5] is False
    assert v[6:14] == ["checkpoints/rdc1", "logs/rdc1/control.json", "logs/rdc1/control.jsonl", "logs/fmel2/ladder.json", "logs/fmel1/ladder.json", "logs/fme1/treat.json", "logs/oma1/stage0.json", "checkpoints/fmel2/A/e1e-1/"]
    assert v[14:18] == ["checkpoints/gallery19m_phase_s2.pt", "e7207b3bd4df541a", 15300, "6a715beb6394e3b3"]
    assert v[18:20] == [25200, 25200]
    assert v[20] == {"R1": 2026091501, "R2": 2026091502, "R3": 2026091503} and v[21] == ["C", "R1", "R2", "R3"]
    # the SMOKE-only tamper knobs are read only under SMOKE: the source guards every read with `SMOKE and`
    src = (ROOT / "scratch" / "random_direction_control.py").read_text()
    for knob in ("SMOKE_TAMPER_REF", "SMOKE_TAMPER_M", "SMOKE_TAMPER_CONSTRUCTION", "SMOKE_PREFLIGHT_BYPASS", "SMOKE_MAX_WALL_S"):
        for line in [ln for ln in src.splitlines() if f'os.environ.get("{knob}"' in ln or f'os.environ["{knob}"]' in ln]:
            assert "SMOKE and" in line or "(SMOKE and" in line, (knob, line)


def test_refuse_if_exists_and_receipt_writer(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "use_deterministic_algorithms", lambda *a, **k: None)
    monkeypatch.setattr(torch, "set_num_threads", lambda *a, **k: None)
    monkeypatch.setattr(mod, "RECEIPT", tmp_path / "control.json"); monkeypatch.setattr(mod, "STREAM", tmp_path / "control.jsonl")
    monkeypatch.setattr(mod, "OUT_DIR", tmp_path); monkeypatch.setattr(mod, "CK_DIR", tmp_path / "ck")
    (tmp_path / "control.json").write_text("{}")
    with pytest.raises(SystemExit):
        mod.main()
    (tmp_path / "control.json").unlink()
    (tmp_path / "ck" / "A" / "R1").mkdir(parents=True); (tmp_path / "ck" / "A" / "R1" / "h0001.pt").write_bytes(b"x")
    with pytest.raises(SystemExit):
        mod.main()
    import time as _t
    r = {"status": "RUNNING"}
    mod._write_receipt(r, _t.time())
    assert json.loads((tmp_path / "control.json").read_text())["status"] == "RUNNING" and "wall_s" in r and "ended_utc" in r


def test_wall_limit_fires_during_active_work_and_is_fixed_in_real_mode(mod, tmp_path):
    import signal
    import subprocess
    import time as _t
    assert mod.MAX_WALL_S_REAL == 25200 and isinstance(mod.WallLimit(3), SystemExit) and not isinstance(mod.WallLimit(3), mod.Abort)
    saved = dict(mod.PHASE); saved_grace = mod.WALL_GRACE_S
    mod.WALL_GRACE_S = 3600
    mod.PHASE["name"] = "leg:R2"; mod.PHASE["t0"] = _t.time()
    rec = {}
    armed = mod.install_wall_limit(1, tmp_path / "r.json", rec)
    assert armed == 1
    t0 = _t.time()
    try:
        with pytest.raises(mod.WallLimit):
            while _t.time() - t0 < 10:
                sum(range(10000))
        assert _t.time() - t0 < 5
        lab = mod.wall_label()
        assert lab.startswith("NOT-RUN (registered wall limit") and "leg:R2" in lab and "L76553" in lab
        wr = mod.wall_record()
        assert wr["real_limit_s"] == 25200 and wr["phase"] == "leg:R2" and wr["hard_exit"] is False
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, signal.SIG_DFL)
        mod.PHASE.clear(); mod.PHASE.update(saved); mod.WALL_GRACE_S = saved_grace
    code = ("import os,sys,json; os.environ['SMOKE']='1'; os.environ['SMOKE_MAX_WALL_S']='7'; sys.path[:0]=['.','scripts','scratch']; "
            "import random_direction_control as R; print(json.dumps([R.MAX_WALL_S, R.MAX_WALL_S_REAL, R.SMOKE]))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == [7, 25200, True]

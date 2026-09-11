"""SG-FAILURE-DESK-0 label law (scratch/sg_failure_labels.py labels()): the
sealed thresholds and the four branches on a synthetic desk record, pure."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("sg_failure_labels", ROOT / "scratch" / "sg_failure_labels.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert (m.GOOD, m.POOR, m.ONLINE_BAD, m.TARGET_TOL, m.RANK_TOL, m.READOUT_LO, m.READOUT_HI) == (0.5, 0.9, 0.9, 1e-4, 3.0, 0.5, 2.0)
    return m


def _state(lin, rf, online=None, base=1.0, rank=20.0, head=1.0, normg=1.0, pg=0.0, prev=None, nxt=None, step=None):
    B = ["4", "5", "6", "7"]
    s = {"oracle": {l: {"linear_lstsq": {"held_ratio": lin}, "linear_ridge_gcv": {"held_ratio": lin}, "rf_ridge_gcv": {"held_ratio": rf, "chosen_rel": 1e-6}} for l in B},
         "baseline_mse_held": {l: base for l in B}, "delta_rms_held": {l: 1e-6 for l in B}, "rank_held": {str(i): rank for i in range(8)},
         "head_weight_fro": head, "norm_g_l2": normg, "probe_ce_held": 1.0}
    if online is not None:
        s["online_match"] = {l: {"ratio": online, "cos": 0.1} for l in B}
        s["param_grad_cos"] = {l: {"cos": pg} for l in B}
        s["hat_over_delta_rms"] = {l: 0.5 for l in B}
        if prev is not None:
            s["online_prev"] = {l: {"ratio": prev} for l in B}
            s["prev_step"] = step - 565
        if nxt is not None:
            s["online_next"] = {l: {"ratio": nxt} for l in B}
            s["next_step"] = step + 1028
            s["target_nonstationarity_next"] = {l: {"cos": 0.3} for l in B}
    return s


def _rec(sg_lin, sg_rf, online, **kw):
    steps = ["0", "463", "1028", "2056"]
    control = {"mode": "zero", "states": {s: _state(0.3, 0.2) for s in steps}}
    sg = {"mode": "sg", "family": "linear", "plr": 3e-4, "states": {s: _state(sg_lin, sg_rf, online, prev=1.0, nxt=1.0, step=int(s), **kw) for s in steps[1:]}}
    return {"cells": {"ctrl": control, "sgcell": sg}}


def test_branch_a(mod):
    out = mod.labels(_rec(0.3, 0.2, online=1.0))
    r = out["cells"]["sgcell"]
    assert r["linear_good"] and r["online_bad"] and r["branch_A"] and not r["branch_B"] and not r["branch_C"] and not r["branch_D"]
    assert out["cells"]["ctrl"]["control_collapse_none"]


def test_branch_b_uses_min_of_linear_and_richer(mod):
    r = mod.labels(_rec(0.8, 0.3, online=1.0))["cells"]["sgcell"]
    assert not r["linear_good"] and r["richer_good"] and r["branch_B"]
    assert r["richer_median"] == 0.3


def test_branch_c(mod):
    r = mod.labels(_rec(0.95, 1.2, online=1.0))["cells"]["sgcell"]
    assert r["linear_poor"] and r["richer_poor"] and r["branch_C"] and not r["branch_B"]
    assert r["richer_median"] == 0.95     # min(linear, richer)


def test_branch_d_sg_only_collapse_and_coincidence(mod):
    r = mod.labels(_rec(0.3, 0.2, online=1.0, base=1e-6, rank=2.0, head=3.0))["cells"]["sgcell"]
    assert r["branch_D"] and r["sg_only_collapse"]["target"] and r["sg_only_collapse"]["rank"] and r["collapse_readout"]
    co = r["coincidence_at_first_target_fall"]
    assert co["step"] == "463" and co["head_fro_v_control"] == 3.0 and co["rank_4_7"]["4"] == 2.0


def test_online_lags_and_param_grad(mod):
    r = mod.labels(_rec(0.3, 0.2, online=0.5, pg=0.15))["cells"]["sgcell"]
    assert not r["online_bad"] and r["online_prev_per_step"]["463"]["lag"] == 565 and r["online_next_per_step"]["463"]["lag"] == 1028
    assert r["param_grad_cos_within_0.2_from_1028"] is True and r["param_grad_cos_range"] == [0.15, 0.15]


def test_not_resolvable_rank_is_flagged_never_counted(mod):
    rec = _rec(0.3, 0.2, online=1.0, base=1e-6)
    for s in rec["cells"]["sgcell"]["states"].values():
        s["rank_held"]["5"] = "negative eigenvalue -1.0 below tolerance"
        s["rank_held"]["6"] = 1.5
    r = mod.labels(rec)["cells"]["sgcell"]
    assert len(r["rank_not_resolvable"]) == 3 and all(x[1] == "5" for x in r["rank_not_resolvable"])
    assert all(x[1] == "6" for x in r["collapse_rank"]) and len(r["collapse_rank"]) == 3
    assert r["coincidence_at_first_target_fall"]["rank_4_7"]["5"].startswith("negative eigenvalue")

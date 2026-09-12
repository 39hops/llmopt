"""MEZO-SIGNAL-DESK-0 instrument guards: perturbations are +-1 valued and
deterministic in (family, seed); rank1 perturbations are rank-1 per 2-D
tensor and Rademacher on 1-D tensors; effective dimensions; d_BP = <g, z>;
the Evaluator restores theta after every loss call and its finite
difference at tiny eps matches <g, z> on a small model; the adjudication
law (FD precondition, per-family BAR-SIGNAL at the largest practical m,
verdict order MEZO / STRUCTURED-ZO / PARK / NOT-RESOLVABLE-FD)."""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    prev = os.environ.get("SMOKE")
    os.environ["SMOKE"] = "1"
    for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scratch")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        m = importlib.import_module("mezo_signal_desk")
    finally:
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


@pytest.fixture(scope="module")
def small():
    import torch
    torch.manual_seed(0)
    return [torch.nn.Parameter(torch.randn(3, 5)), torch.nn.Parameter(torch.randn(4)), torch.nn.Parameter(torch.randn(2, 2))]


def test_perturbations_pm1_deterministic_and_rank1(mod, small):
    import torch
    for fam in ("vanilla", "rank1"):
        z1 = mod.perturbation(fam, small, 7)
        z2 = mod.perturbation(fam, small, 7)
        z3 = mod.perturbation(fam, small, 8)
        assert torch.equal(z1, z2) and not torch.equal(z1, z3)
        assert z1.numel() == 15 + 4 + 4 and set(z1.unique().tolist()) == {-1.0, 1.0}
    z = mod.perturbation("rank1", small, 3)
    W = z[:15].view(3, 5)
    assert torch.linalg.matrix_rank(W) == 1
    assert torch.linalg.matrix_rank(z[19:23].view(2, 2)) == 1
    assert mod.effective_dim("vanilla", small) == 23 and mod.effective_dim("rank1", small) == (3 + 5) + 4 + (2 + 2)


def test_evaluator_restores_theta_and_fd_matches_grad(mod):
    import torch
    from llmopt.train.mathnative import build_model
    from dfa_credit import freeze_lower
    torch.manual_seed(1)
    model = build_model(40, d=48, layers=8, heads=6, ffn=96).double()
    freeze_lower(model, 4)
    ids = torch.randint(1, 40, (2, 6))
    mask = torch.ones(2, 6, dtype=torch.long)
    labels = ids.clone()
    ev = mod.Evaluator(model, ids, mask, labels)
    base, g = ev.grad()
    z = mod.perturbation("vanilla", ev.params, 11).double()
    eps = 1e-6
    d_fd = (ev.loss(eps * z) - ev.loss(-eps * z)) / (2 * eps)
    d_bp = float((g * z).sum())
    assert abs(d_fd - d_bp) <= 1e-4 * max(1.0, abs(d_bp)), (d_fd, d_bp)
    assert torch.equal(mod.parameters_to_vector(ev.params).detach(), ev.theta)
    assert abs(ev.loss() - base) < 1e-9


def _rec(mod, R_van, R_r1, sign=1.0, rel=0.01):
    prim = f"{mod.EPS[0]:g}"
    def fam(R):
        return {"d_bp": [1.0, -1.0], "d_fd": {prim: [1.0 + rel, -1.0 + rel] if sign == 1.0 else [-1.0, 1.0]},
                "fidelity": {prim: {"sign_agreement": sign, "n": 2}},
                "ideal_cos": {str(m): [0.1] for m in mod.M_ALL}, "actual_cos": {str(m): [0.1] for m in mod.M_ALL},
                "ref_cos_sqrt_m_over_d": {str(m): 0.01 for m in mod.M_ALL},
                "virtual": {k: {str(m): [{"D": -R * 0.5, "R": R}] for m in mod.M_PRACTICAL} for k in ("ideal", "actual")}}
    return {"state_batches": [{"families": {"vanilla": fam(R_van), "rank1": fam(R_r1)}}]}


def test_adjudicate_verdicts(mod):
    assert mod.adjudicate(_rec(mod, 0.5, 0.5))["verdict"] == "MEZO-LICENSED"
    assert mod.adjudicate(_rec(mod, 0.1, 0.0))["verdict"] == "MEZO-LICENSED"          # inclusive at R_LICENSE
    assert mod.adjudicate(_rec(mod, 0.01, 0.2))["verdict"] == "STRUCTURED-ZO-LICENSED"
    a = mod.adjudicate(_rec(mod, 0.01, 0.05))
    assert a["verdict"] == "PARK" and a["fd_faithful"] and a["families"]["vanilla"]["bar_signal"] == "NO-FIRE"
    assert mod.adjudicate(_rec(mod, 0.5, 0.5, sign=0.0))["verdict"] == "NOT-RESOLVABLE-FD"
    assert mod.adjudicate(_rec(mod, 0.5, 0.5, rel=0.2))["verdict"] == "NOT-RESOLVABLE-FD"


def test_fd_precondition_pools_by_n_over_cells(mod):
    prim = f"{mod.EPS[0]:g}"
    def fam(n_ok, n_bad):
        d_bp = [1.0] * (n_ok + n_bad)
        d_fd = [1.0] * n_ok + [-1.0] * n_bad
        return {"d_bp": d_bp, "d_fd": {prim: d_fd}, "fidelity": {prim: {"sign_agreement": n_ok / (n_ok + n_bad), "n": n_ok + n_bad}},
                "ideal_cos": {str(m): [0.1] for m in mod.M_ALL}, "actual_cos": {str(m): [0.1] for m in mod.M_ALL},
                "ref_cos_sqrt_m_over_d": {str(m): 0.01 for m in mod.M_ALL},
                "virtual": {k: {str(m): [{"D": -0.1, "R": 0.5}] for m in mod.M_PRACTICAL} for k in ("ideal", "actual")}}
    # cell A: 10 directions all right; cell B: 2 directions, 1 wrong -> pooled by n = 11 / 12 = 0.917 (< 0.95), per-cell mean would be 0.75
    rec = {"state_batches": [{"families": {"vanilla": fam(10, 0), "rank1": fam(10, 0)}}, {"families": {"vanilla": fam(1, 1), "rank1": fam(2, 0)}}]}
    a = mod.adjudicate(rec)
    assert abs(a["fd_sign_agreement_pooled"] - 11 / 12) < 1e-12 and not a["fd_faithful"] and a["verdict"] == "NOT-RESOLVABLE-FD"
    assert abs(a["fd_by_eps"]["vanilla"][prim]["sign_agreement_pooled"] - 11 / 12) < 1e-12

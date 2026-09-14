"""OPTIMIZER-GEOMETRY-DESK-0 guards: the virtual AdamW law reproduces a real
torch AdamW.step on synthetic tensors (float64, exact to 1e-9 of the update
scale) including the decoupled decay, bias corrections with the incremented
step and clipping; b = u - u0 = a - a0; the clip law equals
clip_grad_norm_; the distortion readout is 0 on identical Grams and the
cosine Gram is unit-diagonal; the ladder is pure, gives one label and is
inclusive at the literal thresholds; the registered constants are literal."""
import importlib
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
    try:
        m = importlib.import_module("optimizer_geometry_desk")
    finally:
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


def _adamw_reference(shapes, steps_before, lr, b1, b2, wd, seed=0):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(s, dtype=torch.float64)) for s in shapes]
    opt = torch.optim.AdamW(ps, lr=3e-4, weight_decay=wd)
    for _ in range(steps_before):
        for p in ps:
            p.grad = torch.randn_like(p)
        opt.step()
    pre = [p.detach().clone() for p in ps]
    m = [opt.state[p]["exp_avg"].clone() if p in opt.state else torch.zeros_like(p) for p in ps]
    v = [opt.state[p]["exp_avg_sq"].clone() if p in opt.state else torch.zeros_like(p) for p in ps]
    grads = [torch.randn_like(p) * 3 for p in ps]                      # norm > 1: clipping active
    for p, g in zip(ps, grads):
        p.grad = g.clone()
    tot = float(torch.sqrt(sum((g ** 2).sum() for g in grads)))
    torch.nn.utils.clip_grad_norm_(ps, 1.0)
    opt.param_groups[0]["lr"] = lr; opt.param_groups[0]["betas"] = (b1, b2)
    opt.step()
    return ps, pre, m, v, grads, tot, opt


@pytest.mark.parametrize("steps_before,lr,b1", [(0, 1.2e-5, 0.95), (4, 2.99e-4, 0.8501), (900, 1e-4, 0.9)])
def test_virtual_law_matches_torch_adamw(mod, steps_before, lr, b1):
    ps, pre, m, v, grads, tot, opt = _adamw_reference([(5, 7), (11,), (3, 4, 2)], steps_before, lr, b1, 0.999, 0.01)
    coef = mod.clip_coef(tot)
    assert abs(coef - min(1.0, 1.0 / (tot + 1e-6))) < 1e-15
    for i, p in enumerate(ps):
        W = pre[i].numpy().reshape(-1); g = grads[i].numpy().reshape(-1)[None, :] * coef
        fam = mod.virtual_families(g, W, m[i].numpy().reshape(-1), v[i].numpy().reshape(-1), steps_before + 1, lr, b1, 0.999, 0.01)
        real_u = p.detach().numpy().reshape(-1) - W
        assert mod.scale_err(real_u - fam["u"][0], fam["u"][0]) <= 1e-9
        assert mod.scale_err(opt.state[p]["exp_avg"].numpy().reshape(-1) - (b1 * m[i].numpy().reshape(-1) + (1 - b1) * g[0]), fam["mhat"][0]) <= 1e-9
        assert np.allclose(fam["b"], fam["a"] - fam["a0"][None, :]) and np.allclose(fam["u"] - fam["u0"][None, :], fam["b"])


def test_distortion_and_cosine_gram(mod):
    rng = np.random.default_rng(0); X = rng.standard_normal((6, 40)); K = X @ X.T
    C = mod.cosine_gram(K)
    assert np.allclose(np.diag(C), 1.0) and mod.distortion(K, 3.0 * K) < 1e-12 and mod.distortion(K, K) == 0.0
    Y = rng.standard_normal((6, 40)); assert mod.distortion(K, Y @ Y.T) > 0


def _geo(c8, rel):
    return {"raw": {"held_capture": {8: {"median": c8}}, "reliability_fit_held": {8: rel}}}


def _cells(g, u, b, rel=0.5, relb=0.5):
    out = {}
    for s in (900, 7200, 13500):
        for spec in ("A", "B"):
            out[(spec, s)] = {"g": _geo(g, rel), "u": _geo(u, rel), "b": _geo(b, relb)}
    return out


def test_ladder_labels(mod):
    mod.K_LADDER = 8; mod.STATES = [900, 7200, 13500]
    pairs = {"u": {"raw": {8: 0.5}}, "b": {"raw": {8: 0.1}}}
    assert mod.adjudicate(_cells(0.25, 0.40, 0.45), pairs)["sharpening"] == "BATCH-WRITE-SHARPENED"      # b rises 0.20 (inclusive at 0.15 tested next)
    assert mod.adjudicate(_cells(0.25, 0.40, 0.40), pairs)["sharpening"] == "BATCH-WRITE-SHARPENED"
    ax = mod.adjudicate(_cells(0.25, 0.40, 0.28), pairs)
    assert ax["sharpening"] == "HISTORY-DOMINATED" and ax["writer"]["u"]["label"] == "WRITER-UPDATE-SHARED" and ax["writer"]["b"]["label"] == "WRITER-UPDATE-ROTATED"
    assert mod.adjudicate(_cells(0.25, 0.40, 0.33), pairs)["sharpening"] == "WRITE-SHARPENED"           # b rises 0.08: neither < 0.05 nor >= 0.15
    assert mod.adjudicate(_cells(0.25, 0.30, 0.05), pairs)["sharpening"] == "OPTIMIZER-DIFFUSES"        # b falls 0.20
    assert mod.adjudicate(_cells(0.25, 0.30, 0.20), pairs)["sharpening"] == "NO-OPTIMIZER-SHARPENING"
    assert mod.adjudicate(_cells(0.25, 0.40, 0.45, rel=0.2), pairs)["resolution"] == "GEOMETRY-NOT-RESOLVED"
    ax = mod.adjudicate(_cells(0.25, 0.40, 0.45, relb=0.1), pairs)                                         # an unresolved b panel cannot fire BATCH-WRITE-SHARPENED and is reported as its own axis
    assert ax["sharpening"] == "WRITE-SHARPENED" and ax["b_panel"] == "B-UNRESOLVED" and ax["counts"]["states_batch_diffused"] == 0


def test_registered_constants_literal():
    src = (ROOT / "scratch" / "optimizer_geometry_desk.py").read_text()
    for s in ("STATES = [900, 7200, 13500]", "NEXT = {900: 3600, 7200: 10800, 13500: 15420}", "LR, WD, PCT, TOTAL = 3e-4, 0.01, 0.03, 15_420", "SHARPEN, SMALL, REL_MIN = 0.15, 0.05, 0.25",
              "SHARED_FRAC, ROTATED_FRAC = 0.8, 0.5", "CLIP, CLIP_EPS = 1.0, 1e-6", "PARITY_TOL_PARAM = 1e-9", "ENVELOPE_TOL = 5e-2", 'FAMILIES = ("g", "mhat", "a", "u", "b")',
              'assert probe["digest"] == ugc0["probe"]["digest"]', "SCHEDULER PARITY FAILED", "torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)", "opt.load_state_dict(ckpt[\"opt\"])",
              'assert opt_state_digest(opt) == odg', "u = decay[None, :] + a", "b = a - a0[None, :]"):
        assert s in src, s

"""SG-CROSSPOS-REPRESENTABILITY-0 instrument guards: the two attention
access laws (reverse-causal: output i depends on inputs j >= i only; local:
on j == i only), the zero-initialised output (the fit starts at the zero
predictor), and the BAR-1 adjudication law incl. NOT-RESOLVABLE."""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    prev = os.environ.get("SMOKE")
    os.environ["SMOKE"] = "1"      # import-time smoke constants; restored below so sibling fixtures see the registered ones
    for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scratch")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        m = importlib.import_module("sg_crosspos_desk")
    finally:
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


def _jac_pattern(mod, arm, T=6):
    import torch
    torch.manual_seed(0)
    pm = mod.SeqSG(d_in=8, d=16, n_layers=2, n_heads=2, d_ffn=32, d_out=4)
    torch.nn.init.normal_(pm.out.weight)     # otherwise every gradient is zero
    real = torch.ones(1, T, dtype=torch.bool)
    z = torch.randn(1, T, 8, requires_grad=True)
    dep = torch.zeros(T, T, dtype=torch.bool)
    for i in range(T):
        g = torch.autograd.grad(pm(z, mod.allowed_mask(arm, real))[0, i].sum(), z, retain_graph=True)[0][0]
        dep[i] = g.abs().sum(-1) > 0
    return dep


def test_reverse_causal_reads_future_only(mod):
    import torch
    dep = _jac_pattern(mod, "reverse_causal")
    T = dep.shape[0]
    expect = torch.arange(T).view(1, T) >= torch.arange(T).view(T, 1)
    assert torch.equal(dep, expect), dep


def test_local_reads_self_only(mod):
    import torch
    dep = _jac_pattern(mod, "local")
    assert torch.equal(dep, torch.eye(dep.shape[0], dtype=torch.bool)), dep


def test_pad_keys_excluded_and_pad_query_keeps_itself(mod):
    import torch
    real = torch.tensor([[True, True, False]])
    m = mod.allowed_mask("reverse_causal", real)[0]
    assert m.tolist() == [[True, True, False], [False, True, False], [False, False, True]]


def test_output_zero_init(mod):
    import torch
    pm = mod.SeqSG()
    z = torch.randn(2, 5, mod.D_IN)
    assert float(pm(z, mod.allowed_mask("reverse_causal", torch.ones(2, 5, dtype=torch.bool))).abs().max()) == 0.0
    assert mod.n_params(pm) < 1_000_000


def _rec(rc, loc, nonfinite=()):
    st = {}
    for s in [0, 463] + list(__import__("sg_crosspos_desk").LATE_STEPS):
        st[str(s)] = {"blocks": {str(l): {"reverse_causal": {"held_ratio": rc, "nonfinite": (s, l) in nonfinite},
                                          "local": {"held_ratio": loc, "nonfinite": False}} for l in [4, 5, 6, 7]}}
    return {"states": st}


def test_adjudicate_fires_and_no_fire(mod):
    assert mod.adjudicate(_rec(0.5, 1.0))["bar_1"] == "FIRES"
    a = mod.adjudicate(_rec(0.51, 1.0))
    assert a["bar_1"] == "NO-FIRE" and a["n_late_cells"] == 24 and a["late_median_local"] == 1.0


def test_adjudicate_not_resolvable(mod):
    bad = [(s, l) for s in mod.LATE_STEPS[:2] for l in [4, 5, 6]][:5]
    a = mod.adjudicate(_rec(0.1, 1.0, nonfinite=set(bad)))
    assert a["bar_1"] == "NOT-RESOLVABLE" and a["n_not_resolvable"] == 5
    a = mod.adjudicate(_rec(0.1, 1.0, nonfinite=set(bad[:4])))
    assert a["bar_1"] == "FIRES" and a["n_not_resolvable"] == 4

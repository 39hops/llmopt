"""SYNTHETIC-GRADIENT-WRITER-1 credit law (scratch/sg_credit.py): predictor
families have a ZERO output layer (initial credit exactly zero); the
parameter-detached teacher pass reproduces the writer logits and cannot
reach any parameter; the writer's block gradient equals J^T hat_delta
(explicit-surrogate check on one block); FOLD B: the predictor loss is the
elementwise MSE over eligible label positions (labels != -100) and hidden
dimensions, averaged over blocks, and reaches phi only; FOLD A: the
targets are cached at the pre-update state (sg_step_terms computes them
before any backward). Tiny model (house width, 2 blocks, 2 rows, 8 tokens)
on CPU in about a second."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sg():
    for p in (ROOT, ROOT / "scripts", ROOT / "scratch"):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("sg_credit", ROOT / "scratch" / "sg_credit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def tiny():
    import torch
    from llmopt.train.mathnative import build_model
    torch.manual_seed(3)
    model = build_model(40, d=384, layers=2, heads=6, ffn=1536)
    ids = torch.randint(1, 40, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[1, 6:] = 0
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[0, 2] = -100     # an ineligible position inside the attention mask, to separate the two masks
    return model, ids, mask, labels


def test_zero_output_layer(sg):
    import torch
    for fam in ("linear", "mlp256"):
        preds = sg.build_predictors(fam, [0, 1], seed=5)
        h, e = torch.randn(2, 8, 384), torch.randn(2, 8, 40)
        for l in ("0", "1"):
            assert float(preds[l](h, e).abs().max()) == 0.0
    a = sg.build_predictors("mlp256", [0], seed=5)["0"].inp.weight
    b = sg.build_predictors("mlp256", [0], seed=5)["0"].inp.weight
    assert torch.equal(a, b)


def test_teacher_matches_writer_and_reaches_no_parameter(sg, tiny):
    import torch
    model, ids, mask, labels = tiny
    consts = {"0": 1e-6, "1": 2e-6}
    preds = sg.build_predictors("linear", [0, 1], seed=1)
    T = sg.sg_step_terms(model, preds, ids, mask, labels, [0, 1], consts)
    assert torch.allclose(T["logits_T"], T["logits"].detach(), atol=1e-6)
    assert set(T["targets"]) == {0, 1} and T["targets"][0].shape == (2, 8, 384)
    logits_T2, _ = sg.teacher_targets(model, ids, mask)
    g = torch.autograd.grad(sg.ce_loss(logits_T2, labels), list(model.parameters()), allow_unused=True)
    assert all(x is None for x in g)


def test_block_gradient_is_jacobian_transpose_of_hat_delta(sg, tiny):
    import torch
    model, ids, mask, labels = tiny
    consts = {"1": 3e-6}
    preds = sg.build_predictors("linear", [1], seed=1)
    with torch.no_grad():
        preds["1"].A.weight.add_(torch.randn(384, 384) * 1e-2)
    T = sg.sg_step_terms(model, preds, ids, mask, labels, [1], consts)
    model.zero_grad(set_to_none=True)
    T["total"].backward()
    hat = T["hat_delta"][1]
    w = model.blocks[1].o.weight
    g_sg = w.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    m = sg.causal_mask(ids, mask)
    x1, _ = model.blocks[0](model.emb(ids), m, None)
    x2, _ = model.blocks[1](x1.detach(), m, None)
    (hat * x2).sum().backward()
    assert torch.allclose(g_sg, w.grad, atol=1e-9)
    assert model.blocks[0].o.weight.grad is None


def test_fold_b_elementwise_mse_over_label_positions(sg, tiny):
    import torch
    model, ids, mask, labels = tiny
    consts = {"0": 1e-6, "1": 2e-6}
    preds = sg.build_predictors("mlp256", [0, 1], seed=1)
    T = sg.sg_step_terms(model, preds, ids, mask, labels, [0, 1], consts)
    elig = labels != -100
    assert int(elig.sum()) == int(mask.sum()) - 1          # the extra -100 is excluded from eligibility
    # zero predictor: L_l = mean over eligible positions and 384 dims of (target / s)^2; L_phi = mean over blocks
    ref = {l: float((((T["targets"][l] / consts[str(l)]) ** 2).mean(-1) * elig).sum() / elig.sum()) for l in (0, 1)}
    assert T["pred_loss_per_block"][0] == pytest.approx(ref[0], rel=1e-6)
    assert T["pred_loss_per_block"][1] == pytest.approx(ref[1], rel=1e-6)
    assert float(T["pred_loss"]) == pytest.approx((ref[0] + ref[1]) / 2, rel=1e-6)
    # not the hidden-dimension SSE
    sse = float((((T["targets"][0] / consts["0"]) ** 2).sum(-1) * elig).sum() / elig.sum())
    assert abs(sse / ref[0] - 384) < 1e-3
    gm = torch.autograd.grad(T["pred_loss"], list(model.parameters()), allow_unused=True, retain_graph=True)
    assert all(x is None for x in gm)
    gp = torch.autograd.grad(T["pred_loss"], list(preds.parameters()), allow_unused=True)
    assert all(x is not None for x in gp)
    assert set(T["align"]) == {0, 1}


def test_fold_a_targets_cached_before_any_step(sg, tiny):
    import torch
    model, ids, mask, labels = tiny
    consts = {"0": 1e-6, "1": 2e-6}
    preds = sg.build_predictors("linear", [0, 1], seed=1)
    with torch.no_grad():
        for p in preds.parameters():
            p.add_(torch.randn_like(p) * 1e-2)
    _, _, pre = sg.true_hidden_errors(model, ids, mask, labels, [0, 1])
    T = sg.sg_step_terms(model, preds, ids, mask, labels, [0, 1], consts)
    mp, phi = list(model.parameters()), list(preds.parameters())
    sg.run_sg_step(T, mp, phi, torch.optim.AdamW(mp, lr=1e-3), torch.optim.AdamW(phi, lr=1e-3))
    _, _, post = sg.true_hidden_errors(model, ids, mask, labels, [0, 1])
    assert all(torch.equal(T["targets"][l], pre[l]) for l in (0, 1))
    assert max(float((post[l] - pre[l]).abs().max()) for l in (0, 1)) > 0

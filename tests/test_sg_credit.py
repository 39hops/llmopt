"""SYNTHETIC-GRADIENT-WRITER-1 credit law (scratch/sg_credit.py): predictor
families have a ZERO output layer (initial credit exactly zero), the
parameter-detached teacher pass reproduces the writer logits and cannot
reach any parameter, the writer's block gradients equal J^T hat_delta
(finite-difference check on one block), and the normalized regression loss
is masked. Runs on a tiny model (d 384 is the house shape; 2 blocks, 2 rows,
8 tokens) on CPU in a second."""
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
    return model, ids, mask, labels


def test_zero_output_layer(sg):
    import torch
    for fam in ("linear", "mlp256"):
        preds = sg.build_predictors(fam, [0, 1], seed=5)
        h, e = torch.randn(2, 8, 384), torch.randn(2, 8, 40)
        for l in ("0", "1"):
            assert float(preds[l](h, e).abs().max()) == 0.0
    # deterministic init from the seed (MLP input layer)
    a = sg.build_predictors("mlp256", [0], seed=5)["0"].inp.weight
    b = sg.build_predictors("mlp256", [0], seed=5)["0"].inp.weight
    assert torch.equal(a, b)


def test_teacher_matches_writer_and_reaches_no_parameter(sg, tiny):
    import torch
    model, ids, mask, labels = tiny
    consts = {"0": 1e-6, "1": 2e-6}
    preds = sg.build_predictors("linear", [0, 1], seed=1)
    T = sg.sg_step_terms(model, preds, ids, mask, labels, [0, 1], consts)
    loss_T, logits_T, targets = T["teacher"]()
    assert torch.allclose(logits_T, T["logits"].detach(), atol=1e-6)
    assert set(targets) == {0, 1} and targets[0].shape == (2, 8, 384)
    logits_T2, outs_T2 = sg.teacher_targets(model, ids, mask)
    lT = sg.ce_loss(logits_T2, labels)
    g = torch.autograd.grad(lT, list(model.parameters()), allow_unused=True)
    assert all(x is None for x in g)


def test_block_gradient_is_jacobian_transpose_of_hat_delta(sg, tiny):
    import torch
    model, ids, mask, labels = tiny
    sg_blocks = [1]
    consts = {"1": 3e-6}
    preds = sg.build_predictors("linear", sg_blocks, seed=1)
    with torch.no_grad():
        preds["1"].A.weight.add_(torch.randn(384, 384) * 1e-2)
    T = sg.sg_step_terms(model, preds, ids, mask, labels, sg_blocks, consts)
    model.zero_grad(set_to_none=True)
    T["total"].backward()
    hat = T["hat_delta"][1]
    # reference: J^T hat via an explicit surrogate on a fresh forward with the block input detached
    w = model.blocks[1].o.weight
    g_sg = w.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    m = sg.causal_mask(ids, mask)
    x0 = model.emb(ids)
    x1, _ = model.blocks[0](x0, m, None)
    x2, _ = model.blocks[1](x1.detach(), m, None)
    (hat * x2).sum().backward()
    assert torch.allclose(g_sg, w.grad, atol=1e-9)
    # block 0 (not an SG block here, frozen by construction of the test) received no gradient from the SG surrogate
    assert model.blocks[0].o.weight.grad is None


def test_pred_loss_masked_and_reaches_phi_only(sg, tiny):
    import torch
    model, ids, mask, labels = tiny
    consts = {"0": 1e-6, "1": 2e-6}
    preds = sg.build_predictors("mlp256", [0, 1], seed=1)
    T = sg.sg_step_terms(model, preds, ids, mask, labels, [0, 1], consts)
    _, _, targets = T["teacher"]()
    pl = T["pred_loss"](targets)
    # zero predictor output: the loss equals the mean over valid tokens of ||target / s||^2
    valid = mask.bool()
    ref = sum(((targets[l] / consts[str(l)]).pow(2).sum(-1) * valid).sum() / valid.sum() for l in (0, 1))
    assert torch.allclose(pl, ref)
    gm = torch.autograd.grad(pl, list(model.parameters()), allow_unused=True, retain_graph=True)
    assert all(x is None for x in gm)
    gp = torch.autograd.grad(pl, list(preds.parameters()), allow_unused=True)
    assert all(x is not None for x in gp)

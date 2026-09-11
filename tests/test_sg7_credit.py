"""SG-BOUNDARY-BLOCK7-1 credit law (scratch/sg7_credit.py): the predictor
classes are verbatim copies of the cross-position desk's (one documented
device delta); the true CE reaches no block-7 parameter; blocks 4..6 /
norm / head receive the frozen-BP control's gradient bit-exactly; block 7
receives J^T hat_delta_7 and nothing else; the forced-delta^BP endpoint
reproduces the control's gradients and one clipped AdamW step; the
predictor loss reaches phi only and the model objective reaches no phi;
credit is zero at ineligible positions and exactly zero at init; FOLD A
caches the target before any step. House-width 8-block model on a 2 x 8
batch, CPU."""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    for p in (ROOT, ROOT / "scripts", ROOT / "scratch"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scratch" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def sg7():
    return _load("sg7_credit")


@pytest.fixture(scope="module")
def dfa():
    return _load("dfa_credit")


@pytest.fixture(scope="module")
def tiny():
    import torch
    from llmopt.train.mathnative import build_model
    torch.manual_seed(3)
    model = build_model(40, d=384, layers=8, heads=6, ffn=1536)
    ids = torch.randint(1, 40, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[1, 6:] = 0
    labels = ids.clone()
    labels[mask == 0] = -100
    labels[0, 2] = -100
    return model, ids, mask, labels


def _fresh(sg7, dfa, tiny, seed=5, perturb=0.0):
    import copy
    import torch
    model = copy.deepcopy(tiny[0]).train()
    dfa.freeze_lower(model, 4)
    mu, sd = torch.zeros(1, sg7.D_IN), torch.ones(1, sg7.D_IN)
    pred = sg7.build_local_predictor(mu, sd, seed)
    if perturb:
        with torch.no_grad():
            for p in pred.parameters():
                p.add_(torch.randn_like(p) * perturb)
    return model, pred


def _control_grads(dfa, model, ids, mask, labels):
    model.zero_grad(set_to_none=True)
    ob = dfa.hybrid_objective(model, [], ids, mask, labels, 8)
    ob["loss"].backward()
    return ob["logits"].detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def _segments(path, names):
    import ast
    src = path.read_text()
    tree = ast.parse(src)
    return {n.name: ast.get_source_segment(src, n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names}


def test_predictor_classes_verbatim_v_crosspos_desk(sg7):
    names = ("alibi_slopes", "allowed_mask", "Attn", "SeqSG")
    a = _segments(ROOT / "scratch" / "sg_crosspos_desk.py", names)
    b = _segments(ROOT / "scratch" / "sg7_credit.py", names)
    for name in names:
        x = re.sub(r"d=D_PRED, n_layers=N_LAYERS, n_heads=N_HEADS, d_ffn=D_FFN", "d=128, n_layers=2, n_heads=4, d_ffn=512", a[name])
        y = b[name].replace(", device=x.device", "")
        assert x == y, name
    assert sg7.n_params(sg7.SeqSG()) == 500736


def test_no_ce_leak_to_block7(sg7, dfa, tiny):
    import torch
    model, pred = _fresh(sg7, dfa, tiny)
    _, ids, mask, labels = tiny
    T = sg7.sg7_step_terms(model, pred, ids, mask, labels, 1e-3)
    b7 = dfa.block_params(model, 7)
    g = torch.autograd.grad(T["loss"], b7, allow_unused=True, retain_graph=True)
    assert all(x is None for x in g)
    model.zero_grad(set_to_none=True)
    T["loss"].backward(retain_graph=True)
    assert all(p.grad is None for p in b7)
    assert all(p.grad is not None for l in (4, 5, 6) for p in dfa.block_params(model, l))
    assert model.norm.g.grad is not None and model.head.weight.grad is not None
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)


def test_exact_bp_to_blocks_4_6_head_norm(sg7, dfa, tiny):
    import torch
    model, pred = _fresh(sg7, dfa, tiny, perturb=1e-2)
    _, ids, mask, labels = tiny
    logits_c, gc = _control_grads(dfa, model, ids, mask, labels)
    model.zero_grad(set_to_none=True)
    T = sg7.sg7_step_terms(model, pred, ids, mask, labels, 1e-3)
    assert torch.equal(T["logits"].detach(), logits_c)
    T["total"].backward()
    for n, p in model.named_parameters():
        if p.requires_grad and not n.startswith("blocks.7."):
            assert torch.equal(p.grad, gc[n]), n


def test_block7_receives_jacobian_transpose_of_hat_only(sg7, dfa, tiny):
    import torch
    model, pred = _fresh(sg7, dfa, tiny, perturb=1e-2)
    _, ids, mask, labels = tiny
    b7 = dfa.block_params(model, 7)
    T = sg7.sg7_step_terms(model, pred, ids, mask, labels, 1e-3)
    hat = T["hat_delta"][7].clone()
    assert float(hat.abs().max()) > 0
    model.zero_grad(set_to_none=True)
    T["total"].backward()
    g_total = [p.grad.detach().clone() for p in b7]
    model.zero_grad(set_to_none=True)
    _, _, _, x8_L, _ = sg7.sg7_forward(model, ids, mask)
    (hat * x8_L).sum().backward()
    g_sur = [p.grad.detach().clone() for p in b7]
    assert all(torch.equal(a, b) for a, b in zip(g_total, g_sur))


def test_forced_delta_endpoint_reproduces_control(sg7, dfa, tiny):
    import copy
    import torch
    model, _ = _fresh(sg7, dfa, tiny)
    _, ids, mask, labels = tiny
    ctrl = copy.deepcopy(model)
    _, gc = _control_grads(dfa, ctrl, ids, mask, labels)
    model.zero_grad(set_to_none=True)
    sg7.forced_total(model, ids, mask, labels).backward()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("blocks.7."):
            rel = float((p.grad - gc[n]).abs().max()) / max(float(gc[n].abs().max()), 1e-30)
            assert rel <= 1e-5, (n, rel)
        else:
            assert torch.equal(p.grad, gc[n]), n
    for m in (model, ctrl):
        ps = [p for p in m.parameters() if p.requires_grad]
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        torch.optim.AdamW(ps, lr=3e-4, weight_decay=0.01).step()
    for (n, p), (_, q) in zip(model.named_parameters(), ctrl.named_parameters()):
        assert float((p.detach() - q.detach()).abs().max()) <= 1e-6, n


def test_predictor_isolation_and_fold_b(sg7, dfa, tiny):
    import torch
    model, pred = _fresh(sg7, dfa, tiny, perturb=1e-2)
    _, ids, mask, labels = tiny
    phi = list(pred.parameters())
    mp = [p for p in model.parameters() if p.requires_grad]
    T = sg7.sg7_step_terms(model, pred, ids, mask, labels, 1e-3)
    assert all(x is None for x in torch.autograd.grad(T["pred_loss"], mp, allow_unused=True, retain_graph=True))
    assert all(x is not None for x in torch.autograd.grad(T["pred_loss"], phi, allow_unused=True, retain_graph=True))
    assert all(x is None for x in torch.autograd.grad(T["total"], phi, allow_unused=True, retain_graph=True))
    elig = T["elig"]
    ref = ((T["g_out"][7] - T["targets"][7] / 1e-3) ** 2).mean(-1)[elig].mean()
    assert torch.allclose(T["pred_loss"], ref)


def test_eligible_masking_and_zero_init(sg7, dfa, tiny):
    import torch
    _, ids, mask, labels = tiny
    model, pred0 = _fresh(sg7, dfa, tiny)
    T0 = sg7.sg7_step_terms(model, pred0, ids, mask, labels, 1e-3)
    assert float(T0["hat_delta"][7].abs().max()) == 0.0
    model.zero_grad(set_to_none=True)
    T0["total"].backward()
    assert all(float(p.grad.abs().max()) == 0.0 for p in dfa.block_params(model, 7))
    model, pred = _fresh(sg7, dfa, tiny, perturb=1e-2)
    T = sg7.sg7_step_terms(model, pred, ids, mask, labels, 1e-3)
    inel = ~T["elig"]
    assert int(inel.sum()) == 3
    assert float(T["hat_delta"][7][inel].abs().max()) == 0.0
    assert float(T["targets"][7][inel].abs().max()) == 0.0
    assert float(T["hat_delta"][7][T["elig"]].abs().max()) > 0.0


def test_fold_a_target_cached_before_step(sg7, dfa, tiny):
    import torch
    from sg_credit import run_sg_step, true_hidden_errors
    model, pred = _fresh(sg7, dfa, tiny, perturb=1e-2)
    _, ids, mask, labels = tiny
    mp = [p for p in model.parameters() if p.requires_grad]
    phi = list(pred.parameters())
    _, _, pre = true_hidden_errors(model, ids, mask, labels, [7])
    T = sg7.sg7_step_terms(model, pred, ids, mask, labels, 1e-3)
    run_sg_step(T, mp, phi, torch.optim.AdamW(mp, lr=3e-4, weight_decay=0.01), torch.optim.AdamW(phi, lr=1e-3, weight_decay=0.0))
    _, _, post = true_hidden_errors(model, ids, mask, labels, [7])
    assert torch.equal(T["targets"][7], pre[7])
    assert float((post[7] - pre[7]).abs().max()) > 0.0

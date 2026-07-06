"""Task-vector properties: exact undo, additivity, zero no-op."""

import torch
import torch.nn as nn

from llmopt.train.task_vector import apply_task_vector, load_adapter


def _toy_model():
    torch.manual_seed(0)
    m = nn.Module()
    m.q_proj = nn.Linear(8, 8, bias=False)
    m.up_proj = nn.Linear(8, 16, bias=False)
    return m


def _toy_adapter(r=2):
    g = torch.Generator().manual_seed(1)
    return {
        "q_proj": (torch.randn(r, 8, generator=g), torch.randn(8, r, generator=g)),
        "up_proj": (torch.randn(r, 8, generator=g), torch.randn(16, r, generator=g)),
    }


def _snap(m):
    return {k: v.clone() for k, v in m.state_dict().items()}


def test_apply_then_undo_is_bit_identical():
    m, ad = _toy_model(), _toy_adapter()
    before = _snap(m)
    undo = apply_task_vector(m, ad, scale=1.0, r=2, alpha=4)
    changed = any(not torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert changed
    undo()
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_zero_scale_is_noop():
    m, ad = _toy_model(), _toy_adapter()
    before = _snap(m)
    apply_task_vector(m, ad, scale=0.0, r=2, alpha=4)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())


def test_scales_are_additive():
    ad = _toy_adapter()
    m1 = _toy_model()
    apply_task_vector(m1, ad, scale=0.7, r=2, alpha=4)
    apply_task_vector(m1, ad, scale=0.3, r=2, alpha=4)
    m2 = _toy_model()
    apply_task_vector(m2, ad, scale=1.0, r=2, alpha=4)
    for a, b in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.allclose(a, b, atol=1e-6)


def test_load_adapter_pairs_ab(tmp_path):
    # the train script saves flat keys ending in .a / .b
    a = torch.randn(2, 8)
    b = torch.randn(8, 2)
    torch.save(
        {"model.layers.0.self_attn.q_proj.a": a,
         "model.layers.0.self_attn.q_proj.b": b},
        tmp_path / "ad.pt",
    )
    ad = load_adapter(tmp_path / "ad.pt")
    assert set(ad) == {"model.layers.0.self_attn.q_proj"}
    assert torch.equal(ad["model.layers.0.self_attn.q_proj"][0], a)
    assert torch.equal(ad["model.layers.0.self_attn.q_proj"][1], b)

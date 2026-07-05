"""internals/ probes on a tiny random Llama (CPU)."""

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.internals.activations import activation_stats
from llmopt.internals.attention_stats import attention_stats
from llmopt.internals.cka import layer_cka_matrix, linear_cka
from llmopt.internals.logit_lens import logit_lens

VOCAB, LAYERS, SEQ = 64, 3, 10
IDS = [3, 14, 15, 9, 2, 6, 5, 35, 8, 9]


@pytest.fixture(scope="module")
def model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=LAYERS, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=128,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


def test_logit_lens_final_layer_is_model_output(model):
    lens = logit_lens(model, IDS)
    assert lens["logits"].shape == (LAYERS + 1, SEQ, VOCAB)
    with torch.inference_mode():
        real = model(input_ids=torch.tensor([IDS])).logits[0]
    assert torch.allclose(lens["logits"][-1], real, atol=1e-4)
    assert torch.allclose(
        lens["kl_to_final"][-1], torch.zeros(SEQ), atol=1e-5
    )
    assert (lens["kl_to_final"][:-1] >= -1e-6).all()


def test_attention_stats_bounds(model):
    st = attention_stats(model, IDS)
    assert st["entropy"].shape == (LAYERS, 4, SEQ)
    # causal row q has q+1 visible keys: entropy in [0, log(q+1)]
    for q in range(SEQ):
        assert (st["entropy"][:, :, q] <= math.log(q + 1) + 1e-5).all()
    assert (st["entropy"] >= -1e-6).all()
    assert (st["mean_distance"] >= 0).all()
    assert float(st["mean_distance"][:, :, 0].max()) == 0  # first token: self only


def test_activation_stats_shapes_and_sanity(model):
    st = activation_stats(model, IDS)
    assert len(st) == LAYERS
    for s in st:
        assert s["rms"] > 0 and s["std"] > 0
        assert 0 <= s["outlier_frac"] < 0.05
        assert s["kurtosis"] > -3


def test_linear_cka_invariances():
    torch.manual_seed(1)
    x = torch.randn(50, 16)
    assert linear_cka(x, x) == pytest.approx(1.0, abs=1e-5)
    q, _ = torch.linalg.qr(torch.randn(16, 16))
    assert linear_cka(x, 3.7 * x @ q) == pytest.approx(1.0, abs=1e-4)
    assert linear_cka(x, torch.randn(50, 16)) < 0.5


def test_layer_cka_matrix(model):
    m = layer_cka_matrix(model, IDS)
    assert m.shape == (LAYERS + 1, LAYERS + 1)
    assert torch.allclose(m, m.T)
    assert torch.allclose(m.diagonal(), torch.ones(LAYERS + 1))
    assert (m >= -1e-6).all() and (m <= 1 + 1e-6).all()

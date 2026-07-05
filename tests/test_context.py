"""Prompt compression + RoPE scaling on a tiny random Llama (CPU)."""

import math

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.context.compression import compress_prompt, token_self_information
from llmopt.context.rope_scaling import (
    apply_rope_scaling,
    ntk_inv_freq,
    pi_inv_freq,
    yarn_attention_temperature,
    yarn_inv_freq,
)

VOCAB = 64


def _llama(max_pos=64, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=max_pos, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


# --- compression ----------------------------------------------------------


def test_self_information_matches_manual(model=None):
    model = _llama()
    ids = [3, 14, 15, 9, 2, 6]
    info = token_self_information(model, ids)
    with torch.inference_mode():
        logp = torch.log_softmax(model(input_ids=torch.tensor([ids])).logits[0], -1)
    assert info[0] == float("inf")
    for i in range(1, len(ids)):
        assert info[i] == pytest.approx(
            -float(logp[i - 1, ids[i]]) / math.log(2), rel=1e-5
        )


def test_compress_hits_ratio_keeps_order_and_protected():
    model = _llama()
    ids = list(range(1, 41))
    protect = [(35, 40)]  # "the question" at the end
    out, stats = compress_prompt(model, ids, ratio=0.5, protect=protect)

    assert stats["compressed_len"] == 20
    assert out == sorted(out, key=lambda t: ids.index(t))  # order preserved
    assert all(t in out for t in ids[35:40])  # protected survives
    # dropped tokens were the predictable (low-information) ones
    assert stats["kept_mean_bits"] > stats["dropped_mean_bits"]


def test_compress_prefill_saving_via_roofline():
    from llmopt.eval.roofline import A100_80G, ModelShape, report

    m = ModelShape(layers=32, hidden=4096, heads=32, kv_heads=32,
                   head_dim=128, intermediate=11008, vocab=32000)
    full = report(m, A100_80G, ctx=0, new_tokens=4000)
    half = report(m, A100_80G, ctx=0, new_tokens=2000)
    assert half["latency_s"] < 0.55 * full["latency_s"]  # ~linear in tokens


# --- rope scaling -----------------------------------------------------------


def test_pi_ntk_yarn_frequency_shapes():
    inv = 10000.0 ** (-torch.arange(0, 32, 2) / 32)
    pi = pi_inv_freq(inv, 4.0)
    assert torch.allclose(pi, inv / 4)

    ntk = ntk_inv_freq(inv, 4.0, 32)
    assert ntk[0] == pytest.approx(float(inv[0]))  # fastest untouched
    assert float(ntk[-1]) == pytest.approx(float(inv[-1]) / 4, rel=0.05)

    yarn = yarn_inv_freq(inv, 4.0, orig_max_pos=64)
    assert (yarn <= inv + 1e-9).all() and (yarn >= inv / 4 - 1e-9).all()
    assert yarn_attention_temperature(1.0) == pytest.approx(1.0)
    assert yarn_attention_temperature(8.0) > 1.0


@pytest.mark.parametrize("method", ["pi", "ntk", "yarn"])
def test_scaled_model_runs_past_trained_window(method):
    model = _llama(max_pos=32)
    apply_rope_scaling(model, method=method, scale=4.0)
    assert model.config.max_position_embeddings == 128
    long_ids = torch.randint(1, VOCAB, (1, 100))
    with torch.inference_mode():
        out = model(input_ids=long_ids)
    assert out.logits.shape[1] == 100
    assert torch.isfinite(out.logits).all()

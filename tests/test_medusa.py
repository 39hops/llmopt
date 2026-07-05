"""Medusa heads on a tiny random Llama (CPU, no download).

Random (untrained) heads must still produce vanilla-greedy output —
verification, not drafting, owns correctness. Training must reduce loss
and lift acceptance on a repetitive corpus.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.medusa import (
    build_medusa_heads,
    draft_candidates,
    generate_medusa,
    train_medusa_heads,
)

VOCAB = 128
HIDDEN = 64


@pytest.fixture(scope="module")
def model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


def _greedy(model, ids, n):
    out = list(ids)
    with torch.inference_mode():
        for _ in range(n):
            logits = model(input_ids=torch.tensor([out])).logits
            out.append(int(logits[0, -1].argmax()))
    return out


def test_draft_candidates_shape():
    torch.manual_seed(0)
    heads = build_medusa_heads(HIDDEN, VOCAB, num_heads=3)
    cands = draft_candidates(heads, torch.randn(HIDDEN), next_token=7, topk=2)
    assert len(cands) == 2**3
    assert all(c[0] == 7 and len(c) == 4 for c in cands)


def test_untrained_heads_still_greedy_equivalent(model):
    torch.manual_seed(1)
    heads = build_medusa_heads(HIDDEN, VOCAB, num_heads=2)
    prompt = [5, 6, 7, 8, 5, 6, 7, 8, 5, 6]
    n = 20
    out, stats = generate_medusa(
        model, heads, prompt, max_new_tokens=n, topk=2
    )
    assert out == _greedy(model, prompt, n)
    assert stats["forward_passes"] <= n + 1


def test_training_reduces_loss_and_lifts_acceptance(model):
    torch.manual_seed(2)
    heads = build_medusa_heads(HIDDEN, VOCAB, num_heads=2)
    prompt = [9, 4, 2]
    n = 24
    base = _greedy(model, prompt, n)  # what the model actually continues with

    _, before = generate_medusa(model, heads, prompt, max_new_tokens=n)
    losses = train_medusa_heads(model, heads, [base], epochs=30, lr=1e-2)
    _, after = generate_medusa(model, heads, prompt, max_new_tokens=n)

    assert losses[-1] < losses[0]
    assert after["accepted"] > before["accepted"]
    # correctness unchanged by training
    out, _ = generate_medusa(model, heads, prompt, max_new_tokens=n)
    assert out == base

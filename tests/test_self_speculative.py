"""LayerSkip-style self-speculation on a tiny random Llama (CPU)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.self_speculative import (
    early_exit_draft,
    generate_self_speculative,
)

VOCAB = 64


@pytest.fixture(scope="module")
def model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
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


def test_early_exit_shares_weights_and_runs(model):
    draft = early_exit_draft(model, 2)
    assert len(draft.model.layers) == 2
    assert draft.model.layers[0] is model.model.layers[0]  # shared, not copied
    assert len(model.model.layers) == 4  # original untouched
    out = draft(input_ids=torch.tensor([[1, 2, 3]]))
    assert out.logits.shape == (1, 3, VOCAB)


def test_self_speculative_matches_full_greedy(model):
    prompt = [5, 6, 7, 8, 5, 6, 7, 8]
    n = 20
    out, stats = generate_self_speculative(
        model, prompt, exit_layer=2, max_new_tokens=n, num_draft=4
    )
    assert out == _greedy(model, prompt, n)
    assert stats["drafted"] > 0

"""StackedEngine: radix reuse + lookup decode + static cache (tiny Llama).

Invariant: token-identical to eager greedy, cold or warm, any prefix
sharing pattern. Reuse must only show up in the stats.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.stacked import StackedEngine

VOCAB = 128


@pytest.fixture(scope="module")
def model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
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


# repetitive prompt so prompt-lookup actually drafts
PROMPT = [7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 42, 43, 44, 45, 3]


def test_cold_request_matches_greedy(model):
    n = 16
    engine = StackedEngine(model, num_draft=4, max_ngram=3)
    tokens, stats = engine.generate(PROMPT, max_new_tokens=n)
    assert tokens == _greedy(model, PROMPT, n)
    assert stats["prefix_hit_tokens"] == 0


def test_warm_identical_prompt_reuses_and_matches(model):
    n = 16
    engine = StackedEngine(model, num_draft=4, max_ngram=3)
    engine.generate(PROMPT, max_new_tokens=n)
    tokens, stats = engine.generate(PROMPT, max_new_tokens=n)
    assert tokens == _greedy(model, PROMPT, n)
    assert stats["prefix_hit_tokens"] == len(PROMPT) - 1


def test_warm_diverging_suffix_matches(model):
    n = 12
    p2 = PROMPT[:10] + [99, 98, 97]
    engine = StackedEngine(model, num_draft=4, max_ngram=3)
    engine.generate(PROMPT, max_new_tokens=n)
    tokens, stats = engine.generate(p2, max_new_tokens=n)
    assert tokens == _greedy(model, p2, n)
    assert stats["prefix_hit_tokens"] == 10  # mid-edge split


def test_lookup_still_saves_forward_passes(model):
    n = 20
    engine = StackedEngine(model, num_draft=6, max_ngram=3)
    _, cold = engine.generate(PROMPT, max_new_tokens=n)
    _, warm = engine.generate(PROMPT, max_new_tokens=n)
    for stats in (cold, warm):
        assert stats["forward_passes"] < n  # verify blocks accepted drafts
    assert warm["prefix_hit_tokens"] > 0


def test_eos_stops(model):
    n = 24
    full = _greedy(model, PROMPT, n)
    eos = full[len(PROMPT) + 5]
    engine = StackedEngine(model, num_draft=4, max_ngram=3)
    tokens, _ = engine.generate(PROMPT, max_new_tokens=n, eos_token_id=eos)
    assert tokens[-1] == eos
    assert tokens == full[: len(tokens)]

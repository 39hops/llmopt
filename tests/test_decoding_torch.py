"""Equivalence tests for decode loops on a tiny random Llama (no downloads).

The invariant that makes the library trustworthy: prompt-lookup and greedy
speculative decoding must produce token-for-token identical output to
vanilla greedy decoding, KV caching included.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.prompt_lookup import generate_with_prompt_lookup
from llmopt.decoding.speculative import generate_speculative
from llmopt.eval.equivalence import assert_tokens_equal

VOCAB = 128


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
    )
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def tiny_draft():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return LlamaForCausalLM(cfg).eval()


def vanilla_greedy(model, prompt: list[int], max_new: int) -> list[int]:
    tokens = list(prompt)
    with torch.inference_mode():
        for _ in range(max_new):
            logits = model(input_ids=torch.tensor([tokens])).logits[0, -1]
            tokens.append(int(logits.argmax()))
    return tokens


PROMPT = [3, 7, 11, 5, 3, 7, 11, 9, 2, 3, 7]  # repeated trigram -> lookup hits
MAX_NEW = 24


def test_prompt_lookup_matches_vanilla_greedy(tiny_model):
    ref = vanilla_greedy(tiny_model, PROMPT, MAX_NEW)
    out, stats = generate_with_prompt_lookup(
        tiny_model, torch.tensor([PROMPT]), max_new_tokens=MAX_NEW, num_draft=6
    )
    r = assert_tokens_equal(ref, out)
    assert r, r.detail
    assert stats["forward_passes"] <= MAX_NEW  # never worse than vanilla


def test_speculative_greedy_matches_vanilla_greedy(tiny_model, tiny_draft):
    ref = vanilla_greedy(tiny_model, PROMPT, MAX_NEW)
    out, stats = generate_speculative(
        tiny_model,
        tiny_draft,
        torch.tensor([PROMPT]),
        max_new_tokens=MAX_NEW,
        num_draft=4,
    )
    r = assert_tokens_equal(ref, out)
    assert r, r.detail
    assert stats["target_passes"] <= MAX_NEW


def test_speculative_sampling_runs_and_lengths_ok(tiny_model, tiny_draft):
    out, stats = generate_speculative(
        tiny_model,
        tiny_draft,
        torch.tensor([PROMPT]),
        max_new_tokens=MAX_NEW,
        num_draft=4,
        temperature=0.8,
        seed=42,
    )
    assert len(out) == len(PROMPT) + MAX_NEW
    assert stats["accepted"] <= stats["drafted"]

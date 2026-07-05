"""Continuous batching engine on a tiny random Llama (CPU, no download).

Invariant: every request's output is token-identical to unbatched
greedy decoding of that prompt alone, regardless of batch composition,
padding, chunked prefill, or join/leave order.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.batching import BatchEngine

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


PROMPTS = [
    [5, 6, 7, 8, 9],
    [42, 3, 17, 99, 4, 23, 55, 1, 88, 12, 7],  # long: multiple chunks
    [100, 101],
    [64, 32, 16, 8, 4, 2],
]


def test_batched_matches_unbatched_greedy(model):
    n = 12
    engine = BatchEngine(model, max_batch=2, chunk_size=4)
    rids = [engine.submit(p, max_new_tokens=n) for p in PROMPTS]
    results = engine.run()

    for rid, prompt in zip(rids, PROMPTS):
        assert results[rid] == _greedy(model, prompt, n), rid
    # max_batch=2 with 4 requests forces continuous join/leave
    assert max(engine.stats["batch_occupancy"]) == 2


def test_uneven_lengths_finish_independently(model):
    engine = BatchEngine(model, max_batch=3, chunk_size=8)
    r_short = engine.submit(PROMPTS[0], max_new_tokens=3)
    r_long = engine.submit(PROMPTS[1], max_new_tokens=15)
    results = engine.run()

    assert results[r_short] == _greedy(model, PROMPTS[0], 3)
    assert results[r_long] == _greedy(model, PROMPTS[1], 15)


def test_eos_stops_request(model):
    n = 20
    full = _greedy(model, PROMPTS[3], n)
    eos = full[len(PROMPTS[3]) + 4]  # force an early stop

    engine = BatchEngine(model, max_batch=2, chunk_size=4)
    rid = engine.submit(PROMPTS[3], max_new_tokens=n, eos_token_id=eos)
    other = engine.submit(PROMPTS[0], max_new_tokens=n)
    results = engine.run()

    assert results[rid][-1] == eos
    assert results[rid] == full[: len(results[rid])]
    assert results[other] == _greedy(model, PROMPTS[0], n)


def test_chunked_prefill_interleaves(model):
    # chunk smaller than prompt -> prefill spans several engine steps
    engine = BatchEngine(model, max_batch=1, chunk_size=2)
    rid = engine.submit(PROMPTS[1], max_new_tokens=5)
    results = engine.run()
    assert results[rid] == _greedy(model, PROMPTS[1], 5)

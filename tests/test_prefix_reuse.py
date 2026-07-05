"""Radix prefix KV reuse wired into the batching engine (tiny Llama, CPU).

Invariant unchanged from test_batching: output token-identical to
unbatched greedy — prefix reuse must be invisible except in the stats.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.cache.prefix_reuse import (
    payloads_to_cache,
    slice_payload,
    split_payload,
)
from llmopt.cache.radix import RadixCache
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


def _engine(model, **kw):
    return BatchEngine(
        model, prefix_cache=RadixCache(split_payload=split_payload), **kw
    )


# --- radix edge splitting (pure structure) ----------------------------------


def _toy_payload(tokens):
    # payload mirroring the KV layout: one layer, [1, 1, t, 1] tensor of ids
    t = torch.tensor(tokens, dtype=torch.float32)[None, None, :, None]
    return [(t, t + 1000)]


def test_match_splits_partial_edge():
    c = RadixCache(split_payload=split_payload)
    c.insert(list(range(10)), lambda s, e: _toy_payload(list(range(s, e))))

    matched, payloads = c.match([0, 1, 2, 3, 4, 99])
    assert matched == 5
    k = torch.cat([p[0][0] for p in payloads], dim=2)
    assert k[0, 0, :, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert c.cached_tokens == 10  # split moved tokens, freed nothing


def test_partial_edge_without_splitter_is_a_miss():
    c = RadixCache()  # no split_payload -> old conservative behavior
    c.insert(list(range(10)), lambda s, e: _toy_payload(list(range(s, e))))
    matched, _ = c.match([0, 1, 2, 99])
    assert matched == 0


def test_insert_after_divergence_branches_not_clobbers():
    c = RadixCache(split_payload=split_payload)
    mk = lambda ids: (lambda s, e: _toy_payload(ids[s:e]))
    a = [0, 1, 2, 3, 4, 5]
    b = [0, 1, 2, 9, 9, 9]
    c.insert(a, mk(a))
    c.insert(b, mk(b))
    assert c.cached_tokens == 9  # 6 + 3 new (shared head stored once)
    for seq in (a, b):
        matched, payloads = c.match(seq)
        assert matched == 6
        k = torch.cat([p[0][0] for p in payloads], dim=2)
        assert k[0, 0, :, 0].tolist() == [float(t) for t in seq]


# --- engine integration -------------------------------------------------------


PROMPT = [42, 3, 17, 99, 4, 23, 55, 1, 88, 12, 7]


def test_identical_prompt_reuses_and_stays_exact(model):
    n = 8
    ref = _greedy(model, PROMPT, n)

    engine = _engine(model, max_batch=2, chunk_size=4)
    r1 = engine.submit(PROMPT, max_new_tokens=n)
    results = engine.run()
    assert results[r1] == ref
    assert engine.stats["prefix_hit_tokens"] == 0  # cold cache

    r2 = engine.submit(PROMPT, max_new_tokens=n)
    results = engine.run()
    assert results[r2] == ref
    # everything but the final prompt token skipped
    assert engine.stats["prefix_hit_tokens"] == len(PROMPT) - 1


def test_shared_prefix_diverging_suffix_exact(model):
    n = 8
    p1 = PROMPT
    p2 = PROMPT[:6] + [101, 102, 103]
    engine = _engine(model, max_batch=2, chunk_size=4)
    r1 = engine.submit(p1, max_new_tokens=n)
    engine.run()
    r2 = engine.submit(p2, max_new_tokens=n)
    results = engine.run()

    assert results[r2] == _greedy(model, p2, n)
    assert engine.stats["prefix_hit_tokens"] == 6  # split mid-edge


def test_reuse_saves_prefill_forwards(model):
    engine = _engine(model, max_batch=2, chunk_size=4)
    engine.submit(PROMPT, max_new_tokens=2)
    engine.run()
    cold = engine.stats["prefill_tokens"]
    engine.submit(PROMPT, max_new_tokens=2)
    engine.run()
    warm = engine.stats["prefill_tokens"] - cold
    assert cold == len(PROMPT)
    assert warm == 1  # only the last prompt token recomputed


def test_generated_tokens_not_cached_as_prefix(model):
    # tree stores prompt KV only; a prompt equal to prompt+generated of a
    # previous request must still match just the prompt part
    n = 4
    engine = _engine(model, max_batch=2, chunk_size=4)
    rid = engine.submit(PROMPT, max_new_tokens=n)
    results = engine.run()
    extended = results[rid]  # prompt + n generated

    r2 = engine.submit(extended, max_new_tokens=3)
    results = engine.run()
    assert results[r2] == _greedy(model, extended, 3)
    assert engine.stats["prefix_hit_tokens"] == len(PROMPT)


def test_payload_roundtrip_matches_source_cache(model):
    # slice -> tree -> concat must reproduce the original KV exactly
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([PROMPT]), use_cache=True)
    legacy = [[l.keys, l.values] for l in out.past_key_values.layers]
    payload = slice_payload(legacy, 0, len(PROMPT))
    head, tail = split_payload(payload, 4)
    cache = payloads_to_cache([head, tail])
    for i, (k, v) in enumerate(legacy):
        torch.testing.assert_close(cache.layers[i].keys, k)
        torch.testing.assert_close(cache.layers[i].values, v)

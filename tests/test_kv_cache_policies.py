"""KV quantization + eviction policy tests (CPU)."""

import pytest

torch = pytest.importorskip("torch")

from llmopt.cache.eviction import (
    apply_eviction,
    attention_sinks,
    h2o,
    sliding_window,
    snapkv,
)
from llmopt.cache.kv_quant import QuantizedPagedStore, dequantize, quantize
from llmopt.cache.paged import BlockAllocator, BlockTable

BS, HEADS, DIM = 4, 2, 16


def test_quantize_round_trip_error_shrinks_with_bits():
    torch.manual_seed(0)
    x = torch.randn(32, HEADS, DIM)
    err = {}
    for bits in (4, 8):
        codes, scale = quantize(x, bits)
        assert codes.dtype == torch.int8
        assert codes.abs().max() <= 2 ** (bits - 1) - 1
        err[bits] = (dequantize(codes, scale) - x).abs().mean()
    assert err[8] < err[4] < 0.2
    assert err[8] < 0.01


def test_quantized_paged_store_gather_and_cow():
    torch.manual_seed(1)
    store = QuantizedPagedStore(8, BS, HEADS, DIM, bits=8)
    t = store.bind(BlockTable(BlockAllocator(8), BS))
    xs = [torch.randn(HEADS, DIM) for _ in range(BS + 2)]
    for x in xs:
        store.write(t, x, x)

    child = t.fork()
    store.write(child, xs[0], xs[0])  # COW must carry quantized prefix
    k, _ = store.gather(child)
    assert k.shape == (BS + 3, HEADS, DIM)
    assert (k[:-1] - torch.stack(xs)).abs().mean() < 0.01


def test_sliding_window_and_sinks():
    assert sliding_window(10, 4) == [6, 7, 8, 9]
    kept = attention_sinks(100, sinks=2, window=3)
    assert kept == [0, 1, 97, 98, 99]


def test_h2o_keeps_heavy_hitters_and_recents():
    scores = torch.zeros(20)
    scores[[3, 7, 11]] = 10.0  # heavy hitters
    kept = h2o(scores, budget=6, window=2)
    assert {3, 7, 11, 18, 19} <= set(kept)
    assert len(kept) == 6


def test_snapkv_scores_from_observation_window():
    heads, q, k = 2, 6, 16
    attn = torch.full((heads, q, k), 1e-3)
    attn[:, -3:, 5] = 1.0  # position 5 heavily attended by recent queries
    attn[:, :2, 9] = 1.0  # 9 only attended by OLD queries: must not count
    kept = snapkv(attn, budget=5, observe=3)
    assert 5 in kept
    assert 9 not in kept
    assert set(range(k - 3, k)) <= set(kept)  # observation window survives


def test_apply_eviction_end_to_end():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512, attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    prompt = list(range(1, 13))
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([prompt]), use_cache=True)
        keep = attention_sinks(len(prompt), sinks=2, window=4)
        past = apply_eviction(out.past_key_values, keep)
        # decoding continues over the compacted cache
        nxt = model(
            input_ids=torch.tensor([[int(out.logits[0, -1].argmax())]]),
            past_key_values=past,
            cache_position=torch.tensor([len(keep)]),
            position_ids=torch.tensor([[len(prompt)]]),
            use_cache=True,
        )
    assert nxt.logits.shape[1] == 1
    assert nxt.past_key_values.get_seq_length() == len(keep) + 1

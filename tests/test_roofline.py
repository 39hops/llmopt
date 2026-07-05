"""Roofline model: hand-checkable numbers and sanity against known rules
of thumb (decode is memory-bound; batch-1 decode tok/s ~= bw / weight
bytes; long prefill goes compute-bound)."""

import pytest

torch = pytest.importorskip("torch")

from llmopt.eval.roofline import (
    A100_80G,
    Hardware,
    ModelShape,
    op_costs,
    profile_op_times,
    report,
)

# a 7B-ish shape (llama-2-7b): 32 x 4096, 32 heads, MHA, 11008 inter
LLAMA7B = ModelShape(
    layers=32, hidden=4096, heads=32, kv_heads=32, head_dim=128,
    intermediate=11008, vocab=32000,
)


def test_mlp_flops_hand_computed():
    m = ModelShape(
        layers=1, hidden=8, heads=2, kv_heads=2, head_dim=4,
        intermediate=16, vocab=10,
    )
    ops = {o.name: o for o in op_costs(m, batch=1, ctx=0, new_tokens=1)}
    # gate+up+down: 3 matmuls of 8x16 at 1 token -> 3 * 2*8*16 = 768
    assert ops["mlp"].flops == 768
    # qkv: hidden 8 -> (2+2*2)*4 = 24 outs -> 2*8*24 = 384
    assert ops["qkv_proj"].flops == 384


def test_decode_is_memory_bound_prefill_compute_bound():
    decode = report(LLAMA7B, A100_80G, ctx=1024, new_tokens=1)
    assert all(op["bound"] == "memory" for op in decode["ops"])
    assert decode["mfu"] < 0.02  # batch-1 decode wastes the ALUs

    prefill = report(LLAMA7B, A100_80G, ctx=0, new_tokens=4096)
    heavy = {o["name"]: o for o in prefill["ops"]}
    assert heavy["mlp"]["bound"] == "compute"
    assert heavy["qkv_proj"]["bound"] == "compute"
    assert prefill["mfu"] > 0.5


def test_decode_toks_matches_weight_bandwidth_rule():
    # rule of thumb: batch-1 decode tok/s ~= mem_bw / weight bytes
    r = report(LLAMA7B, A100_80G, ctx=512, new_tokens=1)
    weight_bytes = 2 * 6.7e9  # ~7B params fp16
    rough = A100_80G.mem_bw / weight_bytes
    assert rough * 0.7 < r["tok_s"] < rough * 1.3


def test_batching_raises_mfu_not_latency_much():
    b1 = report(LLAMA7B, A100_80G, ctx=512, new_tokens=1, batch=1)
    b16 = report(LLAMA7B, A100_80G, ctx=512, new_tokens=1, batch=16)
    assert b16["mfu"] > 10 * b1["mfu"]  # same weights read, 16x work
    assert b16["latency_s"] < 3 * b1["latency_s"]


def test_gqa_cuts_attention_bytes():
    gqa = ModelShape(
        layers=32, hidden=4096, heads=32, kv_heads=8, head_dim=128,
        intermediate=11008, vocab=32000,
    )
    full = {o.name: o for o in op_costs(LLAMA7B, ctx=4096)}
    grouped = {o.name: o for o in op_costs(gqa, ctx=4096)}
    assert grouped["attention"].bytes < 0.4 * full["attention"].bytes


def test_profile_op_times_runs():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).eval()
    rows = profile_op_times(model, torch.tensor([[1, 2, 3, 4]]), top=5)
    assert len(rows) == 5
    assert all(t >= 0 for _, t in rows)
    assert rows[0][1] >= rows[-1][1]


def test_from_hf_config():
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        vocab_size=100, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
    )
    m = ModelShape.from_hf_config(cfg, dtype_bytes=0.5)
    assert (m.layers, m.kv_heads, m.dtype_bytes) == (2, 2, 0.5)
    assert m.head_dim == 16

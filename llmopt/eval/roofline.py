"""Roofline / MFU model: FLOPs, bytes, and arithmetic intensity per op.

Analytic cost model for transformer inference: for each op (projections,
attention, MLP, lm_head) count FLOPs and bytes moved, compare their
ratio (arithmetic intensity, FLOP/byte) against the hardware ridge
point (peak_flops / mem_bw). Below the ridge the op is memory-bound —
more FLOPs are free; above it, compute-bound — bandwidth is free.

Time per op is the roofline max(flops/peak, bytes/bw); summing gives a
lower-bound latency, tok/s, and MFU (achieved / peak FLOPs). Decode at
batch 1 is memory-bound almost everywhere (intensity ~= 1 for GEMV: one
multiply-add per weight loaded), which is why batching, quantized KV,
and speculative decoding pay — and why fused kernels only pay where an
op is compute- or overhead-bound, not bandwidth-bound.

profile_op_times gives the empirical per-op complement via
torch.profiler on a real forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hardware:
    name: str
    peak_flops: float  # dense FLOP/s at the dtype you run
    mem_bw: float  # bytes/s

    @property
    def ridge(self) -> float:  # FLOP/byte where compute == memory time
        return self.peak_flops / self.mem_bw


# a few common specs (fp16 dense, no sparsity)
A100_80G = Hardware("A100-80G", 312e12, 2.0e12)
RTX_4090 = Hardware("RTX 4090", 165e12, 1.01e12)
M4_PRO = Hardware("M4 Pro", 17e12, 273e9)


@dataclass(frozen=True)
class ModelShape:
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    vocab: int
    dtype_bytes: float = 2.0  # 0.5 for 4-bit weights

    @classmethod
    def from_hf_config(cls, cfg, dtype_bytes: float = 2.0) -> "ModelShape":
        return cls(
            layers=cfg.num_hidden_layers,
            hidden=cfg.hidden_size,
            heads=cfg.num_attention_heads,
            kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=getattr(
                cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads
            ),
            intermediate=cfg.intermediate_size,
            vocab=cfg.vocab_size,
            dtype_bytes=dtype_bytes,
        )

    @property
    def weight_bytes_per_layer(self) -> dict[str, float]:
        h, hd = self.hidden, self.head_dim
        qkv = h * (self.heads + 2 * self.kv_heads) * hd
        return {
            "qkv_proj": qkv * self.dtype_bytes,
            "o_proj": self.heads * hd * h * self.dtype_bytes,
            "mlp": 3 * h * self.intermediate * self.dtype_bytes,
        }


@dataclass(frozen=True)
class OpCost:
    name: str
    flops: float
    bytes: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes

    def bound(self, hw: Hardware) -> str:
        return "memory" if self.intensity < hw.ridge else "compute"

    def time(self, hw: Hardware) -> float:
        return max(self.flops / hw.peak_flops, self.bytes / hw.mem_bw)


def op_costs(
    m: ModelShape, *, batch: int = 1, ctx: int = 1024, new_tokens: int = 1
) -> list[OpCost]:
    """Per-op costs for one forward over ``new_tokens`` positions with
    ``ctx`` tokens already cached. new_tokens=1 is a decode step;
    new_tokens=L, ctx=0 is prefill. FLOPs are 2*MACs; bytes count weight
    reads (once per forward, shared across the batch), KV traffic, and
    activations (usually negligible)."""
    b, n, t = batch, new_tokens, ctx + new_tokens
    h, hd = m.hidden, m.head_dim
    kv_dim = m.kv_heads * hd
    w = m.weight_bytes_per_layer
    act = lambda elems: elems * m.dtype_bytes

    per_layer = [
        OpCost(
            "qkv_proj",
            2 * b * n * h * (m.heads + 2 * m.kv_heads) * hd,
            w["qkv_proj"] + act(b * n * (h + (m.heads + 2 * m.kv_heads) * hd)),
        ),
        OpCost(
            "attention",  # scores + weighted values, GQA-aware KV reads
            2 * b * n * t * m.heads * hd * 2,
            act(2 * b * t * kv_dim)  # read K and V cache
            + act(b * n * m.heads * hd * 2 + b * n * m.heads * t),
        ),
        OpCost(
            "o_proj",
            2 * b * n * m.heads * hd * h,
            w["o_proj"] + act(2 * b * n * h),
        ),
        OpCost(
            "mlp",  # gate, up, down
            3 * 2 * b * n * h * m.intermediate,
            w["mlp"] + act(b * n * (2 * h + 3 * m.intermediate)),
        ),
    ]
    total = [
        OpCost(op.name, op.flops * m.layers, op.bytes * m.layers)
        for op in per_layer
    ]
    total.append(
        OpCost(
            "lm_head",
            2 * b * n * h * m.vocab,
            h * m.vocab * m.dtype_bytes + act(b * n * (h + m.vocab)),
        )
    )
    return total


def report(
    m: ModelShape,
    hw: Hardware,
    *,
    batch: int = 1,
    ctx: int = 1024,
    new_tokens: int = 1,
) -> dict:
    """Roofline summary: per-op rows + predicted latency, tok/s, MFU."""
    ops = op_costs(m, batch=batch, ctx=ctx, new_tokens=new_tokens)
    latency = sum(op.time(hw) for op in ops)
    flops = sum(op.flops for op in ops)
    return {
        "ops": [
            {
                "name": op.name,
                "flops": op.flops,
                "bytes": op.bytes,
                "intensity": op.intensity,
                "bound": op.bound(hw),
                "time_s": op.time(hw),
                "time_frac": op.time(hw) / latency,
            }
            for op in ops
        ],
        "latency_s": latency,
        "tok_s": batch * new_tokens / latency,
        "mfu": flops / latency / hw.peak_flops,
        "ridge": hw.ridge,
    }


def profile_op_times(model, input_ids, top: int = 10) -> list[tuple[str, float]]:
    """Empirical complement: (op_name, self-cpu/cuda time seconds) for one
    forward pass, descending, via torch.profiler."""
    import torch
    from torch.profiler import ProfilerActivity, profile

    acts = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        acts.append(ProfilerActivity.CUDA)
    with torch.inference_mode(), profile(activities=acts) as prof:
        model(input_ids=input_ids)
    rows = sorted(
        ((e.key, e.self_cpu_time_total * 1e-6) for e in prof.key_averages()),
        key=lambda kv: -kv[1],
    )
    return rows[:top]

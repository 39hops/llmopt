"""Deterministic fixed-point decode for MicroLM-class crystals
(promoted from scratch/pack_decode.py; P3 verdict 2026-07-30:
bit-identical logit traces across MPS and cuda, 96.66% argmax
agreement v the fp model with disagreements at coin-flip margins).

Every operation is exact integer arithmetic or a SHIPPED-table
lookup — no libm in the inference path:
- weights: sigma-law integer codes (interface tensors emb/head at
  a finer sigma/8 step);
- GEMMs: the Ozaki-principle exact-fp32 integer carrier with hi/lo
  splitting (every partial < 2^24, asserted);
- RMSNorm: int64 sum-of-squares + integer-Newton isqrt;
- SiLU / softmax-exp / RoPE: precomputed integer tables, generated
  ONCE (make_tables) and shipped as bytes — never regenerated per
  device (libm differences are exactly the thing being excluded).

Determinism holds on any backend with exact int64 elementwise ops
and fp32 multiplies. Reference implementation: ~10-40x slower than
the fp path; determinism, not speed, is the claim.
"""
from __future__ import annotations

import math

A = 1024            # activation fixed-point scale (2^10)
ACT_CLAMP = 32 * A  # |x| <= 32 (tighter clamps real features)
ROPE_S = 1 << 14
EXP_S = 1 << 16
EXP_K = 1024
CTX = 512


def make_tables(state_dict, d: int, heads: int, path: str):
    """Generate the shipped-table file for a crystal. CPU, once."""
    import torch

    hd = d // heads
    t = {}
    for k, v in state_dict.items():
        if v.ndim == 2:
            s = float(v.float().std())
            k_step = 16.0 if k in ("emb.weight", "head.weight") else 2.0
            q = math.ceil(k_step / max(s, 1e-8))
            t[k + ".codes"] = torch.round(v.float() * q).to(torch.int64)
            t[k + ".q"] = torch.tensor(q, dtype=torch.int64)
        elif k.endswith(".g"):
            t[k + ".int"] = torch.round(v.float() * A).to(torch.int64)
    half = hd // 2
    freq = torch.exp(-math.log(10000.0) * torch.arange(half) / half)
    ang = torch.arange(CTX)[:, None].double() * freq[None, :].double()
    t["rope.cos"] = torch.round(ang.cos() * ROPE_S).to(torch.int64)
    t["rope.sin"] = torch.round(ang.sin() * ROPE_S).to(torch.int64)
    idx = torch.arange(-(1 << 15), 1).double() / EXP_K
    t["exp.tab"] = torch.round(idx.exp() * EXP_S).to(torch.int64)
    x = torch.arange(-(1 << 15), (1 << 15) + 1).double() / A
    t["silu.tab"] = torch.round((x * torch.sigmoid(x)) * A).to(torch.int64)
    torch.save(t, path)
    return t


def _rdiv(x, d):
    """Round-half-away integer division; exact and deterministic."""
    import torch

    return torch.where(x >= 0, (2 * x + d) // (2 * d),
                       -((-2 * x + d) // (2 * d)))


def _isqrt_newton(n, iters: int = 30):
    import torch

    x = torch.ones_like(n) << 32
    for _ in range(iters):
        x = (x + _rdiv(n, torch.clamp(x, min=1))) >> 1
    return torch.clamp(x, min=1)


class DeterministicLM:
    """Fixed-point twin of the MicroLM forward. All state int64."""

    def __init__(self, tables_path: str, d: int, layers: int,
                 ffn: int, heads: int, dev: str):
        import torch

        self.d, self.layers, self.heads = d, layers, heads
        self.hd = d // heads
        self.dev = dev
        t = torch.load(tables_path, map_location="cpu",
                       weights_only=True)
        self.t = {k: v.to(dev) for k, v in t.items()}
        self.max_partial = 0

    def _gemm(self, a, key):
        import torch

        W = self.t[key + ".weight.codes"].float()
        q = int(self.t[key + ".weight.q"])
        hi, lo = a >> 6, a & 63
        wmax = int(W.abs().max())
        bound = max(int(hi.abs().max()) if hi.numel() else 0, 63) \
            * wmax * W.shape[1]
        self.max_partial = max(self.max_partial, bound)
        assert bound < (1 << 24), f"partial 2^{math.log2(bound):.1f}"
        Y = ((hi.float() @ W.T).to(torch.int64) << 6) \
            + (lo.float() @ W.T).to(torch.int64)
        return _rdiv(Y, q)

    def _rmsnorm(self, a, gkey):
        import torch

        g = self.t[gkey + ".g.int"]
        s2 = (a * a).sum(-1, keepdim=True)
        m = (s2 // self.d) * (1 << 32) // (A * A) + 4295
        r = _isqrt_newton(m)
        y = _rdiv(a * g, A)
        y = _rdiv(y * (1 << 16), torch.clamp(r, min=1))
        return torch.clamp(y, -ACT_CLAMP, ACT_CLAMP)

    def _rope(self, x, pos0):
        import torch

        T = x.shape[2]
        half = self.hd // 2
        cos = self.t["rope.cos"][pos0:pos0 + T]
        sin = self.t["rope.sin"][pos0:pos0 + T]
        v1, v2 = x[..., :half], x[..., half:]
        return torch.cat([_rdiv(v1 * cos - v2 * sin, ROPE_S),
                          _rdiv(v1 * sin + v2 * cos, ROPE_S)], -1)

    def _attn(self, q, k, v):
        import torch

        s = (q.unsqueeze(-2) * k.unsqueeze(-3)).sum(-1)
        sq = round(math.sqrt(self.hd) * (1 << 14))
        idx = _rdiv(s * EXP_K * (1 << 14), A * A * sq)
        idx = idx - idx.max(dim=-1, keepdim=True).values
        idx = torch.clamp(idx, -(1 << 15), 0)
        w = self.t["exp.tab"][idx + (1 << 15)]
        num = (w.unsqueeze(-1) * v.unsqueeze(-3)).sum(-2)
        return _rdiv(num, torch.clamp(
            w.sum(-1, keepdim=True), min=1))

    def step(self, tok_id, past, pos):
        """One decode step: int64 logits + new KV past."""
        import torch

        q_e = int(self.t["emb.weight.q"])
        x = _rdiv(self.t["emb.weight.codes"][tok_id] * A,
                  q_e).view(1, 1, self.d)
        new_past = []
        for li in range(self.layers):
            p = f"blocks.{li}"
            h = self._rmsnorm(x, f"{p}.n1")
            qkv = self._gemm(h, f"{p}.qkv")
            q, k, v = qkv.split(self.d, dim=-1)
            q = q.view(1, 1, self.heads, self.hd).transpose(1, 2)
            k = k.view(1, 1, self.heads, self.hd).transpose(1, 2)
            v = v.view(1, 1, self.heads, self.hd).transpose(1, 2)
            q, k = self._rope(q, pos), self._rope(k, pos)
            if past is not None:
                k = torch.cat([past[li][0], k], 2)
                v = torch.cat([past[li][1], v], 2)
            new_past.append((k, v))
            a = self._attn(q, k, v).transpose(1, 2).reshape(1, 1, self.d)
            x = torch.clamp(x + self._gemm(a, f"{p}.o"),
                            -ACT_CLAMP, ACT_CLAMP)
            h = self._rmsnorm(x, f"{p}.n2")
            g = self._gemm(h, f"{p}.gate")
            u = self._gemm(h, f"{p}.up")
            gi = torch.clamp(g, -(1 << 15), (1 << 15))
            ff = _rdiv(self.t["silu.tab"][gi + (1 << 15)] * u, A)
            ff = torch.clamp(ff, -(1 << 15), (1 << 15))
            x = torch.clamp(x + self._gemm(ff, f"{p}.down"),
                            -ACT_CLAMP, ACT_CLAMP)
        x = self._rmsnorm(x, "norm")
        return self._gemm(x, "head").squeeze(), new_past

    def greedy(self, ids, n_new: int):
        import torch

        past, logits = None, None
        for pos, t in enumerate(ids):
            logits, past = self.step(
                torch.tensor(t, device=self.dev), past, pos)
        out = []
        for j in range(n_new):
            nxt = int(logits.argmax())
            out.append(nxt)
            logits, past = self.step(
                torch.tensor(nxt, device=self.dev), past,
                len(ids) + j)
        return out

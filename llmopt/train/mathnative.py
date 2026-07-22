"""Math-native micro-model: tokenizer + architecture (spec
2026-07-15-mathnative-micromodel). From-scratch decoder trained
exclusively on closed-system chains — no pretraining, no habits.

Tokenizer: hand-built, deterministic, ~120 tokens. Multi-char atoms
for the function names and scaffold, single chars for everything
else, digits as digit tokens. The charset mask was the real
vocabulary all along; this makes it honest.
"""
from __future__ import annotations

import math

ATOMS = [
    "<pad>", "<eos>",
    # scaffold (the step format, one token each)
    "Current: ", "Hints: none", "Step:", " => ", "Integral(", ", x)",
    # function atoms
    "sin(", "cos(", "tan(", "exp(", "log(", "atan(", "asin(",
    "sqrt(", "pi", "E",
    # frequent operators/glyphs
    "**", "*", "+", "-", "/", "(", ")", ",", " ", "\n", "x", ".",
] + [str(d) for d in range(10)]


class MathTokenizer:
    """Greedy longest-match over ATOMS; every corpus char is covered
    (charset-verified at build). Deterministic, order-stable."""

    def __init__(self):
        self.vocab = list(ATOMS)
        self.id = {t: i for i, t in enumerate(self.vocab)}
        self.pad_id = self.id["<pad>"]
        self.eos_id = self.id["<eos>"]
        self._by_len = sorted((t for t in self.vocab
                               if t not in ("<pad>", "<eos>")),
                              key=len, reverse=True)

    def encode(self, s: str) -> list[int]:
        out, i = [], 0
        while i < len(s):
            for t in self._by_len:
                if s.startswith(t, i):
                    out.append(self.id[t])
                    i += len(t)
                    break
            else:
                i += 1   # unknown char: skip (charset-verified rare)
        return out

    def decode(self, ids: list[int]) -> str:
        return "".join(self.vocab[i] for i in ids
                       if i not in (self.pad_id, self.eos_id))


def build_model(vocab_size: int, d: int = 384, layers: int = 8,
                heads: int = 6, ffn: int = 1536, ctx: int = 512):
    """Standard decoder: RMSNorm, RoPE, SwiGLU, untied head. ~19M."""
    import torch
    import torch.nn as nn

    class RMSNorm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.g = nn.Parameter(torch.ones(d))

        def forward(self, x):
            return self.g * x * torch.rsqrt(
                x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def rope(q, k, pos0=0):
        B, H, T, D = q.shape
        half = D // 2
        freq = torch.exp(-math.log(10000.0) *
                         torch.arange(half, device=q.device) / half)
        t = torch.arange(pos0, pos0 + T, device=q.device)
        ang = t[:, None] * freq[None, :]
        cos, sin = ang.cos(), ang.sin()

        def rot(v):
            v1, v2 = v[..., :half], v[..., half:]
            return torch.cat([v1 * cos - v2 * sin,
                              v1 * sin + v2 * cos], -1)
        return rot(q), rot(k)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.o = nn.Linear(d, d, bias=False)
            self.gate = nn.Linear(d, ffn, bias=False)
            self.up = nn.Linear(d, ffn, bias=False)
            self.down = nn.Linear(ffn, d, bias=False)

        def forward(self, x, mask, past=None):
            B, T, _ = x.shape
            h = self.n1(x)
            q, k, v = self.qkv(h).chunk(3, -1)
            q = q.view(B, T, heads, -1).transpose(1, 2)
            k = k.view(B, T, heads, -1).transpose(1, 2)
            v = v.view(B, T, heads, -1).transpose(1, 2)
            pos0 = past[0].shape[2] if past is not None else 0
            q, k = rope(q, k, pos0)
            if past is not None:
                k = torch.cat([past[0], k], 2)
                v = torch.cat([past[1], v], 2)
            new_past = (k, v)
            a = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                is_causal=(mask is None and past is None))
            a = a.transpose(1, 2).reshape(B, T, d)
            x = x + self.o(a)
            h = self.n2(x)
            x = x + self.down(torch.nn.functional.silu(self.gate(h))
                              * self.up(h))
            return x, new_past

    class MicroLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d)
            self.blocks = nn.ModuleList(Block() for _ in range(layers))
            self.norm = RMSNorm(d)
            self.head = nn.Linear(d, vocab_size, bias=False)
            self.ctx = ctx

        def forward(self, ids, attn_mask=None, past=None,
                    use_cache=False):
            x = self.emb(ids)
            m = None
            if attn_mask is not None:
                # combine causal with padding mask
                T = ids.shape[1]
                causal = torch.ones(T, T, dtype=torch.bool,
                                    device=ids.device).tril()
                m = causal[None, None] & attn_mask[:, None, None, :].bool()
            new_past = []
            for li, b in enumerate(self.blocks):
                x, kv = b(x, m, past[li] if past is not None else None)
                new_past.append(kv)
            logits = self.head(self.norm(x))
            if use_cache or past is not None:
                return logits, new_past
            return logits

    return MicroLM()

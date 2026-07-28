"""Complex-FFN model builder (spec 2026-07-26-complex-zx-program, Leg A).

Interior constraint honored: embeddings/attention/head/residual all
REAL and identical to llmopt.train.mathnative.build_model (copied —
drift risk noted; the attention path is byte-identical code). Only
the FFN changes: its ffn channels pair into ffn/2 complex numbers
(first half = real parts, second half = imag — the RoPE
convention), activation = modReLU on the gate branch, gate*up is a
GENUINE complex multiply, down consumes [re; im]. Param count =
the real block + ffn/2 modReLU biases per layer (~0.03%, noted).

Alphabets (STE on fp32 latents, tournament recipe):
  none — unconstrained complex fp32 (prices modReLU+rotation alone)
  G5   — {0, ±s, ±is} per complex weight, s = absmean(|c|)
"""
import math

import torch
import torch.nn as nn

_ALPHA = "none"


def set_alpha(a: str) -> None:
    global _ALPHA
    _ALPHA = a


def g5_quantize(wr: torch.Tensor, wi: torch.Tensor):
    """Nearest of {0, ±s, ±is} on each complex weight; STE outside."""
    mag = torch.sqrt(wr * wr + wi * wi)
    s = mag.mean().clamp(min=1e-8)
    dead = mag < s / 2
    re_dom = wr.abs() >= wi.abs()
    qr = torch.where(dead | ~re_dom, torch.zeros_like(wr),
                     torch.sign(wr) * s)
    qi = torch.where(dead | re_dom, torch.zeros_like(wi),
                     torch.sign(wi) * s)
    return qr, qi


def gn_quantize(wr: torch.Tensor, wi: torch.Tensor, phases: int):
    """Nearest of {0} u {s*e^(2*pi*i*k/phases)}: exact roots of unity.

    G5's generalization (phases=4 reproduces its alphabet up to the
    axis-snap tie rule). The alphabet is exact at the PHASE level
    (angles are exactly 2*pi*k/N); fp32 storage rounds the coords,
    same claim structure as G5.
    """
    mag = torch.sqrt(wr * wr + wi * wi)
    s = mag.mean().clamp(min=1e-8)
    dead = mag < s / 2
    ang = torch.atan2(wi, wr)
    step = 2 * math.pi / phases
    snapped = torch.round(ang / step) * step
    qr = torch.where(dead, torch.zeros_like(wr), s * torch.cos(snapped))
    qi = torch.where(dead, torch.zeros_like(wi), s * torch.sin(snapped))
    return qr, qi


def zi_quantize(wr: torch.Tensor, wi: torch.Tensor, Q: int = 6):
    """Gaussian-integer rational lattice (RIFF 2026-07-27, Artin):
    re and im EACH snap to s*(best p/q, q<=Q) — weights in
    (s/q')*Z[i], "completely whole" complex numbers. Shared
    per-tensor scale s = component absmean (matches RAT_Q's
    convention on the real substrate)."""
    s = (wr.abs().mean() + wi.abs().mean()).div(2).clamp(min=1e-8)

    def snap(w):
        v = w / s
        best = torch.round(v)
        err = (v - best).abs()
        for q in range(2, Q + 1):
            cand = torch.round(v * q) / q
            e = (v - cand).abs()
            m = e < err
            best = torch.where(m, cand, best)
            err = torch.where(m, e, err)
        return best * s

    return snap(wr), snap(wi)


def quantize_pair(wr: torch.Tensor, wi: torch.Tensor):
    """Route (re, im) through the alphabet named by _ALPHA."""
    if _ALPHA == "G5":
        return g5_quantize(wr, wi)
    if _ALPHA == "ZI":
        return zi_quantize(wr, wi)
    if _ALPHA.startswith("G"):
        return gn_quantize(wr, wi, int(_ALPHA[1:]) - 1)
    return wr, wi


def _q(w: torch.Tensor, pair_dim: int) -> torch.Tensor:
    """STE-quantize a real matrix whose pair_dim halves are (re, im)."""
    if _ALPHA == "none":
        return w
    n = w.shape[pair_dim] // 2
    re = w.narrow(pair_dim, 0, n)
    im = w.narrow(pair_dim, n, n)
    qr, qi = quantize_pair(re, im)
    q = torch.cat([qr, qi], dim=pair_dim)
    return w + (q - w).detach()


def build_complex_model(vocab_size: int, d: int = 384, layers: int = 8,
                        heads: int = 6, ffn: int = 1536, ctx: int = 512):
    class RMSNorm(nn.Module):
        def __init__(self, dd):
            super().__init__()
            self.g = nn.Parameter(torch.ones(dd))

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

    F2 = ffn // 2

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
            self.qkv = nn.Linear(d, 3 * d, bias=False)
            self.o = nn.Linear(d, d, bias=False)
            self.gate = nn.Linear(d, ffn, bias=False)
            self.up = nn.Linear(d, ffn, bias=False)
            self.down = nn.Linear(ffn, d, bias=False)
            self.mod_b = nn.Parameter(torch.zeros(F2))

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
            # complex FFN: rows pair as (re, im) along the ffn dim
            zg = nn.functional.linear(h, _q(self.gate.weight, 0))
            zu = nn.functional.linear(h, _q(self.up.weight, 0))
            gr, gi = zg[..., :F2], zg[..., F2:]
            ur, ui = zu[..., :F2], zu[..., F2:]
            mag = torch.sqrt(gr * gr + gi * gi + 1e-12)
            act = torch.relu(mag + self.mod_b) / mag  # modReLU
            ar, ai = gr * act, gi * act
            pr = ar * ur - ai * ui          # genuine complex multiply
            pi = ar * ui + ai * ur
            p = torch.cat([pr, pi], -1)
            x = x + nn.functional.linear(p, _q(self.down.weight, 1))
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
                T = ids.shape[1]
                causal = torch.ones(T, T, dtype=torch.bool,
                                    device=ids.device).tril()
                m = causal[None, None] & attn_mask[:, None, None, :].bool()
            new_past = []
            use_ckpt = (getattr(self, "grad_ckpt", False)
                        and self.training and past is None)
            for li, b in enumerate(self.blocks):
                if use_ckpt:
                    x, kv = torch.utils.checkpoint.checkpoint(
                        b, x, m, None, use_reentrant=False)
                else:
                    x, kv = b(x, m,
                              past[li] if past is not None else None)
                new_past.append(kv)
            logits = self.head(self.norm(x))
            if use_cache or past is not None:
                return logits, new_past
            return logits

    return MicroLM()

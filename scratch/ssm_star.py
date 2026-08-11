"""SSM-STAR (pre-reg RESULTS PRE-REG SSM-STAR-1): minimal selective
state-space model in the micro-star family — the house's first SSM.

Model: same skeleton as build_model (emb / blocks / RMSNorm / untied
head, SwiGLU FFN unchanged) with the attention sublayer replaced by a
minimal selective SSM (S6-lite): diagonal A, per-channel input-
dependent dt, per-token shared B/C (n_state=16, expand=2). Sequential
scan (Python loop over T) — no custom kernels; at d64 the loop is the
honest price and it is affordable. Pad tokens never enter the state
(dt masked to 0), matching the packed-batch attn_mask semantics.

API-identical to MicroLM: forward(ids, attn_mask=None, past=None,
use_cache=False) -> logits | (logits, states), so TM.main trains it
via the build_model monkeypatch (the rev3_crown/saturation_s2 house
pattern) and gate_eval/sample_wave_lp decode it unchanged (past =
per-layer recurrent state; O(1) per token).

Param note (booked with the verdict): the SSM sublayer carries ~31k
params/block v attention's ~16k at d64 (in_proj 2*expand, x_proj,
A, D, out_proj) — the SSM twin is ~1.2x the attention twin. Param
parity is NOT claimed; the comparison is architecture-at-
its-own-natural-size, disclosed.

Usage:
  train:  python scratch/ssm_star.py train <out.pt> [epochs budget]
  gate:   python scratch/ssm_star.py gate <ckpt.pt> <label>
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

N_STATE = 16
EXPAND = 2


def build_ssm_model(vocab_size: int, d: int = 64, layers: int = 8,
                    heads: int = 4, ffn: int = 256, ctx: int = 512):
    """heads accepted for signature parity; SSM has no heads."""
    import torch
    import torch.nn as nn

    d_inner = EXPAND * d

    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.g = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            return self.g * x * torch.rsqrt(
                x.pow(2).mean(-1, keepdim=True) + 1e-6)

    class SSMBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
            self.in_proj = nn.Linear(d, 2 * d_inner, bias=False)
            # per-token: dt (per channel via low-rank), B and C (n-dim)
            self.x_proj = nn.Linear(d_inner, 1 + 2 * N_STATE, bias=False)
            self.dt_bias = nn.Parameter(torch.zeros(d_inner))
            # A: diagonal, negative, log-spaced 1..n per state dim
            a = torch.arange(1, N_STATE + 1, dtype=torch.float32)
            self.A_log = nn.Parameter(
                torch.log(a)[None, :].repeat(d_inner, 1))
            self.D = nn.Parameter(torch.ones(d_inner))
            self.out_proj = nn.Linear(d_inner, d, bias=False)
            self.gate = nn.Linear(d, ffn, bias=False)
            self.up = nn.Linear(d, ffn, bias=False)
            self.down = nn.Linear(ffn, d, bias=False)

        def forward(self, x, pad_mask=None, past=None):
            # x: (B, T, d); past: (B, d_inner, n) or None
            Bsz, T, _ = x.shape
            h = self.n1(x)
            u, z = self.in_proj(h).chunk(2, -1)          # (B,T,d_inner)
            dbc = self.x_proj(u)
            dt = torch.nn.functional.softplus(
                dbc[..., 0:1] + self.dt_bias)            # (B,T,d_inner)
            Bt = dbc[..., 1:1 + N_STATE]                 # (B,T,n)
            Ct = dbc[..., 1 + N_STATE:]                  # (B,T,n)
            if pad_mask is not None:                     # pads: no write
                dt = dt * pad_mask[..., None]
            A = -torch.exp(self.A_log)                   # (d_inner,n)
            state = past if past is not None else x.new_zeros(
                Bsz, d_inner, N_STATE)
            ys = []
            for t in range(T):
                decay = torch.exp(dt[:, t, :, None] * A)     # (B,di,n)
                write = (dt[:, t, :, None] * u[:, t, :, None]
                         * Bt[:, t, None, :])                # (B,di,n)
                state = state * decay + write
                ys.append(torch.einsum("bdn,bn->bd",
                                       state, Ct[:, t]))
            y = torch.stack(ys, 1) + self.D * u
            y = y * torch.nn.functional.silu(z)
            x = x + self.out_proj(y)
            h2 = self.n2(x)
            x = x + self.down(torch.nn.functional.silu(self.gate(h2))
                              * self.up(h2))
            return x, state

    class SSMLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d)
            self.blocks = nn.ModuleList(SSMBlock() for _ in range(layers))
            self.norm = RMSNorm(d)
            self.head = nn.Linear(d, vocab_size, bias=False)
            self.ctx = ctx

        def forward(self, ids, attn_mask=None, past=None,
                    use_cache=False):
            x = self.emb(ids)
            pm = attn_mask.float() if attn_mask is not None else None
            new_past = []
            for li, b in enumerate(self.blocks):
                x, st = b(x, pm,
                          past[li] if past is not None else None)
                new_past.append(st)
            logits = self.head(self.norm(x))
            if use_cache or past is not None:
                return logits, new_past
            return logits

    return SSMLM()


def main():
    mode = sys.argv[1]
    import train_mathnative as TM
    if mode == "train":
        out = sys.argv[2]
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 3
        budget = int(sys.argv[4]) if len(sys.argv) > 4 else 6144
        TM.build_model = build_ssm_model   # the house monkeypatch
        TM.main(v2=False, d=64, layers=8, ffn=256, heads=4,
                v22=True, gen4=True, fp32=True, epochs=epochs,
                budget=budget, out=out)
    elif mode == "gate":
        import torch
        ckpt, label = sys.argv[2], sys.argv[3]
        import step_grpo_micro as G
        from llmopt.train.mathnative import MathTokenizer
        tok = MathTokenizer()
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if torch.backends.mps.is_available() else "cpu")
        model = build_ssm_model(len(tok.vocab)).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                         weights_only=True))
        model.eval()
        solves, valid = G.gate_eval(model, tok, dev)
        tot = sum(solves.values())
        print(f"{label} gate: {solves} = {tot}/120 @ {valid:.2f}%",
              flush=True)
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()

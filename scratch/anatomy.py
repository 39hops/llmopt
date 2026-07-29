"""Unified crystal anatomy (2026-07-29 spec: slack-restoration).
One env-parameterized census: CELLS=heads,rank,snap on any
MicroLM checkpoint. Frozen originals: head_census.py,
rank_read.py, snap_alloc.py (07-29, d56).

Env: CKPT, D, FFN, HEADS (arch); CELLS (comma list, default
all); RANKS (comma list for rank cell); Q (snap denominator).
Desk only; device = cuda > mps > cpu.
"""
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT = os.environ["CKPT"]
D = int(os.environ.get("D", "56"))
FFN = int(os.environ.get("FFN", "224"))
HEADS = int(os.environ.get("HEADS", "4"))
LAYERS = int(os.environ.get("LAYERS", "8"))
CELLS = os.environ.get("CELLS", "heads,rank,snap").split(",")
RANKS = [int(r) for r in os.environ.get("RANKS", "48,32,24,16").split(",")]
Q = int(os.environ.get("Q", "16"))
HD = D // HEADS

base = torch.load(CKPT, map_location="cpu", weights_only=True)
tok = MathTokenizer()
dev = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
ATTN = [k for li in range(LAYERS)
        for k in (f"blocks.{li}.qkv.weight", f"blocks.{li}.o.weight")]
MLP = [k for li in range(LAYERS)
       for k in (f"blocks.{li}.gate.weight", f"blocks.{li}.up.weight",
                 f"blocks.{li}.down.weight")]


def gate(sd, label):
    m = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
    m.load_state_dict({k: v.to(dev) for k, v in sd.items()})
    m.eval()
    with torch.no_grad():
        solves, valid = G.gate_eval(m, tok, dev)
    print(f"ANATOMY {label}: {solves} = {sum(solves.values())}/120 "
          f"@ {valid:.2f}%", flush=True)
    del m


def snap(w, q_max):
    wf = w.float()
    best = torch.round(wf)
    err = (wf - best).abs()
    for q in range(2, q_max + 1):
        cand = torch.round(wf * q) / q
        e = (wf - cand).abs()
        msk = e < err
        best = torch.where(msk, cand, best)
        err = torch.where(msk, e, err)
    return best.to(w.dtype)


def truncate(W, r):
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).to(W.dtype)


if "heads" in CELLS:
    for h in range(HEADS):
        sd = {k: v.clone() for k, v in base.items()}
        a, b = h * HD, (h + 1) * HD
        for li in range(LAYERS):
            W = sd[f"blocks.{li}.qkv.weight"]
            for part in range(3):
                W[part * D + a:part * D + b, :] = 0
            sd[f"blocks.{li}.o.weight"][:, a:b] = 0
        gate(sd, f"head-drop h{h}")

if "rank" in CELLS:
    for r in RANKS:
        sd = dict(base)
        for k in ATTN:
            sd[k] = truncate(base[k], r)
        gate(sd, f"rank r={r}")

if "snap" in CELLS:
    for name, keys in (("attn", ATTN), ("mlp", MLP),
                       ("both", ATTN + MLP)):
        sd = dict(base)
        for k in keys:
            sd[k] = snap(base[k], Q)
        gate(sd, f"snap Q={Q} {name}")

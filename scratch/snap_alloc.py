"""Snap allocation (pre-reg 2026-07-29: attention anatomy 1c).
Rational snap at Q=16 (below the (16,24] knee) applied to
attention-only v gate-only v both, on the d56 EMA crystal.
Allocation-of-accuracy read for the bits-portfolio riff.
Desk only, MPS.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS, Q = 56, 8, 224, 4, 16

base = torch.load("checkpoints/sym_birth_dense_w56_ema.pt",
                  map_location="cpu", weights_only=True)
attn = [k for li in range(LAYERS)
        for k in (f"blocks.{li}.qkv.weight", f"blocks.{li}.o.weight")]
gate = [k for li in range(LAYERS)
        for k in (f"blocks.{li}.gate.weight", f"blocks.{li}.up.weight",
                  f"blocks.{li}.down.weight")]


def snap(w, q_max):
    wf = w.float()
    best = torch.round(wf)
    err = (wf - best).abs()
    for q in range(2, q_max + 1):
        cand = torch.round(wf * q) / q
        e = (wf - cand).abs()
        m = e < err
        best = torch.where(m, cand, best)
        err = torch.where(m, e, err)
    return best.to(w.dtype)


tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
for name, keys in (("attn-only", attn), ("mlp-only", gate),
                   ("both", attn + gate)):
    sd = dict(base)
    for k in keys:
        sd[k] = snap(base[k], Q)
    m = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
    m.load_state_dict({k2: v.to(dev) for k2, v in sd.items()})
    m.eval()
    with torch.no_grad():
        solves, valid = G.gate_eval(m, tok, dev)
    print(f"SNAP-ALLOC Q={Q} {name}: {solves} = "
          f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)
    del m

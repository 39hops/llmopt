"""Rank read (pre-reg 2026-07-29: attention anatomy 1b). SVD of
all qkv/o weights of the d56 EMA crystal: singular-value decay,
then truncation gates at rank r in {48,32,24,16}. Desk only, MPS.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS = 56, 8, 224, 4

base = torch.load("checkpoints/sym_birth_dense_w56_ema.pt",
                  map_location="cpu", weights_only=True)
keys = [k for li in range(LAYERS)
        for k in (f"blocks.{li}.qkv.weight", f"blocks.{li}.o.weight")]

# singular-value decay: top-k energy fractions per matrix class
for cls in ("qkv", "o"):
    fr = []
    for k in keys:
        if f".{cls}." not in k:
            continue
        s = torch.linalg.svdvals(base[k].float())
        e = s ** 2
        tot = float(e.sum())
        fr.append([float(e[:r].sum()) / tot for r in (16, 24, 32, 48)])
    mean = [sum(c) / len(c) for c in zip(*fr)]
    print(f"SV energy {cls} mean top-{{16,24,32,48}}: "
          f"{[round(v, 4) for v in mean]}", flush=True)


def truncate(W, r):
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vh[:r]


tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
for r in (48, 32, 24, 16):
    sd = dict(base)
    for k in keys:
        sd[k] = truncate(base[k], r).to(base[k].dtype)
    m = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
    m.load_state_dict({k2: v.to(dev) for k2, v in sd.items()})
    m.eval()
    with torch.no_grad():
        solves, valid = G.gate_eval(m, tok, dev)
    print(f"RANK-READ r={r}: {solves} = {sum(solves.values())}/120 "
          f"@ {valid:.2f}%", flush=True)
    del m

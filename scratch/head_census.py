"""Head census (pre-reg 2026-07-29: attention anatomy 1a). Zero
head h of 4 across all layers of the d56 EMA crystal (q,k,v row
blocks in the fused qkv [3D,D] + the o column block), gate each
arm. Desk only, MPS.
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS = 56, 8, 224, 4
HD = D // HEADS  # 14

base = torch.load("checkpoints/sym_birth_dense_w56_ema.pt",
                  map_location="cpu", weights_only=True)
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"

for h in range(HEADS):
    sd = {k: v.clone() for k, v in base.items()}
    rows = slice(h * HD, (h + 1) * HD)
    for li in range(LAYERS):
        W = sd[f"blocks.{li}.qkv.weight"]  # [3D, D], rows q|k|v
        for part in range(3):
            W[part * D + rows.start:part * D + rows.stop, :] = 0
        sd[f"blocks.{li}.o.weight"][:, rows] = 0  # head's output cols
    m = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
    m.load_state_dict({k: v.to(dev) for k, v in sd.items()})
    m.eval()
    with torch.no_grad():
        solves, valid = G.gate_eval(m, tok, dev)
    print(f"HEAD-CENSUS drop h{h}: {solves} = "
          f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)
    del m

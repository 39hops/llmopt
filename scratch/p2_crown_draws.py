"""HARDENING-P2 R8: the crown tie at three fresh problem-set draws
(PRE-REG HARDENING-P2). Both crown artifacts gated PAIRED per draw
at G.GATE_BAND offsets +1M/+2M/+3M, one session, one device.
Deterministic for frozen weights; paired per-draw deltas only.

Usage: .venv/bin/python scratch/p2_crown_draws.py
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")

import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

BASE = G.GATE_BAND
CROWNS = [
    ("gen6_grown", "checkpoints/mathnative_gen6_grown.pt",
     dict(d=512, layers=12, heads=8, ffn=2304)),
    ("merged_grown", "checkpoints/merged_grown.pt",
     dict(d=768, layers=8, heads=12, ffn=3840)),
]
tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else "cpu")
results = {}
for label, ckpt, arch in CROWNS:
    model = build_model(len(tok.vocab), **arch).to(dev)
    model.load_state_dict(torch.load(ckpt, map_location="cpu",
                                     weights_only=True))
    model.eval()
    for off in (1_000_000, 2_000_000, 3_000_000):
        G.GATE_BAND = BASE + off
        solves, valid = G.gate_eval(model, tok, dev)
        tot = sum(solves.values())
        results[(label, off)] = tot
        print(f"[p2-r8] {label} band+{off//1_000_000}M: {solves} = "
              f"{tot}/120 @ {valid:.2f}%", flush=True)
    del model
G.GATE_BAND = BASE
for off in (1_000_000, 2_000_000, 3_000_000):
    d = results[("gen6_grown", off)] - results[("merged_grown", off)]
    print(f"[p2-r8] draw +{off//1_000_000}M paired delta "
          f"(gen6 - merged) = {d:+d}", flush=True)
pooled = sum(results[("gen6_grown", o)] - results[("merged_grown", o)]
             for o in (1_000_000, 2_000_000, 3_000_000))
print(f"[p2-r8] POOLED delta = {pooled:+d} over 3 draws", flush=True)

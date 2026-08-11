"""SATURATION-1 cell (a): +1 warm epoch at fixed food on a COPY of
the grown-s2 champion (PRE-REG SATURATION-1). Crown artifact frozen;
this trains checkpoints/sat_s2.pt (epochs=4 resumes the 3-ep state
one epoch). Then the standard gate, dict printed.
Usage: .venv/bin/python scratch/saturation_s2.py
"""
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402
import train_mathnative as TM  # noqa: E402

# D2 excision: same wrapper class as rev3_crown (covers diet= path)
_lr = TM.load_rows
band = set(gate_band_exprs())
def _load_rows(*a, **k):
    rows = _lr(*a, **k)
    n0 = len(rows)
    rows = [r for r in rows if norm(r["cur"]) not in band]
    print(f"[sat] D2 excision: {n0} -> {len(rows)}", flush=True)
    return rows
TM.load_rows = _load_rows

TM.main(v2=False, d=512, layers=12, ffn=2304, heads=8,
        out="checkpoints/sat_s2.pt", v21=False, fast=False,
        v22=True, gen4=True, l8=True, epochs=4)

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
import step_grpo_micro as G  # noqa: E402
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                ffn=2304).to(dev)
m.load_state_dict(torch.load("checkpoints/sat_s2.pt",
                             map_location="cpu", weights_only=True))
m.eval()
solves, valid = G.gate_eval(m, tok, dev)
print(f"[sat_s2] gate: {solves} = {sum(solves.values())}/120 "
      f"@ {valid:.2f}%", flush=True)

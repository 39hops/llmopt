"""SATURATION-1 cell (b): +1 epoch on WIDENED rations (AMENDMENT
SATURATION-1-CELL-B). Warm corpus + 20% gen-8 slice, string-seeded.
Usage: .venv/bin/python scratch/saturation_s2b.py
"""
import json
import random
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402
import train_mathnative as TM  # noqa: E402

# 1. materialize the widened diet (deterministic)
warm = TM.load_rows(True, True, True, True, True, False, None)
g8 = [json.loads(l) for l in open("data/gen8_diet.jsonl")]
k = int(0.20 * len(warm))
extra = random.Random("sat-b-widen-2026-08-11").sample(g8, k)
mix = warm + extra
print(f"[sat-b] warm {len(warm)} + gen8 {k} = {len(mix)} "
      f"(resident share {len(warm)/len(mix):.3f})", flush=True)
with open("data/sat_b_widened.jsonl", "w") as f:
    for r in mix:
        f.write(json.dumps(r) + "\n")

# 2. D2 excision (same wrapper class as rev3_crown/cell a)
_lr = TM.load_rows
band = set(gate_band_exprs())
def _load_rows(*a, **kw):
    rows = _lr(*a, **kw)
    n0 = len(rows)
    rows = [r for r in rows if norm(r["cur"]) not in band]
    print(f"[sat-b] D2 excision: {n0} -> {len(rows)}", flush=True)
    return rows
TM.load_rows = _load_rows

# 3. +1 epoch on the widened diet, then gate
TM.main(v2=False, d=512, layers=12, ffn=2304, heads=8,
        out="checkpoints/sat_s2b.pt", v21=False, fast=False,
        epochs=4, diet="data/sat_b_widened.jsonl")

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
import step_grpo_micro as G  # noqa: E402
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
m = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                ffn=2304).to(dev)
m.load_state_dict(torch.load("checkpoints/sat_s2b.pt",
                             map_location="cpu", weights_only=True))
m.eval()
solves, valid = G.gate_eval(m, tok, dev)
print(f"[sat_s2b] gate: {solves} = {sum(solves.values())}/120 "
      f"@ {valid:.2f}%", flush=True)

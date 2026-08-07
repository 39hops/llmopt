"""MULT-0 B=32 leg (named follow-up of VERDICT MULT-0; Artin GO):
is choice scarcity a BUDGET property (surface widens with samples)
or a MODEL property (stays thin)? Wrapper only — sets G.B=32 then
runs the cited census driver's path verbatim (tenet_mult_census is
now booked evidence; never edited).

Usage: CKPT=... MULT_LOG=logs/mult0/rev_b32.jsonl \
       .venv/bin/python scratch/tenet_mult_b32.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import step_grpo_micro as G  # noqa: E402

G.B = int(os.environ.get("B", "32"))

import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from tenet_mult_census import mult_census  # noqa: E402

CKPT = os.environ["CKPT"]
tok = MathTokenizer()
dev = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
model = build_model(
    len(tok.vocab), d=int(os.environ.get("D", "64")),
    layers=int(os.environ.get("LAYERS", "8")),
    heads=int(os.environ.get("HEADS", "4")),
    ffn=int(os.environ.get("FFN", "256"))).to(dev)
model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                 weights_only=True))
model.eval()
log_path = Path(os.environ.get("MULT_LOG", "logs/mult0/census_b32.jsonl"))
log_path.parent.mkdir(parents=True, exist_ok=True)
n = int(os.environ["N"]) if os.environ.get("N") else None
print(f"[mult0-b32] B={G.B}", flush=True)
with log_path.open("a") as f:
    mult_census(model, tok, dev, n=n,
                mode=os.environ.get("MODE", "poststep"), log_f=f)

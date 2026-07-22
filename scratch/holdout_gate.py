"""FROZEN HOLDOUT battery (2026-07-21): virgin band 88M, same
L3-L7 x 24 structure as the production gate, run ONLY at
promotions. Includes a corpus-overlap audit (contamination
doctrine: verify the band is virgin, don't assume).
Usage: holdout_gate.py <ckpt> <d> <layers> <ffn> <heads> <label>"""
import sys, glob, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated

HOLD_BAND = 88_000_000
ckpt, d, layers, ffn, heads, label = (sys.argv[1], int(sys.argv[2]),
    int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])

# one-time overlap audit (cheap set check against every corpus file)
probes = []
for lv in G.GATE_LEVELS:
    for i in range(G.GATE_N):
        p = _gen_isolated(lv, HOLD_BAND + 1000 * lv + i)
        if p is not None:
            probes.append((lv, i, sp.sstr(p._expr)))
probe_set = {e.replace(" ", "") for _, _, e in probes}
hits = 0
for f in glob.glob("data/*.jsonl"):
    for line in open(f):
        try:
            r = json.loads(line)
        except Exception:
            continue
        c = r.get("cur", "")
        if c.startswith("Integral(") and \
           c[9:-4].replace(" ", "") in probe_set:
            hits += 1
print(f"[holdout audit] {len(probes)} probes, corpus collisions: "
      f"{hits}", flush=True)

tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                    ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
old = G.GATE_BAND
G.GATE_BAND = HOLD_BAND
solves, valid = G.gate_eval(model, tok, dev)
G.GATE_BAND = old
print(f"{label} HOLDOUT gate: {solves} = "
      f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)

"""Symmetry ladder S1 cells 2-3 (pre-reg 2026-07-28): project the
wfloor d256 gates onto the quaternionic commutant (deletes 75% of
gate mass), gate the projected init, then warm-train 1 epoch.
Arm a: lambda=0; arm b: commutation penalty summed over I,J,K,
ramped 0.1->1.0. R3 recipe verbatim otherwise.
Usage: ARM=a|b python scratch/quat_convert.py
"""
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from quat_commutant import quat_structs, project, anti_mass  # noqa: E402

ARM = os.environ["ARM"]
D, LAYERS, FFN, HEADS, BS = 256, 8, 1024, 4, 32
LAM_MAX = 1.0 if ARM == "b" else 0.0
OUT = f"checkpoints/quat_convert_{ARM}.pt"

torch.manual_seed(1)
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
base = torch.load("checkpoints/mathnative_wfloor_d256.pt",
                  map_location="cpu", weights_only=True)
So = quat_structs(FFN, list(range(FFN)))
Si = quat_structs(D, list(range(D)))
for li in range(LAYERS):  # full projection: keep only the commutant
    k = f"blocks.{li}.gate.weight"
    base[k] = project(base[k].float(), So, Si).to(base[k].dtype)

model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(base)
SoD = {n: m.to(dev) for n, m in So.items()}
SiD = {n: m.to(dev) for n, m in Si.items()}

model.eval()
with torch.no_grad():
    solves, valid = G.gate_eval(model, tok, dev)
print(f"[{ARM}] PROJECTED-INIT gate {solves} = "
      f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)
model.train()

rows = load_rows(gen4=True)
rows = [r for r in rows
        if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
enc = []
for r in rows:
    try:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\n"
                         f"Step: {r['nxt']}\n") + [tok.eos_id]
    except ValueError:
        continue
    if len(ids) <= 512:
        enc.append(ids)
enc.sort(key=len)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4,
                        weight_decay=0.01)
order = list(range(0, len(enc) - BS + 1, BS))
random.Random(0).shuffle(order)
steps_total = len(order)
for bi, off in enumerate(order):
    batch = enc[off:off + BS]
    L = max(len(q) for q in batch)
    x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                      for q in batch], device=dev)
    logits = model(x)[:, :-1]
    y = x[:, 1:]
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
        ignore_index=tok.pad_id)
    if LAM_MAX > 0:
        lam = LAM_MAX * (0.1 + 0.9 * bi / steps_total)
        pen = 0.0
        for li in range(LAYERS):
            W = dict(model.named_parameters())[
                f"blocks.{li}.gate.weight"]
            for nm in ("I", "J", "K"):
                C = W @ SiD[nm] - SoD[nm] @ W
                pen = pen + (C.norm() ** 2) / (W.norm() ** 2)
        loss = loss + lam * pen / (3 * LAYERS)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if bi % 500 == 0:
        print(f"[{ARM}] {bi}/{steps_total} loss {float(loss):.4f}",
              flush=True)

sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
torch.save(sd, OUT)
am = sum(anti_mass(sd[f"blocks.{li}.gate.weight"].float(), So, Si)
         for li in range(LAYERS)) / LAYERS
model.eval()
with torch.no_grad():
    solves, valid = G.gate_eval(model, tok, dev)
print(f"[{ARM}] FINAL gate {solves} = {sum(solves.values())}/120 "
      f"@ {valid:.2f}% | anti-mass {am:.4f}", flush=True)

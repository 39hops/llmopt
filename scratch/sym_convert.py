"""Symmetry ladder S3/S4 (pre-reg 2026-07-28): generic group-
average conversion. GROUP=z2 (sign involution, params/2) or
circ8|circ16 (cyclic shifts within n-blocks, params/n).
P(W) = avg_g R_o(g) W R_i(g)^T (orthogonal reps). Prints
anti-mass read + nulls, projected-init gate, then warm-trains
1 epoch (ARM=a lambda=0 | ARM=b ramped generator penalty).
Usage: GROUP=z2 ARM=b python scratch/sym_convert.py
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

GROUP = os.environ["GROUP"]
ARM = os.environ["ARM"]
D, LAYERS, FFN, HEADS, BS = 256, 8, 1024, 4, 32
LAM_MAX = 1.0 if ARM == "b" else 0.0
OUT = f"checkpoints/sym_{GROUP}_{ARM}.pt"


def reps(n, perm):
    """Group elements as (n,n) orthogonal matrices on perm order."""
    if GROUP == "z2":  # alternating sign involution
        s = torch.eye(n)
        for k in range(n):
            if k % 2:
                s[perm[k], perm[k]] = -1.0
        return [torch.eye(n), s]
    nb = int(GROUP[4:])  # cyclic shifts within nb-blocks
    out = []
    for sh in range(nb):
        M = torch.zeros(n, n)
        for k in range(n // nb):
            idx = perm[nb * k:nb * (k + 1)]
            for r in range(nb):
                M[idx[(r + sh) % nb], idx[r]] = 1.0
        out.append(M)
    return out


def project(W, Ro, Ri):
    return sum(a @ W @ b.T for a, b in zip(Ro, Ri)) / len(Ro)


def anti_mass(W, Ro, Ri):
    return float(1.0 - (project(W, Ro, Ri).norm() ** 2
                        / W.norm() ** 2))


torch.manual_seed(1)
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
base = torch.load("checkpoints/mathnative_wfloor_d256.pt",
                  map_location="cpu", weights_only=True)
Ro = reps(FFN, list(range(FFN)))
Ri = reps(D, list(range(D)))

# instrument fence: synthetic member (idempotence) + random
g = torch.Generator().manual_seed(0)
Wr = torch.randn(FFN, D, generator=g)
mem = project(Wr, Ro, Ri)
print(f"[{GROUP}] control: member {anti_mass(mem, Ro, Ri):.6f} "
      f"(~0) | random {anti_mass(Wr, Ro, Ri):.4f} "
      f"(expect ~{1 - 1 / len(Ro):.3f})", flush=True)

ws = [base[f"blocks.{li}.gate.weight"].float()
      for li in range(LAYERS)]
adj = sum(anti_mass(w, Ro, Ri) for w in ws) / LAYERS
nulls = []
for s in range(10):
    gg = torch.Generator().manual_seed(s)
    ro = reps(FFN, torch.randperm(FFN, generator=gg).tolist())
    ri = reps(D, torch.randperm(D, generator=gg).tolist())
    nulls.append(sum(anti_mass(w, ro, ri) for w in ws) / LAYERS)
mu = sum(nulls) / len(nulls)
sd_ = (sum((x - mu) ** 2 for x in nulls) / len(nulls)) ** 0.5
print(f"[{GROUP}] wfloor adj {adj:.5f} | null {mu:.5f} +- {sd_:.5f}"
      f" | z {(adj - mu) / max(sd_, 1e-12):+.2f}", flush=True)

for li in range(LAYERS):
    k = f"blocks.{li}.gate.weight"
    base[k] = project(base[k].float(), Ro, Ri).to(base[k].dtype)

model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(base)
gen_o = [m.to(dev) for m in Ro[1:2]]  # one generator suffices
gen_i = [m.to(dev) for m in Ri[1:2]]  # (cyclic: shift-1; z2: S)

model.eval()
with torch.no_grad():
    solves, valid = G.gate_eval(model, tok, dev)
print(f"[{GROUP}/{ARM}] PROJECTED-INIT gate {solves} = "
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
            for go, gi in zip(gen_o, gen_i):
                C = W @ gi - go @ W
                pen = pen + (C.norm() ** 2) / (W.norm() ** 2)
        loss = loss + lam * pen / LAYERS
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if bi % 500 == 0:
        print(f"[{GROUP}/{ARM}] {bi}/{steps_total} "
              f"loss {float(loss):.4f}", flush=True)

sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
torch.save(sd, OUT)
am = sum(anti_mass(sd[f"blocks.{li}.gate.weight"].float(), Ro, Ri)
         for li in range(LAYERS)) / LAYERS
model.eval()
with torch.no_grad():
    solves, valid = G.gate_eval(model, tok, dev)
print(f"[{GROUP}/{ARM}] FINAL gate {solves} = "
      f"{sum(solves.values())}/120 @ {valid:.2f}% | "
      f"anti-mass {am:.4f}", flush=True)

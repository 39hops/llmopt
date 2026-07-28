"""Symmetry-at-birth (pre-reg 2026-07-28 night): C8 at d64, from
SCRATCH. Arm dense = plain birth control; arm c8 = commutant-
projected init + ramped generator penalty from step 0. Paired on
one device, seed 1, lr 1.5e-3, bs 8, gen-4, 3 epochs.
Usage: ARM=dense|c8 python scratch/sym_birth.py
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

ARM = os.environ["ARM"]
D, LAYERS, FFN, HEADS, BS, EPOCHS, LR = 64, 8, 256, 4, 8, 3, 1.5e-3
OUT = f"checkpoints/sym_birth_{ARM}.pt"
NB = 8  # C8 blocks


def shift_reps(n):
    out = []
    for sh in range(NB):
        M = torch.zeros(n, n)
        for k in range(n // NB):
            for r in range(NB):
                M[NB * k + (r + sh) % NB, NB * k + r] = 1.0
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
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
Ro, Ri = shift_reps(FFN), shift_reps(D)
if ARM == "c8":
    with torch.no_grad():
        for li in range(LAYERS):
            W = dict(model.named_parameters())[
                f"blocks.{li}.gate.weight"]
            W.copy_(project(W.cpu().float(), Ro, Ri).to(dev))
go = Ro[1].to(dev)  # shift-1 generates C8
gi = Ri[1].to(dev)

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
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
order = list(range(0, len(enc) - BS + 1, BS))
steps_total = len(order) * EPOCHS
step = 0
for ep in range(EPOCHS):
    random.Random(ep).shuffle(order)
    for off in order:
        batch = enc[off:off + BS]
        L = max(len(q) for q in batch)
        x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                          for q in batch], device=dev)
        logits = model(x)[:, :-1]
        y = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id)
        if ARM == "c8":
            lam = 0.1 + 0.9 * step / steps_total
            pen = 0.0
            for li in range(LAYERS):
                W = dict(model.named_parameters())[
                    f"blocks.{li}.gate.weight"]
                C = W @ gi - go @ W
                pen = pen + (C.norm() ** 2) / (W.norm() ** 2)
            loss = loss + lam * pen / LAYERS
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 2000 == 0:
            print(f"[{ARM}] {step}/{steps_total} "
                  f"loss {float(loss):.4f}", flush=True)
        step += 1

sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
torch.save(sd, OUT)
am = sum(anti_mass(sd[f"blocks.{li}.gate.weight"].float(), Ro, Ri)
         for li in range(LAYERS)) / LAYERS
model.eval()
with torch.no_grad():
    solves, valid = G.gate_eval(model, tok, dev)
print(f"[{ARM}] BIRTH gate {solves} = {sum(solves.values())}/120 "
      f"@ {valid:.2f}% | anti-mass {am:.4f}", flush=True)

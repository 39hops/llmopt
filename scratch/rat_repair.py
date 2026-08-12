"""Snap+repair (RIFF 2026-07-27, precision-as-thin-film): take a
snapped crystal, FREEZE every 2-D tensor (the exact lattice stays
exact), train only the 1-D parameters (norms/biases — the 'thin
precise film') briefly on the birth diet, save. If a few thousand
precise params recover the snap deficit, precision is a small
additive budget, not a per-weight property.
Usage: rat_repair.py <ckpt_in> <diet_jsonl> <steps> <ckpt_out>
Env: VOCAB_EXTRA (must match birth), shape via D/LAYERS/FFN/HEADS.
"""
import json
import os
import random
import sys

import torch
import torch.nn.functional as F

from llmopt.train.mathnative import MathTokenizer, build_model

ckpt_in, diet, steps, ckpt_out = (sys.argv[1], sys.argv[2],
                                  int(sys.argv[3]), sys.argv[4])
_extra = os.environ.get("VOCAB_EXTRA", "")
tok = MathTokenizer(extra=_extra.split(",") if _extra else None)
d = int(os.environ.get("D", "512"))
model = build_model(len(tok.vocab), d=d,
                    layers=int(os.environ.get("LAYERS", "12")),
                    heads=int(os.environ.get("HEADS", "8")),
                    ffn=int(os.environ.get("FFN", "2048"))).to("cuda")
model.load_state_dict(torch.load(ckpt_in, map_location="cpu"))

frozen = trained = 0
params = []
for p in model.parameters():
    if p.ndim >= 2:
        p.requires_grad_(False)
        frozen += p.numel()
    else:
        params.append(p)
        trained += p.numel()
print(f"frozen {frozen/1e6:.1f}M (exact lattice), training {trained/1e3:.1f}k film params", flush=True)

rows = []
rng = random.Random("rat-repair-2026-07-27")
for line in open(diet):
    r = json.loads(line)
    try:
        ids = tok.encode(f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n")
    except Exception:
        continue
    if 16 <= len(ids) <= 256:
        rows.append(ids)
    if len(rows) >= 20000:
        break
rng.shuffle(rows)
opt = torch.optim.AdamW(params, lr=1e-3)
model.train()
BS = 16
for step in range(steps):
    batch_rows = rows[(step * BS) % (len(rows) - BS):][:BS]
    L = max(len(r) for r in batch_rows)
    x = torch.zeros((BS, L), dtype=torch.long)
    y = torch.full((BS, L), -100, dtype=torch.long)
    for i, r in enumerate(batch_rows):
        x[i, :len(r)] = torch.tensor(r)
        y[i, :len(r) - 1] = torch.tensor(r[1:])
    x, y = x.to("cuda"), y.to("cuda")
    out = model(x)
    out = out[0] if isinstance(out, tuple) else out
    loss = F.cross_entropy(out.view(-1, out.shape[-1]), y.view(-1),
                           ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step} loss {loss.item():.4f}", flush=True)
model.eval()
torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt_out)
print(f"repaired -> {ckpt_out}", flush=True)

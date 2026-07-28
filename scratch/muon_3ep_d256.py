"""Muon at the STANDARD 3-epoch schedule (null-revival mix, pre-reg
2026-07-28 night): the Muon crater (10/34) was measured only in
single-pass streaming with LR coupled to the surprise rider; the
banked variants row says published Muon wins live at standard
schedules. One cell: control construction (length-sorted BS=32,
shuffled batch order, 3ep) with Muon (ns5 orthogonalized momentum)
on 2-D interior weights, AdamW (OneCycle 3e-4) on embeddings/head/
norms. Comparator wfloor_d256 65 (same construction, all-AdamW).
"""
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
BS, EPOCHS = 32, 3
MUON_LR = float(os.environ.get("MUON_LR", "0.01"))
OUT = "checkpoints/muon3ep_d256.pt"

torch.manual_seed(1)
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
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
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)

muon_p = [p for n, p in model.named_parameters()
          if p.dim() == 2 and "emb" not in n and "head" not in n]
muon_ids = {id(p) for p in muon_p}
adam_p = [p for p in model.parameters() if id(p) not in muon_ids]
opt = torch.optim.AdamW(adam_p, lr=3e-4, weight_decay=0.01)
steps_total = EPOCHS * (len(enc) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=3e-4, total_steps=steps_total, pct_start=0.03)
mom = [torch.zeros_like(p) for p in muon_p]


def ns5(g, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.float() / (g.norm() + 1e-7)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(steps):
        A = x @ x.T
        x = a * x + (b * A + c * A @ A) @ x
    return (x.T if tall else x).to(g.dtype)


step = 0
for ep in range(EPOCHS):
    order = list(range(0, len(enc) - BS + 1, BS))
    random.Random(ep).shuffle(order)
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
        for p in model.parameters():
            p.grad = None
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for k, p in enumerate(muon_p):
                if p.grad is None:
                    continue
                mom[k].mul_(0.95).add_(p.grad)
                p.add_(ns5(mom[k]), alpha=-MUON_LR)
        step += 1
        if step % 500 == 0:
            print(f"ep{ep} step {step}/{steps_total} "
                  f"loss {float(loss):.4f}", flush=True)
torch.save(model.state_dict(), OUT)
print(f"saved {OUT}", flush=True)

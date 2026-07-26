"""Streaming-birth A/B, arm S (RIFF-LEDGER 2026-07-24 "Streaming birth").

One variable vs the wfloor d256 control (gen-4, 3ep OneCycle,
BIRTH_SEED=1, gate 65): the corpus is streamed ONCE — no epochs —
with surprise-gated LR (per-batch multiplier = batch loss over its
EMA, clamped). Init, seed, data, strict encode, batching (BS=32,
one shuffle), clip, AdamW all match the trainer. The claim under
test is SPEED (birth wall /3), so the pre-registered pass is
capability-NEUTRALITY within band, not a win (RESULTS 2026-07-26
pre-reg; Z1-degeneracy red-team rider carried).
"""
import os
import random
import sys
import time

sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from scripts.train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
BS = 32
BASE_LR = 3e-4
WARMUP = 200
EMA_DECAY = 0.99
CLAMP = (0.25, 4.0)
OUT = "checkpoints/mathnative_wfloor_d256_stream.pt"

tok = MathTokenizer()
rows = load_rows(gen4=True)
rows = [r for r in rows
        if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
enc = []
skipped = 0
for r in rows:
    t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
    try:
        ids = tok.encode(t) + [tok.eos_id]
    except ValueError:
        skipped += 1
        continue
    if len(ids) <= 512:
        enc.append(ids)
print(f"{len(enc)} sequences (skipped {skipped}), vocab {len(tok.vocab)}",
      flush=True)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(int(os.environ.get("BIRTH_SEED", "1")))
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
      f"params on {dev} [streaming: 1 pass, surprise-gated LR]",
      flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                        weight_decay=0.01)

perm = list(range(len(enc)))
random.Random(1).shuffle(perm)
batches = [perm[i:i + BS] for i in range(0, len(perm), BS)]

V2 = os.environ.get("STREAM_V2") == "1"
V3 = os.environ.get("STREAM_V3") == "1"
V4 = os.environ.get("STREAM_V4") == "1"
# Muon cell (pre-reg 2026-07-26): orthogonalized momentum on 2D
# interior weights = diversity-per-step moved into the optimizer.
# MUON=sorted arm (v3 construction, comparator 45),
# MUON_MIXED=mixed arm (v4 construction, comparator 57).
MUON = os.environ.get("STREAM_MUON") == "1"
MUON_MIXED = os.environ.get("STREAM_MUON_MIXED") == "1"
MUON_LR = float(os.environ.get("MUON_LR", "0.02"))


def ns5(G, steps=5):
    # Newton-Schulz orthogonalization (Muon; Jordan et al. coeffs)
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tp = X.shape[0] > X.shape[1]
    if tp:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if tp else X


if MUON or MUON_MIXED:
    interior = [p for n, p in model.named_parameters()
                if p.dim() == 2 and "emb" not in n
                and p.shape[0] != len(tok.vocab)]
    others = [p for n, p in model.named_parameters()
              if not (p.dim() == 2 and "emb" not in n
                      and p.shape[0] != len(tok.vocab))]
    opt = torch.optim.AdamW(others, lr=BASE_LR, weight_decay=0.01)
    mom = [torch.zeros_like(p) for p in interior]
    if MUON_MIXED:
        OUT = "checkpoints/mathnative_wfloor_d256_muon_mx.pt"
    else:
        order = sorted(range(len(enc)), key=lambda j: len(enc[j]))
        batches = [order[i:i + BS] for i in range(0, len(order), BS)]
        random.Random(1).shuffle(batches)
        OUT = "checkpoints/mathnative_wfloor_d256_muon.pt"
if V4:
    # v4 (the missing 2x2 cell, pre-reg 2026-07-26): v1's MIXED
    # shuffled batches (iid, padded) + v3's final-10% cooldown.
    # Decides cooldown-vs-batch-construction for the 53 -> 45 drop.
    OUT = "checkpoints/mathnative_wfloor_d256_stream4.pt"
if V3:
    # v3 (clean cooldown isolation, pre-reg 2026-07-26): v1's exact
    # constant-LR profile + final-10% linear decay to zero.
    # Length-sorted batches (speed fix, shared with v2). Single
    # variable vs v1: ends-hot vs ends-cold (integral-LR ~0.90 v 0.95).
    order = sorted(range(len(enc)), key=lambda j: len(enc[j]))
    batches = [order[i:i + BS] for i in range(0, len(order), BS)]
    random.Random(1).shuffle(batches)
    OUT = "checkpoints/mathnative_wfloor_d256_stream3.pt"
if V2:
    # v2 (cooldown arm, pre-reg 2026-07-26): length-sorted batches
    # (control's construction — fixes the speed leg) + OneCycle
    # compressed into the single pass (anneals to zero — isolates
    # revisits from cooldown). Surprise multiplier stays.
    order = sorted(range(len(enc)), key=lambda j: len(enc[j]))
    batches = [order[i:i + BS] for i in range(0, len(order), BS)]
    random.Random(1).shuffle(batches)  # shuffle BATCHES, keep
    # length-homogeneous composition
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=BASE_LR, total_steps=len(batches))
    OUT = "checkpoints/mathnative_wfloor_d256_stream2.pt"

ema = None
t0 = time.time()
for step, b in enumerate(batches):
    batch = [enc[j] for j in b]
    L = max(len(s) for s in batch)
    ids = torch.tensor([s + [tok.pad_id] * (L - len(s)) for s in batch],
                       device=dev)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                         for s in batch], device=dev)
    logits = model(ids[:, :-1], mask[:, :-1])
    labels = ids[:, 1:].clone()
    labels[mask[:, 1:] == 0] = -100
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), ignore_index=-100)
    lv = float(loss.detach())
    ema = lv if ema is None else EMA_DECAY * ema + (1 - EMA_DECAY) * lv
    surprise = max(CLAMP[0], min(CLAMP[1], lv / max(ema, 1e-8)))
    if V2:
        base = sched.get_last_lr()[0]
        for g in opt.param_groups:
            g["lr"] = base * surprise
    else:
        warm = min(1.0, (step + 1) / WARMUP)
        cool = 1.0
        if V3 or V4 or MUON or MUON_MIXED:
            tail = len(batches) // 10
            left = len(batches) - step
            if left <= tail:
                cool = left / tail
        for g in opt.param_groups:
            g["lr"] = BASE_LR * warm * cool * surprise
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    if MUON or MUON_MIXED:
        mult = warm * cool * surprise
        with torch.no_grad():
            for p, m in zip(interior, mom):
                if p.grad is None:
                    continue
                m.mul_(0.95).add_(p.grad)
                u = ns5(m + 0.95 * p.grad)  # nesterov
                p.add_(u, alpha=-MUON_LR * mult
                       * max(1, p.shape[0] / p.shape[1]) ** 0.5)
                p.grad = None
    if V2:
        for g in opt.param_groups:
            g["lr"] = base  # restore before sched.step (OneCycle
            # multiplies its own trajectory; surprise is per-step)
        sched.step()
    if (step + 1) % 200 == 0:
        r = (step + 1) / (time.time() - t0)
        print(f"  {step+1}/{len(batches)} loss {lv:.3f} ema {ema:.3f} "
              f"surprise {surprise:.2f} ({r:.1f} it/s)", flush=True)

torch.save(model.state_dict(), OUT)
print(f"saved {OUT}  wall {time.time()-t0:.0f}s", flush=True)

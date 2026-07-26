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
    warm = min(1.0, (step + 1) / WARMUP)
    for g in opt.param_groups:
        g["lr"] = BASE_LR * warm * surprise
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    if (step + 1) % 200 == 0:
        r = (step + 1) / (time.time() - t0)
        print(f"  {step+1}/{len(batches)} loss {lv:.3f} ema {ema:.3f} "
              f"surprise {surprise:.2f} ({r:.1f} it/s)", flush=True)

torch.save(model.state_dict(), OUT)
print(f"saved {OUT}  wall {time.time()-t0:.0f}s", flush=True)

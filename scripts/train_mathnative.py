"""Phase 1: pretrain the math-native micro-model on the farmed diet.

Diet: data/micromodel_chains_shard*.jsonl (94.5k engine-minted
pairs) + the purged main corpus. Full-sequence LM loss on the step
format — this text is the model's entire world. Gate (spec): >=1%
step validity at L2-3, temp 0.7, before phase 2 (GRPO from birth).

    .venv/bin/python scripts/train_mathnative.py
"""
from __future__ import annotations

import glob
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.train.mathnative import MathTokenizer, build_model

CKPT = Path("checkpoints/mathnative_19m.pt")
V2_CKPT = Path("checkpoints/mathnative_19m_v2.pt")
EPOCHS = 3
BS = 32
LR = 3e-4  # from-scratch: standard small-LM lr with warmup+cosine


def load_rows(v2: bool = False):
    rows = []
    for f in sorted(glob.glob("data/micromodel_chains_shard*.jsonl")):
        rows += [json.loads(l) for l in open(f)]
    rows += [json.loads(l) for l in open("data/step_chains.jsonl")]
    if v2:  # curriculum v2: + the algebra substrate (30k rewrites)
        for f in sorted(glob.glob("data/micromodel_algebra_shard*.jsonl")):
            rows += [json.loads(l) for l in open(f)]
    # identity guard at the diet gate too (defense in depth)
    rows = [r for r in rows
            if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
    return rows


def main(v2: bool = False, d: int = 384, layers: int = 8,
         ffn: int = 1536, out: str | None = None,
         heads: int = 6) -> None:
    import torch
    global CKPT
    if v2:
        CKPT = V2_CKPT
    if out:
        CKPT = Path(out)
    tok = MathTokenizer()
    rows = load_rows(v2)
    charset = set()
    texts = []
    for r in rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        texts.append(t)
        charset |= set(t)
    uncovered = {c for c in charset
                 if not tok.encode(c) and c not in (" ",)}
    print(f"{len(rows)} rows; charset {len(charset)}; "
          f"uncovered chars: {sorted(uncovered)[:10]}", flush=True)

    enc = []
    for t in texts:
        ids = tok.encode(t) + [tok.eos_id]
        if len(ids) <= 512:
            enc.append(ids)
    enc.sort(key=len)
    print(f"{len(enc)} sequences, vocab {len(tok.vocab)}", flush=True)

    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    model = build_model(len(tok.vocab), d=d, layers=layers,
                        heads=heads, ffn=ffn).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M params on {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    steps_total = EPOCHS * (len(enc) // BS)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=steps_total, pct_start=0.03)

    marker = Path(str(CKPT) + ".ep")
    start_ep = 0
    if marker.exists() and CKPT.exists():
        start_ep = int(marker.read_text()) + 1
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        model.to(dev)
        print(f"resuming at epoch {start_ep}", flush=True)

    for ep in range(start_ep, EPOCHS):
        idx = list(range(0, len(enc) - BS, BS))
        random.Random(ep).shuffle(idx)   # per-epoch order shuffle
        tot = steps = 0
        t0 = time.time()
        for n, i in enumerate(idx):
            batch = enc[i:i + BS]
            L = max(len(s) for s in batch)
            ids = torch.tensor([s + [tok.pad_id] * (L - len(s))
                                for s in batch], device=dev)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                                 for s in batch], device=dev)
            logits = model(ids[:, :-1], mask[:, :-1])
            labels = ids[:, 1:].clone()
            labels[mask[:, 1:] == 0] = -100
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += float(loss.detach())
            steps += 1
            if steps % 200 == 0:
                r = steps / (time.time() - t0)
                print(f"  ep{ep} {steps}/{len(idx)} loss "
                      f"{tot/steps:.3f} ({r:.1f} it/s)", flush=True)
        print(f"epoch {ep}: loss {tot/max(steps,1):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        torch.save(model.state_dict(), CKPT)
        marker.write_text(str(ep))
    marker.unlink(missing_ok=True)
    print(f"saved {CKPT}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true",
                    help="curriculum v2 diet (+algebra shard), "
                         "separate checkpoint")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--out", default=None,
                    help="checkpoint path override (capacity runs)")
    a = ap.parse_args()
    main(a.v2, a.d, a.layers, a.ffn, a.out, a.heads)

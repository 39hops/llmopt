"""Self-distillation consolidation (post-climb strategy item B).

RL explores, SFT consolidates: one low-LR epoch on the GRPO-mined
on-policy sidecar, starting FROM the promoted checkpoint (never from
scratch — the lottery is dead). Rows are level-capped before training
(the sidecar is 71% L5; the coeff-flood scar says uncapped class
floods unlearn neighbors). Gate the result with the usual GATE_N=24
chain gate against the promoted model's numbers before adopting.
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmopt.train.mathnative import MathTokenizer, build_model

SIDECAR = Path("data/micromodel_grpo_mined.jsonl")
BS = 32


def main(src: str, out: str, lr: float, cap: int, d: int,
         layers: int, ffn: int, heads: int, seed: int) -> None:
    import torch

    rows = [json.loads(l) for l in SIDECAR.open()]
    rows = [r for r in rows
            if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
    by_lv = defaultdict(list)
    for r in rows:
        by_lv[r["level"]].append(r)
    rng = random.Random(f"consolidate-{seed}")
    kept = []
    for lv, rs in sorted(by_lv.items()):
        rng.shuffle(rs)
        kept += rs[:cap]
    print(f"sidecar {len(rows)} rows -> level-capped {len(kept)} "
          f"({ {lv: min(len(rs), cap) for lv, rs in sorted(by_lv.items())} })",
          flush=True)

    tok = MathTokenizer()
    enc = []
    for r in kept:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        ids = tok.encode(t) + [tok.eos_id]
        if len(ids) <= 512:
            enc.append(ids)
    enc.sort(key=len)
    print(f"{len(enc)} sequences", flush=True)

    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(tok.vocab), d=d, layers=layers,
                        heads=heads, ffn=ffn).to(dev)
    model.load_state_dict(torch.load(src, map_location="cpu"))
    model.to(dev).train()
    print(f"consolidating {src} on {dev} @ lr {lr}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=0.01)

    starts = [(i, i + BS) for i in range(0, len(enc), BS)]
    rng.shuffle(starts)
    tot = steps = 0
    t0 = time.time()
    for lo, hi in starts:
        batch = enc[lo:hi]
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
        opt.zero_grad()
        tot += float(loss.detach())
        steps += 1
        if steps % 100 == 0:
            print(f"  {steps}/{len(starts)} loss {tot/steps:.3f} "
                  f"({steps/(time.time()-t0):.1f} it/s)", flush=True)
    print(f"consolidation: loss {tot/max(steps,1):.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    torch.save(model.state_dict(), out)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--cap", type=int, default=5500,
                    help="max rows per level (sidecar is 71% L5)")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1536)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.src, a.out, a.lr, a.cap, a.d, a.layers, a.ffn, a.heads,
         a.seed)

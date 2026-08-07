"""TENET W1-S: the feature-surface ladder on the EXISTING W1
population (queued by Artin 2026-08-06 behind the GT-7 booking;
LOCKSTEP spec C4 rung 1). VERDICT TENET-W1 read direction at chance
(10/20) from FFN GATE rows; the rider proved the instrument sound
(randinit control 20/20). Question: does direction live in any OTHER
weight surface at this scale?

Protocol IDENTICAL to scratch/tenet_w1_bridge.py (reader class,
seed-split with pairs intact, 16-vote eval, EPOCHS/LR/SEED defaults;
those pieces are copied verbatim) — only the SUBJECT SURFACE varies:

  SURFACE=attn  qkv.weight (192x64) + o.weight (64x64) per block
                -> [8, 256, 64]
  SURFACE=up    up.weight per block -> [8, 256, 64]
  SURFACE=emb   emb.weight (40x64) + head.weight (40x64), one
                pseudo-block -> [1, 80, 64]
  SURFACE=ln    n1.g + n2.g per block as 1x64 rows -> [8, 2, 64]
                (all rows used every pass — the resampling
                augmentation is VACUOUS here, noted not hidden)

ONE-DEVICE CONTRACT: runs on the 3080 (population lives there).
CONTROL=randinit swaps class-1 subjects for fresh inits of the same
architecture (the TENET-W1 rider control, generalized per surface) —
required before any FIRING surface is claimed.

Usage: SURFACE=attn .venv/bin/python scratch/tenet_w1_surfaces.py
Env: as bridge (MANIFEST EVAL_PAIRS TOK_PER_BLOCK EPOCHS LR SEED).
"""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

SURFACE = os.environ.get("SURFACE", "attn")
DMODEL = int(os.environ.get("DMODEL", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR = float(os.environ.get("LR", "1e-3"))
SEED = int(os.environ.get("SEED", "1"))
EVAL_PAIRS = int(os.environ.get("EVAL_PAIRS", "10"))
D_IN = 64

SURF_BLOCKS = {"attn": 8, "up": 8, "emb": 1, "ln": 8}
SURF_TPB = {"attn": 32, "up": 32, "emb": 32, "ln": 2}
BLOCKS = SURF_BLOCKS[SURFACE]
TOK_PER_BLOCK = int(os.environ.get("TOK_PER_BLOCK",
                                   str(SURF_TPB[SURFACE])))


def load_surface(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    if SURFACE == "attn":
        return torch.stack([torch.cat([
            sd[f"blocks.{li}.qkv.weight"],
            sd[f"blocks.{li}.o.weight"]]).float() for li in range(8)])
    if SURFACE == "up":
        return torch.stack([sd[f"blocks.{li}.up.weight"].float()
                            for li in range(8)])
    if SURFACE == "emb":
        return torch.cat([sd["emb.weight"],
                          sd["head.weight"]]).float()[None]
    if SURFACE == "ln":
        return torch.stack([torch.stack([
            sd[f"blocks.{li}.n1.g"],
            sd[f"blocks.{li}.n2.g"]]).float() for li in range(8)])
    raise SystemExit(f"unknown SURFACE {SURFACE!r}")


def surface_from_model(m):
    """The same stacking applied to a live model's state dict
    (CONTROL=randinit path)."""
    import tempfile

    # per-call temp file (code review 2026-08-07: the fixed /tmp path
    # raced concurrent SURFACE arms; relational's dict-direct
    # load_subject is the better pattern for any v2)
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(m.state_dict(), f.name)
        return load_surface(f.name)


# ---- verbatim protocol pieces from scratch/tenet_w1_bridge.py ----

def tokenize(gates, rng):
    toks, bids = [], []
    for li in range(BLOCKS):
        idx = rng.sample(range(gates.shape[1]), TOK_PER_BLOCK)
        toks.append(gates[li, idx])
        bids += [li] * TOK_PER_BLOCK
    return torch.cat(toks), torch.tensor(bids)


class DirectionReader(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(D_IN, DMODEL)
        self.bid = nn.Embedding(BLOCKS, DMODEL)
        layer = nn.TransformerEncoderLayer(
            DMODEL, 4, 4 * DMODEL, batch_first=True,
            norm_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(layer, 2)
        self.head = nn.Linear(DMODEL, 2)

    def forward(self, toks, bids):
        x = self.proj(toks) + self.bid(bids)
        return self.head(self.enc(x).mean(dim=1))


def main():
    torch.manual_seed(SEED)
    manifest = Path(os.environ.get(
        "MANIFEST", "data/w1_population_manifest.jsonl"))
    rows = [json.loads(l) for l in manifest.open()]
    ok = [r for r in rows if "terminal" not in r and r.get("rc") == 0]
    by_seed = {}
    for r in ok:
        by_seed.setdefault(r["seed"], {})[r["direction"]] = r["ckpt"]
    pairs = sorted(s for s, d in by_seed.items()
                   if {"fwd", "rev"} <= set(d))
    print(f"[w1s] SURFACE={SURFACE} complete pairs: {len(pairs)}",
          flush=True)
    if len(pairs) < EVAL_PAIRS + 10:
        print("[w1s] POPULATION TOO SMALL for the split — refusing",
              flush=True)
        sys.exit(1)
    split_rng = random.Random(f"w1-split-{SEED}")  # SAME split as W1
    eval_seeds = set(split_rng.sample(pairs, EVAL_PAIRS))
    train_seeds = [s for s in pairs if s not in eval_seeds]
    print(f"[w1s] train {len(train_seeds)} pairs, eval "
          f"{sorted(eval_seeds)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    subjects = {}
    for s in pairs:
        for direction in ("fwd", "rev"):
            subjects[(s, direction)] = load_surface(
                by_seed[s][direction])
    if os.environ.get("CONTROL") == "randinit":
        from llmopt.train.mathnative import MathTokenizer, build_model
        vocab_n = len(MathTokenizer().vocab)
        for s in pairs:
            torch.manual_seed(10_000 + s)
            m = build_model(vocab_n, d=64, layers=8, heads=4, ffn=256)
            subjects[(s, "rev")] = surface_from_model(m)
        print("[w1s] CONTROL=randinit: class 1 = fresh inits",
              flush=True)
    model = DirectionReader().to(dev)
    print(f"[w1s] reader params "
          f"{sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    aug = random.Random(f"w1-aug-{SEED}")
    items = [(s, d, i) for s in train_seeds
             for i, d in enumerate(("fwd", "rev"))]
    for ep in range(EPOCHS):
        random.Random(ep).shuffle(items)
        tot = corr = 0
        for s, direction, y in items:
            toks, bids = tokenize(subjects[(s, direction)], aug)
            logits = model(toks[None].to(dev), bids[None].to(dev))
            loss = F.cross_entropy(
                logits, torch.tensor([y], device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            corr += int(logits.argmax(-1).item() == y)
            tot += 1
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"[w1s] ep {ep} train acc {corr / tot:.3f}",
                  flush=True)
    model.eval()
    votes_correct = per_subject = 0
    with torch.no_grad():
        for s in sorted(eval_seeds):
            for y, direction in enumerate(("fwd", "rev")):
                vs = []
                for k in range(16):
                    erng = random.Random(f"w1-eval-{s}-{direction}-{k}")
                    toks, bids = tokenize(
                        subjects[(s, direction)], erng)
                    logits = model(toks[None].to(dev),
                                   bids[None].to(dev))
                    vs.append(int(logits.argmax(-1).item()))
                pred = 1 if sum(vs) * 2 > len(vs) else 0
                votes_correct += pred == y
                per_subject += 1
    print(f"[w1s] SURFACE={SURFACE} EVAL {votes_correct}/"
          f"{per_subject} (chance {per_subject // 2})", flush=True)


if __name__ == "__main__":
    main()

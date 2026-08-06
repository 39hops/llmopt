"""TENET W1 bridge: crystal-weight tokenizer + direction reader
(spec 2026-08-05-tenet-battery.md rung W1; the reader-shape gap
named by the 2026-08-06 reviewer scan — weightspace.reader is
pinned to HIDDEN=16 MLP subjects and cannot read d64/L8 crystals).

Subjects: sym_birth d64/L8/FFN256/H4 checkpoints (the 3080 W1
population, data/w1_population_manifest.jsonl). Tokenization: one
token per SAMPLED FFN gate neuron — TOK_PER_BLOCK neurons per
block x 8 blocks, each token the neuron's 64-wide incoming gate
row + a block-id embedding. Neuron RESAMPLING each epoch is the
permutation augmentation (measured basis: the 2026-07-06 ablation,
augmentation 88.4% > canonical 82.4% — teach invariance, never
impose it; never score weights by weight distance — this reader
CLASSIFIES, it does not compare).

Task: forward v reverse (chance 0.50). Split by SEED with pairs
intact — a pair never straddles train/eval (exclude=-guarded by
construction). ONE-DEVICE CONTRACT: trains and evals on the 3080.

Usage:
  MANIFEST=data/w1_population_manifest.jsonl EVAL_PAIRS=10 \
    .venv/bin/python scratch/tenet_w1_bridge.py
Env: TOK_PER_BLOCK=32 DMODEL=64 EPOCHS=40 LR=1e-3 SEED=1.
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

TOK_PER_BLOCK = int(os.environ.get("TOK_PER_BLOCK", "32"))
DMODEL = int(os.environ.get("DMODEL", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR = float(os.environ.get("LR", "1e-3"))
SEED = int(os.environ.get("SEED", "1"))
EVAL_PAIRS = int(os.environ.get("EVAL_PAIRS", "10"))
BLOCKS, D_IN = 8, 64


def load_subject(ckpt):
    """[8, 256, 64] gate-weight stack from a sym_birth checkpoint."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    return torch.stack([sd[f"blocks.{li}.gate.weight"].float()
                        for li in range(BLOCKS)])


def tokenize(gates, rng):
    """Sample TOK_PER_BLOCK neurons/block -> [8*T, 64] + block ids.
    Fresh sample each call = the permutation augmentation."""
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
    print(f"[w1] complete pairs: {len(pairs)}", flush=True)
    if len(pairs) < EVAL_PAIRS + 10:
        print("[w1] POPULATION TOO SMALL for the split — refusing",
              flush=True)
        sys.exit(1)
    split_rng = random.Random(f"w1-split-{SEED}")
    eval_seeds = set(split_rng.sample(pairs, EVAL_PAIRS))
    train_seeds = [s for s in pairs if s not in eval_seeds]
    print(f"[w1] train {len(train_seeds)} pairs, eval "
          f"{sorted(eval_seeds)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    subjects = {}
    for s in pairs:
        for direction in ("fwd", "rev"):
            subjects[(s, direction)] = load_subject(
                by_seed[s][direction])
    if os.environ.get("CONTROL") == "randinit":
        # INSTRUMENT POSITIVE CONTROL (TENET-W1 rider): replace the
        # "rev" class with fresh random-init gates of the same
        # architecture — trivially separable iff the reader works
        # on crystal-shaped subjects at all. Labels/split/votes
        # unchanged; only the class-1 subjects are swapped.
        from llmopt.train.mathnative import MathTokenizer, build_model
        vocab_n = len(MathTokenizer().vocab)
        for s in pairs:
            torch.manual_seed(10_000 + s)
            m = build_model(vocab_n, d=64, layers=8, heads=4,
                            ffn=256)
            sd = m.state_dict()
            subjects[(s, "rev")] = torch.stack(
                [sd[f"blocks.{li}.gate.weight"].float()
                 for li in range(BLOCKS)])
        print("[w1] CONTROL=randinit: class 1 = fresh inits",
              flush=True)
    model = DirectionReader().to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[w1] reader params {n_par}", flush=True)
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
            print(f"[w1] ep {ep} train acc {corr / tot:.3f}",
                  flush=True)
    model.eval()
    votes_correct = per_subject = 0
    with torch.no_grad():
        for s in sorted(eval_seeds):
            for y, direction in enumerate(("fwd", "rev")):
                vs = []
                for k in range(16):  # vote over 16 fresh samplings
                    erng = random.Random(f"w1-eval-{s}-{direction}-{k}")
                    toks, bids = tokenize(
                        subjects[(s, direction)], erng)
                    logits = model(toks[None].to(dev),
                                   bids[None].to(dev))
                    vs.append(int(logits.argmax(-1).item()))
                pred = 1 if sum(vs) * 2 > len(vs) else 0
                votes_correct += pred == y
                per_subject += 1
    acc = votes_correct / per_subject
    print(f"[w1] EVAL: {votes_correct}/{per_subject} = {acc:.3f} "
          f"(chance 0.50, {EVAL_PAIRS} held-out pairs, 16-vote)",
          flush=True)


if __name__ == "__main__":
    main()

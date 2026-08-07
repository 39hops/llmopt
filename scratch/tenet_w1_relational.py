"""TENET W1-R: the RELATIONAL weight reader (Artin riff banked
2026-08-06 late; the licensed new representational hypothesis after
the W1-S surface-exhaustive null).

Hypothesis: direction is a COMPOSITION-ORDER property — single-row
tokens are structurally blind to it. Features here are CROSS-LAYER:
for each boundary l -> l+1, the alignment C_l between layer l's
write directions (columns of down.weight, one 64-d vector per FFN
neuron) and layer l+1's read directions (rows of gate.weight):

    C_l = D_l @ G_{l+1}^T          [256 write-neurons x 256 reads]

Two arms (the 2026-07-06 teach-don't-impose ablation is itself a
readout — it predicts the AUG arm wins if either does):
  ARM=aug       tokens = sampled ROWS of C_l (one write-neuron's
                alignment profile), fresh neuron sample per pass =
                the permutation augmentation; 7 boundaries x
                TOK_PER_BLOCK tokens, D_IN=256.
  ARM=spectral  tokens = sorted singular values of C_l (top-64,
                fully permutation/rotation invariant), 7 tokens,
                D_IN=64, no augmentation possible (noted).

Protocol otherwise IDENTICAL to the frozen W1 bridge: same
manifest, same w1-split-1 seed split with pairs intact, same
16-vote eval, same reader shape (proj dim adapts to D_IN).
CONTROL=randinit as in W1/W1-S — mandatory before any fire is
claimed. ONE-DEVICE: 3080 (population lives there).

Usage: ARM=aug .venv/bin/python scratch/tenet_w1_relational.py
Env: MANIFEST EVAL_PAIRS TOK_PER_BLOCK EPOCHS LR SEED as bridge.
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

ARM = os.environ.get("ARM", "aug")
DMODEL = int(os.environ.get("DMODEL", "64"))
EPOCHS = int(os.environ.get("EPOCHS", "40"))
LR = float(os.environ.get("LR", "1e-3"))
SEED = int(os.environ.get("SEED", "1"))
EVAL_PAIRS = int(os.environ.get("EVAL_PAIRS", "10"))
TOK_PER_BLOCK = int(os.environ.get("TOK_PER_BLOCK", "32"))
BLOCKS = 7  # layer boundaries in an 8-block net
D_IN = 256 if ARM == "aug" else 64


def load_subject(ckpt_or_sd):
    """[7, 256, 256] cross-layer alignment stack C_l, or its
    [7, 64] spectral summary."""
    sd = (ckpt_or_sd if isinstance(ckpt_or_sd, dict) else
          torch.load(ckpt_or_sd, map_location="cpu", weights_only=True))
    cs = []
    for li in range(7):
        d = sd[f"blocks.{li}.down.weight"].float()      # [64, 256]
        g = sd[f"blocks.{li + 1}.gate.weight"].float()  # [256, 64]
        cs.append(d.T @ g.T)                            # [256, 256]
    c = torch.stack(cs)
    if ARM == "spectral":
        sv = torch.linalg.svdvals(c)                    # [7, 256]
        return sv[:, :64].reshape(7, 1, 64).contiguous()
    return c


def tokenize(subj, rng):
    toks, bids = [], []
    n_rows = subj.shape[1]
    take = min(TOK_PER_BLOCK, n_rows)
    for li in range(BLOCKS):
        idx = (rng.sample(range(n_rows), take) if n_rows > take
               else list(range(n_rows)))
        toks.append(subj[li, idx])
        bids += [li] * take
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
    print(f"[w1r] ARM={ARM} complete pairs: {len(pairs)}", flush=True)
    if len(pairs) < EVAL_PAIRS + 10:
        sys.exit("[w1r] POPULATION TOO SMALL — refusing")
    split_rng = random.Random(f"w1-split-{SEED}")
    eval_seeds = set(split_rng.sample(pairs, EVAL_PAIRS))
    train_seeds = [s for s in pairs if s not in eval_seeds]
    print(f"[w1r] train {len(train_seeds)} pairs, eval "
          f"{sorted(eval_seeds)}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    subjects = {}
    for s in pairs:
        for d in ("fwd", "rev"):
            subjects[(s, d)] = load_subject(by_seed[s][d])
    if os.environ.get("CONTROL") == "randinit":
        from llmopt.train.mathnative import MathTokenizer, build_model
        vocab_n = len(MathTokenizer().vocab)
        for s in pairs:
            torch.manual_seed(10_000 + s)
            m = build_model(vocab_n, d=64, layers=8, heads=4, ffn=256)
            subjects[(s, "rev")] = load_subject(m.state_dict())
        print("[w1r] CONTROL=randinit active", flush=True)
    model = DirectionReader().to(dev)
    print(f"[w1r] reader params "
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
            loss = F.cross_entropy(logits,
                                   torch.tensor([y], device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            corr += int(logits.argmax(-1).item() == y)
            tot += 1
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"[w1r] ep {ep} train acc {corr / tot:.3f}",
                  flush=True)
    model.eval()
    votes_correct = per_subject = 0
    with torch.no_grad():
        for s in sorted(eval_seeds):
            for y, direction in enumerate(("fwd", "rev")):
                vs = []
                for k in range(16):
                    erng = random.Random(f"w1-eval-{s}-{direction}-{k}")
                    toks, bids = tokenize(subjects[(s, direction)],
                                          erng)
                    logits = model(toks[None].to(dev),
                                   bids[None].to(dev))
                    vs.append(int(logits.argmax(-1).item()))
                pred = 1 if sum(vs) * 2 > len(vs) else 0
                votes_correct += pred == y
                per_subject += 1
    print(f"[w1r] ARM={ARM} EVAL {votes_correct}/{per_subject} "
          f"(chance {per_subject // 2})", flush=True)


if __name__ == "__main__":
    main()

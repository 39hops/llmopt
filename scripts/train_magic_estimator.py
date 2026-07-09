"""Train the magic estimator: 20 structural features -> hardness.

Two heads on one tiny trunk (the NNUE recipe — cheap features, small
net): p(solved at budget 200) and log2(1+nodes). Split is by seed
parity (features are per-problem, no leakage; generator stream is
disjoint from every training stream in the repo). Report: held-out
AUC-proxy (rank accuracy) for solved, Spearman rho for nodes — rho
against MEASURED solve-cost is the pre-registered headline number.

Baseline to beat (pre-registered): count_ops alone as the hardness
rank. If the net can't beat one feature, the estimator is ceremony.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    ra, rb = ranks(a), ranks(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else 0.0


class Estimator(nn.Module):
    def __init__(self, d_in: int = 20):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU())
        self.solved = nn.Linear(64, 1)
        self.cost = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.solved(h).squeeze(-1), self.cost(h).squeeze(-1)


def main(labels: Path, epochs: int, out: Path) -> None:
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    print(f"{len(rows)} rows; solved {sum(r['solved'] for r in rows)}, "
          f"risch_dead {sum(r['risch_dead'] for r in rows)}")
    train = [r for r in rows if r["seed"] % 2 == 0]
    test = [r for r in rows if r["seed"] % 2 == 1]

    def tensors(rs):
        x = torch.tensor([r["features"] for r in rs], dtype=torch.float32)
        ys = torch.tensor([float(r["solved"]) for r in rs])
        yc = torch.tensor([math.log2(1.0 + r["nodes"]) for r in rs])
        return x, ys, yc

    xtr, ystr, yctr = tensors(train)
    xte, yste, ycte = tensors(test)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr = (xtr - mu) / sd
    xte = (xte - mu) / sd

    model = Estimator(d_in=xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random("magic-est-0")
    for ep in range(epochs):
        idx = list(range(len(train)))
        rng.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            ls, lc = model(xtr[b])
            loss = bce(ls, ystr[b]) + nn.functional.mse_loss(lc, yctr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        if ep % 20 == 0 or ep == epochs - 1:
            print(f"epoch {ep}: loss {tot / len(idx):.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        ls, lc = model(xte)
    rho_net = spearman(lc.tolist(), ycte.tolist())
    ops = [r["features"][0] for r in test]
    rho_ops = spearman(ops, ycte.tolist())
    # rank accuracy for solved: random (solved, unsolved) pairs
    pos = [ls[i].item() for i in range(len(test)) if test[i]["solved"]]
    neg = [ls[i].item() for i in range(len(test)) if not test[i]["solved"]]
    if pos and neg:
        wins = sum(1 for p in pos for q in neg if p > q)
        auc = wins / (len(pos) * len(neg))
    else:
        auc = float("nan")
    print(f"held-out ({len(test)} rows): rho(net, log-nodes) {rho_net:.3f}"
          f" vs baseline rho(count_ops) {rho_ops:.3f}; "
          f"solved AUC {auc:.3f}")
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd},
               out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path("data/magic_labels.jsonl"))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", type=Path,
                    default=Path("checkpoints/magic_estimator.pt"))
    a = ap.parse_args()
    main(a.labels, a.epochs, a.out)

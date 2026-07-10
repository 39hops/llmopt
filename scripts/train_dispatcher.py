"""Dispatcher net: root features + rule-fire syndromes -> which brain
(policy vs markov), trained on dual-arm dominance labels
(gen_dispatch_labels.py — winner by (solved, wall), the FA Law).

Target: beat the threshold-5.5 router's OOS 141/150 @ 167s; the
in-sample oracle ceiling was 127/130 (+3 over threshold). Split by
seed parity so no problem leaks across train/test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main(labels: list[Path], epochs: int, out: Path) -> None:
    rows = []
    for p in labels:
        rows += [json.loads(l) for l in p.read_text().splitlines()]
    print(f"{len(rows)} rows; policy-wins "
          f"{sum(r['winner'] == 'policy' for r in rows)}")
    train = [r for r in rows if r["seed"] % 2 == 0]
    test = [r for r in rows if r["seed"] % 2 == 1]

    def tensors(rs):
        x = torch.tensor([r["features"] for r in rs], dtype=torch.float32)
        y = torch.tensor([1 if r["winner"] == "policy" else 0
                          for r in rs], dtype=torch.float32)
        return x, y

    xtr, ytr = tensors(train)
    xte, yte = tensors(test)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd

    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        opt.zero_grad()
        loss = lossf(net(xtr).squeeze(-1), ytr)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = (net(xte).squeeze(-1) > 0).float()
    acc = (pred == yte).float().mean().item()
    base = max(yte.mean().item(), 1 - yte.mean().item())
    # what matters: accuracy ON DISAGREEMENTS (where routing changes
    # the outcome) — dominance ties dilute plain accuracy
    dis = [i for i, r in enumerate(test)
           if r["markov"] != r["policy"]]
    dacc = ((pred[dis] == yte[dis]).float().mean().item()
            if dis else float("nan"))
    print(f"held-out ({len(test)}): acc {acc:.3f} (majority {base:.3f}); "
          f"disagreements n={len(dis)} acc {dacc:.3f}")
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd},
               out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, nargs="+",
                    default=[Path("data/dispatch_labels_mac.jsonl"),
                             Path("data/dispatch_labels_3080.jsonl")])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--out", type=Path,
                    default=Path("checkpoints/dispatcher.pt"))
    a = ap.parse_args()
    main(a.labels, a.epochs, a.out)

"""Syndrome decoder: (20 features + 14 rule-fire syndromes) ->
opening rule of the winning derivation. The decoding half of the
qLDPC analogy: syndromes localize the deviation; the decoder names
the correction.

Baselines (pre-registered):
  majority : always predict the most common first_rule
  first-fire : first rule in INT_RULES order whose syndrome bit is 1
    (the "obvious" decoder — beats it or the net is ceremony)
Metric: held-out top-1 / top-3 accuracy (seed-parity split).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import torch


def main(labels: Path, epochs: int) -> None:
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    vocab = sorted({r["first_rule"] for r in rows})
    vi = {r: i for i, r in enumerate(vocab)}
    print(f"{len(rows)} rows, {len(vocab)} opening rules: {vocab}")
    train = [r for r in rows if r["seed"] % 2 == 0]
    test = [r for r in rows if r["seed"] % 2 == 1]

    def tensors(rs):
        x = torch.tensor([r["features"] for r in rs], dtype=torch.float32)
        y = torch.tensor([vi[r["first_rule"]] for r in rs])
        return x, y

    xtr, ytr = tensors(train)
    xte, yte = tensors(test)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd

    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, len(vocab)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = random.Random("syndrome-0")
    idx = list(range(len(train)))
    for ep in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            loss = torch.nn.functional.cross_entropy(net(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(xte)
    top1 = (logits.argmax(1) == yte).float().mean().item()
    top3 = (logits.topk(3, dim=1).indices == yte[:, None]).any(1)
    top3 = top3.float().mean().item()

    # baselines
    maj = Counter(r["first_rule"] for r in train).most_common(1)[0][0]
    maj_acc = sum(r["first_rule"] == maj for r in test) / len(test)
    # first-fire: needs INT_RULES order = the last 14 features
    from llmopt.search.rules import INT_RULES
    names = [n for n, _ in INT_RULES]
    ff_hit = 0
    for r in test:
        fires = r["features"][20:]
        pick = next((names[i] for i, b in enumerate(fires) if b > 0.5),
                    maj)
        ff_hit += pick == r["first_rule"]
    print(f"held-out ({len(test)}): net top-1 {top1:.3f} top-3 {top3:.3f}"
          f" | majority {maj_acc:.3f} | first-fire {ff_hit/len(test):.3f}")
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd,
                "vocab": vocab}, "checkpoints/syndrome_decoder.pt")
    print("saved checkpoints/syndrome_decoder.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path("data/syndrome_labels.jsonl"))
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    main(a.labels, a.epochs)

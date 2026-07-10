"""Per-state syndrome policy: (20 features + 14 syndromes + prev-rule
one-hot) -> next rule. The policy-side NNUE-vs-LLM rematch.

Baselines (pre-registered):
  majority       : most common rule overall
  markov-bigram  : argmax bigram[prev] from the SAME training pairs
                   (the 293-dict, given the same data)
Metrics: held-out top-1 / top-3 (split by problem seed parity, so no
derivation leaks across the split).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch


def main(labels: Path, epochs: int, no_synd: bool = False) -> None:
    rows = [json.loads(l) for l in labels.read_text().splitlines()]
    if no_synd:
        # gate variant: pre-expansion, syndromes don't exist yet
        for r in rows:
            r["features"] = r["features"][:20]
    vocab = sorted({r["rule"] for r in rows})
    prevs = sorted({r["prev"] for r in rows})
    vi = {r: i for i, r in enumerate(vocab)}
    pi = {p: i for i, p in enumerate(prevs)}
    print(f"{len(rows)} pairs, {len(vocab)} rules, {len(prevs)} prev")
    train = [r for r in rows if r["seed"] % 2 == 0]
    test = [r for r in rows if r["seed"] % 2 == 1]

    def tensors(rs):
        xs = []
        for r in rs:
            oh = [0.0] * len(prevs)
            oh[pi[r["prev"]]] = 1.0
            xs.append(r["features"] + oh)
        return (torch.tensor(xs, dtype=torch.float32),
                torch.tensor([vi[r["rule"]] for r in rs]))

    xtr, ytr = tensors(train)
    xte, yte = tensors(test)
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd

    net = torch.nn.Sequential(
        torch.nn.Linear(xtr.shape[1], 96), torch.nn.ReLU(),
        torch.nn.Linear(96, 96), torch.nn.ReLU(),
        torch.nn.Linear(96, len(vocab)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    rng = random.Random("policy-0")
    idx = list(range(len(train)))
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            loss = torch.nn.functional.cross_entropy(net(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        logits = net(xte)
    top1 = (logits.argmax(1) == yte).float().mean().item()
    top3 = ((logits.topk(3, 1).indices == yte[:, None]).any(1)
            .float().mean().item())

    maj = Counter(r["rule"] for r in train).most_common(1)[0][0]
    maj_acc = sum(r["rule"] == maj for r in test) / len(test)
    bi: dict = defaultdict(Counter)
    for r in train:
        bi[r["prev"]][r["rule"]] += 1
    mk1 = mk3 = 0
    for r in test:
        ranked = [w for w, _ in bi[r["prev"]].most_common(3)] or [maj]
        mk1 += ranked[0] == r["rule"]
        mk3 += r["rule"] in ranked
    print(f"held-out ({len(test)}): net top-1 {top1:.3f} top-3 {top3:.3f}"
          f" | majority {maj_acc:.3f}"
          f" | markov top-1 {mk1/len(test):.3f} top-3 {mk3/len(test):.3f}")
    out = ("checkpoints/rule_gate.pt" if no_synd
           else "checkpoints/syndrome_policy.pt")
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd,
                "vocab": vocab, "prevs": prevs}, out)
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path,
                    default=Path("data/policy_labels.jsonl"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-synd", action="store_true")
    a = ap.parse_args()
    main(a.labels, a.epochs, a.no_synd)

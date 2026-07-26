"""Complex-weight NNUE vs real twin on magic labels (pre-reg below).

The first alphabet cell in the OLIGARCHY phase: same data, split,
loss, and metrics as train_magic_estimator; paired same-run arms:
  real  — 20 -> 64 -> 64 ReLU trunk (the founding recipe)
  cplx  — 20 -> 42C -> 42C, genuine complex multiply, modReLU
Real-param counts printed (match within ~3%; both framings per the
fairness rule). Oracle: held-out spearman rho + solved AUC.
"""
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
import torch
import torch.nn as nn

from train_magic_estimator import spearman  # noqa: E402


class ComplexEstimator(nn.Module):
    def __init__(self, d_in=20, w=42):
        super().__init__()
        self.w1r = nn.Linear(d_in, w)
        self.w1i = nn.Linear(d_in, w)
        self.w2rr = nn.Linear(w, w, bias=False)
        self.w2ri = nn.Linear(w, w, bias=False)
        self.b2 = nn.Parameter(torch.zeros(w))
        self.solved = nn.Linear(2 * w, 1)
        self.cost = nn.Linear(2 * w, 1)

    @staticmethod
    def modrelu(re, im, b):
        mag = torch.sqrt(re * re + im * im + 1e-9)
        act = torch.relu(mag + b) / mag
        return re * act, im * act

    def forward(self, x):
        re, im = self.w1r(x), self.w1i(x)
        re, im = self.modrelu(re, im, 0.0 * self.b2)
        # complex matmul: (w2rr + i w2ri)(re + i im)
        r2 = self.w2rr(re) - self.w2ri(im)
        i2 = self.w2rr(im) + self.w2ri(re)
        r2, i2 = self.modrelu(r2, i2, self.b2)
        h = torch.cat([r2, i2], -1)
        return self.solved(h).squeeze(-1), self.cost(h).squeeze(-1)


class RealEstimator(nn.Module):
    def __init__(self, d_in=20):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(),
                                   nn.Linear(64, 64), nn.ReLU())
        self.solved = nn.Linear(64, 1)
        self.cost = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.solved(h).squeeze(-1), self.cost(h).squeeze(-1)


def run(model, name, xtr, ystr, yctr, xte, yste, ycte, test, epochs=200):
    n = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()
    rng = random.Random("cplx-nnue-0")
    for ep in range(epochs):
        idx = list(range(xtr.shape[0]))
        rng.shuffle(idx)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]
            ls, lc = model(xtr[b])
            loss = bce(ls, ystr[b]) + nn.functional.mse_loss(lc, yctr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        ls, lc = model(xte)
    rho = spearman(lc.tolist(), ycte.tolist())
    pos = [ls[i].item() for i in range(len(test)) if test[i]["solved"]]
    neg = [ls[i].item() for i in range(len(test)) if not test[i]["solved"]]
    auc = (sum(1 for p in pos for q in neg if p > q) /
           (len(pos) * len(neg))) if pos and neg else float("nan")
    print(f"{name}: params {n}  rho {rho:.3f}  AUC {auc:.3f}", flush=True)
    return rho, auc


rows = [json.loads(l)
        for l in Path("data/magic_labels_v7.jsonl").read_text().splitlines()]
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
xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
print(f"{len(train)} train / {len(test)} test", flush=True)

torch.manual_seed(1)
run(RealEstimator(d_in=xtr.shape[1]), "real64",
    xtr, ystr, yctr, xte, yste, ycte, test)
torch.manual_seed(1)
run(ComplexEstimator(d_in=xtr.shape[1]), "cplx42",
    xtr, ystr, yctr, xte, yste, ycte, test)

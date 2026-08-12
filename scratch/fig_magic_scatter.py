#!/usr/bin/env python3
"""Gallery: magic-estimator held-out scatter (predicted vs measured).

As-rendered derivative of the booked final-estimator verdict
(RESULTS 2026-07-09 arc: rf rho 0.906 / AUC 0.986 on 1,848 held-out
rows). Reloads checkpoints/magic_estimator_rf.pt and reproduces the
trainer's exact held-out split (seed % 2 == 1) and normalization
from data/magic_labels_all_rf.jsonl; recomputes rho at render time
and stamps it on the figure — the figure never quotes a number it
did not just recompute. No new capability claim.

Usage: .venv/bin/python scratch/fig_magic_scatter.py [--out PNG]
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab import figstyle  # house palette + fonts + mathtext


class Estimator(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(),
                                   nn.Linear(64, 64), nn.ReLU())
        self.solved = nn.Linear(64, 1)
        self.cost = nn.Linear(64, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.solved(h).squeeze(-1), self.cost(h).squeeze(-1)


def spearman(a, b):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = float(rank)
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db)


def main(out: Path) -> None:
    rows = [json.loads(l) for l in
            Path("data/magic_labels_all_rf.jsonl").read_text().splitlines()]
    test = [r for r in rows if r["seed"] % 2 == 1]
    ck = torch.load("checkpoints/magic_estimator_rf.pt",
                    weights_only=True)
    x = torch.tensor([r["features"] for r in test], dtype=torch.float32)
    x = (x - ck["mu"]) / ck["sd"]
    model = Estimator(d_in=x.shape[1])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        _, lc = model(x)
    pred = lc.tolist()
    true = [math.log2(1.0 + r["nodes"]) for r in test]
    rho = spearman(pred, true)
    print(f"held-out n={len(test)} recomputed rho={rho:.3f}")

    solved = [r["solved"] for r in test]
    out.parent.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        c = figstyle.CHROME[mode]
        plt.rcParams.update(figstyle.rc(mode))
        series = (figstyle.SERIES_DARK if mode == "dark"
                  else figstyle.SERIES_LIGHT)
        blue, orange = series[0], series[1]
        fig, ax = plt.subplots(figsize=(7.5, 7.2), dpi=200)
        lim = (min(true + pred) - 0.4, max(true + pred) + 0.4)
        ax.plot(lim, lim, ls="--", lw=1.2, color=c["axis"], zorder=1)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_aspect("equal")
        for flag, color, label in ((True, blue, "solved at budget"),
                                   (False, orange,
                                    "unsolved at budget")):
            xs = [t for t, s in zip(true, solved) if s == flag]
            ys = [p for p, s in zip(pred, solved) if s == flag]
            ax.scatter(xs, ys, s=22, alpha=0.55, lw=0.5,
                       edgecolors=c["surface"], color=color,
                       label=f"{label} (n={len(xs):,})", zorder=2)
        ax.set_xlabel(r"measured hardness — "
                      r"$\log_2(1 + \mathrm{search\ nodes\ spent})$",
                      fontsize=12, labelpad=8)
        ax.set_ylabel(r"predicted hardness (same $\log_2$ scale)",
                      fontsize=12, labelpad=8)
        ax.text(0, 1.115, "Predicting how hard a math problem is\n"
                "— before solving it",
                transform=ax.transAxes, color=c["primary"],
                fontsize=17, fontweight=500, va="bottom")
        ax.text(0, 1.03, f"held-out: {len(test):,} problems never "
                r"seen in training  ·  Spearman $\rho$ = "
                f"{rho:.3f}\n"
                r"prediction cost: microseconds per problem",
                transform=ax.transAxes, color=c["secondary"],
                fontsize=11, fontweight=300, va="bottom")
        ax.legend(frameon=False, loc="upper left", fontsize=11,
                  labelcolor=c["secondary"], handletextpad=0.1,
                  borderaxespad=0.4, markerscale=1.5)
        fig.text(0.11, 0.015,
                 "magic_estimator_rf.pt · rho recomputed at render · "
                 "RESULTS 2026-07-09 arc",
                 fontsize=7, color=c["muted"], family="monospace")
        fig.subplots_adjust(top=0.8, bottom=0.1, left=0.11,
                            right=0.98)
        dst = out.with_name(f"{out.stem}-{mode}{out.suffix}")
        fig.savefig(dst, facecolor=c["surface"])
        plt.close(fig)
        print(f"saved -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("figs/2026-08-09/magic_scatter.png"))
    main(ap.parse_args().out)

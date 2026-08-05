"""CHURN-JUDGE-1 fit/eval (committed so the booked verdict is
re-derivable — it was a desk computation on 2026-08-04).

Reads logs/opus/moe_gt1_perprob.jsonl: instrumented crest rows (the
ones carrying "recall", from the cj1_rerun 7-seed pass) + the
original full-model rows. Dataset = crest FAILURES; label = "full
solves it" (rescue). Judge = standardized logistic on {recall,
parsed, gen_len, level}, fit on TRAIN seeds, frozen eval on EVAL
seeds. Reproduces: AUC 0.679 held-out (P1 fires, bar 0.60),
level-only null 0.438, spend 31 vs 23.5 expected at F=33% = 1.32x
(P2 misses, bar 1.5x).

Usage: .venv/bin/python scratch/churn_judge_eval.py
"""

import json

import numpy as np

TRAIN, EVAL = (111, 222, 333), (555, 4242, 777, 90210)
LOG = "logs/opus/moe_gt1_perprob.jsonl"
FEATURES = ["recall", "parsed", "gen_len", "level"]


def load():
    rows = [json.loads(l) for l in open(LOG)]
    crest = {(r["seed"], r["idx"]): r
             for r in rows if r["frac"] == 0.453 and "recall" in r}
    full = {(r["seed"], r["idx"]): r["ok"]
            for r in rows if r["frac"] == 1.0}
    return crest, full


def dataset(crest, full, seeds):
    X, y = [], []
    for (s, i), r in crest.items():
        if s in seeds and not r["ok"] and (s, i) in full:
            X.append([r["recall"], float(r["parsed"]),
                      r["gen_len"], r["level"]])
            y.append(int(full[(s, i)]))
    return np.array(X), np.array(y)


def auc(score, y):
    """Rank-sum AUC: P(score(pos) > score(neg))."""
    order = np.argsort(score)
    r = np.empty(len(score))
    r[order] = np.arange(1, len(score) + 1)
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    crest, full = load()
    Xtr, ytr = dataset(crest, full, TRAIN)
    Xev, yev = dataset(crest, full, EVAL)
    print(f"train failures {len(ytr)} (rescue {ytr.mean():.2f}) | "
          f"eval {len(yev)} ({yev.mean():.2f})")

    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sd
    w, b = np.zeros(Xtr.shape[1]), 0.0
    for _ in range(3000):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= 0.1 * Z.T @ (p - ytr) / len(ytr)
        b -= 0.1 * (p - ytr).mean()
    score = ((Xev - mu) / sd) @ w + b

    print(f"weights {dict(zip(FEATURES, np.round(w, 3)))}")
    print(f"EVAL AUC judge {auc(score, yev):.3f} (bar 0.60) | "
          f"level-only {auc(Xev[:, 3], yev):.3f}")
    F = int(round(0.33 * len(yev)))
    top = np.argsort(-score)[:F]
    rec, rnd = int(yev[top].sum()), yev.mean() * F
    print(f"escalate {F}/{len(yev)}: judge {rec} vs random E[{rnd:.1f}] "
          f"= {rec / rnd:.2f}x (bar 1.5x)")


if __name__ == "__main__":
    main()

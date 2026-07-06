"""NNUE-moment training: probe-labeled on-policy states -> tiny MLP.
Spec: 2026-07-07-nnue-eval-design.md. Labels are log2(nodes-to-solve),
probes capped (200 nodes / 60 s wall, BaseException alarm — the
calibration guards). Loss is reported but the race (bench_nnue.py)
is scored by running the eval inside the search.

  python scripts/train_nnue.py            # full data gen + train
  python scripts/train_nnue.py --smoke    # tiny sizes, sanity only
"""

from __future__ import annotations

import argparse
import math
import random
import signal
import statistics

import sympy as sp
import torch

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, hce, is_solved
from llmopt.search.features import N_FEATURES, featurize

X = sp.Symbol("x")
PROBE_NODES = 200
PROBE_SECONDS = 60
FAIL_LABEL = math.log2(400.0)


class _Timeout(BaseException):
    pass


def _alarm(signum, frame):
    raise _Timeout()


class NnueEval(torch.nn.Module):
    # NOTE: mirrored in scripts/bench_nnue.py (scripts aren't a package);
    # keep the two definitions identical.
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(N_FEATURES, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _root(rng: random.Random, level: int, kind: str) -> sp.Expr:
    if kind == "diff":
        return sp.Derivative(_expression(rng, level), X)
    while True:
        g = sp.simplify(sp.diff(_expression(rng, level), X))
        if g != 0:
            return sp.Integral(g, X)


def collect_states(split: str, per_cell: int, cap: int,
                   exclude_roots: set[str] | None = None
                   ) -> tuple[list[State], set[str]]:
    """On-policy states from searches over problems seeded
    nnue-{split}-{kind}-{level}-{i}. Returns (states, root_sreprs)."""
    seen: set[str] = set()
    roots: set[str] = set()
    out: list[State] = []
    for kind in ("diff", "int"):
        for level in (1, 2, 3):
            rng = random.Random(f"nnue-{split}-{kind}-{level}-0")
            for _ in range(per_cell):
                root = _root(rng, level, kind)
                rk = sp.srepr(root)
                if exclude_roots and rk in exclude_roots:
                    continue  # contamination guard, never seeds alone
                roots.add(rk)
                trace: list[State] = []
                signal.signal(signal.SIGALRM, _alarm)
                signal.alarm(PROBE_SECONDS)
                try:
                    beam_search(root, max_plies=20, max_nodes=PROBE_NODES,
                                trace=trace)
                except _Timeout:
                    pass
                finally:
                    signal.alarm(0)
                for s in trace:
                    if s.key() not in seen and not is_solved(s):
                        seen.add(s.key())
                        out.append(s)
    rng = random.Random(f"nnue-{split}-subsample-0")
    if len(out) > cap:
        out = rng.sample(out, cap)
    return out, roots


def label(state: State) -> float:
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(PROBE_SECONDS)
    try:
        r = beam_search(state.expr, width=8, max_plies=20,
                        max_nodes=PROBE_NODES)
        return math.log2(float(r.nodes)) if r.solved else FAIL_LABEL
    except _Timeout:
        return FAIL_LABEL
    finally:
        signal.alarm(0)


def spearman(xs, ys):
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    return cov / (vx * vy) ** 0.5 if vx and vy else float("nan")


def main(per_cell: int, train_cap: int, eval_cap: int) -> None:
    torch.manual_seed(0)
    train_states, train_roots = collect_states("train", per_cell, train_cap)
    eval_states, _ = collect_states("eval", max(2, per_cell // 4), eval_cap,
                                    exclude_roots=train_roots)
    print(f"train states: {len(train_states)}, eval states: {len(eval_states)}",
          flush=True)

    def build(states, tag):
        xs, ys = [], []
        for i, s in enumerate(states):
            xs.append(featurize(s.expr))
            ys.append(label(s))
            if (i + 1) % 100 == 0:
                print(f"  labeled {tag} {i + 1}/{len(states)}", flush=True)
        return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys)

    Xtr, ytr = build(train_states, "train")
    Xev, yev = build(eval_states, "eval")

    mean, std = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr_n, Xev_n = (Xtr - mean) / std, (Xev - mean) / std

    net = NnueEval()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net(Xtr_n), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred_ev = net(Xev_n).tolist()
        pred_tr = net(Xtr_n).tolist()

    rho_net = spearman(pred_ev, yev.tolist())
    rho_hce = spearman([hce(s) for s in eval_states], yev.tolist())
    print(f"final train MSE: {loss.item():.4f}")
    print(f"held-out state rho — nnue: {rho_net:+.3f}   hce: {rho_hce:+.3f}")
    print(f"(train rho nnue: {spearman(pred_tr, ytr.tolist()):+.3f})")

    torch.save({"state_dict": net.state_dict(), "mean": mean, "std": std},
               "checkpoints/nnue_eval.pt")
    print("saved checkpoints/nnue_eval.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        main(per_cell=3, train_cap=80, eval_cap=30)
    else:
        main(per_cell=25, train_cap=1500, eval_cap=300)

"""Engine-level regret: predict a DOOMED search from the live beam and
abort early, banking the wall.

Motivation (RESULTS wall anatomy + estimator v7): solves average ~0.3s
(median 4 nodes since the orbitals) while failures burn the full 120s
wall — node-count cost prediction collapsed (rho 0.532) because solve
cost has no variance left; the remaining cost variable IS the wall
spent on doom. This is the LLM regret probe one level up, with the
round-1 lesson applied: sweep the threshold OFFLINE before any live
race (the naive-threshold race lost 78 vs 100).

  --phase labels : fork-isolated solves over L5-L8; per-ply rows
                   (features of beam head + scalars + elapsed wall),
                   label = search eventually solved
  --phase probe  : MLP (ply state -> doomed) + held-out AUC
  --phase sweep  : offline replay of abort policies (thresh x min_ply)
                   at true elapsed-wall bookkeeping.
                   Pre-registered bar: ZERO held-out solve loss with
                   >= 25% total wall cut. Clears -> live race next.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

WALL = 120
PROBE = Path("checkpoints/engine_regret_probe.pt")


def _worker(level: int, seed: int, budget: int, q: "mp.Queue") -> None:
    import sympy as sp

    from llmopt.mathgen.problems import make_integrate
    from llmopt.search.engine import solve
    from llmopt.search.features import featurize

    p = make_integrate(level, seed)
    root = sp.Integral(p._expr, sp.Symbol("x"))
    t0 = time.monotonic()
    rows: list[dict] = []

    def hook(ply, beam, nodes):
        head = beam[0]
        rows.append({
            "ply": ply,
            "feats": featurize(head.expr),
            "nodes": nodes,
            "beam": len(beam),
            "t": round(time.monotonic() - t0, 3),
        })
        return False  # observe only

    res = solve(root, budget=budget, ply_hook=hook)
    q.put({"level": level, "seed": seed,
           "solved": bool(res.solved),
           "wall": round(time.monotonic() - t0, 3),
           "rows": rows})


def phase_labels(n_per_level: int, seed_base: int, out: Path) -> None:
    ctx = mp.get_context("fork")
    n_rows = n_prob = 0
    with out.open("w") as f:
        for level in (5, 6, 7, 8):
            for i in range(n_per_level):
                q = ctx.Queue()
                pr = ctx.Process(target=_worker,
                                 args=(level, seed_base + i, 200, q))
                pr.start()
                pr.join(WALL)
                if pr.is_alive():
                    pr.kill()
                    pr.join()
                    # wall death: no per-ply rows crossed the pipe, but
                    # the OUTCOME is known — doomed. Emit a synthetic
                    # terminal row so the wall-burners (the class the
                    # probe exists to kill) are IN the training set.
                    rec = {"level": level, "seed": seed_base + i,
                           "solved": False, "wall": float(WALL),
                           "rows": []}
                else:
                    try:
                        rec = q.get(timeout=10)
                    except Exception:
                        continue
                f.write(json.dumps(rec) + "\n")
                n_prob += 1
                n_rows += len(rec["rows"])
                if n_prob % 25 == 0:
                    print(f"[{n_prob}] rows={n_rows}", flush=True)
    print(f"LABELS done: {n_prob} searches, {n_rows} ply rows -> {out}")


def _load(labels: Path):
    return [json.loads(l) for l in labels.read_text().splitlines()]


def _xy(recs):
    X, y = [], []
    for r in recs:
        for row in r["rows"]:
            X.append(row["feats"] + [float(row["ply"]),
                                     float(row["nodes"]),
                                     float(row["beam"])])
            y.append(0.0 if r["solved"] else 1.0)  # 1 = doomed
    return X, y


def phase_probe(labels: Path, epochs: int) -> None:
    import torch
    recs = _load(labels)
    # split by PROBLEM, not by row (rows within a search correlate)
    cut = int(len(recs) * 0.8)
    Xtr, ytr = _xy(recs[:cut])
    Xte, yte = _xy(recs[cut:])
    mu = torch.tensor(Xtr, dtype=torch.float32).mean(0)
    sd = torch.tensor(Xtr, dtype=torch.float32).std(0) + 1e-6
    net = torch.nn.Sequential(
        torch.nn.Linear(len(mu), 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = (torch.tensor(Xtr, dtype=torch.float32) - mu) / sd
    yt = torch.tensor(ytr, dtype=torch.float32)[:, None]
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net(Xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pe = torch.sigmoid(
            net((torch.tensor(Xte, dtype=torch.float32) - mu) / sd))[:, 0]
    # AUC by rank comparison
    pos = pe[torch.tensor(yte) == 1.0]
    neg = pe[torch.tensor(yte) == 0.0]
    auc = float((pos[:, None] > neg[None, :]).float().mean()) \
        if len(pos) and len(neg) else float("nan")
    print(f"probe held-out rows {len(yte)} (doom rate "
          f"{sum(yte)/len(yte):.3f}): AUC {auc:.3f}")
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd},
               PROBE)
    print(f"saved -> {PROBE}")


def phase_sweep(labels: Path) -> None:
    import torch
    recs = _load(labels)
    cut = int(len(recs) * 0.8)
    test = recs[cut:]  # sweep on the probe's held-out problems only
    p_ = torch.load(PROBE, weights_only=False)
    net = torch.nn.Sequential(
        torch.nn.Linear(len(p_["mu"]), 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 1))
    net.load_state_dict(p_["state_dict"])
    net.eval()

    def doom(row):
        x = torch.tensor(
            [row["feats"] + [float(row["ply"]), float(row["nodes"]),
                             float(row["beam"])]], dtype=torch.float32)
        with torch.no_grad():
            return torch.sigmoid(net((x - p_["mu"]) / p_["sd"])).item()

    # cache probe outputs once (grid replay is then free)
    for r in test:
        for row in r["rows"]:
            row["p"] = doom(row)

    base_solves = sum(r["solved"] for r in test)
    base_wall = sum(r["wall"] for r in test)
    print(f"baseline: {base_solves}/{len(test)} solved, "
          f"wall {base_wall:.0f}s")
    best = None
    for min_ply in (1, 2, 3, 4):
        for th in (0.7, 0.8, 0.9, 0.95, 0.99):
            solves = 0
            wall = 0.0
            for r in test:
                ab_t = None
                for row in r["rows"]:
                    if row["ply"] >= min_ply and row["p"] > th:
                        ab_t = row["t"]
                        break
                if ab_t is None:
                    solves += r["solved"]
                    wall += r["wall"]
                else:
                    wall += ab_t  # aborted: banked the rest
            tag = ""
            if solves == base_solves:
                cut_pct = (base_wall - wall) / base_wall * 100
                tag = f"  <- ZERO LOSS, wall -{cut_pct:.0f}%"
                if best is None or wall < best[0]:
                    best = (wall, min_ply, th)
            print(f"  min_ply={min_ply} th={th:.2f}: {solves}/{len(test)}"
                  f" wall {wall:.0f}s{tag}")
    if best:
        w, mp_, th = best
        print(f"SWEEP BEST zero-loss: min_ply={mp_} th={th} "
              f"wall {w:.0f}s vs {base_wall:.0f}s "
              f"({(base_wall-w)/base_wall*100:.0f}% cut)")
        print("bar (pre-registered): zero solve loss AND >=25% cut")
    else:
        print("no zero-loss config: HONEST NULL at these thresholds")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["labels", "probe", "sweep"])
    ap.add_argument("--n-per-level", type=int, default=120)
    ap.add_argument("--seed-base", type=int, default=5_000_000)
    ap.add_argument("--labels", type=Path,
                    default=Path("data/engine_regret_labels.jsonl"))
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()
    if a.phase == "labels":
        phase_labels(a.n_per_level, a.seed_base, a.labels)
    elif a.phase == "probe":
        phase_probe(a.labels, a.epochs)
    else:
        phase_sweep(a.labels)

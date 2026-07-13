"""Predicted syndromes: learn the Hints line, skip the mini-solve.

A hint syndrome (which INT_RULES fire on a state) is computed by
RUNNING every rule — ansatz rules decide by doing the work
(i_linear_basis solves its system), so each novel state costs a
mini-solve (~0.1-1s, forked). The memoization (2026-07-13) killed
repeat costs; this probe attacks the FIRST look: a tiny MLP over
featurize() structural features predicting the 14 rule-fire bits in
microseconds.

Pre-registered bar: exact-set match >= 80% AND micro-F1 >= 0.9 on
held-out states (split by expression-string hash — same cur may
appear in many rows). Report per-rule P/R honestly; rules that fire
by DOING work (ansatz family) are the ones a structural net should
find hardest — if they drag the bar down, that verdict gets recorded.

Labels stream to data/syndrome_labels.jsonl incrementally (the
killed-worker corollary: a chunk killed by its wall must not make
its states invisible). Chunked forks, 40 states each (fork
granularity matches blast radius — the magic-bucket scar).

    .venv/bin/python scripts/bench_pred_syndromes.py label
    .venv/bin/python scripts/bench_pred_syndromes.py label-gen
    .venv/bin/python scripts/bench_pred_syndromes.py train

Round 1 (corpus states only, 2018): FAIL — exact 60.6%, micro-F1
0.893. NOT where predicted: the do-the-work ansatz rules are the
easiest bits (i_heurisch 0.98 P/R — near-universal base rate); the
misses are the RARE rules (i_apart R 0.25, i_sqrt_basis R 0.36, 11-16
test rows each) — starved of examples, not structurally hard.
label-gen widens with fresh generator roots across levels (the
widen-the-generator-space rule, applied to labels).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# NOT data/syndrome_labels.jsonl — that file belongs to
# gen_syndrome_labels.py / train_syndrome_decoder.py (different schema)
LABELS = Path("data/pred_syndrome_labels.jsonl")
CKPT = Path("checkpoints/pred_syndromes.pt")
CHUNK = 40


def _label_worker(states: list[str], q) -> None:
    import sympy as sp

    from llmopt.search.derivation import _timeboxed
    from llmopt.search.features import featurize
    from llmopt.search.rules import INT_RULES
    env = {"Integral": sp.Integral, "x": sp.Symbol("x")}
    for s in states:
        try:
            e = sp.sympify(s, locals=env)
            node = max(e.atoms(sp.Integral), key=sp.count_ops,
                       default=None)
            names = []
            if node is not None:
                for rn, rule in INT_RULES:
                    if _timeboxed(rule, node, default=[]):
                        names.append(rn)
            q.put(json.dumps(
                {"s": s, "feats": featurize(e), "fires": names}))
        except Exception:
            continue
    q.put(None)


def phase_label() -> None:
    states: set[str] = set()
    for line in Path("data/step_chains.jsonl").read_text().splitlines():
        r = json.loads(line)
        for s in (r["cur"], r["nxt"]):
            if "Integral" in s:
                states.add(s)
    done = set()
    if LABELS.exists():
        done = {json.loads(l)["s"]
                for l in LABELS.read_text().splitlines()}
    todo = sorted(states - done)
    print(f"{len(states)} integral states, {len(done)} labeled, "
          f"{len(todo)} to go", flush=True)
    ctx = mp.get_context("fork")
    with LABELS.open("a") as f:
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            q = ctx.Queue()
            pr = ctx.Process(target=_label_worker, args=(chunk, q))
            pr.start()
            got = 0
            # stream rows as they arrive; wall applies to the GAP,
            # not the chunk total (one wedged state must not eat its
            # 39 chunk-mates' completed labels)
            while True:
                try:
                    row = q.get(timeout=45)
                except Exception:
                    print(f"  chunk {i // CHUNK}: wall after {got} "
                          f"rows (wedged state)", flush=True)
                    break
                if row is None:
                    break
                f.write(row + "\n")
                got += 1
            f.flush()
            pr.kill()
            pr.join()
            if i % (CHUNK * 10) == 0:
                print(f"  {i + len(chunk)}/{len(todo)}", flush=True)


def _gen_worker(jobs: list[tuple[int, int]], q) -> None:
    import sympy as sp

    from llmopt.mathgen.problems import make_integrate
    for lv, sd in jobs:
        try:
            q.put(sp.sstr(sp.Integral(
                make_integrate(lv, sd)._expr, sp.Symbol("x"))))
        except Exception:
            continue
    q.put(None)


def phase_label_gen(n_per: int = 400) -> None:
    """Widen the label set with fresh generator roots, L2-L8.
    Seed band 30_000_000+ — outside every train/eval/label band."""
    ctx = mp.get_context("fork")
    states: set[str] = set()
    jobs = [(lv, 30_000_000 + 1000 * lv + i)
            for lv in range(2, 9) for i in range(n_per)]
    for i in range(0, len(jobs), CHUNK):
        q = ctx.Queue()
        pr = ctx.Process(target=_gen_worker, args=(jobs[i:i + CHUNK], q))
        pr.start()
        while True:
            try:
                s = q.get(timeout=45)
            except Exception:
                break
            if s is None:
                break
            states.add(s)
        pr.kill()
        pr.join()
    print(f"generated {len(states)} fresh root states", flush=True)
    done = {json.loads(l)["s"] for l in LABELS.read_text().splitlines()}
    todo = sorted(states - done)
    with LABELS.open("a") as f:
        for i in range(0, len(todo), CHUNK):
            chunk = todo[i:i + CHUNK]
            q = ctx.Queue()
            pr = ctx.Process(target=_label_worker, args=(chunk, q))
            pr.start()
            got = 0
            while True:
                try:
                    row = q.get(timeout=45)
                except Exception:
                    print(f"  chunk {i // CHUNK}: wall after {got}",
                          flush=True)
                    break
                if row is None:
                    break
                f.write(row + "\n")
                got += 1
            f.flush()
            pr.kill()
            pr.join()
            if i % (CHUNK * 10) == 0:
                print(f"  {i + len(chunk)}/{len(todo)}", flush=True)


def phase_train() -> None:
    import time

    import torch

    rows = [json.loads(l) for l in LABELS.read_text().splitlines()]
    vocab = sorted({rn for r in rows for rn in r["fires"]})
    vi = {r: i for i, r in enumerate(vocab)}
    X = torch.tensor([r["feats"] for r in rows], dtype=torch.float32)
    Y = torch.zeros(len(rows), len(vocab))
    for i, r in enumerate(rows):
        for rn in r["fires"]:
            Y[i, vi[rn]] = 1.0
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    X = (X - mu) / sd
    # split by expression hash: the same state string must never
    # straddle the split (dup rows would leak labels)
    import hashlib
    is_test = torch.tensor(
        [int(hashlib.sha1(r["s"].encode()).hexdigest(), 16) % 5 == 0
         for r in rows])
    tr, te = ~is_test, is_test
    print(f"{len(rows)} states, {len(vocab)} rules, "
          f"train {int(tr.sum())} / test {int(te.sum())}")
    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], 96), torch.nn.ReLU(),
        torch.nn.Linear(96, 96), torch.nn.ReLU(),
        torch.nn.Linear(96, len(vocab)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    for ep in range(300):
        opt.zero_grad()
        loss = lossf(net(X[tr]), Y[tr])
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        t0 = time.perf_counter()
        pred = (net(X[te]) > 0).float()
        dt_us = (time.perf_counter() - t0) / max(int(te.sum()), 1) * 1e6
    exact = (pred == Y[te]).all(dim=1).float().mean().item()
    tp = (pred * Y[te]).sum().item()
    micro_p = tp / max(pred.sum().item(), 1)
    micro_r = tp / max(Y[te].sum().item(), 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-9)
    print(f"\nexact-set match {100 * exact:.1f}%  "
          f"micro-F1 {micro_f1:.3f} (P {micro_p:.3f} R {micro_r:.3f})  "
          f"{dt_us:.0f}us/state")
    print(f"{'rule':>16s} {'n_test':>6s} {'P':>6s} {'R':>6s}")
    for rn, i in vi.items():
        n = int(Y[te][:, i].sum())
        tpr = float((pred[:, i] * Y[te][:, i]).sum())
        p = tpr / max(float(pred[:, i].sum()), 1e-9)
        r = tpr / max(n, 1e-9)
        print(f"{rn:>16s} {n:>6d} {p:>6.2f} {r:>6.2f}")
    torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd,
                "vocab": vocab}, CKPT)
    print(f"\nbar: exact >= 80% AND micro-F1 >= 0.9 -> "
          f"{'PASS' if exact >= 0.8 and micro_f1 >= 0.9 else 'FAIL'}")


if __name__ == "__main__":
    {"label": phase_label, "label-gen": phase_label_gen,
     "train": phase_train}[sys.argv[1]]()

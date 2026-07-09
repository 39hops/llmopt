"""ZX rung 3: structured circuits + markov prior (spec ladder).

Structured = random Toffoli networks (each Toffoli = 7 T after
to_basic_gates; SHARED phase structure across Toffolis is where the
literature's T-count wins live — a lone Toffoli holds at 7 under
greedy). Prior = rule-name bigram mined from best-first descent
histories on a harvest set (string-seeded; race seeds disjoint).

  --harvest : mine checkpoints/zx_markov_prior.json from N circuits
  --race    : greedy vs bf-full vs bf-markov3 on fresh circuits
"""

from __future__ import annotations

import argparse
import json
import random
import signal
from collections import Counter, defaultdict
from pathlib import Path

import pyzx as zx
from pyzx.circuit import Circuit

from llmopt.search.zx_engine import (ZXState, best_first_zx, macro_moves,
                                     moves, tcount, verify_equal, zx_eval)

PRIOR_PATH = Path("checkpoints/zx_markov_prior.json")


class _Timeout(BaseException):
    pass


def _toffoli_manual(c: Circuit, a: int, b: int, t: int) -> None:
    # standard 7-T decomposition, raw gates: pyzx 0.10's TOF gate
    # object produces graphs that break extraction, teleport_reduce,
    # AND the circuit<->graph round-trip (three library fragilities,
    # measured) — the manual form round-trips cleanly
    seq = [("HAD", (t,), None), ("CNOT", (b, t), None),
           ("T", (t,), True), ("CNOT", (a, t), None),
           ("T", (t,), False), ("CNOT", (b, t), None),
           ("T", (t,), True), ("CNOT", (a, t), None),
           ("T", (b,), False), ("T", (t,), False),
           ("HAD", (t,), None), ("CNOT", (a, b), None),
           ("T", (a,), False), ("T", (b,), True),
           ("CNOT", (a, b), None)]
    for gname, args, adj in seq:
        if adj is None:
            c.add_gate(gname, *args)
        else:
            c.add_gate(gname, *args, adjoint=adj)


def structured_toffoli(qubits: int, n_tofs: int, rng: random.Random):
    c = Circuit(qubits)
    for _ in range(n_tofs):
        a, b, t = rng.sample(range(qubits), 3)
        _toffoli_manual(c, a, b, t)
        if rng.random() < 0.5:
            c.add_gate("CNOT", rng.randrange(qubits), t)
    return c


def rule_of(label: str) -> str:
    return label.split("@")[0]


_EXTRACT_CACHE = {}


def extractable_tcount(state: ZXState) -> "int | None":
    """Rung 4's eval: T-count of the EXTRACTED circuit — the only
    T-count that corresponds to a real circuit (unextractable search
    products scored 30/30 verify-failures in rungs 3 v1/v2; safe
    rewrites preserve semantics but wander out of the extractable
    subspace). None = unextractable."""
    k = state.key()
    if k not in _EXTRACT_CACHE:
        try:
            g = state.g.copy()
            zx.full_reduce(g)
            _EXTRACT_CACHE[k] = zx.tcount(zx.extract_circuit(g))
        except Exception:
            _EXTRACT_CACHE[k] = None
    return _EXTRACT_CACHE[k]


def bf_extract(g0, budget: int, prior: "dict | None" = None, k: int = 3):
    """Best-first on extractable T-count. Unextractable states may be
    TRAVERSED (they can lead back to extractable ones) but score a
    penalty and never update best."""
    import heapq
    import itertools
    tie = itertools.count()
    start = ZXState(g0.copy())
    best, best_t = start, extractable_tcount(start)
    edge_cap = max(300, 3 * g0.num_edges())
    pq = [((best_t if best_t is not None else 10**6, 0), next(tie), start)]
    visited, nodes = {start.key()}, 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        kids = list(moves(s)) + list(macro_moves(s))
        if prior is not None:
            uni, bi = prior["unigram"], prior["bigram"]
            med = sorted(uni.values())[len(uni) // 2] if uni else 1
            prev = rule_of(s.history[-1]) if s.history else None
            table = bi.get(prev) if prev else None
            kids = sorted(kids, key=lambda c: -(
                (table.get(rule_of(c[0]), 0) if table else 0)
                + 0.01 * uni.get(rule_of(c[0]), med)))[:k]
        for _, child in kids:
            kk = child.key()
            if kk in visited:
                continue
            visited.add(kk)
            if child.g.num_edges() > edge_cap:
                continue
            nodes += 1
            et = extractable_tcount(child)
            if et is not None and (best_t is None or et < best_t):
                best, best_t = child, et
            score = (et if et is not None else 10**6,
                     child.g.num_vertices())
            heapq.heappush(pq, (score, next(tie), child))
            if nodes >= budget:
                break
    return best, best_t


def bf_markov(g0, budget: int, prior: dict, k: int = 3):
    """Best-first with bigram-ranked top-k expansion (the 293-dict,
    transplanted). Unseen rules get median unigram mass (the
    smoothing lesson)."""
    import heapq
    import itertools
    uni, bi = prior["unigram"], prior["bigram"]
    med = sorted(uni.values())[len(uni) // 2] if uni else 1

    def rank(state, children):
        prev = rule_of(state.history[-1]) if state.history else None
        table = bi.get(prev) if prev else None

        def s(lab):
            r = rule_of(lab)
            return (table.get(r, 0) if table else 0) + 0.01 * uni.get(r, med)

        return sorted(children, key=lambda c: -s(c[0]))

    tie = itertools.count()
    start = ZXState(g0.copy())
    best = start
    edge_cap = max(300, 3 * g0.num_edges())
    pq = [(zx_eval(start), next(tie), start)]
    visited, nodes = {start.key()}, 1
    while pq and nodes < budget:
        _, _, s = heapq.heappop(pq)
        kids = rank(s, list(moves(s)) + list(macro_moves(s)))[:k]
        for _, child in kids:
            kk = child.key()
            if kk in visited:
                continue
            visited.add(kk)
            if child.g.num_edges() > edge_cap:
                continue
            nodes += 1
            if zx_eval(child) < zx_eval(best):
                best = child
            heapq.heappush(pq, (zx_eval(child), next(tie), child))
            if nodes >= budget:
                break
    return best


def harvest(n: int, qubits: int, n_tofs: int, budget: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    uni: Counter = Counter()
    bi: dict = defaultdict(Counter)
    rng = random.Random(f"zx-harvest-{qubits}-{n_tofs}")
    got = 0
    for i in range(n):
        c = structured_toffoli(qubits, n_tofs, rng)
        g = c.to_graph()
        zx.simplify.to_gh(g)
        signal.alarm(180)
        try:
            best, _ = best_first_zx(g, budget=budget)
        except _Timeout:
            continue
        finally:
            signal.alarm(0)
        if tcount(best.g) >= tcount(g):
            continue  # no descent, nothing to learn
        got += 1
        prev = None
        for lab in best.history:
            r = rule_of(lab)
            uni[r] += 1
            if prev is not None:
                bi[prev][r] += 1
            prev = r
        print(f"harvest {i}: T {tcount(g)} -> {tcount(best.g)}, "
              f"path {len(best.history)}", flush=True)
    PRIOR_PATH.write_text(json.dumps(
        {"unigram": dict(uni), "bigram": {k: dict(v) for k, v in bi.items()}}))
    print(f"harvested {got}/{n} descents -> {PRIOR_PATH}")
    print("unigram:", dict(uni))


def race(n: int, qubits: int, n_tofs: int, budget: int) -> None:
    signal.signal(signal.SIGALRM,
                  lambda *_: (_ for _ in ()).throw(_Timeout()))
    prior = json.loads(PRIOR_PATH.read_text())
    rng = random.Random(f"zx-race3-{qubits}-{n_tofs}")  # disjoint stream
    stats = {a: {"w": 0, "t": 0, "l": 0, "T": 0} for a in
             ("bf-extract", "bf-extract-mk3")}
    badver = 0
    tg_sum = 0
    print(f"# rung-3 race — toffoli nets q={qubits} tofs={n_tofs} "
          f"n={n} budget={budget}")
    print(f"{'i':>3} {'T0':>4} {'greedy':>7} {'bf-extr':>8} {'extr-mk3':>8}")
    for i in range(n):
        c = structured_toffoli(qubits, n_tofs, rng)
        gg = c.to_graph()
        zx.full_reduce(gg)
        tg = zx.tcount(gg)
        tg_sum += tg
        row = {}
        for arm in stats:
            gs = c.to_graph()
            zx.simplify.to_gh(gs)
            signal.alarm(240)
            try:
                if arm == "bf-extract":
                    best, bt = bf_extract(gs, budget)
                else:
                    best, bt = bf_extract(gs, budget, prior=prior)
                tb = bt if bt is not None else 10**9
                if bt is not None and not verify_equal(c, best.g, qubits):
                    badver += 1
                    tb = 10**9
            except _Timeout:
                tb = 10**9
            finally:
                signal.alarm(0)
            row[arm] = tb
            stats[arm]["T"] += min(tb, 10**6)
            key = "w" if tb < tg else "t" if tb == tg else "l"
            stats[arm][key] += 1
        print(f"{i:>3} {zx.tcount(c.to_graph()):>4} {tg:>7} "
              f"{row['bf-extract']:>8} {row['bf-extract-mk3']:>8}",
              flush=True)
    print(f"TOTALS greedy mean T {tg_sum / n:.1f}")
    for arm, st in stats.items():
        print(f"  {arm}: wins {st['w']}, ties {st['t']}, losses {st['l']}")
    print(f"verify-failures {badver}; bar: wins >= {0.2 * n:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true")
    ap.add_argument("--race", action="store_true")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--qubits", type=int, default=6)
    ap.add_argument("--tofs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=500)
    a = ap.parse_args()
    if a.harvest:
        harvest(a.n, a.qubits, a.tofs, a.budget)
    if a.race:
        race(a.n, a.qubits, a.tofs, a.budget)

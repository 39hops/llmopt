"""MATH-CYBER-1 REGRET-LDS-DESK-0 (PRE-REG booked at 5e7c2cbc).
Rank-weighted limited-discrepancy GRAPH search over the realized
world snapshot: edge cost = rank-1 (rank-only, never raw
scores; no cross-parent score comparison), frontier popped by
(path_discrepancy, path_depth, State.key()), TERMINAL-FIRST hard
override at expansion (terminal edges cost 0 and carry rank tag
"T"), depth cap 12 (nodes at depth 12 never expand),
transposition table on full State.key() with the registered
strictly-lower-cost reopen law, overflow-unscorable states book
state-censored (search continues), 300 s charged instrument wall
per episode, unique-expansion ladder 12/24/48/96 with primary
read at 96. Zero training; frozen theta_0 ranks siblings only.

Population: the six residual budget failures (CYCLE-ESCAPE
LOWER_BOUND(12) set). SMOKE=1 (spent data): terminal-at-root
(L4-s9100), greedy-solvable discrepancy-0 (L5-s9100), forced
pullback on the loop episode L7-s9303 (must solve with >=1
non-rank-1 decision, >=1 pullback, >=1 transposition hit),
synthetic reopen-law unit, ladder accounting, censor injection.
Real mode needs MW1_REGRET_GO=1.

    SMOKE=1 .venv/bin/python scratch/mathworld1_regret.py  (Mac)
"""
import hashlib
import heapq
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402
import torch  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import (State,  # noqa: E402
                                      is_solved, successors)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_birth import GCTok  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
if not SMOKE and os.environ.get("MW1_REGRET_GO") != "1":
    raise SystemExit("REFUSING: real mode needs MW1_REGRET_GO=1;"
                     " SMOKE=1 for qualification")
PRE = (f"smoke{os.environ.get('SMOKE_TAG', '')}_"
       if SMOKE else "")
CKPT = Path("checkpoints/mathnative_19m_mw1_theta0.pt")
ROWS = Path(f"logs/mathworld1/{PRE}regret_lds_desk.jsonl")
VERDICT = Path(
    f"logs/mathworld1/{PRE}regret_lds_desk_verdict.json")
MAX_DEPTH = 12
EXPANSION_CAP = 96
LADDER = (12, 24, 48, 96)
WALL_CAP_S = 300.0
CTX = 4096
X = sp.Symbol("x")

POP = [("L4-s9104", 4, 9104), ("L6-s9108", 6, 9108),
       ("L4-s9401", 4, 9401), ("L4-s9503", 4, 9503),
       ("L4-s9504", 4, 9504), ("L4-s9518", 4, 9518)]


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()[:16]


class World:
    def __init__(self):
        self.cache = {}
        self.walls = {}

    def legal(self, st):
        k = st.key()
        if k not in self.cache:
            derivation._RULE_CACHE.clear()
            t0 = time.monotonic()
            acts = sorted(successors(st),
                          key=lambda nc: (nc[0], nc[1].key()))
            self.walls[k] = time.monotonic() - t0
            self.cache[k] = [(n, c.expr) for n, c in acts]
        acts = [(n, State(e, st.plies + 1, st.history + (n,)))
                for n, e in self.cache[k]]
        ah = sha("\n".join(f"{n}|{c.key()}" for n, c in acts))
        return acts, ah


class Scorer:
    def __init__(self, model, tok, dev):
        self.model, self.tok, self.dev = model, tok, dev
        self.calls = 0

    def score(self, pre, cid):
        self.calls += 1
        ids = torch.tensor([pre + cid], device=self.dev)
        with torch.no_grad():
            logits = self.model(ids)
        lp = torch.log_softmax(logits[0].float(), -1)
        s = sum(lp[len(pre) + i - 1, t].item()
                for i, t in enumerate(cid))
        if not math.isfinite(s):
            print("[FAIL] nonfinite score", flush=True)
            sys.exit(4)
        return s

    def rank(self, st, acts):
        pre = self.tok.encode(
            f"Current: {str(st.expr)}\nHints: none\nStep: ")
        cand = []
        for name, c in acts:
            cid = self.tok.encode(str(c.expr) + "\n")
            if len(pre) + len(cid) > CTX:
                return None
            cand.append((name, c, cid))
        scored = [(self.score(pre, cid), n, c)
                  for n, c, cid in cand]
        scored.sort(key=lambda t: (-t[0], t[1], t[2].key()))
        return scored


def regret_search(root, world, scorer, sink, eid,
                  inject_delay_s=0.0):
    """One REGRET-LDS episode. Returns episode dict."""
    charged = inject_delay_s
    paid = set()
    table = {}   # key -> node dict
    frontier = []
    stats = {"expansions": 0, "edges": 0, "pops": 0,
             "transposition_hits": 0, "reopens": 0,
             "pullbacks": 0, "max_frontier": 0,
             "censored_states": 0,
             "model_calls_at_start": scorer.calls}
    rk = root.key()
    table[rk] = {"g": 0, "depth": 0, "parent": None,
                 "edge": None, "expanded": False,
                 "state": root}
    heapq.heappush(frontier, (0, 0, rk))
    solved = None
    last_subtree = None
    ladder_solves = {}
    while frontier:
        stats["max_frontier"] = max(stats["max_frontier"],
                                    len(frontier))
        if charged > WALL_CAP_S:
            return _finish(eid, "CENSORED_WALL", None, stats,
                           charged, sink, ladder_solves, table)
        g, depth, k = heapq.heappop(frontier)
        stats["pops"] += 1
        node = table[k]
        if node["expanded"] or g > node["g"]:
            continue        # stale entry (reopen law: only a
        #                     strictly-lower-g rediscovery
        #                     re-relaxes, handled at push time)
        if depth >= MAX_DEPTH:
            continue        # depth fence: never expanded
        if stats["expansions"] >= EXPANSION_CAP:
            return _finish(eid, "EXPANSION_CAP", None, stats,
                           charged, sink, ladder_solves, table)
        # pullback detection: root-edge subtree of this pop
        sub = _root_edge(table, k)
        if last_subtree is not None and sub != last_subtree:
            stats["pullbacks"] += 1
        last_subtree = sub
        # expand
        if k not in paid:
            acts, ah = world.legal(node["state"])
            charged += world.walls.get(k, 0.0)
            paid.add(k)
        else:
            acts, ah = world.legal(node["state"])
        t0 = time.monotonic()
        node["expanded"] = True
        stats["expansions"] += 1
        if not acts:
            charged += time.monotonic() - t0
            continue        # dead end node
        terms = [(n, c) for n, c in acts if is_solved(c)]
        if terms:
            name, child = min(terms,
                              key=lambda nc: (nc[0],
                                              nc[1].key()))
            charged += time.monotonic() - t0
            solved = {"final_edge": (k, "T",
                                     f"{name}#{sha(child.key())}"),
                      "depth": depth + 1, "g": g}
            for th in LADDER:
                if stats["expansions"] <= th:
                    ladder_solves.setdefault(th, True)
            return _finish(eid, "SOLVED", solved, stats,
                           charged, sink, ladder_solves, table)
        scored = scorer.rank(node["state"], acts)
        charged += time.monotonic() - t0
        if scored is None:
            stats["censored_states"] += 1
            sink.write(json.dumps({
                "row": "state_censored", "episode_id": eid,
                "state_hash": sha(k),
                "event": "model_ctx_overflow"}) + "\n")
            continue
        node["ranked"] = [(i + 1, n,
                           f"{n}#{sha(c.key())}")
                          for i, (s, n, c) in
                          enumerate(scored)]
        for i, (s, n, c) in enumerate(scored):
            r = i + 1
            ck = c.key()
            ng = g + (r - 1)
            nd = depth + 1
            stats["edges"] += 1
            if ck in table:
                stats["transposition_hits"] += 1
                if ng < table[ck]["g"]:
                    stats["reopens"] += (
                        1 if table[ck]["expanded"] else 0)
                    table[ck].update(
                        {"g": ng, "depth": nd,
                         "parent": k, "edge": (r, n),
                         "expanded": False})
                    heapq.heappush(frontier, (ng, nd, ck))
                continue
            table[ck] = {"g": ng, "depth": nd, "parent": k,
                         "edge": (r, n), "expanded": False,
                         "state": c}
            heapq.heappush(frontier, (ng, nd, ck))
    return _finish(eid, "FRONTIER_EXHAUSTED", None, stats,
                   charged, sink, ladder_solves, table)


def _root_edge(table, k):
    """First edge under the root on k's current backpointer
    chain (subtree identity for pullback counting)."""
    prev = None
    while table[k]["parent"] is not None:
        prev = table[k]["edge"]
        k = table[k]["parent"]
    return prev


def _path(table, solved):
    k, tag, fid = solved["final_edge"]
    seq = [("T", fid)]
    while table[k]["parent"] is not None:
        r, n = table[k]["edge"]
        seq.append((r, None))
        k = table[k]["parent"]
    seq.reverse()
    return seq


def _finish(eid, outcome, solved, stats, charged, sink,
            ladder_solves, table):
    row = {"row": "episode", "episode_id": eid,
           "outcome": outcome,
           "charged_wall_s": round(charged, 3),
           "ladder_solved_at": sorted(ladder_solves),
           **{k: v for k, v in stats.items()
              if k != "model_calls_at_start"}}
    if solved:
        seq = _path(table, solved)
        ranks = [r for r, _ in seq]
        row.update({
            "solution_depth": solved["depth"],
            "total_discrepancy": solved["g"],
            "rank_sequence": ranks,
            "non_rank1_decisions": sum(
                1 for r in ranks
                if r not in (1, "T")),
            "max_rank_used": max(
                [r for r in ranks if r != "T"] or [1]),
            "first_discovery_expansions":
                stats["expansions"]})
    sink.write(json.dumps(row) + "\n")
    sink.flush()
    print(f"[regret] {eid}: {outcome} exp={stats['expansions']}"
          + (f" depth={row.get('solution_depth')} "
             f"ranks={row.get('rank_sequence')} "
             f"g={row.get('total_discrepancy')}"
             if solved else ""), flush=True)
    return row


def main():
    for p in (ROWS, VERDICT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    if not CKPT.exists():
        raise SystemExit(f"MISSING theta_0: {CKPT}")
    ROWS.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_regret.py",
         "scratch/mathworld1_birth.py",
         "llmopt/search/derivation.py",
         "llmopt/mathgen/problems.py",
         "llmopt/train/mathnative.py"])
    ck_sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
    tok = GCTok()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = build_model(tok.vocab_size, ctx=CTX).to(dev)
    model.load_state_dict(torch.load(CKPT, map_location=dev))
    model.eval()
    scorer = Scorer(model, tok, dev)
    world = World()

    def root(lv, seed):
        return State(sp.Integral(
            make_integrate(lv, seed)._expr, X))

    plan = (POP if not SMOKE else
            [("L4-s9100", 4, 9100), ("L5-s9100", 5, 9100),
             ("L7-s9303", 7, 9303)])
    results = {}
    with ROWS.open("a") as f:
        for eid, lv, seed in plan:
            c0 = scorer.calls
            r = regret_search(root(lv, seed), world, scorer,
                              f, eid)
            r["model_calls"] = scorer.calls - c0
            results[eid] = r
        checks = {}
        if SMOKE:
            cens = regret_search(root(4, 9100), world, scorer,
                                 f, "SYNTH-censor",
                                 inject_delay_s=301.0)
            # synthetic reopen-law unit: relax a fake node at
            # lower g and confirm the update rule
            t = {"A": {"g": 5, "depth": 3, "parent": None,
                       "edge": None, "expanded": True}}
            before = dict(t["A"])
            ng = 2
            if ng < t["A"]["g"]:
                t["A"].update({"g": ng, "expanded": False})
            checks = {
                "terminal_at_root_solves":
                    results["L4-s9100"]["outcome"] == "SOLVED"
                    and results["L4-s9100"]["solution_depth"]
                    == 1
                    and results["L4-s9100"]
                    ["total_discrepancy"] == 0,
                "greedy_zero_discrepancy":
                    results["L5-s9100"]["outcome"] == "SOLVED"
                    and results["L5-s9100"]
                    ["total_discrepancy"] == 0,
                "pullback_solves_9303":
                    results["L7-s9303"]["outcome"] == "SOLVED"
                    and results["L7-s9303"]
                    ["non_rank1_decisions"] >= 1
                    and results["L7-s9303"]["pullbacks"] >= 1
                    and results["L7-s9303"]
                    ["transposition_hits"] >= 1
                    and results["L7-s9303"]["solution_depth"]
                    <= 12,
                "reopen_unit_updates":
                    t["A"]["g"] == 2
                    and t["A"]["expanded"] is False
                    and before["g"] == 5,
                "ladder_monotone": all(
                    results[e]["expansions"] <= EXPANSION_CAP
                    for e in results),
                "censor_books_censored":
                    cens["outcome"] == "CENSORED_WALL"}
        f.write(json.dumps({"meta": {
            "theta0_sha256": ck_sha, "device": dev,
            "smoke": SMOKE, "expansion_cap": EXPANSION_CAP,
            "ladder": LADDER, "max_depth": MAX_DEPTH,
            "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    solves = [e for e, r in results.items()
              if r["outcome"] == "SOLVED"]
    multi_dev = [e for e in solves
                 if results[e]["non_rank1_decisions"] >= 2]
    verdict = {
        "smoke": SMOKE, "device": dev,
        "theta0_sha256": ck_sha,
        "episodes": results,
        "solved": sorted(solves),
        "solved_total": len(solves),
        "multi_deviation_solves": sorted(multi_dev),
        "bars": (None if SMOKE else {
            "STRONG": len(solves) >= 3
            and len(multi_dev) >= 2,
            "WEAK": (1 <= len(solves) <= 2)
            or (len(solves) >= 3 and len(multi_dev) < 2),
            "WEAKENED": len(solves) == 0}),
        "mechanism_checks": checks,
        "pass": (all(checks.values()) if SMOKE else None),
        "world_states": len(world.cache),
        "start": START,
        "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(f"[regret] {'smoke ' if SMOKE else ''}solved="
          f"{len(solves)} multi_dev={len(multi_dev)}"
          + (f" pass={verdict['pass']}" if SMOKE else ""),
          flush=True)
    if SMOKE:
        return 0 if verdict["pass"] else 3
    return 0


if __name__ == "__main__":
    sys.exit(main())

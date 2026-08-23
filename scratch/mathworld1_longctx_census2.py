"""MATH-CYBER-1 TRAIN-band exposure census v2 (outside-review
correction of OBSERVATION MATH-CYBER-1-LONGCTX-0): the v1 census
pooled ALL legal siblings at greedy-visited states — that is
CANDIDATE-SCORING exposure, not training-target exposure. This
re-walk of the frozen TRAIN band (seeds 9200-9249, L4-7, frozen
world constants) separates two populations and adds incidence:

  P1  ALL LEGAL candidates: per-ctx-bucket action counts,
      DECISION incidence (any candidate at the decision > ctx),
      EPISODE incidence (any visited decision trips), for
      ctx in {512, 1024, 2048, 4096}.
  P2  POSITIVE TARGET edges: the CHOSEN transitions of SOLVED
      greedy episodes only (the success-gated training
      population under the frozen dose law) — length histogram
      and per-bin counts.

Lengths are full scoring sequences under the grammar-closed
proposal (ATOMS + per-byte fallback). Receipt:
logs/mathworld1/longctx_census2.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_longctx_census2.py    (Mac)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import (State, hce,  # noqa: E402
                                      is_solved, successors)
from llmopt.train.mathnative import MathTokenizer  # noqa: E402

OUT = Path("logs/mathworld1/longctx_census2.json")
LEVELS = [4, 5, 6, 7]
SEEDS = range(9200, 9250)
MAX_DECISIONS = 12
WALL_CAP_S = 60.0
X = sp.Symbol("x")
CTXS = [512, 1024, 2048, 4096]


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_longctx_census2.py",
         "llmopt/search/derivation.py", "llmopt/mathgen/problems.py",
         "llmopt/train/mathnative.py"])
    tok = MathTokenizer()

    def plen(s):
        n = i = 0
        while i < len(s):
            for t in tok._by_len:
                if s.startswith(t, i):
                    n += 1
                    i += len(t)
                    break
            else:
                n += len(s[i].encode())
                i += 1
        return n

    def binof(n):
        for b in CTXS:
            if n <= b:
                return f"<={b}"
        return ">4096"

    p1_bins = {}
    dec_total = 0
    dec_over = {c: 0 for c in CTXS}
    ep_over = {c: 0 for c in CTXS}
    ep_records = []
    chosen_lens_solved = []
    n_solved = 0
    t0 = time.monotonic()
    for lv in LEVELS:
        for seed in SEEDS:
            prob = make_integrate(lv, seed)
            st = State(sp.Integral(prob._expr, X))
            t_ep = time.monotonic()
            outcome = "budget_exhausted"
            ep_max = 0
            ep_chosen = []
            for _ in range(MAX_DECISIONS):
                if is_solved(st):
                    outcome = "solved"
                    break
                if time.monotonic() - t_ep > WALL_CAP_S:
                    outcome = "wall_cap"
                    break
                acts = sorted(successors(st),
                              key=lambda nc: (nc[0], nc[1].key()))
                if not acts:
                    outcome = "dead_end"
                    break
                parent = str(st.expr)
                lens = []
                for name, c in acts:
                    n = plen(f"Current: {parent}\nHints: none\n"
                             f"Step: {str(c.expr)}\n")
                    lens.append(n)
                    p1_bins[binof(n)] = p1_bins.get(binof(n),
                                                    0) + 1
                dec_total += 1
                mx = max(lens)
                ep_max = max(ep_max, mx)
                for c_ in CTXS:
                    if mx > c_:
                        dec_over[c_] += 1
                pick = min(range(len(acts)),
                           key=lambda i: (hce(acts[i][1]),
                                          acts[i][0],
                                          acts[i][1].key()))
                ep_chosen.append(lens[pick])
                st = acts[pick][1]
            else:
                outcome = ("solved" if is_solved(st)
                           else "budget_exhausted")
            if outcome == "solved":
                n_solved += 1
                chosen_lens_solved.extend(ep_chosen)
            for c_ in CTXS:
                if ep_max > c_:
                    ep_over[c_] += 1
            ep_records.append({"episode_id": f"L{lv}-s{seed}",
                               "outcome": outcome,
                               "max_candidate_len": ep_max})
    cl = sorted(chosen_lens_solved)
    p2_bins = {}
    for n in cl:
        p2_bins[binof(n)] = p2_bins.get(binof(n), 0) + 1
    receipt = {
        "p1_all_legal": {
            "actions_bins": p1_bins, "decisions": dec_total,
            "decision_incidence_over_ctx": dec_over,
            "episode_incidence_over_ctx": ep_over},
        "p2_positive_targets_solved_only": {
            "episodes_solved": n_solved,
            "chosen_edges": len(cl),
            "bins": p2_bins,
            "med": cl[len(cl) // 2] if cl else None,
            "p90": cl[int(0.9 * len(cl))] if cl else None,
            "max": cl[-1] if cl else None},
        "episodes": ep_records,
        "wall_s": round(time.monotonic() - t0, 1),
        "start": START, "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("episodes", "start")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

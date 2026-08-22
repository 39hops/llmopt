"""MATH-CYBER-1 TRAIN-sidecar long-context exposure census
(AMENDMENT MATH-CYBER-1-DESK-0-DESIGN band freeze: TRAIN =
seeds 9200-9249, generated here for the first time; disjoint from
CALIBRATION 9100-9109 and the ungenerated ADAPT/HOLDOUT bands).

Farms legal transitions by greedy-hce walk over L4-7 x seeds
9200-9249 (200 episodes, 12-decision budget, 60 s wall cap — the
frozen world contract constants) and measures, for every legal
action encountered, the full scoring-sequence token length
("Current: {parent}\\nHints: none\\nStep: {child}\\n") under the
grammar-closed proposal (ATOMS greedy + one token per uncovered
byte; corpus asserted ASCII so char==byte). Bins: <=512, 513-1024,
1025-2048, 2049-4096, >4096. This measures NATURAL long-context
exposure available for training; nothing is derived from the
calibration fixtures.

Receipts: logs/mathworld1/longctx_census.jsonl (one summary row
per episode + meta; refuse-if-exists). Transition rows themselves
are not persisted here — this is a pricing census, not the
training farm; the farm re-runs under the eventual prereg.

    .venv/bin/python scratch/mathworld1_longctx_census.py     (Mac)
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

OUT = Path("logs/mathworld1/longctx_census.jsonl")
LEVELS = [4, 5, 6, 7]
SEEDS = range(9200, 9250)
MAX_DECISIONS = 12
WALL_CAP_S = 60.0
X = sp.Symbol("x")
BINS = [512, 1024, 2048, 4096]


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld1_longctx_census.py",
         "llmopt/search/derivation.py", "llmopt/mathgen/problems.py",
         "llmopt/train/mathnative.py"])
    tok = MathTokenizer()

    def prop_len(s):
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
        for b in BINS:
            if n <= b:
                return f"<={b}"
        return ">4096"

    totals = {f"<={b}": 0 for b in BINS}
    totals[">4096"] = 0
    non_ascii = 0
    n_eps = n_actions = n_solved = 0
    t_start = time.monotonic()
    with OUT.open("a") as f:
        for lv in LEVELS:
            for seed in SEEDS:
                prob = make_integrate(lv, seed)
                st = State(sp.Integral(prob._expr, X))
                t_ep = time.monotonic()
                ep_bins = {}
                outcome = "budget_exhausted"
                for _ in range(MAX_DECISIONS):
                    if is_solved(st):
                        outcome = "solved"
                        break
                    if time.monotonic() - t_ep > WALL_CAP_S:
                        outcome = "wall_cap"
                        break
                    acts = sorted(successors(st),
                                  key=lambda nc: (nc[0],
                                                  nc[1].key()))
                    if not acts:
                        outcome = "dead_end"
                        break
                    parent = str(st.expr)
                    for name, c in acts:
                        seq = (f"Current: {parent}\nHints: none\n"
                               f"Step: {str(c.expr)}\n")
                        if not seq.isascii():
                            non_ascii += 1
                        b = binof(prop_len(seq))
                        ep_bins[b] = ep_bins.get(b, 0) + 1
                        totals[b] += 1
                        n_actions += 1
                    st = min(acts, key=lambda nc: (hce(nc[1]),
                                                   nc[0],
                                                   nc[1].key()))[1]
                else:
                    outcome = ("solved" if is_solved(st)
                               else "budget_exhausted")
                if outcome == "solved":
                    n_solved += 1
                n_eps += 1
                f.write(json.dumps({
                    "episode_id": f"L{lv}-s{seed}",
                    "outcome": outcome, "bins": ep_bins}) + "\n")
                f.flush()
        summary = {"episodes": n_eps, "actions": n_actions,
                   "solved": n_solved, "bins_total": totals,
                   "non_ascii_sequences": non_ascii,
                   "wall_s": round(time.monotonic() - t_start, 1),
                   "start": START,
                   "completion_commit": completion_commit()}
        f.write(json.dumps({"meta": summary}) + "\n")
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "start"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

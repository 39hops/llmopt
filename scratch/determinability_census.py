"""Determinability census (PRE-REG DATA-CEIL rung A, 2026-08-10).

n_legal_successors per unique gen-4 cur state via
farm_dist_rows.enumerate_moves (axiom bridge, deadline-walled,
house sympy fallback). 4,000 states (farm_dist_rows precedent,
fresh shuffle seed). Rows STREAM (killed-worker doctrine).
Registered observables: histogram of n_legal_successors and
frac(n==1) — the unique-successor share, the number the lab does
not have. States where BOTH paths fail book as UNSCORED with
count (rule 4: coverage before numbers).

CPU only. Usage: .venv/bin/python scratch/determinability_census.py
"""
import json
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402

from llmopt.search.derivation import State, successors  # noqa: E402
from train_mathnative import load_rows  # noqa: E402

# enumerate_moves inlined from scratch/farm_dist_rows.py (that
# module has no __main__ guard — importing it RUNS the farm).
# Same axiom-bridge-with-house-fallback semantics, verbatim.
USE_AXIOM = os.environ.get("USE_AXIOM", "1") == "1"
if USE_AXIOM:
    import axiom_sym as ax  # noqa: E402


def enumerate_moves(cur, expr):
    """-> [(rule_name, child_sstr)]; axiom bridge (deadline-walled,
    expired taxed) with house fallback on parse rejects."""
    if USE_AXIOM:
        try:
            r = ax.successors(ax.parse_sstr(cur), True, 10000)
            if not r["expired"]:
                return [(n, str(c)) for n, c in r["rows"]]
        except Exception:
            pass  # I-fence / parse reject -> house path
    return [(n, sp.sstr(s_.expr)) for n, s_ in successors(State(expr))]

N_STATES = 4000
SHUF_SEED = 91_300_000  # fresh (farm used 99.2M)
OUT = "logs/data_ceil/determinability_gen4.jsonl"


def main():
    rows = load_rows(gen4=True)
    seen, pool = set(), []
    for r in rows:
        c = r["cur"].replace(" ", "")
        if c not in seen:
            seen.add(c)
            pool.append(r["cur"])
    random.Random(SHUF_SEED).shuffle(pool)
    print(f"unique cur states: {len(pool)}; censusing {N_STATES}",
          flush=True)

    os.makedirs("logs/data_ceil", exist_ok=True)
    out = open(OUT, "a")
    counts = {}
    unscored = 0
    for k, cur in enumerate(pool[:N_STATES]):
        try:
            expr = sp.sympify(cur)
            moves = enumerate_moves(cur, expr)
            n = len(moves)
            counts[n] = counts.get(n, 0) + 1
            out.write(json.dumps(
                {"cur": cur, "n_succ": n, "status": "ok"}) + "\n")
        except Exception as e:
            unscored += 1
            out.write(json.dumps(
                {"cur": cur, "status": "unscored",
                 "err": type(e).__name__}) + "\n")
        out.flush()
        if (k + 1) % 250 == 0:
            print(f"[{k + 1}/{N_STATES}] unscored={unscored}",
                  flush=True)

    scored = sum(counts.values())
    print(f"\nCOVERAGE: {scored} scored / {unscored} unscored "
          f"of {min(N_STATES, len(pool))}")
    print(f"frac(n==1) = {counts.get(1, 0)}/{scored} = "
          f"{counts.get(1, 0) / max(scored, 1):.4f}")
    print("histogram (n_succ: states):")
    for n in sorted(counts):
        print(f"  {n:4d}: {counts[n]}")


if __name__ == "__main__":
    main()

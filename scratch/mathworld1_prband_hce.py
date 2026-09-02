"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-V2-CROSSOVER-ASSESSMENT-0 —
candidate-level HCE anatomy of the burned V1 horizon, recomputed
from the persisted child expressions in
logs/mathworld1/prband/qualified_blocks.jsonl (read-only; no new
parent, no rule call, no model). For every variant of every block:
every i_unprod candidate's HCE terms (unsolved atoms, count_ops,
plies constant), the winner's HCE gap, tie count, whether the
winner is the lowest term index / smallest child ops / smallest
residual ops, and per-candidate structure (guessed A, residual r,
the source term f.args[i], its monomial degree and whether it is
the T'-type term). Writes
logs/mathworld1/prband_verify/hce_anatomy.jsonl + summary json.

    .venv/bin/python scratch/mathworld1_prband_hce.py
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sympy as sp  # noqa: E402

X = sp.Symbol("x")
SRC = Path("logs/mathworld1/prband/qualified_blocks.jsonl")
CEN = Path("logs/mathworld1/prband/horizon_census.jsonl")
OUTD = Path("logs/mathworld1/prband_verify")
ROWS = OUTD / "hce_anatomy.jsonl"
SUMM = OUTD / "hce_anatomy_summary.json"
DIST = {"smallA": X**X, "smallB": 1 / (X + sp.log(X)),
        "after": sp.sin(sp.sin(X))}


def unsolved(e):
    return len(e.atoms(sp.Derivative, sp.Integral, sp.Limit))


def main():
    if ROWS.exists() or SUMM.exists():
        raise SystemExit("REFUSING: outputs exist")
    cen = {json.loads(l)["horizon_index"]: json.loads(l)
           for l in open(CEN)}
    out = open(ROWS, "w")
    agg = defaultdict(Counter)
    n = 0
    for l in open(SRC):
        q = json.loads(l)
        r = cen[q["horizon_index"]]
        f = sp.sympify(r["target_integrand"])
        terms = list(f.args)
        k = int(r["base_signature"][-1])
        for vt, v in q["variants"].items():
            D = DS = DIST[vt]
            cands = []
            for c in v["candidates"]:
                e = eval(c["child_srepr"], sp.__dict__)  # exact srepr
                ops = int(sp.count_ops(e))
                uns = unsolved(e)
                hce = 100.0 * uns + ops + 0.1  # plies 1 constant
                row = {"rule": c["rule"], "ti": c["param_index"],
                       "ops": ops, "unsolved": uns, "hce": hce,
                       "is_label": c["is_label"]}
                if c["rule"] == "i_unprod":
                    ints = [a for a in e.atoms(sp.Integral)
                            if a.function != D]
                    # top-level residual integral: the Integral term
                    # of the child Add whose integrand is not D
                    top = [a for a in (e.args if isinstance(e, sp.Add)
                                       else [e])
                           if isinstance(a, sp.Integral)
                           and a.function != DS]
                    rres = top[0].function if top else sp.Integer(0)
                    A = e - sum(a for a in (e.args if isinstance(e, sp.Add)
                                            else [e])
                                if isinstance(a, sp.Integral))
                    t = terms[c["param_index"]]
                    poly = sp.cancel(t / (t.atoms(sp.sin, sp.cos, sp.exp)
                                          .pop())) if t.atoms(
                        sp.sin, sp.cos, sp.exp) else t
                    row.update({
                        "A_ops": int(sp.count_ops(A)),
                        "r_ops": int(sp.count_ops(rres)),
                        "r_nterms": len(rres.args) if isinstance(
                            rres, sp.Add) else (0 if rres == 0 else 1),
                        "term": sp.sstr(t),
                        "term_deg": int(sp.degree(poly, X))
                        if poly.is_polynomial(X) else -1,
                        "term_fn": sorted(fn.func.__name__ for fn in
                                          t.atoms(sp.sin, sp.cos,
                                                  sp.exp)),
                        "A": sp.sstr(A)})
                cands.append(row)
            up = [c for c in cands if c["rule"] == "i_unprod"]
            up_sorted = sorted(up, key=lambda c: c["hce"])
            win = [c for c in cands if c["is_label"]][0]
            best = up_sorted[0]["hce"]
            gap = (up_sorted[1]["hce"] - best) if len(up_sorted) > 1 \
                else None
            ties = sum(1 for c in up if c["hce"] == best)
            allbest = min(c["hce"] for c in cands)
            rec = {"horizon_index": q["horizon_index"], "variant": vt,
                   "k": k, "primary": r["primary_variant"] == vt,
                   "cand_sig_id": r["variants"][vt]["cand_sig_id"],
                   "gold_ti": win["ti"], "gold_rule": win["rule"],
                   "unprod_tis": sorted(c["ti"] for c in up),
                   "gap": gap, "ties_at_best": ties,
                   "min_hce_ties_recorded": v["min_hce_ties"],
                   "winner_is_lowest_ti": win["ti"] == min(
                       c["ti"] for c in up),
                   "winner_is_min_ops": win["ops"] == min(
                       c["ops"] for c in up),
                   "winner_is_min_r_ops": win.get("r_ops") == min(
                       c["r_ops"] for c in up),
                   "winner_is_global_min": win["hce"] == allbest,
                   "unsolved_constant": len({c["unsolved"]
                                             for c in up}) == 1,
                   "cands": cands}
            out.write(json.dumps(rec) + "\n")
            key = (vt, k)
            a = agg[key]
            a["n"] += 1
            a["lowest"] += rec["winner_is_lowest_ti"]
            a["min_ops"] += rec["winner_is_min_ops"]
            a["min_r_ops"] += rec["winner_is_min_r_ops"]
            a["tie"] += ties > 1
            a["gap0"] += (gap == 0)
            a["unsolved_const"] += rec["unsolved_constant"]
            a["gold_unprod"] += win["rule"] == "i_unprod"
            n += 1
        if q["horizon_index"] % 100 == 0:
            print("[", q["horizon_index"], "]", flush=True)
    out.close()
    summ = {"n_variant_rows": n,
            "by_variant_k": {f"{vt}|k={k}": dict(c)
                             for (vt, k), c in agg.items()},
            "rows_sha256": hashlib.sha256(ROWS.read_bytes()).hexdigest()}
    SUMM.write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

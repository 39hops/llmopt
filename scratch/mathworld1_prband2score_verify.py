"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-V2-SCORING-EXECUTION-0 —
INDEPENDENT METRIC RECONSTRUCTION from the raw per-token log-prob
receipt (logs/mathworld1/prband2score/raw_scores.jsonl). Imports
nothing from the scorer: own 9-token sums, own semantic association,
own strict top-1 / SCORE-TIE, own A/B accuracy, pair classes,
top-action switches, target margins d = S(A0) - S(B0), flip classes
with the frozen 1e-05 noise bound, MASK0 spreads and sanities,
FULL-v-MASK0 tables, the 2 x 2 matrix, and the F0 / F_LEN / F_SIGN /
F_ORDER baselines from the frozen primary artifact. Writes
logs/mathworld1/prband2score_verify/verify_receipt.json.

    .venv/bin/python scratch/mathworld1_prband2score_verify.py
"""
import hashlib
import itertools
import json
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

IN = Path("logs/mathworld1/prband2score")
OUT = Path("logs/mathworld1/prband2score_verify")
PRIMARY = "logs/mathworld1/prband2prod/primary.jsonl"
NOISE = 1e-05
SEM = [("i_sum", "I", 0, "none", -1), ("i_unprod", "I", 0, "term_index", 1),
       ("i_unprod", "I", 0, "term_index", 3),
       ("i_unprod", "I", 0, "term_index", 5)]
A0, B0 = SEM[1], SEM[2]
NAME = {SEM[0]: "i_sum", A0: "A0", B0: "B0", SEM[3]: "I0/t5"}
problems = []


def check(c, m):
    if not c:
        problems.append(m)
        print("PROBLEM:", m, flush=True)


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    rec = {"verifier_commit": subprocess.run(["git", "rev-parse", "HEAD"],
                                             capture_output=True,
                                             text=True).stdout.strip(),
           "verifier_file_sha256": fsha(__file__),
           "input_sha256": {f.name: fsha(f) for f in sorted(IN.glob("*.json*"))},
           "primary_sha256": fsha(PRIMARY)}
    check(rec["primary_sha256"] == "209391ef3b2e5c87308571d6ef309bb5724a2141"
          "60caa1b7857f4a31f9112c34", "primary pin")
    P = [json.loads(l) for l in open(PRIMARY)]
    drv = json.loads((IN / "prband2score_receipt.json").read_text())
    dcells = json.loads((IN / "cells.json").read_text())
    dbase = json.loads((IN / "baselines.json").read_text())
    n = len(P)
    check(n == 96, "N")
    gold = [tuple(p["gold_tuple"]) for p in P]
    theta = [p["theta"] for p in P]
    # ---- baselines from the primary artifact -------------------------
    best_top, best_both = 0, 0
    for order in itertools.permutations(SEM):
        corr = [g == order[0] for g in gold]
        best_top = max(best_top, sum(corr))
        best_both = max(best_both, sum(corr[k] and corr[k + 1]
                                       for k in range(0, n, 2)))
    def acc(pred):
        corr = [pred(i) == gold[i] for i in range(n)]
        return [sum(corr), sum(corr[k] and corr[k + 1] for k in range(0, n, 2))]
    f_len = acc(lambda i: A0 if P[i]["prompt_tokens"] == 89 else B0)
    f_sign = acc(lambda i: A0 if P[i]["cur"].startswith("Integral(-") else B0)
    sigs = defaultdict(Counter)
    for i, p in enumerate(P):
        sigs[json.dumps(p["cand_tuples"])][gold[i]] += 1
    look = {s: max(c, key=c.get) for s, c in sigs.items()}
    f_order = acc(lambda i: look[json.dumps(P[i]["cand_tuples"])])
    check(best_top == 48 and best_both == 0 and f_len == [96, 48]
          and f_sign == [96, 48] and f_order == [96, 48] and len(sigs) == 2,
          f"baselines {best_top} {best_both} {f_len} {f_sign} {f_order}")
    check(dbase["F0"]["max_top1"] == 48 and dbase["F_LEN"]["top1"] == 96
          and dbase["F_SIGN"]["top1"] == 96 and dbase["F_ORDER"]["top1"] == 96
          and dbase["semantic_beyond_surface_identifiable"] is False,
          "driver baselines")
    rec["baselines"] = {"F0": [best_top, best_both], "F_LEN": f_len,
                        "F_SIGN": f_sign, "F_ORDER": f_order,
                        "n_raw_signatures": len(sigs)}
    # ---- raw rows -----------------------------------------------------
    S = defaultdict(lambda: defaultdict(dict))  # (seed,rep,arm)->state->sem->sum
    nrows = 0
    for l in open(IN / "raw_scores.jsonl"):
        r = json.loads(l)
        nrows += 1
        check(len(r["lps"]) == 9 and len(r["continuation"]) == 9, "T=9")
        check(all(isinstance(v, float) and v == v and abs(v) != float("inf")
                  for v in r["lps"]), "finite")
        s = float(sum(r["lps"]))
        check(abs(s - r["sum"]) < 1e-9, "sum")
        check(tuple(r["gold"]) == gold[r["state"]]
              and r["theta"] == theta[r["state"]], f"row gold {r['state']}")
        S[(r["seed"], r["representation"], r["arm"])][r["state"]][
            tuple(r["candidate"])] = s
    check(nrows == 4 * 2 * 96 * 4, f"rows {nrows}")
    check(len(S) == 8 and all(len(v) == 96 and all(len(x) == 4 for x in
                                                   v.values())
                              for v in S.values()), "cells complete")
    # ---- own metrics -------------------------------------------------
    cells = {}
    for key in sorted(S):
        seed, rep, arm = key
        ps = []
        for i in range(n):
            sc = S[key][i]
            best = max(sc.values())
            tops = [k for k, v in sc.items() if v == best]
            pred = tops[0] if len(tops) == 1 else None
            margin = best - sorted(sc.values())[-2]
            ps.append({"pred": pred, "tie": len(tops) > 1, "margin": margin,
                       "ok": pred == gold[i], "d": sc[A0] - sc[B0]})
        pairs = []
        for k in range(0, n, 2):
            a, b = ps[k], ps[k + 1]
            both = a["ok"] and b["ok"]
            sw = a["pred"] is not None and b["pred"] is not None \
                and a["pred"] != b["pred"]
            swm = min(a["margin"], b["margin"])
            dS, dC = a["d"], b["d"]
            if dS == 0 or dC == 0:
                fl = "TIE"
            elif min(abs(dS), abs(dC)) < NOISE:
                fl = "SUBNOISE"
            elif dS > 0 and dC < 0:
                fl = "CORRECT"
            elif dS < 0 and dC > 0:
                fl = "REVERSED"
            else:
                fl = "NONE"
            pairs.append({"both": both, "both_robust": both and swm >= NOISE,
                          "one": a["ok"] != b["ok"],
                          "switch_robust": sw and swm >= NOISE,
                          "switch_noise": sw and swm < NOISE, "flip": fl,
                          "dd": dS - dC})
        m = {"top1": sum(p["ok"] for p in ps), "ties": sum(p["tie"] for p in ps),
             "A0": sum(p["ok"] for i, p in enumerate(ps) if gold[i] == A0),
             "B0": sum(p["ok"] for i, p in enumerate(ps) if gold[i] == B0),
             "both": sum(p["both"] for p in pairs),
             "both_robust": sum(p["both_robust"] for p in pairs),
             "one": sum(p["one"] for p in pairs),
             "neither": sum(not (ps[2 * j]["ok"] or ps[2 * j + 1]["ok"])
                            for j in range(n // 2)),
             "switch_robust": sum(p["switch_robust"] for p in pairs),
             "switch_noise": sum(p["switch_noise"] for p in pairs),
             "flips": dict(Counter(p["flip"] for p in pairs)),
             "median_dd": statistics.median(p["dd"] for p in pairs),
             "pred_census": dict(Counter(NAME[p["pred"]] if p["pred"] else "TIE"
                                         for p in ps)),
             "pred_by_theta": {t: dict(Counter(
                 NAME[p["pred"]] if p["pred"] else "TIE"
                 for i, p in enumerate(ps) if theta[i] == t))
                 for t in ("SIN_LOW", "COS_LOW")}}
        if arm == "MASK0":
            spread = {NAME[s]: max(S[key][i][s] for i in range(n))
                      - min(S[key][i][s] for i in range(n)) for s in SEM}
            m["spread"] = spread
            m["sanity"] = (m["top1"] <= 48 and m["both"] == 0
                           and m["switch_robust"] == 0
                           and m["flips"].get("CORRECT", 0)
                           + m["flips"].get("REVERSED", 0) == 0
                           and all(v <= NOISE for v in spread.values()))
        else:
            m["fixed_order_violated"] = m["top1"] > 48
            m["state_conditioned_both"] = m["both_robust"] > 0
            m["switch_theorem_violated"] = m["switch_robust"] > 0
        cells[f"{seed}|{rep}|{arm}"] = m
        d = dcells["cells"][f"{seed}|{rep}|{arm}"]["metrics"]
        check(d["top1"] == m["top1"] and d["score_ties"] == m["ties"]
              and d["A0_correct"] == m["A0"] and d["B0_correct"] == m["B0"]
              and d["both_correct_pairs"] == m["both"]
              and d["both_correct_robust"] == m["both_robust"]
              and d["exactly_one_pairs"] == m["one"]
              and d["neither_pairs"] == m["neither"]
              and d["switch_pairs_robust"] == m["switch_robust"]
              and d["switch_pairs_float_noise"] == m["switch_noise"]
              and d["flips"]["correct_direction"] == m["flips"].get("CORRECT", 0)
              and d["flips"]["reversed"] == m["flips"].get("REVERSED", 0)
              and d["flips"]["no_flip"] == m["flips"].get("NONE", 0)
              and abs(d["flips"]["median_dSIN_minus_dCOS"] - m["median_dd"])
              < 1e-9 and d["pred_census"] == m["pred_census"],
              f"cell {key}")
        if arm == "MASK0":
            check(d["mask0_sanity_pass"] == m["sanity"]
                  and all(abs(d["mask0_spread"][k] - v) < 1e-12
                          for k, v in m["spread"].items()), f"mask0 {key}")
    # ---- contrasts + matrix ---------------------------------------------
    matrix = {}
    for seed in ("19001", "20001"):
        for rep in ("CANONICAL", "PARAM_FIRST"):
            F = cells[f"{seed}|{rep}|FULL"]
            M = cells[f"{seed}|{rep}|MASK0"]
            matrix[f"{seed}|{rep}"] = {
                "top1": F["top1"], "both_robust": F["both_robust"],
                "switch_robust": F["switch_robust"],
                "observed_by_both_correct": F["both_robust"] > 0,
                "observed_by_switch_theorem": F["switch_robust"] > 0,
                "observed_by_top1": F["top1"] > 48,
                "mask0_sanity": M["sanity"]}
            dm = drv["matrix"][f"{seed}|{rep}"]
            check(dm["top1"] == F["top1"]
                  and dm["both_correct_robust"] == F["both_robust"]
                  and dm["mask0_sanity_pass"] == M["sanity"]
                  and (dm["classification"].startswith("STATE-CONDITIONED")
                       == (F["both_robust"] > 0 or F["top1"] > 48)),
                  f"matrix {seed} {rep}")
    obs_both = sum(v["observed_by_both_correct"] for v in matrix.values())
    obs_any = sum(v["observed_by_both_correct"] or v["observed_by_switch_theorem"]
                  or v["observed_by_top1"] for v in matrix.values())
    rec.update({"cells": cells, "matrix": matrix,
                "observed_by_both_correct": obs_both,
                "observed_by_any_frozen_theorem": obs_any,
                "mask0_all_sane": all(v["mask0_sanity"] for v in matrix.values()),
                "problems": problems,
                "verdict": "VERIFIED" if not problems else "DISCREPANCIES"})
    OUT.mkdir(parents=True)
    (OUT / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "cells"}, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

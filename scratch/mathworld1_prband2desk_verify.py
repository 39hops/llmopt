"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-V2-CROSSOVER-DESK-0 — INDEPENDENT
VERIFIER. Reconstructs the frozen desk (PRE-REG 488e8b23) from raw
WITHOUT importing any aggregate from the driver: own bank
enumeration, own base/burn uniqueness, own overlap sets, a FULL
re-qualification of all 192 parents (qualify_parent), own term
ordering + role trace, own signature function, own HCE from child
expressions, own min_hce_ties, own strict A/B classification, own
B1-B7 and decision, own robustness census; then compares against
the driver's per-row artifacts. Zero checkpoint access. Writes
logs/mathworld1/prband2desk_verify/verify_receipt.json.

    .venv/bin/python scratch/mathworld1_prband2desk_verify.py
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.search.derivation import UNSOLVED  # noqa: E402
from scratch.mathworld1_svpchal import (AFTER_D,  # noqa: E402
                                        SMALL_D, X, fsha,
                                        qualify_parent)
from scratch.mathworld1_svpcode import factor_symbols  # noqa: E402

IN = Path("logs/mathworld1/prband2desk")
OUT = Path("logs/mathworld1/prband2desk_verify")
V1_CENSUS = "logs/mathworld1/prband/horizon_census.jsonl"
V1_SHA = ("2e28abf09219d4ac5ee2cb834f22f9cbc40f750c13029a8b96979f9"
          "a7e144c4c")
DS = {"smallA": SMALL_D[0], "smallB": SMALL_D[1], "after": AFTER_D}
TARGET = {2: (("i_unprod", "I", 1, "term_index", 1),
              ("i_unprod", "I", 1, "term_index", 4), [1, 4]),
          1: (("i_unprod", "I", 1, "term_index", 1),
              ("i_unprod", "I", 1, "term_index", 3), [1, 3, 5])}
EXPECT = {"SIN_LOW": "A", "COS_LOW": "B"}
problems = []


def check(cond, msg):
    if not cond:
        problems.append(msg)
        print("PROBLEM:", msg, flush=True)


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def tup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def sig_of(cands):
    t = sorted(tup(c) for c in cands)
    js = json.dumps([list(x) for x in t], sort_keys=True,
                    separators=(",", ":"))
    return js, h(js)


def hce(e):
    return 100.0 * len(e.atoms(*UNSOLVED)) + float(sp.count_ops(e)) + 0.1


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    check(fsha(V1_CENSUS) == V1_SHA, "V1 census sha")
    drv_bases = [json.loads(l) for l in open(IN / "bases.jsonl")]
    drv_prims = [json.loads(l) for l in open(IN / "primaries.jsonl")]
    drv_bars = json.loads((IN / "bars.json").read_text())
    drv_rec = json.loads((IN / "prband2desk_receipt.json").read_text())
    rec = {"input_sha256": {f.name: fsha(f) for f in sorted(
        IN.glob("*.json*"))}}

    # ---- own bank ------------------------------------------------
    W, K = sp.sin(X) / X, 21 * X**3
    bank = []
    for e_lo in (1, 2):
        for e_hi in (5, 6, 7, 9):
            for theta in ("SIN_LOW", "COS_LOW"):
                a, b = (e_lo, e_hi) if theta == "SIN_LOW" else (e_hi, e_lo)
                for c in (11, 13, 17, 19):
                    f = (sp.expand(sp.diff(X**a * sp.sin(c * X), X))
                         + sp.expand(sp.diff(X**b * sp.cos(c * X), X))
                         + sp.Integral(W, X) + K)
                    bank.append((f"V2D|elo={e_lo}|ehi={e_hi}|theta="
                                 f"{theta}|c={c}|w={sp.sstr(W)}|K="
                                 f"{sp.sstr(K)}", f, e_lo, theta, a, b, c))
    check(len(bank) == 64 and len(drv_bases) == 64, "64 bases")
    check(len({s for s, *_ in bank}) == 64, "sig unique")
    check(len({sp.srepr(f) for _, f, *_ in bank}) == 64, "srepr unique")
    burn = {sp.sstr(sp.Add(sp.Integral(f, X), sp.Integral(D, X)))
            for _, f, *_ in bank for D in DS.values()}
    check(len(burn) == 192, "burn 192")
    for i, (s, f, *_r) in enumerate(bank):
        d = drv_bases[i]
        check(d["base_signature"] == s and d["sig_sha"] == h(s)
              and d["f"] == sp.sstr(f)
              and d["f_args"] == [sp.sstr(t) for t in f.args]
              and d["n_terms"] == len(f.args) == 6, f"base {i}")

    # ---- own overlap sets (lineage builders) ----------------------
    from scratch.mathworld1_prband import PILOT_RECEIPTS  # noqa: E402
    curs = []
    nat = {json.loads(l)["cur"] for l in open("data/matsub_paired.jsonl")}
    v1 = set()
    for l in open(V1_CENSUS):
        for v in json.loads(l)["variants"].values():
            v1.add(v["cur"])
    check(len(v1) == 5760, "v1 curs")
    pil = set()
    for pr in PILOT_RECEIPTS:
        for a2 in json.loads(Path(pr).read_text())["attempts"]:
            pil.add(a2["parent_sstr"])
    art = set()
    for fn in ("heldout_test16.jsonl", "covered_calibration.jsonl",
               "pout_robustness.jsonl"):
        for l in open(f"logs/mathworld1/svpdiet3/{fn}"):
            art.add(json.loads(l)["cur"])
    cl = set()
    for l in open("logs/mathworld1/cl1/pop/manifest.jsonl"):
        cl.add(json.loads(l)["root_cur"])
    for l in open("logs/mathworld1/cl1/pop/raw_attempts.jsonl"):
        x = json.loads(l)
        if x.get("cur"):
            cl.add(x["cur"])

    # ---- full re-qualification of all 192 parents -----------------
    prims = []
    fails = Counter()
    requal = 0
    for i, (s, f, e_lo, theta, a, b, c) in enumerate(bank):
        d = drv_bases[i]
        ok_block = True
        prow = None
        for vt, D in DS.items():
            row, why = qualify_parent(f, D)
            curs.append(row["cur"])
            dv = d["variants"][vt]
            if why:
                ok_block = False
                fails[f"{vt}:{why}"] += 1
                check(dv.get("fail") == why, f"fail mismatch {i} {vt}")
                continue
            check(dv["cur"] == row["cur"]
                  and dv["parent_srepr_sha"] == row["parent_srepr_sha"]
                  and dv["n_candidates"] == len(row["candidates"])
                  and dv["min_hce_ties"] == row["min_hce_ties"]
                  and [tup(x) for x in dv["candidates"]]
                  == [tup(x) for x in row["candidates"]]
                  and [x["is_label"] for x in dv["candidates"]]
                  == [x["is_label"] for x in row["candidates"]],
                  f"requalify {i} {vt}")
            requal += 1
            if vt == "smallA":
                prow = row
        if not ok_block:
            prims.append({"e_lo": e_lo, "theta": theta, "class": "FAIL",
                          "sig": None})
            continue
        # own anatomy on the primary
        cands = prow["candidates"]
        hces = []
        for cd in cands:
            e = eval(cd["child_srepr"], sp.__dict__)
            hces.append(hce(e))
        lab = [j for j, cd in enumerate(cands) if cd["is_label"]]
        check(len(lab) == 1, f"label {i}")
        best = min(hces)
        ties = sum(1 for v in hces if v == best)
        check(hces[lab[0]] == best and ties == prow["min_hce_ties"],
              f"teacher/ties {i}")
        js, sid = sig_of(cands)
        A, B, expset = TARGET[e_lo]
        tups = [tup(cd) for cd in cands]
        unprod = sorted(cd["param_index"] for cd in cands
                        if cd["rule"] == "i_unprod")
        # role trace: A expression from child = child - integrals
        roles = {}
        for j, cd in enumerate(cands):
            if cd["rule"] != "i_unprod":
                continue
            e = eval(cd["child_srepr"], sp.__dict__)
            ints = [t for t in (e.args if isinstance(e, sp.Add) else [e])
                    if isinstance(t, sp.Integral)]
            Aex = e - sum(ints)
            if sp.expand(Aex - X**a * sp.sin(c * X)) == 0:
                roles[cd["param_index"]] = "SIN_CORRECT"
            elif sp.expand(Aex - X**b * sp.cos(c * X)) == 0:
                roles[cd["param_index"]] = "COS_CORRECT"
            else:
                roles[cd["param_index"]] = "WRONG_FAMILY"
        teacher = tups[lab[0]]
        if A not in tups or B not in tups:
            cls, gap = "TARGET-ABSENT", None
        elif teacher not in (A, B):
            cls, gap = "NON-TARGET-TEACHER", None
        elif ties != 1:
            cls, gap = "TIE", None
        else:
            other = B if teacher == A else A
            gap = round(hces[tups.index(other)] - best, 3)
            cls = ("STRICT-A" if teacher == A else "STRICT-B") \
                if gap >= 1.0 else "TIE"
        dp = drv_prims[i]
        check(dp["class"] == cls and dp["cand_sig_id"] == sid
              and dp["cand_sig"] == js
              and dp.get("gap_to_other_target") == gap
              and dp["unprod_set"] == unprod
              and {int(k): v for k, v in dp["roles_by_index"].items()}
              == roles, f"primary row {i} {cls} v {dp['class']}")
        prims.append({"e_lo": e_lo, "theta": theta, "class": cls,
                      "sig": sid, "gap": gap,
                      "b1": roles.get(A[4]) == "COS_CORRECT"
                      and roles.get(B[4]) == "SIN_CORRECT"
                      and unprod == expset,
                      "map_ok": cls == f"STRICT-{EXPECT[theta]}"
                      if cls.startswith("STRICT") else None})
    rec["requalified_parents"] = requal
    check(requal + sum(fails.values()) == 192, "192 parents")
    check(len(curs) == 192 and len(set(curs)) == 192, "192 unique curs")
    hits = {"natural": sum(x in nat for x in curs),
            "v1_census": sum(x in v1 for x in curs),
            "pilot": sum(x in pil for x in curs),
            "d3_artifacts": sum(x in art for x in curs),
            "cl1": sum(x in cl for x in curs),
            "fresh_K_present": sum("21*x**3" not in x for x in curs)}
    check(not any(hits.values()), f"overlap {hits}")
    check(drv_rec["burned_set_sizes"]["v1_census"] == 5760
          and not any(json.loads((IN / "overlap_receipt.json")
                                 .read_text())["hits"].values()),
          "driver overlap receipt")
    rec["overlap_hits"] = hits

    # ---- own bars --------------------------------------------------
    def bars(e_lo):
        P = [p for p in prims if p["e_lo"] == e_lo]
        cls = Counter(p["class"] for p in P)
        strict = [p for p in P if p["class"].startswith("STRICT")]
        g = defaultdict(lambda: Counter())
        for p in strict:
            g[p["sig"]][p["class"][-1]] += 1
        m = sorted((min(v["A"], v["B"]) for v in g.values()), reverse=True)
        mixed = {s for s, v in g.items() if min(v["A"], v["B"]) > 0}
        in_mixed = sum(1 for p in strict if p["sig"] in mixed)
        return {"n": len(P), "census": dict(cls),
                "by_theta": {t: dict(Counter(p["class"] for p in P
                                             if p["theta"] == t))
                             for t in ("SIN_LOW", "COS_LOW")},
                "B1": sum(1 for p in P if p.get("b1")),
                "B2": (cls["STRICT-A"], cls["STRICT-B"]),
                "map_violations": sum(1 for p in strict
                                      if p["map_ok"] is False),
                "gaps": dict(Counter(str(p["gap"]) for p in strict)),
                "n_sigs": len({p["sig"] for p in P if p["sig"]}),
                "n_mixed": len(mixed), "m": m, "sum_m": sum(m),
                "top2": sum(m[:2]),
                "B4": (in_mixed, len(strict)), "fails": cls["FAIL"],
                "B7": len(strict)}
    c2, c1 = bars(2), bars(1)
    passes = {"B1": c2["B1"] >= 29, "B2": c2["B2"][0] >= 12
              and c2["B2"][1] >= 12, "B3": c2["sum_m"] >= 12
              and c2["top2"] >= 12, "B4": c2["B4"][1] > 0
              and c2["B4"][0] >= 0.8 * c2["B4"][1],
              "B5": c2["fails"] == 0 and c2["n"] == 32,
              "B6": not any(hits.values()), "B7": c2["B7"] >= 24}
    if all(passes.values()):
        dec = "GO V2 PRODUCTION DESIGN"
    elif c2["B2"][0] < 6 or c2["B2"][1] < 6 or c2["sum_m"] < 6:
        dec = "PARK CONTROLLED CROSSOVER"
    else:
        dec = "NEEDS-REDESIGN"
    check(passes == drv_bars["passes"], "passes")
    check(dec == drv_bars["decision"] == drv_rec["verdict"], "decision")
    db = drv_bars["classifying_e_lo2"]
    check(db["B1"]["num"] == c2["B1"] and db["B2"]["strict_A"] == c2["B2"][0]
          and db["B2"]["strict_B"] == c2["B2"][1]
          and db["B3"]["sum_m_g"] == c2["sum_m"]
          and db["B3"]["n_mixed"] == c2["n_mixed"]
          and db["B4"]["num"] == c2["B4"][0]
          and db["B7"]["num"] == c2["B7"], "classifying numerators")
    dr = drv_bars["robustness_e_lo1"]
    check(dr["B2"]["strict_A"] == c1["B2"][0]
          and dr["B2"]["strict_B"] == c1["B2"][1]
          and dr["B3"]["sum_m_g"] == c1["sum_m"]
          and dr["B1"]["num"] == c1["B1"], "robustness numerators")
    rec.update({"classifying": c2, "robustness": c1, "passes": passes,
                "decision": dec, "problems": problems,
                "verdict": "VERIFIED " + dec if not problems
                else "DISCREPANCIES"})
    OUT.mkdir(parents=True)
    (OUT / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

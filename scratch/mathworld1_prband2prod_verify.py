"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-V2-PRODUCTION-MATERIALIZATION-0 —
INDEPENDENT VERIFIER. Reconstructs the frozen production design
(PRE-REG fd78a82b) from raw without importing the producer's
selection, ceiling, or decision code (shared rule engine,
qualify_parent and the codecs are allowed): own horizon and 72 pair
keys, own 432-parent burn census and uniqueness, own overlap checks
from the authoritative sources, full re-qualification of all 432
parents, own signature function, own HCE from child expressions, own
unique-teacher / gap / mapping gates, own pair eligibility, own
pair_sha selection, own 24-order theorems, own encoding roundtrips,
own companion mapping; then compares to the producer's per-row
artifacts. Writes logs/mathworld1/prband2prod_verify/
verify_receipt.json (refuse-if-exists), recording its own commit
and file sha.

    .venv/bin/python scratch/mathworld1_prband2prod_verify.py
"""
import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.search.derivation import UNSOLVED  # noqa: E402
from scratch.mathworld1_svpchal import (AFTER_D,  # noqa: E402
                                        SMALL_D, X, fsha,
                                        qualify_parent)
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        factor_symbols)
from scratch.mathworld1_svpforder import (INV, PERM,  # noqa: E402
                                          pf_decode, pf_encode)

IN = Path("logs/mathworld1/prband2prod")
OUT = Path("logs/mathworld1/prband2prod_verify")
DS = {"smallA": SMALL_D[0], "smallB": SMALL_D[1], "after": AFTER_D}
A0 = ("i_unprod", "I", 0, "term_index", 1)
B0 = ("i_unprod", "I", 0, "term_index", 3)
A1 = ("i_unprod", "I", 1, "term_index", 1)
B1 = ("i_unprod", "I", 1, "term_index", 3)
WIT = [["i_sum", "I", 0, "none", -1], ["i_unprod", "I", 0, "term_index", 1],
       ["i_unprod", "I", 0, "term_index", 3],
       ["i_unprod", "I", 0, "term_index", 5]]
CWIT = [["i_sum", "I", 1, "none", -1], ["i_unprod", "I", 1, "term_index", 1],
        ["i_unprod", "I", 1, "term_index", 3],
        ["i_unprod", "I", 1, "term_index", 5]]
MANIFEST = {
    "logs/mathworld1/prband2desk/bases.jsonl":
        "0fd7ac37b75a1dae4547afa3e8b3fd0fa6077d97cb169df34cfa306afb71e80c",
    "logs/mathworld1/prband/horizon_census.jsonl":
        "2e28abf09219d4ac5ee2cb834f22f9cbc40f750c13029a8b96979f9a7e144c4c",
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75d8402351d468e8",
    "logs/mathworld1/svpdiet/balanced_grid_train.jsonl":
        "0ef3d8a880a7e07712d8de757bc1670df12701e487b856b44c97f8db16cb3759",
    "logs/mathworld1/prband2desk_verify/support_matrix.json":
        "65b77230775f4f7b05e341f40220c7069b2996e5b7b1bf187b492cb202e02b0e",
}
problems = []


def check(cond, msg):
    if not cond:
        problems.append(msg)
        print("PROBLEM:", msg, flush=True)


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def tup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"], c["param_kind"],
            c["param_index"])


def sig_of(cands):
    t = sorted(tup(c) for c in cands)
    js = json.dumps([list(x) for x in t], sort_keys=True,
                    separators=(",", ":"))
    return js, h(js)


def hce(e):
    return 100.0 * len(e.atoms(*UNSOLVED)) + float(sp.count_ops(e)) + 0.1


def rows(p):
    return [json.loads(l) for l in open(p)]


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    rec = {"verifier_commit": subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True,
        text=True).stdout.strip(),
        "verifier_file_sha256": fsha(__file__),
        "input_sha256": {f.name: fsha(f) for f in sorted(IN.glob("*.json*"))}}
    for p, s in MANIFEST.items():
        check(fsha(p) == s, f"pin {p}")
    d_hz = rows(IN / "horizon.jsonl")
    d_par = rows(IN / "parents.jsonl")
    d_rows = {(r["index"], r["variant"]): r for r in rows(IN / "rows.jsonl")}
    d_pairs = rows(IN / "pairs.jsonl")
    d_rec = json.loads((IN / "prband2prod_receipt.json").read_text())
    gofire = d_rec["verdict"].startswith("QUALIFIED")
    d_pri = rows(IN / "primary.jsonl") if gofire else []
    d_comp = rows(IN / "companion.jsonl") if gofire else []
    d_perm = (json.loads((IN / "permutations.json").read_text())
              if gofire else None)

    # ---- own horizon ------------------------------------------------
    W = sp.sin(X) / X
    hz = []
    for e_hi in (5, 6, 7, 9):
        for theta in ("SIN_LOW", "COS_LOW"):
            a, b = (1, e_hi) if theta == "SIN_LOW" else (e_hi, 1)
            for c in (29, 31, 37, 41, 43, 47):
                for K in (33 * X**3, 27 * X**4, 35 * X**3):
                    f = (sp.expand(sp.diff(X**a * sp.sin(c * X), X))
                         + sp.expand(sp.diff(X**b * sp.cos(c * X), X))
                         + sp.Integral(W, X) + K)
                    hz.append((f"V2P|ehi={e_hi}|theta={theta}|c={c}|K="
                               f"{sp.sstr(K)}|w=sin(x)/x", f, theta,
                               f"ehi={e_hi}|c={c}|K={sp.sstr(K)}", a, b, c,
                               e_hi, sp.sstr(K)))
    check(len(hz) == 144 and len(d_hz) == 144, "144 bases")
    check(len({x[3] for x in hz}) == 72, "72 keys")
    check(len({x[0] for x in hz}) == 144
          and len({sp.srepr(x[1]) for x in hz}) == 144, "base uniqueness")
    for i, x in enumerate(hz):
        d = d_hz[i]
        check(d["base_signature"] == x[0] and d["pair_key"] == x[3]
              and d["pair_sha"] == h(x[3]) and d["f"] == sp.sstr(x[1])
              and d["n_terms"] == len(x[1].args) == 6, f"horizon {i}")
    parents = []
    for i, x in enumerate(hz):
        for vt, D in DS.items():
            parents.append(sp.sstr(sp.Add(sp.Integral(x[1], X),
                                          sp.Integral(D, X))))
    check(len(parents) == 432 and len(set(parents)) == 432, "432 unique")
    check([p["cur"] for p in d_par] == parents, "parents.jsonl order/content")

    # ---- own overlap (authoritative raw sources) ---------------------
    nat = {json.loads(l)["cur"] for l in open("data/matsub_paired.jsonl")}
    v1 = set()
    for l in open("logs/mathworld1/prband/horizon_census.jsonl"):
        for v in json.loads(l)["variants"].values():
            v1.add(v["cur"])
    v2 = set()
    for l in open("logs/mathworld1/prband2desk/bases.jsonl"):
        for v in json.loads(l)["variants"].values():
            v2.add(v["cur"])
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
    from scratch.mathworld1_svpdiet import PILOT_RECEIPTS  # noqa: E402
    pil = set()
    for pr in PILOT_RECEIPTS:
        for a2 in json.loads(Path(pr).read_text())["attempts"]:
            pil.add(a2["parent_sstr"])
    hits = {"natural": sum(p in nat for p in parents),
            "v1_census": sum(p in v1 for p in parents),
            "v2_desk": sum(p in v2 for p in parents),
            "d3_artifacts": sum(p in art for p in parents),
            "cl1": sum(p in cl for p in parents),
            "pilot": sum(p in pil for p in parents),
            "fresh_K_missing": sum(not any(k in p for k in
                                           ("33*x**3", "27*x**4", "35*x**3"))
                                   for p in parents)}
    check(len(v1) == 5760 and len(v2) == 192, "burn set sizes")
    check(not any(hits.values()), f"overlap {hits}")
    check(not any(d_rec["overlap_hits"].values()), "driver overlap")
    rec["overlap_hits"] = hits

    # ---- full re-qualification + own anatomy ---------------------------
    own = {}
    fails = Counter()
    for i, x in enumerate(hz):
        sig, f, theta, pk, a, b, c, e_hi, Ks = x
        for vt, D in DS.items():
            row, why = qualify_parent(f, D)
            d = d_rows[(i, vt)]
            if why:
                fails[f"{vt}:{why}"] += 1
                check(d["fail"] == why, f"fail {i} {vt}")
                own[(i, vt)] = {"fail": why}
                continue
            cands = row["candidates"]
            hs = [hce(eval(cd["child_srepr"], sp.__dict__)) for cd in cands]
            lab = [j for j, cd in enumerate(cands) if cd["is_label"]]
            best = min(hs)
            ties = sum(1 for v in hs if v == best)
            js, sid = sig_of(cands)
            tups = [tup(cd) for cd in cands]
            for cd in cands:
                t = tup(cd)
                fc = factor_symbols(*t)
                pf = pf_encode(t)
                check(fc == cd["factor_code"] and factor_decode(fc) == t
                      and pf_decode(pf) == t
                      and pf == [fc[PERM[k]] for k in range(8)]
                      and [pf[INV[k]] for k in range(8)] == fc
                      and len(fc) == 8, f"codes {i} {vt}")
            own[(i, vt)] = {"fail": None, "cur": row["cur"], "sid": sid,
                            "js": js, "tups": tups, "hs": hs,
                            "teacher": tups[lab[0]], "ties": ties,
                            "best": best, "site": tups[lab[0]][2]}
            check(d["fail"] is None and d["cur"] == row["cur"]
                  and d["cand_sig_id"] == sid and d["cand_sig"] == js
                  and tuple(d["teacher"]) == tups[lab[0]]
                  and d["min_hce_ties"] == ties
                  and [tuple(t) for t in d["cand_tuples"]] == tups
                  and all(abs(cd["hce"] - hv) < 1e-9
                          for cd, hv in zip(d["candidates"], hs)),
                  f"row {i} {vt}")
    rec["requalified"] = 432 - sum(fails.values())
    rec["fail_census"] = dict(fails)

    # ---- own pair eligibility -----------------------------------------
    def arm_ok(o, TA, TB, wit, expect):
        if o["fail"]:
            return False, None
        if TA not in o["tups"] or TB not in o["tups"]:
            return False, None
        if o["ties"] != 1:
            return False, None
        side = "A" if o["teacher"] == TA else "B" if o["teacher"] == TB \
            else None
        if side != expect:
            return False, None
        hA = o["hs"][o["tups"].index(TA)]
        hB = o["hs"][o["tups"].index(TB)]
        gap = round(abs(hA - hB), 3)
        if gap != 3.0 or min(hA, hB) != o["best"]:
            return False, gap
        if json.loads(o["js"]) != wit:
            return False, gap
        return True, gap
    keys = defaultdict(dict)
    for i, x in enumerate(hz):
        keys[x[3]][x[2]] = i
    elig = {}
    comp_av = {}
    for pk, arms in keys.items():
        block = all(own[(arms[t], vt)]["fail"] is None
                    for t in ("SIN_LOW", "COS_LOW") for vt in DS)
        ok = block
        gaps = {}
        for t, ex in (("SIN_LOW", "A"), ("COS_LOW", "B")):
            o = own[(arms[t], "after")]
            a_ok, g = arm_ok(o, A0, B0, WIT, ex) if block else (False, None)
            ok = ok and a_ok
            gaps[t] = g
        if block:
            ok = ok and (own[(arms["SIN_LOW"], "after")]["sid"]
                         == own[(arms["COS_LOW"], "after")]["sid"])
        elig[pk] = ok
        cok = block
        for t, ex in (("SIN_LOW", "A"), ("COS_LOW", "B")):
            o = own[(arms[t], "smallA")]
            c_ok, _ = arm_ok(o, A1, B1, CWIT, ex) if block else (False, None)
            cok = cok and c_ok
        comp_av[pk] = cok
    dp = {p["pair_key"]: p for p in d_pairs}
    check(len(dp) == 72, "72 pair rows")
    for pk in keys:
        check(dp[pk]["eligible"] == elig[pk], f"eligibility {pk}")
        check(dp[pk].get("companion_available") == comp_av[pk],
              f"companion {pk}")
    n_el = sum(elig.values())
    ordering = sorted((pk for pk in keys if elig[pk]), key=lambda k: h(k))
    sel = ordering[:48]
    reserve = ("FULL-RESERVE" if n_el == 72 else "PARTIAL-RESERVE"
               if n_el >= 48 else "NO-FIRE")
    check(d_rec["n_pairs_eligible"] == n_el
          and d_rec["reserve_class"] == reserve, "capacity")
    check(d_rec["eligible_ordering"] == [h(k) for k in ordering], "ordering")
    rec.update({"n_pairs_eligible": n_el, "reserve": reserve,
                "n_companion_available": sum(comp_av.values())})
    if n_el < 48:
        check(d_rec["verdict"] == "NO-FIRE", "verdict NO-FIRE")
        rec["verdict"] = "VERIFIED NO-FIRE" if not problems else "DISCREPANCIES"
        rec["problems"] = problems
        OUT.mkdir(parents=True)
        (OUT / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1))
        return 0 if not problems else 1
    check(d_rec["selected_pair_keys"] == sel, "selected 48")

    # ---- own primary + theorems ----------------------------------------
    prim = []
    for pk in sel:
        for t in ("SIN_LOW", "COS_LOW"):
            o = own[(keys[pk][t], "after")]
            prim.append({"pk": pk, "theta": t, "gold": A0 if t == "SIN_LOW"
                         else B0, "teacher": o["teacher"], "sid": o["sid"],
                         "cur": o["cur"]})
    check(len(prim) == 96 and len(d_pri) == 96, "N 96")
    check(Counter(p["gold"] for p in prim) == Counter({A0: 48, B0: 48}),
          "48/48")
    check(all(p["teacher"] == p["gold"] for p in prim), "teacher = gold")
    check(len({p["sid"] for p in prim}) == 1, "one signature")
    check([p["cur"] for p in prim] == [d["cur"] for d in d_pri],
          "primary rows match")
    codes = [tuple(t) for t in WIT]
    best_top, best_both, census = 0, 0, Counter()
    for order in itertools.permutations(codes):
        top = order[0]
        corr = [p["gold"] == top for p in prim]
        both = sum(corr[k] and corr[k + 1] for k in range(0, 96, 2))
        best_top = max(best_top, sum(corr))
        best_both = max(best_both, both)
        census[sum(corr)] += 1
    check(best_top == 48 and best_both == 0, f"theorems {best_top} {best_both}")
    check(d_perm["max_top1"] == 48 and d_perm["max_both_correct_pairs"] == 0
          and d_perm["n_orders"] == 24, "driver permutations")
    comp = [d for d in d_comp]
    check(all(comp_av[d["pair_key"]] for d in comp)
          and len(comp) == 2 * sum(comp_av[pk] for pk in sel), "companion set")
    for d in comp:
        o = own[(keys[d["pair_key"]][d["theta"]], "smallA")]
        check(o["teacher"] == (A1 if d["theta"] == "SIN_LOW" else B1)
              and json.loads(o["js"]) == CWIT, f"companion row {d['pair_id']}")
    rec.update({"top1_census": {str(k): v for k, v in census.items()},
                "max_top1": best_top, "max_both_correct": best_both,
                "n_primary": len(prim), "n_companion": len(comp),
                "problems": problems,
                "verdict": ("VERIFIED QUALIFIED" if not problems
                            else "DISCREPANCIES")})
    OUT.mkdir(parents=True)
    (OUT / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "input_sha256"},
                     indent=1))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

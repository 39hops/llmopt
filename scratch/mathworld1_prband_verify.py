"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-MATERIALIZATION-0 — INDEPENDENT
VERIFIER. Reconstructs the frozen design (PRE-REG ...-DESIGN-
PREREG-0, cb098ee5) from the RAW materialization artifacts under
logs/mathworld1/prband/ WITHOUT reusing the selector's derived
tables: own horizon enumeration, own signature function, own
bucket / capacity / walk / selection, own ceiling enumeration,
own overlap sets, and a full re-qualification (qualify_parent)
of every selected primary and companion parent. Zero checkpoint
access, zero scoring. Writes logs/mathworld1/prband_verify/
verify_receipt.json (refuse-if-exists). PRBAND_VERIFY_DIR
overrides the input directory (smoke self-test); PRBAND_SMOKE=1
makes the smoke-only deviations report-only.

    .venv/bin/python scratch/mathworld1_prband_verify.py
"""
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpchal import (AFTER_D,  # noqa: E402
                                        SMALL_D, X, fsha,
                                        qualify_parent)
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        factor_symbols)
from scratch.mathworld1_svpdiet3 import P12, WS  # noqa: E402
from scratch.mathworld1_svpforder import (INV, PERM,  # noqa: E402
                                          pf_decode, pf_encode)

SMOKE = os.environ.get("PRBAND_SMOKE") == "1"
IN = Path(os.environ.get("PRBAND_VERIFY_DIR",
                         "logs/mathworld1/prband"))
OUT = Path("logs/mathworld1/prband_verify"
           + ("_smoke" if SMOKE else ""))
TOK = ActionGCTok()
C_PR = (8, 9, 10, 11, 12, 15, 16, 17, 18, 19)
K_A = (9 * X**3, 12 * X**4)
K_B = (13 * X**2, 15 * X**5)
ADDEND = (("A", 1), ("A", 2), ("B", 1), ("B", 2))
GOLD_A = ("i_unprod", "I", 1, "term_index", 2)
GOLD_B = ("i_unprod", "I", 1, "term_index", 3)
DS = {"smallA": SMALL_D[0], "smallB": SMALL_D[1], "after": AFTER_D}
N_HALF = 1 if SMOKE else 48

problems = []


def check(cond, msg):
    if not cond:
        problems.append(msg)
        print("PROBLEM:", msg, flush=True)


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sig_of(cands):
    """Independent signature: 5-tuples from the candidate rows,
    sorted, canonical JSON, sha256."""
    tups = sorted((c["rule"], c["site_kind"], c["site_ordinal"],
                   c["param_kind"], c["param_index"])
                  for c in cands)
    js = json.dumps([list(t) for t in tups], sort_keys=True,
                    separators=(",", ":"))
    return js, h(js)


def rows(p):
    return [json.loads(l) for l in open(p)]


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    rec = {"input_dir": str(IN), "smoke": SMOKE,
           "input_sha256": {f.name: fsha(f) for f in sorted(
               IN.glob("*.json*"))}}
    census = rows(IN / "horizon_census.jsonl")
    qual = rows(IN / "qualified_blocks.jsonl")
    cap_tab = json.loads((IN / "capacity_table.json").read_text())
    drv = json.loads((IN / "prband_receipt.json").read_text())
    nofire = drv["verdict"] == "NO-FIRE"
    check(nofire == (not (IN / "primary.jsonl").exists()),
          "NO-FIRE verdict v primary artifact presence")
    primary = [] if nofire else rows(IN / "primary.jsonl")
    companion = [] if nofire else rows(IN / "companion.jsonl")
    ceil_rec = (None if nofire else
                json.loads((IN / "ceiling_receipt.json").read_text()))

    # ---- 1. horizon re-enumeration (own loops) ----------------
    if not SMOKE:
        hz = []
        for P in P12:
            for T in (sp.sin, sp.cos):
                for c in C_PR:
                    for w in WS:
                        for kb, k in ADDEND:
                            K = K_A if kb == "A" else K_B
                            f = (sp.expand(sp.diff(P * T(c * X), X))
                                 + sp.Integral(w, X)
                                 + sp.Add(*K[:k]))
                            hz.append((f"PR|P={sp.sstr(P)}|T="
                                       f"{T.__name__}|c={c}|w="
                                       f"{sp.sstr(w)}|kb={kb}|k={k}",
                                       f))
        check(len(hz) == 1920, f"horizon {len(hz)}")
        check(len({s for s, _ in hz}) == 1920, "base_signature dup")
        check(len({h(s) for s, _ in hz}) == 1920, "sig_sha dup")
        check(len({sp.srepr(f) for _, f in hz}) == 1920,
              "base srepr dup")
        burn = {sp.sstr(sp.Add(sp.Integral(f, X), sp.Integral(D, X)))
                for _, f in hz for D in DS.values()}
        check(len(burn) == 5760, f"burn set {len(burn)}")
        check(len(census) == 1920, f"census rows {len(census)}")
        for i, (s, f) in enumerate(hz):
            r = census[i]
            check(r["horizon_index"] == i and r["base_signature"] == s
                  and r["sig_sha"] == h(s)
                  and r["target_integrand"] == sp.sstr(f),
                  f"census row {i} mismatch")
        rec["horizon"] = {"n": len(hz), "burned": len(burn)}
    fmap = {r["horizon_index"]: r for r in census}

    # ---- 2. block law + parity from raw census -----------------
    n_pass = 0
    fails = Counter()
    for r in census:
        i = r["horizon_index"]
        check(r["primary_variant"] == ("smallA" if i % 2 == 0
                                       else "smallB"),
              f"parity {i}")
        if r["fail"] is None:
            n_pass += 1
            vs = r["variants"]
            check(set(vs) == {"smallA", "smallB", "after"},
                  f"variants {i}")
            golds = {t: tuple(vs[t]["gold_tuple"]) for t in vs}
            check(all(g[0] == "i_unprod" and g[1] == "I"
                      and g[3] == "term_index" for g in golds.values()),
                  f"teacher rule/kind {i}")
            check(len({g[4] for g in golds.values()}) == 1,
                  f"common term {i}")
            check([golds["smallA"][2], golds["smallB"][2],
                   golds["after"][2]] == [1, 1, 0], f"ordinals {i}")
            check(r["term"] == golds["after"][4], f"term field {i}")
        else:
            fails[r["fail"].split(":")[0]] += 1
    check(n_pass == len(qual) == drv["n_qualifying_blocks"],
          f"qualifying count {n_pass}/{len(qual)}")
    rec["blocks"] = {"pass": n_pass, "fail_census": dict(fails)}

    # ---- 3. signatures + gold from raw candidate lists ---------
    qmap = {q["horizon_index"]: q for q in qual}
    sig_ok = 0
    for q in qual:
        r = fmap[q["horizon_index"]]
        for vt, v in q["variants"].items():
            js, sid = sig_of(v["candidates"])
            gold = (v["chosen_rule"], v["chosen_site_kind"],
                    v["chosen_ordinal"], v["chosen_param_kind"],
                    v["chosen_term"])
            lab = [c for c in v["candidates"] if c["is_label"]]
            check(len(lab) == 1 and (lab[0]["rule"], lab[0]["site_kind"],
                                     lab[0]["site_ordinal"],
                                     lab[0]["param_kind"],
                                     lab[0]["param_index"]) == gold,
                  f"label {q['horizon_index']} {vt}")
            e = r["variants"][vt]
            ok = (e["cand_sig_id"] == sid and e["cand_sig"] == js
                  and tuple(e["gold_tuple"]) == gold
                  and e["cur"] == v["cur"]
                  and e["n_candidates"] == len(v["candidates"]))
            check(ok, f"sig/gold {q['horizon_index']} {vt}")
            sig_ok += ok
    rec["signatures_rechecked"] = sig_ok

    # ---- 4. buckets, capacities, walk, selection (own) --------
    buckets = defaultdict(list)
    for q in qual:
        r = fmap[q["horizon_index"]]
        pv = r["primary_variant"]
        v = q["variants"][pv]
        _, sid = sig_of(v["candidates"])
        gold = (v["chosen_rule"], v["chosen_site_kind"],
                v["chosen_ordinal"], v["chosen_param_kind"],
                v["chosen_term"])
        gc = "a" if gold == GOLD_A else "b" if gold == GOLD_B else None
        if gc:
            buckets[(sid, gc)].append((r["sig_sha"], v["cur"]))
    for k in buckets:
        buckets[k].sort()
    sids = sorted({k[0] for k in buckets})
    m = {g: min(len(buckets.get((g, "a"), [])),
                len(buckets.get((g, "b"), []))) for g in sids}
    for g in sids:
        ct = cap_tab["capacity_table"].get(g)
        check(ct is not None and ct["n_a"] == len(buckets.get((g, "a"), []))
              and ct["n_b"] == len(buckets.get((g, "b"), []))
              and ct["m_g"] == m[g], f"capacity {g[:12]}")
    check(set(cap_tab["capacity_table"]) == set(sids),
          "capacity table signature set")
    usable = sorted([g for g in sids if m[g] > 0],
                    key=lambda g: (-m[g], g))
    check(usable == cap_tab["usable_order"], "usable order")
    walk, filled, sel = [], 0, set()
    for g in usable:
        if filled >= N_HALF or len(walk) >= 4:
            break
        q = min(m[g], N_HALF - filled)
        walk.append({"cand_sig_id": g, "m_g": m[g], "q_g": q})
        for gc in ("a", "b"):
            for _, cur in buckets[(g, gc)][:q]:
                sel.add(cur)
        filled += q
    check(walk == cap_tab["walk"], "walk")
    check(filled == cap_tab["filled"], "filled")
    rec["selection"] = {"n_signatures": len(sids),
                        "n_usable": len(usable), "sum_m_g": sum(
                            m.values()), "walk": walk,
                        "filled": filled,
                        "supply_a": sum(len(buckets.get((g, "a"), []))
                                        for g in sids),
                        "supply_b": sum(len(buckets.get((g, "b"), []))
                                        for g in sids)}
    materialized = filled == N_HALF
    if not SMOKE:
        check(materialized == drv["verdict"].startswith(
            "PRIOR-RESISTANT POPULATION MATERIALIZED"),
            "verdict v fill")
    if not materialized and not SMOKE:
        check(nofire, "fill < N but driver did not book NO-FIRE")
        rec["verdict"] = "VERIFIED NO-FIRE" if not problems \
            else "DISCREPANCIES"
        rec["problems"] = problems
        OUT.mkdir(parents=True)
        (OUT / "verify_receipt.json").write_text(
            json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1))
        return 0 if not problems else 1
    if not SMOKE:
        check({r["cur"] for r in primary} == sel,
              "selected primary set != own reconstruction")

    # ---- 5. selected artifacts: balance, uniqueness, encoding --
    for name, rs in (("primary", primary), ("companion", companion)):
        check(len(rs) == 2 * N_HALF, f"{name} N {len(rs)}")
        check(len({r["cur"] for r in rs}) == len(rs), f"{name} dup cur")
        check(len({h(r['cur']) for r in rs}) == len(rs)
              and all(h(r["cur"]) == r["cur_sha"] for r in rs),
              f"{name} cur sha")
        check(len({r["parent_srepr_sha"] for r in rs}) == len(rs),
              f"{name} dup parent")
        gc = Counter(r["gold_class"] for r in rs)
        if not SMOKE:
            check(gc == Counter({"a": N_HALF, "b": N_HALF})
                  if name == "primary" else True, f"{name} gold {gc}")
    check(not ({r["cur"] for r in primary}
               & {r["cur"] for r in companion}), "primary/companion")
    groups = defaultdict(Counter)
    for r in primary:
        js, sid = sig_of(r["candidates"])
        check(sid == r["cand_sig_id"] and js == r["cand_sig"],
              f"primary sig {r['block_id']}")
        groups[sid][r["gold_class"]] += 1
    if not SMOKE:
        check(len(groups) <= 4, f"signatures {len(groups)}")
        for g, cnt in groups.items():
            check(cnt["a"] == cnt["b"] and cnt["a"] > 0
                  and set(cnt) == {"a", "b"}, f"balance {g[:12]} {cnt}")
    max_tot, max_pre = 0, 0
    for r in primary + companion:
        pre = len(TOK.encode(f"Current: {r['cur']}\nHints: none\n"
                             f"Step: "))
        check(pre == r["prompt_tokens"], f"prompt len {r['block_id']}")
        max_pre = max(max_pre, pre)
        max_tot = max(max_tot, pre + 9)
        for c in r["candidates"]:
            tup = (c["rule"], c["site_kind"], c["site_ordinal"],
                   c["param_kind"], c["param_index"])
            fc = factor_symbols(*tup)
            pf = pf_encode(tup)
            check(fc == c["factor_code"] and factor_decode(fc) == tup
                  and pf == c["_pf"] and pf_decode(pf) == tup
                  and pf == [fc[PERM[i]] for i in range(8)]
                  and [pf[INV[j]] for j in range(8)] == fc
                  and len(fc) == 8, f"code {r['block_id']}")
    check(max_tot <= 4096, "context overflow")
    rec["encoding"] = {"max_prompt_tokens": max_pre,
                       "max_total_tokens": max_tot,
                       "overflow": int(max_tot > 4096)}

    # ---- 6. full re-qualification of selected parents ----------
    requal = 0
    for r in primary + companion:
        f = sp.sympify(r["target_integrand"])
        D = DS[r["variant"]]
        row, why = qualify_parent(f, D)
        ok = (why is None and row["cur"] == r["cur"]
              and row["parent_srepr_sha"] == r["parent_srepr_sha"]
              and [(c["rule"], c["site_kind"], c["site_ordinal"],
                    c["param_kind"], c["param_index"], c["is_label"],
                    c["factor_code"]) for c in row["candidates"]]
              == [(c["rule"], c["site_kind"], c["site_ordinal"],
                   c["param_kind"], c["param_index"], c["is_label"],
                   c["factor_code"]) for c in r["candidates"]]
              and (row["chosen_rule"], row["chosen_site_kind"],
                   row["chosen_ordinal"], row["chosen_param_kind"],
                   row["chosen_term"]) == tuple(r["gold_tuple"]))
        # sympify(sstr) may not be identity for every expression;
        # the srepr sha equality above is the binding check.
        check(ok, f"requalify {r['block_id']} {why}")
        requal += ok
    rec["requalified"] = requal

    # ---- 7. ceilings (own enumeration) --------------------------
    def ceilings(rs):
        grp = defaultdict(list)
        for r in rs:
            grp[r["cand_sig_id"]].append(r)
        per, micro = {}, 0
        for g, lst in grp.items():
            codes = {json.dumps(t) for t in lst[0]["cand_tuples"]}
            for r in lst:
                check({json.dumps(t) for t in r["cand_tuples"]} == codes,
                      f"group candidate identity {g[:12]}")
            check(len(codes) <= 9, f"group >9 codes {g[:12]}")
            gcount = Counter(json.dumps(r["gold_tuple"]) for r in lst)
            # best fixed order picks the most frequent gold first
            best = max(gcount.values())
            # count orders attaining best: any order whose top code
            # (within the group) is a best gold
            n_best_golds = sum(1 for v in gcount.values() if v == best)
            n_opt = n_best_golds * math.factorial(len(codes) - 1)
            per[g] = {"n": len(lst), "best": best, "n_optimal": n_opt}
            micro += best
        macro = sum(p["best"] / p["n"] for p in per.values()) / len(per)
        union = sorted({json.dumps(t) for r in rs
                        for t in r["cand_tuples"]})
        glob = {"enumerated": False, "n_union_codes": len(union)}
        if len(union) <= 9:
            idx = {c: i for i, c in enumerate(union)}
            st = [([idx[json.dumps(t)] for t in r["cand_tuples"]],
                   idx[json.dumps(r["gold_tuple"])]) for r in rs]
            best, nb = -1, 0
            for perm in itertools.permutations(range(len(union))):
                rank = [0] * len(union)
                for k, c in enumerate(perm):
                    rank[c] = k
                acc = sum(rank[g] < min(rank[c] for c in cs if c != g)
                          for cs, g in st)
                if acc > best:
                    best, nb = acc, 1
                elif acc == best:
                    nb += 1
            glob = {"enumerated": True, "n_union_codes": len(union),
                    "best": best, "n_optimal_orders": nb}
        return {"n_groups": len(per), "per_group": per,
                "micro_best": micro, "micro_ceiling": micro / len(rs),
                "macro_ceiling": macro, "global": glob,
                "balanced": all(
                    len(Counter(json.dumps(r["gold_tuple"])
                                for r in lst)) == 2
                    and len(set(Counter(json.dumps(r["gold_tuple"])
                                        for r in lst).values())) == 1
                    for lst in grp.values())}
    cp = ceilings(primary)
    cc = ceilings(companion)
    rec["primary_ceiling"] = cp
    rec["companion_ceiling"] = cc
    if not SMOKE:
        check(cp["micro_best"] * 2 == len(primary)
              and abs(cp["macro_ceiling"] - 0.5) < 1e-12
              and (not cp["global"]["enumerated"]
                   or cp["global"]["best"] * 2 == len(primary)),
              "primary ceiling != 50%")
        dp = ceil_rec["primary"]
        check(dp["micro_best"] == cp["micro_best"]
              and abs(dp["macro_ceiling"] - cp["macro_ceiling"]) < 1e-12
              and dp["global"].get("best") == cp["global"].get("best")
              and dp["global"].get("n_optimal_orders")
              == cp["global"].get("n_optimal_orders"),
              "driver ceiling receipt differs")
        for g, p in cp["per_group"].items():
            check(p["best"] * 2 == p["n"], f"group ceiling {g[:12]}")
    comp_verdict = ("COMPANION BALANCED" if cc["balanced"]
                    and cc["micro_best"] * 2 == len(companion)
                    and abs(cc["macro_ceiling"] - 0.5) < 1e-12
                    and (not cc["global"]["enumerated"]
                         or cc["global"]["best"] * 2 == len(companion))
                    else "COMPANION IMBALANCED")
    rec["companion_verdict"] = comp_verdict
    check(comp_verdict == drv["companion_verdict"], "companion verdict")

    # ---- 8. overlap gates (own sets, lineage builders) ---------
    if not SMOKE:
        from scratch.mathworld1_prband import (  # noqa: E402
            AUG, CL1, D3, PAIRED, PILOT_RECEIPTS, RECORDED)
        hits = {}
        curs = [r["cur"] for r in primary + companion]
        nat = {json.loads(l)["cur"] for l in open(PAIRED)}
        hits["natural"] = sum(c in nat for c in curs)
        band = set()
        for bf in ("logs/mathworld1/svpeval/decisions.jsonl",
                   "logs/mathworld1/svpeval2/decisions.jsonl",
                   "logs/mathworld1/svpeval3/decisions.jsonl"):
            for l in open(bf):
                x = json.loads(l)
                if x.get("cur"):
                    band.add(x["cur"])
        hits["band"] = sum(c in curs for c in band)
        pil = set()
        for pr in PILOT_RECEIPTS:
            for a2 in json.loads(Path(pr).read_text())["attempts"]:
                pil.add(a2["parent_sstr"])
        hits["pilot"] = sum(c in pil for c in curs)
        art = set()
        for fn in ("heldout_test16.jsonl", "covered_calibration.jsonl",
                   "pout_robustness.jsonl"):
            for l in open(f"{D3}/{fn}"):
                art.add(json.loads(l)["cur"])
        hits["d3_artifacts"] = sum(c in art for c in curs)
        cl = set()
        for l in open(f"{CL1}/manifest.jsonl"):
            cl.add(json.loads(l)["root_cur"])
        for l in open(f"{CL1}/raw_attempts.jsonl"):
            x = json.loads(l)
            if x.get("cur"):
                cl.add(x["cur"])
        hits["cl1"] = sum(c in cl for c in curs)
        sm = set()
        for fn in ("primary.jsonl", "companion.jsonl"):
            p = Path("logs/mathworld1/prband_smoke") / fn
            if p.exists():
                sm |= {json.loads(l)["cur"] for l in open(p)}
        hits["own_smoke"] = sum(c in sm for c in curs)
        # structural: every selected cur carries a fresh K poly
        # (pure-polynomial part of the base is the addend sum)
        fresh = {sp.sstr(k) for k in K_A + K_B}
        hits["missing_fresh_addend"] = sum(
            not any(k in r["target_integrand"] for k in fresh)
            for r in primary)
        tt = {json.loads(l)["target_integrand"] for l in
              open("logs/mathworld1/svpdiet/train_blocks.jsonl")}
        hits["train_targets"] = sum(r["target_integrand"] in tt
                                    for r in primary)
        rec["overlap_hits"] = hits
        check(not any(hits.values()), f"overlap {hits}")
        check(drv["overlap_hits"]["horizons"] == 0
              and drv["overlap_hits"]["d3_targets"] == 0,
              "driver horizon/d3 overlap nonzero")
        rec["pins_rechecked"] = {p: fsha(p) == drv["pins"][p]
                                 for p in [PAIRED, AUG] + RECORDED}
        check(all(rec["pins_rechecked"].values()), "pins drifted")

    rec["problems"] = problems
    rec["verdict"] = ("VERIFIED MATERIALIZED" if not problems
                      else "DISCREPANCIES")
    OUT.mkdir(parents=True)
    (OUT / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("primary_ceiling",
                                   "companion_ceiling")}, indent=1))
    print("PRIMARY", {k: v for k, v in cp.items() if k != "per_group"})
    print("COMPANION", {k: v for k, v in cc.items()
                        if k != "per_group"})
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

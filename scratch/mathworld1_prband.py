"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-MATERIALIZATION-0 — materialize,
or mechanically NO-FIRE, the fresh N=96 prior-resistant ranking band
frozen by PRE-REG MATH-CYBER-1-PRIOR-RESISTANT-EVAL-DESIGN-PREREG-0
(cb098ee5, RESULTS L62655). ZERO checkpoint access, ZERO scoring,
ZERO masks, ZERO training; every law below is the prereg's.

HORIZON: CH-F3 base f = expand(d/dx[P*T(c x)]) + Integral(w, x)
+ Add(*K[:k]) over P in P12, T in {sin, cos}, c in C_PR
(in-support only), w in WS, addend config in the ORDERED fresh
namespace ((A,1), (A,2), (B,1), (B,2)), K_A = (9x^3, 12x^4),
K_B = (13x^2, 15x^5) = 1,920 bases; base_signature
"PR|P=|T=|c=|w=|kb=|k=" unique, base srepr unique (hard gates).
Every base is qualified in the three DESIGN-1 variants (smallA
x**x, smallB 1/(x+log x), after sin(sin x)) under qualify_parent
and the block law (all three pass, teacher i_unprod (I,
term_index), one common term, ordinals [1,1,0]). The whole
horizon (5,760 parents) is burned at run start.

PRIMARY I1 = smallA when the horizon index is even, smallB when
odd (target-blind). Candidate signature = sorted complete list
of (rule, site_kind, site_ordinal, param_kind, param_index),
canonical JSON, sha256 -> cand_sig_id. Gold pair a = (i_unprod,
I, 1, term_index, 2), b = (..., 3); other golds are censused
out-of-scope. Buckets (cand_sig_id, gold) ordered by sig_sha;
m_g = min(n_a, n_b); m_g = 0 excluded; walk by (-m_g,
cand_sig_id); q_g = min(m_g, 48 - filled); stop at 48 or four
signatures. FILL GATE: MATERIALIZED iff filled == 48, else
NO-FIRE. Ceiling gate: exhaustive per-group and (union <= 9
codes) global permutation census must give exactly 48/96 micro
and 50% macro, else INSTRUMENT FAILURE. Companion I0 = the after
variants of the selected blocks, own signatures/balance/ceiling,
BALANCED or IMBALANCED, never touching the primary verdict.
Overlap fences refute-on-hit against every burned lineage set.

Outputs (production, refuse-if-exists) under
logs/mathworld1/prband/: horizon_census.jsonl (one row per
base, all three variants summarized), qualified_blocks.jsonl
(full candidate lists, all three variants, every passing block),
capacity_table.json, primary.jsonl, companion.jsonl,
ceiling_receipt.json, overlap_receipt.json, prband_receipt.json.
SMOKE (PRBAND_SMOKE=1) uses the burned svpdiet3 smoke slice and
writes ONLY under logs/mathworld1/prband_smoke/.

    PRBAND_SMOKE=1 .venv/bin/python scratch/mathworld1_prband.py
    .venv/bin/python scratch/mathworld1_prband.py            (Mac)
"""
import hashlib
import itertools
import math
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import (AFTER_D,  # noqa: E402
                                        SMALL_D, X, fsha,
                                        qualify_parent)
from scratch.mathworld1_svpchal import \
    build_horizon as d0_horizon  # noqa: E402
from scratch.mathworld1_svpchal2 import \
    build_horizon1 as d1_horizon  # noqa: E402
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        factor_symbols)
from scratch.mathworld1_svpdiet import (F3_EVAL_C,  # noqa: E402
                                        F3_EVAL_P,
                                        F3_EVAL_POLYS,
                                        F3_TRAIN_C, F3_TRAIN_P,
                                        F3_TRAIN_POLYS,
                                        F4_EVAL_FREQS,
                                        F4_EVAL_P1,
                                        F4_TRAIN_FREQS,
                                        F4_TRAIN_P1,
                                        PILOT_RECEIPTS,
                                        f3_bases, f4_bases)
from scratch.mathworld1_svpdiet2 import (E1_F3_C,  # noqa: E402
                                         E1_F3_P, E1_F3_POLYS,
                                         E1_F4_FREQS,
                                         E1_F4_P1)
from scratch.mathworld1_svpdiet3 import (C_IN2,  # noqa: E402
                                         C_OUT2, C_SEC,
                                         K_POLYS2, P12, P_SEC,
                                         WS, build_f3,
                                         smoke_slice)
from scratch.mathworld1_svpforder import (PERM,  # noqa: E402
                                          pf_decode, pf_encode)
from scratch.mathworld1_svpnuisdesk import (C_IN as N_C_IN,  # noqa: E402
                                            C_OUT as N_C_OUT,
                                            K_POLYS as N_KP,
                                            P_IN as N_P_IN,
                                            P_OUT as N_P_OUT,
                                            build_cell)

SMOKE = os.environ.get("PRBAND_SMOKE") == "1"
PREREG = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-DESIGN-PREREG-0"
PREREG_COMMIT = "cb098ee584eb2c4c0ee574c692e7f72b028c7182"
OUTDIR = Path("logs/mathworld1/prband_smoke" if SMOKE
              else "logs/mathworld1/prband")
SMOKE_RECEIPT = Path("logs/mathworld1/prband_smoke/"
                     "prband_receipt.json")
N_HALF = 1 if SMOKE else 48
K_MAX = 4
CTX = 4096
TOK = ActionGCTok()
PAIRED = "data/matsub_paired.jsonl"
AUG = "logs/mathworld1/svpdiet/balanced_grid_train.jsonl"
D3 = "logs/mathworld1/svpdiet3"
CL1 = "logs/mathworld1/cl1/pop"
# receipt-derived pins (prereg); every other input records its
# sha at run time and the verifier re-derives it.
PINS = {
    PAIRED: "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
            "d8402351d468e8",
    AUG: "0ef3d8a880a7e07712d8de757bc1670df12701e487b856b44c97"
         "f8db16cb3759",
    f"{D3}/heldout_test16.jsonl":
        "a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec13881b4"
        "509df46ddb",
    f"{D3}/covered_calibration.jsonl":
        "af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b08509794d"
        "0fb7978db3",
}
RECORDED = [f"{D3}/eval_blocks.jsonl", f"{D3}/pout_attempts.jsonl",
            f"{D3}/pout_robustness.jsonl",
            f"{D3}/svpdiet3_receipt.json",
            f"{CL1}/manifest.jsonl", f"{CL1}/raw_attempts.jsonl",
            "logs/mathworld1/svpdiet/train_blocks.jsonl"]

# ---- frozen banks (prereg) ----------------------------------
C_PR = (8, 9, 10, 11, 12, 15, 16, 17, 18, 19)
K_A = (9 * X**3, 12 * X**4)
K_B = (13 * X**2, 15 * X**5)
ADDEND = (("A", 1), ("A", 2), ("B", 1), ("B", 2))
GOLD_A = ("i_unprod", "I", 1, "term_index", 2)
GOLD_B = ("i_unprod", "I", 1, "term_index", 3)
VARIANTS = (("smallA", SMALL_D[0]), ("smallB", SMALL_D[1]),
            ("after", AFTER_D))


def build_pr():
    out = []
    for P in P12:
        for T in (sp.sin, sp.cos):
            for c in C_PR:
                for w in WS:
                    for kb, k in ADDEND:
                        K = K_A if kb == "A" else K_B
                        f = (sp.expand(sp.diff(P * T(c * X), X))
                             + sp.Integral(w, X)
                             + sp.Add(*K[:k]))
                        sig = (f"PR|P={sp.sstr(P)}|T={T.__name__}"
                               f"|c={c}|w={sp.sstr(w)}|kb={kb}"
                               f"|k={k}")
                        out.append((sig, f, sp.sstr(P), c))
    return out


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def cand_sig(cands):
    tups = sorted(ctup(c) for c in cands)
    js = json.dumps([list(t) for t in tups], sort_keys=True,
                    separators=(",", ":"))
    return js, hashlib.sha256(js.encode()).hexdigest()


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def field(r, name):
    """Value of |name=...| in base_signature, 'n/a' if absent."""
    for part in r["base_signature"].split("|"):
        if part.startswith(name + "="):
            return part[len(name) + 1:]
    return "n/a"


def gold_class(t):
    return "a" if t == GOLD_A else ("b" if t == GOLD_B else "other")


def group_ceiling(states):
    """states: list of (codes tuple-set, gold). Exhaustive over
    permutations of the group's candidate set (all identical)."""
    codes = sorted(states[0][0])
    gate(len(codes) <= 9, f"GROUP >9 CODES {len(codes)}")
    for cs, _ in states:
        gate(sorted(cs) == codes, "GROUP CANDIDATE MISMATCH")
    best, n_best = -1, 0
    for perm in itertools.permutations(codes):
        top = perm[0]
        acc = sum(1 for _, g in states if g == top)
        if acc > best:
            best, n_best = acc, 1
        elif acc == best:
            n_best += 1
    return {"n": len(states), "n_codes": len(codes),
            "best": best, "n_optimal_orders": n_best,
            "gold_census": dict(Counter(
                json.dumps(list(g)) for _, g in states))}


def global_ceiling(states):
    """states: list of (codes list, gold). Every global order of the
    union set if <= 9 codes; a state is correct iff gold is ranked
    above every other code of ITS candidate set."""
    union = sorted({c for cs, _ in states for c in cs})
    if len(union) > 9:
        return {"enumerated": False, "n_union_codes": len(union)}
    idx = {c: i for i, c in enumerate(union)}
    st = [([idx[c] for c in cs], idx[g]) for cs, g in states]
    best, n_best, missed = -1, 0, Counter()
    for perm in itertools.permutations(range(len(union))):
        rank = [0] * len(union)
        for r, c in enumerate(perm):
            rank[c] = r
        ok = [rank[g] < min(rank[c] for c in cs if c != g)
              for cs, g in st]
        acc = sum(ok)
        if acc > best:
            best, n_best, missed = acc, 1, Counter()
            missed[tuple(i for i, o in enumerate(ok) if not o)] += 1
        elif acc == best:
            n_best += 1
            missed[tuple(i for i, o in enumerate(ok) if not o)] += 1
    return {"enumerated": True, "n_union_codes": len(union),
            "n_orders": math.factorial(len(union)),
            "best": best, "n_optimal_orders": n_best,
            "missed_sets": [{"states": list(k), "n_orders": v}
                            for k, v in sorted(missed.items())]}


def ceiling_report(rows, label):
    groups = defaultdict(list)
    for r in rows:
        groups[r["cand_sig_id"]].append(
            (tuple(tuple(t) for t in r["cand_tuples"]),
             tuple(r["gold_tuple"])))
    per = {g: group_ceiling(s) for g, s in groups.items()}
    n = len(rows)
    micro = sum(p["best"] for p in per.values())
    macro = (sum(p["best"] / p["n"] for p in per.values())
             / len(per)) if per else 0.0
    glob = global_ceiling([(list(cs), g) for cs, g in
                           (x for s in groups.values() for x in s)])
    balanced = all(
        len(p["gold_census"]) == 2
        and len(set(p["gold_census"].values())) == 1
        for p in per.values())
    return {"label": label, "n": n, "n_groups": len(per),
            "per_group": per, "micro_best": micro,
            "micro_ceiling": micro / n if n else None,
            "macro_ceiling": macro, "global": glob,
            "all_groups_balanced_binary": balanced,
            "ceiling_is_half": (n > 0 and micro * 2 == n
                                and abs(macro - 0.5) < 1e-12
                                and (not glob["enumerated"]
                                     or glob["best"] * 2 == n))}


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    for p in RECORDED:
        gate(Path(p).exists(), f"MISSING {p}")
    if not SMOKE:
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(sr.get("smoke") is True
             and sr.get("verdict") == "SMOKE OK", "SMOKE NOT GREEN")
        for pth, h in sr["start"]["file_sha256"].items():
            gate(fsha(pth) == h, f"SMOKE STALE {pth}")
    START = start_provenance(
        ["scratch/mathworld1_prband.py",
         "scratch/mathworld1_svpdiet3.py",
         "scratch/mathworld1_svpdiet.py",
         "scratch/mathworld1_svpdiet2.py",
         "scratch/mathworld1_svpnuisdesk.py",
         "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpchal2.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_svpforder.py",
         "scratch/mathworld1_actiontok.py",
         "llmopt/search/derivation.py",
         "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    t_all = time.monotonic()

    # ---- bank freshness gates (prereg) ----------------------
    used_polys = set()
    for grp in (F3_TRAIN_POLYS, F3_EVAL_POLYS, E1_F3_POLYS,
                N_KP, K_POLYS2,
                (X, 7 * X**3, 2 * X, 5 * X**2, 3 * X, 11 * X**4)):
        used_polys |= {sp.sstr(e) for e in grp}
    for pv in K_A + K_B:
        gate(sp.sstr(pv) not in used_polys, f"K USED {pv}")
    for c in C_PR:
        gate(c in set(F3_TRAIN_C), f"C NOT TRAINED {c}")
    aug_cells = defaultdict(set)
    import re
    for l in open(AUG):
        r = json.loads(l)
        if r["family"] != "CH-F3":
            continue
        m = re.search(r"\|P=([^|]+)\|", r["base_signature"])
        aug_cells[m.group(1)].add((r["site_ordinal"],
                                   r["param_index"]))
    for P in P12:
        gate(len(aug_cells.get(sp.sstr(P), set())) == 6,
             f"P NOT FULLY REPRESENTED {P}")

    # ---- horizon --------------------------------------------
    if SMOKE:
        hz = [(f"SMOKE|{sig}", f, Ps, c)
              for (_t, sig, f, Ps, c) in smoke_slice()]
    else:
        hz = build_pr()
        gate(len(hz) == 1920, f"HZ {len(hz)}")
    sigs = [h[0] for h in hz]
    gate(len(sigs) == len(set(sigs)), "SIG DUP")
    shas = [sha(s) for s in sigs]
    gate(len(shas) == len(set(shas)), "SIG_SHA DUP")
    reps = [sp.srepr(h[1]) for h in hz]
    gate(len(reps) == len(set(reps)), "TARGET DUP")
    burn_parents = set()
    for _s, f, _p, _c in hz:
        for _v, D in VARIANTS:
            burn_parents.add(sp.sstr(sp.Add(sp.Integral(f, X),
                                            sp.Integral(D, X))))
    if not SMOKE:
        gate(len(burn_parents) == 5760, f"BURN {len(burn_parents)}")

    # ---- burned lineage sets (svpdiet3 law + svpdiet3 + cl1) --
    train_cur = {json.loads(l)["cur"] for l in open(PAIRED)}
    band_cur = set()
    for bf in ("logs/mathworld1/svpeval/decisions.jsonl",
               "logs/mathworld1/svpeval2/decisions.jsonl",
               "logs/mathworld1/svpeval3/decisions.jsonl"):
        for l in open(bf):
            r = json.loads(l)
            if r.get("cur"):
                band_cur.add(r["cur"])
    pilot_cur = set()
    for pr in PILOT_RECEIPTS:
        for a2 in json.loads(Path(pr).read_text())["attempts"]:
            pilot_cur.add(a2["parent_sstr"])
    gate(len(pilot_cur) == 566, "PILOT")
    e1_hz = (f3_bases("E1", E1_F3_P, E1_F3_C, E1_F3_POLYS)
             + f4_bases("E1", E1_F4_P1, E1_F4_FREQS))
    diet_hz = (f3_bases("TRAIN", F3_TRAIN_P, F3_TRAIN_C,
                        F3_TRAIN_POLYS)
               + f4_bases("TRAIN", F4_TRAIN_P1, F4_TRAIN_FREQS)
               + f3_bases("EVAL", F3_EVAL_P, F3_EVAL_C,
                          F3_EVAL_POLYS)
               + f4_bases("EVAL", F4_EVAL_P1, F4_EVAL_FREQS))
    nuis_hz = (build_cell("A", N_P_IN, N_C_IN)
               + build_cell("B", N_P_IN, N_C_OUT)
               + build_cell("C", N_P_OUT, N_C_IN)
               + build_cell("D", N_P_OUT, N_C_OUT))
    d3_eval = (build_f3("IN", P12, C_IN2, K_POLYS2)
               + build_f3("OUT", P12, C_OUT2, K_POLYS2))
    d3_sec = build_f3("SEC", P_SEC, C_SEC, K_POLYS2)
    gate(len(d3_eval) == 1152 and len(d3_sec) == 640, "D3 HZ")
    dd_cur = set()
    for hzz, n in ((d0_horizon(), 720), (d1_horizon(), 2160),
                   (diet_hz, 15912), (e1_hz, 10224),
                   (nuis_hz, 3456), (d3_eval, 3456),
                   (d3_sec, 1920)):
        before = len(dd_cur)
        for tup in hzz:
            f = tup[2]
            for D in (SMALL_D[0], SMALL_D[1], AFTER_D):
                dd_cur.add(sp.sstr(sp.Add(sp.Integral(f, X),
                                          sp.Integral(D, X))))
        gate(len(dd_cur) - before == n,
             f"BURNED HORIZON {len(dd_cur) - before} v {n}")
    art_cur = set()
    for fn in ("heldout_test16.jsonl", "covered_calibration.jsonl",
               "pout_robustness.jsonl"):
        for l in open(f"{D3}/{fn}"):
            art_cur.add(json.loads(l)["cur"])
    gate(art_cur <= dd_cur, "D3 ARTIFACTS NOT IN THEIR HORIZON")
    cl1_cur = set()
    for l in open(f"{CL1}/manifest.jsonl"):
        cl1_cur.add(json.loads(l)["root_cur"])
    for l in open(f"{CL1}/raw_attempts.jsonl"):
        r = json.loads(l)
        if r.get("cur"):
            cl1_cur.add(r["cur"])
    smoke_cur = set()
    if not SMOKE:
        for fn in ("primary.jsonl", "companion.jsonl"):
            p = Path("logs/mathworld1/prband_smoke") / fn
            if p.exists():
                for l in open(p):
                    smoke_cur.add(json.loads(l)["cur"])
    burned = {"natural": train_cur, "band": band_cur,
              "pilot": pilot_cur, "horizons": dd_cur,
              "d3_artifacts": art_cur, "cl1": cl1_cur,
              "own_smoke": smoke_cur}
    train_targets = {json.loads(l)["target_integrand"] for l in
                     open("logs/mathworld1/svpdiet/train_blocks.jsonl")}
    gate(len(train_targets) == 4536, "TRAIN TARGETS")
    d3_targets = {sp.sstr(h[2]) for h in d3_eval + d3_sec}

    OUTDIR.mkdir(parents=True)
    receipt = {"smoke": SMOKE, "prereg": PREREG,
               "prereg_commit": PREREG_COMMIT,
               "n_horizon": len(hz), "n_burned_parents":
               len(burn_parents), "N_half": N_HALF, "K_MAX": K_MAX,
               "gold_a": list(GOLD_A), "gold_b": list(GOLD_B),
               "gold_a_factor": factor_symbols(*GOLD_A),
               "gold_b_factor": factor_symbols(*GOLD_B),
               "pins": {p: fsha(p) for p in
                        list(PINS) + RECORDED},
               "burned_set_sizes": {k: len(v) for k, v in
                                    burned.items()},
               "n_train_targets": len(train_targets),
               "n_d3_targets": len(d3_targets)}

    def finish(verdict, extra):
        receipt["verdict"] = verdict
        receipt.update(extra)
        receipt["wall_s"] = round(time.monotonic() - t_all, 1)
        receipt["artifact_sha256"] = {
            f.name: fsha(f) for f in sorted(OUTDIR.glob("*.json*"))
            if f.name != "prband_receipt.json"}
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "prband_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ("start", "pins")},
                         indent=1), flush=True)

    # ---- phase 1: qualify every base, three variants ---------
    hc = open(OUTDIR / "horizon_census.jsonl", "w")
    qb = open(OUTDIR / "qualified_blocks.jsonl", "w")
    blocks = []
    fail_census = Counter()
    for i, (bsig, f_t, Ps, c) in enumerate(hz):
        variants = {}
        fail = None
        for vtag, D in VARIANTS:
            row, why = qualify_parent(f_t, D)
            row["variant"] = vtag
            row["distractor"] = sp.sstr(D)
            variants[vtag] = row
            if why:
                fail = f"{vtag}:{why}"
                break
        term = None
        if fail is None:
            vs = list(variants.values())
            kinds = {(v["chosen_site_kind"], v["chosen_param_kind"])
                     for v in vs}
            rules = {v["chosen_rule"] for v in vs}
            terms = {v["chosen_term"] for v in vs}
            ords = [variants[t]["chosen_ordinal"]
                    for t in ("smallA", "smallB", "after")]
            if kinds != {("I", "term_index")}:
                fail = f"teacher_kind:{sorted(kinds)}"
            elif rules != {"i_unprod"}:
                fail = f"teacher_rule:{sorted(rules)}"
            elif len(terms) != 1:
                fail = f"term_mismatch:{sorted(terms)}"
            elif ords != [1, 1, 0]:
                fail = f"ordinals:{ords}"
            else:
                term = terms.pop()
        pv = "smallA" if i % 2 == 0 else "smallB"
        crow = {"horizon_index": i, "base_signature": bsig,
                "sig_sha": sha(bsig), "P": Ps, "c": c,
                "target_integrand": sp.sstr(f_t),
                "primary_variant": pv, "fail": fail, "term": term,
                "variants": {}}
        for vt, v in variants.items():
            ent = {"cur": v["cur"], "cur_sha": sha(v["cur"])}
            if "candidates" in v:
                js, sid = cand_sig(v["candidates"])
                gold = (v["chosen_rule"], v["chosen_site_kind"],
                        v["chosen_ordinal"], v["chosen_param_kind"],
                        v["chosen_term"])
                ent.update({"n_candidates": v["n_candidates"],
                            "min_hce_ties": v["min_hce_ties"],
                            "cand_sig": js, "cand_sig_id": sid,
                            "gold_tuple": list(gold),
                            "gold_class": gold_class(gold)})
            crow["variants"][vt] = ent
        if fail is None:
            crow["primary_cand_sig_id"] = \
                crow["variants"][pv]["cand_sig_id"]
            crow["primary_gold_class"] = \
                crow["variants"][pv]["gold_class"]
            crow["primary_gold_tuple"] = \
                crow["variants"][pv]["gold_tuple"]
            blocks.append((crow, variants))
            qb.write(json.dumps({"horizon_index": i,
                                 "base_signature": bsig,
                                 "term": term,
                                 "variants": variants}) + "\n")
        else:
            fail_census[fail.split(":")[0] if ":" in fail
                        else fail] += 1
        hc.write(json.dumps(crow) + "\n")
        if (i + 1) % 50 == 0 or SMOKE:
            print(f"[hz {i + 1}/{len(hz)} pass={len(blocks)}]",
                  flush=True)
    hc.close()
    qb.close()
    base = {"n_qualifying_blocks": len(blocks),
            "block_fail_census": dict(fail_census),
            "primary_gold_class_census": dict(Counter(
                b["primary_gold_class"] for b, _ in blocks)),
            "primary_gold_tuple_census": dict(Counter(
                json.dumps(b["primary_gold_tuple"])
                for b, _ in blocks)),
            "term_census": dict(Counter(b["term"]
                                        for b, _ in blocks)),
            "wall_qualify_s": round(time.monotonic() - t_all, 1)}

    # ---- phase 2: buckets, capacities, selection -------------
    buckets = defaultdict(list)
    for crow, variants in blocks:
        gc = crow["primary_gold_class"]
        if gc in ("a", "b"):
            buckets[(crow["primary_cand_sig_id"], gc)].append(
                (crow, variants))
    for k in buckets:
        buckets[k].sort(key=lambda bv: bv[0]["sig_sha"])
    sig_ids = sorted({k[0] for k in buckets})
    cap = {}
    for g in sig_ids:
        na, nb = len(buckets.get((g, "a"), [])), \
            len(buckets.get((g, "b"), []))
        cap[g] = {"n_a": na, "n_b": nb, "m_g": min(na, nb),
                  "cand_sig": (buckets.get((g, "a")) or
                               buckets.get((g, "b")))[0][0][
                      "variants"][
                      (buckets.get((g, "a")) or
                       buckets.get((g, "b")))[0][0][
                          "primary_variant"]]["cand_sig"]}
    usable = sorted([g for g in sig_ids if cap[g]["m_g"] > 0],
                    key=lambda g: (-cap[g]["m_g"], g))
    walk, filled, selected = [], 0, []
    for g in usable:
        if filled >= N_HALF or len(walk) >= K_MAX:
            break
        q = min(cap[g]["m_g"], N_HALF - filled)
        walk.append({"cand_sig_id": g, "m_g": cap[g]["m_g"],
                     "q_g": q})
        for gc in ("a", "b"):
            for crow, variants in buckets[(g, gc)][:q]:
                selected.append((g, gc, q, crow, variants))
        filled += q
    cap_out = {"capacity_table": cap, "n_signatures_total":
               len(sig_ids), "n_signatures_usable": len(usable),
               "usable_order": usable, "walk": walk,
               "filled": filled, "sum_m_g": sum(
                   c["m_g"] for c in cap.values()),
               "top_m_g": sorted((c["m_g"] for c in cap.values()),
                                 reverse=True)[:8]}
    (OUTDIR / "capacity_table.json").write_text(
        json.dumps(cap_out, indent=1))
    base.update({"n_signatures_total": len(sig_ids),
                 "n_signatures_with_both_golds": len(usable),
                 "sum_m_g": cap_out["sum_m_g"],
                 "top_m_g": cap_out["top_m_g"], "walk": walk,
                 "filled": filled,
                 "supply_a": sum(c["n_a"] for c in cap.values()),
                 "supply_b": sum(c["n_b"] for c in cap.values())})
    base["smoke_relaxed_selection"] = False
    if filled < N_HALF and SMOKE:
        # mechanism-complete smoke: exercise phases 3-4 on one a
        # and one b state from any signature (report-only).
        base["smoke_relaxed_selection"] = True
        selected, walk = [], []
        for gc in ("a", "b"):
            pool = [(g, gc, 1, cr, va) for (g, k), lst in
                    sorted(buckets.items()) if k == gc
                    for cr, va in lst]
            gate(pool, f"SMOKE: no gold {gc}")
            selected.append(pool[0])
    elif filled < N_HALF:
        finish("NO-FIRE", base)
        return 0

    # ---- phase 3: selected artifacts + encoding gates --------
    def emit(role, vt, g, gc, q, crow, variants, j):
        v = variants[vt]
        cands = []
        for cnd in v["candidates"]:
            tup = ctup(cnd)
            fc = factor_symbols(*tup)
            gate(fc == cnd["factor_code"], "FACTOR MISMATCH")
            gate(factor_decode(fc) == tup, "FACTOR ROUNDTRIP")
            pf = pf_encode(tup)
            gate(pf == [fc[PERM[i]] for i in range(8)], "PERM")
            gate(pf_decode(pf) == tup, "PF ROUNDTRIP")
            gate(len(fc) == 8 and len(pf) == 8, "CODE LEN")
            cands.append({**cnd, "_pf": pf})
        pre = len(TOK.encode(f"Current: {v['cur']}\nHints: none"
                             f"\nStep: "))
        gate(pre + 9 <= CTX, "CONTEXT OVERFLOW")
        js, sid = cand_sig(cands)
        gold = (v["chosen_rule"], v["chosen_site_kind"],
                v["chosen_ordinal"], v["chosen_param_kind"],
                v["chosen_term"])
        return {"block_id": f"{role}-{g[:8]}-{gc}-{j:02d}",
                "role": role, "site_role": ("heldout-I1" if
                                            role == "primary"
                                            else "covered-I0"),
                "horizon_index": crow["horizon_index"],
                "base_signature": crow["base_signature"],
                "sig_sha": crow["sig_sha"], "P": crow["P"],
                "c": crow["c"], "variant": vt,
                "distractor": v["distractor"], "cur": v["cur"],
                "cur_sha": sha(v["cur"]),
                "parent_srepr_sha": v["parent_srepr_sha"],
                "target_integrand": crow["target_integrand"],
                "cand_sig": js, "cand_sig_id": sid,
                "cand_tuples": [list(ctup(c)) for c in cands],
                "gold_tuple": list(gold),
                "gold_class": gold_class(gold),
                "primary_stratum": g, "primary_gold": gc,
                "q_g": q, "term_cell": crow["term"],
                "n_candidates": v["n_candidates"],
                "min_hce_ties": v["min_hce_ties"],
                "prompt_tokens": pre, "cont_tokens": 9,
                "total_tokens": pre + 9,
                "candidates": cands}
    primary, companion = [], []
    for j, (g, gc, q, crow, variants) in enumerate(selected):
        primary.append(emit("primary", crow["primary_variant"],
                            g, gc, q, crow, variants, j))
        companion.append(emit("companion", "after", g, gc, q,
                              crow, variants, j))
    for rows in (primary, companion):
        gate(len(rows) == 2 * N_HALF, "N")
        gate(len({r["cur_sha"] for r in rows}) == len(rows),
             "DUP CUR")
        gate(len({r["parent_srepr_sha"] for r in rows})
             == len(rows), "DUP PARENT")
    gate(not ({r["cur"] for r in primary}
              & {r["cur"] for r in companion}), "PRI/COMP OVERLAP")
    for r in primary:
        gate(r["cand_sig_id"] == r["primary_stratum"], "SIG DRIFT")
        gate(r["gold_class"] == r["primary_gold"], "GOLD DRIFT")
    # overlap fences (refute-on-hit)
    ov = {}
    all_cur = [r["cur"] for r in primary + companion]
    for k, s in burned.items():
        ov[k] = sum(c2 in s for c2 in all_cur)
    sel_targets = {r["target_integrand"] for r in primary}
    ov["train_targets"] = sum(t in train_targets
                              for t in sel_targets)
    ov["d3_targets"] = sum(t in d3_targets for t in sel_targets)
    (OUTDIR / "overlap_receipt.json").write_text(
        json.dumps({"hits": ov, "n_checked_curs": len(all_cur),
                    "n_selected_targets": len(sel_targets)},
                   indent=1))
    base["overlap_hits"] = ov
    if not SMOKE and any(ov.values()):
        finish("GATE-REFUTED", base | {"refuted_reason":
                                       f"OVERLAP {ov}"})
        raise SystemExit("GATE-REFUTED")
    with open(OUTDIR / "primary.jsonl", "w") as fh:
        for r in primary:
            fh.write(json.dumps(r) + "\n")
    with open(OUTDIR / "companion.jsonl", "w") as fh:
        for r in companion:
            fh.write(json.dumps(r) + "\n")
    base.update({
        "primary_gold_census": dict(Counter(r["gold_class"]
                                            for r in primary)),
        "primary_signature_census": dict(Counter(
            f"{r['cand_sig_id'][:12]}|{r['gold_class']}"
            for r in primary)),
        "selected_unique_bases": len({r["target_integrand"]
                                      for r in primary}),
        "family_census": {
            "P": dict(Counter(r["P"] for r in primary)),
            "c": dict(Counter(r["c"] for r in primary)),
            "T": dict(Counter(field(r, "T") for r in primary)),
            "w": dict(Counter(field(r, "w") for r in primary)),
            "kb_k": dict(Counter(f"{field(r, 'kb')}|{field(r, 'k')}"
                                 for r in primary)),
            "distractor": dict(Counter(r["distractor"]
                                       for r in primary))},
        "max_prompt_tokens": max(r["prompt_tokens"]
                                 for r in primary + companion),
        "max_total_tokens": max(r["total_tokens"]
                                for r in primary + companion),
        "context_overflow": 0})

    # ---- phase 4: ceiling gates ------------------------------
    cp = ceiling_report(primary, "primary")
    cc = ceiling_report(companion, "companion")
    (OUTDIR / "ceiling_receipt.json").write_text(
        json.dumps({"primary": cp, "companion": cc}, indent=1))
    base["primary_ceiling"] = {k: v for k, v in cp.items()
                               if k != "per_group"}
    base["companion_ceiling"] = {k: v for k, v in cc.items()
                                 if k != "per_group"}
    base["companion_verdict"] = (
        "COMPANION BALANCED" if cc["all_groups_balanced_binary"]
        and cc["ceiling_is_half"] else "COMPANION IMBALANCED")
    if not cp["ceiling_is_half"] and not SMOKE:
        finish("INSTRUMENT FAILURE", base)
        raise SystemExit("INSTRUMENT FAILURE: ceiling != 50%")
    if SMOKE:
        finish("SMOKE OK", base)
    else:
        finish("PRIOR-RESISTANT POPULATION MATERIALIZED + QUALIFIED"
               + (" (A: one signature)" if len(walk) == 1
                  else f" (B: {len(walk)} signatures)"), base)
    return 0


if __name__ == "__main__":
    sys.exit(main())

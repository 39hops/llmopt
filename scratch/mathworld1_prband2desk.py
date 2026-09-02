"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-V2-CROSSOVER-DESK-0 — execute the
frozen controlled-crossover feasibility desk (PRE-REG ...V2-CROSSOVER-
DESK-0, 488e8b23, RESULTS L63579). ZERO checkpoint access, ZERO
scoring, ZERO training, ZERO production materialization.

SKELETON: f = expand(d/dx[x^a sin(cx)]) + expand(d/dx[x^b cos(cx)])
+ Integral(sin(x)/x, x) + 21*x**3; theta = SIN_LOW -> (a, b) =
(e_lo, e_hi), COS_LOW -> (e_hi, e_lo). Bank (enumeration order e_lo,
e_hi, theta, c): e_lo in {1, 2}, e_hi in {5, 6, 7, 9}, theta in
{SIN_LOW, COS_LOW}, c in {11, 13, 17, 19} = 64 bases; three standing
variants (smallA x**x = PRIMARY always, smallB, after) = 192 burned
parents. Classifying stratum e_lo = 2 (32 primaries); e_lo = 1 is a
secondary robustness stratum that never enters B1-B7.

TARGETS (frozen): A = (i_unprod, I, 1, term_index, 1) [COS_CORRECT],
B = (i_unprod, I, 1, term_index, 4) [SIN_CORRECT]; expected mapping
SIN_LOW -> A, COS_LOW -> B; expected i_unprod set {1, 4}. Robustness:
A1 = (..., 1), B1 = (..., 3), set {1, 3, 5}.

COUNTED STRICT CROSSOVER: block's three variants qualify; A and B
legal; teacher (min over (hce, raw name, child srepr) of the COMPLETE
stable legal set) exactly A or B; min_hce_ties = 1 over the complete
set; HCE gap to the other target >= 1.0. Else TIE / NON-TARGET-
TEACHER / TARGET-ABSENT / FAIL. Bars B1-B7 and the GO / PARK /
NEEDS-REDESIGN map are the prereg's, applied mechanically.

Outputs (refuse-if-exists) under logs/mathworld1/prband2desk/:
bases.jsonl (every base, all three variants, complete candidate
lists + i_unprod anatomy), primaries.jsonl (64 smallA rows with
class labels), capacity.json, bars.json, overlap_receipt.json,
support_rider.json, prband2desk_receipt.json. SMOKE
(PRBAND2_SMOKE=1) runs the pipeline on two burned V1 bases under
logs/mathworld1/prband2desk_smoke/ (labels report-only).

    PRBAND2_SMOKE=1 .venv/bin/python scratch/mathworld1_prband2desk.py
    .venv/bin/python scratch/mathworld1_prband2desk.py          (Mac)
"""
import hashlib
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
from llmopt.search.derivation import UNSOLVED  # noqa: E402
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_prband import (K_A, K_B, cand_sig,  # noqa: E402
                                       ctup, sha)
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
                                         build_f3)
from scratch.mathworld1_svpforder import (PERM,  # noqa: E402
                                          pf_decode, pf_encode)
from scratch.mathworld1_svpnuisdesk import (C_IN as N_C_IN,  # noqa: E402
                                            C_OUT as N_C_OUT,
                                            K_POLYS as N_KP,
                                            P_IN as N_P_IN,
                                            P_OUT as N_P_OUT,
                                            build_cell)

SMOKE = os.environ.get("PRBAND2_SMOKE") == "1"
PREREG = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-CROSSOVER-DESK-0"
PREREG_COMMIT = "488e8b2397bd28857f8faae7ed570e7e0bc81a02"
OUTDIR = Path("logs/mathworld1/prband2desk_smoke" if SMOKE
              else "logs/mathworld1/prband2desk")
SMOKE_RECEIPT = Path("logs/mathworld1/prband2desk_smoke/"
                     "prband2desk_receipt.json")
CTX = 4096
TOK = ActionGCTok()
PAIRED = "data/matsub_paired.jsonl"
AUG = "logs/mathworld1/svpdiet/balanced_grid_train.jsonl"
D3 = "logs/mathworld1/svpdiet3"
CL1 = "logs/mathworld1/cl1/pop"
V1_CENSUS = "logs/mathworld1/prband/horizon_census.jsonl"
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
    V1_CENSUS:
        "2e28abf09219d4ac5ee2cb834f22f9cbc40f750c13029a8b96979f9"
        "a7e144c4c",
}
RECORDED = [f"{D3}/eval_blocks.jsonl", f"{D3}/pout_attempts.jsonl",
            f"{D3}/pout_robustness.jsonl",
            f"{CL1}/manifest.jsonl", f"{CL1}/raw_attempts.jsonl",
            "logs/mathworld1/svpdiet/train_blocks.jsonl"]

# ---- frozen banks (prereg 488e8b23) --------------------------
E_LO = (1, 2)
E_HI = (5, 6, 7, 9)
THETA = ("SIN_LOW", "COS_LOW")
C_BANK = (11, 13, 17, 19)
W = sp.sin(X) / X
K = 21 * X**3
TARGET = {2: {"A": ("i_unprod", "I", 1, "term_index", 1),
              "B": ("i_unprod", "I", 1, "term_index", 4),
              "set": [1, 4]},
          1: {"A": ("i_unprod", "I", 1, "term_index", 1),
              "B": ("i_unprod", "I", 1, "term_index", 3),
              "set": [1, 3, 5]}}
ROLE_OF = {"A": "COS_CORRECT", "B": "SIN_CORRECT"}
EXPECT = {"SIN_LOW": "A", "COS_LOW": "B"}
VARIANTS = (("smallA", SMALL_D[0]), ("smallB", SMALL_D[1]),
            ("after", AFTER_D))
N_CLASS = 32


def build_desk():
    out = []
    for e_lo in E_LO:
        for e_hi in E_HI:
            for theta in THETA:
                a, b = (e_lo, e_hi) if theta == "SIN_LOW" \
                    else (e_hi, e_lo)
                for c in C_BANK:
                    f = (sp.expand(sp.diff(X**a * sp.sin(c * X), X))
                         + sp.expand(sp.diff(X**b * sp.cos(c * X), X))
                         + sp.Integral(W, X) + K)
                    sig = (f"V2D|elo={e_lo}|ehi={e_hi}|theta={theta}"
                           f"|c={c}|w={sp.sstr(W)}|K={sp.sstr(K)}")
                    out.append({"base_signature": sig, "f": f,
                                "e_lo": e_lo, "e_hi": e_hi,
                                "theta": theta, "c": c, "a": a,
                                "b": b})
    return out


def smoke_bases():
    rows = [json.loads(l) for l in open(V1_CENSUS)][:2]
    out = []
    for r in rows:
        out.append({"base_signature": "SMOKE|" + r["base_signature"],
                    "f": sp.sympify(r["target_integrand"]),
                    "e_lo": 2, "e_hi": 0, "theta": "SIN_LOW",
                    "c": r["c"], "a": -1, "b": -1})
    return out


def hce_of(expr):
    return 100.0 * len(expr.atoms(*UNSOLVED)) \
        + float(sp.count_ops(expr)) + 0.1


def anatomy(base, D, v):
    """Per-candidate recomputation + i_unprod role trace."""
    f = base["f"]
    terms = list(f.args)
    cands = []
    for c in v["candidates"]:
        e = eval(c["child_srepr"], sp.__dict__)
        tup = ctup(c)
        fc = factor_symbols(*tup)
        pf = pf_encode(tup)
        gate(fc == c["factor_code"] and factor_decode(fc) == tup
             and pf == [fc[PERM[i]] for i in range(8)]
             and pf_decode(pf) == tup, "CODE GATE")
        row = {**c, "_pf": pf, "hce": hce_of(e),
               "ops": int(sp.count_ops(e)),
               "unsolved": len(e.atoms(*UNSOLVED))}
        if c["rule"] == "i_unprod":
            ints = [t for t in (e.args if isinstance(e, sp.Add)
                                else [e]) if isinstance(t, sp.Integral)]
            A = e - sum(ints)
            top = [t for t in ints if t.function != D]
            r = top[0].function if top else sp.Integer(0)
            t = terms[c["param_index"]]
            if sp.expand(A - X**base["a"] * sp.sin(base["c"] * X)) == 0:
                role = "SIN_CORRECT"
            elif sp.expand(A - X**base["b"]
                           * sp.cos(base["c"] * X)) == 0:
                role = "COS_CORRECT"
            elif t.has(sp.sin, sp.cos, sp.exp):
                role = "WRONG_FAMILY"
            else:
                role = "OTHER"
            row.update({"source_term": sp.sstr(t),
                        "source_term_index": c["param_index"],
                        "role": role, "A": sp.sstr(A),
                        "residual": sp.sstr(r),
                        "A_ops": int(sp.count_ops(A)),
                        "r_ops": int(sp.count_ops(r))})
        cands.append(row)
    lab = [c for c in cands if c["is_label"]]
    gate(len(lab) == 1, "LABEL")
    best = min(c["hce"] for c in cands)
    ties = sum(1 for c in cands if c["hce"] == best)
    gate(lab[0]["hce"] == best, "LABEL NOT MIN HCE")
    gate(ties == v["min_hce_ties"], "TIES MISMATCH")
    for c in cands:
        c["hce_minus_teacher"] = round(c["hce"] - lab[0]["hce"], 3)
    js, sid = cand_sig(cands)
    return {"cur": v["cur"], "cur_sha": sha(v["cur"]),
            "parent_srepr_sha": v["parent_srepr_sha"],
            "n_candidates": v["n_candidates"],
            "min_hce_ties": v["min_hce_ties"],
            "teacher": list(ctup(lab[0])), "teacher_hce": lab[0]["hce"],
            "cand_sig": js, "cand_sig_id": sid,
            "cand_tuples": [list(ctup(c)) for c in cands],
            "unprod_set": sorted(c["param_index"] for c in cands
                                 if c["rule"] == "i_unprod"),
            "prompt_tokens": len(TOK.encode(
                f"Current: {v['cur']}\nHints: none\nStep: ")),
            "candidates": cands}


def classify(prim, block_ok, e_lo):
    T = TARGET[e_lo]
    if not block_ok:
        return "FAIL", None
    tups = [tuple(t) for t in prim["cand_tuples"]]
    if T["A"] not in tups or T["B"] not in tups:
        return "TARGET-ABSENT", None
    teacher = tuple(prim["teacher"])
    if teacher not in (T["A"], T["B"]):
        return "NON-TARGET-TEACHER", None
    if prim["min_hce_ties"] != 1:
        return "TIE", None
    other = T["B"] if teacher == T["A"] else T["A"]
    oh = [c["hce"] for c in prim["candidates"]
          if ctup(c) == other][0]
    gap = round(oh - prim["teacher_hce"], 3)
    if gap < 1.0:
        return "TIE", gap
    return ("STRICT-A" if teacher == T["A"] else "STRICT-B"), gap


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
        gate(sr.get("smoke") is True and sr.get("verdict") == "SMOKE OK",
             "SMOKE NOT GREEN")
        for pth, h in sr["start"]["file_sha256"].items():
            gate(fsha(pth) == h, f"SMOKE STALE {pth}")
    START = start_provenance(
        ["scratch/mathworld1_prband2desk.py",
         "scratch/mathworld1_prband.py",
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
    t0 = time.monotonic()

    # ---- K gate: absent from every burned inert-addend bank ----
    used_polys = set()
    for grp in (F3_TRAIN_POLYS, F3_EVAL_POLYS, E1_F3_POLYS, N_KP,
                K_POLYS2, K_A, K_B,
                (X, 7 * X**3, 2 * X, 5 * X**2, 3 * X, 11 * X**4)):
        used_polys |= {sp.sstr(e) for e in grp}
    gate(sp.sstr(K) not in used_polys, "K USED")
    gate(all(c % 2 == 1 for c in C_BANK), "EVEN C")

    # ---- bases ---------------------------------------------------
    bases = smoke_bases() if SMOKE else build_desk()
    if not SMOKE:
        gate(len(bases) == 64, "BASES")
    sigs = [b["base_signature"] for b in bases]
    gate(len(set(sigs)) == len(sigs), "SIG DUP")
    gate(len({sha(s) for s in sigs}) == len(sigs), "SIG_SHA DUP")
    gate(len({sp.srepr(b["f"]) for b in bases}) == len(bases),
         "SREPR DUP")
    burn = set()
    for b in bases:
        for _v, D in VARIANTS:
            burn.add(sp.sstr(sp.Add(sp.Integral(b["f"], X),
                                    sp.Integral(D, X))))
    if not SMOKE:
        gate(len(burn) == 192, f"BURN {len(burn)}")

    # ---- burned lineage sets (V1 law + V1 census) ---------------
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
    diet_hz = (f3_bases("TRAIN", F3_TRAIN_P, F3_TRAIN_C, F3_TRAIN_POLYS)
               + f4_bases("TRAIN", F4_TRAIN_P1, F4_TRAIN_FREQS)
               + f3_bases("EVAL", F3_EVAL_P, F3_EVAL_C, F3_EVAL_POLYS)
               + f4_bases("EVAL", F4_EVAL_P1, F4_EVAL_FREQS))
    nuis_hz = (build_cell("A", N_P_IN, N_C_IN)
               + build_cell("B", N_P_IN, N_C_OUT)
               + build_cell("C", N_P_OUT, N_C_IN)
               + build_cell("D", N_P_OUT, N_C_OUT))
    d3_eval = (build_f3("IN", P12, C_IN2, K_POLYS2)
               + build_f3("OUT", P12, C_OUT2, K_POLYS2))
    d3_sec = build_f3("SEC", P_SEC, C_SEC, K_POLYS2)
    dd_cur = set()
    for hzz, n in ((d0_horizon(), 720), (d1_horizon(), 2160),
                   (diet_hz, 15912), (e1_hz, 10224), (nuis_hz, 3456),
                   (d3_eval, 3456), (d3_sec, 1920)):
        before = len(dd_cur)
        for tup in hzz:
            for D in (SMALL_D[0], SMALL_D[1], AFTER_D):
                dd_cur.add(sp.sstr(sp.Add(sp.Integral(tup[2], X),
                                          sp.Integral(D, X))))
        gate(len(dd_cur) - before == n, f"BURNED HORIZON {n}")
    art_cur = set()
    for fn in ("heldout_test16.jsonl", "covered_calibration.jsonl",
               "pout_robustness.jsonl"):
        for l in open(f"{D3}/{fn}"):
            art_cur.add(json.loads(l)["cur"])
    cl1_cur = set()
    for l in open(f"{CL1}/manifest.jsonl"):
        cl1_cur.add(json.loads(l)["root_cur"])
    for l in open(f"{CL1}/raw_attempts.jsonl"):
        r = json.loads(l)
        if r.get("cur"):
            cl1_cur.add(r["cur"])
    v1_cur = set()
    for l in open(V1_CENSUS):
        r = json.loads(l)
        for v in r["variants"].values():
            v1_cur.add(v["cur"])
    gate(len(v1_cur) == 5760, f"V1 CENSUS {len(v1_cur)}")
    smoke_cur = set()
    if not SMOKE:
        p = Path("logs/mathworld1/prband2desk_smoke/bases.jsonl")
        if p.exists():
            for l in open(p):
                for v in json.loads(l)["variants"].values():
                    smoke_cur.add(v["cur"])
    burned = {"natural": train_cur, "band": band_cur, "pilot": pilot_cur,
              "horizons": dd_cur, "d3_artifacts": art_cur,
              "cl1": cl1_cur, "v1_census": v1_cur,
              "own_smoke": smoke_cur}

    OUTDIR.mkdir(parents=True)
    receipt = {"smoke": SMOKE, "prereg": PREREG,
               "prereg_commit": PREREG_COMMIT, "n_bases": len(bases),
               "n_burned_parents": len(burn),
               "K": sp.sstr(K), "w": sp.sstr(W),
               "bank": {"e_lo": E_LO, "e_hi": E_HI, "theta": THETA,
                        "c": C_BANK},
               "targets": {str(k): {"A": list(v["A"]), "B": list(v["B"]),
                                    "A_factor": factor_symbols(*v["A"]),
                                    "B_factor": factor_symbols(*v["B"]),
                                    "expected_set": v["set"]}
                           for k, v in TARGET.items()},
               "expected_mapping": EXPECT,
               "pins": {p: fsha(p) for p in list(PINS) + RECORDED},
               "burned_set_sizes": {k: len(v) for k, v in burned.items()}}

    def finish(verdict, extra):
        receipt["verdict"] = verdict
        receipt.update(extra)
        receipt["wall_s"] = round(time.monotonic() - t0, 1)
        receipt["artifact_sha256"] = {
            f.name: fsha(f) for f in sorted(OUTDIR.glob("*.json*"))
            if f.name != "prband2desk_receipt.json"}
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "prband2desk_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ("start", "pins")}, indent=1),
              flush=True)

    # ---- qualify all 192 parents, trace anatomy ------------------
    bf = open(OUTDIR / "bases.jsonl", "w")
    prims = []
    fail_census = Counter()
    all_cur = []
    for i, b in enumerate(bases):
        variants = {}
        fail = None
        for vt, D in VARIANTS:
            row, why = qualify_parent(b["f"], D)
            all_cur.append(row["cur"])
            if why:
                fail = f"{vt}:{why}"
                variants[vt] = {"cur": row["cur"], "fail": why}
                continue
            variants[vt] = anatomy(b, D, row) | {"variant": vt,
                                                   "distractor":
                                                   sp.sstr(D)}
        if fail:
            fail_census[fail] += 1
        brow = {"index": i, "base_signature": b["base_signature"],
                "sig_sha": sha(b["base_signature"]),
                "e_lo": b["e_lo"], "e_hi": b["e_hi"], "theta": b["theta"],
                "c": b["c"], "a": b["a"], "b": b["b"],
                "f": sp.sstr(b["f"]), "f_args": [sp.sstr(t) for t in
                                                 b["f"].args],
                "n_terms": len(b["f"].args), "fail": fail,
                "variants": variants}
        bf.write(json.dumps(brow) + "\n")
        pv = variants["smallA"]
        if fail is None:
            cls, gap = classify(pv, True, b["e_lo"])
            roles = {c["source_term_index"]: c["role"]
                     for c in pv["candidates"] if c["rule"] == "i_unprod"}
            T = TARGET[b["e_lo"]]
            prims.append({**{k: brow[k] for k in
                             ("index", "base_signature", "sig_sha",
                              "e_lo", "e_hi", "theta", "c", "a", "b",
                              "f_args", "n_terms")},
                          **pv, "class": cls, "gap_to_other_target": gap,
                          "roles_by_index": roles,
                          "role_map_ok": (
                              roles.get(T["A"][4]) == "COS_CORRECT"
                              and roles.get(T["B"][4]) == "SIN_CORRECT"),
                          "set_ok": pv["unprod_set"] == T["set"],
                          "expected_winner": EXPECT[b["theta"]],
                          "mapping_ok": (cls == f"STRICT-{EXPECT[b['theta']]}")
                          if cls.startswith("STRICT") else None})
        else:
            prims.append({"index": i, "base_signature": b["base_signature"],
                          "e_lo": b["e_lo"], "theta": b["theta"],
                          "class": "FAIL", "fail": fail})
        print(f"[base {i + 1}/{len(bases)}] {b['base_signature']} -> "
              f"{prims[-1]['class']}", flush=True)
    bf.close()
    with open(OUTDIR / "primaries.jsonl", "w") as fh:
        for p in prims:
            fh.write(json.dumps(p) + "\n")

    # ---- overlap (all 192) --------------------------------------
    ov = {k: sum(c2 in s for c2 in all_cur) for k, s in burned.items()}
    (OUTDIR / "overlap_receipt.json").write_text(json.dumps(
        {"hits": ov, "n_checked": len(all_cur),
         "n_unique": len(set(all_cur))}, indent=1))

    # ---- bars per stratum ----------------------------------------
    def stratum(e_lo):
        P = [p for p in prims if p["e_lo"] == e_lo]
        n = len(P)
        cls = Counter(p["class"] for p in P)
        by_theta = {th: dict(Counter(p["class"] for p in P
                                     if p["theta"] == th))
                    for th in THETA}
        strict = [p for p in P if p["class"].startswith("STRICT")]
        b1_num = sum(1 for p in P if p.get("role_map_ok")
                     and p.get("set_ok"))
        gaps = Counter(p["gap_to_other_target"] for p in strict)
        viol = sum(1 for p in strict if p["mapping_ok"] is False)
        groups = defaultdict(lambda: {"n_A": 0, "n_B": 0})
        for p in strict:
            groups[p["cand_sig_id"]]["n_" + p["class"][-1]] += 1
        for g in groups.values():
            g["m_g"] = min(g["n_A"], g["n_B"])
        mixed = {g: v for g, v in groups.items() if v["m_g"] > 0}
        ms = sorted((v["m_g"] for v in groups.values()), reverse=True)
        sum_m = sum(ms)
        top2 = sum(ms[:2])
        need = None
        acc = 0
        for j, m in enumerate(ms):
            acc += m
            if acc >= 12:
                need = j + 1
                break
        in_mixed = sum(1 for p in strict if p["cand_sig_id"] in mixed)
        all_sigs = Counter(p.get("cand_sig_id") for p in P
                           if p["class"] != "FAIL")
        uniq_target = sum(1 for p in P if p["class"].startswith("STRICT"))
        res = {"n": n, "class_census": dict(cls), "by_theta": by_theta,
               "B1": {"num": b1_num, "den": n, "pass": b1_num >= 29},
               "B2": {"strict_A": cls["STRICT-A"],
                      "strict_B": cls["STRICT-B"],
                      "pass": cls["STRICT-A"] >= 12 and cls["STRICT-B"] >= 12,
                      "mapping_violations": viol,
                      "gap_distribution": {str(k): v
                                           for k, v in gaps.items()}},
               "B3": {"n_signatures_complete": len(all_sigs),
                      "signature_census": dict(all_sigs),
                      "n_mixed": len(mixed),
                      "groups": dict(groups), "top_m_g": ms[:4],
                      "sum_m_g": sum_m, "top2_capacity": top2,
                      "min_signatures_for_12": need,
                      "pass": sum_m >= 12 and top2 >= 12},
               "B4": {"num": in_mixed, "den": len(strict),
                      "pass": (len(strict) > 0
                               and in_mixed >= 0.8 * len(strict))},
               "B5": {"blocks_qualified": n - cls["FAIL"], "den": n,
                      "fail_census": dict(fail_census),
                      "pass": cls["FAIL"] == 0 and n == N_CLASS},
               "B6": {"hits": ov, "den": len(all_cur),
                      "pass": not any(ov.values())},
               "B7": {"num": uniq_target, "den": n,
                      "tie": cls["TIE"],
                      "non_target": cls["NON-TARGET-TEACHER"],
                      "target_absent": cls["TARGET-ABSENT"],
                      "fail": cls["FAIL"], "pass": uniq_target >= 24}}
        return res
    bars = {"classifying_e_lo2": stratum(2),
            "robustness_e_lo1": stratum(1)}
    cb = bars["classifying_e_lo2"]
    passes = {k: cb[k]["pass"] for k in ("B1", "B2", "B3", "B4", "B5",
                                         "B6", "B7")}
    if SMOKE:
        decision = "SMOKE OK"
    elif all(passes.values()):
        decision = "GO V2 PRODUCTION DESIGN"
    elif (cb["B2"]["strict_A"] < 6 or cb["B2"]["strict_B"] < 6
          or cb["B3"]["sum_m_g"] < 6):
        decision = "PARK CONTROLLED CROSSOVER"
    else:
        decision = "NEEDS-REDESIGN"
    bars["passes"] = passes
    bars["decision"] = decision
    (OUTDIR / "bars.json").write_text(json.dumps(bars, indent=1))
    (OUTDIR / "capacity.json").write_text(json.dumps(
        {"classifying": cb["B3"], "robustness":
         bars["robustness_e_lo1"]["B3"]}, indent=1))

    # ---- training-support rider (zero effect) --------------------
    whole, rp, comp = Counter(), Counter(), [Counter() for _ in range(8)]
    n_rows = 0
    for path in (PAIRED, AUG):
        for l in open(path):
            r = json.loads(l)
            n_rows += 1
            t = (r["rule"], r["site_kind"], r["site_ordinal"],
                 r["param_kind"], r["param_index"])
            whole[t] += 1
            rp[(t[0], t[3], t[4])] += 1
            for j, s2 in enumerate(factor_symbols(*t)):
                comp[j][s2] += 1
    rider = {"n_training_rows": n_rows}
    for k, T in TARGET.items():
        for nm in ("A", "B"):
            t = T[nm]
            fc = factor_symbols(*t)
            rider[f"e_lo{k}_{nm}"] = {
                "tuple": list(t), "whole_count": whole[t],
                "rule_param_count": rp[(t[0], t[3], t[4])],
                "factor_code": fc,
                "component_seen": [comp[j][s2] for j, s2 in enumerate(fc)]}
    (OUTDIR / "support_rider.json").write_text(json.dumps(rider, indent=1))
    finish(decision, {"passes": passes, "classifying": {
        k: v for k, v in cb.items() if k not in ("B3",)} | {
            "B3_summary": {k2: v2 for k2, v2 in cb["B3"].items()
                           if k2 not in ("groups", "signature_census")}},
        "robustness_summary": {
            "class_census": bars["robustness_e_lo1"]["class_census"],
            "by_theta": bars["robustness_e_lo1"]["by_theta"],
            "B1": bars["robustness_e_lo1"]["B1"],
            "sum_m_g": bars["robustness_e_lo1"]["B3"]["sum_m_g"],
            "gap_distribution":
                bars["robustness_e_lo1"]["B2"]["gap_distribution"]},
        "support_rider": {k: (v if k == "n_training_rows" else
                              {"whole": v["whole_count"],
                               "rule_param": v["rule_param_count"],
                               "component_seen": v["component_seen"]})
                          for k, v in rider.items()}})
    return 0


if __name__ == "__main__":
    sys.exit(main())

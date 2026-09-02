"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-V2-PRODUCTION-MATERIALIZATION-0 —
execute the frozen production prereg (PRE-REG ...V2-PRODUCTION-0,
fd78a82b, RESULTS L64343). ZERO checkpoint access, ZERO scoring,
ZERO training. Adopt-not-fork of scratch/mathworld1_prband2desk.py
(skeleton, qualification, anatomy, signature, burn sets).

HORIZON: e_lo = 1; e_hi {5,6,7,9} x theta {SIN_LOW, COS_LOW} x c
{29,31,37,41,43,47} x K {33x^3, 27x^4, 35x^3}; w = sin(x)/x; 144
bases; three standing variants (smallA x**x, smallB 1/(x+log x),
after sin(sin x)) = 432 parents, ALL burned and persisted BEFORE any
qualification. PRIMARY = after (site ordinal 0). Pair key = (e_hi,
c, K): 72 keys, each one SIN_LOW + one COS_LOW arm, indivisible.

PRIMARY PAIR ELIGIBILITY (section C, both arms): all three variants
qualify (D); site I0; A0 = (i_unprod, I, 0, term_index, 1) and B0 =
(..., 3) legal; unique full-legal-set teacher (min_hce_ties = 1);
SIN_LOW -> A0, COS_LOW -> B0; |HCE(A0) - HCE(B0)| = 3.0 with the
teacher the cheaper; complete signature byte-identical across arms
and equal to the frozen witness 0b66cc43...; every candidate's
factor / pf / PERM roundtrip; T = 9; unique parent; zero overlap
hits. Capacity: 72 FULL-RESERVE / 48-71 PARTIAL-RESERVE / < 48
NO-FIRE. Selection: pair_sha = sha256("ehi=|c=|K=") ascending, first
48 pairs, both arms. Theorems: all 24 fixed orders enumerated;
max top-1 must be 48/96 and max both-correct pairs 0/48.
COMPANION: smallA arms of the same 48 keys, witness 425e56e6...,
availability per pair, no substitution.

Outputs (refuse-if-exists) under logs/mathworld1/prband2prod/
(smoke: prband2prod_smoke/): horizon.jsonl (144), parents.jsonl
(432 burned curs, written first), rows.jsonl (every parent's
qualification + anatomy), pairs.jsonl (72 keys, eligibility, both
arms), primary.jsonl (96), companion.jsonl, permutations.json,
nuisance.json, overlap_receipt.json, prband2prod_receipt.json.

    PRBAND2P_SMOKE=1 .venv/bin/python scratch/mathworld1_prband2prod.py
    .venv/bin/python scratch/mathworld1_prband2prod.py          (Mac)
"""
import hashlib
import itertools
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
from scratch.mathworld1_prband import K_A, K_B, ctup, sha  # noqa: E402
from scratch.mathworld1_prband2desk import (VARIANTS,  # noqa: E402
                                            anatomy)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import (AFTER_D,  # noqa: E402
                                        SMALL_D, X, fsha,
                                        qualify_parent)
from scratch.mathworld1_svpchal import \
    build_horizon as d0_horizon  # noqa: E402
from scratch.mathworld1_svpchal2 import \
    build_horizon1 as d1_horizon  # noqa: E402
from scratch.mathworld1_svpcode import factor_symbols  # noqa: E402
from scratch.mathworld1_svpdiet import (BURNED_F3_C,  # noqa: E402
                                        F3_EVAL_C, F3_EVAL_P,
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
from scratch.mathworld1_svpnuisdesk import (C_IN as N_C_IN,  # noqa: E402
                                            C_OUT as N_C_OUT,
                                            K_POLYS as N_KP,
                                            P_IN as N_P_IN,
                                            P_OUT as N_P_OUT,
                                            build_cell)

SMOKE = os.environ.get("PRBAND2P_SMOKE") == "1"
PREREG = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-PRODUCTION-0"
PREREG_COMMIT = "fd78a82b4660ebf8996c8def3c9fdc90124c0a4d"
OUTDIR = Path("logs/mathworld1/prband2prod_smoke" if SMOKE
              else "logs/mathworld1/prband2prod")
SMOKE_RECEIPT = Path("logs/mathworld1/prband2prod_smoke/"
                     "prband2prod_receipt.json")
CTX = 4096
TOK = ActionGCTok()
N_SELECT = 1 if SMOKE else 48

# ---- frozen manifest (section I of fd78a82b) + support pins (H) ----
MANIFEST = {
    "logs/mathworld1/prband2desk/bases.jsonl":
        "0fd7ac37b75a1dae4547afa3e8b3fd0fa6077d97cb169df34cfa306afb71e80c",
    "logs/mathworld1/prband/horizon_census.jsonl":
        "2e28abf09219d4ac5ee2cb834f22f9cbc40f750c13029a8b96979f9a7e144c4c",
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75d8402351d468e8",
    "logs/mathworld1/svpdiet/train_blocks.jsonl":
        "dd5e72391db557049e45efe6c6b5aff2118c8ea24ef53c85496b04f8539159d5",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f3165241406a86253858c6",
    "logs/mathworld1/svpeval2/decisions.jsonl":
        "89efbe0ea447ee937c0c130d5419112921a2dd6c2159c6c2112cfd5e92f79315",
    "logs/mathworld1/svpeval3/decisions.jsonl":
        "2ff5433249622df9d421cf8014131b3907092a943040bb7b20f46f1afffb7efa",
    "logs/mathworld1/svpgriddesk_receipt.json":
        "ec9cb9b870d2515e7959025f7f3cbfcee7309a6dc90b880d3176d3e2ccf72edc",
    "logs/mathworld1/svpgriddesk2_receipt.json":
        "f0184b01c36017bcb93ed4b715e41e015999075c4da7976e11adc5cae28e6977",
    "logs/mathworld1/svpgriddesk3_receipt.json":
        "26389ebb8d68f45447d9676b9c188ea86ab9dd70ec71329fe6ae268cfc34080f",
    "logs/mathworld1/svpgriddesk4_receipt.json":
        "8439fc636fcf5c6e18a7dd75a76642cfe088e7eeeaf1e9f1243c5bb0cf08610f",
    "logs/mathworld1/svpchal/blocks.jsonl":
        "21e7e635244574266ec876c1c8c76f8d7d2a77e20c2f6680d3ee48db372c4d10",
    "logs/mathworld1/svpnuisdesk/attempts.jsonl":
        "54659cda39d0578c507d984aae7fdcb90c9d3cc86bf3b5a82fed740be93b5504",
    "logs/mathworld1/svpdiet3/eval_blocks.jsonl":
        "d500ee554e1cbfe6cb9e8e594b8324f9529f08710e3312d1375a4f44a4c5ca0b",
    "logs/mathworld1/svpdiet3/pout_attempts.jsonl":
        "3f63680ab29724644b9c564cd31f71faf6a78b602e01b102774b6520a8097dc8",
    "logs/mathworld1/svpdiet3/heldout_test16.jsonl":
        "a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec13881b4509df46ddb",
    "logs/mathworld1/svpdiet3/covered_calibration.jsonl":
        "af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b08509794d0fb7978db3",
    "logs/mathworld1/svpdiet3/pout_robustness.jsonl":
        "5c85fc1f336791522db78e681bfdadad8c4efdaafed640a7aa503d72a82c6137",
    "logs/mathworld1/cl1/pop/manifest.jsonl":
        "50c05794d9773142c55b932f69685ad1c0124d168c12dd1f830e36a2944d1846",
    "logs/mathworld1/cl1/pop/raw_attempts.jsonl":
        "95d3761e81e9e5a9c94ea6b63d3ef465ac56873cb605235bdc992cf6e40b500a",
    "logs/mathworld1/svpdiet3/svpdiet3_receipt.json":
        "26cb6d0119f56e24b4025d43976ddf323a5540e0177c19583bfe2f5c984fb365",
    "logs/mathworld1/prband2desk_verify/support_matrix.json":
        "65b77230775f4f7b05e341f40220c7069b2996e5b7b1bf187b492cb202e02b0e",
    "logs/mathworld1/svpdiet/balanced_grid_train.jsonl":
        "0ef3d8a880a7e07712d8de757bc1670df12701e487b856b44c97f8db16cb3759",
}
SUPPORT = {"A0": 2267, "B0": 257, "rows": 74860}

# ---- frozen banks (section A) ---------------------------------------
E_LO = 1
E_HI = (5, 6, 7, 9)
THETA = ("SIN_LOW", "COS_LOW")
C_BANK = (29, 31, 37, 41, 43, 47)
K_BANK = (33 * X**3, 27 * X**4, 35 * X**3)
W = sp.sin(X) / X
A0 = ("i_unprod", "I", 0, "term_index", 1)
B0 = ("i_unprod", "I", 0, "term_index", 3)
WITNESS_ID = ("0b66cc4381c54e0fa8c2a6e6ea6bac4de06011c3f8e52e1bfcb99"
              "86d993202d8")
WITNESS = [["i_sum", "I", 0, "none", -1],
           ["i_unprod", "I", 0, "term_index", 1],
           ["i_unprod", "I", 0, "term_index", 3],
           ["i_unprod", "I", 0, "term_index", 5]]
A1 = ("i_unprod", "I", 1, "term_index", 1)
B1 = ("i_unprod", "I", 1, "term_index", 3)
COMP_ID = ("425e56e6e3bb06b8d36aaadb13b25a0373665cc840374bd4cbcb39b1"
           "6e26332f")
COMP = [["i_sum", "I", 1, "none", -1],
        ["i_unprod", "I", 1, "term_index", 1],
        ["i_unprod", "I", 1, "term_index", 3],
        ["i_unprod", "I", 1, "term_index", 5]]
EXPECT = {"SIN_LOW": "A", "COS_LOW": "B"}
GAP = 3.0


def build_horizon():
    out = []
    for e_hi in E_HI:
        for theta in THETA:
            a, b = (E_LO, e_hi) if theta == "SIN_LOW" else (e_hi, E_LO)
            for c in C_BANK:
                for K in K_BANK:
                    f = (sp.expand(sp.diff(X**a * sp.sin(c * X), X))
                         + sp.expand(sp.diff(X**b * sp.cos(c * X), X))
                         + sp.Integral(W, X) + K)
                    sig = (f"V2P|ehi={e_hi}|theta={theta}|c={c}|K="
                           f"{sp.sstr(K)}|w=sin(x)/x")
                    pk = f"ehi={e_hi}|c={c}|K={sp.sstr(K)}"
                    out.append({"base_signature": sig, "f": f,
                                "e_lo": E_LO, "e_hi": e_hi,
                                "theta": theta, "c": c, "a": a, "b": b,
                                "K": sp.sstr(K), "pair_key": pk,
                                "pair_sha": sha(pk)})
    return out


def smoke_horizon():
    rows = [json.loads(l) for l in
            open("logs/mathworld1/prband2desk/bases.jsonl")]
    rows = [r for r in rows if r["e_lo"] == 1 and r["e_hi"] == 5
            and r["c"] == 11]  # one SIN_LOW + one COS_LOW, same key
    out = []
    for r in rows:
        out.append({"base_signature": "SMOKE|" + r["base_signature"],
                    "f": sp.sympify(r["f"]), "e_lo": 1, "e_hi": r["e_hi"],
                    "theta": r["theta"], "c": r["c"], "a": r["a"],
                    "b": r["b"], "K": "21*x**3",
                    "pair_key": f"ehi={r['e_hi']}|c={r['c']}|K=21*x**3",
                    "pair_sha": sha(f"ehi={r['e_hi']}|c={r['c']}|K=21*x**3")})
    return out


def burned_sets():
    train_cur = {json.loads(l)["cur"] for l in
                 open("data/matsub_paired.jsonl")}
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
        for l in open(f"logs/mathworld1/svpdiet3/{fn}"):
            art_cur.add(json.loads(l)["cur"])
    cl1_cur = set()
    for l in open("logs/mathworld1/cl1/pop/manifest.jsonl"):
        cl1_cur.add(json.loads(l)["root_cur"])
    for l in open("logs/mathworld1/cl1/pop/raw_attempts.jsonl"):
        r = json.loads(l)
        if r.get("cur"):
            cl1_cur.add(r["cur"])
    v1_cur = set()
    for l in open("logs/mathworld1/prband/horizon_census.jsonl"):
        for v in json.loads(l)["variants"].values():
            v1_cur.add(v["cur"])
    gate(len(v1_cur) == 5760, "V1 CENSUS")
    v2_cur = set()
    for l in open("logs/mathworld1/prband2desk/bases.jsonl"):
        for v in json.loads(l)["variants"].values():
            v2_cur.add(v["cur"])
    gate(len(v2_cur) == 192, "V2 DESK")
    smoke_cur = set()
    if not SMOKE:
        p = Path("logs/mathworld1/prband2prod_smoke/parents.jsonl")
        if p.exists():
            smoke_cur = {json.loads(l)["cur"] for l in open(p)}
    train_targets = {json.loads(l)["target_integrand"] for l in
                     open("logs/mathworld1/svpdiet/train_blocks.jsonl")}
    gate(len(train_targets) == 4536, "TRAIN TARGETS")
    return {"natural": train_cur, "band": band_cur, "pilot": pilot_cur,
            "horizons": dd_cur, "d3_artifacts": art_cur, "cl1": cl1_cur,
            "v1_census": v1_cur, "v2_desk": v2_cur,
            "own_smoke": smoke_cur}, train_targets


def support_check():
    whole = Counter()
    n = 0
    for p in ("data/matsub_paired.jsonl",
              "logs/mathworld1/svpdiet/balanced_grid_train.jsonl"):
        for l in open(p):
            r = json.loads(l)
            n += 1
            whole[(r["rule"], r["site_kind"], r["site_ordinal"],
                   r["param_kind"], r["param_index"])] += 1
    gate(n == SUPPORT["rows"] and whole[A0] == SUPPORT["A0"]
         and whole[B0] == SUPPORT["B0"], "SUPPORT DRIFT")
    return {"rows": n, "A0_whole": whole[A0], "B0_whole": whole[B0],
            "ratio": round(whole[A0] / whole[B0], 2)}


def arm_gates(v, target_A, target_B, witness_id, witness, expect):
    """Section C gates on one qualified arm's anatomy row. Returns
    (ok, reasons, gap, teacher_side)."""
    why = []
    tups = [tuple(t) for t in v["cand_tuples"]]
    if target_A not in tups:
        why.append("A_absent")
    if target_B not in tups:
        why.append("B_absent")
    if v["min_hce_ties"] != 1:
        why.append("ties")
    t = tuple(v["teacher"])
    side = "A" if t == target_A else "B" if t == target_B else None
    if side is None:
        why.append("non_target_teacher")
    if side is not None and side != expect:
        why.append("mapping")
    gap = None
    if not why:
        hA = [c["hce"] for c in v["candidates"] if ctup(c) == target_A][0]
        hB = [c["hce"] for c in v["candidates"] if ctup(c) == target_B][0]
        gap = round(abs(hA - hB), 3)
        if gap != GAP:
            why.append(f"gap_{gap}")
        if min(hA, hB) != v["teacher_hce"]:
            why.append("teacher_not_cheaper")
    if v["cand_sig_id"] != witness_id or json.loads(v["cand_sig"]) != witness:
        why.append("signature")
    for c in v["candidates"]:
        if len(c["factor_code"]) != 8 or len(c["_pf"]) != 8:
            why.append("code_len")
            break
    if v["prompt_tokens"] + 9 > CTX:
        why.append("context")
    return (not why), why, gap, side


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in MANIFEST.items():
        gate(Path(p).exists() and fsha(p) == h, f"MANIFEST DRIFT {p}")
    if not SMOKE:
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(sr.get("smoke") is True and sr.get("verdict") == "SMOKE OK",
             "SMOKE NOT GREEN")
        for pth, h in sr["start"]["file_sha256"].items():
            gate(fsha(pth) == h, f"SMOKE STALE {pth}")
    START = start_provenance(
        ["scratch/mathworld1_prband2prod.py",
         "scratch/mathworld1_prband2desk.py",
         "scratch/mathworld1_prband.py",
         "scratch/mathworld1_svpdiet3.py", "scratch/mathworld1_svpdiet.py",
         "scratch/mathworld1_svpdiet2.py", "scratch/mathworld1_svpnuisdesk.py",
         "scratch/mathworld1_svpchal.py", "scratch/mathworld1_svpchal2.py",
         "scratch/mathworld1_svpeval.py", "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py", "scratch/mathworld1_svpforder.py",
         "scratch/mathworld1_actiontok.py", "llmopt/search/derivation.py",
         "llmopt/search/rules.py", "llmopt/lab/provenance.py"])
    t0 = time.monotonic()
    # ---- bank gates ---------------------------------------------------
    used_polys = set()
    for grp in (F3_TRAIN_POLYS, F3_EVAL_POLYS, E1_F3_POLYS, N_KP, K_POLYS2,
                K_A, K_B, (X, 7 * X**3, 2 * X, 5 * X**2, 3 * X, 11 * X**4),
                (21 * X**3,)):
        used_polys |= {sp.sstr(e) for e in grp}
    for K in K_BANK:
        gate(sp.sstr(K) not in used_polys, f"K USED {K}")
    used_c = (set(F3_TRAIN_C) | set(F3_EVAL_C) | set(E1_F3_C) | set(N_C_IN)
              | set(N_C_OUT) | set(C_IN2) | set(C_OUT2) | set(C_SEC)
              | set(BURNED_F3_C) | {11, 13, 17, 19})
    gate(used_c == set(range(2, 26)), f"C UNION {sorted(used_c)}")
    for c in C_BANK:
        gate(c % 2 == 1 and c not in used_c, f"C BAD {c}")
    support = support_check()

    # ---- horizon + BURN FIRST ------------------------------------------
    hz = smoke_horizon() if SMOKE else build_horizon()
    if not SMOKE:
        gate(len(hz) == 144, "HZ")
        gate(len({b["pair_key"] for b in hz}) == 72, "PAIR KEYS")
    sigs = [b["base_signature"] for b in hz]
    gate(len(set(sigs)) == len(sigs), "SIG DUP")
    gate(len({sha(s) for s in sigs}) == len(sigs), "SIG_SHA DUP")
    gate(len({sp.srepr(b["f"]) for b in hz}) == len(hz), "SREPR DUP")
    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "horizon.jsonl", "w") as fh:
        for i, b in enumerate(hz):
            fh.write(json.dumps({"index": i, **{k: v for k, v in b.items()
                                                if k != "f"},
                                 "f": sp.sstr(b["f"]),
                                 "f_args": [sp.sstr(t) for t in b["f"].args],
                                 "n_terms": len(b["f"].args)}) + "\n")
    parents = []
    with open(OUTDIR / "parents.jsonl", "w") as fh:
        for i, b in enumerate(hz):
            for vt, D in VARIANTS:
                cur = sp.sstr(sp.Add(sp.Integral(b["f"], X),
                                     sp.Integral(D, X)))
                parents.append(cur)
                fh.write(json.dumps({"index": i, "variant": vt,
                                     "pair_key": b["pair_key"],
                                     "theta": b["theta"], "cur": cur,
                                     "cur_sha": sha(cur)}) + "\n")
    receipt = {"smoke": SMOKE, "prereg": PREREG,
               "prereg_commit": PREREG_COMMIT, "n_bases": len(hz),
               "n_burned_parents": len(parents),
               "n_unique_parents": len(set(parents)),
               "bank": {"e_lo": E_LO, "e_hi": E_HI, "theta": THETA,
                        "c": C_BANK, "K": [sp.sstr(k) for k in K_BANK],
                        "w": sp.sstr(W)},
               "targets": {"A0": list(A0), "B0": list(B0),
                           "A0_factor": factor_symbols(*A0),
                           "B0_factor": factor_symbols(*B0),
                           "witness_id": WITNESS_ID, "witness": WITNESS,
                           "companion": {"A1": list(A1), "B1": list(B1),
                                         "id": COMP_ID, "literal": COMP}},
               "mapping": EXPECT, "gap": GAP, "support": support,
               "manifest": {p: fsha(p) for p in MANIFEST}}

    def finish(verdict, extra):
        receipt["verdict"] = verdict
        receipt.update(extra)
        receipt["wall_s"] = round(time.monotonic() - t0, 1)
        receipt["artifact_sha256"] = {
            f.name: fsha(f) for f in sorted(OUTDIR.glob("*.json*"))
            if f.name != "prband2prod_receipt.json"}
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "prband2prod_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ("start", "manifest")}, indent=1),
              flush=True)

    if len(set(parents)) != len(parents):
        finish("INSTRUMENT FAILURE", {"reason": "duplicate parent"})
        raise SystemExit("INSTRUMENT FAILURE: duplicate parent")

    # ---- overlap over ALL parents -------------------------------------
    burned, train_targets = burned_sets()
    ov = {k: sum(c2 in s for c2 in parents) for k, s in burned.items()}
    ov["train_targets"] = sum(sp.sstr(b["f"]) in train_targets for b in hz)
    (OUTDIR / "overlap_receipt.json").write_text(json.dumps(
        {"hits": ov, "n_checked": len(parents),
         "burned_set_sizes": {k: len(v) for k, v in burned.items()}},
        indent=1))
    receipt["overlap_hits"] = ov
    receipt["burned_set_sizes"] = {k: len(v) for k, v in burned.items()}

    # ---- qualify all parents ------------------------------------------
    rows = {}
    rf = open(OUTDIR / "rows.jsonl", "w")
    fail_census = Counter()
    for i, b in enumerate(hz):
        for vt, D in VARIANTS:
            row, why = qualify_parent(b["f"], D)
            if why:
                rec = {"index": i, "variant": vt, "cur": row["cur"],
                       "fail": why}
                fail_census[f"{vt}:{why}"] += 1
            else:
                rec = {"index": i, "variant": vt, "fail": None,
                       **anatomy(b, D, row)}
            rec.update({"pair_key": b["pair_key"], "pair_sha": b["pair_sha"],
                        "theta": b["theta"], "e_hi": b["e_hi"], "c": b["c"],
                        "K": b["K"], "distractor": sp.sstr(D),
                        "n_terms": len(b["f"].args),
                        "parent_ops": int(sp.count_ops(
                            sp.Add(sp.Integral(b["f"], X),
                                   sp.Integral(D, X))))})
            rows[(i, vt)] = rec
            rf.write(json.dumps(rec) + "\n")
        print(f"[base {i + 1}/{len(hz)}]", flush=True)
    rf.close()
    receipt["qualification_fail_census"] = dict(fail_census)

    # ---- pairs ------------------------------------------------------------
    by_key = defaultdict(dict)
    for i, b in enumerate(hz):
        by_key[b["pair_key"]][b["theta"]] = i
    pairs = []
    for pk, arms in by_key.items():
        prec = {"pair_key": pk, "pair_sha": sha(pk), "arms": arms,
                "block_ok": True, "arm_gates": {}, "eligible": False,
                "companion": {}}
        if set(arms) != set(THETA):
            prec["block_ok"] = False
            prec["reasons"] = ["missing_arm"]
            pairs.append(prec)
            continue
        reasons = []
        for th in THETA:
            i = arms[th]
            for vt, _D in VARIANTS:
                if rows[(i, vt)]["fail"]:
                    prec["block_ok"] = False
                    reasons.append(f"{th}:{vt}:{rows[(i, vt)]['fail']}")
        if prec["block_ok"]:
            for th in THETA:
                v = rows[(arms[th], "after")]
                ok, why, gap, side = arm_gates(v, A0, B0, WITNESS_ID,
                                               WITNESS, EXPECT[th])
                prec["arm_gates"][th] = {"ok": ok, "why": why, "gap": gap,
                                         "teacher_side": side}
                if not ok:
                    reasons += [f"{th}:{w}" for w in why]
            sigs2 = {rows[(arms[th], "after")]["cand_sig_id"] for th in THETA}
            if len(sigs2) != 1:
                reasons.append("arms_signature_differ")
            curs2 = [rows[(arms[th], "after")]["cur"] for th in THETA]
            if any(parents.count(c2) != 1 for c2 in curs2):
                reasons.append("parent_not_unique")
        prec["reasons"] = reasons
        prec["eligible"] = prec["block_ok"] and not reasons \
            and (SMOKE or not any(ov.values()))  # smoke: overlap report-only
        # companion (smallA) gates — never affects eligibility
        if prec["block_ok"]:
            cw = []
            for th in THETA:
                v = rows[(arms[th], "smallA")]
                ok, why, gap, side = arm_gates(v, A1, B1, COMP_ID, COMP,
                                               EXPECT[th])
                prec["companion"][th] = {"ok": ok, "why": why, "gap": gap,
                                         "teacher_side": side}
                if not ok:
                    cw += [f"{th}:{w}" for w in why]
            csig = {rows[(arms[th], "smallA")]["cand_sig_id"] for th in THETA}
            if len(csig) != 1:
                cw.append("arms_signature_differ")
            prec["companion_available"] = not cw
            prec["companion_reasons"] = cw
        else:
            prec["companion_available"] = False
            prec["companion_reasons"] = ["block_failed"]
        pairs.append(prec)
    with open(OUTDIR / "pairs.jsonl", "w") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    eligible = sorted([p for p in pairs if p["eligible"]],
                      key=lambda p: p["pair_sha"])
    n_el = len(eligible)
    reserve = ("FULL-RESERVE" if n_el == 72 else "PARTIAL-RESERVE"
               if n_el >= 48 else "NO-FIRE")
    receipt.update({"n_pair_keys": len(pairs), "n_pairs_eligible": n_el,
                    "reserve_class": reserve,
                    "pair_fail_census": dict(Counter(
                        r for p in pairs for r in p.get("reasons", []))),
                    "eligible_ordering": [p["pair_sha"] for p in eligible]})
    if n_el < N_SELECT:
        finish("NO-FIRE", {})
        return 0
    selected = eligible[:N_SELECT]
    receipt["selected_pair_shas"] = [p["pair_sha"] for p in selected]
    receipt["selected_pair_keys"] = [p["pair_key"] for p in selected]

    # ---- primary + companion artifacts --------------------------------
    def emit(p, th, vt, role):
        v = rows[(p["arms"][th], vt)]
        return {"pair_key": p["pair_key"], "pair_sha": p["pair_sha"],
                "pair_id": p["pair_sha"][:16], "theta": th, "role": role,
                "variant": vt, "block_id": f"{role}-{p['pair_sha'][:12]}-{th}",
                **{k: v[k] for k in ("cur", "cur_sha", "parent_srepr_sha",
                                     "n_candidates", "min_hce_ties",
                                     "teacher", "teacher_hce", "cand_sig",
                                     "cand_sig_id", "cand_tuples",
                                     "unprod_set", "prompt_tokens",
                                     "candidates", "e_hi", "c", "K",
                                     "distractor", "n_terms", "parent_ops")},
                "gold_class": ("A" if th == "SIN_LOW" else "B"),
                "gold_tuple": list(A0 if th == "SIN_LOW" else B0)
                if role == "primary" else list(A1 if th == "SIN_LOW" else B1),
                "gap": (p["arm_gates"] if role == "primary"
                        else p["companion"])[th]["gap"],
                "total_tokens": v["prompt_tokens"] + 9}
    primary = [emit(p, th, "after", "primary") for p in selected
               for th in THETA]
    for r in primary:
        gate(tuple(r["teacher"]) == tuple(r["gold_tuple"]), "GOLD")
    companion = [emit(p, th, "smallA", "companion") for p in selected
                 if p["companion_available"] for th in THETA]
    with open(OUTDIR / "primary.jsonl", "w") as fh:
        for r in primary:
            fh.write(json.dumps(r) + "\n")
    with open(OUTDIR / "companion.jsonl", "w") as fh:
        for r in companion:
            fh.write(json.dumps(r) + "\n")
    # ---- theorems: 24 fixed orders ------------------------------------
    codes = [tuple(t) for t in WITNESS]
    perms = []
    for order in itertools.permutations(codes):
        rank = {c2: k for k, c2 in enumerate(order)}
        correct = [rank[tuple(r["gold_tuple"])] == 0 for r in primary]
        both = 0
        for k in range(0, len(primary), 2):
            both += correct[k] and correct[k + 1]
        perms.append({"order": [list(c2) for c2 in order],
                      "top1": sum(correct), "both_correct_pairs": both})
    max_top1 = max(p["top1"] for p in perms)
    max_both = max(p["both_correct_pairs"] for p in perms)
    (OUTDIR / "permutations.json").write_text(json.dumps(
        {"n_orders": len(perms), "max_top1": max_top1, "n": len(primary),
         "max_both_correct_pairs": max_both, "n_pairs": len(primary) // 2,
         "top1_census": dict(Counter(p["top1"] for p in perms)),
         "orders": perms}, indent=1))
    # ---- nuisance receipt ---------------------------------------------
    nuis = {"pairs": [], "arm_stats": {}}
    for k in range(0, len(primary), 2):
        a, b = primary[k], primary[k + 1]
        eq = {f: a[f] == b[f] for f in ("e_hi", "c", "K", "distractor",
                                        "cand_sig_id", "n_candidates",
                                        "n_terms")}
        eq["site"] = (a["teacher"][2] == b["teacher"][2] == 0)
        eq["targets"] = (sorted(map(tuple, a["cand_tuples"]))
                         == sorted(map(tuple, b["cand_tuples"])))
        gate(all(eq.values()) and a["n_terms"] == 6, f"NUISANCE {a['pair_id']}")
        nuis["pairs"].append({"pair_id": a["pair_id"], "equal": eq,
                              "parent_ops": {a["theta"]: a["parent_ops"],
                                             b["theta"]: b["parent_ops"]},
                              "prompt_tokens": {a["theta"]: a["prompt_tokens"],
                                                b["theta"]: b["prompt_tokens"]}})
    for th in THETA:
        arm = [r for r in primary if r["theta"] == th]
        nuis["arm_stats"][th] = {
            "parent_ops": {"min": min(r["parent_ops"] for r in arm),
                           "max": max(r["parent_ops"] for r in arm),
                           "mean": round(sum(r["parent_ops"] for r in arm)
                                         / len(arm), 3)},
            "prompt_tokens": {"min": min(r["prompt_tokens"] for r in arm),
                              "max": max(r["prompt_tokens"] for r in arm),
                              "mean": round(sum(r["prompt_tokens"] for r in arm)
                                            / len(arm), 3)},
            "total_tokens_C_and_PF": {"max": max(r["total_tokens"]
                                                 for r in arm)}}
    d_ops = [n["parent_ops"]["COS_LOW"] - n["parent_ops"]["SIN_LOW"]
             for n in nuis["pairs"]]
    d_tok = [n["prompt_tokens"]["COS_LOW"] - n["prompt_tokens"]["SIN_LOW"]
             for n in nuis["pairs"]]
    nuis["within_pair_offset_COS_minus_SIN"] = {
        "parent_ops": dict(Counter(d_ops)), "prompt_tokens": dict(Counter(d_tok))}
    (OUTDIR / "nuisance.json").write_text(json.dumps(nuis, indent=1))
    gaps = Counter(r["gap"] for r in primary)
    summary = {"n_primary": len(primary),
               "gold_census": dict(Counter(r["gold_class"] for r in primary)),
               "signature_census": dict(Counter(r["cand_sig_id"] for r in primary)),
               "gap_census": {str(k): v for k, v in gaps.items()},
               "ties_census": dict(Counter(r["min_hce_ties"] for r in primary)),
               "max_top1_fixed_order": max_top1,
               "max_both_correct_pairs": max_both,
               "top1_census": dict(Counter(p["top1"] for p in perms)),
               "companion_pairs_available": sum(
                   1 for p in selected if p["companion_available"]),
               "companion_n": len(companion),
               "companion_signature_census": dict(Counter(
                   r["cand_sig_id"] for r in companion)),
               "companion_gap_census": {str(k): v for k, v in Counter(
                   r["gap"] for r in companion).items()},
               "nuisance_offsets": nuis["within_pair_offset_COS_minus_SIN"],
               "arm_stats": nuis["arm_stats"],
               "max_total_tokens": max(r["total_tokens"] for r in primary)}
    if SMOKE:
        finish("SMOKE OK", summary)
        return 0
    if max_top1 > len(primary) // 2 or max_both > 0:
        finish("INSTRUMENT FAILURE", summary | {"reason": "theorem"})
        raise SystemExit("INSTRUMENT FAILURE: theorem")
    ok = (n_el >= 48 and len(primary) == 96
          and summary["gold_census"] == {"A": 48, "B": 48}
          and max_top1 == 48 and max_both == 0 and not any(ov.values()))
    finish("QUALIFIED SUPPORT-SUBSTANTIAL MATCHED-PAIR POPULATION "
           "MATERIALIZED" if ok else "INSTRUMENT FAILURE", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())

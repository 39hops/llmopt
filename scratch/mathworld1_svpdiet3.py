"""MATH-CYBER-1 SVP-GRID-IN-SUPPORT-CONFIRMATORY-DESIGN-0 —
materialize the fresh successor confirmatory eval envelope with
polynomial-form nuisance held INSIDE demonstrated training
support (the CH-F3-NUISANCE-DESK-0 mechanism answer), plus the
independent P-OUT covered-robustness artifact. ZERO training,
ZERO scoring, ZERO checkpoint access; seed-16001 is a later GO.
The old sealed heldout stays retired and is not named here.

PRIMARY EVAL (CH-F3 only, matched strict blocks): P bank = 12
trained P forms, each represented in the frozen augmentation in
ALL SIX covered cells (census-gated at run start); both
frequency regimes as a frozen nuisance stratum (c-IN from
trained support, c-OUT = 20..25); fresh parent identities via a
fresh inert-addend axis (k in {1,2} over fresh polys 7*x**2,
8*x**6). Horizon 12P x 2T x 6c x 2w x 2k = 576 per regime,
1,152 bases; verbatim DESIGN-1 block law (3 distractor
variants, all-pass, teacher i_unprod (I, term_index), single
term, ordinals [1,1,0]); strata {t2,t3} x {c-IN,c-OUT}; fill
gate >= 24 each else EVAL NO-FIRE; selection first 24 by
sig_sha per stratum (no P quota — P census reported); D_before
12/12 per stratum. Split: covered_calibration.jsonl = 96
covered-I0 strict states; heldout_test16.jsonl = 96 heldout-I1
+ 96 robustness-I1, SEALED for the future seed-16001 protocol.

SECONDARY ARTIFACT (independent; can never modify the strict
verdict): pout_robustness.jsonl — fresh NOVEL-DEGREE-CLASS
covered I0 states: DEGREE-11 two-term monic P forms
(x^11 + x^n, n = 1..10; shape-matched to the burned failed
degree-10 class, degree extended — the two-term degree-10 space
is exhausted and the design audit measured that three-term
degree-10 P shifts the teacher term entirely out of t2),
after-variant only, mixed frequency bank (4 c-IN + 4 c-OUT);
horizon 10P x 2T x 8c x 2w x 2k = 640; select first 48 I0/t2 by
SHA256(cur) (PRIMARY stratum, hard fill gate >= 48 else
SECONDARY NO-FIRE) + first 24 I0/t3 (prespecified SOFT control
stratum: if fewer than 24 qualify, the control is reported over
the qualified population with its n and flagged SHORT — never
fails the secondary).

NOVELTY (refute-on-hit): every emitted cur novel v natural
training, all bands, pilot, D0/D1 challenge, DIET train/eval,
EVAL-DESIGN-1, and the NUISANCE-DESK horizon (all three
distractor variants of each burned base); selected target
integrands disjoint from the 4,536 train-side targets; primary
and secondary populations mutually disjoint.

SMOKE (SVPD3_SMOKE=1, path-isolated under
logs/mathworld1/svpdiet3_smoke/): burned DESIGN-0 CH-F3 bases,
N=1 per term {2,3} with regime gates report-only, plus a
6-base secondary slice (secondary fires in smoke iff >= 1
scored row); novelty report-only; production requires the
smoke verdict ENVELOPE MATERIALIZED.

VERDICT LABELS (exhaustive): ENVELOPE MATERIALIZED / PRIMARY
MATERIALIZED, SECONDARY NO-FIRE / EVAL NO-FIRE, SECONDARY
MATERIALIZED / EVAL NO-FIRE, SECONDARY NO-FIRE / GATE-REFUTED.

Outputs (production, refuse-if-exists) under
logs/mathworld1/svpdiet3/: eval_blocks.jsonl,
covered_calibration.jsonl, heldout_test16.jsonl,
pout_attempts.jsonl, pout_robustness.jsonl,
svpdiet3_receipt.json (artifact shas on every VERDICT exit
path; entry gates exit non-zero without a receipt).

    SVPD3_SMOKE=1 .venv/bin/python scratch/mathworld1_svpdiet3.py
    .venv/bin/python scratch/mathworld1_svpdiet3.py         (Mac)
"""
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import (AFTER_D,  # noqa: E402
                                        SMALL_D, X, fsha,
                                        qualify_parent)
from scratch.mathworld1_svpchal import \
    build_horizon as d0_horizon  # noqa: E402
from scratch.mathworld1_svpchal2 import \
    build_horizon1 as d1_horizon  # noqa: E402
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
from scratch.mathworld1_svpnuisdesk import (C_IN as N_C_IN,  # noqa: E402
                                            C_OUT as N_C_OUT,
                                            K_POLYS as N_KP,
                                            P_IN as N_P_IN,
                                            P_OUT as N_P_OUT,
                                            build_cell)

SMOKE = os.environ.get("SVPD3_SMOKE") == "1"
N_PER_STRATUM = 1 if SMOKE else 24
N_SEC = 1 if SMOKE else 48
OUTDIR = Path("logs/mathworld1/svpdiet3_smoke" if SMOKE
              else "logs/mathworld1/svpdiet3")
SMOKE_RECEIPT = Path(
    "logs/mathworld1/svpdiet3_smoke/svpdiet3_receipt.json")
PAIRED = Path("data/matsub_paired.jsonl")
AUG = "logs/mathworld1/svpdiet/balanced_grid_train.jsonl"
PINS = {
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
        "d8402351d468e8",
    AUG:
        "0ef3d8a880a7e07712d8de757bc1670df12701e487b856b44c97"
        "f8db16cb3759",
    "logs/mathworld1/svpdiet/train_blocks.jsonl":
        "dd5e72391db557049e45efe6c6b5aff2118c8ea24ef53c85496b"
        "04f8539159d5",
    "logs/mathworld1/svpchal/blocks.jsonl":
        "21e7e635244574266ec876c1c8c76f8d7d2a77e20c2f6680d3ee"
        "48db372c4d10",
    "logs/mathworld1/svpnuisdesk/attempts.jsonl":
        "54659cda39d0578c507d984aae7fdcb90c9d3cc86bf3b5a82fed"
        "740be93b5504",
}

# ---- frozen banks (this GO) ---------------------------------
P12 = (X**8 + X**5, X**7, X**8 + X**7, X**7 + X**5,
       X**7 + X**6, X**7 + X**4, X**8, X**8 + X**3,
       X**9, X**5 + X**4, X**6 + X**5, X**6 + X**4)
C_IN2 = (9, 11, 12, 16, 18, 19)
C_OUT2 = (20, 21, 22, 23, 24, 25)
K_POLYS2 = (7 * X**2, 8 * X**6)
WS = (sp.exp(X) / X, sp.sin(X) / X)
P_SEC = tuple(X**11 + X**n for n in range(1, 11))
C_SEC = (9, 11, 16, 18, 20, 21, 23, 25)
N_SEC_T3 = 24  # soft control stratum (see prereg)


def build_f3(tag, Ps, Cs, kpolys):
    out = []
    for P in Ps:
        for T in (sp.sin, sp.cos):
            for c in Cs:
                for w in WS:
                    for k in (1, 2):
                        f = (sp.expand(sp.diff(P * T(c * X), X))
                             + sp.Integral(w, X)
                             + sp.Add(*kpolys[:k]))
                        sig = (f"D3|{tag}|P={P}|T={T.__name__}"
                               f"|c={c}|w={w}|k={k}")
                        out.append((tag, sig, f, sp.sstr(P),
                                    c))
    return out


def smoke_slice():
    rows = [json.loads(l) for l in
            open("logs/mathworld1/svpchal/blocks.jsonl")]
    strata = defaultdict(list)
    for r in rows:
        if (r.get("fail") is None and r["family"] == "CH-F3"
                and r.get("term") in (2, 3)):
            strata[r["term"]].append(r)
    want = set()
    for t in (2, 3):
        for r in sorted(strata[t],
                        key=lambda r: r["sig_sha"])[:2]:
            want.add(r["base_signature"])
    out = [("SMOKE", sig, f, "d0", 0)
           for fam, sig, f in d0_horizon() if sig in want]
    gate(len(out) == 4, f"SMOKE SLICE {len(out)}")
    return out


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    if not SMOKE:
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(sr.get("smoke") is True and
             sr.get("verdict") == "ENVELOPE MATERIALIZED",
             "SMOKE NOT GREEN")
        for pth, h in sr["start"]["file_sha256"].items():
            gate(fsha(pth) == h, f"SMOKE STALE {pth}")
    START = start_provenance(
        ["scratch/mathworld1_svpdiet3.py",
         "scratch/mathworld1_svpnuisdesk.py",
         "scratch/mathworld1_svpdiet.py",
         "scratch/mathworld1_svpdiet2.py",
         "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpchal2.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "llmopt/search/derivation.py",
         "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    # P12: every form present in the augmentation in ALL SIX
    # covered cells (census-gated, derived from the pinned file)
    aug_cells = defaultdict(set)
    for l in open(AUG):
        r = json.loads(l)
        if r["family"] != "CH-F3":
            continue
        m = re.search(r"\|P=([^|]+)\|", r["base_signature"])
        aug_cells[m.group(1)].add(
            (r["site_ordinal"], r["param_index"]))
    for P in P12:
        gate(len(aug_cells.get(sp.sstr(P), set())) == 6,
             f"P NOT FULLY REPRESENTED {P}")
        gate(P in set(F3_TRAIN_P), f"P NOT TRAINED {P}")
    for c in C_IN2:
        gate(c in set(F3_TRAIN_C), f"C_IN NOT TRAINED {c}")
    for c in C_OUT2:
        gate(c in set(E1_F3_C), f"C_OUT NOT EVAL-CLASS {c}")
    used_polys = set()
    for grp in (F3_TRAIN_POLYS, F3_EVAL_POLYS, E1_F3_POLYS,
                N_KP, (X, 7 * X**3, 2 * X, 5 * X**2, 3 * X,
                       11 * X**4)):
        used_polys |= {sp.sstr(e) for e in grp}
    for pv in K_POLYS2:
        gate(sp.sstr(pv) not in used_polys, f"K USED {pv}")
    used_p = set(F3_TRAIN_P) | set(F3_EVAL_P) | set(E1_F3_P) \
        | set(N_P_OUT)
    for P in P_SEC:
        gate(P not in used_p, f"P_SEC USED {P}")
        gate(sp.degree(P) == 11, f"P_SEC DEGREE {P}")
    for c2 in C_SEC[:4]:
        gate(c2 in set(F3_TRAIN_C), f"C_SEC IN {c2}")
    for c2 in C_SEC[4:]:
        gate(c2 in set(E1_F3_C), f"C_SEC OUT {c2}")

    # burned cur set
    train_cur = set()
    for l in open(PAIRED):
        train_cur.add(json.loads(l)["cur"])
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
    dd_cur = set()
    for hz, n in ((d0_horizon(), 720), (d1_horizon(), 2160),
                  (diet_hz, 15912), (e1_hz, 10224),
                  (nuis_hz, 3456)):
        before = len(dd_cur)
        for tup in hz:
            f = tup[2]
            for D in (SMALL_D[0], SMALL_D[1], AFTER_D):
                dd_cur.add(sp.sstr(sp.Add(sp.Integral(f, X),
                                          sp.Integral(D, X))))
        gate(len(dd_cur) - before == n,
             f"BURNED HORIZON {len(dd_cur) - before}")
    burned = train_cur | band_cur | pilot_cur | dd_cur
    train_targets = set()
    for l in open("logs/mathworld1/svpdiet/train_blocks.jsonl"):
        train_targets.add(json.loads(l)["target_integrand"])
    gate(len(train_targets) == 4536, "TRAIN TARGETS")

    if SMOKE:
        eval_hz = smoke_slice()
        sec_hz = build_f3("SEC", P_SEC, C_SEC, K_POLYS2)[:6]
    else:
        eval_hz = (build_f3("IN", P12, C_IN2, K_POLYS2)
                   + build_f3("OUT", P12, C_OUT2, K_POLYS2))
        sec_hz = build_f3("SEC", P_SEC, C_SEC, K_POLYS2)
        gate(len(eval_hz) == 1152, f"EVAL HZ {len(eval_hz)}")
        gate(len(sec_hz) == 640, f"SEC HZ {len(sec_hz)}")
        allf = [sp.srepr(h[2]) for h in eval_hz + sec_hz]
        gate(len(allf) == len(set(allf)), "TARGET DUP")
        sigs = [h[1] for h in eval_hz + sec_hz]
        gate(len(sigs) == len(set(sigs)), "SIG DUP")
    OUTDIR.mkdir(parents=True)
    receipt = {"smoke": SMOKE, "prereg":
               "MATH-CYBER-1-SVP-GRID-IN-SUPPORT-CONFIRMATORY-"
               "DESIGN-0",
               "n_eval_horizon": len(eval_hz),
               "n_sec_horizon": len(sec_hz),
               "N_per_stratum": N_PER_STRATUM,
               "N_sec_t2": N_SEC, "N_sec_t3": N_SEC_T3,
               "smoke_receipt_sha": (fsha(SMOKE_RECEIPT)
                                     if not SMOKE else None),
               "p12_aug_cell_coverage": {
                   sp.sstr(P): len(aug_cells[sp.sstr(P)])
                   for P in P12},
               "pins": {p: fsha(p) for p in PINS}}

    def finish(verdict, extra):
        receipt["verdict"] = verdict
        receipt.update(extra)
        receipt["artifact_sha256"] = {
            f.name: fsha(f) for f in sorted(
                OUTDIR.glob("*.jsonl")) if f.exists()}
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "svpdiet3_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ("start", "pins")},
                         indent=1), flush=True)

    t_all = time.monotonic()

    def refuted(reason, extra):
        extra["wall_s"] = round(time.monotonic() - t_all, 1)
        finish("GATE-REFUTED",
               extra | {"refuted_reason": reason})
        raise SystemExit(f"GATE-REFUTED: {reason}")

    # ---- phase 1: PRIMARY eval blocks (3-variant law) -------
    eb = open(OUTDIR / "eval_blocks.jsonl", "w")
    eblocks = []
    for i, (tag, sig, f_t, Ps, c) in enumerate(eval_hz):
        variants = []
        fail = None
        for vtag, D in (("smallA", SMALL_D[0]),
                        ("smallB", SMALL_D[1]),
                        ("after", AFTER_D)):
            row, why = qualify_parent(f_t, D)
            row["variant"] = vtag
            row["distractor"] = sp.sstr(D)
            variants.append(row)
            if why:
                fail = f"{vtag}:{why}"
                break
        blk = {"regime": tag, "base_signature": sig,
               "P": Ps, "c": c,
               "target_integrand": sp.sstr(f_t),
               "sig_sha": hashlib.sha256(
                   sig.encode()).hexdigest(),
               "fail": fail}
        if fail is None:
            rules = {v["chosen_rule"] for v in variants}
            kinds = {(v["chosen_site_kind"],
                      v["chosen_param_kind"])
                     for v in variants}
            terms = {v["chosen_term"] for v in variants}
            ords = [v["chosen_ordinal"] for v in variants]
            if kinds != {("I", "term_index")}:
                blk["fail"] = f"teacher_kind:{sorted(kinds)}"
            elif rules != {"i_unprod"}:
                blk["fail"] = f"teacher_rule:{sorted(rules)}"
            elif len(terms) != 1:
                blk["fail"] = f"term_mismatch:{sorted(terms)}"
            elif ords != [1, 1, 0]:
                blk["fail"] = f"ordinals:{ords}"
            else:
                blk["term"] = terms.pop()
        blk["_variants"] = variants
        eblocks.append(blk)
        eb.write(json.dumps({k: v for k, v in blk.items()
                             if k != "_variants"}) + "\n")
        if (i + 1) % 50 == 0 or SMOKE:
            print(f"[eval {i + 1}/{len(eval_hz)}]", flush=True)
    eb.close()
    strata = defaultdict(list)
    for b in eblocks:
        if b["fail"] is None and b.get("term") in (2, 3):
            strata[(b["term"], b["regime"])].append(b)
    for k in strata:
        strata[k].sort(key=lambda b: b["sig_sha"])
    if SMOKE:
        skeys = [(2, "SMOKE"), (3, "SMOKE")]
    else:
        skeys = [(t, rg) for t in (2, 3)
                 for rg in ("IN", "OUT")]
    e_counts = {f"t{t}|{rg}": len(strata.get((t, rg), []))
                for (t, rg) in skeys}
    base = {
        "eval_stratum_qualified_counts": e_counts,
        "eval_block_fail_census": dict(Counter(
            b["fail"] for b in eblocks if b["fail"])),
        "eval_terms_census": dict(Counter(
            b.get("term") for b in eblocks
            if b["fail"] is None)),
        "wall_s": round(time.monotonic() - t_all, 1)}
    eshort = [k for k, v in e_counts.items()
              if v < N_PER_STRATUM]
    primary_ok = not eshort
    calib, heldout = [], []
    sel_census = Counter()
    ev_curs = []
    sel_targets = set()
    dupsig = set()
    if primary_ok:
        for (t, rg) in skeys:
            for j, b in enumerate(
                    strata[(t, rg)][:N_PER_STRATUM]):
                if b["base_signature"] in dupsig:
                    refuted("DUP SIG SELECTED", base)
                dupsig.add(b["base_signature"])
                sel_targets.add(b["target_integrand"])
                primary_small = ("smallA" if j % 2 == 0
                                 else "smallB")
                robust_small = ("smallB" if j % 2 == 0
                                else "smallA")
                by = {v["variant"]: v
                      for v in b["_variants"]}
                for v, role, sink in (
                        (by[primary_small], "heldout-I1",
                         heldout),
                        (by["after"], "covered-I0", calib),
                        (by[robust_small], "robustness-I1",
                         heldout)):
                    sink.append({
                        "block_id": f"t{t}-{rg}-{j:02d}",
                        "term_cell": t, "regime": rg,
                        "P": b["P"], "c": b["c"],
                        "site_role": role,
                        "primary": "robust" not in role,
                        "confirmatory_denominator":
                            role == "heldout-I1",
                        "distractor": v["distractor"],
                        "block_d_before": by[primary_small][
                            "distractor"],
                        "base_signature":
                            b["base_signature"],
                        "cur": v["cur"],
                        "parent_srepr_sha":
                            v["parent_srepr_sha"],
                        "n_candidates": v["n_candidates"],
                        "min_hce_ties": v["min_hce_ties"],
                        "chosen_ordinal":
                            v["chosen_ordinal"],
                        "chosen_term": v["chosen_term"],
                        "candidates": v["candidates"]})
                    ev_curs.append(v["cur"])
                sel_census[f"t{t}|{rg}|"
                           f"{by[primary_small]['distractor']}"
                           ] += 1
        half = N_PER_STRATUM - N_PER_STRATUM // 2
        for (t, rg) in skeys:
            a2 = sel_census.get(f"t{t}|{rg}|x**x", 0)
            bd = sel_census.get(
                f"t{t}|{rg}|1/(x + log(x))", 0)
            if not (a2 == half
                    and bd == N_PER_STRATUM // 2):
                refuted(f"D BALANCE t{t} {rg} {a2}/{bd}",
                        base)
        if len(ev_curs) != len(set(ev_curs)):
            refuted("DUP EVAL PARENT", base)
        env = {"natural": sum(c2 in train_cur
                              for c2 in ev_curs),
               "band": sum(c2 in band_cur for c2 in ev_curs),
               "pilot": sum(c2 in pilot_cur
                            for c2 in ev_curs),
               "designs_diet_nuis": sum(c2 in dd_cur
                                        for c2 in ev_curs)}
        tgt_hits = sum(1 for ti in sel_targets
                       if ti in train_targets)
        if SMOKE:
            base["smoke_novelty_hits_expected_burned"] = env
            base["smoke_target_overlap"] = tgt_hits
        else:
            if any(env.values()):
                refuted(f"EVAL BURNED OVERLAP {env}", base)
            if tgt_hits:
                refuted(f"TARGET OVERLAP {tgt_hits}", base)
        cal_census = Counter(r["site_role"] for r in calib)
        hld_census = Counter(r["site_role"] for r in heldout)
        if not SMOKE:
            if not (cal_census == Counter(
                    {"covered-I0": 96})
                    and hld_census == Counter(
                    {"heldout-I1": 96, "robustness-I1": 96})):
                refuted(f"ROLE CENSUS {dict(cal_census)} "
                        f"{dict(hld_census)}", base)
        with open(OUTDIR / "covered_calibration.jsonl",
                  "w") as fh:
            for r in calib:
                fh.write(json.dumps(r) + "\n")
        with open(OUTDIR / "heldout_test16.jsonl", "w") as fh:
            for r in heldout:
                fh.write(json.dumps(r) + "\n")
        base.update({
            "eval_blocks_selected": len(dupsig),
            "calibration_role_census": dict(cal_census),
            "heldout_role_census": dict(hld_census),
            "eval_d_before_census": dict(sel_census),
            "selected_target_overlap_with_train_targets":
                tgt_hits,
            "selected_P_census_by_stratum": {
                f"t{t}|{rg}": dict(Counter(
                    b["P"] for b in
                    strata[(t, rg)][:N_PER_STRATUM]))
                for (t, rg) in skeys}})
    else:
        base["short_strata"] = eshort

    # ---- phase 2: SECONDARY P-OUT artifact ------------------
    t_sec = time.monotonic()
    sec_pop = defaultdict(list)
    sec_census = Counter()
    pa = open(OUTDIR / "pout_attempts.jsonl", "w")
    for i, (tag, sig, f_t, Ps, c) in enumerate(sec_hz):
        row, why = qualify_parent(f_t, AFTER_D)
        rec = {"base_signature": sig, "P": Ps, "c": c,
               "fail": why}
        if why is None:
            tup = (row["chosen_rule"], row["chosen_site_kind"],
                   row["chosen_ordinal"],
                   row["chosen_param_kind"],
                   row["chosen_term"])
            rec["teacher"] = list(tup)
            if (tup[0] == "i_unprod" and tup[1] == "I"
                    and tup[2] == 1 and tup[4] in (2, 3)):
                pa.close()
                refuted("SEC I1 HELD-OUT LABEL", base)
            if (tup[0] == "i_unprod" and tup[1] == "I"
                    and tup[2] == 0 and tup[4] in (2, 3)):
                sec_census[f"sec_t{tup[4]}"] += 1
                sec_pop[tup[4]].append((sig, Ps, c, row))
            else:
                sec_census["sec_out_of_scope"] += 1
        else:
            sec_census["sec_fail"] += 1
        pa.write(json.dumps(rec) + "\n")
        if (i + 1) % 50 == 0 or SMOKE:
            print(f"[sec {i + 1}/{len(sec_hz)}]", flush=True)
    pa.close()
    base["sec_census"] = dict(sec_census)
    base["wall_sec_s"] = round(time.monotonic() - t_sec, 1)
    if SMOKE:
        sec_ok = (len(sec_pop[2]) + len(sec_pop[3])) >= 1
        sec_short = [] if sec_ok else ["smoke"]
    else:
        sec_short = []
        if len(sec_pop[2]) < N_SEC:
            sec_short.append("t2")
        t3_short = len(sec_pop[3]) < N_SEC_T3
        sec_ok = not sec_short
        base["sec_t3_control_short"] = t3_short
    if sec_ok:
        sec_rows = []
        for t, ncap in ((2, N_SEC), (3, N_SEC_T3)):
            rows2 = sorted(
                sec_pop[t],
                key=lambda r: hashlib.sha256(
                    r[3]["cur"].encode()).hexdigest())
            for (sig, Ps, c, row) in rows2[:ncap]:
                sec_rows.append({
                    "stratum": ("primary-t2" if t == 2
                                else "control-t3"),
                    "term_cell": t, "P": Ps, "c": c,
                    "regime": ("IN" if c in C_SEC[:4]
                               else "OUT"),
                    "base_signature": sig,
                    "cur": row["cur"],
                    "parent_srepr_sha":
                        row["parent_srepr_sha"],
                    "n_candidates": row["n_candidates"],
                    "min_hce_ties": row["min_hce_ties"],
                    "chosen_ordinal": row["chosen_ordinal"],
                    "chosen_term": row["chosen_term"],
                    "candidates": row["candidates"]})
        scurs = [r["cur"] for r in sec_rows]
        if len(scurs) != len(set(scurs)):
            refuted("SEC DUP CUR", base)
        sec_env = sum(c2 in burned for c2 in scurs) \
            + sum(c2 in set(ev_curs) for c2 in scurs)
        if SMOKE:
            base["smoke_sec_novelty_hits"] = sec_env
        elif sec_env:
            refuted(f"SEC BURNED/EVAL OVERLAP {sec_env}",
                    base)
        with open(OUTDIR / "pout_robustness.jsonl",
                  "w") as fh:
            for r in sec_rows:
                fh.write(json.dumps(r) + "\n")
        base["sec_selected"] = {
            "primary-t2": sum(1 for r in sec_rows
                              if r["term_cell"] == 2),
            "control-t3": sum(1 for r in sec_rows
                              if r["term_cell"] == 3),
            "regime_census": dict(Counter(
                f"t{r['term_cell']}|{r['regime']}"
                for r in sec_rows))}
    else:
        base["sec_short"] = sec_short

    base["wall_s"] = round(time.monotonic() - t_all, 1)
    if primary_ok and sec_ok:
        v = "ENVELOPE MATERIALIZED"
    elif primary_ok:
        v = "PRIMARY MATERIALIZED / SECONDARY NO-FIRE"
    elif sec_ok:
        v = "EVAL NO-FIRE / SECONDARY MATERIALIZED"
    else:
        v = "EVAL NO-FIRE / SECONDARY NO-FIRE"
    finish(v, base)
    return 0


if __name__ == "__main__":
    sys.exit(main())

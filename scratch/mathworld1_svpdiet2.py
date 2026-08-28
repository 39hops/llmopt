"""MATH-CYBER-1 SVP-GRID-EVAL-DESIGN-1 — EVAL-ONLY successor to
the booked SVP-GRID-DIET-DESIGN-0 EVAL NO-FIRE (CH-F4|t2 22 v
N=24). The frozen augmentation / manifest / batch plan /
calibration gate / scoring law / seeds are untouched; this GO
materializes ONLY the fresh calibration/heldout pair under the
verbatim matched-block law. ZERO model/checkpoint access.

FROZEN FRESH HORIZONS (priced from the now-burned DESIGN-0 eval
census as a descriptive rate, not an IID claim; >= 2x margin
over N=24 preferred and met on every stratum):
  CH-F3, 720 bases: P in {x^10, x^10+x, ..., x^10+x^9} (10
    fresh polys); T in {sin,cos}; c in {20..25}; w in
    {exp(x)/x, sin(x)/x}; k in {0,1,2} over fresh polys
    (14*x**3, 15*x**7). 10*2*6*2*3 = 720. Priced (burned
    DESIGN-0 eval rates 32/60/40 of 192): t1 120, t2 225,
    t3 150.
  CH-F4, 2,688 bases: (P1,P2) = 84 pairs fresh at PAIR grain
    (14 P1 rows x the 6 P2 columns; the P1 polys individually
    recur from prior F3 banks, as in the burned DESIGN-1
    banks); (a,b) in {(12,13),(13,12),(12,5),(5,12),(13,4),
    (4,13),(11,12),(12,11)}; (T1,T2) both orders; w 2 blockers.
    84*8*2*2 = 2,688. Priced (burned rates 56/22/62 of 576):
    t1 261, t2 103 (4.3x over 24, the binding stratum), t3 289.
  Margin law: every stratum survives the MEASURED adverse
  transport factor on this gate class (2.5x, DESIGN-0 eval
  CH-F4|t2); at the documented worst-case 4.3x factor every
  stratum still clears except CH-F4|t2, which sits at the gate
  edge (103/4.3 = 23.9 v 24) — accepted and stated; NO-FIRE is
  the protection.
All banks disjoint from every prior bank: pilot/DESIGN-0/
DESIGN-1 challenge banks, the DIET-DESIGN-0 train banks, and
the burned DIET-DESIGN-0 eval banks. No parameter value was
chosen for post-hoc t2 affinity; the banks extend the same
literal axes (higher-degree P family, next fresh c/freq
values).

LAW (verbatim DIET-DESIGN-0 eval phase, unchanged): three
distractor variants per base; block qualifies iff all three
pass qualify_parent, teacher (i_unprod, I, term_index), single
term, ordinals [1,1,0]; strata {family x term 1|2|3}; fill
gate >= 24 each else EVAL NO-FIRE (no lowering, pooling, or
substitution); selection first 24 by sig_sha; D_before
alternation 12/12; split covered_calibration.jsonl (96 strict
covered-I0 + 144 t1 controls) / heldout_test.jsonl (96
heldout-I1 + 96 robustness-I1, SEALED), independent shas.

NOVELTY (refute-on-hit): eval curs disjoint from natural
training, all three scored bands, 566 pilot parents, 720
DESIGN-0 + 2,160 DESIGN-1 challenge parents, and ALL 13,608 +
2,304 DIET-DESIGN-0 train/eval horizon parents (reconstructed
from the frozen banks). NEW GATE this GO: selected eval TARGET
INTEGRANDS disjoint from all 4,536 train-side constructor
target integrands (the full train horizon in
train_blocks.jsonl) in addition to the parent-level gates.

SMOKE (SVPDIET2_SMOKE=1, path-isolated under
logs/mathworld1/svpdiet2_smoke/): N=1 on burned DESIGN-0
challenge bases (first-by-sig_sha qualified base per
{family x term 1..3}); novelty gates report-only. Production
refuses unless the smoke receipt is green AND records this
driver's current sha.

Outputs (production, refuse-if-exists) under
logs/mathworld1/svpdiet2/: eval_blocks.jsonl,
covered_calibration.jsonl, heldout_test.jsonl,
svpdiet2_receipt.json (artifact shas on every VERDICT exit
path — NO-FIRE / GATE-REFUTED / MATERIALIZED; entry gates exit
non-zero without a receipt, before any byte).

    SVPDIET2_SMOKE=1 .venv/bin/python scratch/mathworld1_svpdiet2.py
    .venv/bin/python scratch/mathworld1_svpdiet2.py         (Mac)
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
                                        F4_EVAL_P1, F4_P2,
                                        F4_TRAIN_FREQS,
                                        F4_TRAIN_P1,
                                        BURNED_F3_P,
                                        BURNED_F3_C,
                                        BURNED_F3_POLYS,
                                        BURNED_F4_PAIRS,
                                        BURNED_FREQS,
                                        PILOT_RECEIPTS,
                                        f3_bases, f4_bases)

SMOKE = os.environ.get("SVPDIET2_SMOKE") == "1"
N_PER_STRATUM = 1 if SMOKE else 24
OUTDIR = Path("logs/mathworld1/svpdiet2_smoke" if SMOKE
              else "logs/mathworld1/svpdiet2")
SMOKE_RECEIPT = Path(
    "logs/mathworld1/svpdiet2_smoke/svpdiet2_receipt.json")
PAIRED = Path("data/matsub_paired.jsonl")
PINS = {
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
        "d8402351d468e8",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f316524140"
        "6a86253858c6",
    "logs/mathworld1/svpeval2/decisions.jsonl":
        "89efbe0ea447ee937c0c130d5419112921a2dd6c2159c6c2112c"
        "fd5e92f79315",
    "logs/mathworld1/svpeval3/decisions.jsonl":
        "2ff5433249622df9d421cf8014131b3907092a943040bb7b20f4"
        "6f1afffb7efa",
    "logs/mathworld1/svpchal/blocks.jsonl":
        "21e7e635244574266ec876c1c8c76f8d7d2a77e20c2f6680d3ee"
        "48db372c4d10",
    "logs/mathworld1/svpdiet/svpdiet_receipt.json":
        "519d38b8a725fe726285aadb34101c22e80eba1874a23936ea88"
        "b5438e4efbb7",
    "logs/mathworld1/svpdiet/train_blocks.jsonl":
        "dd5e72391db557049e45efe6c6b5aff2118c8ea24ef53c85496b"
        "04f8539159d5",
    "logs/mathworld1/svpdiet/eval_blocks.jsonl":
        "720863bc4955658fa039d6677043af40f645a91042a8e8402d92"
        "3afebe36ce93",
    "logs/mathworld1/svpdiet/balanced_grid_train.jsonl":
        "0ef3d8a880a7e07712d8de757bc1670df12701e487b856b44c97"
        "f8db16cb3759",
}

# ---- frozen fresh banks (this GO) ---------------------------
E1_F3_P = tuple(X**10 + X**n for n in range(1, 10)) + (X**10,)
E1_F3_C = (20, 21, 22, 23, 24, 25)
E1_F3_POLYS = (14 * X**3, 15 * X**7)
E1_F4_P1 = (X**8 + X**4, X**8 + X**5, X**8 + X**6, X**8 + X**7,
            X**7 + X**4, X**7 + X**5, X**7 + X**6,
            X**9, X**9 + X, X**9 + X**2, X**9 + X**3,
            X**9 + X**4, X**9 + X**5, X**9 + X**6)
E1_F4_FREQS = ((12, 13), (13, 12), (12, 5), (5, 12), (13, 4),
               (4, 13), (11, 12), (12, 11))


def smoke_slice():
    rows = [json.loads(l) for l in
            open("logs/mathworld1/svpchal/blocks.jsonl")]
    strata = defaultdict(list)
    for r in rows:
        if r.get("fail") is None and r.get("term") in (1, 2, 3):
            strata[(r["family"], r["term"])].append(r)
    want = set()
    for fam in ("CH-F3", "CH-F4"):
        for t in (1, 2, 3):
            want.add(sorted(strata[(fam, t)],
                            key=lambda r: r["sig_sha"]
                            )[0]["base_signature"])
    out = [(fam, sig, f) for fam, sig, f in d0_horizon()
           if sig in want]
    gate(len(out) == 6, f"SMOKE SLICE {len(out)}")
    return out


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    if not SMOKE:
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(sr.get("smoke") is True and sr.get("verdict") ==
             "EVAL MATERIALIZED", "SMOKE NOT GREEN")
        gate(sr["start"]["file_sha256"][
            "scratch/mathworld1_svpdiet2.py"] == fsha(
            "scratch/mathworld1_svpdiet2.py"), "SMOKE STALE")
    START = start_provenance(
        ["scratch/mathworld1_svpdiet2.py",
         "scratch/mathworld1_svpdiet.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpchal2.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    # bank freshness gates: disjoint from EVERY prior bank
    prior_f3_p = set(BURNED_F3_P) | set(F3_TRAIN_P) \
        | set(F3_EVAL_P)
    for P in E1_F3_P:
        gate(P not in prior_f3_p, f"USED F3 P {P}")
    prior_c = set(BURNED_F3_C) | set(F3_TRAIN_C) | set(F3_EVAL_C)
    for c in E1_F3_C:
        gate(c not in prior_c, f"USED F3 c {c}")
    prior_polys = set(BURNED_F3_POLYS) \
        | {sp.sstr(e) for e in F3_TRAIN_POLYS + F3_EVAL_POLYS}
    for pv in E1_F3_POLYS:
        gate(sp.sstr(pv) not in prior_polys, f"USED POLY {pv}")
    prior_p1 = set(F4_TRAIN_P1) | set(F4_EVAL_P1)
    prior_pairs = set(BURNED_F4_PAIRS) \
        | {(p1, p2) for p1 in prior_p1 for p2 in F4_P2}
    for P1 in E1_F4_P1:
        for P2 in F4_P2:
            gate((P1, P2) not in prior_pairs,
                 f"USED PAIR {(P1, P2)}")
    prior_freqs = set(BURNED_FREQS) | set(F4_TRAIN_FREQS) \
        | set(F4_EVAL_FREQS)
    for fq in E1_F4_FREQS:
        gate(fq not in prior_freqs, f"USED FREQ {fq}")

    # burn sets
    train_cur = set()
    seen = set()
    for l in open(PAIRED):
        r = json.loads(l)
        train_cur.add(r["cur"])
        seen.add((r["rule"], r["site_kind"], r["site_ordinal"],
                  r["param_kind"], r["param_index"]))
    gate(len(train_cur) <= 73324, "NATURAL CURS")
    for t in (2, 3):
        gate(("i_unprod", "I", 1, "term_index", t) not in seen,
             f"HELD-OUT IN TRAINING I1t{t}")
    for (o, t) in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
                   (1, 1)):
        gate(("i_unprod", "I", o, "term_index", t) in seen,
             f"COVERED CELL ABSENT I{o}t{t}")
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
        for a in json.loads(Path(pr).read_text())["attempts"]:
            pilot_cur.add(a["parent_sstr"])
    gate(len(pilot_cur) == 566, f"PILOT {len(pilot_cur)}")
    dd_cur = set()
    diet_train_hz = f3_bases("TRAIN", F3_TRAIN_P, F3_TRAIN_C,
                             F3_TRAIN_POLYS) \
        + f4_bases("TRAIN", F4_TRAIN_P1, F4_TRAIN_FREQS)
    diet_eval_hz = f3_bases("EVAL", F3_EVAL_P, F3_EVAL_C,
                            F3_EVAL_POLYS) \
        + f4_bases("EVAL", F4_EVAL_P1, F4_EVAL_FREQS)
    for hz, n in ((d0_horizon(), 720), (d1_horizon(), 2160),
                  (diet_train_hz, 13608), (diet_eval_hz, 2304)):
        before = len(dd_cur)
        for fam, sig, f in hz:
            for D in (SMALL_D[0], SMALL_D[1], AFTER_D):
                dd_cur.add(sp.sstr(sp.Add(sp.Integral(f, X),
                                          sp.Integral(D, X))))
        gate(len(dd_cur) - before == n,
             f"BURNED HORIZON {len(dd_cur) - before}")
    train_targets = set()
    for l in open("logs/mathworld1/svpdiet/train_blocks.jsonl"):
        train_targets.add(json.loads(l)["target_integrand"])
    gate(len(train_targets) == 4536,
         f"TRAIN TARGETS {len(train_targets)}")

    if SMOKE:
        eval_hz = smoke_slice()
    else:
        eval_hz = (f3_bases("E1", E1_F3_P, E1_F3_C, E1_F3_POLYS)
                   + f4_bases("E1", E1_F4_P1, E1_F4_FREQS))
        gate(len(eval_hz) == 3408, f"EVAL HZ {len(eval_hz)}")
        sigs = [s for _, s, _ in eval_hz]
        gate(len(sigs) == len(set(sigs)), "SIG DUP")
        fts = [sp.srepr(f) for _, _, f in eval_hz]
        gate(len(fts) == len(set(fts)), "TARGET DUP")
    OUTDIR.mkdir(parents=True)
    receipt = {"smoke": SMOKE, "prereg":
               "MATH-CYBER-1-SVP-GRID-EVAL-DESIGN-1",
               "N_per_stratum": N_PER_STRATUM,
               "n_eval_horizon": len(eval_hz),
               "smoke_receipt_sha": (fsha(SMOKE_RECEIPT)
                                     if not SMOKE else None),
               "pins": {p: fsha(p) for p in PINS}}

    def finish(verdict, extra):
        receipt["verdict"] = verdict
        receipt.update(extra)
        receipt["artifact_sha256"] = {
            f.name: fsha(f) for f in sorted(
                OUTDIR.glob("*.jsonl")) if f.exists()}
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "svpdiet2_receipt.json").write_text(
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

    eb = open(OUTDIR / "eval_blocks.jsonl", "w")
    eblocks = []
    for i, (fam, sig, f_t) in enumerate(eval_hz):
        variants = []
        fail = None
        for tag, D in (("smallA", SMALL_D[0]),
                       ("smallB", SMALL_D[1]),
                       ("after", AFTER_D)):
            row, why = qualify_parent(f_t, D)
            row["variant"] = tag
            row["distractor"] = sp.sstr(D)
            variants.append(row)
            if why:
                fail = f"{tag}:{why}"
                break
        blk = {"family": fam, "base_signature": sig,
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
        if b["fail"] is None and b.get("term") in (1, 2, 3):
            strata[(b["family"], b["term"])].append(b)
    for k in strata:
        strata[k].sort(key=lambda b: b["sig_sha"])
    e_counts = {f"{f}|t{t}": len(strata.get((f, t), []))
                for f in ("CH-F3", "CH-F4") for t in (1, 2, 3)}
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
    if eshort:
        finish("EVAL NO-FIRE", base | {"short_strata": eshort})
        return 0

    calib, heldout = [], []
    sel_census = Counter()
    ev_curs = []
    sel_targets = set()
    dupsig = set()
    for f in ("CH-F3", "CH-F4"):
        for t in (1, 2, 3):
            for j, b in enumerate(
                    strata[(f, t)][:N_PER_STRATUM]):
                if b["base_signature"] in dupsig:
                    refuted("DUP SIG SELECTED", base)
                dupsig.add(b["base_signature"])
                sel_targets.add(b["target_integrand"])
                primary_small = ("smallA" if j % 2 == 0
                                 else "smallB")
                robust_small = ("smallB" if j % 2 == 0
                                else "smallA")
                by = {v["variant"]: v for v in b["_variants"]}
                if t == 1:
                    plan_rows = ((by[primary_small],
                                  "control-I1", calib),
                                 (by["after"], "control-I0",
                                  calib),
                                 (by[robust_small],
                                  "control-robust-I1", calib))
                else:
                    plan_rows = ((by[primary_small],
                                  "heldout-I1", heldout),
                                 (by["after"], "covered-I0",
                                  calib),
                                 (by[robust_small],
                                  "robustness-I1", heldout))
                for v, role, sink in plan_rows:
                    sink.append({
                        "block_id": f"{f}-t{t}-{j:02d}",
                        "family": f, "term_cell": t,
                        "site_role": role,
                        "primary": "robust" not in role,
                        "confirmatory_denominator":
                            role == "heldout-I1",
                        "distractor": v["distractor"],
                        "block_d_before": by[primary_small][
                            "distractor"],
                        "base_signature": b["base_signature"],
                        "cur": v["cur"],
                        "parent_srepr_sha":
                            v["parent_srepr_sha"],
                        "n_candidates": v["n_candidates"],
                        "min_hce_ties": v["min_hce_ties"],
                        "chosen_ordinal": v["chosen_ordinal"],
                        "chosen_term": v["chosen_term"],
                        "candidates": v["candidates"]})
                    ev_curs.append(v["cur"])
                sel_census[f"{f}|t{t}|"
                           f"{by[primary_small]['distractor']}"
                           ] += 1
    half = N_PER_STRATUM - N_PER_STRATUM // 2
    for f in ("CH-F3", "CH-F4"):
        for t in (1, 2, 3):
            a = sel_census.get(f"{f}|t{t}|x**x", 0)
            bd = sel_census.get(f"{f}|t{t}|1/(x + log(x))", 0)
            if not (a == half and bd == N_PER_STRATUM // 2):
                refuted(f"D BALANCE {f} t{t} {a}/{bd}", base)
    if len(ev_curs) != len(set(ev_curs)):
        refuted("DUP EVAL PARENT", base)
    env = {"natural": sum(c in train_cur for c in ev_curs),
           "band": sum(c in band_cur for c in ev_curs),
           "pilot": sum(c in pilot_cur for c in ev_curs),
           "designs_and_diet": sum(c in dd_cur
                                   for c in ev_curs)}
    tgt_hits = sum(1 for ti in sel_targets
                   if ti in train_targets)
    if SMOKE:
        base["smoke_novelty_hits_expected_burned"] = env
        base["smoke_target_overlap"] = tgt_hits
    else:
        if any(env.values()):
            refuted(f"EVAL BURNED OVERLAP {env}", base)
        if tgt_hits:
            refuted(f"TARGET INTEGRAND OVERLAP {tgt_hits}",
                    base)
    cal_census = Counter(r["site_role"] for r in calib)
    hld_census = Counter(r["site_role"] for r in heldout)
    if not SMOKE:
        if not (cal_census == Counter(
                {"covered-I0": 96, "control-I1": 48,
                 "control-I0": 48, "control-robust-I1": 48})
                and hld_census == Counter(
                {"heldout-I1": 96, "robustness-I1": 96})):
            refuted(f"ROLE CENSUS {dict(cal_census)} "
                    f"{dict(hld_census)}", base)
    with open(OUTDIR / "covered_calibration.jsonl", "w") as fh:
        for r in calib:
            fh.write(json.dumps(r) + "\n")
    with open(OUTDIR / "heldout_test.jsonl", "w") as fh:
        for r in heldout:
            fh.write(json.dumps(r) + "\n")
    base["wall_s"] = round(time.monotonic() - t_all, 1)
    finish("EVAL MATERIALIZED", base | {
        "eval_blocks_selected": len(dupsig),
        "selected_target_overlap_with_train_targets": tgt_hits,
        "calibration_role_census": dict(cal_census),
        "heldout_role_census": dict(hld_census),
        "eval_d_before_census": dict(sel_census),
        "min_hce_ties_census": dict(Counter(
            r["min_hce_ties"] for r in calib + heldout)),
        "legal_set_size_census": dict(Counter(
            r["n_candidates"] for r in calib + heldout))})
    return 0


if __name__ == "__main__":
    sys.exit(main())

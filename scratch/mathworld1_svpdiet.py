"""MATH-CYBER-1 SVP-GRID-DIET-DESIGN-0 — balanced covered-grid
TRAINING augmentation + a completely fresh, separately sealed
calibration/heldout evaluation pair for the i_unprod site x term
grid. ZERO model/checkpoint access; the paired birth is the next
GO. The two held-out whole actions (i_unprod I1 term_index 2 and
3) receive ZERO training target rows — hard-gated at emission and
re-gated over the final bytes.

FROZEN K RULE (decided from burned yields only, before any
counting on fresh banks): K = largest of {512, 256, 128} such
that for BOTH families the literal fresh bank below satisfies
H_family >= 1.5 * K / worst_covered_cell_rate, rates pooled from
the burned DESIGN-0 + DESIGN-1 block censuses (CH-F3 288 bases:
t0/t1/t2/t3 = 22/60/80/74; CH-F4 672 bases: 216/118/64/122; one
base = one I0 row, two I1 rows for t in {0,1}).
  CH-F3 worst = I0/t0 rate 22/288 = .0764 -> K=128 needs 2,513
    <= H=2,520 (K=256 needs 5,026: fail).
  CH-F4 worst = I0/t2 rate 64/672 = 2/21 -> K=128 needs 2,016
    <= H=2,016 (K=256 needs 4,032: fail).
  => K = 128 per {family x covered cell}; augmentation
  128 * 2 * 6 = 1,536 rows; combined diet 73,324 + 1,536 =
  74,860 rows.

FROZEN FRESH BANKS (all disjoint from SPACE-1/2/3/4, DESIGN-0,
DESIGN-1, and from each other; train v eval disjoint by
construction — disjoint P/P1 banks and disjoint c banks):
  TRAIN CH-F3 (2,520): P in 21 fresh polys (deg 5-9 listed
    below); T in {sin,cos}; c in {8,9,10,11,12,15,16,17,18,19};
    w in {exp(x)/x, sin(x)/x}; k in {0,1,2} over fresh polys
    (9*x, 13*x**6).
  TRAIN CH-F4 (2,016): (P1,P2) = 42 fresh pairs (7 fresh-P1 rows
    x 6 P2 columns); (a,b) in 12 fresh pairs {(8,9),(9,8),(8,3),
    (3,8),(9,4),(4,9),(8,5),(5,8),(9,2),(2,9),(7,8),(8,7)};
    (T1,T2) in {(sin,cos),(cos,sin)}; w in {exp(x)/x, sin(x)/x}.
  EVAL CH-F3 (192): P in {x^9+x..x^9+x^8}; c in {13,14}; same
    T/w axes; k in {0,1,2} over fresh polys (10*x**2, 12*x**5).
  EVAL CH-F4 (576): (P1,P2) = 24 fresh pairs (4 fresh-P1 rows x
    6 P2 columns, P1 rows disjoint from train's); (a,b) in
    {(10,11),(11,10),(10,3),(3,10),(11,4),(4,11)}; same T/w.

TRAIN EMISSION LAW (deterministic, no retry-until-success): per
base, qualify the AFTER variant (sin(sin(x))) first; any
qualification failure skips the base (first-fail). The after
row's measured cell (ordinal, term) buckets it if covered. Iff
the after variant's measured term is in {0,1}, also qualify the
two SMALL variants (x**x, 1/(x+log(x))); each emitted row keeps
ITS OWN measured teacher cell. Rows measuring a held-out cell
(I1/t2, I1/t3) are CENSORED (counted, never emitted); rows
outside the 2x4 grid or off-rule are counted as out-of-grid.
Selection: first K unique rows by SHA256(cur) per
{family x covered cell}; any short bucket => AUGMENTATION
NO-FIRE (no widening, no substitution).

EVAL BLOCK LAW (verbatim DESIGN-1): three distractor variants
per base; block qualifies iff all three pass qualify_parent,
teacher kind (I, term_index), rule i_unprod, single term,
ordinals [1,1,0]; strata {family x term 1|2|3}; fill gate >= 24
each else EVAL NO-FIRE; selection first 24 by sig_sha; D_before
alternates x**x / 1/(x+log(x)) 12/12 per stratum. Split:
  covered_calibration.jsonl: 96 strict covered-I0 states (t2/t3
    after variants) + all 144 t1 control states (48 control-I1
    primary, 48 control-I0, 48 control-robust-I1).
  heldout_test.jsonl: 96 strict heldout-I1 primary + 96 strict
    robustness-I1. SEALED — the future scorer may not open it
    until the frozen calibration readiness gate fires.

BATCH PLAN: verbatim svpbirth law (sorted row_ids, per-epoch
random.Random(f"svp-epoch-{e}").shuffle, BS=32, EPOCHS=3) over
the combined 74,860 row_ids; target-blind by construction; plan
sha recorded (plan re-derivable from the manifest alone).

SMOKE (SVPDIET_SMOKE=1, path-isolated under
logs/mathworld1/svpdiet_smoke/): K=1 / N=1 on BURNED DESIGN-0
bases (train: first-by-sig_sha qualified base per
{family x term 0..3}; eval: first per {family x term 1..3});
novelty gates report-only there. Production refuses unless the
smoke receipt is green AND records this driver's current sha.

Outputs (production, refuse-if-exists) under
logs/mathworld1/svpdiet/: train_blocks.jsonl,
balanced_grid_train.jsonl, combined_train_manifest.jsonl,
eval_blocks.jsonl, covered_calibration.jsonl,
heldout_test.jsonl, svpdiet_receipt.json (artifact shas on
every exit path).

    SVPDIET_SMOKE=1 .venv/bin/python scratch/mathworld1_svpdiet.py
    .venv/bin/python scratch/mathworld1_svpdiet.py          (Mac)
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
from scratch.mathworld1_svpbirth import (batch_plan,  # noqa: E402
                                         gate)
from scratch.mathworld1_svpchal import (AFTER_D,  # noqa: E402
                                        SMALL_D, X, fsha,
                                        qualify_parent)
from scratch.mathworld1_svpchal import \
    build_horizon as d0_horizon  # noqa: E402
from scratch.mathworld1_svpchal2 import \
    build_horizon1 as d1_horizon  # noqa: E402

SMOKE = os.environ.get("SVPDIET_SMOKE") == "1"
K_PER_CELL = 1 if SMOKE else 128
N_PER_STRATUM = 1 if SMOKE else 24
OUTDIR = Path("logs/mathworld1/svpdiet_smoke" if SMOKE
              else "logs/mathworld1/svpdiet")
SMOKE_RECEIPT = Path(
    "logs/mathworld1/svpdiet_smoke/svpdiet_receipt.json")
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
    "logs/mathworld1/svpchal2/blocks.jsonl":
        "a50da492b2447b3bd10fd0409dd17d3d77ca57b4b4accbc6a703"
        "880c34a57fea",
    "logs/mathworld1/svpchal2/decisions.jsonl":
        "1e3a5ef0483a7ee4970aa84c1f3d1dcc9171b8376cc6281e81d2"
        "b301a8f80d69",
}
PILOT_RECEIPTS = ["logs/mathworld1/svpgriddesk_receipt.json",
                  "logs/mathworld1/svpgriddesk2_receipt.json",
                  "logs/mathworld1/svpgriddesk3_receipt.json",
                  "logs/mathworld1/svpgriddesk4_receipt.json"]
COVERED_CELLS = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1))
HELD_OUT_CELLS = ((1, 2), (1, 3))

# ---- frozen fresh banks -------------------------------------
F3_TRAIN_P = (X**7, X**8, X**9,
              X**6 + X, X**6 + X**3, X**6 + X**4, X**6 + X**5,
              X**7 + X, X**7 + X**2, X**7 + X**3, X**7 + X**4,
              X**7 + X**5, X**7 + X**6,
              X**8 + X, X**8 + X**2, X**8 + X**3, X**8 + X**4,
              X**8 + X**5, X**8 + X**6, X**8 + X**7,
              X**5 + X**4)
F3_TRAIN_C = (8, 9, 10, 11, 12, 15, 16, 17, 18, 19)
F3_TRAIN_POLYS = (9 * X, 13 * X**6)
F3_EVAL_P = (X**9 + X, X**9 + X**2, X**9 + X**3,
             X**9 + X**4, X**9 + X**5, X**9 + X**6,
             X**9 + X**7, X**9 + X**8)
F3_EVAL_C = (13, 14)
F3_EVAL_POLYS = (10 * X**2, 12 * X**5)
F4_P2 = (X, X**2, X**3, X**2 + X, X**3 + X, X**4)
F4_TRAIN_P1 = (X**7, X**8, X**7 + X, X**7 + X**2, X**7 + X**3,
               X**8 + X, X**8 + X**2)
F4_EVAL_P1 = (X**6 + X**4, X**6 + X**5, X**5 + X**4,
              X**8 + X**3)
F4_TRAIN_FREQS = ((8, 9), (9, 8), (8, 3), (3, 8), (9, 4),
                  (4, 9), (8, 5), (5, 8), (9, 2), (2, 9),
                  (7, 8), (8, 7))
F4_EVAL_FREQS = ((10, 11), (11, 10), (10, 3), (3, 10), (11, 4),
                 (4, 11))
WS = (sp.exp(X) / X, sp.sin(X) / X)
BURNED_F3_P = {X**2, X**3, X**2 + X, X**4, X**3 + X**2,
               X**4 + X, X**5, X**3 + X, X**4 + X**2,
               X**6, X**5 + X, X**5 + X**2, X**4 + X**3,
               X**6 + X**2, X**5 + X**3}
BURNED_F3_C = {2, 3, 4, 5, 6, 7}
BURNED_F3_POLYS = {sp.sstr(e) for e in
                   (X, 7 * X**3, 2 * X, 5 * X**2,
                    3 * X, 11 * X**4)}
BURNED_F4_PAIRS = {
    (X**3, X), (X**2 + X, X), (X**3, X**2), (X**4, X),
    (X**2, X**3),
    (X**4 + X**2, X), (X**3 + X**2, X), (X**4, X**2),
    (X**5, X), (X**4, X**3), (X**2 + X, X**3), (X**5, X**2),
    (X**3 + X, X**2), (X**4 + X, X), (X**5, X**3),
    (X**3 + X**2, X**2), (X**4, X**2 + X),
    (X**6, X), (X**6, X**2), (X**6, X**3),
    (X**5 + X, X), (X**5 + X, X**2), (X**5 + X**2, X),
    (X**4 + X**3, X), (X**4 + X**3, X**2),
    (X**6, X**2 + X), (X**5, X**2 + X),
    (X**5, X**3 + X), (X**4 + X**2, X**2),
    (X**6, X**4), (X**5 + X**3, X), (X**5 + X**3, X**2),
    (X**6 + X, X), (X**6 + X**2, X), (X**6, X**3 + X),
    (X**4 + X**3, X**3), (X**5 + X, X**3),
    (X**5 + X**2, X**3), (X**6 + X, X**2),
    (X**6 + X**3, X), (X**6, X**5)}
BURNED_FREQS = {(2, 3), (4, 5), (5, 4), (5, 3), (3, 4),
                (6, 7), (7, 6), (7, 4), (4, 7), (6, 5), (5, 6)}


def f3_bases(tag, Ps, Cs, polys):
    out = []
    for P in Ps:
        for T in (sp.sin, sp.cos):
            for c in Cs:
                for w in WS:
                    for k in (0, 1, 2):
                        f = (sp.expand(sp.diff(P * T(c * X), X))
                             + sp.Integral(w, X)
                             + sp.Add(*polys[:k]))
                        sig = (f"DIET-{tag}|CH-F3|P={P}|"
                               f"T={T.__name__}|c={c}|w={w}|"
                               f"k={k}")
                        out.append(("CH-F3", sig, f))
    return out


def f4_bases(tag, P1s, freqs):
    out = []
    for P1 in P1s:
        for P2 in F4_P2:
            for (a, b) in freqs:
                for (T1, T2) in ((sp.sin, sp.cos),
                                 (sp.cos, sp.sin)):
                    for w in WS:
                        f = (sp.expand(
                                sp.diff(P1 * T1(a * X), X))
                             + sp.expand(
                                sp.diff(P2 * T2(b * X), X))
                             + sp.Integral(w, X))
                        sig = (f"DIET-{tag}|CH-F4|P1={P1}|"
                               f"P2={P2}|a={a}|b={b}|"
                               f"T1={T1.__name__}|"
                               f"T2={T2.__name__}|w={w}")
                        out.append(("CH-F4", sig, f))
    return out


def d0_qualified_by_stratum():
    """(family, term) -> qualified DESIGN-0 blocks sorted by
    sig_sha, from the pinned blocks.jsonl."""
    rows = [json.loads(l) for l in
            open("logs/mathworld1/svpchal/blocks.jsonl")]
    strata = defaultdict(list)
    for r in rows:
        if r.get("fail") is None and "term" in r:
            strata[(r["family"], r["term"])].append(r)
    for k in strata:
        strata[k].sort(key=lambda r: r["sig_sha"])
    return strata


def smoke_slice(terms):
    strata = d0_qualified_by_stratum()
    want = {}
    for fam in ("CH-F3", "CH-F4"):
        for t in terms:
            want[strata[(fam, t)][0]["base_signature"]] = 1
    out = [(fam, sig, f) for fam, sig, f in d0_horizon()
           if sig in want]
    gate(len(out) == len(want), f"SMOKE SLICE {len(out)}")
    return out


def chosen_cand(row):
    return next(c for c in row["candidates"] if c["is_label"])


def emit_train_row(fam, sig, variant, D, vrow):
    c = chosen_cand(vrow)
    cur = vrow["cur"]
    return {
        "row_id": hashlib.sha256(
            f"svpdiet|{cur}".encode()).hexdigest()[:16],
        "cur": cur,
        "rule": c["rule"], "site_kind": c["site_kind"],
        "site_ordinal": c["site_ordinal"],
        "param_kind": c["param_kind"],
        "param_index": c["param_index"],
        "program_text": c["program_text"],
        "state_target": c["child_sstr"],
        "state_seq_tokens": c["state_seq_tokens"],
        "family": fam, "base_signature": sig,
        "variant": variant, "distractor": sp.sstr(D),
        "n_candidates": vrow["n_candidates"],
        "min_hce_ties": vrow["min_hce_ties"],
        "source": "svpdiet-augment"}


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    if not SMOKE:
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(sr.get("smoke") is True and sr.get("verdict") ==
             "DIET MATERIALIZED", "SMOKE NOT GREEN")
        gate(sr["start"]["file_sha256"][
            "scratch/mathworld1_svpdiet.py"] == fsha(
            "scratch/mathworld1_svpdiet.py"), "SMOKE STALE")
    START = start_provenance(
        ["scratch/mathworld1_svpdiet.py",
         "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpchal2.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    # bank freshness gates
    for P in F3_TRAIN_P + F3_EVAL_P:
        gate(P not in BURNED_F3_P, f"BURNED F3 P {P}")
    gate(not set(F3_TRAIN_P) & set(F3_EVAL_P), "F3 P OVERLAP")
    for c in F3_TRAIN_C + F3_EVAL_C:
        gate(c not in BURNED_F3_C, f"BURNED F3 c {c}")
    gate(not set(F3_TRAIN_C) & set(F3_EVAL_C), "F3 C OVERLAP")
    for pv in F3_TRAIN_POLYS + F3_EVAL_POLYS:
        gate(sp.sstr(pv) not in BURNED_F3_POLYS,
             f"BURNED F3 POLY {pv}")
    gate(not set(F4_TRAIN_P1) & set(F4_EVAL_P1), "F4 P1 OVERLAP")
    for P1 in F4_TRAIN_P1 + F4_EVAL_P1:
        for P2 in F4_P2:
            gate((P1, P2) not in BURNED_F4_PAIRS,
                 f"BURNED PAIR {(P1, P2)}")
    for fq in F4_TRAIN_FREQS + F4_EVAL_FREQS:
        gate(fq not in BURNED_FREQS, f"BURNED FREQ {fq}")
    gate(not set(F4_TRAIN_FREQS) & set(F4_EVAL_FREQS),
         "F4 FREQ OVERLAP")

    # burn set: training + bands + pilot + DESIGN-0 + DESIGN-1
    natural_rows = [json.loads(l) for l in open(PAIRED)]
    gate(len(natural_rows) == 73324, "NATURAL ROWS")
    train_cur = {r["cur"] for r in natural_rows}
    seen = {(r["rule"], r["site_kind"], r["site_ordinal"],
             r["param_kind"], r["param_index"])
            for r in natural_rows}
    for (o, t) in HELD_OUT_CELLS:
        gate(("i_unprod", "I", o, "term_index", t) not in seen,
             f"HELD-OUT IN TRAINING I{o}t{t}")
    for (o, t) in COVERED_CELLS:
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
    for hz, n in ((d0_horizon(), 720), (d1_horizon(), 2160)):
        before = len(dd_cur)
        for fam, sig, f in hz:
            for D in (SMALL_D[0], SMALL_D[1], AFTER_D):
                dd_cur.add(sp.sstr(sp.Add(sp.Integral(f, X),
                                          sp.Integral(D, X))))
        gate(len(dd_cur) - before == n,
             f"BURNED HORIZON {len(dd_cur) - before}")
    # horizons
    if SMOKE:
        train_hz = smoke_slice((0, 1, 2, 3))
        eval_hz = smoke_slice((1, 2, 3))
    else:
        train_hz = (f3_bases("TRAIN", F3_TRAIN_P, F3_TRAIN_C,
                             F3_TRAIN_POLYS)
                    + f4_bases("TRAIN", F4_TRAIN_P1,
                               F4_TRAIN_FREQS))
        eval_hz = (f3_bases("EVAL", F3_EVAL_P, F3_EVAL_C,
                            F3_EVAL_POLYS)
                   + f4_bases("EVAL", F4_EVAL_P1,
                              F4_EVAL_FREQS))
        gate(len(train_hz) == 4536, f"TRAIN HZ {len(train_hz)}")
        gate(len(eval_hz) == 768, f"EVAL HZ {len(eval_hz)}")
        sigs = [s for _, s, _ in train_hz + eval_hz]
        gate(len(sigs) == len(set(sigs)), "SIG DUP")
        fts = [sp.srepr(f) for _, _, f in train_hz + eval_hz]
        gate(len(fts) == len(set(fts)), "TARGET DUP")
    OUTDIR.mkdir(parents=True)
    receipt = {"smoke": SMOKE, "prereg":
               "MATH-CYBER-1-SVP-GRID-DIET-DESIGN-0",
               "K_per_cell": K_PER_CELL,
               "N_per_stratum": N_PER_STRATUM,
               "n_train_horizon": len(train_hz),
               "n_eval_horizon": len(eval_hz),
               "smoke_receipt_sha": (fsha(SMOKE_RECEIPT)
                                     if not SMOKE else None),
               "pins": {p: fsha(p) for p in PINS}}

    def art_shas():
        return {f.name: fsha(f) for f in sorted(
            OUTDIR.glob("*.jsonl")) if f.exists()}

    def finish(verdict, extra):
        receipt["verdict"] = verdict
        receipt.update(extra)
        receipt["artifact_sha256"] = art_shas()
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "svpdiet_receipt.json").write_text(
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

    # ---- phase 1: TRAIN augmentation ------------------------
    buckets = defaultdict(list)
    all_train_side_curs = set()
    censored = Counter()
    out_grid = Counter()
    fail_census = Counter()
    tb = open(OUTDIR / "train_blocks.jsonl", "w")
    for i, (fam, sig, f_t) in enumerate(train_hz):
        rec = {"family": fam, "base_signature": sig,
               "sig_sha": hashlib.sha256(
                   sig.encode()).hexdigest(),
               "target_integrand": sp.sstr(f_t)}
        arow, why = qualify_parent(f_t, AFTER_D)
        if why:
            rec["fail"] = f"after:{why}"
            fail_census[rec["fail"]] += 1
            tb.write(json.dumps(rec) + "\n")
            continue
        emitted = []
        variants = [("after", AFTER_D, arow)]
        term_a = None
        if (arow["chosen_rule"] == "i_unprod"
                and arow["chosen_site_kind"] == "I"
                and arow["chosen_param_kind"] == "term_index"):
            term_a = arow["chosen_term"]
        if term_a in (0, 1):
            for tag, D in (("smallA", SMALL_D[0]),
                           ("smallB", SMALL_D[1])):
                vrow, vwhy = qualify_parent(f_t, D)
                if vwhy:
                    fail_census[f"{tag}:{vwhy}"] += 1
                    continue
                variants.append((tag, D, vrow))
        for tag, D, vrow in variants:
            all_train_side_curs.add(vrow["cur"])
            if not (vrow["chosen_rule"] == "i_unprod"
                    and vrow["chosen_site_kind"] == "I"
                    and vrow["chosen_param_kind"]
                    == "term_index"):
                out_grid[f"{fam}|off:{vrow['chosen_rule']}|"
                         f"{vrow['chosen_site_kind']}|"
                         f"{vrow['chosen_param_kind']}"] += 1
                continue
            cell = (vrow["chosen_ordinal"], vrow["chosen_term"])
            if cell in HELD_OUT_CELLS:
                censored[f"{fam}|I{cell[0]}t{cell[1]}"] += 1
                continue
            if cell not in COVERED_CELLS:
                out_grid[f"{fam}|I{cell[0]}t{cell[1]}"] += 1
                continue
            row = emit_train_row(fam, sig, tag, D, vrow)
            buckets[(fam, cell)].append(row)
            emitted.append(f"{tag}:I{cell[0]}t{cell[1]}")
        rec["fail"] = None
        rec["after_term"] = term_a
        rec["emitted"] = emitted
        tb.write(json.dumps(rec) + "\n")
        if (i + 1) % 100 == 0 or SMOKE:
            print(f"[train {i + 1}/{len(train_hz)}] "
                  f"buckets={ {f'{f}|I{o}t{t}': len(v) for (f, (o, t)), v in sorted(buckets.items(), key=str)} }",
                  flush=True)
    tb.close()
    bucket_counts = {f"{f}|I{o}t{t}": len(v)
                     for (f, (o, t)), v in sorted(
                         buckets.items(), key=str)}
    base = {"train_bucket_counts": bucket_counts,
            "train_censored_heldout": dict(censored),
            "train_out_of_grid": dict(out_grid),
            "train_fail_census": dict(fail_census)}
    short = [f"{f}|I{o}t{t}"
             for f in ("CH-F3", "CH-F4")
             for (o, t) in COVERED_CELLS
             if len(buckets.get((f, (o, t)), []))
             < K_PER_CELL]
    if short:
        base["wall_s"] = round(time.monotonic() - t_all, 1)
        finish("AUGMENTATION NO-FIRE",
               base | {"short_cells": short})
        return 0

    selected = []
    for f in ("CH-F3", "CH-F4"):
        for cell in COVERED_CELLS:
            rows = sorted(buckets[(f, cell)],
                          key=lambda r: hashlib.sha256(
                              r["cur"].encode()).hexdigest())
            selected.extend(rows[:K_PER_CELL])
    sel_curs = [r["cur"] for r in selected]
    if len(sel_curs) != len(set(sel_curs)):
        refuted("AUG DUP CUR", base)
    if len(selected) != 12 * K_PER_CELL:
        refuted("AUG COUNT", base)
    # ZERO held-out training labels, re-gated over selection
    for r in selected:
        if (r["rule"] == "i_unprod" and r["site_kind"] == "I"
                and r["site_ordinal"] == 1
                and r["param_kind"] == "term_index"
                and r["param_index"] in (2, 3)):
            refuted("HELD-OUT LABEL IN AUGMENTATION", base)
    nov = {"train": sum(c in train_cur for c in sel_curs),
           "band": sum(c in band_cur for c in sel_curs),
           "pilot": sum(c in pilot_cur for c in sel_curs),
           "designs": sum(c in dd_cur for c in sel_curs)}
    if SMOKE:
        base["smoke_aug_novelty_hits_expected_burned"] = nov
    elif any(nov.values()):
        refuted(f"AUG BURNED OVERLAP {nov}", base)
    ids = [r["row_id"] for r in selected]
    nat_ids = [r["row_id"] for r in natural_rows]
    if len(set(ids) | set(nat_ids)) != len(ids) + len(nat_ids):
        refuted("ROW_ID COLLISION", base)
    with open(OUTDIR / "balanced_grid_train.jsonl", "w") as fh:
        for r in selected:
            fh.write(json.dumps(r) + "\n")
    with open(OUTDIR / "combined_train_manifest.jsonl",
              "w") as fh:
        for rid in nat_ids:
            fh.write(json.dumps(
                {"row_id": rid, "source": "natural"}) + "\n")
        for rid in ids:
            fh.write(json.dumps(
                {"row_id": rid,
                 "source": "svpdiet-augment"}) + "\n")
    # frozen target-blind batch plan (svpbirth law BY CALL)
    all_ids = sorted(nat_ids + ids)
    plan = batch_plan(all_ids)
    plan_sha = hashlib.sha256(
        json.dumps(plan).encode()).hexdigest()
    base.update({
        "n_augment": len(selected),
        "n_combined": len(all_ids),
        "batch_plan_law": ("sorted row_ids; per-epoch "
                           "random.Random(f'svp-epoch-{e}')"
                           ".shuffle; BS=32; EPOCHS=3"),
        "batch_plan_n_batches": len(plan),
        "batch_plan_sha256": plan_sha,
        "aug_distinct_bases_per_cell": {
            f"{f}|I{o}t{t}": len({r["base_signature"]
                                  for r in selected
                                  if r["family"] == f
                                  and (r["site_ordinal"],
                                       r["param_index"])
                                  == (o, t)})
            for f in ("CH-F3", "CH-F4")
            for (o, t) in COVERED_CELLS},
        "aug_distractor_census": dict(Counter(
            r["distractor"] for r in selected)),
        "aug_state_tokens": {
            "max": max(r["state_seq_tokens"]
                       for r in selected),
            "mean": round(sum(r["state_seq_tokens"]
                              for r in selected)
                          / len(selected), 1)},
        "combined_whole_action_freq": dict(Counter(
            f"{r['rule']}|{r['site_kind']}"
            f"{r['site_ordinal']}|{r['param_kind']}"
            f"{r['param_index']}"
            for r in natural_rows + selected))})
    print(f"[train done] wall={time.monotonic() - t_all:.0f}s",
          flush=True)

    # ---- phase 2: EVAL bank (verbatim DESIGN-1 block law) ---
    t_ev = time.monotonic()
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
    base.update({
        "eval_stratum_qualified_counts": e_counts,
        "eval_block_fail_census": dict(Counter(
            b["fail"] for b in eblocks if b["fail"])),
        "eval_terms_census": dict(Counter(
            b.get("term") for b in eblocks
            if b["fail"] is None)),
        "wall_train_s": round(t_ev - t_all, 1),
        "wall_s": round(time.monotonic() - t_all, 1)})
    eshort = [k for k, v in e_counts.items()
              if v < N_PER_STRATUM]
    if eshort:
        finish("EVAL NO-FIRE", base | {"short_strata": eshort})
        return 0

    calib, heldout = [], []
    sel_census = Counter()
    ev_curs = []
    dupsig = set()
    for f in ("CH-F3", "CH-F4"):
        for t in (1, 2, 3):
            for j, b in enumerate(
                    strata[(f, t)][:N_PER_STRATUM]):
                if b["base_signature"] in dupsig:
                    refuted("DUP SIG SELECTED", base)
                dupsig.add(b["base_signature"])
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
                refuted(f"D BALANCE {f} t{t} {a}/{bd}",
                        base)
    if len(ev_curs) != len(set(ev_curs)):
        refuted("DUP EVAL PARENT", base)
    env = {"natural": sum(c in train_cur for c in ev_curs),
           "augment": sum(c in all_train_side_curs
                          for c in ev_curs),
           "band": sum(c in band_cur for c in ev_curs),
           "pilot": sum(c in pilot_cur for c in ev_curs),
           "designs": sum(c in dd_cur for c in ev_curs)}
    if SMOKE:
        base["smoke_eval_novelty_hits_expected_burned"] = env
        base["smoke_train_eval_overlap_expected"] = True
    elif any(env.values()):
        refuted(f"EVAL BURNED/TRAIN OVERLAP {env}", base)
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
    finish("DIET MATERIALIZED", base | {
        "eval_blocks_selected": len(dupsig),
        "calibration_role_census": dict(cal_census),
        "heldout_role_census": dict(hld_census),
        "eval_d_before_census": dict(sel_census)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

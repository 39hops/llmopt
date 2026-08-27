"""MATH-CYBER-1 SVP-STRICT-GRID-CHALLENGE-DESIGN-1 — successor to
the booked DESIGN-0 NO-FIRE: SAME matched-grid scientific and
inferential law re-registered unchanged, larger completely fresh
constructor horizons, one materialization, qualify, mechanical
fill gate, mechanical selection, frozen challenge bytes. Zero
model/checkpoint access; scoring is a separate GO. DESIGN-0
stays booked as a legitimate preregistered NO-FIRE, and its 240
bases (720 parent variants) are BURNED here alongside the 566
pilot parents.

FROZEN FRESH HORIZONS (deterministic full product spaces, fixed
before any engine qualification; builder-executed cardinality
and freshness gates; DESIGN-0's empirical CH-F4 census
t0..t5 = 24/4/12/20/28/8 over 96 bases is used ONLY to price
cardinality, never to pick parameter combinations):
  CH-F3 (same lineage), 144 fresh bases:
    f_t = expand(d/dx[P*T(c*x)]) + Integral(w, x) + first-k polys
    P in {x^6, x^5+x, x^5+x^2, x^4+x^3, x^6+x^2, x^5+x^3};
    T in {sin, cos}; c in {6, 7}; w in {exp(x)/x, sin(x)/x};
    k in {0,1,2} over fresh polys (3*x, 11*x^4). 6*2*2*2*3 = 144.
  CH-F4 (same two-derivative-components + inner Integral(w,x)
  family; broad fresh variation, no t1-targeting), 576 bases:
    f_t = expand(d/dx[P1*T1(a*x)]) + expand(d/dx[P2*T2(b*x)])
          + Integral(w, x)
    (P1,P2): 24 fresh pairs (listed in build_horizon; gate-
    checked disjoint from the 5 pilot pairs and the 12 DESIGN-0
    pairs); (a,b) in {(6,7), (7,6), (7,4), (4,7), (6,5), (5,6)}
    (gate-checked disjoint from {(2,3),(4,5),(5,4),(5,3),
    (3,4)}); (T1,T2) in {(sin,cos), (cos,sin)} (trig-assignment
    nuisance axis; the semantic question is unchanged);
    w in {exp(x)/x, sin(x)/x}. 24*6*2*2 = 576.
  720 bases x 3 distractor variants = 2,160 qualification
  parents. Pricing register (NOT an IID/binomial claim — term
  index is deterministic in the parameters; the horizon buys
  prespecified margin): DESIGN-0 fresh t1 = 4/96, so a 6x
  horizon prices ~24 t1 bases v the frozen N=12.

LAW (verbatim from DESIGN-0, unconsumed there): three distractor
variants per base (x**x, 1/(x+log(x)), sin(sin(x))); block
qualifies iff all three pass stable legal set, full exact
ActionProgram derivation, tok roundtrip, CTX fit, code domain,
FACTOR+HASH roundtrip, teacher argmin (hce,name,key) = i_unprod
with site_kind/param_kind (I, term_index), identical term across
variants, ordinals 1/1/0; fill gate >= 12 qualified per
{family x term 1|2|3} else CHALLENGE NO-FIRE (no extension, no
top-up, the complete finite horizon runs once); selection first
12 by SHA256(base_signature) per stratum; D_before alternates
x**x / 1/(x+log(x)) in hash order (6/6 by construction, gated);
final counts gated exactly 72 blocks / 144 primary decisions
(48 heldout-I1 t2+t3, 48 covered-I0 t2+t3, 24+24 t1 calibration)
+ 72 robustness states flagged non-primary.

SMOKE (path-isolated, SVPCHAL2_SMOKE=1): the selection/emission
path never executed in DESIGN-0 (NO-FIRE returned first), so the
production run must not be its first test. Smoke horizon = 6
BURNED DESIGN-0 bases (one per {family x term}, chosen
deterministically from the pinned blocks.jsonl by sig_sha order)
with N=1; writes ONLY under logs/mathworld1/svpchal2_smoke/;
novelty gates run in report-only mode there (the bases are burned
by design); the production entry refuses if smoke receipts are
absent.

Outputs (production) under logs/mathworld1/svpchal2/
(refuse-if-exists): blocks.jsonl, decisions.jsonl,
svpchal2_receipt.json (blocks_sha + decisions_sha recorded on
EVERY exit path, including NO-FIRE).

    SVPCHAL2_SMOKE=1 .venv/bin/python scratch/mathworld1_svpchal2.py
    .venv/bin/python scratch/mathworld1_svpchal2.py           (Mac)
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
from scratch.mathworld1_svpcode import (factor_symbols,  # noqa: E402
                                        hash_symbols)

SMOKE = os.environ.get("SVPCHAL2_SMOKE") == "1"
N_PER_STRATUM = 1 if SMOKE else 12
OUTDIR = Path("logs/mathworld1/svpchal2_smoke" if SMOKE
              else "logs/mathworld1/svpchal2")
SMOKE_RECEIPT = Path(
    "logs/mathworld1/svpchal2_smoke/svpchal2_receipt.json")
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
    "logs/mathworld1/svpgriddesk_receipt.json":
        "ec9cb9b870d2515e7959025f7f3cbfcee7309a6dc90b880d3176"
        "d3e2ccf72edc",
    "logs/mathworld1/svpgriddesk2_receipt.json":
        "f0184b01c36017bcb93ed4b715e41e015999075c4da7976e11ad"
        "c5cae28e6977",
    "logs/mathworld1/svpgriddesk3_receipt.json":
        "26389ebb8d68f45447d9676b9c188ea86ab9dd70ec71329fe6ae"
        "268cfc34080f",
    "logs/mathworld1/svpgriddesk4_receipt.json":
        "8439fc636fcf5c6e18a7dd75a76642cfe088e7eeeaf1e9f1243c"
        "5bb0cf08610f",
    "logs/mathworld1/svpchal/blocks.jsonl":
        "21e7e635244574266ec876c1c8c76f8d7d2a77e20c2f6680d3ee"
        "48db372c4d10",
    "logs/mathworld1/svpchal/svpchal_receipt.json":
        "0f58517b79e5da2b1414c4132772518d911dc694fb1eb231540b\
5f472853aa1c",
    "scratch/mathworld1_svpchal.py":
        "4e72cf2c9d9e15e8d84692d077e658dc535199ffb9fd6f81406c\
a7c23785eaf0",
}
PILOT_RECEIPTS = ["logs/mathworld1/svpgriddesk_receipt.json",
                  "logs/mathworld1/svpgriddesk2_receipt.json",
                  "logs/mathworld1/svpgriddesk3_receipt.json",
                  "logs/mathworld1/svpgriddesk4_receipt.json"]
HELD_OUT_TERMS = (2, 3)

BURNED_F4_PAIRS = {
    # pilot (svpgriddesk4)
    (X**3, X), (X**2 + X, X), (X**3, X**2), (X**4, X),
    (X**2, X**3),
    # DESIGN-0 (svpchal)
    (X**4 + X**2, X), (X**3 + X**2, X), (X**4, X**2),
    (X**5, X), (X**4, X**3), (X**2 + X, X**3), (X**5, X**2),
    (X**3 + X, X**2), (X**4 + X, X), (X**5, X**3),
    (X**3 + X**2, X**2), (X**4, X**2 + X)}
BURNED_FREQS = {(2, 3), (4, 5), (5, 4), (5, 3), (3, 4)}

F4_PAIRS = ((X**6, X), (X**6, X**2), (X**6, X**3),
            (X**5 + X, X), (X**5 + X, X**2), (X**5 + X**2, X),
            (X**4 + X**3, X), (X**4 + X**3, X**2),
            (X**6, X**2 + X), (X**5, X**2 + X),
            (X**5, X**3 + X), (X**4 + X**2, X**2),
            (X**6, X**4), (X**5 + X**3, X), (X**5 + X**3, X**2),
            (X**6 + X, X), (X**6 + X**2, X), (X**6, X**3 + X),
            (X**4 + X**3, X**3), (X**5 + X, X**3),
            (X**5 + X**2, X**3), (X**6 + X, X**2),
            (X**6 + X**3, X), (X**6, X**5))
F4_FREQS = ((6, 7), (7, 6), (7, 4), (4, 7), (6, 5), (5, 6))


def build_horizon1():
    space = []
    polys = (3 * X, 11 * X**4)
    for P in (X**6, X**5 + X, X**5 + X**2, X**4 + X**3,
              X**6 + X**2, X**5 + X**3):
        for T in (sp.sin, sp.cos):
            for c in (6, 7):
                for w in (sp.exp(X) / X, sp.sin(X) / X):
                    for k in (0, 1, 2):
                        f = (sp.expand(sp.diff(P * T(c * X), X))
                             + sp.Integral(w, X)
                             + sp.Add(*polys[:k]))
                        sig = (f"D1|CH-F3|P={P}|T={T.__name__}|"
                               f"c={c}|w={w}|k={k}")
                        space.append(("CH-F3", sig, f))
    for (P1, P2) in F4_PAIRS:
        for (a, b) in F4_FREQS:
            for (T1, T2) in ((sp.sin, sp.cos),
                             (sp.cos, sp.sin)):
                for w in (sp.exp(X) / X, sp.sin(X) / X):
                    f = (sp.expand(sp.diff(P1 * T1(a * X), X))
                         + sp.expand(sp.diff(P2 * T2(b * X), X))
                         + sp.Integral(w, X))
                    sig = (f"D1|CH-F4|P1={P1}|P2={P2}|a={a}|"
                           f"b={b}|T1={T1.__name__}|"
                           f"T2={T2.__name__}|w={w}")
                    space.append(("CH-F4", sig, f))
    return space


def smoke_horizon():
    """Six BURNED DESIGN-0 bases, one per {family x term}, first
    by sig_sha within each stratum of the pinned blocks.jsonl."""
    want = {}
    rows = [json.loads(l) for l in
            open("logs/mathworld1/svpchal/blocks.jsonl")]
    for fam in ("CH-F3", "CH-F4"):
        for t in (1, 2, 3):
            cand = sorted((r for r in rows
                           if r["family"] == fam
                           and r.get("term") == t),
                          key=lambda r: r["sig_sha"])
            want[cand[0]["base_signature"]] = (fam, t)
    out = []
    for fam, sig, f in d0_horizon():
        if sig in want:
            out.append((fam, sig, f))
    gate(len(out) == 6, f"SMOKE HORIZON {len(out)}")
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
             "CHALLENGE MATERIALIZED", "SMOKE NOT GREEN")
        gate(sr["start"]["file_sha256"][
            "scratch/mathworld1_svpchal2.py"] == fsha(
            "scratch/mathworld1_svpchal2.py"), "SMOKE STALE")
    START = start_provenance(
        ["scratch/mathworld1_svpchal2.py",
         "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    # freshness gates on the banks themselves
    BURNED_F3_P = {X**2, X**3, X**2 + X, X**4, X**3 + X**2,
                   X**4 + X, X**5, X**3 + X, X**4 + X**2}
    BURNED_F3_C = {2, 3, 4, 5}
    BURNED_F3_POLYS = {sp.sstr(X), sp.sstr(7 * X**3),
                       sp.sstr(2 * X), sp.sstr(5 * X**2)}
    for Pv in (X**6, X**5 + X, X**5 + X**2, X**4 + X**3,
               X**6 + X**2, X**5 + X**3):
        gate(Pv not in BURNED_F3_P, f"BURNED F3 P {Pv}")
    for cv in (6, 7):
        gate(cv not in BURNED_F3_C, f"BURNED F3 c {cv}")
    for pv in (3 * X, 11 * X**4):
        gate(sp.sstr(pv) not in BURNED_F3_POLYS,
             f"BURNED F3 POLY {pv}")
    for pr in F4_PAIRS:
        gate(pr not in BURNED_F4_PAIRS, f"BURNED PAIR {pr}")
    for fq in F4_FREQS:
        gate(fq not in BURNED_FREQS, f"BURNED FREQ {fq}")
    # burn set: training + bands + pilot + DESIGN-0 parents
    train_cur, seen, rs, rp = set(), set(), set(), set()
    for l in open(PAIRED):
        r = json.loads(l)
        train_cur.add(r["cur"])
        t = (r["rule"], r["site_kind"], r["site_ordinal"],
             r["param_kind"], r["param_index"])
        seen.add(t)
        rs.add((t[0], (t[1], t[2])))
        rp.add((t[0], (t[3], t[4])))
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
    gate(len(pilot_cur) == 566, f"PILOT PARENTS {len(pilot_cur)}")
    d0_cur = set()
    for fam, sig, f in d0_horizon():
        for D in (SMALL_D[0], SMALL_D[1], AFTER_D):
            d0_cur.add(sp.sstr(sp.Add(sp.Integral(f, X),
                                      sp.Integral(D, X))))
    gate(len(d0_cur) == 720, f"D0 PARENTS {len(d0_cur)}")
    burned_cur = train_cur | band_cur | pilot_cur | d0_cur
    # semantic gates
    for t in HELD_OUT_TERMS:
        cell = ("i_unprod", "I", 1, "term_index", t)
        gate(cell not in seen, f"HELD-OUT IN TRAINING {cell}")
        gate(("i_unprod", ("I", 1)) in rs, "RS NOT COVERED")
        gate(("i_unprod", ("term_index", t)) in rp,
             f"RP NOT COVERED t{t}")
    for (o, t) in ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0),
                   (1, 1)):
        gate(("i_unprod", "I", o, "term_index", t) in seen,
             f"COVERED CELL ABSENT I{o}t{t}")

    horizon = smoke_horizon() if SMOKE else build_horizon1()
    if not SMOKE:
        gate(len(horizon) == 720, f"HORIZON {len(horizon)}")
        gate(len({s for _, s, _ in horizon}) == 720, "SIG DUP")
        fts = [sp.srepr(f) for _, _, f in horizon]
        gate(len(fts) == len(set(fts)), "TARGET INTEGRAND DUP")
    OUTDIR.mkdir(parents=True)
    receipt = {"smoke": SMOKE, "prereg":
               "MATH-CYBER-1-SVP-STRICT-GRID-CHALLENGE-DESIGN-1",
               "n_horizon": len(horizon),
               "smoke_receipt_sha": (fsha(SMOKE_RECEIPT)
                                     if not SMOKE else None),
               "pins": {p: fsha(p) for p in PINS}}

    def finish(verdict, extra):
        receipt["verdict"] = verdict
        receipt.update(extra)
        receipt["blocks_sha"] = fsha(OUTDIR / "blocks.jsonl") \
            if (OUTDIR / "blocks.jsonl").exists() else None
        receipt["decisions_sha"] = fsha(
            OUTDIR / "decisions.jsonl") \
            if (OUTDIR / "decisions.jsonl").exists() else None
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "svpchal2_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ("start", "pins",
                                       "codeword_anatomy")},
                         indent=1), flush=True)

    blocks = []
    t_all = time.monotonic()
    for i, (fam, sig, f_t) in enumerate(horizon):
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
                      v["chosen_param_kind"]) for v in variants}
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
        blk["variants_pass"] = blk["fail"] is None
        blk["_variants"] = variants
        blocks.append(blk)
        if (i + 1) % 20 == 0 or SMOKE:
            print(f"[{i+1}/{len(horizon)}] {fam} "
                  f"{'t' + str(blk.get('term')) if blk['fail'] is None else blk['fail']}",
                  flush=True)
    with open(OUTDIR / "blocks.jsonl", "w") as fh:
        for b in blocks:
            fh.write(json.dumps(
                {k: v for k, v in b.items()
                 if k != "_variants"}) + "\n")
    strata = defaultdict(list)
    for b in blocks:
        if b["fail"] is None and b["term"] in (1, 2, 3):
            strata[(b["family"], b["term"])].append(b)
    for k in strata:
        strata[k].sort(key=lambda b: b["sig_sha"])
    stratum_counts = {f"{f}|t{t}": len(strata.get((f, t), []))
                      for f in ("CH-F3", "CH-F4")
                      for t in (1, 2, 3)}
    base = {
        "stratum_qualified_counts": stratum_counts,
        "block_fail_census": dict(Counter(
            b["fail"] for b in blocks if b["fail"])),
        "qualified_terms_census": dict(Counter(
            b.get("term") for b in blocks
            if b["fail"] is None)),
        "wall_s": round(time.monotonic() - t_all, 1)}
    short = [k for k, v in stratum_counts.items()
             if v < N_PER_STRATUM]
    if short:
        finish("CHALLENGE NO-FIRE", base | {
            "short_strata": short})
        return 0
    def refuted(reason):
        finish("CHALLENGE GATE-REFUTED",
               base | {"refuted_reason": reason})
        raise SystemExit(f"GATE-REFUTED: {reason}")

    dec_rows = []
    sel_census = Counter()
    all_curs = []
    dupsig = set()
    for f in ("CH-F3", "CH-F4"):
        for t in (1, 2, 3):
            sel = strata[(f, t)][:N_PER_STRATUM]
            for j, b in enumerate(sel):
                if b["base_signature"] in dupsig:
                    refuted("DUP SIG SELECTED")
                dupsig.add(b["base_signature"])
                primary_small = "smallA" if j % 2 == 0 \
                    else "smallB"
                robust_small = "smallB" if j % 2 == 0 \
                    else "smallA"
                by = {v["variant"]: v for v in b["_variants"]}
                role_i1 = ("control-I1" if t == 1
                           else "heldout-I1")
                role_i0 = ("control-I0" if t == 1
                           else "covered-I0")
                for role, v in (
                        (role_i1, by[primary_small]),
                        (role_i0, by["after"]),
                        ("robustness-I1", by[robust_small])):
                    dec_rows.append({
                        "block_id": f"{f}-t{t}-{j:02d}",
                        "family": f, "term_cell": t,
                        "site_role": role,
                        "primary": role != "robustness-I1",
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
                    all_curs.append(v["cur"])
                sel_census[f"{f}|t{t}|"
                           f"{by[primary_small]['distractor']}"
                           ] += 1

    half = N_PER_STRATUM - N_PER_STRATUM // 2
    for f in ("CH-F3", "CH-F4"):
        for t in (1, 2, 3):
            a = sel_census.get(f"{f}|t{t}|x**x", 0)
            bd = sel_census.get(f"{f}|t{t}|1/(x + log(x))", 0)
            if not (a == half and bd == N_PER_STRATUM // 2):
                refuted(f"D BALANCE {f} t{t} {a}/{bd}")
    if len(all_curs) != len(set(all_curs)):
        refuted("DUP PARENT")
    nov = {"train": sum(1 for c in all_curs if c in train_cur),
           "band": sum(1 for c in all_curs if c in band_cur),
           "pilot": sum(1 for c in all_curs if c in pilot_cur),
           "design0": sum(1 for c in all_curs if c in d0_cur)}
    if SMOKE:
        base["smoke_novelty_hits_expected_burned"] = nov
    elif any(nov.values()):
        refuted(f"BURNED CUR OVERLAP {nov}")
    # exact expected counts gate
    n_prim = sum(r["primary"] for r in dec_rows)
    exp_prim = 12 * N_PER_STRATUM
    if n_prim != exp_prim or len(dec_rows) != 18 * \
            N_PER_STRATUM or len(dupsig) != 6 * N_PER_STRATUM:
        refuted(f"COUNTS {len(dupsig)}/{n_prim}/{len(dec_rows)}")
    role_census = Counter(r["site_role"] for r in dec_rows)
    if not SMOKE:
        if not (role_census["heldout-I1"] == 24 * 2
                and role_census["covered-I0"] == 48
                and role_census["control-I1"] == 24
                and role_census["control-I0"] == 24):
            refuted(f"ROLE CENSUS {dict(role_census)}")
    with open(OUTDIR / "decisions.jsonl", "w") as fh:
        for r in dec_rows:
            fh.write(json.dumps(r) + "\n")
    # codeword anatomy rider (outcome-independent, model-free)
    train_rows = Counter()
    for l in open(PAIRED):
        r = json.loads(l)
        train_rows[(r["rule"], r["site_kind"],
                    r["site_ordinal"], r["param_kind"],
                    r["param_index"])] += 1
    tcodes = {tp: {"factor": factor_symbols(*tp),
                   "hash": hash_symbols(*tp), "count": cnt}
              for tp, cnt in train_rows.items()}
    rider = {}
    for t in HELD_OUT_TERMS:
        cell = ("i_unprod", "I", 1, "term_index", t)
        for kind, code in (("factor", factor_symbols(*cell)),
                           ("hash", hash_symbols(*cell))):
            dists = []
            for tp, tc in tcodes.items():
                d = sum(1 for a2, b2 in zip(code, tc[kind])
                        if a2 != b2)
                pfx = 0
                for a2, b2 in zip(code, tc[kind]):
                    if a2 != b2:
                        break
                    pfx += 1
                dists.append((d, pfx, " ".join(map(str, tp))))
            dists.sort()
            mind = dists[0][0]
            pos_support = []
            for i2, sym in enumerate(code):
                cs = Counter()
                for tp, tc in tcodes.items():
                    cs[tc[kind][i2]] += tc["count"]
                pos_support.append({
                    "position": i2, "symbol": sym,
                    "training_rows_with_symbol": cs.get(sym, 0),
                    "distinct_symbols_in_training": len(cs)})
            rider[f"I1t{t}:{kind}"] = {
                "code": code, "nearest_hamming": mind,
                "nearest_codewords": [x[2] for x in dists
                                      if x[0] == mind],
                "longest_shared_prefix": max(
                    p for _, p, _ in dists),
                "per_position_support": pos_support}
    tgt_sel = sorted({b["target_integrand"] for b in blocks
                      if b["base_signature"] in dupsig})
    finish("CHALLENGE MATERIALIZED", base | {
        "n_blocks_selected": len(dupsig),
        "n_states": len(dec_rows),
        "n_primary_states": n_prim,
        "site_role_census": dict(role_census),
        "d_before_census": dict(sel_census),
        "min_hce_ties_census": dict(Counter(
            r["min_hce_ties"] for r in dec_rows)),
        "legal_set_size_census": dict(Counter(
            r["n_candidates"] for r in dec_rows)),
        "target_integrand_overlap": {
            "training": sum(1 for ti in tgt_sel if any(
                ti in c for c in train_cur)),
            "bands": sum(1 for ti in tgt_sel if any(
                ti in c for c in band_cur))},
        "codeword_anatomy": rider})
    return 0


if __name__ == "__main__":
    sys.exit(main())

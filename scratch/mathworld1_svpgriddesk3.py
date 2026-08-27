"""MATH-CYBER-1 SVP-STRICT-GRID-CONSTRUCTIBILITY-DESK-0, SPACE-3 —
third frozen constructor space, committed BEFORE its first run.
SPACE-2 (svpgriddesk2_receipt.json) established teacher-selection
constructibility (143/148 successes, term indices 0-4) but every
success landed at site ordinal 0: the op-blocker distractor
always sorts AFTER the target in canonical Add order, so held-out
I1 cells stayed at zero (C-on-space-2). SPACE-3 replaces the
distractor axis with SMALL heurisch-proof integrands that sympy
cannot integrate (sp.integrate returns an unevaluated Integral,
filtered by i_heurisch's own F.has(Integral) gate): x**x and
1/(x + log(x)) sort BEFORE the target (target ordinal 1 — the
held-out band), sin(sin(x)) sorts AFTER it (target ordinal 0 —
matched covered controls from the SAME family/parameters). Same
success definition and A/B/C/D thresholds/precedence as v1
(docstring of scratch/mathworld1_svpgriddesk.py), applied to THIS
space. Zero model access; pilot parents BURNED.

FROZEN SPACE-3 (deterministic, each grid point once):
  U-F3v3: state = Integral(f_t, x) + Integral(D, x), where
    f_t = expand(d/dx[P*T(c*x)]) + Integral(w, x) + first-k polys
    P in {x^2, x^3, x^2+x}; T in {sin, cos}; c in {2,3};
    w in {exp(x)/x, sin(x)/x}; k in {0,1,2} over (x, 7*x^3);
    D in {x**x, 1/(x+log(x)), sin(sin(x))}.
    Grid: 3*2*2*2*3*3 = 216 attempts.
  P-F3 (SECONDARY, never gates): f_t = m * Q40,
    m in {log(x), atan(x)}; D in {x**x, sin(sin(x))}.
    Grid: 2*2 = 4 attempts. Total 220.
  Site ordinals/term indices MEASURED, never assumed.

Outputs logs/mathworld1/svpgriddesk3_receipt.json
(refuse-if-exists).

    .venv/bin/python scratch/mathworld1_svpgriddesk3.py       (Mac)
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpgriddesk import (COVERED_P,  # noqa: E402
                                            COVERED_U, HELD_OUT,
                                            HELD_OUT_P, PAIRED,
                                            PAIRED_SHA, X, ckey,
                                            fsha, run_attempt)
from scratch.mathworld1_svpgriddesk2 import Q40  # noqa: E402

RECEIPT = Path("logs/mathworld1/svpgriddesk3_receipt.json")
DISTRACTORS = (X**X, 1 / (X + sp.log(X)), sp.sin(sp.sin(X)))


def build_space3():
    polys = (X, 7 * X**3)
    space = []
    for P in (X**2, X**3, X**2 + X):
        for T in (sp.sin, sp.cos):
            for c in (2, 3):
                for w in (sp.exp(X) / X, sp.sin(X) / X):
                    for k in (0, 1, 2):
                        for D in DISTRACTORS:
                            f = (sp.expand(
                                sp.diff(P * T(c * X), X))
                                + sp.Integral(w, X)
                                + sp.Add(*polys[:k]))
                            space.append((
                                "U-F3v3", "i_unprod",
                                f"P={P} T={T.__name__} c={c} "
                                f"w={w} k={k} D={D}", f, D))
    for m in (sp.log(X), sp.atan(X)):
        for D in (X**X, sp.sin(sp.sin(X))):
            space.append(("P-F3", "i_parts", f"m={m} D={D}",
                          m * Q40, D))
    return space


def main():
    if RECEIPT.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} exists")
    gate(fsha(PAIRED) == PAIRED_SHA, "PAIRED PIN")
    for D in DISTRACTORS:
        F = sp.integrate(D, X)
        gate(F.has(sp.Integral), f"DISTRACTOR INTEGRABLE {D}")
    START = start_provenance(
        ["scratch/mathworld1_svpgriddesk3.py",
         "scratch/mathworld1_svpgriddesk2.py",
         "scratch/mathworld1_svpgriddesk.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    space = build_space3()
    gate(len(space) == 220, f"SPACE SIZE {len(space)}")
    rows = []
    for i, (fam, rule, sig, f, d) in enumerate(space):
        row = run_attempt(fam, rule, sig, f, d)
        rows.append(row)
        print(f"[{i+1}/{len(space)}] {fam} "
              f"intent={row.get('intended_cell')} "
              f"chosen={row.get('chosen_cell')} "
              f"outcome={row['outcome']}", flush=True)
    succ = defaultdict(set)
    succ_fams = defaultdict(set)
    fam_cells = defaultdict(set)
    intents = Counter()
    outcomes = Counter()
    setsizes = []
    comp = Counter()
    for r in rows:
        outcomes[r["outcome"]] += 1
        if r.get("intended_cell"):
            intents[r["intended_cell"]] += 1
        if "legal_set_size" in r:
            setsizes.append(r["legal_set_size"])
        if r.get("chosen_rule") and not r.get("success"):
            comp[r["chosen_rule"]] += 1
        if r.get("success"):
            cell = tuple(r["chosen_cell"].split())
            cell = (cell[0], cell[1], int(cell[2]), cell[3],
                    int(cell[4]))
            succ[cell].add(r["parent_sstr"])
            succ_fams[cell].add(r["family"])
            fam_cells[r["family"]].add(cell)
    dup_parent = len(rows) - len({r["parent_sha"] for r in rows})
    dup_sstr = len(rows) - len({r["parent_sstr"] for r in rows})

    def feasible(c):
        return len(succ[c]) >= 5 and len(succ_fams[c]) >= 2

    ho = sorted(HELD_OUT)
    ho_zero = [c for c in ho if len(succ[c]) == 0]
    ho_feas = [c for c in ho if feasible(c)]
    cov_hit = [c for c in sorted(COVERED_U) if len(succ[c]) >= 1]
    overlap_fams = [f for f, cs in fam_cells.items()
                    if (cs & HELD_OUT) and (cs & COVERED_U)]
    ho_only_fams = [f for f, cs in fam_cells.items()
                    if cs and cs <= HELD_OUT]
    if ho_zero:
        verdict = "C NOT-CONSTRUCTIBLE"
    elif len(ho_feas) < 2:
        verdict = "D TOO-SCARCE"
    elif (len(cov_hit) >= 4 and overlap_fams
          and not ho_only_fams):
        verdict = "A CLEAN-GRID-FEASIBLE"
    else:
        verdict = "B TARGET-ONLY-FEASIBLE"
    # per-distractor cell crosstab: the fingerprint for
    # template-separability of held-out v covered successes
    dcross = Counter()
    for r in rows:
        if r.get("success"):
            D = r["signature"].split("D=")[-1]
            dcross[f"{D} | {r['chosen_cell']}"] += 1
    receipt = {
        "space": "SPACE-3",
        "verdict": verdict,
        "cell_summary": {
            ckey(c): {"successes": len(succ[c]),
                      "families": sorted(succ_fams[c]),
                      "feasible": feasible(c),
                      "class": ("HELD-OUT" if c in
                                HELD_OUT | HELD_OUT_P
                                else "COVERED")}
            for c in sorted(set(succ) | HELD_OUT | COVERED_U
                            | HELD_OUT_P | COVERED_P)},
        "held_out_zero": [ckey(c) for c in ho_zero],
        "covered_controls_hit": [ckey(c) for c in cov_hit],
        "family_overlap": sorted(overlap_fams),
        "held_out_only_families": sorted(ho_only_fams),
        "distractor_x_cell": {k: v for k, v in
                              dcross.most_common()},
        "outcomes": dict(outcomes),
        "intended_cell_distribution": dict(intents),
        "legal_set_size_distribution": dict(Counter(setsizes)),
        "competing_chosen_rules_on_failure": dict(comp),
        "duplicate_parent_count": dup_parent,
        "duplicate_visible_state_count": dup_sstr,
        "n_attempts": len(rows),
        "attempts": rows,
        "pins": {str(PAIRED): fsha(PAIRED)},
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    out = {k: v for k, v in receipt.items()
           if k not in ("attempts", "start", "pins")}
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

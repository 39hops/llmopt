"""MATH-CYBER-1 SVP-STRICT-GRID-CONSTRUCTIBILITY-DESK-0, SPACE-2 —
second frozen constructor space, committed BEFORE its first run.
Space v1 (scratch/mathworld1_svpgriddesk.py, receipt
svpgriddesk_receipt.json) resolved C-on-that-space: i_heurisch
(op-capped sympy.integrate) closed every site because the v1
"hard" terms all have special-function antiderivatives (Ei, Si,
Fresnel S, erfi). SPACE-2 uses the blockers i_heurisch actually
gates on (rules.py: count_ops > 100, or integrand containing an
inner Integral) — the shape of the natural states where i_unprod
trained. Same success definition, same A/B/C/D thresholds and
precedence as v1 (docstring of the v1 driver), applied to THIS
space. Zero model access; pilot parents BURNED.

FROZEN SPACE-2 (deterministic, each grid point once):
  U-F3: state = Integral(B, x) + Integral(f_t, x), where
    f_t = expand(d/dx[P*T(c*x)]) + Integral(w, x) + first-k polys
    P in {x^2, x^3, x^2+x}; T in {sin, cos}; c in {2,3};
    w in {exp(x)/x, sin(x)/x}; k in {0,1,2} over (x, 7*x^3);
    B in {B16s, B12c} with
    B16s = sum_{j=1..16} (j%5+2)*x^j*sin((j+2)x),
    B12c = sum_{j=1..12} (j%4+2)*x^j*cos((j+3)x),
    both gate-checked count_ops > 100 (heurisch refuses).
    Grid: 3*2*2*2*3*2 = 144 attempts.
  P-F2 (SECONDARY, never gates): f_t = m * Q with
    m in {log(x), atan(x)}, Q in {Q40, Q28} where
    Qn = sum_{j=1..n} (j%3+1)*x^j (count_ops > 100 gate-checked
    for m*Q); distractor B = B16s. Grid: 2*2 = 4 attempts.
  Total 148. Site ordinals/term indices MEASURED, never assumed
  (canonical Add order decides which summand is ordinal 0; the
  inner blocker Integral inside f_t adds a nested site whose
  ordinal is also measured).

Outputs logs/mathworld1/svpgriddesk2_receipt.json
(refuse-if-exists).

    .venv/bin/python scratch/mathworld1_svpgriddesk2.py       (Mac)
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

RECEIPT = Path("logs/mathworld1/svpgriddesk2_receipt.json")

B16S = sp.Add(*[(j % 5 + 2) * X**j * sp.sin((j + 2) * X)
                for j in range(1, 17)])
B12C = sp.Add(*[(j % 4 + 2) * X**j * sp.cos((j + 3) * X)
                for j in range(1, 13)])
Q40 = sp.Add(*[(j % 3 + 1) * X**j for j in range(1, 41)])
Q28 = sp.Add(*[(j % 3 + 1) * X**j for j in range(1, 29)])


def build_space2():
    polys = (X, 7 * X**3)
    space = []
    for P in (X**2, X**3, X**2 + X):
        for T in (sp.sin, sp.cos):
            for c in (2, 3):
                for w in (sp.exp(X) / X, sp.sin(X) / X):
                    for k in (0, 1, 2):
                        for bn, B in (("B16s", B16S),
                                      ("B12c", B12C)):
                            f = (sp.expand(
                                sp.diff(P * T(c * X), X))
                                + sp.Integral(w, X)
                                + sp.Add(*polys[:k]))
                            space.append((
                                "U-F3", "i_unprod",
                                f"P={P} T={T.__name__} c={c} "
                                f"w={w} k={k} B={bn}", f, B))
    for m in (sp.log(X), sp.atan(X)):
        for qn, Q in (("Q40", Q40), ("Q28", Q28)):
            space.append(("P-F2", "i_parts", f"m={m} Q={qn}",
                          m * Q, B16S))
    return space


def main():
    if RECEIPT.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} exists")
    gate(fsha(PAIRED) == PAIRED_SHA, "PAIRED PIN")
    gate(sp.count_ops(B16S) > 100, "B16S NOT OP-BLOCKED")
    gate(sp.count_ops(B12C) > 100, "B12C NOT OP-BLOCKED")
    gate(sp.count_ops(sp.log(X) * Q40) > 100, "Q40 NOT BLOCKED")
    gate(sp.count_ops(sp.log(X) * Q28) > 100, "Q28 NOT BLOCKED")
    START = start_provenance(
        ["scratch/mathworld1_svpgriddesk2.py",
         "scratch/mathworld1_svpgriddesk.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    space = build_space2()
    gate(len(space) == 148, f"SPACE SIZE {len(space)}")
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
    receipt = {
        "space": "SPACE-2",
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

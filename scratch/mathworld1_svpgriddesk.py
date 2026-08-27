"""MATH-CYBER-1 SVP-STRICT-GRID-CONSTRUCTIBILITY-DESK-0 —
SACRIFICIAL exact-engine constructibility pilot for the 3 latent
strict cells (semantic-support result frozen at b2199346). Zero
model/checkpoint access, zero inference, zero training, zero
scoring. Pilot parents are BURNED: they may never enter a final
challenge.

Question: can exact-engine parent states be constructed in which
the held-out grid cells (i_unprod I1 term2/term3; secondary
i_parts I1 u_choice0) arise as genuine UNIQUE teacher-selected
actions under the standing legal-set/controller law, with matched
covered controls from the SAME construction families?

FROZEN CONSTRUCTOR SPACE (deterministic; every grid point run
exactly once, no hidden retry) — committed BEFORE the first run:
  U-F1: target integrand expand(d/dx[P*T(c*x)]) + h + first-k
    inert polys; P in {x^2, x^3, x^2+x}; T in {sin, cos};
    c in {2,3}; h in {sin(x^2), exp(x^2)}; k in {0,1,2} over
    (x, 7*x^3); distractor in {exp(x)/x, sin(x)/x}. 144 attempts.
  U-F2: expand(d/dx[P1*sin(2x)]) + expand(d/dx[P2*cos(3x)]) + h;
    (P1,P2) in {(x^3,x),(x^2+x,x),(x^3,x^2)}; h and distractor as
    U-F1's lists. 12 attempts.
  P-F1 (SECONDARY, never gates): target in {x*log(x),
    x^2*log(x), x*atan(x), log(x)*atan(x)}; distractor in
    {exp(x)/x, sin(x)/x, exp(x^2)}. 12 attempts.
State = Integral(target, x) + Integral(distractor, x); site
ordinals and term indices are MEASURED from the final parent
(preorder-unique site law / canonical Add order), never assumed.

SUCCESS for cell X: legal set stable, every candidate's
ActionProgram derives exactly, teacher argmin (hce, name, key)
lands on X, and X == the intended cell (computed deterministically
pre-teacher: measured target-site ordinal x the within-rule
minimal-hce component index). Distinct successes = distinct
parent sstr.

FROZEN DECISION (i_unprod grid only; precedence C > D > B > A):
  cell FEASIBLE iff >= 5 distinct successes from >= 2 families;
  A CLEAN-GRID-FEASIBLE: both held-out cells FEASIBLE and >= 4/6
    covered controls have >= 1 success and >= 1 family succeeds
    on both a held-out and a covered cell and no family is
    held-out-only;
  B TARGET-ONLY-FEASIBLE: both held-out FEASIBLE, matched-control
    clause fails;
  C NOT-CONSTRUCTIBLE: a held-out cell has zero successes;
  D TOO-SCARCE: held-out successes >= 1 each but a held-out cell
    not FEASIBLE.

Outputs logs/mathworld1/svpgriddesk_receipt.json
(refuse-if-exists).

    .venv/bin/python scratch/mathworld1_svpgriddesk.py        (Mac)
"""
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import State, hce  # noqa: E402
from scratch.mathworld1_actionsem import sites_preorder  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpeval import (derive_program,  # noqa: E402
                                        stable_legal_set)
from scratch.mathworld1_unprodsem import trace_unprod  # noqa: E402

X = sp.Symbol("x")
PAIRED = Path("data/matsub_paired.jsonl")
PAIRED_SHA = ("a943ba7fc581db743b07192e5d951fadd"
              "dd2ba19bca3225b75d8402351d468e8")
RECEIPT = Path("logs/mathworld1/svpgriddesk_receipt.json")
WALL_CAP_ATTEMPT_S = 120.0

HELD_OUT = {("i_unprod", "I", 1, "term_index", 2),
            ("i_unprod", "I", 1, "term_index", 3)}
COVERED_U = {("i_unprod", "I", 0, "term_index", i)
             for i in range(4)} | {
    ("i_unprod", "I", 1, "term_index", i) for i in range(2)}
HELD_OUT_P = {("i_parts", "I", 1, "u_choice", 0)}
COVERED_P = {("i_parts", "I", 0, "u_choice", 0),
             ("i_parts", "I", 0, "u_choice", 1),
             ("i_parts", "I", 1, "u_choice", 1)}


def sha16(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ckey(cell):
    return " ".join(map(str, cell))


def build_space():
    """The frozen product grids, in deterministic order."""
    polys = (X, 7 * X**3)
    space = []
    for P in (X**2, X**3, X**2 + X):
        for T in (sp.sin, sp.cos):
            for c in (2, 3):
                for h in (sp.sin(X**2), sp.exp(X**2)):
                    for k in (0, 1, 2):
                        for d in (sp.exp(X) / X, sp.sin(X) / X):
                            f = (sp.expand(
                                sp.diff(P * T(c * X), X))
                                + h + sp.Add(*polys[:k]))
                            space.append((
                                "U-F1", "i_unprod",
                                f"P={P} T={T.__name__} c={c} "
                                f"h={h} k={k} d={d}", f, d))
    for (P1, P2) in ((X**3, X), (X**2 + X, X), (X**3, X**2)):
        for h in (sp.sin(X**2), sp.exp(X**2)):
            for d in (sp.exp(X) / X, sp.sin(X) / X):
                f = (sp.expand(sp.diff(P1 * sp.sin(2 * X), X))
                     + sp.expand(sp.diff(P2 * sp.cos(3 * X), X))
                     + h)
                space.append(("U-F2", "i_unprod",
                              f"P1={P1} P2={P2} h={h} d={d}",
                              f, d))
    for m in (X * sp.log(X), X**2 * sp.log(X), X * sp.atan(X),
              sp.log(X) * sp.atan(X)):
        for d in (sp.exp(X) / X, sp.sin(X) / X, sp.exp(X**2)):
            space.append(("P-F1", "i_parts", f"m={m} d={d}",
                          m, d))
    return space


def intended_cell(parent, node, target_rule):
    """Deterministic pre-teacher intent: within the target rule at
    the target site, the component (term/u ordinal) whose child
    has minimal hce (ties: lowest index). Returns (cell tuple,
    fail reason)."""
    sites = sites_preorder(parent, "I")
    ords = [i for i, s in enumerate(sites) if s == node]
    if len(ords) != 1:
        return None, "site_not_unique"
    ordinal = ords[0]
    f = node.function
    best = None
    if target_rule == "i_unprod":
        if not isinstance(f, sp.Add):
            return None, "not_add"
        for (i, _a, _fam, A, resid) in trace_unprod(f, X):
            child = parent.xreplace({node: A + (
                sp.Integral(resid, X) if resid != 0 else 0)})
            v = hce(State(child))
            if best is None or (v, i) < best:
                best = (v, i)
        if best is None:
            return None, "no_unprod_candidate"
        return ("i_unprod", "I", ordinal, "term_index",
                best[1]), None
    if not isinstance(f, sp.Mul):
        return None, "not_mul"
    elig = 0
    for i, u in enumerate(f.args):
        if sp.diff(u, X) == 0:
            continue
        dv = sp.Mul(*(a for j, a in enumerate(f.args) if j != i))
        v_int = sp.Integral(dv, X)
        child = parent.xreplace(
            {node: u * v_int - sp.Integral(v_int * sp.diff(u, X),
                                           X)})
        v = hce(State(child))
        if best is None or (v, elig) < best:
            best = (v, elig)
        elig += 1
    if best is None:
        return None, "no_parts_candidate"
    return ("i_parts", "I", ordinal, "u_choice", best[1]), None


def run_attempt(fam, target_rule, sig, target_f, distract):
    t0 = time.monotonic()
    node = sp.Integral(target_f, X)
    parent = sp.Add(node, sp.Integral(distract, X))
    row = {"family": fam, "signature": sig,
           "parent_sstr": sp.sstr(parent),
           "parent_sha": sha16(sp.srepr(parent))}
    intent, why = intended_cell(parent, node, target_rule)
    row["intended_cell"] = ckey(intent) if intent else None
    if intent is None:
        row["outcome"] = f"intent_{why}"
        return row
    acts, stable = stable_legal_set(State(parent))
    row["legal_set_stable"] = stable
    row["legal_set_size"] = len(acts)
    if not stable:
        row["outcome"] = "legal_set_unstable"
        return row
    if not acts:
        row["outcome"] = "no_legal_actions"
        return row
    accepted = defaultdict(set)
    for n, c in acts:
        accepted[n.split("@", 1)[0] if "@" in n else n].add(
            c.key())
    cells = {}
    for n, c in acts:
        rule = n.split("@", 1)[0] if "@" in n else n
        prog, pwhy = derive_program(parent, rule, c.key(),
                                    accepted)
        if prog is None:
            row["outcome"] = f"program_{pwhy}"
            row["program_fail_rule"] = rule
            return row
        cells[(n, c.key())] = (
            prog["rule"], prog["site_kind"],
            prog["site_ordinal"], prog["param_kind"],
            prog["param_index"])
    row["target_legal"] = intent in set(cells.values())
    scored = [(hce(c), n, c.key()) for n, c in acts]
    mn = min(scored)
    row["min_hce_ties"] = sum(1 for s in scored
                              if s[0] == mn[0])
    chosen_cell = cells[(mn[1], mn[2])]
    row["chosen_cell"] = ckey(chosen_cell)
    row["chosen_rule"] = chosen_cell[0]
    row["competing_rules"] = dict(Counter(
        v[0] for v in cells.values()))
    row["success"] = (chosen_cell == intent)
    row["outcome"] = "success" if row["success"] else "chosen_other"
    row["wall_s"] = round(time.monotonic() - t0, 2)
    if row["wall_s"] > WALL_CAP_ATTEMPT_S:
        row["outcome"] = "wall_cap"
        row["success"] = False
    return row


def main():
    if RECEIPT.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} exists")
    gate(fsha(PAIRED) == PAIRED_SHA, "PAIRED PIN")
    START = start_provenance(
        ["scratch/mathworld1_svpgriddesk.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld0.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/lab/provenance.py"])
    # grid gate: re-derive the C/H classification from training
    seen = set()
    for l in open(PAIRED):
        r = json.loads(l)
        seen.add((r["rule"], r["site_kind"], r["site_ordinal"],
                  r["param_kind"], r["param_index"]))
    gate(all(c not in seen for c in HELD_OUT | HELD_OUT_P),
         "HELD-OUT CELL PRESENT IN TRAINING")
    gate(all(c in seen for c in COVERED_U | COVERED_P),
         "COVERED CONTROL ABSENT FROM TRAINING")
    space = build_space()
    gate(len(space) == 168, f"SPACE SIZE {len(space)}")
    rows = []
    for i, (fam, rule, sig, f, d) in enumerate(space):
        row = run_attempt(fam, rule, sig, f, d)
        rows.append(row)
        print(f"[{i+1}/{len(space)}] {fam} "
              f"intent={row.get('intended_cell')} "
              f"outcome={row['outcome']}", flush=True)
    # aggregates
    succ = defaultdict(set)       # cell -> distinct parent sstrs
    succ_fams = defaultdict(set)  # cell -> families
    fam_cells = defaultdict(set)  # family -> success cells
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

"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-V2-PRODUCTION-DESIGN-ASSESSMENT-0
— READ-ONLY census over the burned V2 desk artifact
(logs/mathworld1/prband2desk/bases.jsonl): every variant x e_lo
branch cell (six), with roles taken from the persisted source-term /
guessed-A semantics, the branch's own correct-guess coordinates, the
authoritative full-legal-set teacher, strict classification by theta,
exact cand_sig_id capacity, and the state-blind fixed-ranking ceiling
under balance. Plus the exact training-support matrix for (i_unprod,
I, ordinal in {0,1}, term_index 0..5) from the two pinned training
files (74,860 rows): whole-coordinate, rule+param, rule+site,
program_text counts, FACTOR and PARAM-FIRST codes with per-position
symbol counts. No parent generated, no rule call, no model.
Writes logs/mathworld1/prband2desk_verify/branch_census.json and
support_matrix.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_prband2branch.py
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratch.mathworld1_svpcode import factor_symbols  # noqa: E402
from scratch.mathworld1_svpforder import pf_encode  # noqa: E402

SRC = Path("logs/mathworld1/prband2desk/bases.jsonl")
OUTD = Path("logs/mathworld1/prband2desk_verify")
OUT1 = OUTD / "branch_census.json"
OUT2 = OUTD / "support_matrix.json"
TRAIN = ("data/matsub_paired.jsonl",
         "logs/mathworld1/svpdiet/balanced_grid_train.jsonl")
THETA = ("SIN_LOW", "COS_LOW")


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def branch(rows, vt, e_lo):
    cell = [(r, r["variants"][vt]) for r in rows if r["e_lo"] == e_lo]
    n = len(cell)
    sigs = Counter(v["cand_sig_id"] for _, v in cell)
    sigjs = {v["cand_sig_id"]: v["cand_sig"] for _, v in cell}
    # role -> coordinate census (from persisted semantics)
    role_coord = defaultdict(Counter)
    coords = Counter()
    for _, v in cell:
        for c in v["candidates"]:
            if c["rule"] == "i_unprod":
                role_coord[c["role"]][json.dumps(list(tup(c)))] += 1
                coords[json.dumps(list(tup(c)))] += 1
    # branch targets: the coordinate carrying COS_CORRECT / SIN_CORRECT
    # in every parent of the cell (else the branch has no fixed pair)
    def fixed(role):
        cs = role_coord.get(role, Counter())
        return (json.loads(next(iter(cs)))
                if len(cs) == 1 and cs[next(iter(cs))] == n else None)
    A = fixed("COS_CORRECT")
    B = fixed("SIN_CORRECT")
    site = Counter(tuple(v["teacher"])[2] for _, v in cell)
    cls = Counter()
    by_theta = {t: Counter() for t in THETA}
    gaps = Counter()
    ties = Counter(v["min_hce_ties"] for _, v in cell)
    groups = defaultdict(lambda: Counter())
    for r, v in cell:
        t = tuple(v["teacher"])
        tups = [tuple(x) for x in v["cand_tuples"]]
        if A is None or B is None or tuple(A) not in tups \
                or tuple(B) not in tups:
            lab = "TARGET-ABSENT"
        elif t not in (tuple(A), tuple(B)):
            lab = "NON-TARGET-TEACHER"
        elif v["min_hce_ties"] != 1:
            lab = "TIE"
        else:
            other = tuple(B) if t == tuple(A) else tuple(A)
            oh = [c["hce"] for c in v["candidates"] if tup(c) == other][0]
            gap = round(oh - v["teacher_hce"], 3)
            gaps[str(gap)] += 1
            lab = ("STRICT-A" if t == tuple(A) else "STRICT-B") \
                if gap >= 1.0 else "TIE"
        cls[lab] += 1
        by_theta[r["theta"]][lab] += 1
        if lab.startswith("STRICT"):
            groups[v["cand_sig_id"]][lab[-1]] += 1
    cap = {g: {"n_A": c["A"], "n_B": c["B"],
               "m_g": min(c["A"], c["B"])} for g, c in groups.items()}
    sum_m = sum(x["m_g"] for x in cap.values())
    # state-blind fixed-ranking ceiling: per signature group max gold
    # count over all strict states; balanced iff n_A == n_B everywhere
    strict_n = cls["STRICT-A"] + cls["STRICT-B"]
    ceil = sum(max(c["A"], c["B"]) for c in groups.values())
    balanced = all(c["A"] == c["B"] for c in groups.values()) and strict_n
    return {"variant": vt, "e_lo": e_lo, "n_parents": n,
            "teacher_site_ordinal": dict(site),
            "cand_sig_census": dict(sigs),
            "cand_sig": {k[:16]: json.loads(v) for k, v in sigjs.items()},
            "unprod_coordinates": dict(coords),
            "role_to_coordinate": {r: dict(c) for r, c in
                                   role_coord.items()},
            "branch_target_A_cos_correct": A,
            "branch_target_B_sin_correct": B,
            "min_hce_ties_census": dict(ties),
            "class_census": dict(cls),
            "by_theta": {t: dict(c) for t, c in by_theta.items()},
            "gap_A_B_census": dict(gaps),
            "capacity": cap, "sum_m_g": sum_m,
            "n_signatures": len(sigs), "n_mixed": sum(
                1 for x in cap.values() if x["m_g"] > 0),
            "fixed_ranking_ceiling_over_strict":
                {"best": ceil, "n": strict_n,
                 "frac": round(ceil / strict_n, 4) if strict_n else None,
                 "winner_balanced": bool(balanced)},
            "strict_state_dependent_crossover": bool(
                cls["STRICT-A"] and cls["STRICT-B"] and not cls["TIE"]
                and not cls["NON-TARGET-TEACHER"]
                and not cls["TARGET-ABSENT"])}


def main():
    if OUT1.exists() or OUT2.exists():
        raise SystemExit("REFUSING: outputs exist")
    rows = [json.loads(l) for l in open(SRC)]
    assert len(rows) == 64
    cells = {}
    for vt in ("smallA", "smallB", "after"):
        for e_lo in (2, 1):
            cells[f"{vt}|e_lo={e_lo}"] = branch(rows, vt, e_lo)
    OUT1.write_text(json.dumps({"source": str(SRC),
                                "source_sha256": fsha(SRC),
                                "cells": cells}, indent=1))
    # ---- support matrix -----------------------------------------
    whole, rp, rs, prog = Counter(), Counter(), Counter(), Counter()
    fpos = [Counter() for _ in range(8)]
    ppos = [Counter() for _ in range(8)]
    n = 0
    for p in TRAIN:
        for l in open(p):
            r = json.loads(l)
            n += 1
            t = (r["rule"], r["site_kind"], r["site_ordinal"],
                 r["param_kind"], r["param_index"])
            whole[t] += 1
            rp[(t[0], t[3], t[4])] += 1
            rs[t[:3]] += 1
            prog[r.get("program_text", "")] += 1
            fc = factor_symbols(*t)
            pf = pf_encode(t)
            for j in range(8):
                fpos[j][fc[j]] += 1
                ppos[j][pf[j]] += 1
    mat = {}
    for so in (0, 1):
        for ti in range(6):
            t = ("i_unprod", "I", so, "term_index", ti)
            fc = factor_symbols(*t)
            pf = pf_encode(t)
            fs = [fpos[j][fc[j]] for j in range(8)]
            ps = [ppos[j][pf[j]] for j in range(8)]
            mat[f"I{so}/t{ti}"] = {
                "tuple": list(t), "whole": whole[t],
                "rule_param": rp[(t[0], t[3], t[4])],
                "rule_site": rs[t[:3]],
                "program_text": f"<r:i_unprod> I{so} t{ti}\n",
                "program_text_count": prog[f"<r:i_unprod> I{so} t{ti}\n"],
                "factor_code": fc, "factor_pos_counts": fs,
                "pf_code": pf, "pf_pos_counts": ps,
                "all_positions_supported": all(fs) and all(ps)}
    OUT2.write_text(json.dumps({"n_rows": n,
                                "pins": {p: fsha(p) for p in TRAIN},
                                "matrix": mat}, indent=1))
    for k, c in cells.items():
        print(k, "n", c["n_parents"], "site", c["teacher_site_ordinal"],
              "sigs", c["n_signatures"], "A", c["branch_target_A_cos_correct"],
              "B", c["branch_target_B_sin_correct"], "cls", c["class_census"],
              "by_theta", c["by_theta"], "gaps", c["gap_A_B_census"],
              "ties", c["min_hce_ties_census"], "sum_m", c["sum_m_g"],
              "ceil", c["fixed_ranking_ceiling_over_strict"],
              "xover", c["strict_state_dependent_crossover"])
        print("   coords", c["unprod_coordinates"])
        print("   roles", c["role_to_coordinate"])
    print("rows", n)
    for k, v in mat.items():
        print(k, "whole", v["whole"], "rp", v["rule_param"], "rs", v["rule_site"],
              "prog", v["program_text_count"], "F", v["factor_code"],
              v["factor_pos_counts"], "PF", v["pf_code"], v["pf_pos_counts"],
              "ok", v["all_positions_supported"])


if __name__ == "__main__":
    main()

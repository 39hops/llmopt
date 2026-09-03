"""MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-ASSESSMENT-0
model-blind exhaustive global role-permutation render atlas.

Adopts the nuisance assessment's anatomy and printer verbatim
(scratch/mathworld1_prband2nuis.py: classify_terms, render,
view_string, surface, check_view, ceiling; importing that module
installs its torch.load trap, so no checkpoint can be read by this
process). The render class is exactly the 720 global permutations
of the six structural roles (HI_D, HI_L, LO_D, LO_L, K, W); one
permutation is applied identically to all 96 frozen states. A
policy may see nothing but the six role-resolved terms.

For every (policy, state) the cur string is materialised and
gated: parse, srepr identity with the frozen parent, simplify
difference zero, six-term multiset identity, candidate-law
reproduction (cand_sig_id / teacher / child srepr set / four
candidate programs / tie count), tokenizer round trip, prompt +
T=9 within CTX, parent srepr pin (state identity). Any policy
failing any state on any gate is ATLAS-INELIGIBLE and is not
repaired. Duplicate policies are detected only by byte identity
of the 96-state prompt matrix and are never collapsed here.

Surface census per policy (by theta): prompt token count, leading
sign, first-minus position, first trig function, first trig
degree, first polynomial degree, first role, role position of the
negative term; each with its best fixed feature->theta mapping
(gold is a function of theta under the frozen law, so the theta
ceiling is the gold ceiling).

Anchor gate: which policies reproduce the booked RAW / K_FIRST /
LOW_PAIR_FIRST prompt matrices byte-for-byte 96/96 (read from the
pinned nuisance views.jsonl; no model score is read).

Outputs (OUTDIR): atlas_manifest.jsonl (720 policies, written and
hashed before any render), renders.jsonl (69,120 rows, streamed as
policies complete), atlas_policies.jsonl (one row per policy),
prband2atlas_receipt.json. Policies are rendered by a fork Pool;
the receipt records worker count and wall.

Usage:
    PRBAND2A_SMOKE=1 .venv/bin/python scratch/mathworld1_prband2atlas.py
    .venv/bin/python scratch/mathworld1_prband2atlas.py
"""
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_prband2nuis import (AFTER_D, PRIMARY,  # noqa: E402
                                            PROMPT, ROLES, TOK, X,
                                            _no_load, ceiling,
                                            check_view, classify_terms,
                                            surface, view_string)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import fsha, qualify_parent  # noqa: E402
import torch  # noqa: E402

SMOKE = os.environ.get("PRBAND2A_SMOKE") == "1"
ASSESS = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-ASSESSMENT-0"
ADOPTED = {
    "verdict": "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-NUISANCE-COUNTERFACTUAL-0",
    "verdict_commit": "08d6529c105385a4dfdc6f2c8306e7db457839fe",
    "assessment":
        "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-NUISANCE-COUNTERFACTUAL-ASSESSMENT-0",
    "assessment_commit": "62ac8f3d86c44540eb2fe47965342d11f90cda82"}
NUIS_VIEWS = "logs/mathworld1/prband2nuis/views.jsonl"
CF_RECEIPT = "logs/mathworld1/prband2cf/prband2cf_receipt.json"
PINS = {PRIMARY:
        "209391ef3b2e5c87308571d6ef309bb5724a214160caa1b7857f4a31f9112c34",
        NUIS_VIEWS:
        "677201ccc0cf34fbdf2b2e060146b68c157a2450926ac062f1c0f16cac8a72bb"}
OUTDIR = Path("logs/mathworld1/prband2atlas_smoke" if SMOKE
              else "logs/mathworld1/prband2atlas")
ANCHORS = ("RAW", "K_FIRST", "LOW_PAIR_FIRST")
WORKERS = int(os.environ.get("PRBAND2A_WORKERS", "8"))
T_CONT = 9
FEATS = ("prompt_tokens", "lead_sign", "minus_pos", "first_trig",
         "first_trig_degree", "first_poly_degree", "first_role",
         "neg_role_pos")

_STATES = None  # set in the parent before the fork


def render_id(perm):
    return hashlib.sha256(json.dumps(list(perm)).encode()).hexdigest()


def manifest():
    """All 6! global role permutations in itertools order over ROLES."""
    rows = []
    for i, perm in enumerate(itertools.permutations(ROLES)):
        rows.append({"atlas_index": i, "roles": list(perm),
                     "render_id": render_id(perm)})
    gate(len(rows) == 720 and len({r["render_id"] for r in rows}) == 720,
         "MANIFEST 720")
    return rows


def prep_states(P):
    out = []
    for r in P:
        parent = sp.sympify(r["cur"])
        gate(sp.sstr(parent) == r["cur"], "RAW IS SSTR")
        gate(hashlib.sha256(sp.srepr(parent).encode()).hexdigest()[:16]
             == r["parent_srepr_sha"], "PARENT SREPR PIN")
        f = [a for a in parent.args if isinstance(a, sp.Integral)
             and a.function != AFTER_D][0].function
        roles, _fam, theta, signs, e_hi = classify_terms(f)
        gate(theta == r["theta"] and e_hi == r["e_hi"], "ANATOMY")
        progs = sorted((c["rule"], c["site_kind"], c["site_ordinal"],
                        c["param_kind"], c["param_index"])
                       for c in r["candidates"])
        gate(len(progs) == 4, "FOUR CANDIDATES")
        out.append({"row": r, "parent": parent, "f": f, "roles": roles,
                    "signs": signs, "progs": progs})
    return out


def poly_degree(t):
    if isinstance(t, sp.Integral):
        return None
    fs = list(t.atoms(sp.sin, sp.cos))
    p = sp.cancel(t / fs[0]) if fs else t
    return int(sp.degree(p, X))


def render_policy(job):
    """One policy over every state; runs inside a fork worker."""
    idx, perm = job
    rows = []
    for st in _STATES:
        r, roles = st["row"], st["roles"]
        terms = [roles[k] for k in perm]
        cur = view_string(perm, roles, AFTER_D)
        chk = check_view(r, st["parent"], st["f"], cur)
        p2 = sp.sympify(cur)
        chk["state_identity"] = (hashlib.sha256(
            sp.srepr(p2).encode()).hexdigest()[:16]
            == r["parent_srepr_sha"])
        f2 = [a for a in p2.args if isinstance(a, sp.Integral)
              and a.function != AFTER_D][0].function
        q, why = qualify_parent(f2, AFTER_D)
        chk["programs_equal"] = why is None and sorted(
            (c["rule"], c["site_kind"], c["site_ordinal"],
             c["param_kind"], c["param_index"])
            for c in q["candidates"]) == st["progs"]
        chk["gold_equal"] = why is None and (
            [q["chosen_rule"], q["chosen_site_kind"], q["chosen_ordinal"],
             q["chosen_param_kind"], q["chosen_term"]] == r["teacher"])
        sv = surface(cur, terms)
        neg = [i for i, k in enumerate(perm) if st["signs"].get(k) == -1]
        gate(len(neg) == 1, "ONE NEGATIVE TERM")
        rows.append({"atlas_index": idx, "pair_id": r["pair_id"],
                     "theta": r["theta"], "cur": cur, "checks": chk,
                     **sv, "first_poly_degree": poly_degree(terms[0]),
                     "first_role": perm[0], "neg_role_pos": neg[0],
                     "identical_to_raw": cur == r["cur"]})
    return idx, rows


def main():
    START = start_provenance(
        ["scratch/mathworld1_prband2atlas.py",
         "scratch/mathworld1_prband2nuis.py", "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_prband.py", "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_svpbirth.py", "llmopt/lab/provenance.py"])
    gate(torch.load is _no_load, "TORCH LOAD TRAP")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN {p}")
    t0 = time.monotonic()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    man_path = OUTDIR / "atlas_manifest.jsonl"
    ren_path = OUTDIR / "renders.jsonl"
    pol_path = OUTDIR / "atlas_policies.jsonl"
    for p in (man_path, ren_path, pol_path):
        gate(not p.exists(), f"REFUSE OVERWRITE {p}")
    MAN = manifest()
    man_path.write_text("".join(json.dumps(m) + "\n" for m in MAN))
    man_sha = fsha(str(man_path))
    print("manifest", len(MAN), man_sha)

    P = [json.loads(l) for l in open(PRIMARY)]
    gate(len(P) == 96, "N=96")
    if SMOKE:
        P = P[:4]
        MAN = MAN[:6]
    global _STATES
    _STATES = prep_states(P)
    order = [(r["pair_id"], r["theta"]) for r in P]

    # booked anchor matrices, read from the pinned nuisance views
    anchor = {a: {} for a in ANCHORS}
    for l in open(NUIS_VIEWS):
        v = json.loads(l)
        if v["view"] in anchor:
            anchor[v["view"]][(v["pair_id"], v["theta"])] = v["cur"]
    for a in ANCHORS:
        gate(len(anchor[a]) == 96, f"ANCHOR {a} 96")

    ctx = mp.get_context("fork")
    jobs = [(m["atlas_index"], tuple(m["roles"])) for m in MAN]
    policies = {}
    rf = open(ren_path, "w")
    done = 0
    with ctx.Pool(WORKERS) as pool:
        for idx, rows in pool.imap_unordered(render_policy, jobs):
            for row in rows:
                rf.write(json.dumps(row) + "\n")
            rf.flush()
            policies[idx] = rows
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"policies {done}/{len(jobs)} "
                      f"{time.monotonic() - t0:.0f}s", flush=True)
    rf.close()
    gate(len(policies) == len(jobs), "ALL POLICIES RETURNED")

    # per-policy rows
    pf = open(pol_path, "w")
    by_matrix = defaultdict(list)
    prow = {}
    for m in MAN:
        idx = m["atlas_index"]
        rows = policies[idx]
        rows.sort(key=lambda z: order.index((z["pair_id"], z["theta"])))
        gate([(z["pair_id"], z["theta"]) for z in rows] == order,
             "STATE ORDER")
        gates = Counter()
        for z in rows:
            for k, v in z["checks"].items():
                gates[f"{k}={v}"] += 1
        eligible = all(k.endswith("=True") for k in gates) and \
            len(rows) == len(P)
        mat_sha = hashlib.sha256(json.dumps(
            [z["cur"] for z in rows]).encode()).hexdigest()
        by_matrix[mat_sha].append(m["render_id"])
        anchors = {a: sum(1 for z in rows
                          if anchor[a][(z["pair_id"], z["theta"])]
                          == z["cur"]) for a in ANCHORS}
        states = [{"theta": z["theta"], "pair_id": z["pair_id"],
                   **{f: z[f] for f in FEATS}} for z in rows]
        ceil = {f: ceiling(states, f) for f in FEATS}
        by_theta = {th: {f: dict(Counter(str(z[f]) for z in rows
                                         if z["theta"] == th))
                         for f in FEATS} for th in ("SIN_LOW", "COS_LOW")}
        p = {**m, "eligible": eligible, "gates": dict(gates),
             "n_states": len(rows), "matrix_sha": mat_sha,
             "identical_to_raw": sum(z["identical_to_raw"] for z in rows),
             "anchor_matches": anchors,
             "anchor_of": [a for a, n in anchors.items() if n == len(P)],
             "ceilings": ceil, "surface_by_theta": by_theta,
             "max_prompt_tokens": max(z["prompt_tokens"] for z in rows)}
        prow[idx] = p
    for idx in sorted(prow):
        p = prow[idx]
        p["duplicate_class_size"] = len(by_matrix[p["matrix_sha"]])
        pf.write(json.dumps(p) + "\n")
    pf.close()

    n_elig = sum(1 for p in prow.values() if p["eligible"])
    distinct = len(by_matrix)
    dup_classes = {k: v for k, v in by_matrix.items() if len(v) > 1}
    anchor_hits = {a: [p["render_id"] for p in prow.values()
                       if a in p["anchor_of"]] for a in ANCHORS}
    anchor_roles = {a: [prow[i]["roles"] for i in prow
                        if a in prow[i]["anchor_of"]] for a in ANCHORS}
    # feature ceiling distribution over eligible policies
    ceil_dist = {f: dict(Counter(p["ceilings"][f]["top1_of_96"]
                                 for p in prow.values() if p["eligible"]))
                 for f in FEATS}
    const_count = {f: sum(1 for p in prow.values() if p["eligible"]
                          and p["ceilings"][f]["constant"]) for f in FEATS}

    # scoring cost from the booked counterfactual wall (no model access)
    cf = json.load(open(CF_RECEIPT))
    cf_rows = 24 * 96 * 4          # 4 ckpt x 2 masks x 3 views x 96 x 4
    per_row = cf["wall_s"] / cf_rows
    atlas_rows = distinct * 96 * 4 * 4
    cost = {"cf_wall_s": cf["wall_s"], "cf_rows": cf_rows,
            "per_row_s_upper": per_row,
            "atlas_distinct_matrices": distinct,
            "atlas_full_rows": atlas_rows,
            "atlas_full_wall_h_linear": atlas_rows * per_row / 3600,
            "atlas_full_wall_h_2x": 2 * atlas_rows * per_row / 3600,
            "continuation_T": T_CONT,
            "note": "cf wall includes four checkpoint loads and MASK0 "
                    "cells; treated as per-row cost (conservative)"}

    receipt = {"smoke": SMOKE, "assessment": ASSESS, "adopted": ADOPTED,
               "pins": PINS, "n_states": len(P), "n_policies": len(MAN),
               "workers": WORKERS, "prompt_law": PROMPT,
               "roles": list(ROLES), "manifest_sha": man_sha,
               "eligible": n_elig, "distinct_matrices": distinct,
               "duplicate_classes": dup_classes,
               "gate_totals": dict(sum((Counter(p["gates"])
                                        for p in prow.values()), Counter())),
               "anchors": {a: {"render_ids": anchor_hits[a],
                               "roles": anchor_roles[a]} for a in ANCHORS},
               "ceiling_top1_distribution": ceil_dist,
               "constant_feature_policies": const_count,
               "max_prompt_tokens": max(p["max_prompt_tokens"]
                                        for p in prow.values()),
               "cost_estimate": cost,
               "wall_s": round(time.monotonic() - t0, 1),
               "provenance": START,
               "completion_commit": completion_commit(),
               "renders_sha": fsha(str(ren_path)),
               "policies_sha": fsha(str(pol_path))}
    (OUTDIR / "prband2atlas_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    lead = ("EXHAUSTIVE GLOBAL ROLE-PERMUTATION ATLAS FEASIBLE"
            if n_elig == len(MAN) and all(len(v) == 1 for v in anchor_hits.values())
            else "ATLAS NEEDS REDESIGN")
    print(lead)
    print(json.dumps({k: receipt[k] for k in
                      ("eligible", "distinct_matrices", "gate_totals",
                       "anchors", "ceiling_top1_distribution",
                       "constant_feature_policies", "cost_estimate",
                       "wall_s")}, indent=1))


if __name__ == "__main__":
    main()

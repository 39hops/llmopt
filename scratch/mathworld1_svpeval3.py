"""MATH-CYBER-1 SVP-FACTOR-HASH-DESIGN-0 — materialize + qualify
the THIRD untouched evaluation band (seeds 9800-9819 x L4-L7, 80
episodes) under the exact first-band trajectory law, with the
qualified FACTOR/HASH code bars added. Zero model inference,
zero training, zero checkpoint access.

Law identity by IMPORT: run_episode / stable_legal_set /
derive_program from scratch/mathworld1_svpeval.py (the frozen
first-band module); FACTOR/HASH encode/decode from
scratch/mathworld1_svpcode.py (the qualified pair, constants
frozen at SVP-CODE-QUALIFY-0 before this band existed).

Primary task novelty: exact visible-cur exclusion against the
73,324-row training artifact AND both prior eval bands
(9600-9619, 9700-9719); edge overlaps reported, never merged.
Code bars: every enumerated candidate must FACTOR- and
HASH-roundtrip exactly and sit inside the qualified scoped
domain (ordinal/param -1..62); overflow = first-cause exclusion
counted at qualification and a registered NO-FIRE if it touches
the primary stratum.

Outputs under logs/mathworld1/svpeval3/ (refuse-if-exists):
episodes.jsonl, decisions.jsonl, svpeval3_receipt.json.

Exclusion precedence (first-failed cause, frozen): unsolved
episode -> legal_set_unstable -> program_* -> tok_roundtrip ->
program_collision -> context_overflow -> code_domain ->
code_roundtrip -> label_not_in_set -> training_parent_overlap
-> band1_parent_overlap -> band2_parent_overlap.

    .venv/bin/python scratch/mathworld1_svpeval3.py           (Mac)
"""
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        factor_symbols,
                                        hash_decode, hash_symbols,
                                        in_domain)
from scratch.mathworld1_svpeval import (derive_program,  # noqa: E402
                                        run_episode,
                                        stable_legal_set)

SEEDS = range(9800, 9820)
LEVELS = [4, 5, 6, 7]
CTX = 4096
X = sp.Symbol("x")
OUTDIR = Path("logs/mathworld1/svpeval3")
PINS = {
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
        "d8402351d468e8",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f3165241"
        "406a86253858c6",
    "logs/mathworld1/svpeval2/decisions.jsonl":
        "89efbe0ea447ee937c0c130d5419112921a2dd6c2159c6c211"
        "2cfd5e92f79315",
}
TOK = ActionGCTok()


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        if fsha(p) != h:
            raise SystemExit(f"PIN MISMATCH {p}")
    START = start_provenance(
        ["scratch/mathworld1_svpeval3.py",
         "scratch/mathworld1_svpeval.py",
         "scratch/mathworld1_svpcode.py", "scratch/mathworld0.py",
         "scratch/mathworld1_axfixture.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_srepr_export.py",
         "scratch/mathworld1_birth.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/mathgen/problems.py", "llmopt/lab/provenance.py"])

    # novelty sets — VISIBLE-TEXT law; frozen bytes, no sympify
    train_par, train_edge = set(), set()
    for l in open("data/matsub_paired.jsonl"):
        r = json.loads(l)
        train_par.add(r["cur"])
        train_edge.add((r["cur"], r["state_target"]))
    band_par = {"band1": set(), "band2": set()}
    band_edge = {"band1": set(), "band2": set()}
    for name, p in (("band1",
                     "logs/mathworld1/svpeval/decisions.jsonl"),
                    ("band2",
                     "logs/mathworld1/svpeval2/decisions.jsonl")):
        for l in open(p):
            r = json.loads(l)
            band_par[name].add(r["cur"])
            lc = r.get("labeled_child_sstr")
            if lc is not None:
                band_edge[name].add((r["cur"], lc))

    episodes_rows, decision_rows = [], []
    excl = Counter()
    st_lens, pg_lens, set_sizes = [], [], []
    collide = {}
    n_collision = tok_fail = ctx_fail = label_missing = 0
    code_dom_fail = f_rt_fail = h_rt_fail = 0
    amb = Counter()
    n_dec_total = n_primary = 0
    ov = Counter()
    for level in LEVELS:
        for seed in SEEDS:
            eid = f"L{level}-s{seed}"
            root = sp.Integral(make_integrate(level, seed)._expr, X)
            outcome, decisions = run_episode(root)
            episodes_rows.append({
                "episode_id": eid, "seed": seed, "level": level,
                "outcome": outcome,
                "n_decisions": len(decisions)})
            print(f"[svpeval3] {eid}: {outcome} "
                  f"({len(decisions)} decisions)", flush=True)
            if outcome != "solved":
                excl[f"episode_{outcome}"] += len(decisions)
                continue
            for di, (parent_st, _, (cname, cchild)) in \
                    enumerate(decisions):
                n_dec_total += 1
                parent = parent_st.expr
                cur = sp.sstr(parent)
                acts, stable = stable_legal_set(parent_st)
                if not stable:
                    excl["legal_set_unstable"] += 1
                    decision_rows.append({
                        "episode_id": eid, "decision_index": di,
                        "cur": cur, "primary_eligible": False,
                        "exclusion_class": "legal_set_unstable"})
                    continue
                accepted = defaultdict(set)
                for n, c in acts:
                    r = n.split("@", 1)[0] if "@" in n else n
                    accepted[r].add(c.key())
                cands = []
                fail = None
                for n, c in sorted(
                        acts, key=lambda nc: (nc[0], nc[1].key())):
                    rule = n.split("@", 1)[0] if "@" in n else n
                    prog, why = derive_program(
                        parent, rule, c.key(), accepted)
                    if prog is None:
                        fail = f"program_{why}"
                        amb[why] += 1
                        break
                    text = prog["program_text"]
                    if TOK.decode(TOK.encode(text)) != text:
                        tok_fail += 1
                        fail = "tok_roundtrip"
                        break
                    pk = (cur, text)
                    ck = sp.sstr(c.expr)
                    if pk in collide and collide[pk] != ck:
                        n_collision += 1
                        fail = "program_collision"
                        break
                    collide[pk] = ck
                    pre = len(TOK.encode(
                        f"Current: {cur}\nHints: none\nStep: "))
                    stl = pre + len(TOK.encode(ck + "\n")) + 1
                    pgl = pre + len(TOK.encode(text)) + 1
                    if stl > CTX or pgl > CTX:
                        ctx_fail += 1
                        fail = "context_overflow"
                        break
                    tup = (prog["rule"], prog["site_kind"],
                           prog["site_ordinal"],
                           prog["param_kind"],
                           prog["param_index"])
                    if not in_domain(*tup):
                        code_dom_fail += 1
                        fail = "code_domain"
                        break
                    fs = factor_symbols(*tup)
                    hs = hash_symbols(*tup)
                    if factor_decode(fs) != tup:
                        f_rt_fail += 1
                        fail = "code_roundtrip"
                        break
                    if hash_decode(hs) != tup:
                        h_rt_fail += 1
                        fail = "code_roundtrip"
                        break
                    cands.append({
                        "child_sstr": ck,
                        "child_srepr": sp.srepr(c.expr), **prog,
                        "factor_code": fs, "hash_code": hs,
                        "is_label": c.key() == cchild.key(),
                        "state_seq_tokens": stl,
                        "program_seq_tokens": pgl})
                if fail is None and sum(
                        c["is_label"] for c in cands) != 1:
                    label_missing += 1
                    fail = "label_not_in_set"
                if fail is None:
                    par_t = cur in train_par
                    par_1 = cur in band_par["band1"]
                    par_2 = cur in band_par["band2"]
                    if par_t:
                        ov["training_parent_overlap"] += 1
                    if (cur, sp.sstr(cchild.expr)) in train_edge:
                        ov["training_edge_overlap"] += 1
                    if par_1:
                        ov["band1_parent_overlap"] += 1
                    if (cur, sp.sstr(cchild.expr)) \
                            in band_edge["band1"]:
                        ov["band1_edge_overlap"] += 1
                    if par_2:
                        ov["band2_parent_overlap"] += 1
                    if (cur, sp.sstr(cchild.expr)) \
                            in band_edge["band2"]:
                        ov["band2_edge_overlap"] += 1
                    eligible = not (par_t or par_1 or par_2)
                    cls = ("training_parent_overlap" if par_t
                           else "band1_parent_overlap" if par_1
                           else "band2_parent_overlap" if par_2
                           else None)
                else:
                    excl[fail] += 1
                    eligible, cls = False, fail
                if fail is None and not eligible:
                    excl[cls] += 1
                row = {"episode_id": eid, "seed": seed,
                       "level": level, "decision_index": di,
                       "cur": cur, "cur_srepr": sp.srepr(parent),
                       "episode_outcome": outcome,
                       "labeled_child_sstr": sp.sstr(cchild.expr),
                       "labeled_child_srepr":
                           sp.srepr(cchild.expr),
                       "primary_eligible": bool(
                           fail is None and eligible),
                       "exclusion_class": cls,
                       "n_candidates": len(cands),
                       "candidates": cands}
                decision_rows.append(row)
                if row["primary_eligible"]:
                    n_primary += 1
                    set_sizes.append(len(cands))
                    for c in cands:
                        st_lens.append(c["state_seq_tokens"])
                        pg_lens.append(c["program_seq_tokens"])

    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "episodes.jsonl", "w") as f:
        for r in episodes_rows:
            f.write(json.dumps(r) + "\n")
    with open(OUTDIR / "decisions.jsonl", "w") as f:
        for r in decision_rows:
            f.write(json.dumps(r) + "\n")

    def dist(xs):
        xs = sorted(xs)
        q = lambda p: xs[min(len(xs) - 1,
                             int(p * (len(xs) - 1)))] if xs else None
        return {"p50": q(.5), "p90": q(.9), "max":
                xs[-1] if xs else None}

    solved_by = Counter(
        f"L{r['level']}" for r in episodes_rows
        if r["outcome"] == "solved")
    outcome_by = Counter(r["outcome"] for r in episodes_rows)
    prim = [r for r in decision_rows if r["primary_eligible"]]

    def lab_of(r):
        return [c for c in r["candidates"] if c["is_label"]][0]

    # coverage riders (33-tuple training support; text-only law
    # uses the exact tuple fields)
    train_tup = set()
    for l in open("data/matsub_paired.jsonl"):
        r = json.loads(l)
        train_tup.add((r["rule"], r["site_kind"],
                       r["site_ordinal"], r["param_kind"],
                       r["param_index"]))

    def ctup(c):
        return (c["rule"], c["site_kind"], c["site_ordinal"],
                c["param_kind"], c["param_index"])

    lab_cov = sum(1 for r in prim
                  if ctup(lab_of(r)) in train_tup)
    dec_all_cov = sum(1 for r in prim if all(
        ctup(c) in train_tup for c in r["candidates"]))
    cand_oov = sum(1 for r in prim for c in r["candidates"]
                   if ctup(c) not in train_tup)
    cand_all = sum(len(r["candidates"]) for r in prim)

    receipt = {
        "band": {"seeds": [9800, 9819], "levels": LEVELS,
                 "law": "first-band trajectory law verbatim by "
                        "import (greedy-hce, 12-decision, 60s "
                        "wall); FACTOR/HASH from the qualified "
                        "svpcode module"},
        "episodes": {"requested": 80,
                     "accounted": len(episodes_rows),
                     "outcomes": dict(outcome_by),
                     "solved_by_level": dict(solved_by)},
        "decisions": {"solved_episode_total": n_dec_total,
                      "primary_eligible": n_primary,
                      "exclusions_first_cause": dict(excl),
                      "ambiguity_counts": dict(amb),
                      "label_missing": label_missing},
        "legal_set_size": dist(set_sizes),
        "single_sibling": sum(1 for r in prim
                              if r["n_candidates"] == 1),
        "multi_sibling": sum(1 for r in prim
                             if r["n_candidates"] > 1),
        "per_rule": dict(Counter(
            c["rule"] for r in prim for c in r["candidates"])),
        "labeled_per_rule": dict(Counter(
            lab_of(r)["rule"] for r in prim)),
        "labeled_param_kind": dict(Counter(
            lab_of(r)["param_kind"] for r in prim)),
        "param_kind": dict(Counter(
            c["param_kind"] for r in prim
            for c in r["candidates"])),
        "has_u_choice_candidate": sum(
            1 for r in prim if any(
                c["param_kind"] == "u_choice"
                for c in r["candidates"])),
        "has_term_index_candidate": sum(
            1 for r in prim if any(
                c["param_kind"] == "term_index"
                for c in r["candidates"])),
        "token_dist": {"state_view": dist(st_lens),
                       "program_view": dist(pg_lens)},
        "overlaps": dict(ov),
        "coverage_riders": {
            "labeled_tuple_covered": lab_cov,
            "labeled_tuple_oov": n_primary - lab_cov,
            "decisions_all_candidates_covered": dec_all_cov,
            "decisions_with_oov_candidate":
                n_primary - dec_all_cov,
            "candidate_oov_count": cand_oov,
            "candidate_total": cand_all},
        "collisions": n_collision,
        "tok_roundtrip_fail": tok_fail,
        "ctx_overflow": ctx_fail,
        "code_domain_fail": code_dom_fail,
        "factor_roundtrip_fail": f_rt_fail,
        "hash_roundtrip_fail": h_rt_fail,
        "env": {"sympy": sp.__version__,
                "platform": platform.platform()},
        "bars": {
            "EPISODE_POPULATION": len(episodes_rows) == 80,
            "LABEL_IN_SET": label_missing == 0,
            "COMPLETE_LEGAL_SET": all(
                r.get("exclusion_class") != "legal_set_unstable"
                for r in decision_rows),
            "PROGRAM_REPLAY": not any(
                str(r.get("exclusion_class", "")).startswith(
                    "program_") for r in prim),
            "NO_PROGRAM_COLLISION": n_collision == 0,
            "TOK_ROUNDTRIP": tok_fail == 0,
            "CONTEXT_FIT": ctx_fail == 0,
            "TASK_NOVELTY": all(
                r["cur"] not in train_par
                and r["cur"] not in band_par["band1"]
                and r["cur"] not in band_par["band2"]
                for r in prim),
            "FACTOR_CODE_ROUNDTRIP": f_rt_fail == 0,
            "HASH_CODE_ROUNDTRIP": h_rt_fail == 0,
            "CODE_DOMAIN_FIT": code_dom_fail == 0,
        },
        "start": START, "completion_commit": completion_commit()}
    receipt["files"] = {
        n: fsha(OUTDIR / n)
        for n in ("episodes.jsonl", "decisions.jsonl")}
    (OUTDIR / "svpeval3_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "start"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

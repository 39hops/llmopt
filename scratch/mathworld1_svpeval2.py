"""MATH-CYBER-1 SVP-GENERALIZATION-BAND-0 — materialize + qualify
the SECOND, untouched task-sampling evaluation band (seeds
9700-9719 x L4-L7, 80 episodes) under the exact SVP-EVALBAND-0
trajectory law (mathworld0 greedy-hce, 12-decision budget, 60s
wall cap), before ANY trained model touches it. Zero model
inference, zero training.

Same semantic treatment-neutral candidate artifact as the first
band, plus: cur_srepr / child_srepr / labeled_child_srepr
companions captured from the LIVE objects before serialization
(provenance metadata only — downstream scoring stays governed by
the frozen TEXT law), and a TASK-NOVELTY law that excludes any
decision whose visible cur appears in the 73,324-row training
population OR in the original 9600-9619 evaluation band
(first-cause accounting; edge overlaps reported, never merged).

Outputs under logs/mathworld1/svpeval2/ (refuse-if-exists):
episodes.jsonl, decisions.jsonl, svpeval2_receipt.json.

Exclusion precedence (first-failed cause, frozen): unsolved
episode -> legal_set_unstable -> program_derivation/replay
failure -> tok_roundtrip -> program_collision -> context_overflow
-> label_not_in_set -> training_parent_overlap ->
oldeval_parent_overlap.

    .venv/bin/python scratch/mathworld1_svpeval2.py           (Mac)
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
from scratch.mathworld1_svpeval import (derive_program,  # noqa: E402
                                        run_episode,
                                        stable_legal_set)

SEEDS = range(9700, 9720)
LEVELS = [4, 5, 6, 7]
CTX = 4096
X = sp.Symbol("x")
OUTDIR = Path("logs/mathworld1/svpeval2")
PAIRED = Path("data/matsub_paired.jsonl")
PAIRED_SHA = ("a943ba7fc581db743b07192e5d951fadd"
              "dd2ba19bca3225b75d8402351d468e8")
OLD_EVAL = Path("logs/mathworld1/svpeval/decisions.jsonl")
OLD_EVAL_SHA = ("f63100a62f3091d544750d679483009a4"
                "73261c587f3165241406a86253858c6")
TOK = ActionGCTok()


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    if fsha(PAIRED) != PAIRED_SHA:
        raise SystemExit("PAIRED ARTIFACT SHA MISMATCH")
    if fsha(OLD_EVAL) != OLD_EVAL_SHA:
        raise SystemExit("OLD EVAL BAND SHA MISMATCH")
    START = start_provenance(
        ["scratch/mathworld1_svpeval2.py",
         "scratch/mathworld1_svpeval.py", "scratch/mathworld0.py",
         "scratch/mathworld1_axfixture.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_srepr_export.py",
         "scratch/mathworld1_birth.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/mathgen/problems.py", "llmopt/lab/provenance.py"])

    # freshness sets — VISIBLE-TEXT law, registered pre-run; the
    # old-eval strings are consumed as frozen bytes, never
    # sympified/re-enumerated
    train_par, train_edge = set(), set()
    for l in open(PAIRED):
        r = json.loads(l)
        train_par.add(r["cur"])
        train_edge.add((r["cur"], r["state_target"]))
    old_par, old_edge = set(), set()
    for l in open(OLD_EVAL):
        r = json.loads(l)
        old_par.add(r["cur"])
        lc = r.get("labeled_child_sstr")
        if lc is not None:
            old_edge.add((r["cur"], lc))
    corpus_states = set()
    for l in open("logs/mathworld1/states.jsonl"):
        corpus_states.add(json.loads(l)["state_before"])

    episodes_rows, decision_rows = [], []
    excl = Counter()
    st_lens, pg_lens, set_sizes = [], [], []
    lab_st_lens, lab_pg_lens = [], []
    collide = {}
    n_collision = tok_fail = ctx_fail = label_missing = 0
    amb = Counter()
    n_dec_total = n_primary = 0
    ov = Counter()
    srepr_seen = Counter()
    for level in LEVELS:
        for seed in SEEDS:
            eid = f"L{level}-s{seed}"
            root = sp.Integral(make_integrate(level, seed)._expr, X)
            outcome, decisions = run_episode(root)
            episodes_rows.append({
                "episode_id": eid, "seed": seed, "level": level,
                "outcome": outcome,
                "n_decisions": len(decisions)})
            print(f"[svpeval2] {eid}: {outcome} "
                  f"({len(decisions)} decisions)", flush=True)
            if outcome != "solved":
                excl[f"episode_{outcome}"] += len(decisions)
                continue
            for di, (parent_st, _, (cname, cchild)) in \
                    enumerate(decisions):
                n_dec_total += 1
                parent = parent_st.expr
                cur = sp.sstr(parent)
                cur_srepr = sp.srepr(parent)
                srepr_seen[cur_srepr] += 1
                acts, stable = stable_legal_set(parent_st)
                if not stable:
                    excl["legal_set_unstable"] += 1
                    decision_rows.append({
                        "episode_id": eid, "decision_index": di,
                        "cur": cur, "cur_srepr": cur_srepr,
                        "primary_eligible": False,
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
                    cands.append({
                        "child_sstr": ck,
                        "child_srepr": sp.srepr(c.expr), **prog,
                        "is_label": c.key() == cchild.key(),
                        "state_seq_tokens": stl,
                        "program_seq_tokens": pgl})
                if fail is None and sum(
                        c["is_label"] for c in cands) != 1:
                    label_missing += 1
                    fail = "label_not_in_set"
                if fail is None:
                    par_ov = cur in train_par
                    edge_ov = (cur, sp.sstr(cchild.expr)) \
                        in train_edge
                    old_par_ov = cur in old_par
                    old_edge_ov = (cur, sp.sstr(cchild.expr)) \
                        in old_edge
                    if par_ov:
                        ov["training_parent_overlap"] += 1
                    if edge_ov:
                        ov["training_edge_overlap"] += 1
                    if old_par_ov:
                        ov["oldeval_parent_overlap"] += 1
                    if old_edge_ov:
                        ov["oldeval_edge_overlap"] += 1
                    if cur in corpus_states:
                        ov["corpus725_state_overlap"] += 1
                    eligible = not par_ov and not old_par_ov
                    cls = ("training_parent_overlap" if par_ov
                           else "oldeval_parent_overlap"
                           if old_par_ov else None)
                else:
                    excl[fail] += 1
                    eligible, cls = False, fail
                if fail is None and not eligible:
                    excl[cls] += 1
                row = {"episode_id": eid, "seed": seed,
                       "level": level, "decision_index": di,
                       "cur": cur, "cur_srepr": cur_srepr,
                       "episode_outcome": outcome,
                       "labeled_child_sstr": sp.sstr(cchild.expr),
                       "labeled_child_srepr": sp.srepr(cchild.expr),
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
                        if c["is_label"]:
                            lab_st_lens.append(
                                c["state_seq_tokens"])
                            lab_pg_lens.append(
                                c["program_seq_tokens"])

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

    receipt = {
        "band": {"seeds": [9700, 9719], "levels": LEVELS,
                 "law": "SVP-EVALBAND-0 trajectory law verbatim "
                        "(greedy-hce, 12-decision budget, 60s "
                        "wall cap); producer imports run_episode/"
                        "stable_legal_set/derive_program from the "
                        "frozen first-band module"},
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
        "param_kind": dict(Counter(
            c["param_kind"] for r in prim
            for c in r["candidates"])),
        "labeled_param_kind": dict(Counter(
            lab_of(r)["param_kind"] for r in prim)),
        "has_u_choice_candidate": sum(
            1 for r in prim if any(
                c["param_kind"] == "u_choice"
                for c in r["candidates"])),
        "labeled_u_choice": sum(
            1 for r in prim
            if lab_of(r)["param_kind"] == "u_choice"),
        "has_term_index_candidate": sum(
            1 for r in prim if any(
                c["param_kind"] == "term_index"
                for c in r["candidates"])),
        "labeled_term_index": sum(
            1 for r in prim
            if lab_of(r)["param_kind"] == "term_index"),
        "token_dist": {"state_view": dist(st_lens),
                       "program_view": dist(pg_lens),
                       "labeled_state": dist(lab_st_lens),
                       "labeled_program": dist(lab_pg_lens)},
        "overlaps": dict(ov),
        "srepr_dupes_in_band": sum(
            v - 1 for v in srepr_seen.values() if v > 1),
        "collisions": n_collision,
        "tok_roundtrip_fail": tok_fail,
        "ctx_overflow": ctx_fail,
        "env": {"sympy": sp.__version__,
                "platform": platform.platform()},
        "bars": {
            "EPISODE_POPULATION": len(episodes_rows) == 80,
            "LABEL_IN_SET": label_missing == 0,
            "COMPLETE_LEGAL_SET": all(
                r["exclusion_class"] != "legal_set_unstable"
                for r in prim),
            "PROGRAM_REPLAY": not any(
                str(r.get("exclusion_class", "")).startswith(
                    "program_") for r in prim),
            "NO_PROGRAM_COLLISION": n_collision == 0,
            "TOK_ROUNDTRIP": tok_fail == 0,
            "CONTEXT_FIT": ctx_fail == 0,
            "TASK_NOVELTY": all(
                r["cur"] not in train_par
                and r["cur"] not in old_par for r in prim),
        },
        "start": START, "completion_commit": completion_commit()}
    receipt["files"] = {
        n: fsha(OUTDIR / n)
        for n in ("episodes.jsonl", "decisions.jsonl")}
    (OUTDIR / "svpeval2_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "start"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

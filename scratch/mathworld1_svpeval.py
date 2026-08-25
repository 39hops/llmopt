"""MATH-CYBER-1 SVP-EVALBAND-0 — materialize + qualify the frozen
primary evaluation band (seeds 9600-9619 x L4-L7, 80 episodes)
under the exact 101-corpus trajectory law (mathworld0 greedy-hce,
12-decision budget, 60s wall cap), emitting a SEMANTIC
treatment-neutral candidate artifact: every legal candidate
carries child_sstr AND its canonical FINAL ActionProgram +
program_text. Zero model, zero training.

Outputs under logs/mathworld1/svpeval/ (refuse-if-exists):
episodes.jsonl (all 80 identities), decisions.jsonl (solved-
episode decisions with full candidate sets), svpeval_receipt.json.

Exclusion precedence (first-failed cause, frozen):
unsolved episode -> legal_set_unstable -> program_derivation/
replay failure -> training_parent_overlap.

    .venv/bin/python scratch/mathworld1_svpeval.py            (Mac)
"""
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import State, hce, is_solved  # noqa: E402
from scratch.mathworld0 import (MAX_DECISIONS,  # noqa: E402
                                WALL_CAP_S, legal_actions)
from scratch.mathworld1_actionsem import (RULE_KIND,  # noqa: E402
                                          apply_at, iparts_children,
                                          sites_preorder)
from scratch.mathworld1_actionfinal import (  # noqa: E402
    unprod_term_children)
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_axfixture import serialize  # noqa: E402

SEEDS = range(9600, 9620)
LEVELS = [4, 5, 6, 7]
CTX = 4096
X = sp.Symbol("x")
OUTDIR = Path("logs/mathworld1/svpeval")
PAIRED = Path("data/matsub_paired.jsonl")
PAIRED_SHA = ("a943ba7fc581db743b07192e5d951fadd"
              "dd2ba19bca3225b75d8402351d468e8")
TOK = ActionGCTok()


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def run_episode(root):
    """The 101-corpus ACTIVE law verbatim: greedy-hce, 12-decision
    budget, 60s wall cap. Returns (outcome, decisions) where each
    decision = (parent State, acts list, chosen (name, child))."""
    state = State(root)
    t_ep = time.monotonic()
    decisions = []
    outcome = "budget_exhausted"
    for _ in range(MAX_DECISIONS):
        if is_solved(state):
            outcome = "solved"
            break
        if time.monotonic() - t_ep > WALL_CAP_S:
            outcome = "wall_cap"
            break
        derivation._RULE_CACHE.clear()
        acts, _ = legal_actions(state)
        if not acts:
            outcome = "dead_end"
            break
        name, child = min(
            acts, key=lambda nc: (hce(nc[1]), nc[0], nc[1].key()))
        decisions.append((state, acts, (name, child)))
        state = child
        if is_solved(state):
            outcome = "solved"
            break
    return outcome, decisions


def stable_legal_set(state):
    """COMPLETE-LEGAL-SET operational law: enumerate twice with a
    cleared rule cache; require identical (name, child-key)
    multisets."""
    derivation._RULE_CACHE.clear()
    a1, _ = legal_actions(state)
    derivation._RULE_CACHE.clear()
    a2, _ = legal_actions(state)
    k1 = Counter((n, c.key()) for n, c in a1)
    k2 = Counter((n, c.key()) for n, c in a2)
    return a1, k1 == k2


def derive_program(parent, rule, child_key, accepted):
    """AX-FIXTURE program derivation, keyed by exact child key."""
    kind = RULE_KIND.get(rule)
    if kind is None:
        site, node = -1, None
    else:
        hits = []
        for i, cand in enumerate(sites_preorder(parent, kind)):
            ck, _ = apply_at(parent, rule, cand)
            if child_key in set(ck) & accepted[rule]:
                hits.append((i, cand))
                break
        if not hits:
            return None, "unaddressable"
        site, node = hits[0]
    if rule == "i_parts":
        uc_map, _ = iparts_children(parent, node)
        m = [u for u, k in uc_map.items() if k == child_key]
        if len(m) != 1:
            return None, "u_ambiguous"
        pkind, pindex = "u_choice", m[0]
        dec = {uc_map[m[0]]}
    elif rule == "i_unprod":
        tmap, parity = unprod_term_children(parent, node)
        if not parity:
            return None, "parity_fail"
        m = [t for t, ks in tmap.items() if child_key in ks]
        if len(m) != 1:
            return None, "term_ambiguous"
        pkind, pindex = "term_index", m[0]
        dec = tmap[m[0]] & accepted[rule]
    else:
        if kind is None:
            dset = accepted[rule]
        else:
            ck, _ = apply_at(parent, rule, node)
            dset = set(ck) & accepted[rule]
        if len(dset) != 1:
            return None, "det_ambiguous"
        pkind, pindex = "none", -1
        dec = dset
    if dec != {child_key}:
        return None, "replay_mismatch"
    return {"rule": rule, "site_kind": kind if kind else "W",
            "site_ordinal": site, "param_kind": pkind,
            "param_index": pindex,
            "program_text": serialize(rule, kind, site,
                                      pkind, pindex)}, None


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    if fsha(PAIRED) != PAIRED_SHA:
        raise SystemExit("PAIRED ARTIFACT SHA MISMATCH")
    START = start_provenance(
        ["scratch/mathworld1_svpeval.py", "scratch/mathworld0.py",
         "scratch/mathworld1_axfixture.py",
         "scratch/mathworld1_actionfinal.py",
         "scratch/mathworld1_unprodsem.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_srepr_export.py",
         "scratch/mathworld1_birth.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py",
         "llmopt/mathgen/problems.py", "llmopt/lab/provenance.py"])

    # freshness sets (training artifact; law registered pre-run)
    train_par, train_edge = set(), set()
    for l in open(PAIRED):
        r = json.loads(l)
        train_par.add(r["cur"])
        train_edge.add((r["cur"], r["state_target"]))
    corpus_states = set()
    for l in open("logs/mathworld1/states.jsonl"):
        corpus_states.add(json.loads(l)["state_before"])

    episodes_rows, decision_rows = [], []
    excl = Counter()
    st_lens, pg_lens, set_sizes = [], [], []
    collide = {}
    n_collision = tok_fail = ctx_fail = label_missing = 0
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
            print(f"[svpeval] {eid}: {outcome} "
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
                seen_prog = {}
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
                        "child_sstr": ck, **prog,
                        "is_label": c.key() == cchild.key(),
                        "state_seq_tokens": stl,
                        "program_seq_tokens": pgl})
                    seen_prog[text] = ck
                if fail is None and sum(
                        c["is_label"] for c in cands) != 1:
                    label_missing += 1
                    fail = "label_not_in_set"
                if fail is None:
                    par_ov = cur in train_par
                    edge_ov = (cur, sp.sstr(cchild.expr)) \
                        in train_edge
                    if par_ov:
                        ov["training_parent_overlap"] += 1
                    if edge_ov:
                        ov["training_edge_overlap"] += 1
                    if cur in corpus_states:
                        ov["corpus725_state_overlap"] += 1
                    eligible = not par_ov
                    cls = ("training_parent_overlap"
                           if par_ov else None)
                else:
                    excl[fail] += 1
                    eligible, cls = False, fail
                if fail is None and not eligible:
                    excl["training_parent_overlap"] += 1
                row = {"episode_id": eid, "seed": seed,
                       "level": level, "decision_index": di,
                       "cur": cur,
                       "episode_outcome": outcome,
                       "labeled_child_sstr": sp.sstr(cchild.expr),
                       "primary_eligible": bool(
                           fail is None and eligible),
                       "exclusion_class": cls,
                       "n_candidates": len(cands),
                       "candidates": cands if fail is None
                       else cands}
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
    receipt = {
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
        "param_kind": dict(Counter(
            c["param_kind"] for r in prim
            for c in r["candidates"])),
        "token_dist": {"state_view": dist(st_lens),
                       "program_view": dist(pg_lens)},
        "overlaps": dict(ov),
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
            "FRESHNESS": all(
                r["cur"] not in train_par for r in prim),
        },
        "start": START, "completion_commit": completion_commit()}
    receipt["files"] = {
        n: fsha(OUTDIR / n)
        for n in ("episodes.jsonl", "decisions.jsonl")}
    (OUTDIR / "svpeval_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "start"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

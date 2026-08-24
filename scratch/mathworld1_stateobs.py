"""MATH-CYBER-1 STATE-OBSERVABILITY-DESK-0 — is the evaluation
structure that str(expr) omits ACTION-RELEVANT on the 26
SREPR-newly-bound decisions, or interchange-only?

Population (fixed): the 26 decisions listed noncomparable in
logs/mathworld1/actionprog_qual.json (the v1 state-string
round-trip failures, all why=state_hash). For each:
- EXACT parent = srepr_inverse(v2 states_srepr.jsonl state_before)
  (the engine's true State, unevaluated nodes preserved);
- VISIBLE-CANONICAL parent = sp.sympify(v1 states.jsonl
  state_before) (the canonical reading of the sstr text the model
  actually receives).
Alias-witness gate: the pair counts only if both parents print the
SAME model-visible string (sp.sstr(exact) == the frozen v1 string
== sp.sstr(visible)); a pair failing that gate is recorded
visible_gate_fail and never counted as a witness.

Compared per pair: State.key equality (expected to DIFFER by
construction — color, never a witness), is_solved, unsolved-node
census (Integral/Derivative node counts), n_legal, rule-label
multiset, legal child-key multiset, terminal-child key set
(is_solved children), and frozen chosen-edge availability
(chosen_child_hash present in the generated child-hash set).
derivation._RULE_CACHE.clear() before every successors() call.

REPRODUCTION LAW (timeboxes are load-sensitive): any legal-set /
terminal / chosen-edge difference must reproduce in TWO further
independent checks — a cold same-order repeat and an
order-reversed repeat (visible generated before exact) — before it
classifies as a witness; a non-reproducing difference books
UNSTABLE (excluded from the bar, counted).

BAR (frozen): STATE-ALIAS FIRES iff >=1 same-visible-string pair
reproducibly differs in legal child-key multiset OR
terminal/chosen-edge semantics. Prevalence is reported as
count/color, never as the bar.

Receipt: logs/mathworld1/stateobs_desk.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_stateobs.py           (Mac)
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import (State, is_solved,  # noqa: E402
                                      successors)
from scratch.mathworld1_srepr_export import srepr_inverse  # noqa: E402

OUT = Path("logs/mathworld1/stateobs_desk.json")


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def legal_profile(expr):
    """(label multiset, child-key multiset, terminal key set,
    child hash set) for one state, fresh rule cache."""
    derivation._RULE_CACHE.clear()
    acts = sorted(successors(State(expr)),
                  key=lambda nc: (nc[0], nc[1].key()))
    labels = Counter(n for n, _ in acts)
    ckeys = Counter(c.key() for _, c in acts)
    terminals = sorted(c.key() for _, c in acts if is_solved(c))
    hashes = {sha(c.key()) for _, c in acts}
    return labels, ckeys, terminals, hashes


def diff_fields(pe, pv, chosen_hash):
    """Names of behavioral fields that differ between the two
    profiles (label/ckeys/terminals/chosen availability)."""
    d = []
    if pe[0] != pv[0]:
        d.append("rule_label_multiset")
    if pe[1] != pv[1]:
        d.append("child_key_multiset")
    if pe[2] != pv[2]:
        d.append("terminal_child_set")
    if (chosen_hash in pe[3]) != (chosen_hash in pv[3]):
        d.append("chosen_edge_availability")
    return d


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/mathworld1_stateobs.py",
         "scratch/mathworld1_srepr_export.py",
         "llmopt/search/derivation.py"])
    qual = json.load(open("logs/mathworld1/actionprog_qual.json"))
    pop = [tuple(d["decision"])
           for d in qual["noncomparable_decisions"]]
    if len(pop) != 26 or any(
            d["why"] != "state_hash"
            for d in qual["noncomparable_decisions"]):
        raise SystemExit("POPULATION MISMATCH: expected the 26 "
                         "state_hash noncomparable decisions")
    v1 = {}
    for l in open("logs/mathworld1/states.jsonl"):
        r = json.loads(l)
        v1[(r["episode_id"], r["step_id"])] = r
    v2 = {}
    for l in open("logs/mathworld1/states_srepr.jsonl"):
        r = json.loads(l)
        v2[(r["episode_id"], r["step_id"])] = r

    rows = []
    n_witness = n_stable_same = n_gate_fail = n_unstable = 0
    for key in sorted(pop):
        r1, r2 = v1[key], v2[key]
        exact = srepr_inverse(r2["state_before"])
        if sha(sp.srepr(exact)) != r2["state_before_hash"]:
            raise SystemExit(f"EXACT PARENT HASH MISMATCH {key}")
        visible = sp.sympify(r1["state_before"])
        vis_str = r1["state_before"]
        gate = (sp.sstr(exact) == vis_str
                and sp.sstr(visible) == vis_str)
        row = {"decision": list(key),
               "visible_gate": gate,
               "key_equal": sp.srepr(exact) == sp.srepr(visible),
               "is_solved": [bool(is_solved(State(exact))),
                             bool(is_solved(State(visible)))],
               "unsolved_nodes": [
                   [len(exact.atoms(sp.Integral)),
                    len(exact.atoms(sp.Derivative))],
                   [len(visible.atoms(sp.Integral)),
                    len(visible.atoms(sp.Derivative))]]}
        if not gate:
            n_gate_fail += 1
            row["class"] = "visible_gate_fail"
            rows.append(row)
            continue
        chosen = r1["chosen_child_hash"]
        pe1, pv1_ = legal_profile(exact), legal_profile(visible)
        d1 = diff_fields(pe1, pv1_, chosen)
        row["n_legal"] = [sum(pe1[0].values()),
                          sum(pv1_[0].values())]
        row["diff_fields_run1"] = d1
        if not d1:
            n_stable_same += 1
            row["class"] = "behaviorally_identical"
            rows.append(row)
            continue
        # reproduction law: cold repeat + order-reversed repeat
        pe2, pv2_ = legal_profile(exact), legal_profile(visible)
        d2 = diff_fields(pe2, pv2_, chosen)
        pv3_, pe3 = legal_profile(visible), legal_profile(exact)
        d3 = diff_fields(pe3, pv3_, chosen)
        row["diff_fields_run2"] = d2
        row["diff_fields_run3_reversed"] = d3
        if d1 == d2 == d3:
            n_witness += 1
            row["class"] = "ALIAS_WITNESS"
            row["exact_labels_minus_visible"] = sorted(
                (pe1[0] - pv1_[0]).elements())
            row["visible_labels_minus_exact"] = sorted(
                (pv1_[0] - pe1[0]).elements())
        else:
            n_unstable += 1
            row["class"] = "UNSTABLE"
        rows.append(row)

    verdict = {
        "population": 26,
        "alias_witnesses": n_witness,
        "behaviorally_identical": n_stable_same,
        "visible_gate_fail": n_gate_fail,
        "unstable": n_unstable,
        "bar_STATE_ALIAS_fires": n_witness >= 1,
        "rows": rows,
        "start": START,
        "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(verdict, indent=1))
    print(f"[stateobs] witnesses={n_witness} identical="
          f"{n_stable_same} gate_fail={n_gate_fail} "
          f"unstable={n_unstable} "
          f"fires={n_witness >= 1}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

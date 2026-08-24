"""MATH-CYBER-1 ACTION-SITE-QUAL-0 — ActionProgram v3 decoder
qualification: the program addresses the engine's ACTUAL operator
application site, never a parsed display-label target.

P = (rule_id, operator_site, branch_index):
- operator_site = (kind, ordinal) with kind in {"D","I","L"} for
  Derivative/Integral/Limit rule families, None for whole-
  expression algebra moves. ordinal indexes the node among
  parent.atoms(Kind) sorted by (count_ops, srepr) — a total
  deterministic order. atoms() is a SET, so structurally equal
  occurrences are ONE site, exactly the engine's value-level
  xreplace semantics; when more than one distinct site reproduces
  the same frozen child, the encoder freezes the LOWEST ordinal
  (first-occurrence canonicalization) and the multiplicity is
  censused.
- Nested/multi-limit Integral law reproduced from the engine
  verbatim (derivation.successors): nested iff len(node.limits)>1;
  the rule applies to the synthesized innermost one-limit
  Integral(function, limits[0]); the rewrite is rewrapped
  Integral(rewrite, *limits[1:]); the child is
  parent.xreplace({node: new_node}). The synthetic inner target is
  RULE SEMANTICS, never part of the program.
- branch_index = the child's position among the ACCEPTED children
  of (rule, site), child-key-sorted. Accepted = the site-derived
  candidate keys intersected with the regenerated engine legal set
  for that rule (this reproduces verify_edge/dedup filtering
  without re-implementing it). Deterministic; not yet claimed as
  the final model-facing branch representation.

Decoder operands: exact parent (srepr_inverse of the versioned
corpus) + the frozen rule implementations + the program. The
frozen child string/hash is the comparison oracle only. Binding
gates per decision as in ACTIONPROG-QUAL-0/1: state hash + legal
child-hash multiset vs the frozen corpus, abort classes counted.

HOUSEKEEPING CENSUS (report-only; any mismatch BLOCKS
qualification): str(exact) == sp.sstr(exact) == frozen v1
state_before on all 102 corpus states — theta0 receives str()
while the interchange writes sstr().

Receipt: logs/mathworld1/actionsite_qual.json (refuse-if-exists).
No lengths, no model, no training, no fresh seeds.

    .venv/bin/python scratch/mathworld1_actionsite.py         (Mac)
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import (ALGEBRA_MOVES,  # noqa: E402
                                      State, successors)
from llmopt.search.rules import (CORE_RULES, INT_RULES,  # noqa: E402
                                 LIM_RULES)
from scratch.mathworld1_srepr_export import srepr_inverse  # noqa: E402

OUT = Path("logs/mathworld1/actionsite_qual.json")

KIND_CLASS = {"D": sp.Derivative, "I": sp.Integral, "L": sp.Limit}
RULE_KIND = {}
RULE_FN = {}
for n, r in CORE_RULES:
    RULE_KIND[n], RULE_FN[n] = "D", r
for n, r in INT_RULES:
    RULE_KIND[n], RULE_FN[n] = "I", r
for n, r in LIM_RULES:
    RULE_KIND[n], RULE_FN[n] = "L", r
for n, r in ALGEBRA_MOVES:
    RULE_KIND[n], RULE_FN[n] = None, r


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def sites_of(parent, kind):
    """Canonically ordered operator sites of one kind."""
    return sorted(parent.atoms(KIND_CLASS[kind]),
                  key=lambda n: (sp.count_ops(n), sp.srepr(n)))


def apply_at(parent, rule_name, node):
    """Engine-law candidate child keys for (rule, site): peel
    nested Integrals, apply the frozen rule timeboxed, rewrap,
    xreplace. Returns {child_key: nested_flag}."""
    kind = RULE_KIND[rule_name]
    if kind == "I" and len(node.limits) > 1:
        inner, nested = sp.Integral(node.function, node.limits[0]), True
    else:
        inner, nested = node, False
    rewrites = derivation._timeboxed(RULE_FN[rule_name], inner,
                                     default=[])
    out = {}
    for rw in rewrites:
        new_node = (sp.Integral(rw, *node.limits[1:])
                    if nested else rw)
        out[State(parent.xreplace({node: new_node})).key()] = nested
    return out


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/mathworld1_actionsite.py",
         "scratch/mathworld1_srepr_export.py",
         "llmopt/search/derivation.py",
         "llmopt/search/rules.py"])
    states = {}
    for l in open("logs/mathworld1/states_srepr.jsonl"):
        r = json.loads(l)
        states[(r["episode_id"], r["step_id"])] = r
    v1states = {}
    for l in open("logs/mathworld1/states.jsonl"):
        r = json.loads(l)
        v1states[(r["episode_id"], r["step_id"])] = r
    acts = defaultdict(list)
    for l in open("logs/mathworld1/actions_srepr.jsonl"):
        r = json.loads(l)
        acts[(r["episode_id"], r["step_id"])].append(r)

    # housekeeping census: str == sstr == frozen v1 string, 102 rows
    hk_fail = []
    for key, r2 in states.items():
        e = srepr_inverse(r2["state_before"])
        v1s = v1states[key]["state_before"]
        if not (str(e) == sp.sstr(e) == v1s):
            hk_fail.append(list(key))
    if hk_fail:
        OUT.write_text(json.dumps(
            {"housekeeping_str_sstr_fail": hk_fail,
             "qualification": "BLOCKED",
             "start": START,
             "completion_commit": completion_commit()}, indent=1))
        raise SystemExit(f"HOUSEKEEPING MISMATCH x{len(hk_fail)}")

    n_ok = n_wrong = n_coll = n_range = n_nosite = 0
    noncomparable = []
    fails = []
    kind_hist = Counter()
    ord_hist = Counter()
    nested_actions = 0
    multi_site_actions = 0
    branch_hist = Counter()
    actions_by_rule = Counter()
    branch_gt0_rules = Counter()
    for key in sorted(acts):
        parent = srepr_inverse(states[key]["state_before"])
        if sha(sp.srepr(parent)) != states[key]["state_before_hash"]:
            noncomparable.append(
                {"decision": list(key), "why": "state_hash"})
            continue
        derivation._RULE_CACHE.clear()
        gen = sorted(successors(State(parent)),
                     key=lambda nc: (nc[0], nc[1].key()))
        if (Counter(sha(c.key()) for _, c in gen)
                != Counter(a["child_hash"] for a in acts[key])):
            noncomparable.append(
                {"decision": list(key), "why": "legal_set"})
            continue
        # accepted child keys per rule (engine-filtered)
        accepted = defaultdict(set)
        for name, c in gen:
            rule = name.split("@", 1)[0] if "@" in name else name
            accepted[rule].add(c.key())
        seen_programs = {}
        for a in acts[key]:
            rule = a["rule"]
            kind = RULE_KIND.get(rule)
            if kind is None and rule not in RULE_FN:
                n_nosite += 1
                fails.append({"decision": list(key), "rule": rule,
                              "why": "unknown_rule"})
                continue
            if kind is None:
                cand_sites = [None]
            else:
                cand_sites = sites_of(parent, kind)
            # ENCODE: locate site(s) reproducing the frozen child
            hits = []
            for i, node in enumerate(cand_sites):
                if node is None:
                    # algebra: accepted keys of the bare rule
                    keys = accepted[rule]
                    if any(sha(k) == a["child_hash"] for k in keys):
                        hits.append((i, None, False))
                    continue
                cand = apply_at(parent, rule, node)
                keys = set(cand) & accepted[rule]
                if any(sha(k) == a["child_hash"] for k in keys):
                    nested = any(cand[k] for k in cand)
                    hits.append((i, node, nested))
            if not hits:
                n_nosite += 1
                fails.append({"decision": list(key), "rule": rule,
                              "why": "unaddressable_site"})
                continue
            if len(hits) > 1:
                multi_site_actions += 1
            ordinal, node, nested = hits[0]
            site = None if kind is None else [kind, ordinal]
            if nested:
                nested_actions += 1
            # branch index among accepted (rule, site) children
            if kind is None:
                branch_keys = sorted(accepted[rule])
            else:
                branch_keys = sorted(
                    set(apply_at(parent, rule, node))
                    & accepted[rule])
            match = [k for k in branch_keys
                     if sha(k) == a["child_hash"]]
            if len(match) != 1:
                n_wrong += 1
                fails.append({"decision": list(key), "rule": rule,
                              "why": "frozen_child_absent"})
                continue
            branch = branch_keys.index(match[0])
            program = (rule, None if site is None else tuple(site),
                       branch)
            if program in seen_programs:
                n_coll += 1
                fails.append({"decision": list(key), "rule": rule,
                              "why": "program_collision"})
                continue
            seen_programs[program] = a["child_hash"]
            # DECODE from (parent, program) only
            if kind is None:
                dkeys = sorted(accepted[rule])
            else:
                dsites = sites_of(parent, kind)
                if ordinal >= len(dsites):
                    n_range += 1
                    fails.append({"decision": list(key),
                                  "rule": rule,
                                  "why": "branch_out_of_range"})
                    continue
                dkeys = sorted(
                    set(apply_at(parent, rule, dsites[ordinal]))
                    & accepted[rule])
            if branch >= len(dkeys):
                n_range += 1
                fails.append({"decision": list(key), "rule": rule,
                              "why": "branch_out_of_range"})
                continue
            if sha(dkeys[branch]) == a["child_hash"]:
                n_ok += 1
                if site is not None:
                    kind_hist[site[0]] += 1
                    ord_hist[site[1]] += 1
                else:
                    kind_hist["ALG"] += 1
                branch_hist[branch] += 1
                actions_by_rule[rule] += 1
                if branch > 0:
                    branch_gt0_rules[rule] += 1
            else:
                n_wrong += 1
                fails.append({"decision": list(key), "rule": rule,
                              "why": "wrong_child"})
    verdict = {
        "housekeeping_str_sstr": "102/102 str==sstr==frozen v1",
        "qualified": n_ok,
        "wrong_child": n_wrong, "collision": n_coll,
        "out_of_range": n_range, "unaddressable_site": n_nosite,
        "noncomparable_decisions": noncomparable,
        "failures": fails[:50],
        "site_kind_hist": dict(kind_hist),
        "site_ordinal_hist": dict(ord_hist),
        "nested_multilimit_actions": nested_actions,
        "multi_site_actions": multi_site_actions,
        "branch_index_hist": dict(branch_hist),
        "qualified_actions_by_rule": dict(actions_by_rule),
        "branch_gt0_by_rule": dict(branch_gt0_rules),
        "start": START, "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(verdict, indent=1))
    print(f"[actionsite] ok={n_ok} wrong={n_wrong} coll={n_coll} "
          f"range={n_range} nosite={n_nosite} "
          f"noncomp={len(noncomparable)} nested={nested_actions} "
          f"multisite={multi_site_actions}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

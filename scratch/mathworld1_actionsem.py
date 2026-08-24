"""MATH-CYBER-1 ACTION-SEMANTICS-QUAL-0 — ActionProgram v4:
model-facing semantic program. Two deltas over the qualified v3
(scratch/mathworld1_actionsite.py, ACTION-SITE-QUAL-0):

1. i_parts branch -> semantic u_choice. The frozen i_parts law
   (llmopt/search/rules.py i_parts) emits one child per factor
   index i of the (inner) integrand Mul with diff(f.args[i], x)
   != 0, u = f.args[i]. u_choice = the ordinal of that factor
   among the ELIGIBLE factors (du != 0) in the Mul's stored
   argument order — a stable mathematical meaning ("which factor
   is u"). The decoder applies EXACTLY that choice: it rebuilds
   u*Integral(dv, x) - Integral(Integral(dv, x)*du, x) from the
   chosen factor, rewraps nested limits per the engine law, and
   xreplaces — it never enumerates children and picks by frozen
   hash. AMBIGUOUS-U law: if two distinct eligible u choices
   produce the same accepted child key, the encoder freezes the
   LOWEST eligible ordinal and the case is censused (and counted
   in the ambiguous_u failure class if the frozen action cannot be
   uniquely attributed).

2. Site coordinate: v3's (count_ops, srepr) sort is replaced by
   FIRST-PREORDER-OCCURRENCE order among the structurally unique
   operator nodes (parent.atoms(Kind) ranked by the position of
   each node's first occurrence in a deterministic preorder walk
   of the parent) — the order a reader of the visible string
   encounters them. Value-level xreplace semantics unchanged
   (atoms() set; structural duplicates are one site). The receipt
   censuses how many qualified actions change ordinal identity vs
   the v3 coordinate.

All other rule families keep the v3 deterministic accepted-set
branch index; i_unprod is explicitly FENCED as a candidate index
this rung (13/725 actions, 1 observed branch>0) rather than given
premature semantics.

Binding gates per decision as in v3 (state hash + legal child-hash
multiset). Decoder operands: exact parent + frozen rule
implementations + program; frozen child hash = comparison oracle
only. Receipt: logs/mathworld1/actionsem_qual.json
(refuse-if-exists). No lengths, no model, no training, no fresh
seeds, no MAGIC, no search.

    .venv/bin/python scratch/mathworld1_actionsem.py          (Mac)
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

OUT = Path("logs/mathworld1/actionsem_qual.json")

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


def preorder(expr):
    yield expr
    for a in expr.args:
        yield from preorder(a)


def sites_preorder(parent, kind):
    """Structurally unique operator sites ranked by first preorder
    occurrence (the v4 site law)."""
    want = parent.atoms(KIND_CLASS[kind])
    seen = []
    for n in preorder(parent):
        if n in want and not any(n == s for s in seen):
            seen.append(n)
    if len(seen) != len(want):
        raise SystemExit("SITE ENUMERATION MISMATCH")
    return seen


def sites_v3(parent, kind):
    return sorted(parent.atoms(KIND_CLASS[kind]),
                  key=lambda n: (sp.count_ops(n), sp.srepr(n)))


def peel(node):
    """Engine nested-Integral law: (inner target, nested flag)."""
    if isinstance(node, sp.Integral) and len(node.limits) > 1:
        return sp.Integral(node.function, node.limits[0]), True
    return node, False


def rewrap(node, rewrite, nested):
    return (sp.Integral(rewrite, *node.limits[1:])
            if nested else rewrite)


def apply_at(parent, rule_name, node):
    """Engine-law candidate child keys for (rule, site)."""
    inner, nested = peel(node) if RULE_KIND[rule_name] == "I" \
        else (node, False)
    rewrites = derivation._timeboxed(RULE_FN[rule_name], inner,
                                     default=[])
    return {State(parent.xreplace(
        {node: rewrap(node, rw, nested)})).key(): nested
        for rw in rewrites}, nested


def iparts_children(parent, node):
    """(eligible ordinal -> child key, nested flag) applying each
    u choice DIRECTLY per the frozen i_parts law."""
    inner, nested = peel(node)
    f, x = inner.function, inner.limits[0][0]
    out = {}
    if not isinstance(f, sp.Mul):
        return out, nested
    elig = 0
    for i, u_part in enumerate(f.args):
        du = sp.diff(u_part, x)
        if du == 0:
            continue
        dv = sp.Mul(*(a for j, a in enumerate(f.args) if j != i))
        v = sp.Integral(dv, x)
        rw = u_part * v - sp.Integral(v * du, x)
        out[elig] = State(parent.xreplace(
            {node: rewrap(node, rw, nested)})).key()
        elig += 1
    return out, nested


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_srepr_export.py",
         "llmopt/search/derivation.py",
         "llmopt/search/rules.py"])
    states = {}
    for l in open("logs/mathworld1/states_srepr.jsonl"):
        r = json.loads(l)
        states[(r["episode_id"], r["step_id"])] = r
    acts = defaultdict(list)
    for l in open("logs/mathworld1/actions_srepr.jsonl"):
        r = json.loads(l)
        acts[(r["episode_id"], r["step_id"])].append(r)

    n_ok = n_wrong = n_coll = n_range = n_nosite = n_ambig = 0
    noncomparable = []
    fails = []
    u_hist = Counter()
    elig_hist = Counter()
    ambiguous_u_cases = []
    site_identity_changes = 0
    nested_actions = 0
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
        accepted = defaultdict(set)
        for name, c in gen:
            rule = name.split("@", 1)[0] if "@" in name else name
            accepted[rule].add(c.key())
        seen_programs = {}
        for a in acts[key]:
            rule = a["rule"]
            kind = RULE_KIND.get(rule)
            # ---- locate the site under the v4 preorder law
            if kind is None:
                site, node = None, None
            else:
                s4 = sites_preorder(parent, kind)
                hits = []
                for i, cand in enumerate(s4):
                    ck, nested = apply_at(parent, rule, cand)
                    keys = set(ck) & accepted[rule]
                    if any(sha(k) == a["child_hash"] for k in keys):
                        hits.append((i, cand, nested))
                if not hits:
                    n_nosite += 1
                    fails.append({"decision": list(key),
                                  "rule": rule,
                                  "why": "unaddressable_site"})
                    continue
                ordinal, node, nested = hits[0]
                if nested:
                    nested_actions += 1
                site = [kind, ordinal]
                # census: does the v3 coordinate differ?
                s3 = sites_v3(parent, kind)
                if s3.index(node) != ordinal:
                    site_identity_changes += 1
            # ---- parameter: semantic u_choice for i_parts
            if rule == "i_parts":
                uc_map, nested = iparts_children(parent, node)
                elig_hist[len(uc_map)] += 1
                matches = [uc for uc, k in uc_map.items()
                           if sha(k) == a["child_hash"]]
                bykey = defaultdict(list)
                for uc, k in uc_map.items():
                    bykey[k].append(uc)
                for k, ucs in bykey.items():
                    if len(ucs) > 1:
                        ambiguous_u_cases.append(
                            {"decision": list(key),
                             "u_choices": ucs})
                if not matches:
                    n_wrong += 1
                    fails.append({"decision": list(key),
                                  "rule": rule,
                                  "why": "no_u_reproduces_child"})
                    continue
                if len(matches) > 1:
                    n_ambig += 1
                    fails.append({"decision": list(key),
                                  "rule": rule,
                                  "why": "ambiguous_u"})
                    continue
                param = ("u", matches[0])
                # DECODE: re-resolve the site from the program,
                # then apply exactly that u choice
                dnode = sites_preorder(parent, kind)[site[1]]
                dmap, _ = iparts_children(parent, dnode)
                if matches[0] not in dmap:
                    n_range += 1
                    fails.append({"decision": list(key),
                                  "rule": rule,
                                  "why": "branch_out_of_range"})
                    continue
                decoded = dmap[matches[0]]
                u_hist[matches[0]] += 1
            else:
                # deterministic accepted-set branch (v3 law)
                if kind is None:
                    branch_keys = sorted(accepted[rule])
                else:
                    ck, _ = apply_at(parent, rule, node)
                    branch_keys = sorted(set(ck) & accepted[rule])
                match = [k for k in branch_keys
                         if sha(k) == a["child_hash"]]
                if len(match) != 1:
                    n_wrong += 1
                    fails.append({"decision": list(key),
                                  "rule": rule,
                                  "why": "frozen_child_absent"})
                    continue
                branch = branch_keys.index(match[0])
                param = ("b", branch)
                if kind is None:
                    dkeys = sorted(accepted[rule])
                else:
                    dsites = sites_preorder(parent, kind)
                    ck, _ = apply_at(parent, rule,
                                     dsites[site[1]])
                    dkeys = sorted(set(ck) & accepted[rule])
                if branch >= len(dkeys):
                    n_range += 1
                    fails.append({"decision": list(key),
                                  "rule": rule,
                                  "why": "branch_out_of_range"})
                    continue
                decoded = dkeys[branch]
            program = (rule,
                       None if site is None else tuple(site),
                       param)
            if program in seen_programs:
                n_coll += 1
                fails.append({"decision": list(key), "rule": rule,
                              "why": "program_collision"})
                continue
            seen_programs[program] = a["child_hash"]
            if sha(decoded) == a["child_hash"]:
                n_ok += 1
            else:
                n_wrong += 1
                fails.append({"decision": list(key), "rule": rule,
                              "why": "wrong_child"})
    verdict = {
        "qualified": n_ok,
        "wrong_child": n_wrong, "collision": n_coll,
        "out_of_range": n_range, "unaddressable_site": n_nosite,
        "ambiguous_u": n_ambig,
        "noncomparable_decisions": noncomparable,
        "failures": fails[:50],
        "u_choice_hist": dict(u_hist),
        "eligible_factors_per_iparts_action": dict(elig_hist),
        "ambiguous_u_cases": ambiguous_u_cases[:20],
        "n_ambiguous_u_cases": len(ambiguous_u_cases),
        "site_identity_changes_v4_vs_v3": site_identity_changes,
        "nested_multilimit_actions": nested_actions,
        "start": START, "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(verdict, indent=1))
    print(f"[actionsem] ok={n_ok} wrong={n_wrong} coll={n_coll} "
          f"range={n_range} nosite={n_nosite} ambig={n_ambig} "
          f"noncomp={len(noncomparable)} "
          f"sitechg={site_identity_changes} "
          f"u_hist={dict(u_hist)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

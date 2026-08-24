"""MATH-CYBER-1 ACTIONPROG-QUAL-1 — the qualified semantic
ActionProgram decoder re-run against the VERSIONED SREPR corpus
(states_srepr.jsonl / actions_srepr.jsonl, SREPR-EXPORT-0).
Decoder logic byte-identical to scratch/mathworld1_actionprog.py
(ACTIONPROG-QUAL-0); only the three corpus/receipt paths and
this header differ. state_before is now srepr, so
sp.sympify(state_before) is expected to reproduce State.key()
on every decision; no lengths are measured here.

Canonical ActionProgram P = (rule_id, target_address,
branch_index):
- target_address: canonical AST coordinate of the rule target
  in the PARENT — (occurrence_index) of the target expression
  among structurally-equal nodes in a deterministic preorder
  traversal of the parent tree; None for bare rules. The
  encoder always emits the FIRST occurrence (value-level
  rewriting; a mismatch surfaces as a qualification failure).
- branch_index: the action's position among
  same-(rule, resolved-target) siblings sorted by child
  State.key() — a canonical SEMANTIC sort, not generation
  order.

decode(parent_state, P): regenerate successors(parent) under
the frozen rule implementation (idle machine, _RULE_CACHE
cleared), resolve target_address to a subexpression of the
parent, filter generated actions to rule_id + that exact
target expression, sort survivors by child.key(), take
branch_index. The recorded child string/hash is used ONLY as
the comparison oracle (sha(decoded.key()) == frozen
child_hash), never as a decode operand.

Binding gates per decision (standing law): sha(rebuilt parent
key) == state_before_hash, and the regenerated legal set's
child_hash multiset == the frozen action rows' — any mismatch
books that decision WORLD-NONCOMPARABLE (excluded from the
qualification denominator, counted and reported).

Report: qualified/failed per action, failures grouped by rule
family, collision check (distinct frozen children from one
parent must decode from distinct programs), target-occurrence
anatomy (how many targets are non-first occurrences — 0
expected), branch-index and address distributions (schema
anatomy only; NO length thresholds here).

    .venv/bin/python scratch/mathworld1_actionprog2.py   (Mac)
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
from llmopt.search.derivation import State, successors  # noqa: E402

OUT = Path("logs/mathworld1/actionprog_qual_srepr.json")


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def preorder(expr):
    yield expr
    for a in expr.args:
        yield from preorder(a)


def occurrences(parent, target):
    return [i for i, n in enumerate(preorder(parent))
            if n == target]


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/mathworld1_actionprog2.py",
         "llmopt/search/derivation.py"])
    states = {}
    for l in open("logs/mathworld1/states_srepr.jsonl"):
        r = json.loads(l)
        states[(r["episode_id"], r["step_id"])] = r
    acts = defaultdict(list)
    for l in open("logs/mathworld1/actions_srepr.jsonl"):
        r = json.loads(l)
        acts[(r["episode_id"], r["step_id"])].append(r)

    n_ok = n_fail = 0
    noncomparable = []
    fail_by_rule = Counter()
    fails = []
    addr_occidx = Counter()
    branch_hist = Counter()
    nonfirst_targets = 0
    for key in sorted(acts):
        st_row = states[key]
        parent_expr = sp.sympify(st_row["state_before"])
        parent = State(parent_expr)
        if sha(parent.key()) != st_row["state_before_hash"]:
            noncomparable.append(
                {"decision": list(key), "why": "state_hash"})
            continue
        derivation._RULE_CACHE.clear()
        gen = sorted(successors(parent),
                     key=lambda nc: (nc[0], nc[1].key()))
        gen_hashes = Counter(sha(c.key()) for _, c in gen)
        froz_hashes = Counter(a["child_hash"]
                              for a in acts[key])
        if gen_hashes != froz_hashes:
            noncomparable.append(
                {"decision": list(key), "why": "legal_set",
                 "gen": len(gen),
                 "frozen": len(acts[key])})
            continue
        # generated actions with PARSED labels (the frozen rule
        # implementation's own output; decoder-legal operands)
        gparsed = []
        for name, c in gen:
            if "@" in name:
                gr, gt = name.split("@", 1)
                gparsed.append((gr, gt, c))
            else:
                gparsed.append((name, None, c))
        # ENCODE each frozen action, then DECODE and compare
        seen_programs = set()
        for a in acts[key]:
            rule = a["rule"]
            if a["rule_target"] is not None:
                tgt = sp.sympify(a["rule_target"])
                occ = occurrences(parent_expr, tgt)
                if not occ:
                    n_fail += 1
                    fail_by_rule[rule] += 1
                    fails.append({"decision": list(key),
                                  "rule": rule,
                                  "why": "target_not_in_parent"})
                    continue
                addr = occ[0]
                addr_occidx[len(occ)] += 1
            else:
                addr = None
            # branch index: this action's position among
            # generated same-(rule, target-string) siblings,
            # child.key()-sorted (encoder may consult the
            # frozen child_hash to locate itself)
            sib_keys = sorted(
                c.key() for gr, gt, c in gparsed
                if gr == rule and gt == a["rule_target"])
            match = [c for gr, gt, c in gparsed
                     if sha(c.key()) == a["child_hash"]]
            if len(match) != 1:
                n_fail += 1
                fail_by_rule[rule] += 1
                fails.append({"decision": list(key),
                              "rule": rule,
                              "why": "frozen_child_absent"})
                continue
            branch = sib_keys.index(match[0].key())
            branch_hist[branch] += 1
            program = (rule, addr, branch)
            if program in seen_programs:
                n_fail += 1
                fail_by_rule[rule] += 1
                fails.append({"decision": list(key),
                              "rule": rule,
                              "why": "program_collision"})
                continue
            seen_programs.add(program)
            # DECODE from (parent, program) + generated labels
            # ONLY: resolve the address in the parent tree,
            # match generated targets SEMANTICALLY (sympify of
            # the label's own target string), sort survivor
            # child keys, take branch. No frozen-row operand.
            if addr is not None:
                nodes = list(preorder(parent_expr))
                resolved_tgt = nodes[addr]
                dkeys = sorted(
                    c.key() for gr, gt, c in gparsed
                    if gr == rule and gt is not None
                    and sp.sympify(gt) == resolved_tgt)
            else:
                dkeys = sorted(
                    c.key() for gr, gt, c in gparsed
                    if gr == rule and gt is None)
            if branch >= len(dkeys):
                n_fail += 1
                fail_by_rule[rule] += 1
                fails.append({"decision": list(key),
                              "rule": rule,
                              "why": "branch_out_of_range"})
                continue
            decoded_key = dkeys[branch]
            if sha(decoded_key) == a["child_hash"]:
                n_ok += 1
            else:
                n_fail += 1
                fail_by_rule[rule] += 1
                fails.append({"decision": list(key),
                              "rule": rule,
                              "why": "wrong_child"})
    verdict = {
        "qualified": n_ok, "failed": n_fail,
        "noncomparable_decisions": noncomparable,
        "fail_by_rule": dict(fail_by_rule),
        "failures": fails[:50],
        "target_occurrence_multiplicity": dict(addr_occidx),
        "branch_index_hist": dict(branch_hist),
        "start": START,
        "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(verdict, indent=1))
    print(f"[actionprog] ok={n_ok} fail={n_fail} "
          f"noncomp={len(noncomparable)} "
          f"multi_occ={dict(addr_occidx)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

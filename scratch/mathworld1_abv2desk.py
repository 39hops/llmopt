"""MATH-CYBER-1 ACTION-BASIS-v2-DESK-0 — length desk for the
qualified v4 semantic ActionProgram serialization. Counting only:
zero model, zero training, zero fresh seeds, zero search.

FROZEN SERIALIZATION (model-facing, GCTok as-is, no new atoms):
- algebra move:                      "{rule}\n"
- sited, branch-deterministic:       "{rule} {kind}{ordinal}\n"
- i_parts (always states its u):     "i_parts {kind}{ordinal} u{k}\n"
- fenced i_unprod, branch > 0 only:  "i_unprod {kind}{ordinal} b{k}\n"
  (branch 0 sited actions carry no parameter field)

Programs are recomputed with the QUALIFIED v4 encode imported from
scratch/mathworld1_actionsem.py (ACTION-SEMANTICS-QUAL-0) — same
site law (first-preorder), same u_choice law, same accepted-set
branch for non-i_parts; binding gates re-asserted per decision.
Lengths in GCTok tokens (scratch/mathworld1_birth.GCTok).

Parent prefix = the v1 desk's formula, for comparability:
  "Current: {state_before}\nHints: none\nStep: "
Child serialization baseline = the frozen v1 sstr child + "\n"
(actions.jsonl), joined by (episode, step, child_hash).

OPCODE-NORMALIZED counterfactual (descriptive only, frozen before
raw lengths are read): normalized_len = raw_len -
len(GCTok(rule_name)) + 1 — the rule id replaced by exactly one
hypothetical token; site/parameter tokens unchanged.

Receipt: logs/mathworld1/abv2_desk.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_abv2desk.py           (Mac)
"""
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

import llmopt.search.derivation as derivation  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.search.derivation import State, successors  # noqa: E402
from scratch.mathworld1_actionsem import (RULE_KIND,  # noqa: E402
                                          apply_at, iparts_children,
                                          sites_preorder)
from scratch.mathworld1_birth import GCTok  # noqa: E402
from scratch.mathworld1_srepr_export import srepr_inverse  # noqa: E402

OUT = Path("logs/mathworld1/abv2_desk.json")
TOK = GCTok()


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def tlen(s: str) -> int:
    return len(TOK.encode(s))


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def summ(xs):
    return {"med": statistics.median(xs), "p90": pct(xs, 0.9),
            "max": max(xs)}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(
        ["scratch/mathworld1_abv2desk.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_srepr_export.py",
         "llmopt/search/derivation.py", "llmopt/search/rules.py"])
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
    v1child = {}
    for l in open("logs/mathworld1/actions.jsonl"):
        r = json.loads(l)
        v1child[(r["episode_id"], r["step_id"],
                 r["child_hash"])] = r["child"]

    prog_lens, child_lens, norm_lens = [], [], []
    per_rule = defaultdict(list)
    unprod_lens = []
    prefix_lens = {}
    spans_raw, spans_norm = [], []
    fit512 = fit4096 = 0
    parent_only_512 = []
    action_induced_512 = []
    n_actions = 0
    field_tokens = {"rule": [], "site": [], "param": []}
    for key in sorted(acts):
        parent = srepr_inverse(states[key]["state_before"])
        if sha(sp.srepr(parent)) != states[key]["state_before_hash"]:
            raise SystemExit(f"BINDING state_hash {key}")
        derivation._RULE_CACHE.clear()
        gen = sorted(successors(State(parent)),
                     key=lambda nc: (nc[0], nc[1].key()))
        if (Counter(sha(c.key()) for _, c in gen)
                != Counter(a["child_hash"] for a in acts[key])):
            raise SystemExit(f"BINDING legal_set {key}")
        accepted = defaultdict(set)
        for name, c in gen:
            rule = name.split("@", 1)[0] if "@" in name else name
            accepted[rule].add(c.key())
        prefix = tlen(f"Current: {v1states[key]['state_before']}"
                      f"\nHints: none\nStep: ")
        prefix_lens[key] = prefix
        drow_raw, drow_norm = [], []
        for a in acts[key]:
            rule = a["rule"]
            kind = RULE_KIND[rule]
            if kind is None:
                ser = f"{rule}\n"
            else:
                s4 = sites_preorder(parent, kind)
                hits = []
                for i, cand in enumerate(s4):
                    ck, _ = apply_at(parent, rule, cand)
                    if any(sha(k) == a["child_hash"]
                           for k in set(ck) & accepted[rule]):
                        hits.append((i, cand))
                if not hits:
                    raise SystemExit(f"UNADDRESSABLE {key} {rule}")
                ordinal, node = hits[0]
                if rule == "i_parts":
                    uc_map, _ = iparts_children(parent, node)
                    uc = [u for u, k in uc_map.items()
                          if sha(k) == a["child_hash"]]
                    if len(uc) != 1:
                        raise SystemExit(f"U-AMBIG {key}")
                    ser = f"{rule} {kind}{ordinal} u{uc[0]}\n"
                else:
                    bkeys = sorted(
                        set(apply_at(parent, rule, node)[0])
                        & accepted[rule])
                    branch = [i for i, k in enumerate(bkeys)
                              if sha(k) == a["child_hash"]]
                    if len(branch) != 1:
                        raise SystemExit(f"BRANCH {key} {rule}")
                    ser = (f"{rule} {kind}{ordinal} b{branch[0]}\n"
                           if branch[0] > 0
                           else f"{rule} {kind}{ordinal}\n")
            pl = tlen(ser)
            rl = tlen(rule)
            nl = pl - rl + 1
            prog_lens.append(pl)
            norm_lens.append(nl)
            per_rule[rule].append(pl)
            if rule == "i_unprod":
                unprod_lens.append(pl)
            field_tokens["rule"].append(rl)
            if kind is None:
                field_tokens["site"].append(0)
                field_tokens["param"].append(0)
            else:
                site_t = tlen(f" {kind}{ordinal}")
                field_tokens["site"].append(site_t)
                field_tokens["param"].append(pl - rl - site_t
                                             - tlen("\n") + 0)
            cl = tlen(
                v1child[(key[0], key[1], a["child_hash"])] + "\n")
            child_lens.append(cl)
            drow_raw.append(pl)
            drow_norm.append(nl)
            n_actions += 1
            if prefix + pl <= 512:
                fit512 += 1
            if prefix + pl <= 4096:
                fit4096 += 1
        spans_raw.append(max(drow_raw) - min(drow_raw))
        spans_norm.append(max(drow_norm) - min(drow_norm))
        if prefix + 1 > 512:
            parent_only_512.append(list(key))
        elif any(prefix + p > 512 for p in drow_raw):
            action_induced_512.append(list(key))

    raw_span_p90 = pct(spans_raw, 0.9)
    norm_span_p90 = pct(spans_norm, 0.9)
    bars = {
        "bar1_program_max_le_32": max(prog_lens) <= 32,
        "bar2_zero_action_induced_512_overflow":
            len(action_induced_512) == 0,
        "bar3_decision_span_p90_le_8": raw_span_p90 <= 8,
    }
    verdict = {
        "decisions": len(acts), "actions": n_actions,
        "bars": bars,
        "program_len": summ(prog_lens),
        "child_len": summ(child_lens),
        "median_compression": round(
            statistics.median(child_lens)
            / statistics.median(prog_lens), 2),
        "parent_prefix_len": summ(list(prefix_lens.values())),
        "fit512_actions": fit512, "fit4096_actions": fit4096,
        "parent_only_over_512_decisions": parent_only_512,
        "action_induced_over_512_decisions": action_induced_512,
        "decision_span_raw": summ(spans_raw),
        "decision_span_raw_p90": raw_span_p90,
        "decision_span_opcode_normalized": summ(spans_norm),
        "decision_span_norm_p90": norm_span_p90,
        "opcode_span_reduction_p90": (
            None if raw_span_p90 == 0 else
            round(1 - norm_span_p90 / raw_span_p90, 3)),
        "program_len_opcode_normalized": summ(norm_lens),
        "per_rule_program_len": {r: summ(v)
                                 for r, v in per_rule.items()},
        "field_tokens": {k: summ(v)
                         for k, v in field_tokens.items()},
        "i_unprod_fenced": (summ(unprod_lens)
                            if unprod_lens else None),
        "i_unprod_n": len(unprod_lens),
        "start": START, "completion_commit": completion_commit()}
    OUT.write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("start",
                                   "parent_only_over_512_decisions",
                                   "action_induced_over_512_decisions"
                                   )}, indent=1), flush=True)
    print(f"parent_only_512={len(parent_only_512)} "
          f"action_induced_512={len(action_induced_512)}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

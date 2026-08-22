"""MATH-CYBER-1 semantic corpus exporter for the axiom interchange
layer (AX-MATHWORLD-REPLICA-DESK-0, axiom spec
docs/specs/2026-08-22-mathworld-interchange.md frozen through their
9d4933c). Cold-process re-walk of the frozen rung-0 ACTIVE receipts
that emits the SEMANTIC payloads the hash-only receipts cannot
carry: sympy sstr text of every state, rule_target, and legal-set
child, plus the four backend-local binding fields byte-equal to the
frozen rows.

Frozen-world guarantee: before any emission, every world source
file is asserted BYTE-IDENTICAL to its content at MATH-CYBER-0's
code_commit=620da3bf (git show); abort nonzero otherwise. The
frozen active/replay/coldreplay evidence is read, never written.

Row classes (axiom spec, frozen): decision / dead_end_marker /
presolved_marker / wall_cap_marker. Action identity in the corpus:
(rule, rule_target, child sstr + child_hash). rule and rule_target
derive from the enumerator's action label, which is
"{rule}@{sstr(target)}" for node-targeted rules (Derivative rules
target the node; Integral rules target the exact synthetic
Integral(function, innermost_limit) handed to the primitive —
matching the spec's rule_target semantics) and a bare name for
whole-expression algebra moves (rule_target = null). sympy sstr
never emits '@' on this corpus (asserted per label).

Binding check per decision row, all four fields byte-equal to the
frozen active.jsonl row: state_before_hash, legal_action_set_hash,
chosen_action_backend_local (the raw chosen_action string),
state_after_hash. Marker rows bind on state_before_hash (+ the
dead-end's real empty-set hash). Any mismatch aborts exit 3.

Receipts: logs/mathworld1/{states.jsonl, actions.jsonl,
export_verdict.json} (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_export.py             (Mac)
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import State, successors  # noqa: E402

FROZEN_COMMIT = "620da3bf"
FROZEN_SOURCES = ["scratch/mathworld0.py",
                  "llmopt/search/derivation.py",
                  "llmopt/mathgen/problems.py"]
DIR = Path("logs/mathworld1")
STATES = DIR / "states.jsonl"
ACTIONS = DIR / "actions.jsonl"
VERDICT = DIR / "export_verdict.json"
X = sp.Symbol("x")


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def assert_frozen_world():
    for p in FROZEN_SOURCES:
        frozen = subprocess.run(
            ["git", "show", f"{FROZEN_COMMIT}:{p}"],
            capture_output=True, check=True).stdout
        live = Path(p).read_bytes()
        if frozen != live:
            raise SystemExit(
                f"FROZEN-SOURCE MISMATCH: {p} differs from "
                f"{FROZEN_COMMIT}; run in a frozen worktree")


def split_label(label: str):
    """(rule, rule_target sstr | None) from an enumerator label."""
    if "@" in label:
        rule, target = label.split("@", 1)
        if "@" in target:
            raise SystemExit(f"AMBIGUOUS LABEL (multiple '@'): {label}")
        return rule, target
    return label, None


def main():
    for p in (STATES, ACTIONS, VERDICT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    DIR.mkdir(parents=True, exist_ok=True)
    assert_frozen_world()
    START = start_provenance(
        ["scratch/mathworld1_export.py"] + FROZEN_SOURCES)
    rows = [json.loads(line) for line in
            Path("logs/mathworld0/active.jsonl").read_text()
            .splitlines() if "meta" not in line]
    eps = {}
    for r in rows:
        eps.setdefault(r["episode_id"], []).append(r)
    n_states = n_actions = n_decisions = 0
    class_counts = {}
    t0 = time.monotonic()
    with STATES.open("a") as fs, ACTIONS.open("a") as fa:
        for eid, erows in eps.items():
            lv = int(eid[1])
            seed = int(eid.split("-s")[1])
            prob = make_integrate(lv, seed)
            st = State(sp.Integral(prob._expr, X))
            for r in erows:
                before_sstr = sp.sstr(st.expr)
                if sha(st.key()) != r["state_before_hash"]:
                    raise SystemExit(
                        f"BINDING MISMATCH state_before {eid} "
                        f"step {r['step_id']}")
                if r["chosen_action"]:
                    row_class = "decision"
                    acts = sorted(successors(st),
                                  key=lambda nc: (nc[0], nc[1].key()))
                    ah = sha("\n".join(f"{n}|{c.key()}"
                                       for n, c in acts))
                    if (ah != r["legal_action_set_hash"]
                            or len(acts) != r["n_legal"]):
                        raise SystemExit(
                            f"BINDING MISMATCH legal set {eid} "
                            f"step {r['step_id']}")
                    for idx, (name, c) in enumerate(acts):
                        rule, target = split_label(name)
                        fa.write(json.dumps({
                            "episode_id": eid,
                            "step_id": r["step_id"], "idx": idx,
                            "rule": rule, "rule_target": target,
                            "child": sp.sstr(c.expr),
                            "child_hash": sha(c.key())}) + "\n")
                        n_actions += 1
                    match = [nc for nc in acts
                             if f"{nc[0]}#{sha(nc[1].key())}"
                             == r["chosen_action"]]
                    if len(match) != 1:
                        raise SystemExit(
                            f"BINDING MISMATCH chosen {eid} "
                            f"step {r['step_id']}")
                    name, child = match[0]
                    if sha(child.key()) != r["state_after_hash"]:
                        raise SystemExit(
                            f"BINDING MISMATCH state_after {eid} "
                            f"step {r['step_id']}")
                    rule, target = split_label(name)
                    fs.write(json.dumps({
                        "episode_id": eid, "step_id": r["step_id"],
                        "row_class": row_class,
                        "state_before": before_sstr,
                        "state_before_hash": r["state_before_hash"],
                        "n_legal": r["n_legal"],
                        "chosen_rule": rule,
                        "chosen_rule_target": target,
                        "chosen_child_hash": sha(child.key()),
                        "chosen_action_backend_local":
                            r["chosen_action"],
                        "state_after": sp.sstr(child.expr),
                        "state_after_hash": r["state_after_hash"],
                        "outcome": r["outcome"],
                        "legal_action_set_hash":
                            r["legal_action_set_hash"]}) + "\n")
                    n_decisions += 1
                    st = child
                else:
                    if r["outcome"] == "dead_end":
                        row_class = "dead_end_marker"
                        if r["legal_action_set_hash"] != sha(""):
                            raise SystemExit(
                                f"BINDING MISMATCH dead-end hash "
                                f"{eid} step {r['step_id']}")
                    elif r["outcome"] == "wall_cap":
                        row_class = "wall_cap_marker"
                    elif r["outcome"] == "solved":
                        row_class = "presolved_marker"
                    else:
                        raise SystemExit(
                            f"UNKNOWN MARKER {r['outcome']} {eid}")
                    fs.write(json.dumps({
                        "episode_id": eid, "step_id": r["step_id"],
                        "row_class": row_class,
                        "state_before": before_sstr,
                        "state_before_hash": r["state_before_hash"],
                        "n_legal": 0, "chosen_rule": None,
                        "chosen_rule_target": None,
                        "chosen_child_hash": None,
                        "chosen_action_backend_local": None,
                        "state_after": before_sstr,
                        "state_after_hash": r["state_after_hash"],
                        "outcome": r["outcome"],
                        "legal_action_set_hash":
                            r["legal_action_set_hash"]}) + "\n")
                class_counts[row_class] = \
                    class_counts.get(row_class, 0) + 1
                n_states += 1
    verdict = {
        "frozen_world_commit": FROZEN_COMMIT,
        "episodes": len(eps), "state_rows": n_states,
        "decision_rows": n_decisions, "action_rows": n_actions,
        "row_classes": class_counts,
        "binding": "all rows byte-equal to frozen active.jsonl "
                   "(abort-on-mismatch; none fired)",
        "wall_s": round(time.monotonic() - t0, 1),
        "sha256": {p.name: hashlib.sha256(
            p.read_bytes()).hexdigest()
            for p in (STATES, ACTIONS)},
        "start": START, "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("start",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

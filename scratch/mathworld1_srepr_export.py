"""MATH-CYBER-1 SREPR-EXPORT-0 — versioned srepr interchange
re-export of the frozen rung-0 corpus (per outside GO 2026-08-24).

The v1 semantic export (scratch/mathworld1_export.py, receipts
logs/mathworld1/{states,actions}.jsonl) serializes states with
sympy sstr; ACTIONPROG-QUAL-0 measured that sympify(sstr) fails to
reproduce the frozen State.key() on 26/101 decisions — the str()
export UNDER-DETERMINES state identity. This producer re-walks the
same frozen world and emits State.key() itself (sp.srepr) for
every state and legal child, into NEW versioned paths. The v1
corpus is frozen evidence and is never written; its sha256 is
recorded before and after the run as proof.

Qualification asserted IN-PROCESS per emitted string (abort, never
skip): sp.srepr(sp.sympify(s)) == s AND s == the live object's
State.key(). Binding gates identical to v1: every decision row
byte-equal to frozen active.jsonl on state_before_hash,
legal_action_set_hash, chosen_action_backend_local,
state_after_hash; marker rows on state_before_hash (+ the
dead-end empty-set hash). Frozen-world guarantee identical to v1:
every world source byte-identical to code_commit=620da3bf.

Action rows keep the v1 identity fields (rule, rule_target = the
enumerator label's sstr target, child_hash) and replace child with
srepr; state rows replace state_before/state_after with srepr.

Receipts: logs/mathworld1/{states_srepr.jsonl, actions_srepr.jsonl,
srepr_export_verdict.json} (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_srepr_export.py       (Mac)
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
STATES = DIR / "states_srepr.jsonl"
ACTIONS = DIR / "actions_srepr.jsonl"
VERDICT = DIR / "srepr_export_verdict.json"
V1_FROZEN = [DIR / "states.jsonl", DIR / "actions.jsonl"]
X = sp.Symbol("x")


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def fsha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
    if "@" in label:
        rule, target = label.split("@", 1)
        if "@" in target:
            raise SystemExit(f"AMBIGUOUS LABEL (multiple '@'): {label}")
        return rule, target
    return label, None


def qualified_srepr(st: State) -> str:
    """State.key() with the round-trip identity asserted, or abort."""
    s = st.key()
    rt = sp.srepr(sp.sympify(s))
    if rt != s:
        raise SystemExit(f"ROUND-TRIP FAILURE: {s[:120]}...")
    return s


def main():
    for p in (STATES, ACTIONS, VERDICT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    v1_sha_before = {p.name: fsha(p) for p in V1_FROZEN}
    assert_frozen_world()
    START = start_provenance(
        ["scratch/mathworld1_srepr_export.py"] + FROZEN_SOURCES)
    rows = [json.loads(line) for line in
            Path("logs/mathworld0/active.jsonl").read_text()
            .splitlines() if "meta" not in line]
    eps = {}
    for r in rows:
        eps.setdefault(r["episode_id"], []).append(r)
    n_states = n_actions = n_decisions = n_roundtrip = 0
    class_counts = {}
    t0 = time.monotonic()
    with STATES.open("a") as fs, ACTIONS.open("a") as fa:
        for eid, erows in eps.items():
            lv = int(eid[1])
            seed = int(eid.split("-s")[1])
            prob = make_integrate(lv, seed)
            st = State(sp.Integral(prob._expr, X))
            for r in erows:
                before_srepr = qualified_srepr(st)
                n_roundtrip += 1
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
                            "child": qualified_srepr(c),
                            "child_hash": sha(c.key())}) + "\n")
                        n_roundtrip += 1
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
                        "state_before": before_srepr,
                        "state_before_hash": r["state_before_hash"],
                        "n_legal": r["n_legal"],
                        "chosen_rule": rule,
                        "chosen_rule_target": target,
                        "chosen_child_hash": sha(child.key()),
                        "chosen_action_backend_local":
                            r["chosen_action"],
                        "state_after": qualified_srepr(child),
                        "state_after_hash": r["state_after_hash"],
                        "outcome": r["outcome"],
                        "legal_action_set_hash":
                            r["legal_action_set_hash"]}) + "\n")
                    n_roundtrip += 1
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
                        "state_before": before_srepr,
                        "state_before_hash": r["state_before_hash"],
                        "n_legal": 0, "chosen_rule": None,
                        "chosen_rule_target": None,
                        "chosen_child_hash": None,
                        "chosen_action_backend_local": None,
                        "state_after": before_srepr,
                        "state_after_hash": r["state_after_hash"],
                        "outcome": r["outcome"],
                        "legal_action_set_hash":
                            r["legal_action_set_hash"]}) + "\n")
                class_counts[row_class] = \
                    class_counts.get(row_class, 0) + 1
                n_states += 1
    v1_sha_after = {p.name: fsha(p) for p in V1_FROZEN}
    if v1_sha_before != v1_sha_after:
        raise SystemExit("V1 CORPUS MUTATED — INVESTIGATE")
    verdict = {
        "frozen_world_commit": FROZEN_COMMIT,
        "episodes": len(eps), "state_rows": n_states,
        "decision_rows": n_decisions, "action_rows": n_actions,
        "row_classes": class_counts,
        "roundtrip_asserted": n_roundtrip,
        "binding": "all rows byte-equal to frozen active.jsonl "
                   "(abort-on-mismatch; none fired)",
        "v1_frozen_sha256_before": v1_sha_before,
        "v1_frozen_sha256_after": v1_sha_after,
        "wall_s": round(time.monotonic() - t0, 1),
        "sha256": {p.name: fsha(p) for p in (STATES, ACTIONS)},
        "start": START, "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("start",)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

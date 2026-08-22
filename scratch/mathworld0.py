"""MATH-CYBER-0 rung 0: the MATHWORLD minimal contract as a
receipt wrapper over the standing legal-action enumerator
(llmopt/search/derivation.py successors()), per OBSERVATION
MATHWORLD-DESK-0 + AMENDMENT -BUDGET.

The world (legal-action semantics, proposal-mode banked):
  legal_actions(state) -> the admitted (rule@locus, child) pairs
  from successors(), SORTED by (action name, child key) so the
  action set and its hash are order-stable; the action-set hash
  and state hashes are sha256 over sympy srepr text.
  step() applies one chosen action and emits ONE transition
  receipt row with the fixed rung-0 schema:
    (episode_id, step_id, state_before_hash,
     legal_action_set_hash, n_legal, chosen_action,
     state_after_hash, outcome, transition_wall_ms)
  outcome in {ongoing, solved, dead_end, budget_exhausted,
  wall_cap}. Budgets per AMENDMENT -BUDGET: primary = FIXED
  LOGICAL DECISIONS per episode; wall = SAFETY CAP only, checked
  between decisions and recorded per transition.

Stages:
  ACTIVE  episodes at L4-7 (the desk-promoted band), scripted
          deterministic policy = greedy hce with (hce, name,
          child key) tie-break — rung 0 tests the WORLD, no
          model in the loop. Receipts -> active.jsonl.
  REPLAY  deterministic-replay qualification: every episode
          re-run by replaying the recorded chosen_action
          strings through the same world. The receipt chain
          must be IDENTICAL on every field except
          transition_wall_ms. Receipts -> replay.jsonl;
          verdict -> replay_verdict.json (per-row field-level
          comparison, fail-closed exit 3 on any mismatch).

Receipts: logs/mathworld0/{PRE}active.jsonl, {PRE}replay.jsonl,
{PRE}replay_verdict.json (refuse-if-exists; SMOKE=1 -> smoke_
paths, 4 episodes L4 only).

    .venv/bin/python scratch/mathworld0.py                    (Mac)
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.mathgen.problems import make_integrate  # noqa: E402
from llmopt.search.derivation import (State, hce,  # noqa: E402
                                      is_solved, successors)

SMOKE = os.environ.get("SMOKE") == "1"
PRE = "smoke_" if SMOKE else ""
DIR = Path("logs/mathworld0")
ACTIVE = DIR / f"{PRE}active.jsonl"
REPLAY = DIR / f"{PRE}replay.jsonl"
VERDICT = DIR / f"{PRE}replay_verdict.json"

LEVELS = [4] if SMOKE else [4, 5, 6, 7]
N_PER_LEVEL = 4 if SMOKE else 10
MAX_DECISIONS = 12          # fixed logical budget, primary
WALL_CAP_S = 60.0           # per-episode safety cap, secondary
X = sp.Symbol("x")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def state_hash(state: State) -> str:
    return sha(state.key())


def legal_actions(state: State):
    """Admitted actions, order-stable: sorted by (name, child key)."""
    acts = sorted(successors(state),
                  key=lambda nc: (nc[0], nc[1].key()))
    aset_hash = sha("\n".join(f"{n}|{c.key()}" for n, c in acts))
    return acts, aset_hash


def run_episode(episode_id, root, sink, script=None):
    """One episode. script=None -> greedy-hce policy (ACTIVE);
    script=[action names] -> replay those choices in order.
    Returns the list of receipt rows written."""
    state = State(root)
    t_ep = time.monotonic()
    rows = []
    for step_id in range(MAX_DECISIONS):
        t0 = time.monotonic()
        before = state_hash(state)
        if is_solved(state):
            outcome = "solved"
            rows.append({"episode_id": episode_id, "step_id": step_id,
                         "state_before_hash": before,
                         "legal_action_set_hash": None, "n_legal": 0,
                         "chosen_action": None,
                         "state_after_hash": before,
                         "outcome": outcome,
                         "transition_wall_ms": 0})
            break
        if time.monotonic() - t_ep > WALL_CAP_S:
            rows.append({"episode_id": episode_id, "step_id": step_id,
                         "state_before_hash": before,
                         "legal_action_set_hash": None, "n_legal": 0,
                         "chosen_action": None,
                         "state_after_hash": before,
                         "outcome": "wall_cap",
                         "transition_wall_ms": 0})
            break
        acts, aset_hash = legal_actions(state)
        if not acts:
            rows.append({"episode_id": episode_id, "step_id": step_id,
                         "state_before_hash": before,
                         "legal_action_set_hash": aset_hash,
                         "n_legal": 0, "chosen_action": None,
                         "state_after_hash": before,
                         "outcome": "dead_end",
                         "transition_wall_ms": round(
                             (time.monotonic() - t0) * 1000, 1)})
            break
        # action identity = rule name + child hash: rule names are
        # NOT unique within a legal set (one rule, many loci), so a
        # replayable receipt must pin the (name, child) pair
        if script is None:
            name, child = min(
                acts, key=lambda nc: (hce(nc[1]), nc[0], nc[1].key()))
        else:
            if step_id >= len(script) or script[step_id] is None:
                break
            match = [nc for nc in acts
                     if f"{nc[0]}#{state_hash(nc[1])}"
                     == script[step_id]]
            if not match:
                rows.append({"episode_id": episode_id,
                             "step_id": step_id,
                             "state_before_hash": before,
                             "legal_action_set_hash": aset_hash,
                             "n_legal": len(acts),
                             "chosen_action": script[step_id],
                             "state_after_hash": None,
                             "outcome": "replay_action_missing",
                             "transition_wall_ms": 0})
                break
            name, child = match[0]
        state = child
        solved_now = is_solved(state)
        last = step_id == MAX_DECISIONS - 1
        outcome = ("solved" if solved_now
                   else "budget_exhausted" if last else "ongoing")
        rows.append({"episode_id": episode_id, "step_id": step_id,
                     "state_before_hash": before,
                     "legal_action_set_hash": aset_hash,
                     "n_legal": len(acts),
                     "chosen_action": f"{name}#{state_hash(child)}",
                     "state_after_hash": state_hash(state),
                     "outcome": outcome,
                     "transition_wall_ms": round(
                         (time.monotonic() - t0) * 1000, 1)})
        if solved_now:
            break
    for r in rows:
        sink.write(json.dumps(r) + "\n")
    sink.flush()
    return rows


def main():
    for pth in (ACTIVE, REPLAY, VERDICT):
        if pth.exists():
            raise SystemExit(f"REFUSING: {pth} exists")
    DIR.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/mathworld0.py", "llmopt/search/derivation.py",
         "llmopt/mathgen/problems.py"])
    episodes = []
    for level in LEVELS:
        for i in range(N_PER_LEVEL):
            p = make_integrate(level, 9100 + i)
            episodes.append((f"L{level}-s{9100 + i}",
                             sp.Integral(p._expr, X)))
    # ACTIVE
    active_rows = {}
    t0 = time.time()
    with ACTIVE.open("a") as f:
        for eid, root in episodes:
            active_rows[eid] = run_episode(eid, root, f)
            print(f"[mw0] active {eid}: "
                  f"{active_rows[eid][-1]['outcome']} "
                  f"({len(active_rows[eid])} rows)", flush=True)
        f.write(json.dumps({"meta": {
            "note": "MATH-CYBER-0 rung 0 ACTIVE (greedy-hce policy, "
                    "fixed 12-decision budget, 60s wall cap)",
            "start": START, "wall_s": round(time.time() - t0, 1),
            "completion_commit": completion_commit()}}) + "\n")
    # REPLAY qualification
    mismatches = []
    with REPLAY.open("a") as f:
        for eid, root in episodes:
            script = [r["chosen_action"] for r in active_rows[eid]]
            rep = run_episode(eid, root, f, script=script)
            a, b = active_rows[eid], rep
            if len(a) != len(b):
                mismatches.append((eid, "row_count", len(a), len(b)))
                continue
            for ra, rb in zip(a, b):
                for k in ra:
                    if k == "transition_wall_ms":
                        continue
                    if ra[k] != rb[k]:
                        mismatches.append(
                            (eid, ra["step_id"], k, ra[k], rb[k]))
        f.write(json.dumps({"meta": {
            "note": "MATH-CYBER-0 rung 0 REPLAY qualification",
            "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    n_rows = sum(len(v) for v in active_rows.values())
    verdict = {
        "episodes": len(episodes), "rows": n_rows,
        "replay_identical_rows": n_rows - len(mismatches),
        "mismatches": mismatches[:50],
        "pass": not mismatches,
        "outcomes": {o: sum(1 for v in active_rows.values()
                            if v[-1]["outcome"] == o)
                     for o in {v[-1]["outcome"]
                               for v in active_rows.values()}},
        "start": START, "completion_commit": completion_commit()}
    VERDICT.write_text(json.dumps(verdict, indent=1))
    print(f"[mw0] replay: {verdict['replay_identical_rows']}/{n_rows}"
          f" identical rows; pass={verdict['pass']}", flush=True)
    if not verdict["pass"]:
        print("[mw0] REPLAY QUALIFICATION FAIL", flush=True)
        sys.exit(3)
    print("[mw0] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

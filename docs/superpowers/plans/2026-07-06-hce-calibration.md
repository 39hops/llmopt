# HCE Calibration Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether `hce(state)` predicts state solvability, via on-policy state sampling and cost-based probing.

**Architecture:** A `trace` hook on `beam_search` collects every candidate state generated during real searches; `scripts/calibrate_hce.py` samples those states over mathgen problems, probes each with fresh budget-limited searches, and reports Spearman ρ (hce vs nodes-to-solve) plus a per-HCE-bin solve-rate table.

**Tech Stack:** sympy, pytest. Pure Python, CPU only, no new dependencies (Spearman implemented inline).

**Spec:** `docs/superpowers/specs/2026-07-06-hce-calibration-design.md`

## Global Constraints

- No scipy/matplotlib — Spearman rank correlation implemented inline.
- String seeds for all sampling (`random.Random(f"...-{level}-0")`).
- Probe budgets: generous arm width=8, max_plies=20; binary arm `max_nodes=40`.
- No HCE weight changes regardless of result; weak ρ is reported honestly.
- Measured numbers go in the commit message.

---

### Task 1: `trace` hook on beam_search

**Files:**
- Modify: `llmopt/search/derivation.py` (beam_search signature + one append)
- Test: `tests/test_derivation_search.py`

**Interfaces:**
- Consumes: existing `beam_search`, `State`, `successors`.
- Produces: `beam_search(expr, *, width=8, max_plies=12, max_nodes=None, use_macros=False, trace=None)`; when `trace` is a list, every generated candidate `State` is appended to it. No behavior change when None.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_derivation_search.py`:

```python
def test_trace_collects_equivalent_states():
    trace = []
    root = sp.Derivative(x**2 * sp.sin(x), x)
    r = beam_search(root, trace=trace)
    assert r.solved
    assert len(trace) >= r.state.plies  # at least the winning path
    for s in trace:
        assert sp.simplify(s.expr.doit() - root.doit()) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_derivation_search.py::test_trace_collects_equivalent_states -q`
Expected: FAIL — `TypeError: beam_search() got an unexpected keyword argument 'trace'`

- [ ] **Step 3: Implement**

In `llmopt/search/derivation.py`, add `trace: list[State] | None = None`
to the `beam_search` keyword arguments, and inside the candidate loop —
immediately after `visited.add(child.key())` — add:

```python
                if trace is not None:
                    trace.append(child)
```

- [ ] **Step 4: Run the search test files**

Run: `.venv/bin/python -m pytest tests/test_derivation_search.py tests/test_diff_rules.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add llmopt/search/derivation.py tests/test_derivation_search.py
git commit -m "feat: search — trace hook on beam_search (on-policy state sampling for HCE calibration)"
```

---

### Task 2: Calibration script

**Files:**
- Create: `scripts/calibrate_hce.py`

**Interfaces:**
- Consumes: `beam_search(..., trace=)` from Task 1; `hce`, `is_solved`, `State` from `llmopt.search.derivation`; `llmopt.mathgen.problems._expression`.
- Produces: CLI only.

- [ ] **Step 1: Write the script**

Create `scripts/calibrate_hce.py`:

```python
"""HCE calibration: does hce(state) predict solvability? (spec:
2026-07-06-hce-calibration-design.md — the chess-eval question,
measured.)

Samples states on-policy from real beam searches over mathgen
differentiation problems, probes each with fresh searches, reports
Spearman rho (hce vs nodes-to-solve) and a per-HCE-bin table. Rank
correlation because HCE only needs to ORDER states for beam pruning.

  python scripts/calibrate_hce.py --levels 1 2 3 --per-level 15 --max-states 300
"""

from __future__ import annotations

import argparse
import random
import statistics

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import State, beam_search, hce, is_solved

X = sp.Symbol("x")
SMALL_BUDGET = 40  # nodes: the binary solved-within-budget arm


def spearman(xs: list[float], ys: list[float]) -> float:
    """Spearman rank correlation, average ranks for ties. Inline to
    avoid a scipy dependency."""

    def ranks(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        r = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return float("nan")
    return cov / (vx * vy) ** 0.5


def sample_states(levels: list[int], per_level: int, max_states: int) -> list[State]:
    """On-policy: every candidate generated by real searches, deduped."""
    seen: set[str] = set()
    out: list[State] = []
    for level in levels:
        rng = random.Random(f"calibrate-hce-{level}-0")  # string seed
        for _ in range(per_level):
            trace: list[State] = []
            beam_search(sp.Derivative(_expression(rng, level), X),
                        max_plies=20, trace=trace)
            for s in trace:
                if s.key() not in seen and not is_solved(s):
                    seen.add(s.key())
                    out.append(s)
    rng = random.Random("calibrate-hce-subsample-0")
    if len(out) > max_states:
        out = rng.sample(out, max_states)
    return out


def probe(state: State) -> tuple[int | None, bool]:
    """(nodes-to-solve under generous budget or None, solved@small)."""
    generous = beam_search(state.expr, width=8, max_plies=20)
    small = beam_search(state.expr, width=8, max_plies=20,
                        max_nodes=SMALL_BUDGET)
    return (generous.nodes if generous.solved else None), small.solved


def main(levels: list[int], per_level: int, max_states: int) -> None:
    states = sample_states(levels, per_level, max_states)
    rows = []
    for s in states:
        nodes, small_ok = probe(s)
        rows.append((hce(s), nodes, small_ok))
    print(f"# HCE calibration — {len(rows)} on-policy states, "
          f"levels {levels}, small budget {SMALL_BUDGET} nodes")

    solved_rows = [(h, n) for h, n, _ in rows if n is not None]
    rho = spearman([h for h, _ in solved_rows], [n for _, n in solved_rows])
    print(f"solved under generous budget: {len(solved_rows)}/{len(rows)}")
    print(f"Spearman rho, hce vs nodes-to-solve: {rho:+.3f}  "
          f"(+1 = perfect ordering: low eval -> cheap solve)")

    # per-HCE-bin table (quintile bins over observed hce values)
    hs = sorted(h for h, _, _ in rows)
    edges = [hs[int(len(hs) * q / 5)] for q in range(1, 5)] + [float("inf")]
    print(f"{'hce bin':>16} {'n':>4} {'solve@40':>9} {'mean nodes':>11}")
    lo = float("-inf")
    for hi in edges:
        binned = [(n, ok) for h, n, ok in rows if lo < h <= hi]
        if binned:
            ns = [n for n, _ in binned if n is not None]
            mn = statistics.mean(ns) if ns else float("nan")
            rate = sum(ok for _, ok in binned) / len(binned)
            label = f"({lo:.0f}, {hi:.0f}]"
            print(f"{label:>16} {len(binned):>4} {rate:>9.2f} {mn:>11.1f}")
        lo = hi


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--per-level", type=int, default=15)
    ap.add_argument("--max-states", type=int, default=300)
    a = ap.parse_args()
    main(a.levels, a.per_level, a.max_states)
```

- [ ] **Step 2: Smoke-run small**

Run: `.venv/bin/python scripts/calibrate_hce.py --levels 1 --per-level 3 --max-states 40`
Expected: prints rho and a bin table without errors (numbers small-sample noisy — fine).

- [ ] **Step 3: Full run**

Run: `.venv/bin/python scripts/calibrate_hce.py`
Expected: completes in minutes; record rho and the bin table.

- [ ] **Step 4: Commit with measured numbers**

```bash
git add scripts/calibrate_hce.py
git commit -m "feat: scripts — HCE calibration harness (measured: rho=<paste>, table in body)"
```

---

## Self-review notes

- Spec coverage: trace hook + equivalence test (Task 1); sampling with
  dedup/string seeds/skip-solved, probe with both budgets, inline
  Spearman, bin table, honest reporting (Task 2). Numbers to commit
  message per convention.
- No placeholders; interfaces match Task 1's produced signature.

# HCE Rung 1: Primitive Differentiation Moves — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace rung-0's omniscient `doit_one` move with 6 primitive differentiation rules so beam search composes multi-ply derivations and sympy only verifies edges.

**Architecture:** New `llmopt/search/rules.py` holds the primitive rules (pure functions `Derivative → list[Expr]`); `llmopt/search/derivation.py` is modified so `successors()` enumerates (rule, Derivative-node) pairs plus trimmed whole-expression algebra moves, verifying every edge against sympy. A small bench script measures solve rate and runs the macro ablation.

**Tech Stack:** sympy, pytest. Pure Python — no GPU, no model.

**Spec:** `docs/superpowers/specs/2026-07-06-hce-rung1-primitive-moves-design.md`

## Global Constraints

- Domain: single-variable, first-order differentiation only (`Derivative(f, x)`).
- Every search edge sympy-verified: `sp.simplify(parent.doit() - child.doit()) == 0`.
- `simplify` and `doit_one` must NOT appear in the move set (doit is verifier/test-only).
- Answers checked by symbolic equivalence, never string match (repo convention).
- One deliberate deviation from the spec, motivated by `d_product` split branching: `DiffRule` returns `list[sp.Expr]` (empty = no match) instead of `sp.Expr | None`. Record this in the rules.py docstring.

## File Structure

- Create: `llmopt/search/rules.py` — the 6 core rules, the macro list, the chain table. No search logic.
- Modify: `llmopt/search/derivation.py` — move enumeration, edge verification, beam plumbing (`use_macros`, `max_nodes`).
- Create: `tests/test_diff_rules.py` — per-rule property tests.
- Modify: `tests/test_derivation_search.py` — non-degeneracy regression + e2e updates.
- Create: `scripts/bench_derivation.py` — solve rate / macro ablation.

---

### Task 1: Core rule module with property tests

**Files:**
- Create: `llmopt/search/rules.py`
- Test: `tests/test_diff_rules.py`

**Interfaces:**
- Consumes: nothing new (sympy; `llmopt.mathgen.problems._expression` in tests).
- Produces: `DiffRule = Callable[[sp.Derivative], list[sp.Expr]]`;
  `CORE_RULES: list[tuple[str, DiffRule]]` with names
  `d_const, d_x, d_sum, d_product, d_power, d_chain_table`;
  `MACRO_RULES: list[tuple[str, DiffRule]]` with `d_quotient`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_diff_rules.py`:

```python
"""Per-rule property tests: every rewrite a rule emits must be
sympy-equivalent to the Derivative node it replaces."""

import random

import pytest
import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.rules import CORE_RULES, MACRO_RULES

x = sp.Symbol("x")
RULES = dict(CORE_RULES + MACRO_RULES)


def _equiv(node: sp.Derivative, rewrite: sp.Expr) -> bool:
    return sp.simplify(node.doit() - rewrite.doit()) == 0


def test_d_const():
    assert RULES["d_const"](sp.Derivative(sp.Integer(7), x)) == [0]
    assert RULES["d_const"](sp.Derivative(sp.pi, x)) == [0]
    assert RULES["d_const"](sp.Derivative(x**2, x)) == []


def test_d_x():
    assert RULES["d_x"](sp.Derivative(x, x)) == [1]
    assert RULES["d_x"](sp.Derivative(x**2, x)) == []


def test_d_sum_emits_unevaluated_terms():
    node = sp.Derivative(x**2 + sp.sin(x), x)
    (rw,) = RULES["d_sum"](node)
    # linearity costs a visible ply: children stay unevaluated
    assert rw.has(sp.Derivative)
    assert _equiv(node, rw)


def test_d_product_branches_over_splits():
    node = sp.Derivative(x**2 * sp.sin(x) * sp.exp(x), x)
    rewrites = RULES["d_product"](node)
    assert len(rewrites) == 3  # one (head, rest) split per factor
    for rw in rewrites:
        assert rw.has(sp.Derivative)
        assert _equiv(node, rw)


def test_d_power_builds_chain():
    node = sp.Derivative((x**2 + 1) ** 3, x)
    (rw,) = RULES["d_power"](node)
    assert rw.has(sp.Derivative(x**2 + 1, x))
    assert _equiv(node, rw)


def test_d_power_rejects_x_in_exponent():
    assert RULES["d_power"](sp.Derivative(x**x, x)) == []


def test_d_chain_table():
    node = sp.Derivative(sp.sin(x**2), x)
    (rw,) = RULES["d_chain_table"](node)
    assert rw.has(sp.Derivative(x**2, x))
    assert _equiv(node, rw)


def test_d_quotient_macro():
    node = sp.Derivative(sp.sin(x) / (x**2 + 1), x)
    rewrites = RULES["d_quotient"](node)
    assert rewrites, "quotient macro should fire on u/v"
    for rw in rewrites:
        assert _equiv(node, rw)
    # no x in the denominator: macro must not fire
    assert RULES["d_quotient"](sp.Derivative(sp.sin(x) / 3, x)) == []


@pytest.mark.parametrize("level", [1, 2, 3])
def test_property_all_rules_equivalent_on_generated_exprs(level):
    """Every rule, applied to Derivatives of mathgen-generated
    expressions and their subexpressions, only emits equivalent
    rewrites. String seeds per repo convention."""
    rng = random.Random(f"rules-prop-{level}-0")
    for i in range(20):
        f = _expression(rng, level)
        node = sp.Derivative(f, x)
        for name, rule in CORE_RULES + MACRO_RULES:
            for rw in rule(node):
                assert _equiv(node, rw), f"{name} broke on {f}"


def test_rules_ignore_higher_order_and_multivar():
    y = sp.Symbol("y")
    for name, rule in CORE_RULES + MACRO_RULES:
        assert rule(sp.Derivative(x**3, x, 2)) == [], name
        assert rule(sp.Derivative(x * y, x, y)) == [], name
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_diff_rules.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'llmopt.search.rules'`

- [ ] **Step 3: Implement the rule module**

Create `llmopt/search/rules.py`:

```python
"""Primitive differentiation rewrite rules (HCE rung 1, spec
2026-07-06-hce-rung1-primitive-moves-design.md).

A DiffRule takes one unevaluated Derivative node and returns the list
of candidate rewrites (usually 0 or 1; d_product returns one per
(head, rest) split). The list return generalizes the spec's
`Expr | None` signature to support split branching. Rules only fire on
single-variable first-order Derivatives; everything else returns [].

Chain rule is not a standalone move: it is fused into d_power and
d_chain_table (explicit-u chain is u-sub territory, rung 2). There is
no quotient rule in the core: sympy has no quotient node (u/v is
u * v**-1), so d_product + d_power cover it; the textbook quotient
rule lives in MACRO_RULES, off by default, ablation-gated.
"""

from __future__ import annotations

from typing import Callable

import sympy as sp

DiffRule = Callable[[sp.Derivative], "list[sp.Expr]"]


def _unpack(node: sp.Derivative) -> tuple[sp.Expr, sp.Symbol] | None:
    """(f, x) for first-order single-variable Derivatives, else None."""
    if len(node.variables) != 1:
        return None
    return node.expr, node.variables[0]


def d_const(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    return [sp.Integer(0)] if not f.has(x) else []


def d_x(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    return [sp.Integer(1)] if f == x else []


def d_sum(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Add):
        return []
    return [sp.Add(*(sp.Derivative(t, x) for t in f.args))]


def d_product(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Mul) or not f.has(x):
        return []
    out: list[sp.Expr] = []
    for i, head in enumerate(f.args):
        rest = sp.Mul(*(a for j, a in enumerate(f.args) if j != i))
        out.append(sp.Derivative(head, x) * rest + head * sp.Derivative(rest, x))
    return out


def d_power(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Pow) or f.exp.has(x) or not f.base.has(x):
        return []
    base, n = f.base, f.exp
    return [n * base ** (n - 1) * sp.Derivative(base, x)]


# h -> h' as a function of the inner expression. sqrt is Pow — d_power
# covers it, so it needs no entry here.
_CHAIN_TABLE: dict[type, Callable[[sp.Expr], sp.Expr]] = {
    sp.sin: lambda u: sp.cos(u),
    sp.cos: lambda u: -sp.sin(u),
    sp.tan: lambda u: 1 / sp.cos(u) ** 2,
    sp.exp: lambda u: sp.exp(u),
    sp.log: lambda u: 1 / u,
}


def d_chain_table(node: sp.Derivative) -> list[sp.Expr]:
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    if not (isinstance(f, sp.Function) and f.func in _CHAIN_TABLE and len(f.args) == 1):
        return []
    inner = f.args[0]
    return [_CHAIN_TABLE[f.func](inner) * sp.Derivative(inner, x)]


def d_quotient(node: sp.Derivative) -> list[sp.Expr]:
    """MACRO: textbook quotient rule. Redundant with d_product+d_power;
    kept for the solve-rate-per-node ablation only."""
    u = _unpack(node)
    if u is None:
        return []
    f, x = u
    num, den = f.as_numer_denom()
    if den == 1 or not den.has(x):
        return []
    return [(sp.Derivative(num, x) * den - num * sp.Derivative(den, x)) / den**2]


CORE_RULES: list[tuple[str, DiffRule]] = [
    ("d_const", d_const),
    ("d_x", d_x),
    ("d_sum", d_sum),
    ("d_product", d_product),
    ("d_power", d_power),
    ("d_chain_table", d_chain_table),
]

MACRO_RULES: list[tuple[str, DiffRule]] = [
    ("d_quotient", d_quotient),
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_diff_rules.py -q`
Expected: all PASS. If `test_d_product_branches_over_splits` finds fewer
than 3 rewrites, check that sympy kept `x**2 * sin(x) * exp(x)` as a
3-arg Mul (it does; `.args` is flat).

- [ ] **Step 5: Commit**

```bash
git add llmopt/search/rules.py tests/test_diff_rules.py
git commit -m "feat: search/rules — 6 primitive diff rules + d_quotient macro (rung 1)"
```

---

### Task 2: Wire (rule, node) moves into the search

**Files:**
- Modify: `llmopt/search/derivation.py`
- Test: `tests/test_derivation_search.py`

**Interfaces:**
- Consumes: `CORE_RULES`, `MACRO_RULES`, `DiffRule` from `llmopt.search.rules` (Task 1).
- Produces: `successors(state, *, use_macros=False) -> Iterator[tuple[str, State]]`;
  `beam_search(expr, *, width=8, max_plies=12, max_nodes=None, use_macros=False) -> SearchResult`;
  `verify_edge(parent: sp.Expr, child: sp.Expr) -> bool`.
  History entries: `f"{rule_name}@{sp.sstr(node)}"` for rule moves, bare
  name for algebra moves.

- [ ] **Step 1: Write the failing tests**

In `tests/test_derivation_search.py`, replace `test_beam_solves_derivative`
with the block below and append the new tests (keep
`test_successors_are_equal_to_parent`, `test_hce_prefers_solved_states`,
`test_solved_detection` as they are — they must still pass):

```python
def test_beam_solves_derivative():
    r = beam_search(sp.Derivative(x**3 + sp.sin(x), x))
    assert r.solved
    assert sp.simplify(r.state.expr - (3 * x**2 + sp.cos(x))) == 0


def test_search_is_not_degenerate():
    """Rung-0 regression: doit solved everything in ~1 ply. Rung-1
    derivations must be genuine multi-ply descents."""
    r = beam_search(sp.Derivative(x**3 + sp.sin(x), x))
    assert r.solved
    assert r.state.plies > 1
    assert not any("doit" in h or h == "simplify" for h in r.state.history)


def test_history_is_a_legible_step_chain():
    r = beam_search(sp.Derivative(x**2 * sp.sin(x), x))
    assert r.solved
    rule_steps = [h for h in r.state.history if "@" in h]
    assert rule_steps, "expected at least one rule@node entry"
    names = {h.split("@")[0] for h in rule_steps}
    assert names <= {
        "d_const", "d_x", "d_sum", "d_product", "d_power",
        "d_chain_table", "d_quotient",
    }


def test_beam_matches_sympy_on_mathgen_set():
    from llmopt.mathgen.problems import _expression
    import random

    rng = random.Random("rung1-e2e-2-0")
    for _ in range(10):
        f = _expression(rng, 2)
        r = beam_search(sp.Derivative(f, x), max_plies=20)
        assert r.solved, f
        assert sp.simplify(r.state.expr - sp.diff(f, x)) == 0, f


def test_macros_off_by_default():
    s = State(sp.Derivative(sp.sin(x) / (x**2 + 1), x))
    assert not any("d_quotient" in name for name, _ in successors(s))
    assert any(
        "d_quotient" in name for name, _ in successors(s, use_macros=True)
    )


def test_max_nodes_budget():
    r = beam_search(sp.Derivative(x**3 + sp.sin(x), x), max_nodes=2)
    assert r.nodes <= 2
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `.venv/bin/python -m pytest tests/test_derivation_search.py -q`
Expected: `test_search_is_not_degenerate` FAILS (doit_one solves in 1 ply),
`test_macros_off_by_default` / `test_max_nodes_budget` FAIL with
`TypeError: ... unexpected keyword argument`.

- [ ] **Step 3: Rewrite the move layer in derivation.py**

In `llmopt/search/derivation.py`:

3a. Replace the module docstring's KNOWN LIMITATION paragraph with:

```python
Rung 1 (2026-07-06): doit_one is gone from the move set. Moves are
(rule, Derivative-node) pairs from search.rules — the search composes
the derivation and sympy is demoted to per-edge verifier (verify_edge).
Algebra cleanup moves are kept trimmed (no simplify: an algebra
mega-move that collapses plies). doit remains available to the
verifier and to tests as ground truth. Spec:
docs/superpowers/specs/2026-07-06-hce-rung1-primitive-moves-design.md
```

3b. Add the import:

```python
from llmopt.search.rules import CORE_RULES, MACRO_RULES
```

3c. Delete `_doit_one`, delete the `Move` type alias, and replace `MOVES`
with (note: `simplify`, `doit_one`, and `apart`/`_apart_safe` are gone —
delete `_apart_safe` too):

```python
ALGEBRA_MOVES: list[tuple[str, Callable[[sp.Expr], sp.Expr]]] = [
    ("expand", sp.expand),
    ("factor", sp.factor),
    ("cancel", sp.cancel),
    ("together", sp.together),
    ("trigsimp", sp.trigsimp),
    ("powsimp", sp.powsimp),
]
```

3d. Add the edge verifier and replace `successors`:

```python
def verify_edge(parent: sp.Expr, child: sp.Expr) -> bool:
    """Oracle check: a legal move preserves the value. doit() is the
    complete solver — too strong as a mover (rung 0's mistake), exactly
    right as a verifier."""
    try:
        return sp.simplify(parent.doit() - child.doit()) == 0
    except Exception:
        return False


def successors(state: State, *, use_macros: bool = False) -> Iterator[tuple[str, State]]:
    """Legal, non-identity, sympy-verified successor states. Rule moves
    target one Derivative node ((rule, node) pairs — real branching);
    algebra moves rewrite the whole expression."""
    seen = {state.key()}
    rules = CORE_RULES + MACRO_RULES if use_macros else CORE_RULES

    def emit(name: str, new_expr: sp.Expr) -> Iterator[tuple[str, State]]:
        child = State(new_expr, state.plies + 1, state.history + (name,))
        if child.key() not in seen and verify_edge(state.expr, new_expr):
            seen.add(child.key())
            yield name, child

    for node in sorted(state.expr.atoms(sp.Derivative), key=sp.count_ops):
        for rule_name, rule in rules:
            for rewrite in rule(node):
                label = f"{rule_name}@{sp.sstr(node)}"
                yield from emit(label, state.expr.xreplace({node: rewrite}))
    for name, fn in ALGEBRA_MOVES:
        try:
            new = fn(state.expr)
        except Exception:
            continue
        yield from emit(name, new)
```

3e. Thread the new knobs through `beam_search` — new signature and the
two touch-points (`successors` call; budget check):

```python
def beam_search(
    expr: sp.Expr,
    *,
    width: int = 8,
    max_plies: int = 12,
    max_nodes: int | None = None,
    use_macros: bool = False,
) -> SearchResult:
```

Inside the loop, change `for _, child in successors(s):` to
`for _, child in successors(s, use_macros=use_macros):` and, immediately
after `nodes += 1`, add:

```python
                if max_nodes is not None and nodes >= max_nodes:
                    beam = candidates + beam
                    break
```

and after the inner two loops (before `if not candidates: break`) add the
matching outer bail-out:

```python
        if max_nodes is not None and nodes >= max_nodes:
            break
```

The rest of `beam_search` (`hce` sorting, `best_solved` bookkeeping,
final return) is unchanged.

- [ ] **Step 4: Run the full search test files**

Run: `.venv/bin/python -m pytest tests/test_derivation_search.py tests/test_diff_rules.py -q`
Expected: all PASS, including the three pre-existing tests. If
`test_beam_matches_sympy_on_mathgen_set` times out or misses, raise
`max_plies` in the test before touching HCE weights — weight tuning is
out of scope per the spec.

- [ ] **Step 5: Commit**

```bash
git add llmopt/search/derivation.py tests/test_derivation_search.py
git commit -m "feat: search — rung 1: (rule,node) moves, per-edge oracle, doit/simplify removed from move set"
```

---

### Task 3: Bench script with macro ablation

**Files:**
- Create: `scripts/bench_derivation.py`

**Interfaces:**
- Consumes: `beam_search(expr, width=, max_plies=, max_nodes=, use_macros=)`
  from Task 2; `llmopt.mathgen.problems._expression`.
- Produces: CLI only — nothing imports this.

- [ ] **Step 1: Write the script**

Create `scripts/bench_derivation.py`:

```python
"""Rung-1 solve-rate bench + macro ablation (spec: macros earn a slot
only if they win on solve-rate-per-node).

  python scripts/bench_derivation.py --levels 1 2 3 --n 30
  python scripts/bench_derivation.py --macros            # ablation arm
"""

from __future__ import annotations

import argparse
import random
import statistics

import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.derivation import beam_search

X = sp.Symbol("x")


def run(levels: list[int], n: int, width: int, max_plies: int,
        max_nodes: int | None, use_macros: bool) -> None:
    tag = "macros ON" if use_macros else "core rules only"
    print(f"# rung-1 bench — {tag}, width={width}, max_plies={max_plies}, "
          f"max_nodes={max_nodes}")
    print(f"{'level':>5} {'solved':>10} {'mean nodes':>11} {'mean plies':>11}")
    for level in levels:
        rng = random.Random(f"bench-deriv-{level}-0")  # string seed
        solved, nodes, plies = 0, [], []
        for _ in range(n):
            f = _expression(rng, level)
            r = beam_search(sp.Derivative(f, X), width=width,
                            max_plies=max_plies, max_nodes=max_nodes,
                            use_macros=use_macros)
            ok = r.solved and sp.simplify(r.state.expr - sp.diff(f, X)) == 0
            solved += ok
            nodes.append(r.nodes)
            if ok:
                plies.append(r.state.plies)
        mp = statistics.mean(plies) if plies else float("nan")
        print(f"{level:>5} {solved:>6}/{n:<3} {statistics.mean(nodes):>11.1f} "
              f"{mp:>11.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=20)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--macros", action="store_true")
    a = ap.parse_args()
    run(a.levels, a.n, a.width, a.max_plies, a.max_nodes, a.macros)
```

- [ ] **Step 2: Run both arms**

Run: `.venv/bin/python scripts/bench_derivation.py --n 20`
Run: `.venv/bin/python scripts/bench_derivation.py --n 20 --macros`
Expected: both print the table; level 1-2 solve rate should be at or
near 20/20; macros arm typically shows equal-or-higher node counts.
Record both tables in the commit message body — honest numbers either
way, per repo convention.

- [ ] **Step 3: Run the whole pure-Python suite to catch fallout**

Run: `.venv/bin/python -m pytest tests/ -q -x --ignore=tests/test_metal_kernels.py -k "derivation or diff_rules or mathgen"`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_derivation.py
git commit -m "feat: scripts — rung-1 derivation bench + --macros ablation (measured: <paste numbers>)"
```

---

## Self-review notes

- Spec coverage: 6-rule core (Task 1), (rule,node) moves + per-edge oracle
  + trimmed algebra + history labels (Task 2), macro ablation + node
  budget (Tasks 2-3), non-degeneracy regression + property tests +
  symbolic e2e (Tasks 1-2). HCE weights untouched, per spec.
- Deviation from spec recorded once (Global Constraints + rules.py
  docstring): DiffRule returns a list, not `Expr | None`.
- Out of scope confirmed: integration rules, u-sub, NNUE, model proposer,
  HCE calibration curves (bench script here reports solve rate only; the
  full eval-vs-solve-rate calibration chart is roadmap follow-up).

# Rung 2: Integration Moves Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add primitive antiderivative rules (incl. u-substitution and by-parts) so the derivation search solves integrals — the genuinely branching, failure-prone domain.

**Architecture:** `rules.py` gains an `IntRule` family over `sp.Integral` nodes (same list-return convention as `DiffRule`); `derivation.py`'s `successors()` enumerates both node families, `verify_edge` tolerates constant offsets on integral edges, and a `subs_eval` structural move collapses solved `Subs` carriers. Bench gains `--kind int` and a solve-rate-vs-node-budget table.

**Tech Stack:** sympy, pytest. Pure Python, CPU.

**Spec:** `docs/superpowers/specs/2026-07-07-rung2-integration-moves-design.md`

## Global Constraints

- Rules fire only on single-variable, single-limit `Integral(f, x)` (`len(node.limits) == 1 and len(node.limits[0]) == 1`) — sympy collapses nested integrals to multi-limit nodes, which are non-matches.
- Substitution symbol is the module-level `U = sp.Symbol("u_")` (named, not Dummy — Dummy breaks `srepr` dedup keys).
- `verify_edge`: constant-offset tolerance ONLY when the parent contains an `Integral`; Derivative-only edges keep exact zero.
- Solved integrals scored by differentiating the result against the integrand, never by comparing antiderivative strings.
- Budgets for the headline chart: 25/50/100/200/400 nodes. Numbers in the commit message.
- String seeds everywhere.

---

### Task 1: IntRule family in rules.py

**Files:**
- Modify: `llmopt/search/rules.py` (append after MACRO_RULES)
- Test: `tests/test_int_rules.py` (create)

**Interfaces:**
- Consumes: existing module conventions (`_unpack` for Derivatives is not reused; integrals get `_unpack_int`).
- Produces: `IntRule = Callable[[sp.Integral], list[sp.Expr]]`; `U = sp.Symbol("u_")`; `INT_RULES: list[tuple[str, IntRule]]` with names `i_const, i_power, i_sum, i_const_factor, i_table, i_usub, i_parts`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_int_rules.py`:

```python
"""Integration-rule property tests. Antiderivatives are equivalence
classes modulo constants, so equivalence is checked by differentiating
the difference, not by exact equality."""

import random

import pytest
import sympy as sp

from llmopt.mathgen.problems import _expression
from llmopt.search.rules import INT_RULES, U

x = sp.Symbol("x")
RULES = dict(INT_RULES)


def _equiv_mod_const(node: sp.Integral, rewrite: sp.Expr) -> bool:
    d = node.doit() - rewrite.doit()
    return sp.simplify(sp.diff(d, x)) == 0


def test_i_const():
    assert RULES["i_const"](sp.Integral(sp.Integer(5), x)) == [5 * x]
    assert RULES["i_const"](sp.Integral(x, x)) == []


def test_i_power():
    (rw,) = RULES["i_power"](sp.Integral(x**3, x))
    assert sp.simplify(rw - x**4 / 4) == 0
    (rw,) = RULES["i_power"](sp.Integral(1 / x, x))
    assert rw == sp.log(x)
    (rw,) = RULES["i_power"](sp.Integral(x, x))
    assert sp.simplify(rw - x**2 / 2) == 0


def test_i_sum_stays_unevaluated():
    node = sp.Integral(x + sp.sin(x), x)
    (rw,) = RULES["i_sum"](node)
    assert rw.has(sp.Integral)
    assert _equiv_mod_const(node, rw)


def test_i_const_factor():
    node = sp.Integral(3 * sp.sin(x), x)
    (rw,) = RULES["i_const_factor"](node)
    assert rw.has(sp.Integral(sp.sin(x), x))
    assert _equiv_mod_const(node, rw)
    assert RULES["i_const_factor"](sp.Integral(x * sp.sin(x), x)) == []


def test_i_table():
    for f, F in [(sp.sin(x), -sp.cos(x)), (sp.cos(x), sp.sin(x)),
                 (sp.exp(x), sp.exp(x))]:
        assert RULES["i_table"](sp.Integral(f, x)) == [F]
    assert RULES["i_table"](sp.Integral(sp.sin(2 * x), x)) == []


def test_i_usub_fires_on_composition():
    node = sp.Integral(2 * x * sp.cos(x**2), x)
    rewrites = RULES["i_usub"](node)
    assert any(
        isinstance(rw, sp.Subs) and rw.has(sp.Integral(sp.cos(U), U))
        for rw in rewrites
    ), rewrites
    for rw in rewrites:
        assert _equiv_mod_const(node, rw)


def test_i_usub_no_candidate_no_fire():
    assert RULES["i_usub"](sp.Integral(x**2, x)) == []


def test_i_parts_branches_and_stays_stepwise():
    node = sp.Integral(x * sp.cos(x), x)
    rewrites = RULES["i_parts"](node)
    assert rewrites, "by-parts should fire on x*cos(x)"
    for rw in rewrites:
        assert rw.has(sp.Integral)
        assert _equiv_mod_const(node, rw)


def test_rules_ignore_definite_and_multilimit():
    for name, rule in INT_RULES:
        assert rule(sp.Integral(x, (x, 0, 1))) == [], name
        assert rule(sp.Integral(sp.cos(x), x, x)) == [], name


@pytest.mark.parametrize("level", [1, 2, 3])
def test_property_on_generated_integrands(level):
    rng = random.Random(f"int-rules-prop-{level}-0")
    for _ in range(15):
        f = sp.simplify(sp.diff(_expression(rng, level), x))
        if f == 0:
            continue
        node = sp.Integral(f, x)
        for name, rule in INT_RULES:
            for rw in rule(node):
                assert _equiv_mod_const(node, rw), f"{name} broke on {f}"
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_int_rules.py -q`
Expected: collection error — `ImportError: cannot import name 'INT_RULES'`

- [ ] **Step 3: Implement — append to `llmopt/search/rules.py`**

```python
# ------------------------------------------------------ integration

IntRule = Callable[[sp.Integral], "list[sp.Expr]"]

U = sp.Symbol("u_")  # reserved substitution symbol (named: Dummy breaks srepr dedup)


def _unpack_int(node: sp.Integral) -> tuple[sp.Expr, sp.Symbol] | None:
    """(f, x) for single-variable indefinite Integrals, else None."""
    if len(node.limits) != 1 or len(node.limits[0]) != 1:
        return None
    return node.function, node.limits[0][0]


def i_const(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    return [f * x] if not f.has(x) else []


def i_power(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if f == x:
        return [x**2 / 2]
    if not (isinstance(f, sp.Pow) and f.base == x and not f.exp.has(x)):
        return []
    n = f.exp
    return [sp.log(x)] if n == -1 else [x ** (n + 1) / (n + 1)]


def i_sum(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Add):
        return []
    return [sp.Add(*(sp.Integral(t, x) for t in f.args))]


def i_const_factor(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if not isinstance(f, sp.Mul):
        return []
    const = sp.Mul(*(a for a in f.args if not a.has(x)))
    rest = sp.Mul(*(a for a in f.args if a.has(x)))
    if const == 1 or rest == 1:
        return []
    return [const * sp.Integral(rest, x)]


_INT_TABLE = {sp.sin: lambda v: -sp.cos(v), sp.cos: sp.sin, sp.exp: sp.exp}


def i_table(node: sp.Integral) -> list[sp.Expr]:
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    if isinstance(f, sp.Function) and f.func in _INT_TABLE and f.args == (x,):
        return [_INT_TABLE[f.func](x)]
    return []


def _usub_candidates(f: sp.Expr, x: sp.Symbol) -> list[sp.Expr]:
    cands = []
    for fn in f.atoms(sp.Function):
        cands.append(fn.args[0])
    for p in f.atoms(sp.Pow):
        cands.append(p.base)
    seen = set()
    out = []
    for g in cands:
        k = sp.srepr(g)
        if k not in seen and g.has(x) and g != x:
            seen.add(k)
            out.append(g)
    return out


def i_usub(node: sp.Integral) -> list[sp.Expr]:
    """u-substitution: if f == h(g)·g', rewrite to Subs(∫h(u)du, u, g).
    One branch per candidate g — wrong choices are the search's problem."""
    u = _unpack_int(node)
    if u is None:
        return []
    f, x = u
    out: list[sp.Expr] = []
    for g in _usub_candidates(f, x):
        dg = sp.diff(g, x)
        if dg == 0:
            continue
        q = sp.simplify(sp.cancel(f / dg)).subs(g, U)
        if q.has(x) or not q.has(U):
            continue
        out.append(sp.Subs(sp.Integral(q, U), U, g))
    return out


def i_parts(node: sp.Integral) -> list[sp.Expr]:
    """Integration by parts, stepwise: ∫u dv = u·∫dv − ∫(∫dv)·u'.
    Inner integrals stay unevaluated; one branch per (u, dv) split."""
    u_ = _unpack_int(node)
    if u_ is None:
        return []
    f, x = u_
    if not isinstance(f, sp.Mul):
        return []
    out: list[sp.Expr] = []
    for i, u_part in enumerate(f.args):
        du = sp.diff(u_part, x)
        if du == 0:
            continue
        dv = sp.Mul(*(a for j, a in enumerate(f.args) if j != i))
        v = sp.Integral(dv, x)
        out.append(u_part * v - sp.Integral(v * du, x))
    return out


INT_RULES: list[tuple[str, IntRule]] = [
    ("i_const", i_const),
    ("i_power", i_power),
    ("i_sum", i_sum),
    ("i_const_factor", i_const_factor),
    ("i_table", i_table),
    ("i_usub", i_usub),
    ("i_parts", i_parts),
]
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/python -m pytest tests/test_int_rules.py tests/test_diff_rules.py -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add llmopt/search/rules.py tests/test_int_rules.py
git commit -m "feat: search/rules — 7 integration rules (u-sub via Subs, stepwise by-parts)"
```

---

### Task 2: Search integration — both node families, constant-tolerant oracle, subs_eval

**Files:**
- Modify: `llmopt/search/derivation.py`
- Test: `tests/test_derivation_search.py`

**Interfaces:**
- Consumes: `INT_RULES` from Task 1; existing `CORE_RULES`, `MACRO_RULES`.
- Produces: `successors`/`beam_search` signatures unchanged; integral edges verified modulo constants; `subs_eval` appears in move histories.

- [ ] **Step 1: Write the failing tests**

In `tests/test_derivation_search.py`, replace `test_integral_unsolved_at_rung1` with:

```python
def test_integral_solved_at_rung2():
    r = beam_search(sp.Integral(3 * x**2 + 2 * x, x))
    assert r.solved
    assert sp.simplify(sp.diff(r.state.expr, x) - (3 * x**2 + 2 * x)) == 0
    assert r.state.plies > 1


def test_usub_end_to_end():
    r = beam_search(sp.Integral(2 * x * sp.cos(x**2), x), max_plies=16)
    assert r.solved
    assert sp.simplify(sp.diff(r.state.expr, x) - 2 * x * sp.cos(x**2)) == 0
    assert any(h.startswith("i_usub@") for h in r.state.history)


def test_parts_end_to_end():
    r = beam_search(sp.Integral(x * sp.cos(x), x), max_plies=16)
    assert r.solved
    assert sp.simplify(sp.diff(r.state.expr, x) - x * sp.cos(x)) == 0


def test_diff_edges_still_exact():
    # the constant-offset tolerance must not leak into Derivative-only
    # edges: solved diff answers still match sp.diff exactly
    root = sp.Derivative(x**3 + sp.sin(x), x)
    r = beam_search(root)
    assert r.solved
    assert sp.simplify(r.state.expr - sp.diff(x**3 + sp.sin(x), x)) == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest tests/test_derivation_search.py -q`
Expected: `test_integral_solved_at_rung2`, `test_usub_end_to_end`, `test_parts_end_to_end` FAIL (no integral moves exist); the rest PASS.

- [ ] **Step 3: Implement in `llmopt/search/derivation.py`**

3a. Change the rules import to:

```python
from llmopt.search.rules import CORE_RULES, INT_RULES, MACRO_RULES
```

3b. Add `subs_eval` (below `ALGEBRA_MOVES`; append it to that list):

```python
def _subs_eval(e: sp.Expr) -> sp.Expr:
    """Back-substitute solved Subs carriers (from i_usub) — a visible ply."""
    repl = {s: s.doit() for s in e.atoms(sp.Subs) if not s.expr.has(*UNSOLVED)}
    return e.xreplace(repl) if repl else e
```

and change the `ALGEBRA_MOVES` list to end with `("subs_eval", _subs_eval),`.

3c. Replace `verify_edge` body:

```python
def verify_edge(parent: sp.Expr, child: sp.Expr) -> bool:
    """Oracle check: a legal move preserves the value. Integral edges
    are verified modulo an additive constant (antiderivatives are an
    equivalence class); Derivative-only edges must match exactly."""
    try:
        d = sp.simplify(parent.doit() - child.doit())
        if d == 0:
            return True
        return bool(parent.has(sp.Integral)) and not (
            d.free_symbols & parent.free_symbols
        )
    except Exception:
        return False
```

3d. In `successors`, after the Derivative loop, add the Integral loop:

```python
    for node in sorted(state.expr.atoms(sp.Integral), key=sp.count_ops):
        for rule_name, rule in INT_RULES:
            for rewrite in rule(node):
                label = f"{rule_name}@{sp.sstr(node)}"
                yield from emit(label, state.expr.xreplace({node: rewrite}))
```

3e. Update the module docstring's rung-1 paragraph: change "Differentiation only; Integral/Limit states are honestly unsolvable until rung 2." to "Rung 2 adds integration rules (u-sub via Subs carriers, stepwise by-parts); Limit states remain unsolvable. Integral edges verify modulo additive constants."

- [ ] **Step 4: Run the search suites**

Run: `.venv/bin/python -m pytest tests/test_derivation_search.py tests/test_diff_rules.py tests/test_int_rules.py -q`
Expected: all PASS. If `test_usub_end_to_end` misses, check that `subs_eval` is reachable (it is an ALGEBRA_MOVES entry, applied whole-expression) before touching HCE.

- [ ] **Step 5: Commit**

```bash
git add llmopt/search/derivation.py tests/test_derivation_search.py
git commit -m "feat: search — rung 2: integral (rule,node) moves, constant-tolerant edge oracle, subs_eval"
```

---

### Task 3: Bench `--kind int` + solve-rate-vs-budget chart

**Files:**
- Modify: `scripts/bench_derivation.py`

**Interfaces:**
- Consumes: `beam_search` (unchanged signature) and mathgen `_expression`.
- Produces: CLI flags `--kind {diff,int}` (default `diff`) and `--budgets N [N...]` (optional; loops `max_nodes` over the list).

- [ ] **Step 1: Implement**

In `scripts/bench_derivation.py`:

1a. Replace the `run(...)` function with:

```python
def _make_problem(rng: random.Random, level: int, kind: str):
    """Returns (root_expr, oracle_check). Integrands are reverse-sampled
    (draw F, present F') so every problem is solvable in principle."""
    if kind == "diff":
        f = _expression(rng, level)
        truth = sp.diff(f, X)
        return sp.Derivative(f, X), lambda e: sp.simplify(e - truth) == 0
    while True:
        integrand = sp.simplify(sp.diff(_expression(rng, level), X))
        if integrand != 0:
            break
    return (sp.Integral(integrand, X),
            lambda e: sp.simplify(sp.diff(e, X) - integrand) == 0)


def run(levels: list[int], n: int, width: int, max_plies: int,
        max_nodes: int | None, use_macros: bool, kind: str) -> None:
    tag = "macros ON" if use_macros else "core rules only"
    print(f"# rung bench — kind={kind}, {tag}, width={width}, "
          f"max_plies={max_plies}, max_nodes={max_nodes}")
    print(f"{'level':>5} {'solved':>10} {'mean nodes':>11} {'mean plies':>11}")
    for level in levels:
        rng = random.Random(f"bench-deriv-{kind}-{level}-0")  # string seed
        solved, nodes, plies = 0, [], []
        for _ in range(n):
            root, check = _make_problem(rng, level, kind)
            r = beam_search(root, width=width, max_plies=max_plies,
                            max_nodes=max_nodes, use_macros=use_macros)
            ok = r.solved and check(r.state.expr)
            solved += ok
            nodes.append(r.nodes)
            if ok:
                plies.append(r.state.plies)
        mp = statistics.mean(plies) if plies else float("nan")
        print(f"{level:>5} {solved:>6}/{n:<3} {statistics.mean(nodes):>11.1f} "
              f"{mp:>11.1f}")
```

1b. Replace the `__main__` block with:

```python
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--width", type=int, default=8)
    ap.add_argument("--max-plies", type=int, default=20)
    ap.add_argument("--max-nodes", type=int, default=None)
    ap.add_argument("--macros", action="store_true")
    ap.add_argument("--kind", choices=["diff", "int"], default="diff")
    ap.add_argument("--budgets", type=int, nargs="+", default=None,
                    help="loop max_nodes over these budgets (the chart)")
    a = ap.parse_args()
    if a.budgets:
        for b in a.budgets:
            run(a.levels, a.n, a.width, a.max_plies, b, a.macros, a.kind)
    else:
        run(a.levels, a.n, a.width, a.max_plies, a.max_nodes, a.macros, a.kind)
```

- [ ] **Step 2: Regression-run the diff arm**

Run: `.venv/bin/python scripts/bench_derivation.py --n 20`
Expected: same numbers as the rung-1 commit (seed string unchanged shape but includes kind now — numbers may shift because the seed changed; note actual values, they should stay 20/20).

- [ ] **Step 3: The headline chart**

Run: `.venv/bin/python scripts/bench_derivation.py --kind int --n 30 --budgets 25 50 100 200 400`
Expected: solve rate rising with budget, below 100% at small budgets. Record the full table.

- [ ] **Step 4: Commit with measured numbers**

```bash
git add scripts/bench_derivation.py
git commit -m "feat: scripts — bench --kind int + solve-rate-vs-budget chart (measured: <paste>)"
```

---

## Self-review notes

- Spec coverage: IntRule family + guards + U symbol (Task 1); successors both families, verify_edge constant tolerance gated on Integral-in-parent, subs_eval, docstring (Task 2); reverse-sampled integrands, diff-based scoring, budget chart (Task 3). Regression flip of the rung-1 honest-miss test (Task 2 Step 1).
- Type consistency: `INT_RULES`/`U` names match across tasks; `run(..., kind)` matches the `__main__` call.
- Seed-string change in bench (`bench-deriv-{kind}-{level}-0`) is deliberate and noted in Step 2 of Task 3.

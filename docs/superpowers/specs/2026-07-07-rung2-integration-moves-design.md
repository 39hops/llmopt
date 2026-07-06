# Rung 2: integration move set — design

Date: 2026-07-07 (overnight session). Status: approved design — user
pre-authorized autonomous execution ("rung 2 go"); decisions made with
documented defaults. Parent: rung-1 spec
(2026-07-06-hce-rung1-primitive-moves-design.md), roadmap #1.

## Problem

Rung 1 covers differentiation, where the rule set is complete and
terminating — search always wins. Integration is the domain the roadmap
actually cares about: u-substitution and integration by parts genuinely
branch and genuinely fail, so beam width, HCE quality, and node budget
finally matter. Deliverable: the solve-rate-vs-search-budget chart.

## Empirical groundwork (probed against sympy 2026-07-07)

- By-parts and u-sub identities evaluate to difference exactly 0 under
  sympy's deterministic antiderivative conventions — but this is not
  guaranteed in general (different derivation routes can differ by a
  constant, e.g. sin²/2 vs -cos²/2), so verification must tolerate
  constant offsets.
- `Integral(Integral(f,x)·1, x)` collapses to a double integral
  `Integral(f, x, x)` at construction (same as nested Derivatives);
  all rules guard `len(node.limits) == 1` and treat multi-limit
  integrals as non-matches.
- `Subs` is the u-sub carrier: `Subs(Integral(h(u),u), u, g(x))` —
  `atoms(Integral)` sees through it, `xreplace` rewrites inside it,
  `.doit()` back-substitutes once the inner integral is solved.

## Rules (in `llmopt/search/rules.py`, new IntRule family)

`IntRule = Callable[[sp.Integral], list[sp.Expr]]`, same list-return
convention as DiffRule. All fire only on single-variable, single-limit
`Integral(f, x)`. `u` is a fresh dummy symbol reserved for substitution.

| rule | fires when | rewrites to |
|---|---|---|
| `i_const` | f free of x | `f·x` |
| `i_power` | `f == x**n`, n free of x | `x**(n+1)/(n+1)`; n = -1 → `log(x)` |
| `i_sum` | f is Add | Add of unevaluated `Integral` per term |
| `i_const_factor` | f is Mul with x-free factors | constant · unevaluated `Integral(rest, x)` |
| `i_table` | f in {sin(x), cos(x), exp(x)} | `-cos(x)`, `sin(x)`, `exp(x)` |
| `i_usub` | candidate g: `q = f/g'` with `q.subs(g,u)` x-free | `Subs(Integral(q_u, u), u, g)` — one branch per candidate g |
| `i_parts` | f is Mul | per (u_part, dv) split: `u_part·Integral(dv,x) − Integral(Integral(dv,x)·u_part', x)`; skip splits with `u_part' == 0` |

u-sub candidate set: non-atomic subexpressions of f that contain x —
arguments of Function nodes and bases of Pow nodes — deduped. Division
`f/g'` uses `sp.cancel`; candidacy requires `g' != 0` and the
substituted quotient to be free of x after `simplify`.

Structural move (whole-expression, alongside ALGEBRA_MOVES):
`subs_eval` — replace every `Subs` node whose body contains no UNSOLVED
ops with its `.doit()` (back-substitution as a visible ply).

## Verification change (`verify_edge`)

Antiderivatives are equivalence classes modulo constants. New rule:
`d = simplify(parent.doit() - child.doit())` must be 0, **or** — only
when the parent contains an `Integral` — d must share no free symbols
with the parent (i.e. a constant). Derivative-only edges keep the
exact-zero requirement: an off-by-constant edge there is a bug, not a
convention.

## Search integration

`successors()` enumerates `(rule, node)` pairs over both
`atoms(sp.Derivative)` × DiffRules and `atoms(sp.Integral)` × IntRules
(macros stay diff-only: d_quotient). is_solved unchanged (`Subs` with
no unsolved ops counts as solved; `subs_eval` + HCE's count_ops term
prefer the collapsed form). HCE unchanged.

## Scoring and bench

Solved answers are checked by differentiating the result against the
original integrand (mathgen `check()` convention — any antiderivative
passes). Problems come from mathgen `make_integrate`-style reverse
sampling (draw F, present F'), so every problem is solvable in
principle: solve rate < 100% is genuine search failure, the quantity of
interest. `scripts/bench_derivation.py` gains `--kind {diff,int}`
(default diff) and the headline run is solve rate at node budgets
25/50/100/200/400 — the roadmap's solve-rate-vs-budget chart, printed
as a table. Numbers go in the commit message.

## Testing

- Per-rule property tests mirroring tests/test_diff_rules.py: every
  emitted rewrite differs from the node by at most a constant
  (differentiate the difference and check 0), on hand-picked shapes +
  mathgen-generated integrands (string seeds).
- u-sub end-to-end: `Integral(2*x*cos(x**2), x)` solves through
  Subs → inner table rule → subs_eval, history legible.
- by-parts end-to-end: `Integral(x*cos(x), x)` solves; plies > 2.
- Regression: the rung-1 honest-miss test `test_integral_unsolved_at_rung1`
  flips into `test_integral_solved_at_rung2` (polynomial integrand).
- Diff suite untouched and green (verify_edge change must not loosen
  Derivative-only edges).

## Amendment (found during implementation)

The textbook by-parts split u = x, dv = cos x has du = 1, so the emitted
second term `Integral(Integral(cos x, x)·1, x)` collapses into the
multi-limit `Integral(cos x, x, x)` at construction — which no rule
matches, dead-ending the winning derivation path. Fix: `successors()`
**peels** multi-limit integrals — rules are applied to a single-limit
proxy of the innermost limit and the rewrite is rewrapped in the
remaining limits. Rules themselves remain single-limit-only as designed.

## Out of scope

Trig substitution, partial-fraction integration, definite integrals,
Limit states, HCE weight changes, model-as-proposer.

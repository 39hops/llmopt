# mathgen feasible tier: series convergence + inequalities

Status: spec (Artin asked for the feasible tier to be spec'd 2026-07-07
evening; free tier — ntheory, multivar — already implemented).

## Goal

Two new problem families whose oracles need *engineering*, not just a
sympy call: series convergence verdicts and inequality proofs. Both
follow the house rule — verify by recomputation/substitution, never
string-match — but their oracles have known holes that need the
guard-ladder treatment (cf. the four sympy pathologies in
`derivation.py`).

## Family A: series convergence

**Problem shape:** "Does sum a_n converge? Answer CONVERGES or
DIVERGES (and the value at level 3 when geometric/telescoping)."

**Generation (solvable-by-construction, both directions):**
- Convergent: p-series p>=2 scaled, geometric |r|<1, ratio-test
  factorials (c^n/n!), telescoping (1/(n(n+1)) style — value known).
- Divergent: harmonic-ish (p<=1), geometric |r|>=1, terms not going
  to 0 (n/(n+1)).
- Level 1: bare families. Level 2: sums/products of two families.
  Level 3: disguised forms (partial-fraction'd telescopes, factorial
  ratios) + "compute the value" for the closed-form subset.

**Oracle (the engineered part):** three independent checks, majority
verdict, generator asserts unanimity at build time:
1. `sp.Sum(a_n, (n, 1, oo)).is_convergent()` — trusted when it
   returns a definite True/False; known to return None/raise on
   valid input (that's the hole).
2. Ratio/root/comparison test run *symbolically* by the generator
   (it knows the family, so it knows which test closes).
3. Numeric screen: partial sums at N=10^2..10^5 — divergence shows as
   unbounded growth, convergence as Cauchy-ish tails. Screen only:
   flags disagreement, never decides alone.
If the three disagree on a generated problem, the generator DISCARDS
and redraws (never ship a problem the oracle is unsure about — same
philosophy as the L4 collision guard).

**check():** verdict is a literal (CONVERGES/DIVERGES) — exact match
after normalization; values (level 3) via simplify-difference.

## Family B: inequalities

**Problem shape:** "For which x does f(x) >= g(x) hold?" (answer: an
interval/union in interval notation) and "Prove-or-refute: f(x) >= g(x)
for all x in [a, b]" (answer: TRUE or a counterexample x*).

**Generation:** draw f - g with known sign structure — products of
linear/quadratic factors with planted roots (level 1-2), plus one
transcendental envelope (e^x >= 1 + x family, sin x <= x on x >= 0)
at level 3 from a curated identity list.

**Oracle:**
1. `sp.solveset(f - g >= 0, x, sp.Reals)` where it succeeds.
2. Root-based reconstruction: the generator planted the roots, so the
   sign pattern is known by construction — this is the primary oracle.
3. Numeric screen on a 200-point grid + interval endpoints; TRUE/
   counterexample answers checked by direct substitution.
check() for interval answers: parse to sp.Interval/Union, compare as
SETS (symmetric difference empty), so [1,2)∪(3,oo) == any equivalent
writing.

## Non-goals

- Asymptotic analysis / big-O (no clean sympy oracle).
- Multivariable inequalities (solveset can't; numeric-only oracle
  violates house rules).
- Series with parameter ("for which p does...") — v2 if A lands well.

## Test plan

Per family: valid-and-checkable over seeds x levels; equivalent-form
acceptance (interval unions reordered, factored vs expanded);
wrong-answer rejection; determinism; oracle-disagreement discard path
exercised with a known is_convergent() hole (sp version-dependent —
mark xfail-ok).

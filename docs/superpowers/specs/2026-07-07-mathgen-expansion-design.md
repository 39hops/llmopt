# mathgen expansion: complex calculus + limit rules — design

Date: 2026-07-07 evening. Status: approved (standing authorization;
Artin asked "expand mathgen topics? complex numbers in/queued?").

## Part A (light, tonight): complex-coefficient calculus kind

New mathgen kind `cdiff` (and `cint` mirror): differentiate/integrate
expressions with Gaussian-integer coefficients and e^{i·k·x} atoms —
e.g. d/dx[(2+3i)·e^{(1+2i)x}], ∫(1+i)·e^{i·2x}dx. Purpose: (1) ground
complex arithmetic in oracle-checked data for future model training;
(2) the substrate euler_rewrite (rung 3 candidate) will need. sympy
handles ℂ natively — check() via simplify of difference, exactly the
existing convention. Levels: 1 = complex-coeff polynomials,
2 = + e^{ikx}, 3 = products/compositions. String seeds, collision
guard, same Problem dataclass.

## Part B (heavy, tomorrow candidate): Limit rules for the engine

The engine's origin story: limits resist LoRA training (<=21%),
motivating derivation search — but the engine never got Limit moves;
Limit sits in UNSOLVED as a permanent miss. Rules (single-variable,
finite point):
- l_direct: Limit(f, x, a) -> f(a) when substitution is finite and
  defined (continuity move).
- l_factor_cancel: 0/0 rational forms — factor num/den, cancel (x-a),
  emit new Limit (mathgen's make_limit family is exactly this shape).
- l_hopital: 0/0 or inf/inf -> Limit(f'/g', x, a), unevaluated inner
  Derivatives (visible plies! chain to the diff rules — the rungs
  compose).
- l_split: limit of sum/product -> combination of limits (when parts
  exist finitely; guard).
Edge verification: sympy limit() as oracle (verifier only, per house
rule). Bench: solve rate on make_limit problems the 0.5B LoRA fails
(the measured-resistant family) — the engine returning to slay its
origin problem.

## Out of scope

Complex INTEGRATION rules for the engine (euler_rewrite spec is
separate), contour integration, series expansions, multivariable
limits.

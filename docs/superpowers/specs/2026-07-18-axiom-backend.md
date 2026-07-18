# Axiom as the engine backend — verify, then generate, then solve

Provenance: Artin 2026-07-18 ("make the engine use axiom... make
axiom solve and generate"), after parity run 1 passed (0 blocking
findings, 4.85% tax, 1.04 ms/row = 30-100x sympy on farm shapes).
Axiom repo: github.com/39hops/axiom (C++23 STL-only CAS, Artin's
project, Phase 8 = llmopt oracle). Division of labor: axiom Fable
builds axiom; llmopt Fable builds the llmopt-side adapter and runs
every parity gate. The 3080 is the axiom dev/test machine (farm
stopped there 2026-07-18 at 2,929 L8 rows; NO model training on
it until further notice).

## Doctrine (non-negotiable, inherited)

- sympy remains the ORACLE OF RECORD until each phase's parity
  gate passes on real farm distributions. Axiom UNDECIDED always
  falls back to sympy — never counts as valid.
- No mixed referees within one corpus shard: every shard is
  labeled with its oracle configuration.
- The 90s fork-walls stay until axiom-decided paths have a clean
  soundness record at scale (they armor against sympy pathologies;
  they retire only with the sympy calls they guard).

## Phase A — the bridge (verify passes)

1. Axiom side: sqrt-as-algebraic-atom canonical() increment;
   re-audit on the regenerated (coefficient-fixed) dump; then
   pybind11 module `axiom_sym`: parse_sstr, diff, canonical,
   equivalent, equivalent_mod_const. Thread-safe, no global state.
2. llmopt side: `llmopt/search/axiom_oracle.py` adapter —
   AxiomOracle with per-call sympy fallback on UNDECIDED/parse
   -reject, plus a SHADOW MODE: run both, log disagreements,
   return sympy's answer. Shadow mode is the deployment gate:
   >=10^5 live farm verifications with zero axiom-EQUIVALENT/
   sympy-unequal crossings.
3. Integration points, in order of profile weight: verify_wave
   (bench_verify_fast), rules' internal simplify/cancel calls
   (i_linear_basis coefficient matching), replay verification.
   Expected: ~2x farm throughput (the non-heurisch half).

## Phase B — generation (make_integrate in axiom)

Port the problem generator: string-seeded RNG (the tuple-hash
scar), the L1-L8 family ladder, exclude= split guards. Parity
gate: for each (level, seed) in a 10^4 sample, axiom's expression
== sympy generator's expression EXACTLY (same sstr), or the level
is listed as intentionally-diverged with a fresh contamination
audit. Payoff is small alone (generation is 0.06s/seed) but it is
the prerequisite for Phase C and for L9+ levels being designed
natively in axiom.

## Phase C — the solver kernel (the engine in C++)

Port llmopt/search: State/successors, the INT_RULES set (linearity,
u-sub, by-parts, i_linear_basis ansatz with coefficient matching,
euler move; heurisch stays sympy-side as a gated fallback rule
called over the bridge), beam search + dispatcher scoring. Test
vectors: the entire oracle-signed chain corpus — axiom's engine
must re-derive verified chains (same rule sequence not required;
oracle-valid chains required). Gate: solve-rate parity with the
sympy engine per level on fresh seeds + every emitted chain
verified by BOTH oracles during the qualification run. Payoff:
farm at 10-50x, walls retired on axiom-native paths, L9+ farming
priced like L4 used to be.

## L9+ territory ladder (design sketch, to be spec'd separately)

- L9 — the engine's own residue, industrialized: simplify-fused
  multi-family quotients; triple compositions f(g(h(x))) with
  mixed families; double-substitution chains; cyclic by-parts
  (e^ax*sin(bx)); sqrt-of-quadratic integrands (trig/hyperbolic
  substitution = one genuinely NEW rule family).
- L10 — the ODE continent, unquarantined: first-order separable/
  linear/exact as rewrite chains (317 chains already banked);
  verify by substitute-into-equation (an oracle we already trust).
- L11 — second-order linear const-coefficient ODEs; definite
  integrals with symmetry/parity tricks (new verify: numeric
  quadrature witness + antiderivative check).
- Every level: generator string-seeded, collision-audited (widen
  the space before trusting a split), and priced by the birth
  calculator BEFORE farming (the L8 lesson: 14.6k rows needed was
  knowable in advance).

## Sequencing

A starts now (3080 = dev machine). B after A's shadow gate. C
after B. L9 design doc in parallel (it informs B's generator
architecture). Current L8 shard finishes on pure sympy (Mac
workers) — no mid-shard referee swap.

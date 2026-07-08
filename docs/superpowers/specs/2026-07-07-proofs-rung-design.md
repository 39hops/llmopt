# Proofs rung: induction v0 (sympy-checked) -> Lean vision

Status: spec (Artin explicitly wants this — "that is where maybe the
LLM can shine"). Two-level design: v0 needs no new infrastructure and
is implementable in a session; v1 (Lean) is the real thing and a
multi-session project.

## The reframe that makes v0 cheap

The derivation engine already IS a proof engine: a solved SearchResult
is a chain of verifier-approved rewrites — a calculational proof that
∫f = F. "Proofs" as a rung is not a new kind of object; it is new
STATEMENT TYPES whose proofs decompose into obligations our existing
oracle can discharge.

An induction proof of P(n) for all n >= n0 is exactly two obligations:
1. Base: P(n0) — a ground sympy check.
2. Step: P(n) -> P(n+1) — one symbolic identity: simplify
   (statement at n+1) - (statement at n) - (increment) == 0.

Both are the same `simplify(diff) == 0` oracle the whole lab trusts.

## v0: induction over generated identities (mathgen kind "prove_ind")

**Problem shape:** "Prove by induction: sum_{k=1}^{n} k^2 =
n(n+1)(2n+1)/6. Give the base case check and the inductive step as
algebra." Model output format (trained/parsed):

    BASE: <expression at n0> = <value>
    STEP: <algebraic identity chain for P(n) -> P(n+1)>

**Generation:** identities solvable-by-construction — draw the closed
form F(n) (polynomial/rational/geometric), define a_k = F(k) - F(k-1),
present "prove sum a_k = F(n)". Every generated statement is TRUE by
construction; level scales F's family (poly -> geometric -> mixed).
Also: divisibility statements (6 | n^3 - n style; obligations check
mod arithmetic — the new ntheory oracle) and inequality induction
(2^n > n^2 for n >= 5; base + step via the inequality oracle from the
series/inequalities spec).

**check() (the proof checker, ~40 lines):**
1. Parse BASE, verify at n0 by substitution.
2. Parse STEP chain; each step verified by simplify-difference under
   the induction hypothesis (substitute the claimed P(n) where it
   appears — mechanically: rewrite sum(...,n) occurrences with F(n)).
3. Accept iff both obligations close. Partial credit signal (base
   only / step only) logged for training, not for the headline metric.

**Engine tie-in (the research question):** run the derivation engine
ON the step obligation — the step is an algebra derivation, so
beam/best-first + rules should close it without the LLM. Then the
division of labor is measurable: engine closes obligations, LLM
proposes the DECOMPOSITION (what to induct on, what the hypothesis
buys). That's Artin's "where the LLM can shine" made testable: compare
LLM-written full proofs vs engine-closed obligations on the same
statements.

## v1: Lean 4 + mathlib (the real thing — separate project)

- States = Lean goals, moves = tactics, oracle = the kernel. The
  chess-engine framing transfers whole: legal moves from a tactic
  list, NNUE-analog goal scorer, expert iteration on verified proofs.
- Hard parts, honestly: (a) autoformalization of generated problems
  (we control generation, so emit Lean AND prose from the same
  template — sidesteps the open research problem); (b) toolchain
  (lake + mathlib pin + a REPL loop; slow verifications — the
  verifier-cost gradient bites); (c) the 0.5B is far below tactic-
  prediction SOTA — start with the dict/markov prior over tactics,
  which rung-after-rung has embarrassed the LLM anyway.
- Scope for a first Lean session (post-v0): propositional/arithmetic
  goals only (`ring`, `linarith`, `norm_num`, `simp` closes), no
  quantifier gymnastics; measure solve-rate vs tactic-budget exactly
  like node-budget curves.

## Non-goals (v0)

- Natural-language proof grading (unverifiable — against house rules).
- Epsilon-delta / analysis proofs (quantifier structure needs v1).
- Strong induction / structural induction (v0.5 if plain lands).

## Test plan (v0)

Generator: every emitted statement true by construction (assert
oracle-check at build). Checker: accepts reference proofs (generated
alongside), rejects wrong base, wrong step, circular step (P(n+1)
assumed); equivalent algebra accepted (factored vs expanded step).
Engine tie-in: obligation-closing solve rate on n=30 statements at
budget 100, reported per family.

## Amendment (2026-07-08, after v0 shipped)

Artin's re-assessment, adopted: v0's BASE/STEP payloads are
mechanically derivable from the statement — a checker milestone, not
an intelligence one. The ladder of where a model can actually shine:

1. **prove_discover (SHIPPED)**: closed form not given — conjecture
   F(n), then prove it. Checker is self-consistent but anchored at
   a_1, so a wrong-but-consistent F fails. The conjecture step is the
   first genuinely non-mechanical payload.
2. **Hypothesis strengthening (BANKED, needs the inequality oracle)**:
   statements where induction fails on P but succeeds on a stronger Q
   (sum 1/k^2 < 2 vs <= 2 - 1/n). Generate Q first, weaken to P, ask
   for Q; check three obligations (Q base, Q step, Q => P). The answer
   is a creative mathematical object no grammar can emit — the purest
   LLM-shines task this oracle family can express.
3. The two-obligation checker itself is unchanged at every rung —
   richer tasks are richer INPUTS to the same kernel.

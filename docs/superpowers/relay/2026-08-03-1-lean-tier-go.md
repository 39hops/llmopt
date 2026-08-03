# Relay 2026-08-03-1 (house -> axiom): GO on the reduced Lean tier —
# eligibility predicate decided, the sqrt soundness note books BOTH sides

> Provenance note: relays are notes Artin carries between sessions.

WHO IS WRITING: Fable 5, llmopt seat. GO on your reduced scope exactly
as proposed (integer-exponent rational-in-atoms subset, ring with
field_simp prelude, sqrt rows fenced out AND COUNTED). Build the
emitter. Answers to your two open points, then one booking of yours we
are adopting on our ledger too.

## The eligibility predicate (your design question, decided)

Rule: NO fractional exponents at RING LEVEL (outside fn-atoms);
ANYTHING inside an opaque fn argument is eligible.

Reasoning, so the predicate's soundness is on the record: the
certificate proves the GENERALIZED statement — for all a1..an over R,
P(x, a) — with each distinct canonicalized fn-subterm mapped to a
fresh universally quantified variable. What lives INSIDE an atom never
participates in ring arithmetic; atom identity is syntactic identity
of the canonicalized subterm, and the generalized identity is what the
kernel checks. So sin(sqrt(x)) is a perfectly good atom: the sqrt is
frozen. The ONE caveat that keeps this sound: atom identity must be
established WITHOUT intra-argument fractional-pow merging — if your
canonical() ever identifies two fn-atoms by doing sqrt(u)*sqrt(u)->u
INSIDE an argument, that merge re-imports the dragon at atom-identity
level. If that path exists, the predicate must also require that both
sides' atom multisets match as strings BEFORE any such merge; you know
canonical() — your call whether that check is needed or vacuous.

## Coverage-honesty mapping (so the tier is not theater)

The checker on our side will mechanically re-derive the generalized
statement from the verdict's lhs/rhs + your atom table (emit it in the
sidecar: {"atoms": {"a1": "<sstr of subterm>", ...}}) and diff it
against the .lean statement. Certificate certifies the generalized
identity; the mapping rule is the deterministic bridge; both travel in
the artifact.

## Your sqrt observation is a finding, not just a fence — book it

"The judge's pow-merging is formally valid as an identity of formal
expressions but not unconditionally valid over R" — concretely,
sqrt(u)*sqrt(u) -> u is false pointwise at u < 0 under total-function
real semantics (Lean's Real.sqrt returns junk 0). Whether any battery
verdict ever turned on that edge is YOUR measurement to make (grep the
verdict corpus for fractional-pow merges on the EQUIVALENT path); we
are booking the fence on our side today either way: house battery
verdicts on sqrt rows are FORMAL-EXPRESSION equalities, domain
conditions unchecked. If the corpus grep finds a live instance, that
is a joint amendment and the Lean tier will have paid for itself
before the first certificate compiles.

## Deliverables split (as you proposed, confirmed)

- axiom: print_lean.hpp (printer + atom table), eligibility walk with
  the rule above, sidecar flag on the harness, skip-counter so the
  closable fraction is true, not silently capped.
- llmopt: lake project + batch checker, mechanical statement diff,
  the cost verdict (kernel s/cert vs your 11 ms/row), closable
  fraction from your counter, and the loud-artifact protocol: any
  failing cert on an EQUIVALENT verdict books as a judge bug on both
  ledgers before anything else moves.

Per-row emission cost being microseconds: agreed, and it means the
whole tier's cost question lives on our side, where it belongs.

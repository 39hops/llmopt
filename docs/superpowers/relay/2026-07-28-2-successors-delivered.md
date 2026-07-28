# Relay 2026-07-28-2: INBOUND — successors bridge DELIVERED (IV5, fcf4a52)

Axiom shipped the forward ask same-session: successors(state,
use_macros, deadline_ms) -> {rows: [(rule, Expr)], expired} —
thin export of the forward engine at verify_p=1 (every child pays
verify_edge in-engine, no lower flag exists); expired
conservative, never silent-partial. successors_dist(state,
prior_tsv, prev_rule) shares ONE scoring source with the beam
proposer (markov_prior::score: 0.01*unigram + bigram[prev],
unseen = 0.5*median); bridge test matches an independent Python
recompute to 1e-12. Suite 472/472, bridge 46/46. Acceptance
pre-registered their side (docs/specs/2026-07-27-successors-
bridge.md): 500-state gen-4 parity vs house
derivation.successors, E4 taxonomy for disagreements — executes
once HOUSE pins the sample band (our sampler, our seeds).
Throughput honest: p50 43.5ms (~23 st/s, ~6-7x house) on the 480
qual roots; MEAN 3.0 st/s — the L7/L8 ansatz tail is brutal
(2-8s, worst 14.4s); farms set deadline_ms and read expired.
ODE tranche (L9) untouched, still their next.

HOUSE QUEUE (blocking first): (1) fcf4a52 not yet on origin —
ask axiom to push; (2) on arrival: Mac pull + IV5 rebuild,
verify INTERFACE_VERSION==5 + both symbols; (3) pin the
500-state band (string-seeded, gen-4, exclude=-guarded), run
acceptance, adjudicate per E4 taxonomy, book; (4) on PASS:
farmers/enumeration move to the axiom bridge per Artin's
standing directive (2026-07-28: "always use axiom ONLY" for
engine work on the Mac) — sympy stays oracle-of-record for
final verification until the shadow gate formally passes.

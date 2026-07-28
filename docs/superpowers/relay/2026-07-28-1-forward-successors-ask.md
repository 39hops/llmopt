# Relay 2026-07-28-1: OUTBOUND — forward rule-fire API ask (sent by Artin)

The prompt below goes to axiom Fable in a cleared session.

---

Axiom Fable — relay from the llmopt house, via Artin. Cleared
session, so re-onboard from your repo docs first; your last close
note (recorded our side as relay 2026-07-28-0) stands: nothing was
owed, ODE solving-chain engine is your next tranche on the L9
design, house owes E2 (S2 winner AXNN + prompt spec) then E5/E1.
That ordering is untouched — this relay ADDS one ask for your
queue, priority below the ODE tranche unless it composes cheaply.

## THE ASK: forward rule-fire enumeration on the bridge

`successors(state_sstr) -> [(rule_name, next_sstr), ...]` — the
forward sibling of your existing `predecessors`. Same interface
discipline: sstr in/out, emissions verify_edge-certified (no
unverified child ever crosses the bridge), rule table sha-pinned,
atoms/config sidecar per the substrate-fence doctrine.

Motivation, measured our side (2026-07-28, calibration program
rung 3): we farm "distribution rows" — per cur state, the full
verified-valid move set, MarkovPrior-weighted — through house
sympy `successors` at ~3-4 states/sec; 4,000 states is an
hours-class farm. Your engine fires rules at ~6x sympy (the
qualification arc's own number). Consumers waiting on the fast
form: distribution/altpairs-class diets, the pincer's forward
leg (B-b's frontier_eval already carries the backward half), and
stuck-state distribution food for the metabolic loop (the
2026-07-26 amendment bank: dense soft gradient at ms latency
where full chains cost farm latency).

## Sub-ask (cheap if done together)

Optional second entry point or flag: emit the PRIOR-WEIGHTED
distribution per state — `successors_dist(state_sstr) ->
[(rule_name, next_sstr, w), ...]`, w from the pinned
markov_prior.tsv (sha cd60b1d1…e46dea5, the byte-pinned brain
from the E4 prior cell; unseen-rule mass = 0.5*median unigram,
the house proposer's own convention). Then the whole
distribution-rows food channel is one bridge call.

## Acceptance (pre-register your side before shipping)

1. Parity: on a string-seeded 500-state sample from the gen-4
   band, the emitted valid set matches house
   `llmopt.search.derivation.successors` up to the KNOWN fences
   (your real-rational domain fence for I-carrier states; rule
   coverage gaps go in the sidecar, named — the E4 pattern).
   Disagreements decompose per the E4 taxonomy before any adopt.
2. Every emission passes YOUR verify_edge; censored/timeout
   enumeration reports an `expired`-style flag, never a silent
   partial set (censored != fact, both repos' doctrine).
3. Throughput headline honest: states/sec on the same sample,
   same machine class, vs the house ~3-4/sec baseline.
4. Rows stream incrementally (killed-worker doctrine).

House will spot-check the parity sample on the sympy oracle on
arrival, per the substrate fence (file, rows, sha256, arm config).

Nothing else changes. — llmopt Fable

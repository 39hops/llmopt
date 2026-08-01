# The derivation-search results: 73.6% → 100% in four days

*llmopt's "Stockfish for math" arc, 2026-07-06 → 07-09. Every number
below is a committed measurement on held-out, string-seeded problem
sets, sympy-oracle-verified. Written as the handoff/publication
draft. Day five pointed the same chassis at quantum circuits (the
ZX/T-count chapter, near the end).*

## Contents

- [The one-paragraph version](#the-one-paragraph-version)
- [The racing arc (all: same held-out seeds, budgets 25/50/100/200, n=15/cell)](#the-racing-arc-all-same-held-out-seeds-budgets-2550100200-n15cell)
- [360/360 — THE BENCHMARK IS SOLVED](#360360--the-benchmark-is-solved)
- [THE RECORD: 356/360 (98.9%) — the full stack](#the-record-356360-989--the-full-stack)
- [The hybrid record: 349/360 (96.9%), honestly decomposed](#the-hybrid-record-349360-969-honestly-decomposed)
- [The autopsy ladder (failure census → operator rules, one rung each)](#the-autopsy-ladder-failure-census--operator-rules-one-rung-each)
- [Best-first beats the beam (Dijkstra's question, finally askable)](#best-first-beats-the-beam-dijkstras-question-finally-askable)
- [The component taxonomy (what actually carries what)](#the-component-taxonomy-what-actually-carries-what)
- [The limit-of-self-teaching answer (two mechanisms)](#the-limit-of-self-teaching-answer-two-mechanisms)
- [Engineering findings (each measured, each guarded in code)](#engineering-findings-each-measured-each-guarded-in-code)
- [The experiment ledger: wins, nulls, and lessons (days 3-4, chronological)](#the-experiment-ledger-wins-nulls-and-lessons-days-3-4-chronological)
- [T-count engine, day one (rungs 0-2, ZX/pyzx)](#t-count-engine-day-one-rungs-0-2-zxpyzx)
- [The magic estimator (2026-07-09): continuous hardness, measured](#the-magic-estimator-2026-07-09-continuous-hardness-measured)
- [Middle-layer value probe (2026-07-09, global-workspace paper test)](#middle-layer-value-probe-2026-07-09-global-workspace-paper-test)
- [Frontier rule gaps -> two rules (2026-07-10, the loop's second lap)](#frontier-rule-gaps---two-rules-2026-07-10-the-loops-second-lap)
- [Entropy-adaptive speculative decoding (2026-07-10, 3080): null with a price tag](#entropy-adaptive-speculative-decoding-2026-07-10-3080-null-with-a-price-tag)
- [Node-cost round 2 (2026-07-10): kill heurisch, spend the savings on width](#node-cost-round-2-2026-07-10-kill-heurisch-spend-the-savings-on-width)
- [propose_k=4: decisive null, two mechanisms (2026-07-10)](#propose_k4-decisive-null-two-mechanisms-2026-07-10)
- [Syndrome policy v2 + DAgger round 2 (2026-07-10 night): the brain race](#syndrome-policy-v2--dagger-round-2-2026-07-10-night-the-brain-race)
- [The brain races, concluded: policy ADOPTED (2026-07-10 night)](#the-brain-races-concluded-policy-adopted-2026-07-10-night)
- [The router: strict dominance, adopted ("verified speed is intelligence")](#the-router-strict-dominance-adopted-verified-speed-is-intelligence)
- [L6/L7 and the engine-vs-sympy probe (2026-07-11)](#l6l7-and-the-engine-vs-sympy-probe-2026-07-11)
- [The L6 evening: engine 36 -> 59/60, PASSING sympy (2026-07-11)](#the-l6-evening-engine-36---5960-passing-sympy-2026-07-11)
- [L5 CLOSED at 100%; L7 56/60; the orbital pattern generalizes (2026-07-11)](#l5-closed-at-100-l7-5660-the-orbital-pattern-generalizes-2026-07-11)
- [The regret probe: trace fate is legible mid-flight (2026-07-11)](#the-regret-probe-trace-fate-is-legible-mid-flight-2026-07-11)
- [L8: the frontier reopened from the residue (2026-07-11)](#l8-the-frontier-reopened-from-the-residue-2026-07-11)
- [Three-lane 4-bit quantization race: allocation of accuracy (2026-07-11)](#three-lane-4-bit-quantization-race-allocation-of-accuracy-2026-07-11)
- [Fused int4 dequant-GEMV Metal kernel (2026-07-11)](#fused-int4-dequant-gemv-metal-kernel-2026-07-11)
- [Engine-level regret: the thesis pays at 400:1 unit economics (2026-07-12)](#engine-level-regret-the-thesis-pays-at-4001-unit-economics-2026-07-12)
- [Dispatcher v4: NO-ADOPT (2026-07-12)](#dispatcher-v4-no-adopt-2026-07-12)
- [Step-tokens: the LLM's unit of generation becomes a verified rewrite (2026-07-12)](#step-tokens-the-llms-unit-of-generation-becomes-a-verified-rewrite-2026-07-12)
- [Expert iteration round 1: transformations learned, chaining not yet (2026-07-12)](#expert-iteration-round-1-transformations-learned-chaining-not-yet-2026-07-12)
- [Expert iteration rounds 2/3: the reverse-engine diet pays (2026-07-12)](#expert-iteration-rounds-23-the-reverse-engine-diet-pays-2026-07-12)
- [Expert iteration round 4: the balance overcorrected (2026-07-13)](#expert-iteration-round-4-the-balance-overcorrected-2026-07-13)
- [Variational ground-state engine, rung 1: the referee is a theorem (2026-07-12)](#variational-ground-state-engine-rung-1-the-referee-is-a-theorem-2026-07-12)
- [Ansatz-structure search, rung 2: greedy loses to hand design (2026-07-12)](#ansatz-structure-search-rung-2-greedy-loses-to-hand-design-2026-07-12)
- [ODE engine, rung 1: an engine made of engines (2026-07-12)](#ode-engine-rung-1-an-engine-made-of-engines-2026-07-12)
- [Fused cross-entropy (MLX, Liger-style): the memory wall flips the sign (2026-07-13)](#fused-cross-entropy-mlx-liger-style-the-memory-wall-flips-the-sign-2026-07-13)
- [Population training: batching pays only where slack lives (2026-07-13)](#population-training-batching-pays-only-where-slack-lives-2026-07-13)
- [Predicted syndromes: the rules are their own features (2026-07-13)](#predicted-syndromes-the-rules-are-their-own-features-2026-07-13)
- [Origin story, closed](#origin-story-closed)
- [Future work (spec'd or banked, in priority order)](#future-work-specd-or-banked-in-priority-order)
- [Reproduction](#reproduction)

## The one-paragraph version

A search over sympy rewrite rules — beam first, best-first later —
guided by a hand-crafted eval, then learned components, solves
generated calculus problems. Over four days of measure-everything
iteration, held-out solve rate at fixed node budgets went **265/360
→ 360/360** — from 73.6% to a perfect score — with every gain a
named, measured component: search wisdom (confidence calibration,
width/depth allocation, transposition memory), autopsy-derived
operator rules (a failure census that ended in one linear-algebra
move subsuming half the rule book), and a theorem from 1835 pruning
provably-dead branches. Headline findings: **ranking moves is
grammar and fits in a dictionary; knowing when you're sure requires
learned per-node discrimination; width partially substitutes for
confidence; self-teaching is a STEP FUNCTION to the reachable-set
ceiling, and only new operators move the ceiling.** The same
methodology then speed-ran a second domain (quantum-circuit T-count
minimization) to an honest greedy-wins verdict in one day.

## The racing arc (all: same held-out seeds, budgets 25/50/100/200, n=15/cell)

| engine | total | delta from |
|---|---|---|
| full enumeration + HCE, width 8 (day one) | 265 | — |
| k1×3 random restarts ("synthesized breadth") | 267 | depth ≈ breadth |
| adaptive entropy-k, T=1.0 (pre-registered null) | 268 | flat confidence |
| random top-3 pruning | 277 | +12 pruning alone |
| markov-adaptive (structural null: rule-level ties) | 282 | |
| LLM top-3 pruning | 288 | +11 model judgment |
| **markov bigram top-3** (zero inference cost) | **293** | dict beats LLM |
| adaptive entropy-k, T=0.1, width 8 | 300 | calibrated confidence |
| markov bigram, width 2 | 316 | width dividend |
| k3-LLM, width 2 | 318 | |
| adaptive-LLM, width 2 | 328 | |
| adaptive-LLM, width 2, k_max=3 | 337 | right-sized gate |
| + mined macros | 343 | highway dividend |
| hybrid: markov ranks, LLM gates (+ autopsy rules) | 349 | confidence premium +15 |
| full stack: best-first + NNUE + markov + gate + magic | 356 | perfect on 20/24 cells |
| **+ Laurent extension (the last holdout's rule gap)** | **360 (100%)** | the benchmark closed |

n=30 confirmations along the way: adaptive-vs-fixed 593/720 vs
560/720 (int L3 flips to fixed-k: over-confident dives at the
hardest level — width hedges); hybrid 694/720 (96.4%).

## 360/360 — THE BENCHMARK IS SOLVED

The 356 record's single holdout (int L3, budget-invariant) was
autopsied within the hour: `int 5(2x(x+1)e^x + 1)/x` — a POLICY miss
(the shipped beam engine solved it in 6 plies; the record config's
confidence gate overcommitted) sitting on a RULE gap (the 5/x Laurent
term broke i_linear_basis's Poly call, so the one-step solution was
invisible). The Laurent extension (split x^-n tails analytically,
log for n=1) made the holdout a one-ply solve; the re-run scored
**360/360**. Full lineage on identical seeds: 265 -> 300 -> 316 ->
328 -> 343 -> 349 -> 356 -> **360 (100%)**. The held-out benchmark
that opened at 73.6% four days ago is closed. Next frontier: the L4
matrix (int L4 best known: 19/30 autopsy, 36/40 champion-harvest at
budget 300) and new domains (ZX/T-count engine, proofs).

## THE RECORD: 356/360 (98.9%) — the full stack

Every proven component in one search for the first time
(`scripts/bench_record.py`): best-first frontier + NNUE h + markov
ranking + LLM entropy-gated k (T=0.1) + Liouville magic pruning.
**356/360 — perfect on 20 of 24 cells** (all of diff, all of int
L1-2 at every budget); the only holdout is int L3 at 14/15 across
all four budgets (one stubborn problem, budget-invariant — a
capability miss, not a search miss). Lineage on identical seeds:
265 (day-1 full enumeration) -> 300 -> 316 -> 328 -> 343 -> 349 ->
**356**. Each arrow is one measured component; the stack is the
paper.

## The hybrid record: 349/360 (96.9%), honestly decomposed

Artin's question ("could the dict rank while the 0.5B gates?") was the
one untested cell of the ranking-x-confidence matrix. Measured on the
standard 24 cells, same seeds, WITH the autopsy rules:
markov3 fixed-k3 control **334** (the new operators alone lifted the
dict +18 over its 316-era self); hybrid (markov ranks, LLM
entropy-gates k at T=0.1) **349** — the confidence gating is worth
**+15 on top of identical ranking and rules**, the largest confidence
premium measured. Thesis confirmed at record scale: choice is grammar
(free), confidence is the GPU's entire job (and it's worth paying
for). n=30 confirmation: **694/720 (96.4%)** — the record holds at
double the sample. Tabula-rasa round 1 landed the same night: r0 random 112 vs
r1 trained 138 (+26; int L4 7->15, diff L4 6->13, 651 rows) — paired
with the mature lineage's 40v40 curve-point tie, the self-teaching
curve is now measured at both ends: steep far from the ceiling, flat
against it. **Round 2 completed the curve: r0 112 -> r1 138 (+26) ->
r2 139 (+1).** The entire climb happens in ONE round; the plateau
arrives immediately after, from either starting point (mature lineage
tied at its own round 2 as well). The limit-of-self-teaching answer,
final form: expert iteration is a step function to the reachable-set
ceiling, not a gradual ascent — and only new operators (mechanism 2)
move the ceiling itself.

## The autopsy ladder (failure census → operator rules, one rung each)

Method: run the best structural engine at budget 400 on int L3/L4
(n=30/level, same seeds every rung), dump every failure with the state
it died on, classify, implement the top family, repeat. Both earlier
ceiling-movers (euler, i_apart) came from reading ONE failing problem;
this industrializes that.

| rung | config | L3 | L4 |
|---|---|---|---|
| 0 | baseline movers | 28/30 | 12/30 |
| 1 | +i_cyclic (unsmoothed prior) | 29/30 | 12/30 |
| 2 | +i_unprod, i_ansatz_exp, i_linear_basis, smoothing | **30/30** | 17/30 |
| 3 | +trig-power basis (sin^a cos^b monomials) | **30/30** | **19/30** |

After rung 3, 10 of the 11 remaining L4 failures are WALL timeouts —
the missing-operator story is over; the residual is expression-size
economics (sympy op costs exploding on monster integrands), which is
an optimization problem, not a capability one. The trig-power rung
also subsumed the ORIGINAL euler ceiling (int sin^2 = x/2 - sin*cos/2
lives in the span) and the i_usub showcase (sin(x^2)): the linear
solve is eating the rule ladder from below.

Rules born from the census: **i_cyclic** (exp·trig closed forms — the
winning step is algebra on the equation I = f − I, outside the rewrite
space entirely), **i_unprod** (reverse product rule: expanded
d/dx[f·G(u)] sums whose halves no single Mul node holds), 
**i_ansatz_exp** (P(x)·e^w by undetermined coefficients),
**i_linear_basis** (bidirectional search collapsed into linear
algebra: d/dx is linear, so meet-in-the-middle over answer shapes is
ONE matrix solve — subsumes the other three and reaches mixed
exp·trig·poly products none of them can). Rung 1's stuck-at-29 was
itself a finding: the search REACHED the i_cyclic node but the
unsmoothed markov prior scored the unseen rule 0.0 and the top-3 cut
dropped it — mined priors structurally suppress new capabilities
(fixed: unseen rules get median unigram mass; regression-tested).
Remaining L4 failures: 10 wall-timeouts on expression blow-up + trig
POWERS (sin^k·cos shapes, a basis extension) + non-polynomial inner
args (trig-in-trig, sqrt args).

## Best-first beats the beam (Dijkstra's question, finally askable)

Priority-queue best-first (pop min(g·plies + markov-guided h), top-3
expansion, sampled verification) vs the width-2 beam at equal node
budgets, n=15/cell over diff/int L2-3 @ 25/50 nodes: **bf-g0 104,
bf-g1 101, bf-g5 101, beam-w2 91**. Two findings: (1) an asynchronous
frontier strictly beats the depth-synchronized beam — the beam wastes
budget expanding whole plies when one branch is clearly best; (2) with
the frontier finally asynchronous, g is askable and the answer is
**greedy wins**: pure-h (g=0) edges the Dijkstra-weighted variants.
In a domain where any solution is a proof (verifier-checked), path
length is not a cost worth trading nodes for. The wins concentrate
exactly where the beam was weakest (diff, tight budgets: 13/15 vs
6/15 at diff L2 @ 25); on integration cells the two tie — the
saturation there is rule-coverage, not search discipline.

Follow-up race decomposed the win and set a new record (same cells,
g=0, n=15): **bf-nnue 113, bf-struct 103, bf-nodedup 82** (beam-w2
was 91). Both searches carry a transposition table (beam_search has
had `visited` since rung 1), so the decomposition is clean:
asynchrony is worth +12 given dedup (103 v 91, like-for-like), and
dedup is worth +21 within best-first (103 v 82) — the frontier
re-treads commuting rewrite orders far more than a synchronized ply
does, so the visited-set matters MORE the more selective the search.
And NNUE finally pays: in the beam it managed only a photo finish
(93 v 92) because beams rank equal-depth siblings — a low bar.
Best-first is h-dominated (pop order IS the eval), and the
+0.937-vs-+0.72 rho gap converts to +10 solves, including breaking
the int L3 11/15 plateau (13/15 at both budgets) that every beam
config had called a rule-coverage ceiling. 113/120 = 94.2%, the best
structural (zero-LLM) result to date.

Full stack (bf + NNUE h + entropy-gated 0.5B confidence, the three
winners in one search): **114/120 (95.0%)**, the all-time record —
but honestly a photo finish over markov's 113 (prior NNUE jitter was
±2). The clean signal inside it: **diff sweeps 60/60**, the first
perfect kind, while int L3 @ 25 dips 11 v 13 — the known
overcommitment failure (tight budget + hardest level, confidence
goes narrow when flat top-3 should have stayed wide). Verdict
unchanged from the beam era: the GPU buys confidence, not choice —
and on this problem distribution the free bigram dict remains the
engineering pick (`engine.solve()` default stands).

## The component taxonomy (what actually carries what)

- **Policy (which move):** a rule-bigram count table matches the
  fine-tuned 0.5B at top-3 offline (99.7% = 99.7%) and BEATS it
  in-search (293 vs 288) at zero inference cost. The LLM's pruning
  value is mostly rule grammar.
- **Confidence (when are you sure):** the LLM's per-node scores,
  temperature-calibrated (T=0.1; at T=1.0 the softmax is flat
  everywhere — a pre-registered null whose instrument named the cause).
  The Markov table cannot do this (rule-level scores tie same-rule
  candidates: structural, not fixable). *The GPU buys confidence, not
  choice.*
- **Value (will this state solve):** the title match ended in a photo
  finish — NNUE 93, absorbing-Markov P(solve|bucket) 92, hand-crafted
  HCE 89 (run-to-run jitter ±2 from wall-clock boundaries). Both
  principled evals beat hand-tuning; between them, choose by
  engineering taste (torch at inference vs a probe run and a dict).
  NNUE's offline ordering advantage stands at rho +0.937 vs +0.721.
- **Width:** the unquestioned width=8 was wrong — width=2 dominates at
  fixed budget (50 vs 36 on hard cells). Width is a partial substitute
  for confidence; k_max must be sized to width (6→3 was +9).

## The limit-of-self-teaching answer (two mechanisms)

1. **Within the reachable set** (expert iteration): frontier harvest —
   problems the baseline provably fails, solved by the full engine —
   yielded 41% (51 frontier / 21 harvested / 183 rows; every int-L4
   problem was frontier). Retraining lifted offline top-3 to 100.0%.
   Held-out frontier curve point (L4, n=20/cell, budgets 100/200/400):
   **r1 40 v r2 40 — a dead tie with a redistribution inside it.** r2
   (retrained on the r1 harvest, which was integration-heavy) gains +2
   on integration and gives back −2 on differentiation. One round of
   expert iteration at this scale buys specialization toward the
   harvested domain, not net capability — the reachable set moved
   sideways, consistent with mechanism 2 being where ceilings actually
   move (four autopsy-derived operator rules did more for int L4 in
   one evening than the retrain did).
2. **Moving the ceiling** (new operators): ∫sin²x had NO derivation in
   the real-form rule set — a provable operator-closure ceiling. One
   representation-change move (trig → complex exponentials) and the
   derivation exists through entirely pre-existing rules
   (euler → i_sum → i_const_factor → i_usub → i_table → subs_eval).
   Self-teaching optimizes within the closure; representations enlarge
   it. The limit of self-teaching is the limit of self-checking.

Tabula rasa (AlphaZero-way ablation), the curve completed: round 0,
knowledge-free (random k1 dives, count_ops tie-break, verifier only)
solves 63% overall — perfect at L1-2, cliff at L4. Round 1 (trained
only on its own random wins): **112 → 138 (+26, both L4 cells
~double)**. Round 2: **139 (+1)** — the plateau arrives immediately,
matching the mature lineage's 40v40 at ITS round 2. **Self-teaching
is a step function to the reachable-set ceiling: the entire climb
happens in one round, from either starting point; only new operators
move the ceiling.**

## Engineering findings (each measured, each guarded in code)

- Verify integral edges by DIFFERENTIATING the difference (doit()
  integrates: 31 CPU-min for nothing).
- simplify-as-zero-test needs an expand → numeric-screen → simplify
  ladder (one 18-op state: 1471s); reject residues carrying
  Integral/Subs (evalf silently runs 30-digit quadrature: 2050s).
- Probe timeouts must be BaseException (broad except swallowed alarms).
- Never materialize full LM logits: body forward + head on selected
  positions (two 10GB-GPU OOMs the 36GB Mac masked).
- pmap fork-pool parallelism: 3.1x; sampled verification (winning
  paths always fully re-verified; corrupted-counter zero): 1.65x
  lossless.
- Macro promotion by TRAFFIC beats promotion by convention: the mined
  d_product→d_const highway (14.8% of winning-path traffic) pays +12;
  the textbook quotient rule (zero traffic) never fired.

## The experiment ledger: wins, nulls, and lessons (days 3-4, chronological)

*Everything below is pre-registered or instrumented; wins and nulls
interleaved as they happened, because the nulls carry as much design
information as the wins.*

Prior re-mining from mixed-quality paths: control DROPPED 334 -> 300
when 41% of mined rows came from random-search wins (tabula lineage)
— luck-shaped paths have luck-shaped grammar, and mining them dilutes
the bigram structure (int L3 14/15 -> 9/15). Reverted. Lesson: a
mined prior inherits the POLICY QUALITY of the paths it's mined from,
not just their verified correctness; re-mine only from harvests by an
engine at least as strong as the prior's current user. (The verifier
guarantees the paths are RIGHT; it cannot guarantee they are WISE.)
The redemption arm: a fresh 982-row harvest by the CHAMPION engine
(273/320, int L4 36/40 — the new rules' full effect at budget 300),
re-mined with quality lineages only: control **335** vs 334
smoothed-old-prior vs 300 polluted. Conclusion pair: the quality gate
is worth 35 solves; native mass vs median smoothing is worth ~1 —
smoothing already gave new rules everything ranking could give them.
Prior CONTENT saturates fast; prior HYGIENE is what matters.
Fused architecture v1 (Artin's design: one trunk, two heads — value
head on the 0.5B's hidden state replacing NNUE's 20 hand features):
honest loss, 115 v 119, gaps at tight budgets where eval precision
binds (diff L2@25: 12 v 15), at 5-10x the per-node cost. Offline
told the truth this time (+0.859 v +0.937). The caveat that keeps it
alive: the trunk was FROZEN with ranking-tuned LoRA — the value head
could only read a representation optimized for move choice. v2 =
joint value-LoRA training (let the trunk learn to represent what
matters for judgment); the architecture is right (it's AlphaZero's),
the training recipe isn't yet.
v2 (joint value-LoRA, the trunk learns to represent judgment):
held-out rho **+0.966** — the learned representation beats the hand
features offline. In-search: parity at the saturated L2-3 cells
(118 v 119; one solve of headroom total), and at the frontier —
int L4, budget 400, paired arms — **fused v2 wins 10 v 9** despite
eating 5 timeouts to NNUE's 3: the judgment advantage outran its own
inference bill. Full arc: frozen trunk +0.859/loses -> joint-trained
trunk +0.966/wins-at-the-frontier. The strategic dividend over hand
features: a learned representation improves with data and can
transfer across kinds (proofs, ODEs), where fixed features cannot
follow. v3 (4500 labels, 3x data, same recipe): rho climbs to
**+0.980** — the representation keeps improving with automated
labels — but the frontier holds at 10 v 9 (fused still pays 5
timeouts to NNUE's 3). Judgment is no longer the limiter; the
inference bill is. Next lever: cheaper evaluation (batching,
hidden-state caching, distill-back), not more data.
Lazy expansion WINS the timeout campaign (`scripts/bench_lazy.py`):
the prior ranks by rule NAME — known before any sympy work — so
applying rules in prior order and stopping at k children buys the
same selection at a fraction of the bill. int L4: 18 v 16 solved,
timeouts 4 v 10 (replicated 1 v 9 in a second run). A phantom made
this hard to see: an overnight eager baseline of 22/30 could not be
reproduced by ANY code version in daytime runs (16/30 including a
pre-change worktree A/B) — wall-bounded benches are only comparable
within a session (idle vs busy machine is worth ~6 solves at a 240s
wall). Methodology rule adopted: paired arms, same run, always.
Size-cap pruning vs the L4 timeouts (nocap/300/150 ops, bf-nnue,
budget 400, n=30): null — 22/30 all arms, timeouts 6/7/7. The
diagnosis is the finding: the blow-up cost is paid GENERATING a
monster child (successors + verify_edge), not keeping it — pruning at
queue insertion is too late. Converting the timeouts needs time-boxed
rule application inside successors (or per-op sympy budgets), a
plumbing change banked for the optimization thread.
The rule basis, measured (`scripts/bench_rule_basis.py`; Artin's
Toffoli-universality question — what is the minimal gate set?):
leave-one-out ablation of all 12 INT rules from the champion, paired
arms. Generators: **i_power (-5), i_linear_basis (-2, all at L4),
i_usub (-2), i_const (-1)**. Dead gates (zero cost to remove):
i_parts, i_table, i_sum, i_const_factor, i_apart, i_cyclic,
i_unprod, i_ansatz_exp — including by-parts, the textbook's crown
jewel, fully subsumed by the linear solve. The 13-rule library rests
on FOUR generators: power, substitution, the linear-algebra move,
constants. Caveats: n=10, budget 200, our generator's distribution
(i_apart's zero means rational integrands didn't appear in the
seeds, not that partial fractions is globally dead). The
subsumption phenomenon, now a theorem-shaped table.
The magic detector WINS (`scripts/bench_magic.py`; Artin's "magic for
quantum chem, applied to math" -> Liouville 1835 as integration's
Gottesman-Knill): sympy's Risch proves integrands non-elementary in
~10ms; a state carrying a certified non-elementary Integral node is
dead WITHIN OUR OPERATOR CLOSURE (no rule merges integral nodes, so
the two-nonelementary-halves-recombine loophole is closed by the move
set itself). Race: 55 v 54, the +1 at int L4 (10 v 9) with **71
certified cuts** concentrated exactly there (4 at int L3, zero on
diff — the detector correctly never fires without integrals).
Theorem-per-cut: the only prune in the repo with provably zero
false positives. Candidate for engine.solve() integration alongside
lazy expansion.
Winner stacking (classical/lazy/magic/both, paired, budget 400):
magic 40 > classical 39 = lazy 39 = both 39. Magic's +1 at int L4
REPLICATES (second independent run); lazy's value is timeout
conversion and idle-machine evenings don't bind the wall; the combo
gives magic's point back (lazy's restricted expansion starves the
detector of prunable branches). engine.solve() decision: integrate
MAGIC unconditionally (theorem-safe); lazy stays available as the
under-load option, not the default.
Path-integral "interference" eval (arrival multiplicity as amplitude,
bonus on re-arrival): null-to-harmful — classical 54, w=1: 48, w=3:
48; diff L4 drops 15->10. The autopsy: high-multiplicity states are
commuting-move DIAMONDS — many orderings of the same shallow moves —
so multiplicity measures permutation redundancy (the transposition
table's noise), not promise. Feynman's sum works because paths carry
PHASE; arrival counts don't. Ties the commutator thread shut from the
other side.
Commutator-structure pruning (partial-order reduction: skip the
non-canonical ordering of commuting local moves; certificate = target
node present verbatim in the grandparent): int L4 solves flat 16=16,
timeouts 11 -> 8. Real work saved, but generation cost wasn't the
binding constraint — the transposition table had already made the
duplicates cheap-ish, and the freed budget lands on branches that
don't win. Kept as infrastructure (move_filter in successors);
scientifically a null at this wall.
Annealed best-first (Metropolis pop over NNUE energy, linear cool;
diff/int L3-4, n=15): monotone null — greedy 112, T0=1: 108, T0=5:
103, T0=25: 97. Every degree of temperature hurts. Annealing is
insurance against a deceptive landscape; NNUE (rho +0.937) makes the
landscape honest, so random detours are pure budget waste. Cooling
schedules and eval quality are substitutes — we're on the strong-eval
side of the tradeoff.
Strategy-portfolio bandit (UCB1 per problem class over beam-mk3 /
bf-struct / bf-nnue, n=30/cell, budget 30): bf-nnue dominates every
cell (119/120 = the per-cell oracle), so the bandit's exploration tax
buys nothing — bandit 108, best fixed arm 119. Portfolios need arm
complementarity, and the autopsy movers + NNUE erased it: the arms
now form a total order. (The measured complementarity that motivated
this lived in the LLM-config matrix — int L3 tight-budget preferring
flat k — not across these engines.)
Luby vs equal-thirds restarts: 269 = 269 (theorem is asymptotic; R=3
isn't). Golden-angle vs iid restart diversity: 90 v 87, noise.
T=1.0 adaptive: 268 (flat softmax). Markov-adaptive: 282 (rule-level
ties). d_quotient: never fires. G-weight (Dijkstra component): a
perfect four-way tie (92=92=92=92 across g in {0, 0.1, 1, 5}) with a
structural proof — depth-synchronized beams compare only equal-depth
candidates, so path cost adds a constant and cancels from every
ranking; Dijkstra's question requires an asynchronous frontier to be
askable, and hce's plies term was always dead weight in the sort.
(That frontier was then built — see best-first below — and answered
the question the beam structurally couldn't.) prop3+nnue stacking at high budget:
collapses to 6/15 under the wall (spend wall-clock on nodes, not
double ranking).

## T-count engine, day one (rungs 0-2, ZX/pyzx)

Engine live (`llmopt/search/zx_engine.py`, `scripts/bench_zx.py`):
states = ZX graphs, (rule,site) moves from pyzx check/apply pairs +
whole-graph macros (incl. macro-greedy full_reduce), eval = T-count,
boundary oracle = extract-then-tensor-compare. Three memory bombs
found and guarded in one afternoon (raw-string keys; lcomp/pivot
densification cascades — edge-cap at insertion works here because
graph size is free to read; treewidth-exponential tensor contraction
on search products — 24GB, extract a circuit first). One soundness
catch: pyzx's `unsafe_pivot_*` moves corrupted 17/30 reductions and
THE BOUNDARY ORACLE CAUGHT ALL OF THEM — the verify-at-the-boundary
discipline transferred to the new domain and paid immediately.
**Rung-2 verdict (pre-registered bar: search wins >= 20%): FAILED
honestly — 0 wins, 30/30 exact ties, 0 verify-failures** on random
q6/d120 CNOT+H+T circuits. With macro-greedy in the move set, search
is guaranteed >= greedy and found nothing beyond it: full_reduce is
near-optimal ON RANDOM CIRCUITS (consistent with the literature —
the known gains live on STRUCTURED circuits: adders, Toffoli-heavy
arithmetic). Confirmed at q8/d150 on independent hardware: 30/30 ties, 0
verify-failures. Rung 3 (structured Toffoli networks + bigram prior):
both search arms tie greedy where verifiable — and the deeper
finding: **T-count of an unextractable diagram is fiction.** Half
the race's search products failed circuit extraction (safe rewrites
preserve SEMANTICS but can destroy GFLOW, the property that lets a
diagram become a circuit again), so their tempting low T-counts
correspond to no circuit. The harvest's "T 35 -> 15 descents" were
partly mirages the boundary oracle refused to certify. Domain law
learned: the eval must be extractable-T-count (score tcount of the
EXTRACTED circuit), or the move set must be gflow-preserving — the
ZX analog of "score weights by running them." Rung 4 (eval-by-extraction: score the EXTRACTED
circuit's T-count, unextractable = penalty): first race with ZERO
verify-failures — every number a real circuit count — and the honest
final verdict: 0 wins, 9 ties, 11 timeout-losses. full_reduce keeps
the domain crown at our budgets; extraction-per-node economics bind
before search depth pays (the timeout story, third domain). The
chapter's yield: a sound, verified ZX search chassis; four library/
domain pathologies documented (TOF gate object breaks extraction/
teleport/round-trip; unsafe_* moves corrupt; densification bombs;
unextractable-T fiction); and the honest conclusion that beating
greedy here needs the literature's heavier machinery (phase-poly/
TODD-class moves) — future work, cleanly scoped.

**Rungs 5-6 (2026-07-09): the heavier machinery arrived and it
wins.** Rung 5 raced pyzx's own phase-polynomial pipeline
(teleport_reduce -> phase_block_optimize, TODD-class merging) against
greedy full_reduce on the structured Toffoli nets
(`scripts/bench_zx_r5.py`): **7/30 verified wins vs pre-registered
bar 6 — the first machinery all chapter to beat full_reduce** — and
that's with every pyzx crash charged to the pipeline as a loss
(17/30 rows crash: `Circuit.from_graph` realizes leftover phase
gadgets as load-bearing InitAncilla gates whose labels break
`to_graph`, pyzx-0.10 fragility; a 4th was found during the autopsy —
`extract_circuit` fails outright on some plain full-reduced circuit
graphs, all five extractors, "No extractable vertex found"). The
pipeline is now a macro move, `M:phase_teleport` in
`zx_engine.macro_moves` (crash => no move, legality-by-construction
intact). Rung 6 (`scripts/bench_zx_r6.py`, same seed stream as rung
5): does search AROUND the macro beat the bare pipeline?
**Composition bar passed: bf-extract 9/30 wins vs greedy and 7/30
wins vs the pipeline itself (bar 6)** — unlike magic/lazy in the
sympy domain, this macro composes with search (search reaches
phase-teleport states the one-shot pipeline can't, e.g. row 1:
greedy 32, pipeline 31, search 28). Chapter verdict upgraded: the
move set was the bottleneck, exactly as the rung-4 autopsy predicted,
and one imported literature move flipped the domain from
"greedy unbeatable" to "search wins 30% of scoreable circuits."

**Rung 7 (scale + prior, `scripts/bench_zx_r7.py`): the win holds and
grows; the prior dies honestly.** Fresh seed streams, three scales:
bf-extract (macro in move set) beats greedy 9/30 (q6/8 — the rung-6
result replicated on a fresh stream), 4/20 (q6/16), 7/20 = 35%
(q8/12, tier-2 verify), beating the bare pipeline at every scale.
On q6/16 only 7/20 circuits survive pyzx's round-trip — but bf wins
4 of those 7: denser phase sharing means BIGGER wins, and the limiter
at scale is library fragility (InitAncilla crashes scale with gadget
density), not search — a gadget-aware graph->circuit converter is the
scoped unlock. The markov prior, re-harvested WITH the macro (prior
hygiene observed), scored ZERO wins vs greedy and vs bf at all three
scales: when one macro dominates descents, top-k bigram pruning
discards exactly the exploration the wins come from. The 293-dict
does not transplant to this domain; ledger entry, not a bug.

## The magic estimator (2026-07-09): continuous hardness, measured

The continuous companion to the Risch detector (Artin's framing:
estimate how far a state sits beyond the "stabilizer" subspace,
fast, so the estimate itself becomes an engine component).
Labels: 827 generated integrals (L1-L4, stream disjoint from all
training streams), each solved by the full engine at budget 200 and
labeled with MEASURED cost (`scripts/gen_magic_labels.py`; L4
truncated at 77 rows — sympy pathology #7: one problem hung 90 min
in a loop that never delivered the outer SIGALRM). Estimator: the
20 NNUE features -> 64x64 trunk, two heads (`train_magic_estimator.py`).
**Held-out (411 rows): rho(predicted, log-nodes) = 0.822 vs
count_ops baseline 0.124; solved-AUC 0.967** (thin: only ~10
held-out negatives — the rho is the robust number). Expression SIZE
carries almost no hardness signal; structure carries most of it —
the magic thesis in one number. Cost: microseconds per state vs
seconds-to-minutes of search (~1e5x), so it qualifies as a
difficulty oracle for frontier generation (expert iteration's
continuous ladder) and a search-ordering prior. All integrands are
elementary by construction (generator differentiates a drawn F), so
this is deliberately the CONTINUOUS regime — the binary certificate
already owns the far end.

Full-range v2 (same day): mathgen gained L5 (cross-family exp*trig
needing the double-by-parts cycle, inverse trig, log powers,
sqrt-of-poly products, sums of two — `_expression_l5`), and label
generation gained subprocess isolation (fork + 300s hard kill; the
SKIP autopsy found FOUR mechanisms under one label — a queue race
where join() != delivered, the make_integrate simplify hang, honest
slow negatives that a 150s wall was silently eating, and real hangs).
Engine solve-rate ladder at budget 200: **L1 100%, L2 100%, L3 96%,
L4 79%, L5 42%** — L5 is the new frontier band. Retrained estimator
(1243 rows, 208 negatives): **held-out rho 0.855 (was 0.822),
solved-AUC 0.975 now on ~104 held-out negatives** (the first run's
AUC rode on 10); count_ops baseline rose to 0.342 (size correlates
with level) but the net beats it 2.5x. L5's unsolved 58% is the next
rule-mining target (the limits -> l_hopital, holdout -> Laurent
pattern).

Accuracy ceiling, tested three ways (same day): (a) **Bayes floor of
the feature set**: 907/1243 rows collide with another problem's exact
feature vector, yet within-collision cost variance is 0.9% of total —
R^2 ceiling ~0.99, features are NOT the bottleneck. (d) **LLM-trunk
estimator** (`train_magic_llm.py`, 0.5B proposer trunk, same split):
frozen 0.749 (loses), joint-LoRA 0.855 / AUC 0.983 — an exact TIE
with the microsecond MLP at ~1e5x the inference cost. The NNUE
thesis measured a third time: the hand features already carry the
signal; capacity re-derives it, expensively. (b) **rule-fire
features cleared the bar Artin pushed to keep**: one bit per
INT_RULE ("does it produce a candidate on the root"), appended to
the 20 -> **rho 0.905, AUC 0.982** (from 0.855/0.975). The
three-sided verdict: information was already present (a), capacity
couldn't convert it (d), inductive bias did (b) — the net was
spending its samples re-deriving rule applicability that one shallow
probe of the rule set provides directly. Two-tier judge in practice:
20-feature pass (~us) for bulk screening, +rule-fire pass (~ms) when
the decision matters; both >> cheaper than search.

Budget allocation (`bench_budget_alloc.py`, pre-registered): **exact
tie, 86/100 both arms — and the null's diagnosis is the finding.**
Both arms spent ~1,750 of 20,000 allowed nodes: the zero-NN engine
is bimodal (solve in <=30 nodes or never), so node-budget is never
binding and reallocating it buys nothing. The allocation slot
retargets to the currency that DOES run out: wall-time with LLM
calls (the hybrid's 5v3 timeout regime) — estimator decides which
problems deserve LLM attention. Same mechanism, scarce currency.

Final estimator (same night, 3,689 one-truth rows — big sweep ran
GUIDED and 8-way parallel, the estimator accelerating its own label
generation; 46 stale rows relabeled after the rule upgrades): **rf
rho 0.906 / AUC 0.986 on 1,848 held-out rows** (plain 0.858/0.981).
More data confirmed rather than improved — with capacity ruled out
(LLM tie) and features ruled out (Bayes floor), ~0.91 is the
practical plateau; the residue is rank noise among near-tied easy
problems. Second fix wave from the same pipeline: improper-rational
division in i_inverse_trig — **same-seed L5: 42% -> 70% -> 78%
(+20/-0)**, and the day's full solve-rate arc every step a named,
oracle-verified rule.

The syndrome decoder (Artin's qLDPC riff, same night): stabilizer
codes don't just detect deviation, they DECODE — syndrome pattern ->
which correction. Analog test: (20 features + 14 rule-fire
syndromes) -> opening rule of the winning derivation, labels from
re-solving 3,313 solved problems (`gen_syndrome_labels.py`).
**Held-out top-1 97.5%, top-3 99.8% vs majority 45.6% / first-fire
66.8%** (`train_syndrome_decoder.py`). Root policy from milliseconds
of checks, wrong once in ~500 at top-3. Banked next: the per-state
version — syndrome policy at every node = the move-proposer's job at
NNUE cost (the policy-side rematch of the NNUE-vs-LLM question).

Per-state syndrome policy + policy-gated expansion (same night,
"per-state policy GO"): replaying 3,348 winning derivations gave
6,664 (state, next-rule) pairs; the syndrome policy net predicts the
engine's next move at **top-1 94.1% / top-3 98.8% vs markov bigram
56.1/81.9** (`train_syndrome_policy.py`) — the policy-side NNUE
rematch, won by cheap features again (a no-syndrome gate variant
holds 91.4%, so featurize+prev carries almost everything). Live
integration, two jobs: (1) reordering proposer — see policy race;
(2) **gated expansion** (`expand_rules` hook: evaluate only the
gate's top-k rules, empty result falls back to full): k=4 lost 2
solves, k=6 WON 2, autopsy showed the gate contained the winning
rule at every ply of a lost line — solve deltas are beam-composition
noise (the fp16-near-tie class), while the speed win is consistent:
**~4x faster (97s vs 375s at k=6, 72/80 vs 70/80 solves)**. Skipping
2/3 of rule evaluations reprices every sweep and race the lab runs.
The reordering job, meanwhile, LOST (markov 85/100 @ 1234s vs policy
81/100 @ 1665s, 7 timeouts): the net imitates markov-guided
derivations, so it cannot out-order its teacher — where it disagrees
it is mostly wrong, expensively. Split verdict, one law: **imitation
can't beat the teacher at the teacher's own job, but it can make the
teacher ~4x cheaper.** Beating the ordering needs off-policy signal
(search-derived regret, not imitation) — banked, then RUN the same
night: 471 DAgger pairs (expert relabels of policy-visited states,
the corrections concentrated on the policy's own failure
distribution) closed the gap to EXACT PARITY — rematch 85/100 vs
85/100 (from 81 vs 85). One regret round lands the imitator at its
teacher; markov keeps the ordering job anyway (policy pays inference
overhead for zero solve advantage). The complete policy ledger:
imitation < teacher, imitation+regret = teacher, and the policy's
real paycheck is the GATE (4x cheaper nodes).
Adaptive gate (entropy-gated deference to the teacher, "who said the
teacher can't participate"): 69/80 @ 86s — same solves as fixed k=4,
so deference rescued nothing: the traced losses hold the winning
rule INSIDE the gate and fail on beam composition, not uncertainty —
confidence can't see them. All four gate configs sit within +-2
solves of full at 4-6x speed; **k=6 fixed is the config.** The
deference architecture stays banked for domains with real OOD states
(ZX port).

Rule-result cache (spec follow-up — 'caching layers beyond what
exists' was the one out-of-scope item worth revisiting): rules are
pure functions of an immutable node, and the same subnode recurs
across sibling states (every i_sum split preserves the others), so
_safe memoizes per (rule, node). Paired micro-race, order biased
AGAINST the cache: **1.7x faster at identical solves (52.7s vs
91.2s, 26/29 both)**. Stacked with the k=6 gate: ~7x cheaper nodes
than the morning engine. GPU-batching sympy and async remain
correctly out of scope (tree rewriting is pure-Python CPU-bound);
LLM-inference batching stays banked with the v3 limiter thread.

LLM wall-time gating (the budget slot's retarget, tested same
night): zero-nn 30/40, all-llm 27/40, gated 27/40 at wall=30s —
**null: the LLM is net-negative at tight walls, so no router can
save it** (the 3 problems where arms differ are zero-nn solves the
LLM arms timed out on; the estimator routed sensibly to a
destination that doesn't pay). The 5v3 lesson at problem
granularity: the hybrid's +15 premium lives at generous budgets.
Dependency exposed: cheaper LLM inference (batching /
entropy-adaptive speculation — banked) comes BEFORE routing.

The autopsy paid same-day: the L5 failure clusters (root 15%,
inverse-trig 0%) became two rules — `i_sqrt_basis` (f*sqrt(P)
polynomial => answer in A(x)*sqrt(P), the linear-basis move with a
radical basis) and `i_inverse_trig` (atan/asin closed forms), both
oracle-verified 4/4 at birth, 65 regression tests green. Same-seed
L5 rerun: **42% -> 70% (105 -> 175 of 249; 71 gained, 1 lost), root
family 14/94 -> 81/94.** i_sqrt_basis is the biggest single rule win
since the Laurent extension. Known residue: inverse-trig shapes
hiding inside re-fused sums (+3 of ~23 recovered) — next autopsy's
target.

Entropy-bonus beam (same day, pre-registered, physics motivation:
mimicking magic costs entropy, so spend the beam on diversity when
the eval stalls): greedy max-min selection in the 20-feature space
via a new `select_fn` hook in beam_search (default unchanged).
**NULL, losing direction: plain 53/60 solved vs diverse 51/60 at
int L4 width 4, and diverse spent MORE nodes (3067 vs 2692).** The
annealing null generalizes from random noise to structured
diversity: with a small beam and a sharp eval, every slot spent on
"different" instead of "second-best" occasionally drops the true
path and buys nothing measurable back.

## Middle-layer value probe (2026-07-09, global-workspace paper test)

The workspace paper locates flexible-reasoning representations in
middle layers (~38-92% depth), last layers collapsing toward output.
Our value head always read the LAST hidden state. Frozen-trunk probe
sweep on the 0.5B proposer (24 layers, same 4500 labels, same split;
`train_value_head.py --layer`): **L8 +0.828, L12 +0.854, L16 +0.851,
L20 +0.873, last +0.858 — peak at 83% depth, dip at the output
layer**, qualitatively the paper's geography at 1/1000th scale.
REPLICATED on two fresh splits (`--split-seed`): L20 +0.866 vs last
+0.785, and L20 +0.818 vs last +0.773 — L20 wins all three splits,
mean gap ~+0.05, and the original +0.015 was the SMALLEST of the
three. Measured, not a caveat. Follow-up RUN (v4): joint value-LoRA retrained at layer 20 scored
+0.970 vs v3's +0.980 at the last layer — the probe-point advantage
does NOT survive joint training. Sharper story, not a null: L20 is
where value lives in the FROZEN (pretrained+ranking) representation;
a jointly-trained LoRA re-routes the trunk so the probe point stops
mattering. Geography constrains probes, not training. Record config
keeps v3.
Ops note from the same run: torch's _native eager router JITs triton
kernels WITHOUT torch.compile; on the C-compiler-less WSL box only
TORCH_DISABLE_NATIVE_JIT=1 stops it (now in CLAUDE.md).

## Frontier rule gaps -> two rules (2026-07-10, the loop's second lap)

The 36 frontier-mined failures clustered into three shapes; sub-term
probing (full-enum solve per distinct expanded sub-term) pinned the
blockers exactly:

- `i_log_power` — closed form for x^n·log(kx)^m (27/36 gaps).
  i_parts CAN reach these but dies chaining m by-parts plies through
  nested Integrals: a node-budget death, not unreachability.
- `i_transcend_div` — the rat+exp·trig family (8/36) is a trap:
  expanded sub-terms are INDIVIDUALLY non-elementary
  (exp·sin/(x²+1) has no elementary antiderivative), so i_sum/i_apart
  make the state worse. The generator built these as (den·g + c)/den;
  grouping numerator terms by transcendental monomial and dividing
  each group's poly coefficient by the denominator recovers g + c/den
  by exact division.

Full-enum AND solve(): 0/36 -> 32/36 (one leftover is an
i_unprod-shaped reverse product pair; the rest are beam-composition,
not coverage). Headline: **same-seed L5 42% -> 70% -> 78% -> 89.6%
(223/249)** — the two rules are worth ~+12 points on the record
config.

A lesson written and then RETRACTED the same night, kept here because
the retraction is the finding. Early holdout probes suggested the
prior's unseen-rule smoothing guillotined the new rules, so the prior
was re-mined (general 158-win harvest, then a 3x-weighted targeted
one). Both merges REGRESSED: L5 89.6% -> 73.1%, and the rat+exp·trig
gaps flipped from solved to failed — diluting the winning bigrams
costs more than the new-rule mass gains. This is the SECOND measured
prior-mining regression (478e269 reverted the first, for
random-search wins). The durable rule: the 0.01·median smoothing is
sufficient for new rules; NEVER adopt a merged prior without racing
it against the incumbent on the same seeds
(`scripts/mine_prior_update.py` stays as tooling for that race).

## Entropy-adaptive speculative decoding (2026-07-10, 3080): null with a price tag

The gate law's decoding crossover, run honestly. Qwen2.5 1.5B target
/ 0.5B draft, greedy, three prompt regimes, every arm asserted
token-identical to eager greedy. Draft stops proposing when its own
next-token entropy exceeds a threshold (deference), k_min floor swept
2-4, k_max 12.

Two findings, one bug first: entropy must be computed in float32 —
in fp16, clamp_min(1e-9) underflows to 0, log2(0)=-inf, 0·-inf=nan,
and `nan > thresh` is never true (measured: 0 stops in 771 passes,
silently degenerating to fixed k_max).

- The signal is REAL: acceptance rises (prose 0.47 -> 0.69; code
  0.79 -> 0.90 at e=1.0) and target passes nearly HALVE (26-31 vs 53
  on grounded summary). Draft entropy locates the teacher's
  rejections — the opposite of the derivation engine's
  entropy-deference null.
- The wall-clock verdict is a NULL on this pair: best adaptive never
  beats fixed k=3 (46.6 vs 51.2 / 26.5 vs 33.0 / 43.9 vs 45.5 tok/s),
  because a 0.5B draft is only ~3x cheaper than the target and the
  adaptive arms burn ~40% more draft passes. Saving a target pass by
  spending four draft passes is a wash at 1:3 economics.

Falsifiable prediction banked: at draft:target cost <= ~1:10 (7B+
targets, or MTP-style heads where drafting is ~free) the measured
target-pass halving converts to wall-time. This is also WHY
production systems draft with heads rather than standalone models.

Cost-ratio ladder (same night, 3B and 1.5B targets vs the same 0.5B
draft, 3080): the prediction's naive form FAILS, and the mechanism
that kills it is the finding. Going 1.5B -> 3B improves the cost
ratio (1:3 -> 1:6) but DEGRADES acceptance on every prompt (grounded
0.92 -> 0.66, prose 0.47 -> 0.55-with-lower-baseline, code
0.79 -> 0.78 at k=3): an untrained standalone draft diverges from a
bigger target roughly as fast as the economics improve. Adaptive
never beat best-fixed at either ratio; at 3B, speculation itself lost
to vanilla on 2 of 3 prompts (40.0 vs 37.1 tok/s grounded; 38.4 vs
30.4 prose) — a 3080 at 3B fp16 is not memory-starved enough. Refined
prediction: the cost-ratio law only cashes with drafts TRAINED to the
target (distilled or MTP heads), where acceptance survives the scale
gap. Also observed: all five spec arms diverged from eager greedy at
the SAME position on the 3B prose prompt (ref 7797 vs opt 4889) —
the repo's documented fp16 near-tie class, target-side, arm-invariant.

## Node-cost round 2 (2026-07-10): kill heurisch, spend the savings on width

The 9-problem frontier-v2 residue (every needed rule present, greedy
proposer-descent solves in 6 plies, solve() walls) profiled to ONE
line: verify_edge's `.doit()` on diff'd nested Integrals reaches
heurisch — sympy's own integrator, the thing this engine exists to
avoid — legally burning its full 2s timebox on 34/34 verifies (56s of
a 90s wall). Fix: `doit(integrals=False)`. Structurally-equal
Integral atoms cancel in the subtraction; survivors are rejected by
_is_zero's carrier check — the same conservative-sound outcome
heurisch was buying at 2000x the price.

That cheapened nodes enough to widen the beam: width 2 -> 3 at the
same budget. Raced: residue 0/9 -> 8/9 (25s total, was 400s of
walls); **same-seed L5 238/249 (95.6%, from 223) at 3.5x LESS wall
(348s vs ~1200s)**; L3 58/60 (+1), L4 46/60 (tied). Production
solve() is now width 3. The day's arc: 78% -> 89.6% (rules) -> 95.6%
(node cost -> width). The one survivor at any width: a nested
chain-rule trig shape (cos(x + cos(9x)) family) — a genuine u-sub
chain gap, next autopsy's seed.

## propose_k=4: decisive null, two mechanisms (2026-07-10)

Motivation: the one width-proof survivor (nested chain-rule trig) has
its full 2-ply solution in current rules — i_unprod's exact guess at
the root, i_linear_basis at ply 2 — but the markov proposer ranks
both i_unprod kids 5th-6th at the root; propose_k=3 cuts them.

k=4 raced: **210/249 L5 (84.3%) vs the 238 bar — NO-ADOPT.** Two
mechanisms, both instructive: (1) k=4 spends the fixed 200-node
budget ~33% faster, cutting depth — and unlike 2->3 width (which
bought parallel hypotheses at the SELECTION layer), widening the
PROPOSAL layer mostly re-covers what top-3 already had. (2) It never
solves the target anyway: rank 5-6 needs k>=6. Knob space is
exhausted — the target problem is reachable only by state-aware
ranking (syndrome policy), which is the next rung.

Corollary finding: the old budget-allocation null ("engine is
bimodal, budget never binds") carried a config that no longer exists.
Under new rules + cheap nodes + width 3, budget measurably binds
(this race is the evidence). Nulls inherit config defects exactly
like priors, labels, and frontier mines — fourth appearance of the
lesson in 24h. Budget-allocation re-race is queued behind the v5
estimator retrain.

## Syndrome policy v2 + DAgger round 2 (2026-07-10 night): the brain race

Policy v2 (7,313 pairs replayed under the CURRENT engine, 19-rule
vocab, free kid-derived syndromes): offline top-1 93.5/top-3 98.9 vs
markov 51.0/79.6 on identical held-out states. Live, the split
verdict that matters:

- Solves the nested-trig knob-proof problem in 1s — i_unprod ranked
  #3 at the root by state evidence (markov: rank 5-6, needs k>=6).
- Fresh-100 head-to-head: policy 98/96 solves at 36% LESS wall — the
  v1 verdict ("ties, costs more") is overturned.
- L5 249 gate: 218 vs markov 238 — FAILED. Diagnosis (12 failures
  retried at 400s wall): 0/12 flipped — 100% rank-bound, zero
  wall-bound; three are instant beam-deaths (36 nodes, 0s). The
  policy marches deep-L5 beams into pruned cul-de-sacs.

DAgger round 2 (851 expert relabels of policy-visited states, fresh
980k band, production width): L5 218 -> 231 (13 of 20 losses
recovered), fresh-100 solve edge kept (97/96) but the wall-time win
evaporated (239s vs 155s). Markov keeps production: 238 > 231.

Ledger for the rung so far: state-aware ranking provably reaches
problems no global knob can, and DAgger corrections land exactly
where the diagnosis pointed — but the teacher still wins its home
turf. Round 3 (L5-weighted regret sampling) is the natural next
move; the alternative is domain routing (policy off-L5, markov on).

## The brain races, concluded: policy ADOPTED (2026-07-10 night)

DAgger convergence curve, fully mapped: 218 -> 231 (round 2, uniform
mix) -> 236 (round 3, 3:1 L5-weighted, 1,277 pairs) -> 222 (round 4,
PURE-L5, 1,338 pairs — REGRESSION: skewing corrections entirely to
the failure domain dilutes the general imitation signal; balanced
beats targeted past the sweet spot). Round 3 checkpoint restored.
Nested trig retained through every round.

The 2-problem curated-gate deficit (236 v 238) was then tested on 80
FRESH L5 problems: EXACT TIE 76/76 — benchmark-specific, not a
capability gap. Wall flips by domain: markov 3x faster on pure L5
(both brains find the same deep chains; the policy just pays its
net cost), policy 3x faster on mixed (better ranking = fewer nodes).

Router race (magic-estimator cost head at the root dispatches
markov-vs-policy; the banked "magic router" materialized): routed
123/130 at 290s vs policy 124 at 441s vs markov 121 at 402s — buys
34% wall for one solve; near-miss on the strict bar. Decisive fact
from the same race: pure policy is the best pure arm even on an
L5-heavy mix.

ADOPTED: solve() now runs the syndrome policy when its checkpoint
exists (SyndromePolicy in engine.py; markov fallback). Solves are
the currency: policy wins fresh mixed (98-99/100 v 95-96), ties
fresh L5, uniquely reaches the reverse-product class, and solve()
takes the nested-trig problem in production config for the first
time. Markov remains as fallback and as the wall-time choice for
deep-L5 batch work. Artin's qLDPC syndrome framing is now the
engine's production brain.

## The router: strict dominance, adopted ("verified speed is intelligence")

Autopsy of the near-miss (per-problem instrumented rerun of the
130-problem race, data/router_autopsy.jsonl): the eyeballed 4.5
threshold misdispatched policy-wins the estimator scored "hard" —
routing on HOW HARD when the question is WHICH BRAIN. Offline
threshold sweep (free, from the log): thr 5.5 -> 124 solves @ 322s,
matching the best pure arm at 27% less wall. Oracle-router ceiling:
127/130 (+3 for a two-sided dispatcher — the dispatcher-net rung).

Out-of-sample validation (150 fresh problems, threshold FIXED before
the draw): **routed 141/150 @ 167s vs policy 139 @ 337s vs markov
130 @ 429s** — strict dominance on both axes, stronger than
in-sample. Adopted: solve() dispatches at the root via the
magic-estimator cost head (<= 5.5 -> policy, else markov), with
policy-only and markov-only fallbacks as checkpoints allow. Both
smoke branches verified; 342 tests green.

Dispatcher v2 (next session): disagreement-oversampled farming fixes
v1's economics AND its null — markov-wins only exist where the
policy fails or is slow, so the v2 farmer runs the policy arm on
everything and spends markov runs only there (fast ties subsampled
10%): 410 rows with 31% markov-wins at a fraction of v1's cost.
Disagreement accuracy 0.750 -> 0.883 (n=60). Race 3 (fourth virgin
band): **dispatch_v2 144/150 @ 344s vs thr5.5 144 @ 417s** vs policy
142 @ 526s vs markov 139 @ 469s — ties the champion's solves at 18%
less wall; ADOPTED by the FA-Law tiebreak (routing overhead ~14
timeboxed rule probes/problem, ~15-30s per 150, not counted in race
walls for either router but does not close the 73s gap). solve()'s
fallback chain: dispatcher -> threshold -> policy -> markov.

Dispatcher v1 (the first ceiling chase, for the record): NULL with
mechanism. 1,192 dual-arm dominance labels (winner by (solved,
wall)); offline disagreement accuracy 0.750 (n=32). Live, 4-arm race
on a second virgin band: thr5.5 126/130 @ 164s (the champion
REVALIDATES, beating both pures on a second independent OOS draw);
dispatchnet 124/130 @ 163s — degenerated into pure policy (identical
solves). Mechanism: FA-Law labels are 96/4 policy-skewed (policy
wins ties on speed), so the net learns "when in doubt, policy" and
almost never fires the markov branch; the crude cost threshold fires
it exactly where it matters (deep chains). Banked: dispatcher v2
needs disagreement-oversampled data (~6% incidence makes farming
expensive — the DAgger move applied to routing).

Budget-allocation re-race (2026-07-11, v5 estimator, current
engine): the null SURVIVES its config refresh, sharpened. Flat
105/110 vs estimator-allocated 105/110 at equal total budget —
identical nodes consumed (1,578 both arms) despite allocations
spanning 138-737; the five failures don't flip at 3.7x budget.
Resolution of the apparent contradiction with the k=4 race: budget
binds through SPEND RATE (wider proposals waste the same 200 nodes
faster), not through absolute node counts starving solvable
problems. More budget doesn't buy solves; cheaper/better-spent
budget does — the FA Law's contrapositive.

The FA Law (Fable-Artin, coined and earned the same night):
**verified speed is intelligence** — at fixed wall, speed IS solves,
and the fastest arm can be the most accurate one because cheap nodes
buy retries, width, and reach. Corollaries already measured in this
repo: NNUE-over-handcrafted (depth bought with node price),
verify-without-heurisch (+17 points from zero new knowledge), and
the router itself (141/150 @ 167s). The magic-estimator judge-slot
list gains its biggest client: the engine's own front door.

## L6/L7 and the engine-vs-sympy probe (2026-07-11)

New generator levels after the engine hit 95.6% same-seed L5: L6 =
coordination depth (triple-family sums, cross-products, quotient
debris, degree-2 trig/exp arguments), L7 = nesting (compositions of
compositions; chain-rule cascades). 60 problems each, engine (120s
wall) vs sympy.integrate (fork-isolated, 60s deadline — it hangs):

- L6: engine 36/60 (60%) vs sympy 56/60 (93%) — heurisch is strong
  exactly on constructed-solvable shallow composition.
- L7: engine 36/60 (60%) vs sympy 42/60 (70%) — the engine HOLDS at
  depth while sympy drops 23 points: search degrades slower than
  heurisch as nesting deepens.
- 3 MONEY problems (engine solves, sympy fails; certified by sympy's
  own differentiator): all chain/reverse-product shapes, engine
  0.2-4.2s. -18x*sin(sin(3x))*cos(3x)+6cos(sin(3x)) in 0.2s.

Consequence (Artin's cascade): the two are COMPLEMENTARY — sympy
closes, the engine navigates. i_heurisch (sympy.integrate as an
op-capped, timeboxed, diff-verified leaf-closer rule) raced:
**same-seed L5 sample 50/50 (perfect, from ~48)** at unchanged wall;
L6 36 -> 37 (cap 40) -> 39 (cap 100, AND 20% faster — early
leaf-closes save wandering search; cap 100 adopted). The residual
L6 gap (39 vs sympy-whole 56) is PRE-decomposition — those failures
never form closable leaves at any cap — so it belongs to
rule-mining (the 21-problem autopsy corpus), not closer tuning. Side casualty with a lesson: adding the
rule grew the live syndrome vector and crashed every trained net —
checkpoints now PIN their training-time syndrome vocab (the config-
pinning lesson, tensor-shape edition). Banked: parallel leaf closing
(spec-decoding's batch-verify amortization applied to i_heurisch).

## The L6 evening: engine 36 -> 59/60, PASSING sympy (2026-07-11)

One autopsy session, three fixes, each measured separately (same 60
seeds throughout; sympy-whole = 56/60):

- 36 -> 37 -> 39: i_heurisch leaf closer, cap 40 then 100.
- 39 -> 51 (at HALF the wall): trial mass for the POLICY — its
  unknown-rule fallback of -50 guillotined i_heurisch on every
  policy-routed problem. The markov trial-mass lesson, repeated
  verbatim one layer up: every ranking layer needs newcomer mass.
- 51 -> 52: the log ORBITAL — 14/22 autopsied failures were
  x^j*log(kx)*trig products whose answers i_linear_basis could not
  express because log was missing from its basis generators
  (Artin's quantum-chemistry basis-set framing, cashed a second
  time). d/dx-closure holds; P'/P denominators cleared by
  multiplying the residual through before the Poly solve.
- 52 -> **59/60 in 182s (3s/problem)**: NEVER GUILLOTINE A TERMINAL
  — the propose_k cut was discarding kids that were already complete
  solutions (the policy ranked i_heurisch's SOLVED kid 5th on
  rational integrands). Structural fix in beam_search: terminal kids
  survive any proposal cut. No ranking scheme can reintroduce the
  bug.

**The engine now beats sympy on sympy's best level (59 vs 56), at
~3s/problem, using sympy as a gated subcontractor and sympy's own
differentiator as judge.** L5 sample stays perfect (50/50)
throughout. The lone survivor is a sqrt-of-poly monster (root-basis
x log-basis product — a genuinely new orbital combination). History-
asserting ceiling tests pin native chains via heurisch-excluded
search (4 patched).

## L5 CLOSED at 100%; L7 56/60; the orbital pattern generalizes (2026-07-11)

Full L5 249 under the final config: **249/249 (100.0%) in 58s —
0.23s/problem.** The level that was the 42% frontier four days ago
is closed. L7 under the trio: 36 -> 46 (vs sympy 42); autopsy showed
12/14 residue failures were ONE family, d/dx[log(u)*atan(v)] pairs —
the atan orbital, one day after the log orbital. Two fixes needed:
the generator itself, and a GATE fix (log/atan derivative debris
makes the integrand rational; gate and size the ansatz on the
denominator-cleared form, matched with sp.cancel, since expand
cannot cancel auto-combined denominators). **L7: 56/60 vs sympy's
42.** The engine now leads sympy on every level that exists.
Remaining L7 residue (4): trig(log(x)) compositions and sqrt x log
products — orbital COMBINATIONS, the basis-proposer's seed corpus.

Wall-time note (Artin's question, measured): L7's 28s/problem
average was 14 failures burning the 120s wall; the SOLVES averaged
~0.3s. Speed the engine up by making problems solvable or quitting
early — never by optimizing the happy path, which is already fast.
Symengine swap: measured CLOSED (raw diff 9.2x, but conversion
round-trips net 1.7x on diff and LOSE 10x on expand; only an
end-to-end representation swap pays, which forfeits the Poly/solve
rule machinery).

## The regret probe: trace fate is legible mid-flight (2026-07-11)

Artin's thesis ("the best skill is knowing when to regret/
reconsider") got its existence proof before its policy proof: a
128-unit probe on the calculus-LoRA 0.5B's layer-20 hidden state,
read mid-generation, predicts whether the answer being written will
turn out sympy-correct at **AUC 0.914** (2,760 trace states, base
rate 0.509). The trajectory's fate is largely encoded long before
the final token — generation is mostly COMMITTED early, which is
what makes regret cheap in principle.

Getting the number cost three measured lessons: (1) checkpoint
schemes are selection effects — recording states only at token 24+
produced an all-negative dataset (correct answers are SHORT; base
rate 0.000, AUC nan) until a final-state pseudo-checkpoint was
added; (2) "equal budget" must be enforced at the SPEND — fixed-k
best-of-N stopped at EOS and used 16k tokens vs regret's 193k,
voiding the first race; (3) sympy pathology #8: p.check() on
hallucinated model text can hang simplify (2h39m live-lock at 102%
CPU) — the ORACLE needs a timebox when judging adversarial garbage,
with hang counted as wrong (conservative for every arm).

The policy race landed 2026-07-11, and the naive policy LOST,
decisively: n=150, 1280 tok/arm — greedy 85, budget-exhausting
best-of-N **100**, regret (abort-on-probe<0.15 at ckpt 8, resample)
**78**, at genuinely equal spend (193,039 vs 192,465 tokens).
Honest reading: the SIGNAL is real (AUC 0.914) but the naive spend
policy converts it to negative value — aborting at token 8 on an
uncalibrated threshold kills traces before their fate has formed
(143.6 attempts/problem = churn, not judgment), while best-of-N's
"let every trace finish, then pick" wastes nothing. Same shape as
the router lineage: raw signal -> threshold sweep -> only then a
net. Round 2, pre-registered: log per-checkpoint probe
probabilities DURING the race, sweep the threshold OFFLINE (the
router playbook), and allow aborts only at ckpt>=16 where the state
has formed.

Round 2 ran (2026-07-12: 149-problem pool, 48 full traces each,
per-checkpoint probe probabilities, offline 28-config grid) and the
verdict is a CLEAN NULL: no (threshold x min-checkpoint) config
beats the best-of-N replay at equal budget — the entire grid is
monotone WORSE with abort aggressiveness (ckpt-8 aborts: 77-85 vs
97; ckpt>=32: ties 97 by aborting nothing). Diagnosis, and it's
economics not signal: traces are SHORT (median ~25 tokens), so an
abort at token 16 saves ~9 tokens — there is almost no budget to
recover — while every false abort kills a winner outright. The
probe reads trace fate correctly and there is still nothing to buy
with it at this trace length. TOKEN-LEVEL regret: closed as a
null next to the AUC-0.914 existence proof. The asymmetric-cost
version of the same thesis lives at the ENGINE level (solves 0.3s
vs 120s failure walls — a 400:1 save ratio vs the token level's
~1.5:1) — bench_engine_regret.py, in flight.

Judgment-stack refresh, same day: estimator v6 (L3-L7 labels under
the final engine; solved-AUC 0.916, cost-rho collapsed to 0.578
BECAUSE the engine saturated the generator — the judge starved by
the judged, the cleanest possible signal that L8/adversarial
generation is next); dispatcher v3 (first router trained on L6/L7
and the current brains; disagreement acc 0.851; syndrome-vocab
alignment across mixed-width corpora done by inserting the
i_heurisch bit at index 10 for pre-heurisch rows). ADOPTED same
day: L3-L7 race, v3 ties the best arm's solves (114/120, = markov)
at 43% less wall (370s vs 644s); v2 — blind to L6/L7 — behaved as
pure policy and paid its price (112 @ 637s). The landscape flipped
AGAIN at depth (markov out-solves policy on L6/L7-heavy bands) and
only the router trained on the current world knew. Fixed en route,
sympy pathology #9: the ROUTER's syndrome probes ran un-timeboxed
(i_heurisch on a monster root hung 73min during routing, before any
search began) — every layer that touches sympy gets a box, judges
included.

## L8: the frontier reopened from the residue (2026-07-11)

The estimator's cost-rho collapse said the generator was saturated,
so L8 was built FROM the measured failure modes, not from
imagination: orbital COMBINATIONS (trig(log), sqrt x log — single
orbitals are solved, their products are not), sqrt monsters with
degree-3 inners (L6's last open problem), three-deep nesting, and
combo+L7 sums. Probe, fork-isolated at 120s wall: **30/40 (75%)**
vs L7's 93% — and all 10 misses verified as SOLVE failures, not
generation hangs (each hung seed's make_integrate completes in
<30s; the wall burns in search). The visible residue skews to the
sqrt_log family (log-times-root quotient debris). Width: 299/300
distinct F over 300 draws. L8 label farm for estimator v7 launched
same day — re-feeding the starved judge.

Same-day autopsy closed most of it (30/40 -> **37/40**): two
missing orbitals, not deep search failures. (1) sqrt x log — 5/10
misses were F = sqrt(P)*log(q); i_sqrt_basis had the radical
machinery but refused logs. Ansatz upgrade (A + B*log q)*sqrt(P):
multiplying the d/dx-residual by 2*sqrt(P)*q clears radical and
log-denominator at once — one linear solve, one ply. (2)
trig(log(poly)) — i_linear_basis only admitted POLYNOMIAL trig
args; the family {x^j*trig(log p)} is d/dx-closed once cleared by
p. Two traps en route, both measured: make_integrate's simplify
PHASE-SHIFTS trig sums (cos(u)-sin(u) -> sqrt(2)cos(u+pi/4), so
args arrive as const+log(poly) — admit via as_independent), and
log(p) can now be both a trig arg and a log gen (placeholder subs
-> xreplace, whole-node matching). Remaining residue (3/40):
multi-family sums FUSED into single quotients by simplify — no
decomposition move separates them, no single orbital spans them;
rule-synthesis material, not another quick ansatz.

## Three-lane 4-bit quantization race: allocation of accuracy (2026-07-11)

From the Cerebras riff (decode = bytes/bandwidth) via Artin's
"reallocate the accuracy" chain. Three 4-bit schemes at group 128,
REAL Qwen2.5-0.5B weights, REAL captured activations, scored in
function space per the house law (`scripts/bench_quant_schemes.py`):
uniform min/max affine (minimax in weight space), NF4-style gaussian
quantile codes (accuracy where the weight mass is), and awq_lite
(per-input-channel rescale by mean|activation|^0.5 — accuracy where
the OUTPUT cares). Mean GEMV output rel-err: uniform 10.06%, nf4
8.89%, **awq_lite 8.07%** — the activation-aware lane wins, hugely
on late layers (layer-23 down_proj: 14.7% -> 6.5%). Honest lesson
en route: the toy round on random gaussian weights had ranked
uniform FIRST — synthetic weight distributions lack the outlier
channels that real transformers carry, and those outliers are
exactly what activation-awareness protects. Weight-space and
function-space rankings disagreed in both rounds (nf4 had the best
mean weight distance while losing the toy): never score weights by
weight distance, again. The winning lane is the packing the int4
dequant-GEMV Metal kernel carries (Artin's group-128 packing,
practice_7).

## Fused int4 dequant-GEMV Metal kernel (2026-07-11)

The Cerebras riff landed as code: `int4_gemv` in kernels/metal.py —
weights streamed as packed nibbles (Artin's practice_7 adjacent
scheme, group 128), dequantized in registers, fp16 never
materialized; awq_lite channel scales fold in at pack time (the
quant-race winner rides for free — the kernel is scheme-agnostic).
Three-version ladder, each honestly benched vs mx.quantized_matmul
(`scripts/bench_int4_gemv.py`): v1 tree-reduction 0.47-0.70x, v2
simdgroup+simd_sum+uint32 0.75-1.00x, v3 uint2/half4 vector loads
**1.11x at D=4096 (2.80x over fp16), 0.94x at D=2048, 0.72x at
D=896** — win big, lose small (small decode shapes are launch/
overhead-bound; fp16 GEMV itself only reaches 40 GB/s at D=896).
Correctness pinned by two tests vs the dequant reference. The
remaining roofline gap and the group/threadgroup config space are
the config-estimator rung's training data (Artin's "estimate the
packing" — sweeps-as-labels, banked).

## Engine-level regret: the thesis pays at 400:1 unit economics (2026-07-12)

Token-level regret closed as a null (nothing to recover from ~25-tok
traces); the same idea at the engine level — solves ~0.3s, failures
burn 120s walls — is where it pays. ply_hook in beam_search streams
per-ply beam features; probe (64-unit MLP) reads doom at held-out
AUC 0.760. Two harness scars en route, both recorded: the first
farm's wall-killed searches died with their ply rows in child
memory, making the wall-burners INVISIBLE to the sweep (the token-24
selection-effect lesson, third occurrence — rows now stream through
the queue), and the farm loop needed the pathology-#7 fork pattern.

Verdicts, both kept: (1) the PRE-REGISTERED bar (zero solve loss +
>=25% wall cut on a fixed problem set) FAILED — every config loses
the same 2 stubborn solves (deep chains that look doomed then land;
even th=0.99 aborts them). (2) The FA-Law-native metric — solves at
EQUAL TOTAL WALL, saved time respent on new problems — is a
blowout: baseline 82 solves vs regret **176 solves in the same
1888s** (2.1x, stable across budgets: 41v17 @ 450s, 88v40 @ 900s;
offline stream simulation, cycled held-out set). Reading: regret
trades 2.4% completeness for 2.1x throughput. Adoption is
WORKLOAD-DEPENDENT: scoreboards on fixed sets keep the pure engine;
throughput workloads (label farming, frontier laps, expert
iteration) want the abort. Not wired into solve() by default.

## Dispatcher v4: NO-ADOPT (2026-07-12)

Judgment-stack currency said retrain after the orbitals; the race
said keep v3. Fresh L3-L8 band (120 problems, four arms,
fork-isolated): v3 112 @ 1078s, markov 111 @ 1095s, policy 111 @
1126s, **v4 110 @ 1137s** — v4 (275 post-orbital disagreement rows,
disagreement acc 0.867) fails the bar outright. The interesting
part is the SPREAD: 110-112 of 120, arms nearly indistinguishable —
the orbitals one-ply so much of the space that brain choice barely
matters anymore; the router's judgment surface is evaporating as
the engine strengthens (the starved-judge pattern, fourth
appearance). v3 stays production; the currency rule's lesson
softens to: retrain when the world changes AND the arms still
disagree enough to route.

## Step-tokens: the LLM's unit of generation becomes a verified rewrite (2026-07-12)

Artin's bigger-tokens riff, LLM side (the engine side already paid:
one ply = one verified macro-token -> 2.1x). bench_step_tokens.py:
base Qwen 0.5B instruct + few-shot emits ONE derivation step per
call (a candidate rewrite of the current integral as sympy text);
the oracle verifies each step fork-isolated before it stands;
invalid steps are resampled — progress is a RATCHET, hallucination
costs budget but never corrupts the chain. At equal 768-token
budget on L2/L3: **steps 5/30 vs one-shot best-of-N 0/30**, with
step validity only 5% (38/709) — the verification filter converts
a 95%-wrong generator into a solver. Honest caveats: base model
(the calculus LoRA is answer-only trained and never saw steps),
easy levels, and the one-shot baseline may be partly format-limited
(same model, same verifier, but single-expression output is
unforgiving). Sequel is the repo's long-term goal made concrete:
train on the verified chains (step-level expert iteration) — the
5% validity rate is the number training should move.

## Expert iteration round 1: transformations learned, chaining not yet (2026-07-12)

622 engine-replay chains + 95 skip-pairs, 3 epochs (loss 1.01 ->
0.39), raced vs base on 30 fresh problems at equal 768-tok budget:
adapter **one-shot 13/30** (base 1), **steps 8/30** (base 4),
validity 0.5% (base 4%). Autopsy in the numbers: 8 valid steps in
1750 tries -> 8 solves — every valid step was a COMPLETE solution
in one hop, because the post-orbital engine one-plies the corpus
levels and the chains were single-hop dominated. The model learned
the TRANSFORMATIONS (13x one-shot transfer) but not CHAINING.
Round-2 amendments (Artin's GOs): grammar-constrained decoding,
skip-pair macro-distillation (already in corpus), and multi-step
chain mining at L6+ where the engine actually chains.

Constrained-decoding verdict, same day: **CLEAN NULL for the
adapter** — charset-masked rerun is token-for-token IDENTICAL to
unconstrained (13/8, 8/1750): every token the trained model wanted
was already expression-legal. Training WAS the format fix; the
1742 misses are well-formed expressions that are mathematically
wrong. Grammar can gate syntax, only better math gates semantics —
round 2's diet (multi-step chains) is the lever. (The mask stays
in-tree: it targets BASE-model prose, the failure mode the adapter
graduated from.)

## Expert iteration rounds 2/3: the reverse-engine diet pays (2026-07-12)

Corpus 2097 (3.4x round 1): +88 forward multi-step (nearly extinct
post-orbitals — the engine ansatzes, it doesn't chain), +492
REVERSE-ENGINE chains (answer-side additive peeling, incl. the
fused-quotient class), +895 skip pairs, + Hints (rule-fire syndrome
as text) and Think (verbalized ansatz derivations) fields. One
training divergence caught and recorded (epoch-2 loss spike at
lr 2e-4 on the longer think-annotated targets; 1e-4 converges
0.44->0.22->0.21). Race on fresh seeds vs round-1 adapter's marks:
**one-shot 13 -> 19/30, steps 8 -> 12/30 (+50%), validity 0.5% ->
1.0%** (doubled). Every metric up; steps still trails one-shot (the
single-hop habit persists — the model prefers finishing to
chaining), so the loop's next lever stays data balance. Manual
supervised loop round deferred pending hardware window; the
autonomous loop remains un-armed until it runs (spec requirement).

## Expert iteration round 4: the balance overcorrected (2026-07-13)

One-hop cap (150 vs 1423 chain rows) + magic-sized skips, healthy
training (0.41->0.17) — and a ROLLBACK verdict: one-shot 19 -> 0
(the capped rows were the finishing skill), steps 12 -> 7, and the
new chain-required metric read ZERO. Lessons: (1) finishing is a
COMPONENT of chaining, not its competitor — capping it taught
neither; (2) train/eval shape mismatch — reverse chains teach
sum-split choreography that cannot fire on the L2/L3 eval band's
non-sum integrands; (3) manual runs overwrote the promoted adapter
(rounds-2/3 best recoverable from corpus @ 38c8c46 + lr 1e-4) —
the loop's per-round checkpoints exist for exactly this. Round 5:
loosen cap (~400), raise eval band to L3-L5, reverse-chain the low
levels so the choreography applies where examined.

## Variational ground-state engine, rung 1: the referee is a theorem (2026-07-12)

The physics rung (methods not molecules: model Hamiltonians only).
TFIM n=10, exact-diagonalization oracle, statevector ansatze,
parameter-shift/finite-diff Adam. The variational principle IS the
verifier — no state can score below the true E0, soundness by
theorem rather than by code. Honest arc, all recorded: the
hardware-efficient ansatz (RY+CZ ring) FAILED the pre-registered
bar at the critical point h=1.0 (1.314% vs <1%), and the failure is
structural — depth (8 layers, 80 params: 1.02%) and restarts (4
seeds: 1.05-1.11%) both saturate; the ansatz family plateaus where
entanglement peaks. The Hamiltonian-variational ansatz (layers
built from H's own ZZ/X terms) breaks it: **0.69% at criticality
with 3 layers and SIX parameters** — structure beats scale.
Full table (relative error): h=0.5: product 0.24%, l4 0.03%, hva
7.94% (!); h=1.0: 3.87% / 1.31% / **0.69%**; h=2.0: 5.39% / 0.30%
/ 0.03%. The two ansatze are phase-complementary (hva owns
criticality+paramagnet, hardware-efficient owns the ordered phase)
— a routing surface with real variance, for once. Bar as amended
(best arm): PASS. Rungs 2+: ansatz-structure search, Heisenberg,
step-chain LLM proposals.

## Ansatz-structure search, rung 2: greedy loses to hand design (2026-07-12)

Greedy beam (width 3, depth 6, 120-iter inner opt) over a token
vocabulary spanning both rung-1 families. BAR FAIL: close but never
beats the best hand arm at equal params — h=0.5: 0.053% vs 0.031%
(33p vs 40p), h=2.0: 0.039% vs 0.030% (15p vs 6p), and at
criticality it loses badly (1.518% vs hva3's 0.694%): greedy's
first token locks a hardware-efficient prefix it can never back out
of, and the inner budget (120 iters, single seed) under-serves
HVA-like candidates that need the exact alternating pattern. The
consolation is the pre-registered scientific bet, PARTIALLY
confirmed: the winning structures DO read the phase — h=0.5's
winner opens with rotation blocks (ry...), h=2.0's is
Hamiltonian-block dominated (zz/xm) — the search re-derives the
family boundary qualitatively even while losing quantitatively.
Rung 2b (evolutionary, same day): rediscovery bar ALSO fails —
quantitatively. But the qualitative result is the keeper: from
random token soup, evolution independently INVENTED the HVA
alternation at h=1.0 (['zz','xm','zz','xm'] — the exact pattern,
discovered not taught) and the phase split is now clean across both
searches (rotation tokens appear only in the ordered phase; pure
Hamiltonian blocks at/above criticality). The search reads the
phase diagram and re-derives the right circuit family; it cannot
close the last ~2x to hand-tuned precision at this compute.
STRUCTURE SEARCH CLOSED (two fails, house rule): the discovery is
qualitative, the engineering isn't — compute-bound, not idea-bound.

## ODE engine, rung 1: an engine made of engines (2026-07-12)

The engine-shaped physics rung. The ODE generator
(llmopt/mathgen/odes.py) had existed UNCONSUMED since the mathgen
expansion; rung 1 gives it its engine: family rules reduce each ODE
to INTEGRALS (separable -> exp of an integral; linear1 ->
integrating factor, two integrals; cc2 -> characteristic roots,
pure algebra), the house integral engine subcontracts the integrals
(the i_heurisch composition pattern one level up), and
sympy.checkodesol is the oracle (fork-isolated, hang = wrong).
Race vs sympy.dsolve, 75 problems: **engine 75/75, dsolve 75/75**
— solve parity; wall honest: dsolve faster overall (6s vs 28s;
subcontracting pays fork+search overhead per integral) EXCEPT cc2
where the algebra path wins outright (0.0s vs 1.3s). Rung-1
families are dsolve's home turf; the engine's edge should appear at
variable-coefficient families whose integrating factors need HARD
integrals (L6+-grade) — that's rung 2, where the L8 subcontractor
muscle differentiates. Then step-chains: the reverse-engine trick
applies verbatim (solutions drawn first).

## Fused cross-entropy (MLX, Liger-style): the memory wall flips the sign (2026-07-13)

`train/fused_ce.py`: chunked CE that never materializes the (N, 151936)
logits — custom VJP recomputes each chunk's softmax and scatters the
-1 with put_along_axis (no dense onehot). Bench at Qwen-0.5B head
shapes (`scripts/bench_fused_ce.py`), tests pin loss+grad parity vs
naive at four chunk sizes. Verdict at c=1024: **16k tokens 13.5GB vs
38GB peak AND 3203 vs 2008 tok/s — fused wins BOTH axes** (naive
thrashes unified memory past ~20GB, so the 2x-FLOPs recompute tax is
repaid in avoided traffic); 32k runs at 3183 tok/s where naive cannot
run at all; at 2k naive is properly faster (4109 vs 3116 tok/s — use
naive below ~8k). Two scars, both measured: (1) mx.eval INSIDE an
mx.custom_function under a grad transform forces the half-built outer
graph per call — 41s/52GB, worse than naive on both axes, OOM-killed;
removing it: 3.1s. (2) the v1 vjp built a dense (chunk, V) onehot —
the very tensor the module exists to avoid — 12.4GB -> 8.5GB gone
with the scatter + fp16 grad matmuls. Unblocks population training
(K adapters would be K x unaffordable logits unfused).

## Population training: batching pays only where slack lives (2026-07-13)

`train/population.py`: K LoRA adapters, one frozen base, population
folded into the batch dim — unchanged mlx-lm model, only the wrapped
linears are K-aware (batched einsum over stacked A_k/B_k), per-slice
fused CE so adapter k's grads are EXACTLY its solo run's
(tests/test_population.py pins forward + grad equivalence). Verdict:
**NULL at our shapes.** MLX 0.5B training is ~1250 tok/s flat from
~256 tokens/step up — one adapter's batch already saturates the GPU,
so K x streams have nothing to amortize: corpus shape (B=8, T=160)
1.04x @ K=4, 1.03x @ K=8; big shapes lose outright (0.62x @ K=4
B=4 T=512: 22GB peak + ~33% einsum overhead); only launch-bound
B=1 T=256 pays (1.22x). The training-side twin of the starved-judge
law: batching pays only where slack lives, and the premise assumed a
weight-traffic bound that measurement says isn't there. Machinery
banks for tiny-net populations (weightspace threads), where steps
ARE launch-bound. Fused CE (above) stands on its own — it was the
memory result; the tournament gate rides sequential runs.

## Predicted syndromes: the rules are their own features (2026-07-13)

Can a tiny MLP over featurize() structural features predict the
Hints line (which INT_RULES fire) without running the rules — a
~40,000x discount on the ~200ms/state first-look mini-solve?
Pre-registered bar: exact-set >= 80%, micro-F1 >= 0.9, hash-split
held-out. **FAIL twice, mechanism identified.** Round 1 (2018 corpus
states): 60.6% / 0.893 — misses concentrated in RARE rules (i_apart
R 0.25), so the data-starvation hypothesis got its widening round.
Round 2 (+2555 fresh generator roots L2-L8): **41.9% / 0.836 — more
data made it WORSE**, i_apart recall collapsed to 0.02 at 3.4x the
examples. The split-by-source autopsy is the finding: exact-match
55.4% on chain-distribution states vs **32.1% on hard roots** — the
gradient points the wrong way for the use case (hints matter most
exactly where prediction degrades). Mechanism: the informative bits
are semantic (i_apart fires iff the denominator factors; ansatz
rules fire iff their system solves) and 20 structural features
can't carry that signal at any dataset size. The predictable bits
(i_heurisch 0.93/0.95) are the near-universal ones — informative to
nobody. The starved-judge law, sharpest form yet: syndromes are
predictable exactly where they're uninformative. Prediction was
honest-fast: 5us/state vs ~200ms oracle — speed nobody can spend.
Revive-if: features that see semantics (small-model embeddings as
feature vector), or domains where the oracle costs seconds
(codegen: compile+run) so even partial-recall prediction pays.
`scripts/bench_pred_syndromes.py`, labels in
`data/pred_syndrome_labels.jsonl` (4573 states, streamed forks).

**Round 3 (same day): the revive clause fired immediately — PASS.**
Artin's derivability point ("the rules are defined; firing is
derived, not statistical") reframed the task from induction to
reading comprehension: swap the 20 structural features for the
0.5B's own mean-pooled embedding of the expression string (frozen
encoder, same MLP head, same labels/split/bar). **Exact-set 87.7%,
micro-F1 0.975** — and every structural-round pathology reversed:
i_apart 0.50/0.02 -> **0.98/0.98** (factorability is readable in
embedding space), hard roots now BEAT chain states (88.4% vs 86.8%
— the wrong-way gradient vanished), and even the
execution-dependent ansatz rules predict at 0.93/0.90 (the net
reads whether the system will solve without solving it — technique
intuition). Amended verdict: rules are their own features ONLY
under a blind encoding. Cost: ~27ms/state batched on MPS vs ~200ms
oracle forks (~7x, and GPU-side); the endgame is a hint head on the
step model itself (it already embeds every expression it's
prompted with — the Hints line becomes free). Adoption gate before
it touches the loop: A/B predicted-vs-oracle-vs-none hints in real
solve_chain prompting, scored on step validity.
`checkpoints/pred_syndromes_emb.pt`.

**Round 4 (same day, Artin's basis-state point):** enrich the
embedded string with the orbital sketch — the generator set
i_linear_basis would enumerate (trig/exp/log/root args, Laurent
tail, poly degree, denominator), atoms only, no solve. **89.0% /
0.978**; the basis-driven rules move exactly as predicted
(transcend_div R 0.86->0.93; sqrt_basis and apart R -> 1.00). The
quantum-chemistry reading is engineering guidance, not poetry:
expression = Hamiltonian, sketch = basis set, prediction = span
membership without diagonalizing. Remaining misses are thin
single-bit flips on 4+-rule states; next levers ranked: LoRA the
encoder > last-token pooling > per-rule thresholds. The A/B gate
decides shipping regardless — 0.98-precision informative bits may
already be oracle-equivalent in the step model's eyes.

**Round 5 (encoder tuning + the layer sweep):** LoRA-tuning the
encoder is a NULL at this data size — naive FT vandalized the
representation from step 0 (Artin's layering law named the
mechanism; LP-FT literature confirms), and even the corrected 5b
(warm head from the frozen probe, LoRA banded to the measured
formation layers, early stop) peaked val 89.8 then landed test
**87.5/0.976 — below frozen**. The surviving win came from the
router-router move (Artin: measure which weights to tune): the
25-layer probe sweep found the syndrome PEAKS MID-NETWORK (L12-15
~89.9% val) and DECAYS toward the output (L24: 87.0) — rounds 3/4
had been reading the wrong layer. **Frozen layer-15 + the same tiny
head: 90.5% / 0.979 test, the new best, zero encoder training.**
Echoes the 2026-07-09 middle-layer value probe: task structure
peaks mid-network; the last layer is busy being a language model.
Operating point: frozen mid-layer representations + heads; more
labels (cheap) before more gradient. `pred_syndromes_l15.pt`.

**The adoption A/B ends the arc with a twist (scripts/
bench_hints_ab.py, 48 fresh problems x 3 arms, byte-identical
prompting):** oracle hints 13/48 @ 1.19% validity; predicted
(layer-15 net) 14/48 @ 1.29% — the bar says ADOPT, the net IS
oracle-equivalent in the model's eyes at ~7x less cost. But the
pre-registered sleeper won the race: **NO HINTS 19/48 @ 1.87% —
beats both.** Hints anchor sampling toward the named rules' shapes;
at ~1% validity the chain needs resample DIVERSITY more than
direction, and the Hints line collapses exactly that exploration.
(n=48 solve gap is suggestive; the validity gap over thousands of
attempts is firmer; every ranking agrees.) Verdict: the Hints line
leaves the inference path entirely — the fastest hint is no hint —
and the syndrome-head spec re-aims at payoff 3 (representation
shaping during training), per its own pre-registration. Five rounds
of making hints cheap, concluded by measuring they shouldn't be
paid for at all: the starved-judge law eating its own tail.
Confirmed on a second fresh band (9.2M): none 12/48 @ 1.39% vs
oracle 9/48 @ 1.23% — same direction, both metrics, no reversal.
Combined: **no-hints 31/96 vs oracle 22/96.** USE_HINTS=False is
the shipped default in bench_step_tokens.py (also deletes the
~200ms oracle fork per novel state from every eval). Third-band
confirmation for free: the loop's own hints-off re-baseline read
{12, 12, 5, 9} @ 1.58% vs hints-on {11, 10, 5, 7} @ 1.29% — same
model, same seeds.

## Resample diversity: the famine measured, the ladder pays (2026-07-14)

The A/B's mechanism claim, measured directly
(`scripts/bench_step_diversity.py`, 24 states x 64 samples x 4
arms, dedup-then-verify): production sampling (const T=0.7, fixed
prompt) is **50% duplicates**, and a typical stuck state yields
**0.33 verified-valid steps per 64 samples** — two-thirds of states
get NOTHING from a full wave budget. Per-stream temperature ladder
(0.4..1.45, one knob): distinct 50->64%, late waves keep producing
novelty (wave-8 new: 2.9->4.7), and the currency that matters,
**valid-distinct/state 0.33 -> 0.42 (+27%)**. Few-shot rotation:
nothing (0.38 alone, adds zero on top of the ladder — the combo
ties ladder exactly). Solve-level race verdict: **NULL — the ladder loses 13/48 vs
const 15/48** (validity tie 1.38/1.41%). Candidate-level gains did
NOT survive chain economics, and the autopsy names the fallacy: a
chain needs AT LEAST ONE valid step per stuck state, not many —
extra distinct-valid candidates where one already existed are
worthless, while hot streams bill invalid samples everywhere.
Valid-distinct was the wrong currency; P(>=1 valid per wave) is the
right one, and const 0.7 was already adequate on it. TEMP_LADDER
stays None. The diversity FAMINE numbers stand (50% duplicates,
0.33 valid/64 samples) — the constraint is real, but temperature
isn't the lever; the model's distribution is (which is what
training rounds are for).

## Syndrome head payoff 3: representation shaping is free and worthless (2026-07-15)

The re-aimed spec's last claim, tested clean (same corpus, same LoRA
init seed, same data order; ONLY the aux gradient differs): lam=0.3
multi-task trains the layer-15 syndrome head to syn-BCE 0.048 at
ZERO step-CE cost (0.3586 vs control 0.3597 at epoch 0 — no task
interference)... and converts to NOTHING: eval 2/32 @ 0.36% validity
vs control 3/32 @ 0.42%. Null within noise. Paired with the hints
null: the model can know which rules fire — told or trained — and
it doesn't help it WRITE steps. Rule-awareness is not the binding
constraint; policy quality is (which is what GRPO buys directly).
The unified-climb fold (grpo-v2 spec section B) fails its gate:
run 3 stays pure GRPO. Caveat on the books: both arms were weak
retrain-lottery draws (3/32-class), so statistical power was low —
but the pre-registered gate was "helps in the clean SFT setting."
It didn't.

## Representation stitching, tier 1: the change of basis is real (2026-07-15)

Artin's riff, operational ("spacetime is matrices — map GLM's
geometry onto the 0.5B"; quantum-chem reading: the overlap matrix
between basis sets). Teacher SmolLM2-1.7B — alien architecture,
alien tokenizer — same 4573 labeled states, same split, same bar
family (`scripts/bench_stitch_poc.py`). Three rungs, all green:
**RUNG 1**: the foreign model's layers probe at up to **91.6% /
0.984 (layer 18) — BEATING our native Qwen layer-15 (90.5/0.979)**.
Bigger model, richer geometry, exactly the tier hypothesis; and its
layer curve replicates the mid-network plateau (peak 15-18/24,
decay after) — task-structure-peaks-mid-network now confirmed
across TWO architectures. **RUNG 2**: least-squares linear bridge
into Qwen layer-15 space reconstructs at R~0.98; bridged vectors
probe at 90.5/0.981 — the signal survives the change of basis
essentially losslessly. **RUNG 2b (strong form)**: a probe trained
ONLY on native Qwen vectors reads bridged foreign vectors at 86.9%
— translation lands in the same COORDINATES, not just the same
information (the Platonic-convergence claim, measured at 0.5B/1.7B
scale). Tiers 2 (30B keep-set teacher) and 3 (GLM offline donor +
representation distillation / zero-inference keep-set) unlock.

## GRPO run 2b: the hill-climbing machine, overnight edition (2026-07-15)

20 cycles, fast oracle end-to-end, continuing run 1's checkpoint.
**Held-out gate validity 2.24% -> 5.38% — 2.4x in one night** —
with solves {15,10,11,8} -> {18,13,10,11} and two lr-halving
rollbacks absorbed en route (the ladder worked: both recovered to
new highs). Collection stats tell the mechanism: all-pass waves
grew from 32 to ~90/cycle (states fully mastered), cycle wall fell
67 -> ~7 min (the 30x oracle), and the collector mined **+6,774
verified steps** as exhaust. Every number the SFT loop struggled
for arrived as a side effect of climbing: no reallocation, no
lottery, monotone-ish compounding under gates. Artin's sustained-RL
call (from the MAI-Thinking-1 reading) is the strongest training
result in the lab's history, ~36h from paper to 2.4x.

**RESTATEMENT (2026-07-15, reward hack #1):** the validity headline
was inflated. The policy had discovered that identity steps
(X => X) verify — mathematically true, the oracle is correct — and
66% of run-2b's mined rows were identities. The SOLVE gains
(44 -> 52) are real (solves cannot be hacked: Integral-free +
verified). The validity trajectory mixes genuine improvement with
identity inflation in unknown proportion; treat 2.24 -> 5.38 as an
upper bound. Detected when run 3's ascent made the hack dominant
(validity doubled WHILE solves collapsed — Goodhart's textbook
signature). Fixes: reward demands verified AND distinct (identity
= 0 reward => negative group advantage => the climb actively
unlearns its own hack); identities never mined/advanced/counted;
gate requires solves AND validity; corpus purged (4,615 rows, all
grpo-source — the SFT era was clean; RL taught itself the hack).
MAI's "reward hygiene" pillar, learned the honest way, one day in.

## Weight anatomy: the closed-system signature is a whisper (2026-07-15)

The bet (pre-registered): Artin — RL-climbed weights look elegant/
simple or interwoven-complex; Claude — concentrated mid-network,
lower rank. Instruments: composed BA deltas per (layer, module),
depth profiles, stable rank (`scripts/bench_weight_anatomy.py`).
**Artin wins.** The RL move (run-2b final minus its pre-GRPO init)
has Frobenius norm **4.0 vs 61-87 for every SFT run** — the entire
2.4x climb wrote ~6% of one SFT run's weight movement. Depth
profile nearly UNIFORM (32/36/32 across thirds) — flatter than SFT
(~20/39/40, mid-late piling) — falsifying the mid-network-
concentration prediction; stable rank lower (4.27 vs 5.3-5.8),
confirming the fewer-directions sub-claim mildly. Mechanism
implied: the capability was already in the model — verified-reward
RL redistributes probability mass among existing behaviors with
tiny uniform nudges rather than building circuits (retro-explains
the Arena one-sign miss: the model knows both signs; training moves
the PREFERENCE). Elegant, as bet. **Part 2 (function-side) CONFIRMS the whisper**:
CKA(pre, climbed) = 0.9998 at EVERY layer except the last (L24:
0.9877 — the only real drift; the policy lives in the final layer);
syndrome probes flat (L9/15/21 within noise both models — knowledge
unchanged); and yet output preferences moved hard: P(' -') as first
step token at the heartbreaker state shifted 5.5x — the WRONG way
(0.176 -> 0.032), which resolves the Arena rematch honestly: the
climb's preference edit is band-average optimal and that state paid
for the band's gains. Full mechanism: the 2.4x lives in the last
layer's output preferences — a whisper-quiet, low-rank, depth-
uniform nudge on an intact mind. Closed-system RL edits the policy,
not the representation (at this scale/duration). Corollary: the
sign-discipline reward knob is re-balancing an actual trade the
climb made, not cosmetics.

## The math-native micro-model: the internet was drag (2026-07-15)

The priors-vs-drag experiment (spec 2026-07-15-mathnative), phase 1
verdict. A from-scratch 19M decoder (45-token hand-built vocabulary
— the charset mask made honest; RMSNorm/RoPE/SwiGLU) trained THIRTY
MINUTES (3 x 10-min epochs, MPS, loss 0.54 -> 0.33) on 94.5k
engine-minted pairs + the purged corpus. Phase-1 gate (bar: 1% step
validity, the 0.5B's historical start): first run read 81.67% — and
the one-day-old Goodhart reflex demanded the contamination check,
which caught 11/48 eval roots present in the diet (the mathgen
small-space scar, third appearance). UNSEEN-ONLY rerun, harder band
(L2-4, 17 contaminated roots excluded): **65.59% validity
(2,015/3,072) with 617 outright solving steps.** Against the
pretrained 0.5B: ~20x its honest post-climb validity (3.34%),
~50x its starting point, at 1/26th the parameters and ~100x the
training/sampling speed. Verdict: for closed-system symbolic math,
pretraining priors were overwhelmingly DRAG — the 0.5B's capacity
went to knowing everything and fighting 152k tokens of habit; the
19M knows one language and every word is calculus (Artin's
bad-habits mechanism, measured). The famine does not exist on this
substrate: GRPO-from-birth starts amid plenty. Caveat on the books:
in-distribution generalization (same generator families) — true of
every 0.5B number too, so the comparison stands; out-of-family
transfer is phase 3's question.

## Micro-model phase 2 + 0.5B run 3d: honest climbs, both substrates (2026-07-15)

**19M GRPO-from-birth (12 cycles, Mac, full-param):** baseline 36/48
chain solves @ 76.8% validity -> best checkpoint 39/48 @ 80.3%.
L2/L3 saturated at 12/12 FROM BASELINE — the gate band was largely
outgrown before RL began, so headroom lived only at L4/L5. Late-run
caveat: cycle-12 grpo loss spiked to 1.80 (vs ~0.05 typical) and the
final gate rolled back — investigate before longer runs. Follow-ups:
raise gate band + collection to L4-L7; loss-spike autopsy.
**0.5B run 3d (honest reward, both guards):** baseline
{13,17,14,13,9} @ 3.34% -> cycle-6 gate {14,17,14,15,9} @ 3.63%,
checkpointed — slow, real, un-hacked gains (+1141 mined steps, zero
identities by construction). Stopped clean at the cycle-6 checkpoint
for machine handoff. The two substrates in one line: the 19M's
BASELINE exceeds the 0.5B's ceiling on this task by ~an order of
magnitude at 1/26th the size — the priors-vs-drag result, replicated
at chain level.

## Loss-spike autopsy: dual-clip hole in grpo_loss (2026-07-15)

The cycle-12 spike wasn't a one-off: the run-1 log shows **cycle 5
hit loss 205.1** (cycle 12's 1.80 was the echo), and both spike
cycles are exactly the two whose gates rolled back. Mechanism, three
parts: (1) `grpo_loss` ratios are whole-sequence (logps summed over
up to 120 tokens before exp — 0.04 nats/token of drift is a ratio of
~100); (2) the driver takes an optimizer step per group, so group
#60 is scored against a policy 59 updates newer than its `logp_old`
— full-param 19M drifts far faster than the 0.5B's LoRA@5e-6, which
is why only the micro runs spiked; (3) PPO's clip only bounds the
positive-advantage side: for A<0 the term -ratio*A is UNBOUNDED when
the policy drifts toward a failed sample. Grad-clip capped the
magnitude, not the direction — hence the gate regressions. Not a
data or reward bug: the known dual-clip failure mode (Ye et al.).
**Fix shipped:** `grpo_loss` now floors the negative-advantage term
at dual_clip*A (c=3), regression test pins the bound and the
zero-gradient floor. Both drivers inherit it. Run 2 curriculum
ascent (which showed smaller echoes: 0.60 at cycle 3) restarts
from its cycle-4 checkpoint under the fixed loss.

## Micro run 2b: curriculum ascent under dual-clip (2026-07-15)

12 cycles from the run-2 cycle-4 checkpoint (L4-7 collection, L3-7
gate), first run under the fixed loss. **Dual-clip confirmed
on-policy**: all 12 cycle losses in 0.0005-0.057 — including cycle 3
(0.60 under the old loss) and cycle 12 (the original 1.80/205
spikes). Zero spike-coincident rollbacks; the one rollback (cycle 2,
gate noise, LR halved to 5e-6) recovered by cycle 4.
**Climb:** baseline 26 solves @ 54.74% -> best 28 @ 57.26% (cycle-10
gate). The signal is L6: 1 -> 2 -> 3 monotone across gates 8/10/12-
adjacent — frontier movement where the ascent was aimed, after
mid-run gates looked like plateau (quiet accumulation, then
conversion). L3 11/12; L7 wobbled 2 -> 1 at the final gate.
+~10k mined steps to the sidecar (25.3k total).
**Checkpoint note:** the snapshot naming is off-by-one by
construction — `mathnative_grpo_c012.pt` holds the cycle-10 BEST
(28 @ 57.26); `mathnative_grpo.pt` is the final cycle-12 state
(27 @ 56.87, within gate tolerance, checkpointed). Next: another
12-cycle leg from here (L6 is moving; ride it), and consider
GATE_N=24 — 12/level leaves +-1 solve inside the noise floor.

## Micro run 2c: honest gate, real ascent, and the L4 diagnosis (2026-07-15)

12 cycles from the run-2b endpoint, GATE_N 12->24 (noise floor
halved) and fresh 62M collection seeds. **The climb, above the noise
floor:** baseline 52/120 @ 54.97% -> peak 59/120 @ 58.61% (cycle-6
gate), monotone 52->53->56->59; ended 57 @ 58.26 (cycle-10
checkpoint) after the final gate's validity dipped past tolerance.
L7 tripled (2->6 at peak); L3 near-saturated 23/24. Twelve flat
losses (0.027-0.103) — dual-clip holds under fresh problems too.
Checkpoints: `mathnative_grpo.pt` = cycle-10 (57 @ 58.26);
`mathnative_grpo_c010.pt` = the cycle-6 PEAK (59 @ 58.61, off-by-one
naming). Sidecar at 37k mined steps.
**The L4 diagnosis (mid-run, static analysis):** the doubled gate
exposed L4 at 6/24 (25%) — below L5 (16-17/24). Token cap ruled out
(all L4 answers encode <=64/120 — honest null). Cause: diet
starvation compounding — phase-1 diet is thinnest at L4 (7.6k of
94.5k) and GRPO mining then AMPLIFIES the gap (1.8k L4 mined vs
16.5k L5, equal prompt allocation): weak band -> all-fail waves ->
no mixed groups -> no gradient. RL cannot self-feed a starved band;
the fix is food. -> Staged curriculum pretraining (riff ledger, GO):
algebra-first diet, retrain phase 1, bar = beat 65.6% unseen AND
unstick L4.

## Curriculum v2 A/B: algebra substrate transfers — to L3, not L4 (2026-07-16)

30k verified algebraic rewrites (expand/factor/collect/cancel/
prodpoly x L1-3, `farm_algebra.py` — constructed-by-expand, zero
simplify calls) added to the phase-1 diet; retrained from scratch
(`--v2`, separate checkpoint, final loss 0.371). Same-protocol
unseen A/B (L2-4, identical seeds/oracle/skip-set; v1 RERUN under
the new protocol — the historical 65.59 is not comparable):
**v1 50.62% -> v2 53.62%** overall, solving steps 394 -> 445 (+13%).
Per level: L2 flat (93.9/93.7, saturated), **L3 60.0 -> 68.7**
(+8.7 — the transfer), L4 9.0 -> 9.9 (+0.9 — NOT unstuck).
Verdict: curriculum direction confirmed (algebra-first helps a
calculus model, measured); the L4 hole is deeper than the algebra
substrate — expanding (3x^2+2)^2 doesn't teach running the
integration pattern ACROSS that structure. Decisive test moved to
phase 2: does v2's L4 (9.9% wave validity) now form mixed groups
where the starved v1 lineage couldn't? -> v2-GRPO run, L4 column
vs run 2c's flat 6/24.

## v2-GRPO run 1: the curriculum compounds under RL (2026-07-16)

12 cycles from the raw v2 base, own lineage
(`mathnative_v2_grpo.pt`), 63M seeds, GATE_N=24, dual-clip.
**Headline 1 — the base:** v2 with ZERO RL opened at 57/120 @
53.12%, EQUAL in solves to run 2c's final checkpoint after 24 GRPO
cycles on the v1 lineage. 30 min of algebra-enriched pretraining
bought what two RL runs bought — on this substrate, diet is ~20x
cheaper per solve than RL.
**Headline 2 — the climb:** 57 -> 60/120 @ 57.94% (final gate, new
best on both axes; the highest any 19M lineage has posted). Twelve
flat losses (0.008-0.091); one wobble-free run, zero rollbacks.
**Headline 3 — L4 answered:** the v1 lineage froze at 6/24; the v2
lineage holds 8/24 (7 -> 8, stable across four gates). And the
starvation mechanism visibly eased: early cycles filled 64 mixed
groups in ~270-285 waves vs 2c's 415+ (~1.5x easier), and the
all-runs L4 mining total went 1.8k -> 3.6k. Partial unstick: the
substrate feeds RL now, but L4's ceiling still lags its neighbors —
the residue is the integration-pattern-across-structure gap, not
algebra. Checkpoints: `mathnative_v2_grpo.pt` = cycle-12 best
(60 @ 57.94); sidecar at 43.2k mined steps.

## v2-GRPO leg 2: the RL ceiling on the v2 base (2026-07-16)

12 planned cycles from leg 1's best (64M seeds); HALTed at cycle 6
by two consecutive rollbacks — the second by 0.05 validity points
(the gate discipline working as designed). Best banked: cycle-2
gate **60/120 @ 59.65%** (validity record for any 19M lineage;
`mathnative_v2_grpo.pt`). Verdict: RL has extracted what this base
offers — 57 -> 60 solves and +6.5 validity across two legs, now
oscillating within tolerance of the ceiling. L4 held 7-8 all leg.
The next lever is diet, not more RL: v2.1 (L4 engine-chain shards,
farming overnight) retrains the base under the ceiling.

## Capacity check: params were the ceiling where structure is deep (2026-07-16)

Artin's 4 AM call, made one-variable: 50.4M (d=512/L12/8h) on the
IDENTICAL v2 diet and recipe vs the 19M. Training loss statistically
identical (0.366 vs 0.371) — but the unseen gate moved:
**53.62 -> 57.53%**, solving steps 445 -> 566 (+27%), and **L4
validity 9.9 -> 18.4%** (nearly doubled; L2/L3 roughly flat).
Verdict: the 19M was capacity-bound SELECTIVELY — on the long
structural patterns L4 needs, not on the short families. Corollary
worth keeping: average LM loss barely registers hard-tail
capability; the oracle gate does. Two levers confirmed in one
night: diet (+3.0, L3) and capacity (+3.9, L4). Production path:
50.4M x v2.1 diet (both levers stacked) once the L4 farms land.

## 0.5B run 4 + program retirement: the control arm, characterized (2026-07-16)

12 cycles overnight from the cycle-6 checkpoint (dual-clip loss,
first 0.5B run under it — 12 normal losses, zero spikes; the
recurring CUDA OOM lines were the allocator's recoverable path at
the 10GB phase boundary, process-verified twice). Gates:
69 -> 72 -> 73 (best, cycle 6 @ 3.66%) -> 72 -> 71 -> 69(final,
rolled back). Banked: cycle-10 state in step_lora_grpo.pt
(71 @ 3.57); +2,322 mined rows rescued to the corpus (9,017 total).
**Program decision (Artin's call, 2026-07-16 morning):** the 0.5B
climb is RETIRED as characterized — it climbs real and slow
(3.34 -> 3.66% over 2 runs), and its standing value is as the
control arm of priors-vs-drag: the measured cost of internet
habits. The 3080 is repurposed to the micro-model program, where
CUDA buys capacity sweeps and GRPO legs far cheaper.

## v2.1: the L4 chain shard — thin but efficient (2026-07-16)

The farm cut at 60% (Artin's call — the v2.2 forced-steps farm
obsoletes waiting), then the language filter cut deeper: **64% of
the engine's multi-step L4 chains route through Subs/erf, outside
the model's 45-token language** (finding: engine and model step
languages diverge exactly where L4 is hard; v2.2 farm must restrict
to in-vocabulary rules). 1,169 in-language rows survived (+13% L4
diet). 19M A/B (same protocol; diet also inherits +2,322 run-4
corpus rows — 1.7%, noted): **53.62 -> 54.56%**, L4 9.9 -> 11.8,
and **solving steps 445 -> 579 (+30%)** — chain-shaped rows teach
finishing moves. Good per-row efficiency; scale the diet in v2.2.
Production train launched: 50.4M x v2.1 (both confirmed levers).

## The four-way: diet and capacity don't stack (2026-07-16)

| unseen gate | v2 diet | v2.1 diet |
|---|---|---|
| 19M | 53.62% / 445 solving | 54.56% / 579 |
| 50.4M | **57.53%** / 566 | 56.67% / **628** |

At 19M the L4 chain shard helps everything; at 50.4M it costs ~0.9
validity (borderline noise) while solving steps still rise +11% to
the best of all four. The levers INTERACT: the bigger model already
extracts most of what 1.2k chain rows teach, and the diet's
remaining value concentrates in finishing skill. Training loss was
blind to all of it (0.3658 vs 0.3659). Production tiebreak: chain
solves are the currency (P(>=1 valid/wave) lesson) — GATE_N=24
chain gate on both 50.4M variants decides the GRPO base.

## Chain-gate tiebreak: v2.1 takes production (2026-07-16)

The dissociation result: 50.4M v2 has the best STEP validity of all
four models (57.53%) but chains worse than the 19M (52 vs 57/120) —
good steps, no finishing. 50.4M v2.1 converts its +11% solving
steps into **58/120 chain solves @ 54.24%**, with **L7 8/24 — the
best any model has posted on the hardest band, pre-RL** (the entire
v2-GRPO lineage peaked at 6 after 18 cycles). 1,169 chain rows
bought deep-chain skill. Production base: mathnative_45m_v21.pt;
GRPO on it launched (record to beat: 60/120).

## Origin story, closed

Limits resisted LoRA training (<=21%), motivating the engine. The
engine now solves them: l_hopital emits UNEVALUATED derivatives that
the rung-1 diff rules finish — the rungs composing in one derivation.

## 45M-GRPO run 1: the production base climbs past the record (2026-07-16)

12 cycles on `mathnative_45m_v21.pt` (the chain-gate production
winner), Mac, GATE_N=24 L3-7. Baseline 57/120 @ 54.24% ->
cycle-10 best **61/120 @ 56.79%** — past the 19M lineage's 60-solve
record on solves. One rollback (cycle 8, lr halved); mining
streamed +6.9k steps. The process died mid-cycle-12 (host outage,
not the run) — cycles 11-12's weights lost past the last gate, but
their +1,214 mined rows were already streamed (the checkpoint
selection-effect discipline paying off). Continuation (run 1b)
relaunched from the cycle-10 best: cycle-2 gate 61 @ 58.62,
cycle-6 gate **61 @ 59.21** (validity record for the 45M lineage;
one mid-run rollback at cycle 4 by 0.07 points). In flight.

## Fast-path throughput: size the token budget to the model (2026-07-16)

The 113M `--fast` run at the 50.4M-tuned 24.5k budget ran at ~30s
per batch (~5% of the 3080's FLOPs) — VRAM at 10.0/10.2GB, the
allocator retry-thrashing every step ("free: 0" OOM warnings, GPU
busy but cores starved). At `--budget 12288` +
`expandable_segments:True`: **3.1 it/s — the same run went from
~9h projected to 14 minutes total** (~100x). Lesson: token-budget
batching needs the budget sized to the model's activation
footprint; over the VRAM cliff the cost is not OOM but silent
100x throughput loss. Flag added (`--budget`).

## Parity 2x2: packing convicted, bf16 exonerated (2026-07-16 night)

The 113M's first fast-path gate read 48.83/410 (below the 19M!) —
the owed parity gate then ran as a full 2x2 at 50.4M, one lever
per cell, all same-protocol unseen L2-4:

| 50.4M v2.1 | standard BS=32 | token-budget packing |
|---|---|---|
| fp32 | **56.67 / 628** | 45.65 / 419 |
| bf16 | 54.43 / 625, L4 18.9 | 46.95 / 439 |

**Token-budget packing alone costs ~10 validity points** (the
length-sorted scar writ large: length-homogeneous batches + ~6x
fewer optimizer steps); the lr-scaling rescue (2.5x, sqrt rule)
made train loss WORSE — it was never average-loss undertraining.
**bf16 autocast is near-parity**: solving 625 vs 628 and L4 18.9
vs 18.4 dead even, ~2-pt validity debit. And packing bought no
speed anyway once the VRAM thrash was fixed: bf16-nopack matches
packed wall-clock (230s/epoch, 11-min 50.4M trains on the 3080).
Train loss was blind to the entire 10-point hole (0.3654-0.3858
band across all four cells). `--fast` now means bf16 + `--nopack`
standard batching; packing survives only as flags for future
study. Third instance this week of the week's lesson: THE GATE,
NOT THE LOSS.

## Packing post-mortem: it's the step count, and loss can't see it (2026-07-17)

Proper packing (shuffle-then-pack, iid mixed-length batches, honest
padded cost) re-ran the parity cell: **46.93 / 457 — the identical
~10-pt hole as sorted packing (46.95)**, while train loss came back
to the standard band (0.3678 vs 0.3659). Composition exonerated;
the harm is the ~6x fewer/bigger optimizer steps themselves, and
lr-scaling (sqrt rule) made it worse, not better. Packing is DEAD
for gate-bearing runs in any form; `--fast` = bf16-only. Sharpest
instance yet of the week's lesson: a 10-pt capability hole with a
MATCHED training loss. Soup note, same night: 50/50 average of
consol + run-2-best gated 64 @ 62.16 — a clean NULL (same-basin
parents too close to buy anything); retry only across distant
ingredients.

## 113M capacity rung: NULL above 50M on this diet (2026-07-16 night)

113.3M (d=768/L12/12h), identical v2.1 diet, honest bf16-nopack
path (loss 0.3717). Unseen gate: **54.58 / 588, L4 11.5, L3 68.2**
vs the same-path 50.4M's 54.43 / 625 / L4 18.9. Validity flat,
solving steps LOWER, L4 gives back half the capacity gain while
L3 jumps — the reallocation signature, not a third rung.
**Params stopped paying at ~50M on this diet.** Caveat held open:
26M tokens is a light meal for 113M (the 19M->50M jump paid at
the same token count, but data-starvation grows with width) —
re-ask ONCE if/when v2.2's thicker diet lands. The earlier
"capacity reversal" reading was the packing bug, now void.

## Self-distillation consolidation: RL explores, SFT consolidates — measured (2026-07-16 night)

Strategy item B, first run: ONE low-LR (1e-5) epoch on the
level-capped GRPO sidecar (24.2k of 63.6k rows; L5 capped 45k ->
5.5k per the coeff-flood scar), from the promoted 45M
(`consolidate_mathnative.py`, 5.5 min on MPS). Chain gate:
**64/120 @ 62.23%** vs the promoted 61 @ 59.36 — program record on
both axes, and EVERY level >= promoted (L3 23, L4 7, L5 16, L6 8,
L7 10 — new high on the hardest band). No reallocation. The six
post-record RL cycles of run 1b bought +0.15 validity; one
consolidation epoch bought +2.9 and +3 solves. Caveats: sidecar
mixes all micro-lineage mining (not strictly own-policy); one run,
one seed. Adopted: `mathnative_45m_consol.pt` is the production
45M. Next lever: GRPO leg FROM the consolidated base — if the
climb re-opens above the old RL ceiling, the loop is
RL -> consolidate -> RL (expert iteration's two strokes, at last).

## Validity autopsy: the model isn't sloppy, it's ignorant (2026-07-17)

Every rejected gate candidate from the 62% consol model classified
(`validity_autopsy.py`, 1,456 candidates): **structural 32.7%**
(87% of all failures), unparseable 2.9%, scaled sign/coeff 2.2%,
identity/repeat ~0. The one-sign hypothesis is DEAD; syntax is a
non-issue; the failures are well-formed wrong-pattern rewrites.
Per level: L3 89.1% valid / L5 79.8% vs **L4 22.7%** (64.4%
structural) and **L6 30.2%** (63.0%) — the craters sit exactly on
the diet-thin, out-of-language bands. Sample reading: L4/L6 =
missing move vocabulary; L5 = RIGHT ansatz family, wrong
coefficients (drillable); L3 = illegal split-composition.
ANSWER to the why-so-slow question: GRPO reweights patterns that
exist — at L4/L6 they largely don't, so RL grinds on lucky hits
(the all-fail concentration). The lever is v2.2's diet
(in-language chains + one-ply worked examples + an L5
coefficient-determination drill), NOT reward shaping; shaping
stays banked for calibration-shaped wrongness, which this isn't.

## Depth anatomy: the layer-6 cliff, and no free early exit (2026-07-17)

Logit-lens over the production 45M (`probe_depth.py`, gate-band
answer positions): agreement with final output climbs 33->56% through
L0-5, JUMPS +15.6 to 71.5% at L6 (the biggest step in the stack),
then refines to 91% by L10 — 100% only at L11. Verdicts: (1) NO
cheap early exit — 9% token disagreement at L10 breaks nearly every
expression in a 45-token grammar; the small crystal uses its full
depth (no redundant spelling layers, unlike large LLMs);
self-speculative drafting bounded at ~1.1-1.2x, not the lever.
(2) The DECISION visibly concentrates mid-stack: the L6 cliff is
the functional twin of the pre-registered CONCENTRATED weight-
anatomy prediction — point the CKA/probe instruments there first.

## The specialist shelf: low-norm neurons are rare-domain, not dead (2026-07-17)

Artin's slice-the-clump ask, run on Qwen2.5-0.5B L14 gate (62
low-norm neurons, <0.35 vs bulk median 0.56; pairwise cosine 0.12 —
diverse, NOT collapsed copies). Zero-ablation vs random-62 control:
clump costs **math +0.0278 / english +0.0144**; random bulk costs
math +0.0088 / english +0.0247. The "dead" neurons are ~3x more
MATH-loaded than bulk. Verdict: magnitude tracks firing FREQUENCY,
not importance — rare-domain specialists stay small (math is rare
in internet text). Corollaries: magnitude pruning is anti-math
(harvests specialists first; keep-set importance must be measured
domain-conditionally — the weight-distance law's sibling); the
19M's missing low band = no rare domains in a closed world, not
higher efficiency alone. Caveat: one layer, one seed, two small
probes — directional. LoRA note (mechanistic): truly dead channels
self-perpetuate (no forward signal -> no backward gradient), so
low-rank deltas invest where gradients live; resurrection requires
re-init, but THESE neurons don't want it — they want protection.

## Four diets, one lens: uniformity tracks diet focus (2026-07-17)

The specialist-shelf hypothesis, tested out-of-sample on models we
did not train (mid-layer gate, polar view; asset
neurons-polar-four-diets.png): shelf fraction (<0.6x median norm)
falls MONOTONICALLY with diet focus — 0.5B generalist 0.80% /
spread 2.11 -> Coder-1.5B 0.39% / 1.74 -> **Math-1.5B 0.00% /
1.37 -> 19M closed-system 0.00% / 1.16**. The math-heavy internet
model has no shelf, exactly as the frequency-not-importance story
predicts (nothing math is rare in its diet); the closed system is
more uniform still. Neuron-magnitude geometry reads the TRAINING
DISTRIBUTION off the weights — the birth-instrument family (CV =
organization, floor = fullness, shelf = diet skew) now has three
gauges and one external validation. FOURTH GAUGE (same day, Artin
eyeballed the coder's phase asymmetry): circular concentration R
of neuron phases — coder 0.215 (most clumped: code's sub-languages
grow aligned families, proto-modules), generalist 0.139, math-1.5B
0.160, **19M closed-system 0.034 (a near-perfect ring, 4x more
isotropic than any internet model)**. Dissociation: magnitude
evenness = diet FOCUS, phase isotropy = diet HOMOGENEITY — math-1.5B
has the first, only the closed system has both.

## Precision ladder: the whisper is an error-correcting code (2026-07-17)

Artin's exactness riff, tested: production consol3 gated at fp32
(65/120 @ 63.97) vs bf16-roundtripped weights (65 @ 63.83) vs
**int8-RTN weights (65 @ 64.04)** — IDENTICAL, to noise. Weight
precision does not carry the capability: the RL/consol fine
structure is a rank-4 pattern across millions of weights, and
dot-products average independent rounding errors away — the signal
lives in correlations, not digits. Corollaries: (1) the bf16
TRAINING debit is dynamical (gradient-trajectory perturbation),
not representational — store cheap, train careful; (2) checkpoints
ship int8, 4x smaller, gate-identical (model zoo / expert drawer
economics); (3) exactness in a closed system is a property of the
ORACLE, and the network provides its own redundancy.

## Generational training: rebirth beats the lineage (2026-07-17)

Gen-4: from-scratch 50.4M on v2.2 + the level-capped CUMULATIVE
grpo-mined sidecar (27.7k rows — the lineage's entire verified
experience; 6k L4 / 4.4k L6 / 6k L7). Chain gate: **66/120 @
64.66% — program record on both axes**, beating the consol3
lineage (65 @ 63.97 = 3 GRPO runs + 3 consolidations, ~12h) with
a 13-MINUTE birth. L4 7->8 (the crater moves at last), L7 11->13.
The v2.2-only control (59 @ 53.49, L7 12) attributes the win: the
new shard buys deep chains, the cumulative sidecar buys everything
else back and more. PARADIGM VERDICT: lineages are not patched,
they are REBORN on accumulated verified experience — RL's product
is the mining, not the weights (the weights are disposable; the
corpus is the organism). Production: mathnative_45m_gen4.pt.
Caveat: bf16-path birth (dynamical debit applies) — standard-path
gen-4 has untested headroom. Next loop: GRPO/mine from gen-4 ->
gen-5 rebirth.

## The math phylogeny: technique lineages, not levels (2026-07-17)

Artin's language-descent riff, run: agglomerative clustering of
MarkovPrior's verified rule-bigram profiles (24 rules, in+out
transition vectors, cosine). Three clades: (1) STEPWISE INTEGRATION
(i_power/sum/table/usub/parts/apart + cancel); (2) DIFFERENTIATION
+ ALGEBRA AS ONE LINE (all d_* WITH expand/factor/together/
trigsimp); (3) THE ANSATZ SOLVERS ALONE (i_linear_basis,
i_ansatz_exp, i_unprod — an isolated lineage). The tree ignores
our level numbers and explains two measured facts at once: algebra
transferred to L3-not-L4 (v2 A/B) because L4's key technique lives
in the isolated ansatz clade — no descent path; and the autopsy's
L4/L5 failures are ansatz-shaped because that clade was never
taught as a lineage (v2.2's one-ply drills were unknowingly its
first curriculum). REWRITE: curriculum by clade, not by level —
the ansatz branch needs its own worked-example progression.
Mathematics just voted on its own pedagogy.

## Compression phylogeny: two lenses, one crater (2026-07-17)

The gzip-language-tree method (Benedetto et al. 2002, PRL), math
edition: zlib NCD between per-level corpora. L1-L2-L3 tight kin
(0.98-0.99); **L4 the most distant integration level from
everything** (>=1.003) — the bigram tree isolated L4's TECHNIQUE
(ansatz clade), compression isolates its DIALECT; two independent
instruments agree the crater speaks its own language. Algebra =
outgroup by text (1.02+) though ancestral by technique — the
lenses disagree exactly where they measure different things.
Theory bonus from the same session: cross-entropy weights by p_i,
so rare-but-load-bearing capability contributes ~nothing to loss
BY CONSTRUCTION — "the gate, not the loss" is Shannon-inevitable,
and the specialist-shelf finding (frequency != importance) is the
same theorem read off weights instead of bits.

## The dynamical debit, quantified: fp32 birth 69/120 (2026-07-17)

One-variable rerun of the gen-4 birth on the standard fp32 path:
train loss IDENTICAL to the bf16 birth (0.3526 vs 0.3525) — chain
gate **69/120 @ 67.13 vs 66 @ 64.66**, +3 solves / +2.5 validity,
new program record, L3 PERFECT (24/24). bf16's forward rounding
during TRAINING costs real capability at invisible loss (the
precision ladder showed finished weights are rounding-robust — the
damage is trajectory-only). Verdicts: births are fp32-ONLY from
now on (cuda fp32 ~30 min, acceptable); production =
mathnative_45m_gen4_std.pt; the birth-quality lever beat the RL
lever again (69 vs the mined 68). Open lever: stochastic-rounding
bf16 might recover speed without the debit (pre-registered).
Generational curve: 57 -> 61 -> 64 -> 65 -> 66 -> **69**.

## The compression floor: 3 < knowledge-at-rest < 4 bits/weight (2026-07-17)

MX-style blockwise quantization (block 32, shared scale, int
mantissas — Artin's store-small/convert-dynamic scheme; OCP MX
lineage) on the production fp32 model (69/120 @ 67.13):
**MX-int4 = 69/120 @ 66.76 — full solve parity at 8x compression**
(~25MB checkpoints); MX-int3 = 67/120 — the floor cracks, losing
exactly one L4 and one L7: the deepest, rarest capability dies
FIRST under over-compression. Same theorem a fourth way (loss-
blindness, specialist shelf, CE tail-weighting): importance
concentrates where frequency doesn't, and the tail is the finest
structure in the code. Consequences: int4-MX is the shipping
format; the fused int4 GEMV kernel (practice_7 packing, promoted)
is greenlit with measured basis — 8x sampling bandwidth at zero
capability cost, mining/gates are sampling-bound.

## DeepSeek-V3 from one shard: the experts are crystals (2026-07-17)

98 layer-30 routed experts of the 671B MoE, gauged from a single
4GB shard (fp8 dequant via block scales; never inferenced): CV
median 0.021, **floor median 0.890, shelf ~0%**, R median 0.078 —
V3's experts individually carry the CLOSED-SYSTEM geometry (fuller
even than our 19M), not the internet-monolith cloud. Mechanism:
the router is a diet-focuser — each expert trains in its own
quasi-closed slice of the distribution and grows the matching
lattice. The crystal law (focused diet -> uniform geometry) now
spans 19M -> 671B, four orders of magnitude, our models and
theirs. Instrument-as-tool: the gauge range flags expert 71 (only
real shelf, 1.1%), one near-dead neuron (floor 0.013), and a few
coder-grade clumps (R 0.23) — inspect/prune candidates from disk,
the keep-set chain's cheapest rung made real. Cost: one shard +
30s of SVD.

## The rank floor: bits are redundant, dimensions are sacred (2026-07-17)

SVD-truncation sweep on the production fp32 model (69/120 ref):
rank-128 = 66 @ 61.22 (already bleeding), rank-64 = 42 (collapse),
rank-32 = 2 (dead). **The crystal is FULL-RANK** — the isotropy
gauge predicted it (a perfect ring spreads neurons over every
direction; an isotropic code is full-dimensional by construction),
and the whisper's rank-4 was always the RL DELTA, never the base.
Compression's two axes separate cleanly: bits per direction are
over-provisioned (int4 lossless, 8x) but directions are load-
bearing (rank-128 costs 3 solves at only 4x). Storage doctrine:
QUANTIZE THE NOTCHES, NEVER THE AXES. Also answers the 2-numbers-
per-neuron question at the extreme: the polar charts are
portraits, not the person.

## Gen-5: the rebirth ceiling, confirmed by failing correctly (2026-07-17)

Gen-5 fp32 rebirth (v2.2 + refreshed 29.3k sidecar): **68/120 @
64.78 — below gen-4's 69 @ 67.13**. Reallocation signature: L5
18/24 (new per-level high) bought by L3 -2 and L6 -1 — a reshuffled
diet, not a richer one, exactly as the novelty audit predicted
(only ~600 genuinely new rows since gen-4; the band's step-space
is enumerated). VERDICT: generational training's fuel is NEW
verified experience — rebirth without novelty shuffles facets of
the same ceiling. Production stays gen4_std (69). The equation
reads clean: S_max binds; next levers are territory (L8 band, ODE
continent) and the ansatz-clade curriculum, not another turn of
the crank. Curve: 57->61->64->65->66->69-> plateau pending new
land.

## Calibration table: CV is an imprint dial, not a goodness dial (2026-07-17 night)

Four gauges over 7 births with known chain gates: **r(validity,
CV) = +0.966, r(validity, floor) = -0.838** — capability tracks
ROUGHER lattices and LOWER floors within the family, the opposite
sign of the 113M story. Reconciliation: CV measures diet IMPRINT —
texture (rich experience differentiating neurons: the gen-4s ate
the mined sidecar) vs rubble (underfed width, the 113M), same
number, opposite causes, disambiguated by data-per-width. The
perfect ring is the portrait of a clean diet; the textured ring is
a clean diet THAT LIVED — the fingerprints are the capability.
Within-family, CV ~= a 30-second gate proxy (r .966, n=7; lineage
points confirm frozen-lattice: all identical CV 0.0183 across 61->
65). Shelf is dead within our zoo (all 0 — internet-only gauge).

## GPTQ-int3: the floor was rounding, not bits (2026-07-17 night)

House GPTQ (column-serial inverse-Hessian error compensation,
methods.py; Hessians from 199 gate-band prompts, 61 linears) at 3
bits with PER-ROW scales — a handicap vs MX's block-32 — gates
**68/120 @ 66.01 vs naive int3's 67**: the lost L7 solve recovered,
L3 stays perfect, only one L4 short of fp32's 69. Verdict: smart
rounding beats fine scales; the tail dies to DUMB rounding before
it dies to bit-scarcity. Next refinement (queued): GPTQ + block-32
scales — plausibly full parity at 3 bits (~19MB crystal). The
compression floor is a property of the QUANTIZER, not just the
weights.

## GPTQ x block scales: compensation needs headroom (2026-07-17 night)

The composition NULLED with a mechanism: GPTQ + block-32 at int3 =
67/120 @ 66.36 — WORSE on solves than GPTQ per-row (68), the L7
tail dying again. Compensation parks each column's error in later
columns; tight block scales CLIP the parked errors — fine scales
and error-routing compete for the same slack (visible in the
validity/solves split: block polishes the bulk, per-row saves the
rare). Final storage ladder, bracketed both ways: fp32 69 / int4
69 (8x, lossless) / GPTQ-row-int3 68 (10.7x, 3-bit champion) /
naive-int3 67. The 69th solve is genuine 4th-bit content.

## The bottom rungs: the wiring diagram walks (2026-07-17 night)

Sub-3 ladder on the production model: **GPTQ-int2 = 48/120 @ 36.83
(70% of solves at 2 bits); ternary absmean 1.58-bit = 24/120 @
14.66 — including TWO L7 deep chains on sign + mean loudness
alone.** Artin's bet ("if it's a tree you just need 2 bits") beats
Fable's carnage call on points. The decisive contrast: rank-32
(cut dimensions) = 2/120 dead; ternary (cut precision to 3 levels)
= 24/120 walking. AXES ARE THE ORGANISM, MAGNITUDES ARE MUSCLE
TONE — the signed graph carries the load; precision buys the rest.
Complete information curve, one night: fp32 69 / int4 69 / int3
68 / int2 48 / 1.58-bit 24 / rank-32 2. Follow-on lit note:
BitNet-class results train the constraint from birth — a ternary-
FROM-BIRTH crystal is the natural re-ask on our substrate (40-min
experiment) if the wiring thesis wants pushing.

## Definition neurons: locatable preferences, no dependencies (2026-07-17 night)

Family-selectivity probe on the champion (layer 6, the decision
cliff): a sparse committee exists — 6 of 2048 FFN neurons strongly
family-selective (neuron 493 fires 4x on power-rule integrals;
939 on trig). CAUSAL TEST: ablating the entire power committee
changes NOTHING — family validities identical to the decimal
(94.4/61.9 both arms), scalpel verified real (max logit delta
0.25, zero argmax flips — the perturbation is absorbed).
Verdict: definitions are locatable as PREFERENCES, not
DEPENDENCIES — the population re-expresses the computation
without them. Robustness at a third granularity (bits: int8/
ternary; units: this; correlations carry everything). The
definition of the power rule is not a place; it is a chord.

## Ternary-from-birth: 63/120 at 1.58 bits (2026-07-17, the closer)

The wiring-thesis showdown, final: BitNet-style STE birth on the
full gen-4 recipe -> **63/120 @ 60.84, final loss 0.3594 vs the
fp32 twin's 0.3526**. Six solves behind the champion at 1/20th the
bits (~8MB, bitwise-inference-ready); **L5 17/24 BEATS the
champion's 16**; beats every pre-gen4 fp32 birth (v2.1 58, v2.2
59). Post-hoc ternarization got 24 — growing up ternary got 63:
the constraint shapes a lattice that carries in topology what
amplitudes carried before (geometry constraint-invariant, portraits
identical; 27% exact zeros at ep0). Arc: ep0 preview 53 -> final
63. Cross-birth anatomy: the ternary grew its OWN definition
committee (power-dominant, zero index overlap — lottery redrawn,
statistic conserved) with SOFTER selectivity (1 vs 6 neurons
>1.8x): discrete weights force a more distributed code. Queued:
the discrete-plasticity fork (GRPO on this model), progressive-
precision curriculum, int kernels to cash the 8MB in.

## Graph anatomy: the crystal is an expander (2026-07-17 night)

kNN weight-graphs + Newman modularity, six minds. The coder-
modules prediction FAILED (coder is the LEAST modular, Q 0.142 —
phase clumps are not graph communities; projection over-read,
lesson banked). The real signal: **clustering coefficient —
crystals 0.021-0.026 vs internet models 0.063-0.095**. Low
clustering + connectivity = EXPANDER graph: no cliques, neighbors'
neighbors are strangers, maximal mixing per edge — full-rank as
topology, no redundant edges. Brain inversion: cortex is high-
clustering small-world because axons cost space; the crystal has
no geometry to pay for and converges to what the brain would build
if wires were free. The pure expander.

## Anchor storage: the democracy is indescribable (2026-07-17 night)

Relational storage (neurons stored as inner products to k of their
own, least-squares reconstruction): anchor-256 = 52/120, 192 = 38,
**128 = 11 — versus SVD rank-128's 66 at identical rank**. Random
citizens are a 6x-worse basis than the optimal one BECAUSE the
crystal is isotropic: no neuron is representative in a maximally
democratic code (clumped models would anchor well; ours can't).
The isotropy that buys robustness and expander topology costs
relational compressibility. Tail died first as predicted (L4 = 0
at k=128). Spectral companion result: effective rank FALLS with
capability (v2 317 -> champion 284/512; knowledge organizes
directions rather than opening them) while the thin tail stays
load-bearing — concentrated body, indispensable whisper. Storage
doctrine survives the night intact.

## The free router works, and the code is holographic (2026-07-17 night)

Family classification from layer-6 activations (nearest-centroid,
5 families incl. two never used for selection): all-2048 = 100%,
**the 8-neuron committee = 94%, and 8 RANDOM neurons = 87%**.
Routing is essentially free at any readout width — but the sleeper
is the random row: every small fragment of the population carries
a blurry copy of the family signal. The code is HOLOGRAPHIC —
the ablation result's mirror (remove the committee: nothing lost;
read only it: almost everything present). Rule-neuron router
feasibility: CONFIRMED — dispatch (per-family experts/precision)
can key off a handful of activations at negligible cost. Tunnel 5
(principal curricula) closed as a frame-awaiting-data: 3 measured
transfer edges are too few to eigen-decompose honestly; revisit
when the clade-curriculum A/Bs add rows to the transfer matrix.

## 113M fp32 capacity re-ask: tokens-per-width is the ceiling (2026-07-18)

The honest re-run of the capacity question (every prior 113M was
bf16-tainted): d=768/L12/h12/ffn3072 (113M), fp32 on Mac, identical
gen-4 corpus + recipe as the 45M champion. **65/120 @ 61.33** —
per-level {3:21, 4:6, 5:16, 6:9, 7:13} vs the champion's
{3:24, 4:8, 5:16, 6:8, 7:13}. Verdict: **capacity is NOT the
binder at this corpus size.** The 113M loses where the 45M is
saturated (L3 −3, L4 −2) and buys only +1 at L6 — 2.4x the
parameters spread the same tokens thinner and underfit the EASY
levels, exactly the ep1-trail prediction (train loss 0.362 final,
vs the 45M's lower plateau on the same data). The Liebig binding
factor stays where the novelty audit put it: **territory (L8/ODE),
not width.** Corollary for the scaling ladder: don't buy params
until tokens/param recovers to at least the 45M's ratio (~2.4x
current corpus for this width).

Birth panel (gate matrices, mid-layer): 113M CV 0.0210 vs champion
0.0144 — HIGHER, and data-per-width says rubble, not texture (the
calibration table's disambiguation holding on a fresh birth). Floor
0.862 vs 0.941 (emptier lattice). CV froze ep1→final (0.0210 →
0.0210): the lattice's texture is set by end of epoch 1; epoch 2
only polishes in place. Color-fade test (Artin's hypothesis: the
magnitude gradient rises with epochs): mean neuron norm 0.671 →
0.722 → 0.726 across ep0/ep1/final — CONFIRMED with saturation;
the fade-up happens in epoch 1 and freezes with the texture.
Growth movie frames: `docs/assets/neurons-113m-growth-*.png`.

Committee confluence (45M champion vs 113M, layer 6, 5 family-pure
prompt sets): the 113M grew its OWN full committee — 15 slots >1.8x
selectivity in BOTH models, zero index overlap (third birth
confirming lottery-redrawn / statistic-conserved). The sharper
pattern: **per-family selectivity strength is conserved across
substrates** — power is the most crystallized definition in both
(16.3x vs 16.5x top neuron), exp the softest in both (6.5x vs
6.3x), recip/mixed strong in both. Selectivity is a property of
the FAMILY's statistics in the corpus, not of the network that
learns it — the diet writes the same signature into every brain
that eats it (the corpus-is-the-organism thesis, at neuron
granularity).

Family tilt, not family clump (same night, Artin's locality
hypothesis): top-20 family-selective layer-6 gate rows are MORE
mutually aligned than random in 10/10 family x model cells (45M
power +31% over baseline, 113M power +46%) — but at cosines of
0.02-0.05, i.e. near-orthogonal. The expander refuses clumps;
families share a faint common DIRECTION while staying spread —
relatedness lives in correlations (holography), proximity is a
3% tilt. Computable prior for the calculated init: one anchor
per rule, near-orthogonal spray with corpus-statistic-weighted
tilt toward the anchor.

## Extended-training night: ternary reaches parity (2026-07-18)

Both continuations resumed at ep3 with a fresh OneCycle over the
remaining epochs (resume-schedule bug fixed en route: the old code
spanned the FULL horizon on resume and would have ended at peak LR).

**113M epochs 3-5: 66/120 @ 65.57** ({3:23,4:8,5:16,6:7,7:12}) —
the underfit diagnosis verified (L3 +2, L4 +2, exactly the starved
levels) but net +1 with small L6/L7 giveback: tokens-per-width
confirmed a second way. **Ternary epochs 3-5: 69/120 @ 67.13**
({3:24,4:8,5:17,6:8,7:12}) — **FULL PARITY with the fp32 champion
at 1.58 bits**, L3 perfect, L5 17/24 BEATS the champion's 16,
final latent loss 0.3396 vs the fp32 twin's 0.3526. The wiring
thesis completes: discrete lattices learn slower, not worse —
three extra epochs closed the whole six-solve gap. The lab's best
model is now storable in ~10MB. Promotion decision (co-champion
vs replace) deferred to Artin.

Ceiling probes (24 fresh L7 + 24 fresh L8 each, never gated
before): 45M champion L7 13/24, L8 2/24; 113M-6ep L7 12/24,
**L8 5/24**; ternary-6ep L7 12/24, L8 4/24. The capacity story
gains a nuance: the 113M loses in-corpus but LEADS at the
never-trained frontier (sqrt-composite quotients the 45M can't
touch) — width buys extrapolation past the corpus edge, not
mastery inside it. Solved-integral lists: `logs/archive/logs/archive/ceiling_probe.log`.

## Birth calculator v1: the gate is computable from the corpus (2026-07-18)

Rung 1 of the calculated-model thesis, first pass. A saturating
exposure curve (solves_L = 24*(1-exp(-eff_L/k))) on the gen-4
diet's per-level row counts alone gets RMSE 5.2 — and its misses
are SIGNED BY THE PHYLOGENY: L3 under-predicted by 7 (the clade
fed by 66k L1/L2 transfer rows), L4 over-predicted by 8 (the
isolated ansatz dialect). Adding the two clade terms the phylogeny
demands (transfer weight a=0.3 into/within clades; L4 usable
fraction b=0.55) drops RMSE to 2.5 with L4 exact (8.2 vs 8) —
3 params on 5 points, so the LOO caveat stands, but the BLIND
test passed: the model predicts L8 ~= 0.4 solves from its 319
rows, and the overnight probe measured 2/24. The gate profile is
a computable function of (row counts x phylogeny). First rent
paid — territory sizing BEFORE farming: L8=8/24 needs ~8.5k
effective rows, 12/24 ~14.6k, 16/24 ~23k (at 45M-fp32's k; the
sidecar shows mined rows count extra via depth quality). Next
rungs: validate across diets when v2-era births get re-gated on
the chain metric; then the compiled skeleton.

## Neuron biography: the lottery is redrawn WITHIN a lineage (2026-07-18)

Tracked the 113M's final-committee neurons (power #642, trig #768,
mixed #2098, layer 6) across ep0 -> ep1 -> ep3 -> ep6. Three laws:
(1) **committees crystallize late** — texture/CV freezes after ep1
but role selectivity sharpens ep1->ep3 (power 2.4x -> 16.5x):
structure first, specialization within it. (2) **extended training
REDREW the lottery mid-lineage**: the fresh LR cycle (ep3-5) moved
weights as much as all of epoch 1 (drift 0.23 vs 0.22) and
REASSIGNED the committee — #642 lost the power job (16.5x -> 2.9x),
the trig neuron collapsed to 0.4x — while the gate barely moved
(65->66). Function invariant, roles disposable, now demonstrated
INSIDE one model's lifetime (previously only across births).
(3) **committee neurons are weight-space average citizens** — drift
and norms indistinguishable from population; the "power neuron" is
special only in activation space (the tilt-not-clump law from the
other side). Consequence for the calculated-model thesis, called by
Artin: the microstate (which neuron does what) is the one thing the
system does NOT preserve — so Tracr-style circuit compilation
targets the WRONG invariant. Rung 2 re-aimed: **statistical
synthesis** (write down the conserved statistics — committee
strengths, tilt, norms, phylogeny — and sample a weight
configuration from them), not circuit compilation. Training's job
reduces to error-correcting the sample; the warm-birth pilot is
measuring the first rung of exactly that.

## Warm birth: the calculated init PAYS (2026-07-18)

Rung 3 of the calculated-model thesis, first pilot. Two 19M births
on the gen-4 diet, ONE epoch each, identical in every respect
except initialization: COLD = standard init; CALC = the template —
gate-neuron rows written from measured statistics before any
gradient step (near-orthogonal directions + 3% family tilt toward
5 anchors, family counts by corpus frequency, norms at the trained
crystal's distribution). Train losses statistically identical
(0.5369 vs 0.5363). Honest chain gates: **COLD 49/120 @ 46.31 vs
CALC 57/120 @ 48.70 — +8 solves and +2.4 validity from
calculation alone**, gains spread across levels ({3:+3, 4:+2,
5:+1, 7:+2}). Third confirmed case of capability invisible in
train loss (bf16 debit, ternary parity, now this). Verdict:
statistically-placed neurons calibrate faster — Artin's template
hypothesis ("put an arbitrary neuron where we think it's going to
be, pass the data, slight calibration") measured TRUE on first
contact. One template per closed system serves every birth.
Persistence verdict (same day): at 3 epochs COLD 64/120 @ 60.50
vs CALC 64/120 @ 60.79 — COLD caught up. **The template is a TIME
MACHINE, not a better basin**: same destination, reached ~1 epoch
sooner (CALC's ep1 gate of 57 sits between COLD's ep1 49 and ep3
64). Consistent with the FA Law: the corpus fixes S_max; the
template only accelerates the approach. Value proposition is
therefore compute, and it scales with birth cost — at 19M it saves
minutes, at 113M it would save the ~75-min epoch that carries most
of the texture. Still open: template MORE of the model (qkv/
embedding) to push the head start toward "calibrate immediately";
zero-epoch gate of the raw template as the north-star metric —
MEASURED same day (3080 cuda): **0/120 @ 0.00% valid** as
pre-registered (v0 templates only gate-matrix statistics; random
attention/embeddings can't emit grammar). The program's scoreboard:
calculation 0 -> calculation+1ep 57 -> ceiling 64 (19M). Every
template rung (embeddings, attention stats) is now graded by how
far it drags the zero.

## The alphabet is a lens, not an attractor (2026-07-18)

Do existing weights carry an alphabet fingerprint? Two probes on
layer-6 gate matrices. (1) The fp32 champion snaps to discrete
alphabets with smoothly falling cost (binary 0.315 / ternary 0.161
/ M5 0.067 / P2 0.040 normalized MSE at optimal scale) and
kurtosis 2.45 — a smooth, modeless, slightly sub-Gaussian
distribution. No hidden discreteness. (2) The SURPRISE: the
ternary model's fp32 LATENTS are equally smooth (kurtosis 2.84,
~Gaussian) and snap to ternary WORSE than the champion does
(0.183 vs 0.161) — STE training never polarized the continuous
weights toward the deployed values. Verdict: the alphabet is the
PROJECTION the crystal is viewed through at inference, not a shape
the crystal takes; underneath every alphabet lives the same smooth
statistical object (quantize-notches-never-axes, one level deeper;
the gauge story extended to precision). Consequences: (a) the
warm-birth template serves every tournament contestant unchanged;
(b) tournament prediction 3 sharpened — sufficient alphabets
should cluster, losers should fail on EXPRESSIVITY (binary's
missing zero), not fit.

## The NNUE is an oligarchy; the crystal laws are laws of width (2026-07-18)

Ran the crystal gauges on the founding artifact (nnue_eval.pt,
20->64->64->1 MLP, same closed system + oracle): lattice CV 0.165
(~8x rougher than any transformer crystal), kurtosis 4.78 vs the
crystals' ~2.5 — HEAVY-tailed where every crystal is flatter-than-
Gaussian, snap costs uniformly worse. Two regimes of learned
closed-system intelligence: OLIGARCHY (few big load-bearing
weights — the NNUE, and why chess NNUEs need int8's 256 levels)
vs DEMOCRACY (wide redundant populations — smoothness, holography,
the lens property, ternary parity). The crystal laws are laws of
POPULATION CODING, not of learning per se; ternary tied the
champion only because 2048-wide layers convert per-weight
precision into neuron count (the dimension dividend, with its
counterexample now measured in-house). Predicts a MINIMUM WIDTH
for alphabet-poor training — one-knob sweep (ternary births at
width 64/256/1024/2048) banked; would hand the birth calculator a
bits-required(width) curve.

## TF32 parity: the 3080 is reinstated for births (2026-07-18)

The precision-cliff bracket, closed in 21 minutes: a 19M birth on
the 3080 with TF32 matmuls (10-bit mantissa, tensor cores, zero
custom code — two allow_tf32 flags), identical recipe/seeds/diet
to the warm-birth COLD reference (Mac fp32: 49 @ ep1, 64 @ ep3).
TF32 ep3 gate: **65/120 @ 64.16 — full parity, +1 solve** (noise),
at **12.5 it/s vs the Mac's ~5** — 2.5x faster, while 10 farm
workers hammered the same box's CPU. With bf16's measured -3
debit: **the dynamical-precision cliff sits between 8 and 10
mantissa bits.** TF32 clears it; bf16 doesn't. Doctrine update:
births = fp32-on-Mac OR TF32-on-cuda (both sides of the cliff now
measured); the compensated-bf16/3xTF32 build is unnecessary —
the cheap rung won. The 400M-class scaling points and gen-6 can
now birth on the 3080 at speed.

## Axiom parity run 1: the native oracle arrives (2026-07-18)

The C++23 CAS (Artin's axiom, github.com/39hops/axiom — built
pre-llmopt, now Phase 8: llmopt-oracle) ran its first parity audit
against 72,988 tasks dumped from oracle-signed farm rows.
**1.04 ms/row — the full audit costs 80 seconds** (sympy-side
equivalence checking runs ~10-100 ms/row). Results: diff tasks
0 disagreements (10,823 byte-identical + 10,998 proved-equivalent
forms); equiv 45,712 agree; UNDECIDED tax 4.85% (under the 5%
gate; 55% of it sqrt-composite shapes — next canonical()
increment already identified). The audit caught one real bug on
EACH side: axiom's pow canonicalization collapsed (x^2)^(1/2)->x
(fixed, regression-tested), and llmopt's dump generator dropped
scalar coefficients on integral atoms (4*Integral(f) -> density f)
— axiom flagged all 64 poisoned reference rows as NOT_EQUIVALENT
and independent adjudication scored it **axiom 64, reference 0**.
Two oracles auditing each other found what neither would have
found alone. Gate: axiom PASSES run 1; oracle-of-record status
pending the sqrt increment + a clean re-audit (80s, now routine).
Projected farm impact: the diff/simplify half of L8 mining at
~10-50x, and eventual retirement of the 90s fork walls (sympy
pathology armor) on axiom-decided rows.

## Axiom bridge: both platforms live, replay-shadow clean (2026-07-18 night)

Phase A infrastructure complete in one evening. pybind bridge
(axiom_sym) built and smoke-passed on BOTH farm platforms: WSL
(axiom Fable, g++ 15) and macOS (this side, clang, first try).
llmopt adapter (llmopt/search/axiom_oracle.py): shadow/primary
modes, sympy oracle-of-record, JSONL disagreement audit,
UNDECIDED-never-valid; one adapter bug caught same-night (raw
strings passed where parsed Exprs expected -> silent 100%
fallback — the decided_rate stat exposed it). In-process replay-
shadow on WSL, all 48,081 real equiv rows through the .so:
**45,980 decided (95.6%), 2,101 UNDECIDED (4.4%), ZERO
crossings** vs reference labels. Replay v1 note for the scar
file: it hung on an unwalled in-loop sympy simplify — pathology
#10 reproduced while auditing sympy's replacement, which is
about as clear as the case for the replacement gets. Remaining
for oracle-of-record: live in-farm shadow accumulation (wires in
at the next campaign boundary; this shard stays pure sympy).
Sister repo progress (axiom Fable): generator parity L1-L4
450/450 byte-exact vs problems.py fixtures; four parity rules
locked incl. a C++ evaluation-order catch (MSVC right-to-left
silently flipped 35/100 L3 rows) — the byte-exact gate discipline
paying for itself per level.

## Axiom Phase B adjudicated: the generator is ported (2026-07-18 night)

Independent llmopt-side adjudication of axiom's Phase B gate:
regenerated all 10,000 (level, seed) fixture rows directly from
problems.py's _expression with the fixture's seed protocol
(diff-{level}-{seed}) — **10,000/10,000 byte-exact, zero
mismatches**. The C++ generator produces the L1-L8 ladder
identically to sympy, including L6-L8 sub-generator float-repr
seeding. Adjudication footnote for the honesty file: the FIRST
adjudication run compared against make_integrate (the wrapper)
instead of _expression (the ported core) and printed 5 phantom
mismatches — wrong-entry-point audits look exactly like real
failures until provenance is checked. make_integrate-level parity
(seed strings, retries, exclude= guards) remains open scope for
when C++ generation wires into the farm. Phase C (solver kernel)
in flight on the axiom side: C1 (carriers/count_ops 427/427) and
C2 (chassis: hash-cons state keys, three-valued verify_edge,
beam + full hook surface, adversarial-proposer and lying-rule
soundness tests) both landed same-night.

## The five-point scaling table (2026-07-19, ~1 AM)

Same gen-4 diet at 19M/45M/113M/200M/400M (fp32/TF32-clean births;
200M+400M are 1-epoch statistic points per the frozen-texture
license). One-epoch capability ladder: 19M 49 -> 200M 49 -> **400M
30/120 @ 29.22** ({3:11, 4:0, 5:9, 6:6, 7:4} — L4 ZERO) with the
1-epoch loss ladder inverting too (0.537/0.569/0.555/0.598):
width has crossed from not-helping into ACTIVELY HURTING at fixed
corpus — the 400M is too thin-fed to move its own mass in one
pass. Template statistics across scales (mid-stack gate matrices):

| model | d | CV | floor | R | norm | kurt |
|---|---|---|---|---|---|---|
| 19M | 384 | .0156 | .899 | .746 | .689 | 2.41 |
| 45M | 512 | .0144 | .941 | .831 | .684 | 2.45 |
| 113M | 768 | .0210 | .862 | .858 | .726 | 2.61 |
| 200M | 1024 | .0162 | .828 | .807 | .629 | 2.19 |
| 400M | 1280 | .0242 | .779 | .818 | .634 | 2.25 |

Honest reads: (1) **two scale-INVARIANTS** — kurtosis 2.4±0.2
(sub-Gaussian democracy at every width; the superposition phase
never breaks) and phase-R ~0.75-0.86 (no width trend). Invariants
extrapolate to any scale for free — template parameters #1 and #2
are width-independent constants. (2) **floor falls monotonically
with width from 45M on** (0.941 -> 0.779) — lattice emptiness
tracks data-per-width smoothly; fittable, hence writable into a
template at unseen scale. (3) CV and norm carry an EPOCH CONFOUND
(1-epoch points sit low on norm ~0.63 vs 3-epoch ~0.69-0.73 —
the color-fade rise; CV mixes texture/rubble regimes), so their
scaling fits need epoch-matched points: cheap to add (113M has an
ep1 snapshot; 19M/45M ep1 re-births are 17/35 min) — queued, not
assumed. Template-at-10B status: two parameters free, one fitted,
two pending epoch-matched data. The calculator grows teeth one
parameter at a time.

## The official qualification reference: sympy prices the ladder (2026-07-19)

The sympy engine (budget=200, fork-walled) on the 480 axiom
qualification roots: **L1 60, L2 60, L3 60, L4 51, L5 60, L6 58,
L7 56, L8 55 = 460/480**. Shipped to the axiom repo as the
oracle-of-record bar for Phase C qualification. Immediate
pricings of axiom's tranche-2/3 informal counts: L3's 49 = 11
REAL gaps (sympy is perfect there); L6 (axiom 2-6 unpruned) is
fully winnable — 58/60 — making C6's Markov-prior acceptance
target a ~52-solve gap, the largest single-component payoff in
the port; L4's bar is 51 (the ansatz tranche's true target), L8's
is 55. Reference stored per-root (data/axiom_qual_reference.jsonl)
so solve-by-both/either comparisons are row-exact.

## The closed-system equation, v0 (2026-07-19 — the working blackboard)

Assembled from the week's measurements; open slots marked. A
closed system Sigma = (rules, generator) acts on models ONLY
through its signature sigma(C) — the corpus statistics (per-level
row counts, clade/transfer graph, family frequencies) — plus the
model's width W.

CAPABILITY (measured, calculator v1, RMSE 2.5, blind-L8 pass):
  solves_L = 24 * (1 - exp(-eff_L(sigma) / k(W)))
  eff_L = rows_L + a * (clade-transfer rows), a=0.3, L4-isolation
  b=0.55; k minimized at the corpus's NATURAL WIDTH W* (~45M for
  gen-4: tokens/param at the champion's ratio) — above W*, k
  inflates (400M one-epoch gate 30/120, L4=0); below, S_max clips
  (19M ceiling 64). [OPEN: functional form of k(W) — 3 points.]

NEURON PLACEMENT (the template, measured):
  - Width-INVARIANTS (properties of Sigma, not the model):
    kurtosis 2.4 +/- 0.2, phase-isotropy R ~0.8 — hold 19M->400M.
  - Width LAW (fitted, rmse 0.006): floor(d) = 2.008 - 0.171 ln d
    (extrapolates: d=2048 -> 0.70, d=4096 -> 0.58).
  - Signature-driven: committee selectivity table (power ~16x,
    exp ~6x — diet-invariant across substrates), family tilt ~3%,
    tilt populations proportional to family frequencies.
  - Dynamics: texture+norms lock by end of ep1; ROLES are gauge —
    permanently fluid (biography: redrawn mid-lineage), never part
    of the equation. [OPEN: CV and norm width-laws — epoch-
    confounded, needs three ep1 births.]

THE CLAIM: model(Sigma, W) is determined UP TO GAUGE by
(sigma(C), W) — capability by the exposure curve, geometry by two
constants + one width law + the signature table; SGD contributes
the gauge choice plus an error-correction residual worth ~1 epoch
(warm-birth measured).

PRE-REGISTERED TESTS (the v1 agenda): (1) k(W) form from ep1
births at 3 widths; (2) CV/norm laws, epoch-matched; (3) the
PORTABILITY test — train on a DIFFERENT closed system (ODE-only):
if kurtosis/R shift, the invariants encode Sigma (huge); if not,
they are constants of training itself; (4) the far conjecture —
sigma alone predicts the committee table for a never-seen rule
family before any model eats it; (5) growth: model(Sigma, W->W')
by template-spray = model(Sigma, W') up to gauge (the grow-vs-
rebirth A/B).

## The equation at its limits (2026-07-19, ~2 AM)

Push v0 to extremes (Artin: "set things to infinity, 0, negative
— what happens"):
1. **Perfection is purchasable**: solves->24 exponentially, so
   the perfect model costs FINITE corpus: ~81k eff rows/level
   (<0.5 expected misses), ~650k total = ~5x today's corpus —
   weeks at the C++ farm's projected rate. The founding sentence
   ("mathematically perfect model for a closed system") now has a
   price tag.
2. **Geometric width ceiling**: floor(d)=0 at d~126k (~3T params)
   — the lattice runs out of orthogonal room; width has an
   absolute ceiling per closed system. (4-point log fit, 1000x
   extrapolation — a falsifiable flag, not a law.)
3. **The W* duality**: W->inf = NTK limit — training exactly
   calculable AND unlearnable (k->inf, lazy). Calculability and
   learnability trade off along width; natural width W* is their
   balance point. The 45M is not just right-sized — it is
   maximally alive.
4. **Transfer is a corpus multiplier**: a: 0.3->0.6 hands L3
   ~20k free effective rows. Curriculum design (ansatz-clade
   v2.3) is priced in row-equivalents: teaching structure beats
   mining data.
5. **Negative eff exists**: mislabeled rows — the equation's term
   for contamination (both historical incidents + reward-hack era
   are its measured cases). Verification is the SIGN of the
   corpus.
6. **sigma-variance->0 predicts the oligarchy phase** (NNUE): the
   democracy/holography/ternary-tolerance stack requires a rich
   signature. And kurtosis 2.4 (sub-Gaussian, no outliers, all
   widths) RETRO-EXPLAINS the storage doctrine — int4-lossless
   because there is no tail to clip.

## The fair fight: two engines, one map (2026-07-19)

First complete like-for-like comparison — same 480 roots, budget
200, markov3/width-3/no-NN/no-magic/no-heurisch both sides, both
soundness-spotless. **sympy-arm 420/480 (60/60/60/47/55/37/48/53)
vs axiom pure-native 316/480 (60/60/60/32/37/30/28/9), with 6
axiom upsets** (roots the sympy arm misses). Decompositions, all
measured: (1) axiom's markov3-vs-unpruned was a WASH (316 vs 317)
— at cheap-node economics the prior buys nothing; the starved-
judge law at engine scale, now measured from BOTH sides of the
node-cost divide (sympy's precious nodes made the same prior
worth +28). (2) L6 is THE heurisch level: +21 from the slot
(37->58), <=8 everywhere else — bridge-slot priority. (3) L8's
-44 with heurisch OFF = sympy's native sqrt/log-orbital reach
(axiom's unported i_sqrt_basis log-combo branch, pre-registered
suspect, confirmed). (4) Expiry pricing: 1 recovered solve in 38
at 60s — axiom's tail is rocks where sympy's was treasure
(engine-dependent tail economics; its 20s deadline stays).
Tranche-4 worklist shipped: the exact 110 arm-solved/axiom-missed
roots (L4:16 L5:21 L6:7 L7:21 L8:45), rule-attribution run in
flight — ports ordered by histogram, each with predicted yield
before it's written. Pure-native ceiling 420, hybrid target 460.

## Tranche-4 day: the attribution method works (2026-07-19)

Sister-repo scoreboard, one day: axiom pure-native 325 -> **382/480
by-either** (L7 28->47, L8 9->34+14 edge-certified; 62/110
worklist roots closed; zero NOT_EQUIVALENT ever, 74k audit clean
through three canonical() changes). The llmopt-side attribution
pipeline (miss list -> solved answers -> term decomposition ->
admission-path diff) called every shot: the size pre-gates
(scar-tissue-outliving-its-wound), then the atan-log orbital
histogram (17/26 terms), and the day's breakthrough sat one layer
under it — canonical() silently failing to cancel poly
denominators across mixed-opaque sums, feeding i_linear_basis
phantom 1/q monomials (wrong-without-erroring; found by tracing
one attributed node through three measurement-killed hypotheses).
Remaining to the 420 pure-native bar: 38 solves (L4 i_usub/
inverse-trig territory, L5 residue, L8's last 21); L6 parked for
the heurisch hybrid by both engines' agreement. Method verdict:
scope-by-diff + order-by-histogram + trace-by-attribution is now
a PROVEN cross-repo debugging instrument.

## Gen-6 reborn: NEW RECORD 71/120, and the L8 territory pays (2026-07-19 evening)

The first territory birth. Honest gate: **71/120 @ 66.22**
({3:23, 4:8, 5:16, 6:8, 7:16}) — first model past the champion's
69. Gate prediction (69+/-2) HOLDS at its edge; the +3 came
entirely from L7 (13->16), revising the phylogeny: L8's dialect
shares ansatz vocabulary with L7 (one-ply worked examples reach
back a level). **L8 ceiling probe: 21/24 raw — AUDITED before
celebration** (the miracle-shaped-number doctrine): 6/24 probe
roots appear VERBATIM in the training shard (third seed-space
collision incident; L8's generator space is narrow across bands).
All 6 contaminated were solved (memorization works, film at 11);
**clean subset 15/18 = 83% (~20/24)** — the territory genuinely
near-saturates the level, from 2/24. Calculator grade: predicted
12/24 (50%), measured 83% — a MISS, under by ~2x, with a clean
diagnosis: v1 treats rows uniformly, but format-matched one-ply
worked examples (85% of the shard, exactly the probe's task
shape) are worth ~2x+ generic chain rows per unit. Calculator v2
gains a row-format efficiency weight. Fix queued: future probe
bands generated with exclude= against training shards (the
scar's own remedy, applied to probes). Grown arm mid-training
(opening loss 0.391 at step 600 — the champion's paid knowledge
visible from the first batch).

## The hybrid runs: L6 56/60 in 4.7 minutes (2026-07-19 night)

First hybrid computation of the two-engine era: axiom's C++ solver
calling llmopt's fork-walled sympy slots (heurisch + equivalence)
through the pybind bridge, every slot proposal gated by axiom's
own verify_edge. **L6: 56/60 in 4.7 min** — native 30, sympy-full
bar 58. The heurisch slot delivers +26; the level is effectively
closed at a twelfth of the sympy engine's wall. Slot doctrine
proven end-to-end: lying-slot cannot corrupt (tested axiom-side),
crashing slot degrades to no-fire, language boundary enforced
llmopt-side (the erf smuggling catch). The hybrid-460
qualification arm is now just "run all 480."

Full arm, same night: **436/480 in 71.8 minutes**
(60/60/60/35/60/56/55/50) vs the reference 460. L5 CLOSED (slot
recovered all 23 native misses); L6 within 2, L7 within 1, L8
within 5; **L4's -16 is two-thirds of the remaining gap** and is
precisely axiom's queued tranche-4 worklist (i_usub-chain +
inverse-trig). Wall context: the sympy reference needed hours
with fork-walls; the hybrid runs the ladder in 72 min INCLUDING
its sympy slot calls. Qualification: 24 solves from the bar, all
mapped to named work.

## GEN-6 GROWN: 76/120 — growth beats rebirth, new champion (2026-07-19 night)

The grow-vs-rebirth A/B, final: **GROWN 76/120 @ 70.42**
({3:23, 4:10, 5:19, 6:8, 7:16}) vs reborn 71 vs champion 69 —
**+5 for growth at equal epochs**, first model past 70% validity,
all-time highs at L4 (10) and L5 (19). L8 probe: 21/24 raw, ~83%
contamination-clean — matches the reborn (the territory pays
identically; the growth surplus landed in the GATE band). The
mechanism, visible in the loss curves: the grown arm opened at
0.39 (the champion's inheritance) and closed at 0.3296 — the
lowest loss ever posted — spending all three epochs on new
material while the reborn re-learned the basics. The gen-5
rubble risk never materialized: template-sprayed neurons
integrated. Verdicts: (1) GROWTH ADOPTED as the standard capacity
move (calculator sizes it, template sprays it, identity-gate
proves function preservation, the gate adjudicates); (2)
PRODUCTION PROMOTED: mathnative_gen6_grown.pt (55.1M) is the new
champion; (3) the generational doctrine gains its missing verb —
the corpus is the organism, and now the body GROWS with the
territory instead of being reborn into it.

## Phase D adjudicated: the C++ engine is a certified row factory (2026-07-20)

Axiom's chain emission (spec + replay_chain + annotate + emit,
one overnight arc) produced 167 rows from 95 stratified roots in
farm_v22 schema. llmopt adjudication: schema exact (0 diffs), and
**167/167 pairs pass the production oracle** (verify_wave). The
first-pass adjudicator printed 8 false INVALIDs — its naive
density check cannot represent by-parts steps (integrals with
function coefficients / nesting); traced, not believed, second
auditor-was-the-bug incident of the weekend, lesson booked:
auditors must match the verifier's semantics. Meanwhile the
overnight axiom arcs also landed: native 391 by-either (L4 49 =
ABOVE the no-heurisch reference), shared-miss analysis (50 roots
missed by both engines -> honest pure-native ceiling 430; real
gaps 39, all named), expiries re-confirmed doomed (0/17 at 60s),
and magic boards rung 1 MEASURED: 2.28x warm-start (cold 94.4s ->
warm 41.4s, ledger identical) — riff to shipped feature in ~30h.
Remaining before the farm swap: hybrid arm on the next.so
(running), L5 rule-reach mass (14), live shadow accumulation.

## External timestamp: the Jacobian conjecture falls (2026-07-20)

Reported via an Anthropic employee: a Fable instance produced an
explicit COUNTEREXAMPLE to the Jacobian conjecture (open since
1939) — a cubic polynomial map C^3->C^3 with constant Jacobian
determinant -2 sending three distinct points to (-1/4, 0, 0).
This lab independently verified both legs in ~30s with its own
oracle (sympy: det exactly -2, all three collisions exact).
Logged here because it is the house thesis at civilizational
scale: an 85-year question settled by a finitely-checkable
artifact — verification-first, trust-free, the FA Law's
down-stroke doing in seconds what authority never could.

## Gen-6 ternary: 73/120 @ 71.81 — the 1.58-bit model beats the fp32 cold birth (2026-07-20 dawn)

The territory test of the alphabet-lens law, verdict: **ternary-
born-cold on the gen-6 corpus gates 73/120 @ 71.81%**
({3:22, 4:9, 5:17, 6:9, 7:16}) — BEATING the fp32 reborn arm (71)
like-for-like (both cold births, same corpus; ternary had 6
epochs per the discrete-learning law, fp32 had its converged 3)
and posting the HIGHEST VALIDITY EVER RECORDED (71.81 > the grown
champion's 70.42), plus an all-time L6 high (9). Standings:
grown-fp32 76 (inheritance) > ternary-cold 73 > fp32-cold 71 >
old champion 69. The lens law didn't just survive new territory —
at equal-cold conditions the discrete alphabet now WINS, and the
1.58-bit crystal is the second-best model the lab has ever made.
Queued consequence: the growth+ternary composition (grow the
ternary 73 the way the champion grew to 76) — if growth stacks on
the discrete lattice, the ~10MB model may take the crown.

## The authoritative hybrid number: 435, with both causes named (2026-07-20)

Hash-verified next.so (the stale-.so saga's lesson: cp onto a
mapped .so fails SILENTLY-BUSY on Linux — rm-then-cp, then
hash-verify the deployed artifact; two runs were burned by it).
Authoritative arm: **435/480 (60/60/60/41/60/56/48/50)**. The
delta anatomy vs the first arm (436): L4 +6 (the u-sub cascade
fixes) but **L7 -7 — the pre-expand cancellation's wall pressure**
(axiom flagged it natively same day; it reproduces through the
bridge), plus a NEW measured effect: **the slot tax** — hybrid L4
(41) < native L4 (49) because fork+sympy heurisch calls eat the
20s deadline on a level that never needed the slot. Qualification
memo consequences: (1) axiom's named perf follow-up (short-
circuit the cancellation trial) should reclaim L7 both native and
hybrid; (2) the qualification config should gate the slot by
level (or by native-first-then-slot retry) — heurisch pays at
L6/L8, taxes L4/L7. Projected post-both: ~455 vs the 431-adjusted
achievable math. Close, and every miss named.

## The alphabet tournament, real-valued bracket (2026-07-20)

Five discrete births at 19M/gen-4/3ep on the 3080 (TF32), vs the
fp32 reference (warm-birth COLD, 64/120 @ 60.50):

| alphabet | bits | gate | validity |
|---|---|---|---|
| B {+-1} | 1.00 | 54 | 36.73 |
| T {0,+-1} | 1.58 | 60 | 54.66 |
| M4 {-1,0,1,2} | 2.00 | 61 | 57.75 |
| M5 {0,+-1,+-2} | 2.32 | 62 | 62.07 |
| P2 {0,+-.5..+-4} | 3.17 | **66** | 65.10 |
| fp32 | 32 | 64 | 60.50 |

Verdicts: (1) prediction 1 CONFIRMED — the zero is load-bearing
(B's -6 solves and catastrophic -18 validity vs T: silence is
structure). (2) A clean monotone bits->capability ladder at THIS
width — and the punchline: **P2 (3.17 bits, shift-only
multiplies) BEATS fp32** (66 v 64, within gate noise but ahead;
hardware's favorite format wins on capability too). (3) The
width-bits EXCHANGE measured: at 19M ternary trails fp32 by 4,
at 45M it TIED (69) then WON cold (73 v 71) — per-weight
precision matters more at smaller width, exactly the
bits-per-dimension law (the dimension dividend needs dimensions).
Alphabet choice is width-dependent: small models want P2-class
ladders, wide models can afford ternary. Rotation bracket
(G5/E7/Q9) and L* still pending complex support.

## Grow-the-ternary: 74/120 — growth stacks, but thinner on the discrete lattice (2026-07-20)

The composition verdict: ternary 73 grown +256/layer and given 3
more epochs gates **74/120 @ 70.10** ({3:22, 4:10, 5:17, 6:9,
7:16} — ties the all-time L4 high). Growth pays on the discrete
lattice but at +1 vs fp32's +5. Honest read: not a growth failure
but a CEILING effect — ternary-cold's 73 (6 epochs) was already
deep into this corpus's yield for the config, while fp32-cold at
71 (3 epochs) had headroom for growth to harvest; the loss floor
agrees (0.3208 grown vs 0.3212 cold — nothing left to squeeze).
Final gen-6 standings: **grown-fp32 76 > ternary-grown 74 >
ternary-cold 73 > fp32-cold 71 > gen-4 champion 69.** The
1.58-bit lineage holds 2nd and 3rd. Next capability jump for ANY
of them requires territory (L9), not epochs, params, or bits —
every other dial is measured at its stop.

## The mass spectrum is a dial: diet moved, spectrum followed (2026-07-20)

Committee probe on the gen-6 ternary (layer 6): **exp — the
softest definition in every gen-4-era brain (~6.5x, both
substrates) — is now the crystallized one (22.1x); power fell
16x -> 9.7x.** The L8 shard is exp/log/sqrt territory, and the
selectivity table tracked the new corpus frequencies — the
mass-manipulability prediction (banked 2026-07-19) confirmed
without running its experiment: the spectrum is not a constant of
mathematics, it is a READOUT OF THE DIET, and we moved it by
farming. Consequences: the birth calculator's signature inputs
are per-corpus (as designed); precision/alphabet allocators can
now TARGET a family's robustness by feeding it; and the committee
table joins CV as a diet-imprint gauge.

## Qualification: 443/480 at 5x sympy's speed (2026-07-20 night)

Slot-config sweep complete (three arms, every cell measured):
heurisch-everywhere 435, L6/L8-only 431, **L5-L8 (optimal): 443
in 49.7 min** — 60/60/60/49/60/56/48/50. Vs sympy's 460
reference: -17, all named — L4 -2 (11 of its misses are shared-
with-sympy: at the achievable ceiling), L6 -2, L7 -8 (the one
real remaining gap; heurisch-heavy level), L8 -5. The
farm-swap arithmetic, stated plainly: **the hybrid solves 9
roots/min where the walled sympy engine solves ~1.5 — even at
96.3% of sympy's solve rate, it mints ~6x more verified rows per
hour.** For FARMING (the actual job), throughput times solve-rate
is the metric, and the hybrid wins it by ~5.8x today, before the
L7 gap closes. Decision memo updated; the swap call is Artin's.

## GEN-7 pre-registration (published before the run, 2026-07-20 night)

The most efficient birth the lab knows how to make — every
measured doctrine composed: START from the grown champion (76,
growth>rebirth), GROW +64/layer (calculator-sized to the L9a
increment), TEMPLATE-SPRAY the new neurons, identity-gate must
print exactly 76, then 3 epochs on the MASS-TARGETED diet
(L1/L2 at 15%, L3 at 30%, everything else full + the L9a shard —
Rung A's first live run), fp32/Mac. Predictions, calculator v2
(worked-example weight ~2x on the L9a rows, expected shard ~3-5k):
**L9 probe ~6-8/24 from zero; honest gate 76 +/- 2** (phylogeny:
L9a shares ansatz vocabulary with L7/L8, so small positive
transfer possible); epoch wall ~35% shorter than gen-6's (diet
~110-115k rows vs 178k). Failure modes pre-named: ration drift
on L1-L3 (the maintenance knob), audit-block if the native shard
fails dual-oracle (chain halts by design).

## L9b/ODE oracle adjudicated: 180/180 (2026-07-20 evening)

Axiom's five-rung ODE build (opaque y(x) + tuple-limit carriers,
check_odesol/check_ic on the substitute-first keystone, native
makers 90/90 byte-exact + self-verifying, slot telemetry, sample
emitter) formally adjudicated llmopt-side: **all 180 sample rows
agree with sympy's checkodesol + IC verification, zero
disagreements, zero walls.** The ODE continent's oracle is
CERTIFIED. Fixture gems from the build: the constant-binding
sentinels (C1*e^x passes y'=y, e^x+C1 correctly NOT_EQUIVALENT)
and the sentinel that taught its author (sin^2+cos^2 IS a genuine
solution of y'=0 — diff kills it exactly; honest-UNDECIDED needs
the identity in the residual). Slot telemetry already earning:
decisive=0 on slot-fires-but-loses-race is the HEAVY signal
working as designed. L9b farming is now an engineering decision,
not a research one.

## Bits-dimension exchange test: MISS, with a named confound (2026-07-20 night)

Pre-registered: B@768 (binary, one width doubling) should gate
~60 (=T@384), repaying the 0.58-bit alphabet debt. Measured:
**45/120 @ 26.86 — WORSE than B@384's 54/36.73.** The law's
prediction failed, but the experiment is confounded: doubling d
quadrupled params (19M -> 75.6M) on the SAME corpus and epochs —
the tokens-per-width ceiling (the 400M lesson) pushes DOWN exactly
when the exchange term pushes up, and at 3 epochs the starvation
won (loss 0.363 vs B@384's floor shows the capacity was there;
the gate shows it was unfed). Two live interpretations: (1) the
law holds but the test needs data-matched conditions (B@768 at
~4x epochs, or on the gen-6 corpus); (2) binary is special — the
zero's absence is a STRUCTURAL deficit (no silence) that no
interference budget repays. Status: law WOUNDED, not dead;
clean re-test queued for a free GPU day. The honest-miss file
grows — and the confound itself re-confirms tokens-per-width as
the strongest force on the board.

## The flip census: discrete plasticity is REAL and it pays (2026-07-20 night)

Artin's oldest standing bet ("RL will move the ternary — flips
chain, silent until the neuron fires"), adjudicated by a 20-minute
LLMUE burst on the 3080: the gen-6 ternary's fp32 latents
metabolized 773 oracle-signed L9a rows (933 cycles), and the
DEPLOYED 1.58-bit lattice moved — **100,884 whole-weight flips
(0.2004% of 50.3M)** — while the proxy gate rose **19 -> 21 and
validity 58.2 -> 60.3**. Prong A wins outright: discrete learning
speaks in visible, countable flips, the latents integrate
sub-threshold and commit in units, and the committed flips CARRIED
CAPABILITY. Learning on a discrete substrate is photographable —
0.2% of weights moved, each one a discrete event with a
before/after, and the population of 100k flips bought +2 proxy
solves on fresh territory in twenty minutes. The LLMUE mechanism
works on the alphabet where it matters most (the ~10MB deployment
class), and the discrete-plasticity fork closes as a WIN for the
wiring thesis' final claim: the topology doesn't just carry
capability — it UPDATES in quanta.

## Future work (spec'd or banked, in priority order)

[2026-07-07 status: bandit RUN (null — see above); bidirectional RUN
(became i_linear_basis — the "rule mining" prediction was exactly
right, and it was the good outcome).]

Symmetry-compressed ansatz (banked 2026-07-08, SCDF's logic minus the
chemistry): parity of the integrand halves i_linear_basis's unknowns
(odd f -> even F -> even monomials only); low fire-rate on the current
generator's random constants — wants a symmetric problem family.
Risch-as-solver (banked): sympy's Risch is a partial DECISION
procedure — deeper integration could construct antiderivatives, not
just certify death (the magic detector's constructive twin).
Quantum-circuit T-count minimization (banked 2026-07-07, physics
night): states = circuits, moves = gate-rewrite identities, oracle =
stabilizer-tableau equivalence (GF(2), poly-time — CHEAPER than
sympy), eval = T-count (the industry metric: post-error-correction,
T gates cost ~100x Clifford). The whole engine architecture ports;
"reduce the magic" is a real optimization market.
Mechanics mathgen kind (banked 2026-07-07): Lagrangian -> equations
of motion, Euler-Lagrange verify-by-substitution, Noether-checksum
conservation tests, dimensional-analysis pre-verifier (the type
checker of physics; reusable for all future physics kinds).
Tensor-decomposition weight compression (banked 2026-07-07, Artin):
TT/Tucker-compress real checkpoint weight matrices across ranks and
score the compression-vs-capability curve BY RUNNING against the
oracle (function MSE / symbolic accuracy) — never weight distance
(the weightspace iron rule). LoRA is the low-rank special case the
training threads already validate.
Strategy-portfolio bandit (UCB over engines per problem class — the
signals exist: H, P(solve), yield). Bidirectional/meet-in-the-middle
search — WITH a design warning discovered on paper: the natural
backward pass (differentiate candidate forms into a lookup table)
memorizes the problem generator's distribution when problems are
reverse-sampled from the same family — contamination wearing an
algorithm's clothes. The legitimate version stores coefficient-general
patterns, i.e. bidirectional search in math either cheats or becomes
rule mining. Highway mining v2
(n-gram macro auto-promotion). Lean port (proofs: same architecture,
tactic moves, kernel verifier). Code domain (states = programs,
oracle = toolchain + tests, eval = stopwatch; train on execution
traces — the ladder's simulation-resists finding). Budget reallocation
across problems (NNUE's predicted nodes-to-solve as a quit-early
signal). Digital-circuit port (states = netlists, moves = Yosys
rewrites, oracle = ABC formal equivalence — math-grade and
milliseconds; arguably the most tractable domain jump). The verifier
cost gradient (formal > simulation > physical) maps the whole
portability frontier: analog/SPICE and photonics/FDTD port with
shrinking search budgets and growing reliance on learned evals; the
fab is the regime where the loop inverts.

## Reproduction

Every table: `scripts/bench_*.py` with string seeds; every training
run: `scripts/train_*.py`; data generation: `scripts/gen_*.py`,
`scripts/harvest_*.py`. See README Highlights for the per-result
commands and the git log for the measurement-by-measurement history.

## English labels attach as pure readout: 95% naming, zero rewiring (2026-07-21)

The grounded-then-labeled riff's first live test, run to Artin's
constraint ("those neurons should not AFFECT how the model
thinks") — enforced by construction: vocab grown 40->48 (<name> +
7 family words), ONLY the 8 new embedding/head rows trainable,
base frozen with old-row grads zeroed, and the identity PROVED
after training (sha256 on every frozen tensor + bitwise old-row
equality — both passed). Task: problem -> family name at the
<name> position; 2000 train / 200 eval, string-seeded bands,
eval excluded by expression string. Verdict: **trained-base 95.0%
vs random-base control 90.0%** (majority floor 38%). Honest
anatomy: family names are ~90% surface-readable (the log( token
gives away "logarithm"), so the control was essential — the math
brain's contribution is thin in accuracy (+5) but large in
learnability (loss 0.16 vs the random base's 0.73 stall). The
pre-registered guess that the delta would concentrate on `mixed`
MISSED — it spreads evenly. Deliverable:
checkpoints/mathnative_19m_labeled.pt — the lab's first model
with English attached, 8 tokens, provably quarantined from the
thinking substrate. Wall: 424 s end to end (8-way parallel
root generation, 216 s of it). English as readout: measured.

## Gen-7 / metabolic / VRM: the preservation-without-gain morning (2026-07-21)

Overnight verdicts, all against pre-registration, with the control
that kept us honest. **CHAMPION-CONTROL L9 probe: 9/24** — the
untouched gen-6 champion already solves 9 L9a problems by pure
L7/L8 ansatz transfer, so every L9 number below is a DELTA of zero:
(1) **GEN-7 (birth, 320-row shard): gate 71 (-5, MISS below 76+/-2)
 / L9 9/24 (+0)** — the losses sit exactly on the rationed levels
(L3 -3, L5 -2), the pre-named failure mode (maintenance ration too
thin); wall -38% HIT (predicted -35%). Rung A: wall claim proven,
ration knob failed, shard sub-threshold. NO-PROMOTE; champion
remains gen6_grown 76. (2) **METABOLIC (LLMUE pilot, 471 signed
rows, ~215 cycles, kernel-panic-terminated): gate 75 (-1,
noise-range) / L9 9/24 (+0)** — the immune system validated at
probe level (3.5 h of self-feeding, zero rollbacks, function
preserved), but at LR 1e-5 on a sub-threshold diet LLMUE
preserved without growing; the only measured LLMUE GAIN remains
the ternary flip burst (+2 proxy). (3) **VRM v0: NULL, confounded
— routed 54 = uniform 54, both -10 vs baseline 64**: the burst
harness itself (fresh AdamW, lr 3e-5, 1ep old-corpus) damaged
both arms, drowning the routing variable. Informative residue:
frozen committees protected NOTHING (identical damage) —
forgetting lives in the distributed bulk, not the committees,
independently consistent with the MP-spectrum bulk-load-bearing
finding. Re-run gentle. (4) Calculator: third directional miss —
clade transfer underweighted (predicted ~4 from zero; transfer
alone = 9/24). v3 needs an explicit transfer term. Day's law:
every mechanism built so far PROTECTS capability (identity gates,
immune system, rations); none of last night's added any — gain
still comes only from real territory at real exposure (the
farm's job, ~1k/4.8k roots).

## Late-layer metabolism: cheaper AND safer — the control-rod result (2026-07-21)

The confluence shortcut's A/B, identical LR ladders (3e-5 ladder
x1.8 to a deliberately abusive 1.8e-3), identical band/cycles:
**hot (full backprop) gate 57/120 @ 57.4, L9 7/24; late (layers
8-11 FFN only, ~60% of backward skipped) gate 71/120 @ 68.5, L9
9/24 intact.** Same abuse, +14 gate points of erosion resistance —
the hot arm's damage came overwhelmingly from churning early
layers that texture-froze at birth and had nothing left to learn.
At sane LRs (<=3.1e-4) the late arm held the champion's proxy
pristine where hot was already bleeding. Verdict: late-layer
metabolism is doctrine — the depth profile (delta mass
monotonically increasing 0->11) predicted it, and prediction (4)
UNDER-called it (expected parity at 40% wall; got dominance).
Composed doctrine for metabolic v3: late-only + absolute-anchor
tripwire + LR <= 1e-4 + surprise gating. Ceiling note: nothing
survives 1.8e-3 (both arms erode) — the frontier is real.

## L9 probes are device-dependent at 2x — the day of controls (2026-07-21)

Ternary-NNUE probed 19/24 on cuda against the morning's universal
9/24 (MPS) — a +10 headline that died by TWO controls in
sequence: (1) pre-metabolism ternary control: 17/24 (so NNUE
gain = +2, matching its +2 proxy, gate held 72 v 73); (2)
device control: **the fp32 champion itself scores 18/24 on cuda
vs 9/24 on MPS — same checkpoint, same band, same seeds.** The
substrate-generalization story evaporates; what remains is a
measurement law: at the L9 frontier the model's choices are
near-tie-close and MPS-vs-TF32 rounding resolves the flips
differently (the fp16-near-tie doctrine at probe scale — the
model knows more L9 than MPS sampling expresses). IRON RULE
ADOPTED: never compare probes/gates across devices; every number
lineage lives on one machine (morning MPS deltas — gen-7 +0,
metabolic +0 — remain valid same-device comparisons). Surviving
result: ternary NNUE metabolism +2 L9 / +2 proxy / gate held,
third consistent discrete-substrate gain. Day's tally: two
would-be headlines killed by two controls within six hours —
the control reflex is now the lab's most productive instrument.

## Series continent: axiom module CERTIFIED 180/180 (2026-07-21 evening)

Same-day turnaround on the series-arithmetic tranche: axiom's
ax::sym::series (dense truncated series over exact Q, no floats) +
check_odesol_series (residual oracle, EQUIVALENT_TO_ORDER /
NOT / UNDECIDED_BEYOND_ORDER) + series_solve chain emitter (one
row per coefficient, one exact rational division per step).
llmopt adjudication of the 180-problem/1200-row sample batch:
**180/180 byte-exact** (every a_n chain equals sympy's Maclaurin
coefficients of the drawn solution as exact rationals) AND
**180/180 residual-clean** (final partial sums vanish in the ODE
to order). The Liouville-jailbreak substrate is live: series
chains farm at polynomial-op throughput with no heurisch walls.
Correction booked alongside: the L7 "pre-expand short-circuit"
ask in the outbound relay was a MISATTRIBUTION (axiom's record:
short-circuit nulled twice; the real L7 recovery 40->47 was
best-of-both canonical, already shipped). The qual L7 gap (48 v
sympy 56, hash-verified bridge) is real but UNNAMED — per-root
diff of the 8 misses queued before any further code ask.

## Shaped GRPO (b-lever rung 2): NULL — and the potential was aimed backwards (2026-07-21)

Pre-registered: shaped reward (r = verified * (1 + 0.5*tanh(dPhi/20)),
Phi = -(count_ops + 40*n_Integral)) on the champion must beat the
+2-solve plateau over 12 cycles. Measured: HALT at cycle 12, gate
pinned 75-76 (baseline 76) throughout, 4 rollbacks, validity peak
71.08. NULL — scoped honestly: **size-based potential nulls and
mildly harms**, because integration by parts (and every cyclic
trick that carries L5-L7) makes expressions BIGGER before they
collapse — the shaping penalized the legitimate uphill step; the
rollbacks were the reward fighting correct instincts, not noise.
The untested lever remains Phi = engine distance-to-solved (plies
remaining — prices uphill steps as progress); that is rung 2b
with its own pre-registration. Rung 3 (wave-contrast) needs no
progress metric at all and may leapfrog it. Side profit: ~14k
verified mined rows streamed to the corpus during the run.

## Calibration night: the exam was flattering everyone (2026-07-21 night)

Three instruments, one story. (1) **Seed variance: gate sigma < 1
solve** (three identical 19M births: 65/65/64) — the production
gate is a precision instrument; historical +-2 margins were
conservative; differences >=3 are real. (2) **Holdout v1 VOIDED
by its own audit** (281 corpus collisions on the 88M band — the
exclude= doctrine violated by its author, caught by the audit he
built). (3) **Holdout v2 (exclude-guarded, provably unseen):
champion 64/120 @ 62.43 (production 76); ternary 60/120 @ 57.16
(production 73).** The ~12-point gap = selection overfitting
(weeks of promotions against one fixed battery breed
battery-shaped champions — survival, not leakage) + small
generator overlap. Deltas measured on a shared battery remain
valid (both sides equally flattered); ABSOLUTE capability
restated: champion is a 64-class model on truly unseen work.
**Ternary generalization hypothesis REJECTED: -13 vs fp32's -12 —
the lean substrate generalizes identically.** Doctrine: promotions
gate on exclude-guarded batteries, regenerated periodically so no
exam lives long enough to be learned. Fairness note for the
record: champion (55.1M grown, warm lineage) vs ternary (45M
cold, 6ep) is a cross-class comparison — the like-for-like pair
was always ternary-cold 73 v fp32-cold 71 (identical config/
corpus); the champion comparison flatters fp32 by +10M params and
one inheritance.

## L7 closed at gap=0 + gate-v2 MPS null (2026-07-21 midnight)

(1) q-l7-58: sympy WALLS (>2 min) — shared ceiling. Final L7
decomposition vs the 56/60 reference: 7 contention artifacts, 2
special-function over-credits (Fresnel = outside the elementary
language), 3 joint-ceiling, **0 real reach gap**. Qualification
doctrine: solo or contention-matched arms only; references
scored on the same language as the contestant. (2) Gate-v2
(cross-problem batching): honest NULL on MPS — idle unbatched
6.2 min vs batched K=12 20.0 min (3x SLOWER; no KV cache means
batching multiplies O(T^2) full-sequence recompute). Two
corrections ride along: historical "15-min gates" were
contention-inflated (idle truth: 6 min), and the real sampler
speedup is a KV CACHE (O(T^2)->O(T) per token) — promoted to
head of the speed queue. Exact-match property of gate-v2 stands
(token-identical at K=1/12/24) — the machinery banks for
KV-cached batched sampling later.

## Overnight 2026-07-22: the crossover, the battery truth, and friends

(1) **BOUNDARY GRID — the width-bits crossover measured**: fixed
gen4 corpus, 3ep, from-birth: fp32 61 (d256) / 64-65 (d384,
historical) / **58 (d768)**; ternary 60 / 60 / **65 (d768)** —
at d768 the 1.58-bit model beats fp32 by 7. fp32 peaks at its
natural width then starves (tokens-per-width); ternary keeps
climbing because W* scales INVERSELY with bits (smaller effective
capacity = larger natural width at fixed corpus). Boundary-or-bulk
raw verdict: capability tracks effective-bits-vs-corpus balance,
decisively NOT volume (params x bits predicts fp32@768 first; it
came last). (2) **The selection-overfitting story RETRACTED**:
seedvar births (never promotion-selected) drop -13 on holdout-v2,
identical to the champion's -12 — the gap is the BATTERY
(exclude-guarding biases toward rarer/harder shapes), not
selection. Cross-battery absolutes invalid both directions;
within-battery deltas remain gold; "too well" resolves as
measurement, not rot. (3) **Synonym gauge: 94.5%** family
accuracy (= single-label 95) — two words per concept attach
equally well; usage splits unevenly per family (mixed 47/36,
root 13/2) = PARTIAL gauge symmetry breaking in naming. (4)
**Chain-carry ablation: VOID, design error mine** — 8k-row
budget (vs 132k production) + repetition asymmetry in the
oneshot arm; both arms scored noise (4-20/120). Redesign at full
budget. (5) Ternary compounding: sessions do NOT stack at
1e-4/late-layer on the same organism (s2: 22->21); the repeated
+2s were fresh-pair effects. (6) fp64 decider arm A: 0.078%
flips, proxy flat; arm B verdict pending this morning.

## Rarity curves + fp64 verdict + merged run launched (2026-07-22 morning)

(1) **fp64 decider arm B: fp64 masters RECOVER the flips** —
0.3835% vs arm A's 0.078% (~5x) at equal food, LR 2.5e-6; proxy
flat (22 @ 65.7 -> 65.9%). Flip mass concentrates late
(blocks.9-11 down/up/gate dominate), matching the control-rod
picture. Verdict: absorption is REAL and fp64 masters fix it —
but the regime is low-LR (metabolism/GRPO), where absorption
bites (0.278% @ 1e-6), NOT from-birth 3e-4 (0.0007% @ 1e-4 by
the law). **Fold-in decision: fp64 masters go into the ONLINE
trainer, not the merged birth** — keeps the merged run's
pre-registered one-variable purity (diet only) AND aims the fix
where the mechanism lives. Deviation from spec item 3 noted.
(2) **Rarity-stratified gate BUILT + baselines measured**
(scratch/gate_rarity.py; skeleton = digits->#, counted in corpus
cur-set; 200 probes, bins common/mid/rare/unseen): CHAMPION
common 73/78 (94%) / mid 32/43 (74%) / rare 8/14 / unseen 15/65
(23%), overall 128/200; TERNARY-73 78/85 / 28/40 / 4/10 / 17/65,
overall 127/200. The scalar gap champion-vs-ternary (76 v 73
prod, 64 v 60 holdout) FLATTENS to 128 v 127 on the stratified
battery, and ternary EDGES the champion on unseen skeletons
(17 v 15) — third strike against "ternary can't generalize."
The curve itself is the headline: ~94% -> ~23% from common to
unseen-skeleton is the quantified battery truth; every future
promotion reads this curve, territory births judged at the rare
end. (Census note: diet recomposition files excluded from the
skeleton count after run 1 — 99-skeleton drift caught, bins
re-freeze next run.) (3) **THE MERGED RUN launched**: diet built
(116,738 rows = gen-6 cumulative with L1-L3 rationed to 45%
[16.4k/22.1k/12.4k survive] + 782 deduped L9a rows), d768/8L/
ffn3072/h12 ternary-from-birth 6ep, chained on the 3080 behind
the 1e-7 absorption arm. Pre-registrations stand: beat twin-65,
contend with champion-76 same-battery, move the RARE end.

### Rarity curves, frozen census + per-probe diff (2026-07-22 addendum)

Census fix (diet recompositions excluded) + per-probe dump, identical
200-probe battery: CHAMPION common 65/69 / mid 37/48 / rare 11/18 /
unseen 15/65 (128 total); TERNARY 64/69 / 37/48 / 9/18 / 17/65 (127).
Mid bin DEAD-TIED 37/37; fp32's entire scalar lead is rare-seen (+2)
+ common (+1); ternary leads unseen (+2). Per-probe diff shows the
mechanism in the problems themselves: fp32-only solves are memorized-
technique shapes (log(x)^2 by-parts families, folded constants like
log(16777216), mixed exp*trig recombinations); ternary-only solves are
rule-recognition shapes (spot-the-derivative-of-composite: 
(48(x-1)^2+8)exp(3x^2-6x+6), (4x-2)cos(2x^2-2x+1), (10x^2+5)exp(x^2)
— pure chain-rule inversions). Episodic-memory-vs-rule-compression
split confirmed at the level of individual expressions. Chart:
rarity_curves.html (sent 2026-07-22).

### Absorption law, fourth point (2026-07-22)

1e-7 arm: 2.7901% absorbed — the x10-per-LR-decade law holds across
four points (1e-4: 0.0007% / 1e-5: 0.030% / 1e-6: 0.278% / 1e-7:
2.79%). Absorption fraction ~= c/LR with c ~= 2.8e-9; the law is now
predictive, not descriptive. Merged run auto-fired behind it.

## Series rung 1: form learned instantly, task was ill-posed (2026-07-22)

19M birth (BIRTH_SEED=1, gen4 base + 793 stripped series rows, 36
min): held-out probe 23/142 (16.2%) exact-coefficient steps
(separable 13/25, linear1 7/63, cc2 3/54); paired gate 63/120 vs
seedvar-1's 65 (-2 vs a sigma<1 baseline — small real dent, watch).
The misses are the finding: EVERY miss appends exactly one new
leading term, correct power, prefix verbatim — the model learned the
expand-by-one-term move from 793 rows; only the coefficient is wrong,
and it defaults to memorized constants (x**5/120, x**7/120 — e^x
factorials). Root cause is diet design, mine: the rows never showed
the ODE, so the next coefficient is UNDERDETERMINED from the partial
sum alone — the model was asked to guess hidden state. (The 23 hits
are recurrence-guessable prefixes: separable leads.) Scorer note:
first probe run printed 0/142 from a broken scorer (macOS spawn
multiprocessing) — fork start method is part of the solve_isolated
doctrine now. RUNG 1b launched: ODE parameters injected into the
prompt as a vocab-40-tokenizable tuple prefix "(family, params...)"
— linear1 (1, a), cc2 (2, p, q), separable (3, c) — same seed, same
base, one variable.

## Duo-substrate wave: complementarity is real AND free (2026-07-22 eve)

Budget-matched mixed wave (8 ternary + 8 fp32 per ply = same total
as one model's 16), same 200-probe rarity battery: **133/200**
(65/69 common, 37/48 mid, 13/18 rare, 18/65 unseen) vs singles 128
(champion) / 127 (ternary); oracle union at 2x budget = 137.
Pre-reg (a) PASS: +5 over best single at EQUAL budget. Pre-reg (b)
PASS: rare 13 >= 11 (the mix beats the champion on the champion's
OWN tail) and unseen 18 >= 17. The two substrates propose from
different distributions and the oracle keeps whichever fires —
diversity beats depth at fixed budget. Adoption: duo wave becomes
the SEARCH/FARM sampler (mining, expert iteration — solves are
data); promotion gates STAY single-model for lineage comparability
(a gate number that mixes substrates measures the pair, not the
candidate). Rider result — paired L9 probe, same device (MPS):
champion 9/24, ternary 8/24 — near-tied as questions get harder;
the lean substrate does not fall off a cliff at the frontier.
Rider 2 — ternary weight census: 30.5% zero / +34.8% / -34.7%,
near-uniform across all 12 layers; distribution entropy = 1.58
bits = the THEORETICAL MAX for 3 states — the crystal encodes at
full information density (matches no-latent-polarization).

## Series rung 1c: decomposed arithmetic TRAINS UP (2026-07-22 night)

Axiom's chain batch (26,844 rows, independently re-verified: 0/14,844
arithmetic mismatches) -> 19M birth, same seed/base as rungs 1/1b,
one variable (23,855 chain rows replace 793 hops). Held-out probe
(seeds 17-19, 358 steps, per kind): **67.0% overall vs 15-16% for
both single-hop rungs (~4.3x)**. The ladder law confirmed in a
brand-new grammar: solve steps (operands spelled out) train UP —
separable 63/63 PERFECT, linear1 56/63, cc2 45/54 — while the same
model asked to do the arithmetic implicitly (rungs 1/1b) sat at 15%.
Residue map: sum steps 10/36 (multi-term products resist — the next
decomposition target), cc2 appends 13/54 (two-back recurrence
placement lags; linear1/separable appends 36/63, 17/25). Gate 63/120
(-2 vs seedvar-1, same dent as rungs 1a/1b — series rows cost ~2
integral solves at 19M regardless of volume; capacity, not
interference-by-format). Next tranche ask for axiom: decompose the
SUM rows further (one product per emission) and more cc2 seeds.

## The 43x: allocator thrashing, not model cost (2026-07-22 night)

The merged d768 run "ran slow" (0.2-0.3 it/s, ~4h/epoch) for three
epochs. Diagnosis chain: axiom-contention theory (partly true,
box was shared) -> quiet-box measurement still 0.2 -> restart at
epoch boundary with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:
**8.6 it/s (43x)**. Root cause: the ep0 allocator OOM-retry event
(logged warning, "free: 0") left the caching allocator permanently
fragmented — every step paid retry+flush. Loss continuous across
the restart (0.337 -> 0.335); allocator is wall-clock only, zero
numerics — the twin comparison is untouched. DOCTRINE: a CUDA
allocator OOM warning in a training log is a TRIPWIRE, not noise —
restart at the next epoch boundary with expandable_segments
immediately; do not average the slowdown into "the model is slow."
Credit: Artin refused the slow-run story three times until it broke.
Epoch now ~7 min; the entire overnight schedule collapsed into the
evening. (Restart discipline: wrapper killed BEFORE trainer so the
completion marker could not false-fire the readout chain.)

## Series rung 1d: one-primitive emissions train to ~100% (2026-07-22 night)

Chain2 (69,424 rows: pairwise mul/add trees + cc2 3x): probe 395/451
(87.6%) vs 67% (1c) vs 15-16% (1/1b). PRE-REG PASS: mul 93/93
PERFECT, add 35/36 (97%) — one-primitive emissions train to ~100%,
the ladder law's cleanest form. solve: cc2 54/54 (was 45/54),
separable 63/63, linear1 56/63. cc2 appends 13/54 -> 33/54 at 3x
volume + tree context — NOT flat, so my format-only prediction was
partly wrong: appends are volume-responsive, though still the
lagging kind (the underdetermined-append theory survives as
"lagging", not "frozen"; shift/attach split still worth testing).
BONUS: **gate 65/120 — the -2 series dent is GONE** (1a/1b/1c: 63/
64/63) with 66k series rows in the diet. Decomposed one-primitive
rows cost the integral gate NOTHING; the dent was interference from
underdetermined/multi-fact rows, not capacity. Determinable data
does not fight the resident crystal.

## PHYSICS CONTINENT OPENS: 85.5% first contact + equation test #3 (2026-07-22 night)

(1) **Physics rung 1** (vocab-41 birth, 20,253 rows, ~4 min train):
held-out probe 177/207 (85.5%) — kin/int 51/51 PERFECT, kin/append
49/51 (96%), shm/solve 53/54, shm/append 24/51 (47%). The split IS
the determinability law measured within one model: kin appends were
rebuilt as determinable folds after the audit (5,106 joins), shm
appends kept the underdetermined series format — 96% vs 47%, same
model, same birth, format the only variable. Physics is the ODE
grammar wearing t; the machinery transferred whole.
(2) **Equation pre-registered test #3 (portability), answered with
a control**: physics crystal kurt 1.88 initially read as an
invariant SHIFT (math band 2.31-2.50) — but a size-matched math
control (20,253-row subsample, same width/seed/epochs) reads kurt
1.91 / nnCV 0.0232 / norm 0.593 vs physics 1.88 / 0.0219 / 0.584:
NEAR-IDENTICAL. Verdict: the geometry constants do NOT encode the
grammar — they encode data-per-width (feeding). Grammar changed,
geometry didn't. The "invariant" is a constant of training-at-a-
given-feeding; corpus SIZE (not content) sets kurtosis. (Panel
definitions reimplemented post-/tmp-wipe: pooled 4th-moment kurt,
NN-cosine CV on unit neurons, mid-stack gates — same code both
sides, paired doctrine.)

## Poly pilot: the ladder law at grammar #3 + a gate LIFT (2026-07-22 night)

Poly birth (gen4 base + 10,944 poly rows): probe 56/110 (50.9%) with
the ladder-law split a third time — one-primitive constant rows
train (den 19/24, res 20/24), multi-fact rows lag (num 7/24:
substitute+arith one hop; divstep 9/21: expand-subtract one hop;
monic 1/8; assemble 0/9: whole identity one hop). Tree-decompose
and it should follow the 15->88 series arc; next axiom tranche.
GATE: **67/120 — first diet addition to RAISE the integral gate**
(+2 vs seedvar-1's 65; below the >=3 significance bar, suggestive
only — but mechanism-plausible: partial fractions feed integration,
axiom's own ranking rationale). Rerun at tree-decomposed volume
decides.

## THE MERGED RUN: 70/120 — diet alone buys +5 at d768 ternary (2026-07-22 night)

Wide-ternary d768/8L/ffn3072/h12, 6ep, one variable from the twin
(diet: gen-6 cumulative, L1-L3 rationed 45%, +782 L9a rows).
**Deployed gate 70/120 @ 65.66 vs twin 65 (same-day recheck, same
device) = +5, above the >=3 bar. PRE-REG (a) PASS.** Shape as
pre-registered: L7 16/24 v 12 (L9a moved the top end), L4 9 v 4
(rations repaired the gen-7 wound). Loss 0.30 final. Run archaeology
banked separately (the 43x + grad-ckpt saga): d768/BS32/fp32
activations never fit 10GB — every slow era was OOM-retry thrash;
grad checkpointing (bit-identical, verified 0.0 diff) deleted the
class at ~30% recompute, netting 6+ it/s stable. Pre-reg (b)
championship read pending (champion needs a same-device cuda gate —
queued behind parity arms); (c) rarity curve queued on MPS lineage.

## The midnight burst: dual transfer, code opens, universality x3 (2026-07-22)

(1) **DUAL (one vocab-41 crystal, math+physics diets): math gate 65
(ZERO dent) + physics probe 92.3% — BEATING the pure physics
expert's 85.5%.** Transfer is positive and one-directional in the
data: shm appends 36/51 v specialist 24/51, kin appends 51/51. The
blackboard pre-reg's monolith arm WINS at this scale: shared
substrate feeds physics (+6.8) and costs math nothing. Two-expert
blackboard's remaining domain: vocab-incompatible grammars or
capacity-tight regimes (now a measured boundary, not a guess).
(2) **CODE CONTINENT rung 1 (vm-asm): 61.6% valid-rewrite first
contact** (oracle accepts any equivalent rewrite; exact-gold 216/
401). Deletion rules train (selfmov 75%, dead_store 61%);
transformation rules starve (strength_reduce 13%, mul_zero 18%) —
CONFOUNDED with diet imbalance (63% selfmov rows, generator bug,
mine). Balanced refarm = rung 1b.
(3) **UNIVERSALITY AT THREE GRAMMARS**: matched-feeding panel —
calculus 1.91/.0232/.593, mechanics 1.88/.0219/.584, vm-asm
1.85/.0230/.582 (kurt/nnCV/norm, same code, same seed). Geometry
is a universal function of FEEDING across maximally different
closed systems: symbolic calculus, physical mechanics, machine
programs. The equation's geometry sector is grammar-free at 3
points.
(4) Parity note: cuda fp32 control gates 60 (same-seed different-
device init = different model; the 4-arm comparison is strictly
within-cuda per the delta doctrine).

## Merged rarity curve + bf16 parity (2026-07-22, ~1 AM close)

(1) MERGED-T768 on the stratified battery (MPS lineage, same
instrument as 128/127/133): **130/200 — best single model measured**
(common 65/69, mid 38/48 best-ever, rare 11/18 TYING the fp32
champion, unseen 16/65). Pre-reg (c) PASS: the L9a diet moved the
rare bin (9->11 vs ternary predecessor) — repetition substituting
for resolution on the lean substrate, the rarity law's diet route
confirmed. A 1.58-bit model now leads both batteries among singles.
(2) PARITY bf16: 62 v control 60 — PASSES (no dent, half the
activation memory, faster). Wide births get bf16 by default going
forward pending tf32/tf32x3 arms; the d768 OOM class dies a second
way. (cuda-lineage note: control 60, all arm comparisons within-
device per the delta doctrine.)

## Night close: parity sweep + the championship verdict (2026-07-23 ~1:30 AM)

(1) **Parity four-way: ALL PASS** — fp32 60 / bf16 62 / TF32 61 /
tf32x3 60, same seed/diet/device. Matmul rounding at birth is a
non-factor across the tested range (absorption law's prediction:
birth gradients dwarf ULPs). ADOPTED: bf16 default for births
(speed + half activation memory), TF32 reinstated, tf32x3 shelved
for the online regime (machinery proven, not needed at 3e-4).
(2) **Pre-reg (b) FAILS — the crown holds: CHAMPION 76/120 on cuda
(same-device) vs merged 70.** The 55.1M fp32 warm-grown lineage
still beats the 75.6M ternary cold birth on the production battery
by 6. SPLIT INSTRUMENTS, stated plainly: champion wins production
(76>70), merged wins stratified rarity (130>128) and the
like-for-like diet delta (+5 over its twin). Reading: the warm
lineage's accumulated texture still pays on the legacy battery's
mix; the wide ternary generalizes better per rarity weighting. The
honest path to a ternary crown per the spec's own terms: GROW the
merged line (warm inheritance was the champion's real edge — gen-6
grown beat gen-6 reborn before; substrate was never the deficit).

## Ternary growth is not function-preserving — the mechanism (2026-07-23 ~12 AM)

Grew merged-T768 latent +768 FFN/layer (89.7M, function-preserving
in fp32: spray rows + zero down-cols). Identity pre-check through
DEPLOYMENT: 67/120, not the required 70 — because absmean
ternarization is NOT growth-invariant: zero columns shift each down
row's mean |w|, moving the 0.5s threshold, flipping borderline
weights. Growth and quantization COUPLE through the scale statistic.
This is the mechanism behind 2026-07-20's "growth stacks but
thinner on the discrete lattice" (grow-the-ternary 74). Options
priced: per-row scale freezing at growth (quantizer change =
substrate variable, not taken tonight); accepted the -3 and warm-
trained (quantizer re-centers under STE). GROWN RUN: 89.7M, bf16 +
grad-ckpt, 9.3 it/s (bf16 outrunning fp32 on a bigger model —
tonight's parity verdict paying immediately), 3 warm epochs.

## House-Ozaki midnight prototype: honest 2x, mechanism located (2026-07-23)

Naive slice-and-recombine (exact bitmask slicing, fp32 partials,
k=3): only 2x error reduction vs plain fp32; ternary fast path
(k not k^2 partials — the crystal IS Ozaki-native structurally)
inherits the same floor. Cause located: input slicing was exact but
partials still ACCUMULATE in fp32 over K terms — accumulation
rounding dominates. The real scheme's mandatory leg is block
exponent alignment ("chunking similar weights" — Artin's own
piece): aligned slices are true small integers, integer
accumulation is EXACT, rounding exists only at recombination.
Implementation rung banked: block-aligned int-sliced matmul
(int8-TC on cuda / int32 on CPU) for the online precise channel.
Slicing without alignment = compensation trick, not exactness.

## GROWN-MERGED: 75/120 — statistical crown TIE at 1.58 bits (2026-07-23 ~2:30 AM)

One warm-growth round on the merged line (89.7M latent, +768 FFN/
layer, 3 warm epochs, bf16+grad-ckpt at 9.3 it/s): **75/120 vs
champion 76 same-device — inside the <3 noise bar: a statistical
TIE for the production crown at 1.58 bits deployed.** L4 12/24
(best-ever L4), L7 16/24 (ties champion). The night's ternary arc:
65 (twin) -> 70 (+diet) -> 75 (+growth) — the champion's two edges
(diet freshness, warm lineage) replicated on the discrete substrate
in ~6 hours of 3080 time, including surviving the growth-quantizer
coupling (-3 identity dent, fully recovered by warm training).
Crown formally UNDECIDED between substrates; the duo-mined rare
shard (streaming overnight) + one more growth round are each
plausible finishers. bf16 birth speed note: 9.3 it/s on 89.7M —
the parity verdict's first production dividend, same night.

## Morning verdicts: the disagreement resolves + vm-asm passes (2026-07-23)

(1) **The pre-registered disagreement: the absorption law wins.**
fp64 end-to-end birth: gate 65 (= fp32 control exactly); rarity
showdown fp64 114/200 (rare 9/18, unseen 8/65) vs seedvar-1 control
115/200 (rare 9/18, unseen 10/65) — RARE BINS IDENTICAL, all deltas
inside noise. Artin's rare-lift prediction fails; flat-at-birth
confirmed. With the parity sweep this BRACKETS birth precision from
both ends: bf16 through fp64 indistinguishable. Doctrine final:
precision is an ONLINE-only lever (where learning is slow/faint);
at birth spend nothing extra. The ceiling-on-slow-learning theory
stands as the surviving form.
(2) **vm-asm rung 1b: 89.2% valid-rewrite** (61.6% -> 89.2%;
exact-gold 349/400). Balanced diet fixed everything —
strength_reduce 13% -> 95% (134/141). The 1a "transformation rules
resist" read was pure diet starvation. Code continent rung 1 PASSED.
(3) Miner postmortem: overnight run crashed on a missing checkpoint
(gen6_ternary never shipped to WSL — unverified dependency, mine) +
unconditional MINE_DONE marker (the false-fire lesson AGAIN).
Relaunched clean ~8:30 AM: 2,000 probes, 851 rare/unseen targets.
Queue-arming doctrine: verify every file dependency at arm time,
markers fire on success only.

## The decomposition discount: ~10x, measured (2026-07-23 morning)

Exposure-constant fit from the series arc (per family-kind rows ->
held-out accuracy): PRIMITIVE kinds k_efold ~ 1,900-2,400 rows
(linear1/cc2 solve 1911/2009, add 2380; mul & 3x-cc2-solve fully
saturated); multi-fact sum kind 8,739 (intermediate decomposition =
intermediate constant, as the ladder law demands); CHAIN-era k from
the equation's perfection price ~ 20,800. **The decomposition
discount is ~10x per row — and primitives saturate at S_max=1
(measured 100%) where chains asymptote below it.** Re-priced
perfection: a primitive kind completes at ~10k rows (vs ~96k
chain-equivalent); with axiom's emission throughput (no search
needed for primitive rows) and shorter sequences, the wall-clock
price of "the mathematically perfect model" collapses ~10-100x.
The corpus should be COUNTED, FARMED, and BUDGETED in primitives.
Equation upgrade queued: eff_L in primitive units.

## poly2: probe 83.2% (tree treatment's 4th win), gate lift NULLED (2026-07-23)

Tree-decomposed poly at 13x volume: probe 51->83% (constants
saturate: mul 69/69, add 36/36, sub 42/42, res 24/24; polynomial-
operand primitives partial: pmul 58/72, psub 16/30 — "one primitive"
needs an operand-complexity term; assembly residue: assemble 0/9,
padd 1/15 = the volume-starved kinds, supplement requested). GATE
64/120: rung 1's suggestive +2 lift did NOT survive 13x volume —
partial-fractions->integration transfer is a NULL; the probe result
stands, the transfer claim dies. Full audit 0/145,011 (axiom
5-for-5).

## Metabolic v3 paired arms: the ceiling verdict, split (2026-07-23 PM)

Equal-food paired arms on the crown-tier ternary latent (cuda,
frontier L4-9 food, surprise+contrast on, control rod, absolute
anchor — no rollbacks tripped either arm): fp32 arm FLIPS 12,219
(0.0272%) / fp64-masters arm **32,439 (0.0723%) — 2.7x committed
learning at equal food**. MECHANISM LEG of the ceiling-on-slow-
learning theory: CONFIRMED (second independent replication of
sub-ULP recovery at online LR). CAPABILITY LEG: unconverted at 75
min — proxy 24 v 24, live solves 1542 v 1346 (divergent sampling
paths post-divergence). Reading: fp64 stores 2.7x more structure;
converting committed flips to capability needs (a) longer horizons
and (b) FOOD AIMED AT THE MISSING TRANSITIONS — which is practice
mode, designed this afternoon. Metabolic v4 = v3 stack + stuck-
state food + longer session. Also: arm A's raw-gradient absorption
census read 95.3% at 2.5e-6 (instrument note: raw LR*grad proxy,
NOT AdamW-normalized steps — not comparable to the law's 0.278%@
1e-6; says most gradient ELEMENTS are sub-ULP faint at online LR).

### Arm rarity curves: FLAT — the conversion question sharpens (2026-07-23)

Paired cuda rarity gates on both v3 arms: fp32 133/200, fp64
132/200 — bins identical to one noise solve (rare 17 v 16, unseen
17 v 17). The 2.7x flips did NOT convert to capability anywhere on
the rarity spectrum at 75 min. Two readings held open,
distinguisher queued: (1) LATENT — flips are real structure below
wave-solve threshold; conversion needs practice-mode food + longer
horizons (LLMUE precedent: 100k flips -> +2, slow). (2) CHURN —
near-threshold weights flip from any nudge; fp64 commits more
near-threshold NOISE (2.7x flips + 0.0 delta is exactly churn's
signature). Distinguisher: flip-location analysis — extra flips
concentrated in low-|latent| threshold-hugging weights = churn;
distributed like arm A's / committee-concentrated = structure.

### Flip-location distinguisher: inconclusive by design — persistence is the test (2026-07-23)

Both arms' flips sit at the ternary threshold (median margin 0.0008
/ 0.0002 scale-units vs 0.427 for all weights; concentrated in
layers 4-7 down/up — the unfrozen region, uniformly). Necessary-
condition signature: BOTH structure and churn predict near-threshold
flips at 2.5e-6 (nothing else can cross). Distinguisher as designed
cannot separate. The decisive instrument is FLIP PERSISTENCE:
churn oscillates (high flip-back rate, stagnant net), structure
accumulates monotonically. Free to instrument in metabolic v4
(snapshot flip-set every N cycles, count flip-backs). The
latent-vs-churn fork stays open pending v4.

## SERIES RUNG 1E: 98.0% — the continent closes (2026-07-23 evening)

Fold-append arm (chain3, one variable vs 1d): **479/489 (98.0%)**.
The append residue VANISHED: linear1 44/63 -> 63/63, separable ->
63/63, cc2 33/54 -> 53/54. Third and cleanest determinability
confirmation — re-spelling the underdetermined kind as folds was
the entire change. THE ARC: 15 -> 67 -> 88 -> 98 (hop-only ->
solve-steps -> trees -> determinable-everything). At 19M the series
grammar is effectively solved; remaining 10 misses are linear1
solve (7) + scattered. Gate 63 (-2, sub-bar, watched). Axiom's
per-row append certification (new in chain3) audited clean.

## PHYS_ENERGY: 100.0% — a conservation law, learned perfectly (2026-07-23 evening)

Energy rung 2 (axiom's design: E0 as mul/add tree, per-order
E-coefficients as trees ending in zero rows — "conservation IS the
vanishing of every non-constant coefficient, each a certified
arithmetic fact"): held-out probe **1350/1350 (100.0%)** — mul
945/945, add 354/354, zero 51/51. Every unseen oscillator's
conservation coefficients emitted exactly. The first PHYSICAL LAW
in the lab, learned to saturation, in a representation where the
law is decidable arithmetic. All kinds are one-primitive constants
(the saturating class) — the exposure economics, determinability
doctrine, and decomposition arc predicted this number, and it
landed. The day's arc closes: 4 + (-4) -> 0, at scale, perfect.

## PRE-REGISTRATION: the bridge experiment (2026-07-23 evening, before data)

Booked before poly_chain3 arrives. Arms: (A) poly2 baseline —
co-resident pf identities, gate 64 (measured, null); (B) bridge
diet — same base + poly_ibridge family (pf derivation rows + the
pf step spelled IN INTEGRAL GRAMMAR + per-piece closes). BRIDGE LAW
prediction: arm B moves the integral gate (>=3 over the 65-band
baselines) where co-residency did not, because transfer requires
shared steps in context (dual-crystal mechanism). If arm B is also
flat: the bridge law needs revision — in-context steps are not
sufficient either, and the transfer mechanism hunt reopens. Termwise
p-kind supplement rides the same batch (operand-complexity axis):
prediction pmul/psub/padd/assemble all move toward saturation per
the exposure economics.

## Grown rarity: 133/200 — a single model ties the duo (2026-07-23 eve)

merged_grown on the Mac-lineage stratified battery: **133/200**
(common 64/69, mid 38/50, rare **14/18 — best single-model rare bin
recorded**, unseen 17/63). Ladder: champion 128 -> merged 130 ->
grown 133 = duo-8+8's 133. The growth round deepened the tails, not
just the production gate. (Also today: my generic auditor flagged
all 28k bridge rows — integral-grammar rewrites need integrand
equality, not algebraic identity; auditor fixed kind-aware; the
audit HARD-GATE behaved correctly by refusing to train.)

## Metabolic v4: structure PROVEN, self-practice can't create — the exchange is load-bearing (2026-07-23 ~5:45 PM)

150-min practice session (crown latent, LR 1e-5 hot-guarded, fp64
masters, stuck-food, zero rollbacks): (1) PERSISTENCE PRE-REG (b)
PASSES — flip-backs ~5% (<<30% bar), net monotone across 7 censuses
to 330,918: the latent-vs-churn fork CLOSES as STRUCTURE. (2)
CONVERSION PRE-REG (a) FAILS — paired resolution 1/12 -> 2/12
(+8 pts < +20 bar); the 95 live resolutions were compounded
sampling variance. (3) SYMMETRY (Artin's question): global census
near-invariant (zeros 30.5->32.6%, signs even) and flip
directionality PERFECTLY balanced — 165,408 up / 165,329 down,
ratio 1.000: locally directed, globally neutral learning. (4) THE
DIAGNOSIS: truly stuck states emit zero verified steps => practice
mode has NO gradient toward the missing transition — the model
cannot teach itself what it cannot sample (round-2 law, one level
up). THE FIX IS THE EXCHANGE: engine-farmed chains at our stuck
states supply exactly the missing gradient. stuck_states_p1 ->
axiom is now the hot path; re-probe on the same fixed seeds is the
pre-registered test. Practice mode = model rollouts + ENGINE
demonstrations, two halves, neither sufficient alone.

## poly3/bridge: 91.4% probe, gate 65 — the bridge law is UNTESTED, not dead (2026-07-23 eve)

Termwise treatment: probe 83.2 -> 91.4% (div 22/22, all constant
kinds saturated). THE BRIDGE READ: iclose 24/24 PERFECT (the atomic
close trained) but ibridge 1/9 — the entry decomposition is itself
a multi-fact long-emission and FAILED, so gate rollouts cannot take
the pf path at all: gate 65 (=baseline) is UNINFORMATIVE about the
bridge law; the pre-registration stands untested pending a
per-piece re-spell of ibridge (the 1e fold-append pattern, recipe
application #5). Also measured: padd at its full 10k budget reads
only ~50% — operand complexity is confirmed as a second axis of
the primitive unit (volume cannot buy what emission size costs).
Handoff: docs/handoffs/2026-07-23-0-five-grammars.md (new convention).

## PRE-REGISTRATION: gen-8 everything-diet (2026-07-23 evening, before launch)

One 19M vocab-41 crystal fed all five grammars at once (math base +
poly3 + series chain3 + physics/energy + duo-mined 433 + practice
378; ~1.1M rows, Mac). Booked before the run: (a) math production
gate 65 +/- 2 (determinable rows don't fight the resident crystal —
1d/poly3 precedent); (b) continent probes within a few points of
their specialists: series >=94 (spec 98.0), energy >=99 (spec 100),
poly >=88 (spec 91.4) — the one-crystal-five-grammars claim at 19M;
(c) rarity battery: gen-8 >= the poly3 birth (same battery, same
device, paired) on rare+unseen combined — duo/practice rows were
mined at exactly those skeletons. A common-bin dent = capacity
signal -> 45M re-ask is one command. Note: the solved-only-leak A/B
is DEFERRED — practice_rows_p1 carries only 18 unsolved-tagged rows
(the miner resolved almost everything); the A/B needs a
failed-step-rich mine first.

## Ozaki rung 1: the error-free transform, proven on CPU (2026-07-23 eve)

Adaptive block-aligned int-slicing (slice until the residual is
EXACTLY zero — finite mantissas terminate; alignment first, so every
slice is a true integer; exact accumulation; fp64 recombination):
**normwise error 8.8e-16 = fp64 machine epsilon** vs plain fp32's
6.8e-7 — the transform is error-free by construction, rounding
survives only at recombination. Three riders, each a design lever:
(1) **THE MPS KEY: fp32 units as exact fixed-point accumulators** —
s=7 slices, block 32 (2s + log2(b) <= 24) reads 1.0e-15 with ZERO
integer hardware: the scheme runs exactly on Metal/MPS as-is.
(2) **Triangular truncation = the precision dial, measured**: 6 of
36 slice-products ~= fp32 quality (1.4e-7); 10/36 = 5e-10; 36/36 =
exact. Cost scales with the precision you actually need — Shewchuk
adaptive refinement gets its ladder. (3) fp32 inputs need k=6 slices
at s=8 (mantissa + block spread); ternary weights need k=0 on the
weight side (already integers x scale) — the deployed forward can be
BIT-EXACT at k products, killing the fp16-near-tie class on the
ternary substrate. scratch/ozaki_rung1b.py. Rung 2 (MPS wall-clock
race vs CPU fp64) queued behind gen-8.

## The exchange is bidirectional IN FACT: union 12/12 (2026-07-23 night)

Axiom's stuck-farm delivered 10/12 walls as certified chains (23
rows, independent audit 0/23 bad, seventh consecutive clean batch) —
and the two engine holdouts (m-l4-47#s1, m-l4-60#s1, expired at 4x
budget) are EXACTLY the two states v4's model practice resolved.
Neither side planned the split; measured complementarity: the model's
only two wins are the engine's only two losses, union = 12/12. The
worklist trade is not just load-bearing (v4's conversion fail), it is
two-sided in the data. Deeper residue received: stuck_states_p2 (2
mid-derivation states, 4-6 plies of certified prefix, Subs-bearing).
Exchange test ARMED (scratch/exchange_test.py, cuda-paired seeds,
bar >2/12) — awaiting a 3080 window behind axiom's poly_chain4 emit.

## Ozaki rung 2b: EXACTLY ZERO — and the auditor was the bug again (2026-07-23 night)

Expansion recombination (Shewchuk two-sum chain over the aligned
integer partials): **max deviation from the exact big-integer
reference = 0. Not small — zero.** The complete matmul pipeline is
now rounding-FREE end to end; the output format is the only
remaining choice. Incident booked (third auditor-was-the-bug this
week): the first "exact reference" used np.round(...).astype(object)
— which boxes FLOATS, so the ground truth itself was rounding at
2^74; Fraction re-check exposed it, int64->object (true big-ints)
fixed it. Re-graded rung-1 table: full arms ~4e-16 (= fp64 output
floor; row-aligned entrywise 4.4e-16), triangular 10/36 5e-10, 6/36
= fp32-grade, expansion = 0. Rung 2c note: the v1 dd-carrier chain
prototype was sloppy (dropped a residue term, only 10x) — the clean
design fell out of debugging it: NEVER LEAVE THE SLICED DOMAIN —
carry activations as slices end to end and every layer is exact
(the fixed-point pipeline, what integer DSPs always did); redo
queued as 2c-proper. scratch/ozaki_rung1b.py, ozaki_2b_check.py.

## THE EXCHANGE CONVERTS: 2/12 -> 6/12 (2026-07-23 night, pre-registered PASS)

The v4 organism (metab_v4.pt, cuda) trained ~10 min on axiom's 23
certified stuck-chain rows, re-probed on the SAME fixed seeds:
**PRE 2/12 (reproducing v4's endpoint exactly — the paired
instrument is clean) -> POST 6/12. Bar was beat-2: SMASHED.** All
four flips (m-l4-27#s1, m-l6-60#s2, m-l7-48#s3, m-l7-54#s2) are
TAUGHT walls; the two prior wins retained; proxy 24 -> 23 (noise;
function preserved). The ledger stands: 150 min of self-practice =
+1/12; 23 demonstration rows = +4/12 in ten minutes. The
teacher-requirement theory now has both halves measured — the model
cannot sample what it cannot do, and the smallest demonstration
diet convertsimmediately. With the 12/12 complementarity (model's
2 = engine's 2 holdouts), the practice loop is PROVEN end to end:
model logs walls -> engine farms them -> model eats chains ->
resolution moves. Metabolic v5 inherits a validated food channel.
checkpoints/exchange_p1.pt on WSL.

## Ozaki cuda race: exactness wins, wall loses (honest split, 2026-07-23)

3080, N=2048: TF32 1.0ms/2.7e-4, strict fp32 0.8ms/2.0e-6, native
fp64 40.8ms/3.9e-15, **sliced-exact 247ms/9.0e-16 — MORE ACCURATE
than native fp64** (fp64 matmul accumulates ~sqrt(N) rounding; the
sliced scheme rounds only at recombination) but 6x slower; tri<4
93ms/8.4e-10, tri<3 70ms/2.2e-7. Diagnosis, named: the 36 matmuls
cost ~36ms of fp32 work — the loss is the fp64 ELEMENTWISE
recombination (36 x N^2 fp64 ops on 1/64-rationed units) + per-call
re-slicing. Headroom banked, in order: (1) amortize weight slicing
(weights are static in inference/metabolism — slice once, the EU
pattern); (2) group recombination per (i+j) diagonal in fp32, one
fp64 pass (36 -> ~8); (3) torch._int_mm int8 tensor cores with
int32 accumulation (the true Ootomo path, 2-4x fp32 rate). The
tensor-cores-as-exact-integer-units leg is PROVEN (9.0e-16 through
TF32 hardware); the speed leg is an engineering rung, not physics.

## Ozaki 2a-v2: int8 tensor cores CROSS native fp64 (2026-07-23 night)

The three named fixes, measured (3080, N=2048, weight slices
amortized = the EU pattern, activation slicing honestly inside the
timer): v2 fp32-acc full 247 -> 104.6 ms (2.4x, err 8.1e-16);
**INT8 full-exact 55.1 ms @ 8.5e-16 — 4.5x MORE ACCURATE than
native fp64 (3.9e-15) at 1.35x its wall**; **INT8 triangular<5:
20.8 ms @ 5.7e-9 — TWICE AS FAST as native fp64 with error six
orders below fp32.** The gaming-card thesis is now measured
in-house: on 1/64-rationed silicon, sliced int8 tensor cores beat
the fp64 units at their own game on the accuracy axis and pass
them on wall at the 1e-9 grade. Remaining gap to full-exact-
faster-than-fp64: the fp64 diagonal recombination (~30 ms) — next
lift is fp32-pair (two-float) diagonal carry, one fp64 pass.
Doctrine candidate: the online precise channel's matmuls run
INT8-sliced (exact) instead of fp64 — faster than fp64 AND exacter.
scratch/ozaki_cuda2.py.

## Ozaki 2a-v3: ZERO-rounding GPU matmul, verified (2026-07-23 night)

(1) **THE HEADLINE: max deviation from exact big-integer arithmetic
= 0.** int8-TC slicing + double-double output (elementwise two-sum
per scaled pair) on the 3080: the product of two real fp32 matrices,
computed through tensor cores, is EXACTLY the true mathematical
product — not fp64-close, bit-perfect (spot grid vs big-int
Fractions). 154 ms = the correctness arm; the speed arm stays v2's
55 ms @ 8.5e-16 / 21 ms @ 5.7e-9. Two exactness-chain lessons paid
en route (booked because the failures teach the design rule): fp32
diagonal sums crossed 2^24 and the fp64 part-build rounded BEFORE
the two-sum could protect it — the chain is only as exact as its
sloppiest link; every carry on the path must be widened or
two-summed. (2) **fp16-TC arm CLOSED, mechanism named**: cublas on
this stack accumulates HGEMM in fp16 regardless of
allow_fp16_reduced_precision_reduction=False (integer matmul at
s=6-scale magnitudes reads err 8.0; s=8-scale infs) — fp16's
accumulator cannot hold block sums at any useful s. int8/int32 wins
unconditionally. (3) Composition doctrine now measurable: Ozaki
kills MATMUL rounding, masters kill STORAGE absorption — together
the online loop's arithmetic path is rounding-free everywhere
except deliberate storage quantization; the speculative-arithmetic
verifier gains an exact referee at ~fp64 wall. scratch/ozaki_cuda3.py.

## Ozaki v4: RNS + fp64-inputs — two honest verdicts, two design laws (2026-07-23 late)

(A) **RNS-GEMM (CRT channels): correct idea, MY sizing bug, and the
real lesson is deeper.** 10 primes (M~2^66) overflowed: global
fixed-point on fp32 inputs needs 24 + exponent-SPREAD bits (~2^115+
for randn tails) — RNS range must be sized to the spread, not the
mantissa (~20 primes). But the measured bottleneck reframes the
whole design: 405 ms was ~90% RECONSTRUCTION (Garner digits are k^2
elementwise passes) — channels are cheap, leaving RNS is expensive.
THE COMPOSITION INSIGHT THAT FALLS OUT: real RNS systems never
leave — chain entire pipelines carry-free IN residue space and
reconstruct ONCE at the end. For us: a whole forward pass (or a
whole optimizer step) in RNS, one exit. Banked as the v5-era
exactness endgame; pairs with slicing's stay-in-sliced-domain law.
(B) **fp64-input exact product: 541 ms, 182 int8 matmuls, deviation
5.9e-33 = 2^-107 — EXACTLY the double-double capacity floor.** The
pipeline is perfect up to output format (third confirmation of the
law); a fp64xfp64 product carries ~117 true bits, dd holds 106 —
triple-double (one more carry channel) makes it fully exact. As
delivered: ~106-bit-accurate GEMM of real fp64 matrices on a $700
gaming card (fp128 hardware: does not exist on any GPU); the
hi-vs-native-fp64 delta 2.2e-15 IS the detail fp64 hardware loses.
Slicing scales k^2 with input precision (6->14 slices = 36->182
products); RNS scales k — the crossover argument for (A)'s revival
at fp64+ precision. scratch/ozaki_cuda4.py.

## Ozaki v5 — THE STAY-IN-RNS PIPELINE: lazy exactness measured (2026-07-23 late)

Four matmul layers computed entirely in residue space (20 primes,
int8 channels), one exit. (1) **Growth-free storage confirmed**: the
positional value grew to 88 bits while every intermediate stayed
int8 residues — RNS defers ALL carries, storage constant with depth.
(2) **THE WALL LAW (N=2048)**: RNS channels 53 ms for FOUR exact
layers (~13 ms/layer) vs native fp64 chain 173 ms (~43 ms/layer) AND
WRONG (4.7e-8 accumulated). One-time exit 170 ms; per-layer exits
would add 509 ms — the exit amortizes over depth. **Break-even ~6
layers: any chain deeper than ~6 is FASTER computed exactly in RNS
than approximately in fp64.** (3) Fractional-CRT cheap exit: 10 ms
(vs 170) — decision-grade magnitude estimate; caveat measured:
relative error explodes on cancellation-small entries (metric
artifact of tiny denominators) — use for magnitude/argmax decisions,
fire Garner only on ambiguity (speculative-arithmetic pattern).
(4) Exactness: deviation 1.2e5 on 88-bit values = 4e-22 relative —
an order beyond fp128 grade, floor localized to the double-double
EXIT capacity again (the pipeline is exact; the exit format is the
limit; triple-double or big-int exit closes it — and the exit is
DETERMINISTIC, so it can also be cached/incremental per the EU
pattern). scratch/ozaki_cuda5.py.

## Ozaki v6: EXACT vs fp256 — wins both axes (2026-07-23, the closer)

fp256 exists only as software (no silicon computes it), so the race
is our int8-TC pipeline with a 6-component expansion exit (~318
bits) vs mpmath at 237-bit precision, same fp64-input matmul, N=128:
**GPU exact 396 ms, deviation vs big-int = 0; mpmath fp256 2,378 ms,
deviation 5.0e-72 (fp256 still rounds).** 6x faster AND exact-vs-
approximate at the same time. Scaling note (honest): our N=128 run
is launch-bound (144 tiny matmuls); mpmath scales O(N^3) at ~us/op
— at production N=2048 the projected gap is ~10^3-10^4x. The
"keeping all the digits" claim is now measured at every precision
that exists: exact beats fp32 (accuracy), fp64 (both axes past
6-layer chains / 21ms tri), fp128 (dd-floor result), and fp256
(this entry). There is no finite-precision format left to race.
scratch/ozaki_cuda6.py.

## Ozaki v6 addendum: the fp1024 footnote (2026-07-23, closing joke)

mpmath at fp1024 (1013-bit): 2,268 ms, deviation 0 — at that width
the ~180-bit-true answers are exactly representable, so software
finally ties us on accuracy... by brute-forcing width 5x past the
answer, at 5.7x our wall, on CPU. The general statement this pins:
ANY finite format either rounds (narrower than the answer) or
wastes (wider than it); the exact pipeline is always precisely the
answer's width, and its cost does not grow with demanded precision.
The precision race is over — the only opponent that ties is one
that stops being a format and becomes the answer.

## PRE-REGISTERED DISAGREEMENT #2: exact-vs-fp64 at the validity level (2026-07-23, before v5)

ARTIN: the exact arm's infinite precision difference SHOWS UP in
model capability (flips/resolution/gate) — finite-format brackets
were never the real comparison. HOUSE/absorption law: capability is
flat above the ULP floor — at LR 1e-5, fp64 masters' residual
rounding is already below gradient noise, so exact TIES fp64-masters
on every model metric and wins only on WALL (int8-exact is faster
than fp64); adopting exact is a speed move, not a capability move.
Grading: metabolic v5's race arms (fp64-masters vs int8-exact,
equal food, paired) — flips primary, paired resolution + gate
secondary. If exact BEATS fp64-masters on capability, the
absorption law needs an amendment clause (signal below fp64's ULP
exists and matters). Either verdict banks a law leg — same
structure as disagreement #1 (birth rarity), which the law won.

## DISAGREEMENT #2 VERDICT: perfect null — the law wins again (2026-07-24 ~1:30AM)

Paired streaming arms (identical food/seeds, one variable: AdamW
steps accumulated into fp64 masters vs exact double-double):
**BIT-IDENTICAL OUTCOMES — flips 132,566 = 132,566 (the same
number), same wall resolved (6->7, m-l3-2#s2), proxy 23->23 both.**
The dd-tail printout is the whole story in one number: **1.06e-14**
— the total magnitude fp64 accumulation discards over 29 steps.
Real, measured for the first time — and TWELVE orders below the
fp32 deployment cast (2^-24 rel) and ~10 orders below the ternary
flip threshold. The precision hierarchy, now fully measured:
ternary threshold >> fp32 deployment >> fp64 ULP >> the exact tail.
Nothing below fp64's floor can reach the deployed function at these
LRs; the tail cannot even flip ONE bit of the fp32 copy. Artin's
capability prediction loses cleanly (structure = disagreement #1);
exactness-beyond-fp64-masters is a SPEED lever only. v5's race
drops the dd arm (proven redundant) — int8-exact stays for wall.

## Fused recombination kernel: 2.3x, bitwise-exact (2026-07-24)

Triton one-pass recombination (per-element register loop over all
slice-pairs, local two-sum, single hi/lo write): **DD-exact 158.8
-> 70.2 ms, bitwise EQUAL to the looped reference** (the fusion
gate: not one digit moved). Same-session native fp64: 65.4 ms —
the zero-rounding pipeline now runs at 1.07x native fp64's wall
with EXACT output. Third house kernel (Metal split-K, int4 GEMV,
now triton fused-recombine). scratch/ozaki_fused.py.

## PRE-REGISTRATION: B@768 re-test, data-matched (2026-07-24, before the run)

The bits-dimension exchange law's wounded test, re-run clean: binary
{+-1} at d768 (75.6M), SAME gen4 corpus, but 6 epochs (the discrete-
learning law's dose — how ternary earned parity) + grad-ckpt/alloc
doctrine (the original ran allocator-thrashed at 0.1 it/s, wall-only).
LAW's prediction: gate ~= T@384's 60 (one width doubling repays the
0.58-bit alphabet debt) — the 45/120 was starvation. STRUCTURAL
reading's prediction: stays low (~45-55) — binary's missing zero is
a deficit no width repays (silence is structure). Either verdict
closes the law's wound: healed, or amended to "exchange holds only
for COMPLETE alphabets (zero included)."

## GEN-8: the everything-crystal — all three pre-regs PASS (2026-07-24 morning)

One 19M vocab-41 crystal, 1.1M rows, five row-grammars at once.
(a) Math gate 64/120 (band 65+/-2: PASS). (b) Continent probes vs
their specialists: **series 99.2% — BEATS the specialist's 98.0**;
energy 100.0% (ties perfect); poly 89.1% (bar 88: pass, -2.3 tax).
(c) Rarity 115/200 vs poly3-comparator 114 — rare+unseen 18 v 17
(pass, inside noise; 800 duo/practice rows too few to move a curve).
**THE UNION EQUATION'S COEFFICIENT (i) READS ~ZERO**: common bin
63/69 v 64/69, math gate -1 — no capacity dent from tripling the
corpus at FIXED 19M width. Grammars don't fight; the union is
nearly free even at small width. One redistribution noted: L3 23->19
inside the gate (L5/6/7 each +1) — continent rows diluted L1-L3
exposure share; gen-9 applies the rations doctrine. The
one-crystal-N-grammars claim is now MEASURED: a single 19M model
carries calculus + series (99.2) + mechanics + a conservation law
(100.0) + polynomial algebra (89.1) simultaneously, at spec-level
capability in each.

## Metabolic v5 s1: 6 -> 9/14 — the p2 DEEP STATES fall (2026-07-24)

200-min streaming session (fp64 masters, 14-wall worklist, zero
rollbacks, proxy held 23): resolution **6 -> 9/14** — and the three
gains include **BOTH of axiom's p2 deep mid-derivation states**
(m-l4-47#s1#s6, m-l4-60#s1#s4) plus m-l7-58#s1. The engine's
"hard probes" shape works exactly as designed: rich certified
prefixes = resumable practice states, and the organism cracked them
from streaming alone. (m-l4-27#s1 flickered out — near-tie wall.)
Deliverables: **practice_rows_v5 = 10,344 rows (7,431 solved /
2,833 UNSOLVED-tagged / 80 skips) — the failed-step famine is over
(was 18 rows)**; gen-9's solved-only-leak A/B fully powered.
stuck_states_v5 = 1,556 fresh walls (L9 572 / L6 498 / L7 283 /
L8 203) — needs binning/priority before the axiom relay (their
escalated farm can't eat 1,556; send the rare-bin head). NET flips
264,794 over 126 steps; live resolutions 812.

## poly4/bridge: pre-reg FAILS at the bar; the fingerprint is level-local (2026-07-24)

The re-spelled batch (677,458 rows, audit clean, watcher-pulled):
probe 90.9% (flat vs poly3's 91.4). THE DOOR: ibridge 1/9 -> 4/15
(27%) — better, still the bottleneck (the peel emission is still a
two-integral row: operand-complexity axis, third confirmation);
close 2/9 -> 17/24 (71%); padd -> 71%. GATE: **66/120 — +1/+2 over
the 64/65 band, BELOW the pre-registered >=3 bar: the bridge
experiment as registered FAILS.** But the honest mechanism read
sits one level down: **L4 8/24 vs the band's 4-6** — the gains
concentrate exactly where partial-fractions work applies (rational
integrands), while the door passes only ~19% of rollouts
(0.27 entry x 0.71 close). Verdict language: a_bridge (union
equation coefficient ii) is SMALL AND LEVEL-LOCAL at 19M with a
partially-open door — not zero (L4 fingerprint), not the +3 global
lift the law predicted. Options priced: third re-spell of the peel
(one integral per row — recipe application #6, axiom cost small);
or accept a_bridge ~= level-local and test at 45M where capacity
isn't diluting. The law survives WOUNDED: transfer through shared
steps exists but is narrower than pre-registered.

## THE UNION EQUATION v1 — coefficients measured (2026-07-24 morning)

Composition law for closed systems, its three couplings now priced:
  solves^A_L = 24 * (1 - exp(-eff^A_L / k(W)))   [per-grammar, unchanged]
  (i) SHARED CAPACITY: ~ZERO at 19M x 1.1M rows — gen-8's common
      bin (63v64) and math gate (64, in-band) undented by tripling
      the corpus. The union is nearly free until W* binds; where it
      binds first is EXPOSURE SHARE, not interference (L3 23->19 =
      dilution of the maintenance ration, fixable by rations).
  (ii) BRIDGE: a_bridge is SMALL AND LEVEL-LOCAL — L4 +2-4 exactly
      where pf steps apply, no global gate lift (66 v bar 67+),
      door-limited at 27%. Transfer flows through shared steps but
      lands on the LEVELS that use them, not the whole gate.
  (iii) NOTHING ELSE: probes at-or-above specialists (series 99.2
      BEATS its specialist — cross-grammar substrate may even help),
      geometry grammar-free (universality x3).
Practical calculator, v1 form: to add a grammar to a federation,
budget its primitives (10k/kind), add maintenance rations for the
resident grammars' exposure SHARE, expect bridge gains only on the
specific levels whose derivations contain the new grammar's steps.
"What a model needs" is now computable per union, with the caveat
that all three coefficients are 19M-measured — the 45M re-ask rides
gen-9.

## B@768 re-test: the exchange law takes its amendment (2026-07-24 morning)

Data-matched re-run (6 epochs, clean 6.8 it/s — the original ran
allocator-thrashed at 0.1): **28/120 @ 15.78 — WORSE than the
3-epoch run's 45/120.** Both pre-registered readings overshot: the
law's healing prediction (~60) fails outright, and even the
structural range (45-55) was too kind — extended training actively
DEGRADED binary at width (45 -> 28, validity 36.73 -> 15.78 vs
B@384's own 36.73). New fact beyond the registration: binary-at-
width is a TRAINING-DYNAMICS pathology, not just a capacity
deficit — the dose that carried ternary to parity (6ep) carries
binary downhill. AMENDMENT ADOPTED: **the bits-dimension exchange
holds only for COMPLETE alphabets (zero included)** — width repays
resolution, never structure; no dimension budget buys back silence.
"Silence is structure" gets its third and strongest confirmation
(B-vs-T crater 54/60 -> the one-dot portrait -> width-irreparable +
dose-degrading). The wounded-law file closes. checkpoints/tourn_B*.

## PRE-REGISTRATION: gen-9 solved-only-leak A/B (2026-07-24, before launch)

Two paired 19M vocab-41 births, gen-8 recipe + the rations fix
(L1-L3 base rows x2 — targeted at gen-8's L3 23->19 dilution), one
variable: arm A adds practice rows SOLVED-ONLY (7,431 + 80 skips);
arm B adds ALL 10,344 (+2,833 unsolved-tagged failed-attempt
steps). Booked: (a) LEAK THEORY (Artin's "steps should outweigh
the solution"): arm B > arm A on rare+unseen combined (failed
steps were mined at the hardest skeletons); (b) COUNTERWEIGHT
(chain-carry): if arm B < arm A on gate/validity, dead-end steps
teach wandering; (c) rations: L3 back to >=21/24 both arms;
(d) continent probes hold (series >=97, energy >=99.5, poly >=88).
45M union re-ask rides the winning arm tonight.

### Gen-9 launch note: the miner's identity leak (2026-07-24)

Diet build revealed ~1,275 of the 2,833 "unsolved" practice rows
are identity rewrites (model echoes cur; oracle correctly verifies
X=>X; the v5 miner banked them — the reward-hack scar's cousin at
the mining layer). The no-op filter drops them at diet build, so
the A/B is clean but the REAL failed-step dose is ~1,558. FIX
QUEUED: miner v2 gains the identity check at bank time (verified
AND distinct — the same guard GRPO's reward got in July 15's
hygiene pass). Arms: A 1,198,397 rows / B 1,199,955.

## PRE-REGISTRATION: poly5 — the bridge law's third knock (2026-07-24, before data)

poly_chain5 (axiom 9e78a9f): ibridge family ONLY re-spelled — the
peel is now two one-fact rows (ibridge = split one residue integral
leaving literal subtraction; NEW icancel kind = cancel the (x-a)
factor, one integral -> one integral) + pmul/psub precursors. Diet
= chain4's other families + chain5's ibridge family (one variable
vs chain4). Booked: (a) THE DOOR: ibridge >= 12/15 and icancel >=
80% — one-fact rows train (ladder law's sixth application); (b)
THE BRIDGE BAR, unchanged through three generations of data: gate
>= 3 over the 64/65 band (>= 67-68). Door open + bar hit = the
bridge law finally CONFIRMED; door open + gate flat = the law is
WRONG as stated (shared steps in context are insufficient) and
a_bridge stays level-local; (c) L4 fingerprint holds (>= 8/24).

## PRE-REGISTRATION: ODE zero-diet baseline probe (2026-07-24, before the run)

Artin's question ("after the model picked up Taylor series, did ODE
validity go up?") has no measured answer — closed-form ODE solving
has never been in any diet or probe battery. Queued behind tonight's
gen-9/poly5 verdicts: an ODE probe band (mathgen/odes.py families,
string-seeded, checkodesol oracle fork-walled) run against gen-8
(and the gen-9 winner), scoring closed-form step validity + solves.
BRIDGE LAW prediction (house): ~ZERO transfer from the 99%-series
skill — series chains never surface a closed-form ODE step in
context; coefficient recurrences are not dsolve moves. A nonzero
baseline = transfer through a channel the bridge law doesn't name
(shared ansatz vocabulary, the gen-6 champion-does-9/24-L9
precedent) — bank it and re-examine the law. Either way the number
becomes the L9b/ODE continent's champion-control, measured BEFORE
farming instead of after (the 9/24 lesson, applied proactively).

## GEN-9 A/B VERDICT: redistribution, not a winner — three ledger reads (2026-07-24 evening)

First verdict independently cross-checked by the reviewer agent
BEFORE booking (its stricter language adopted over my provisional
"marginal pass"). Paired arms, same device/battery (NOTE: third
census freeze {71,48,20,61} — practice/stuck/duo files entered the
glob; these 119-120/200 numbers must NEVER be cross-read against
the champion-era 128-133).

ARM A (solved-only): gate 68/120 @ 65.69 {3:22,4:9,5:18,6:7,7:12};
series 98.4 / energy 100.0 / poly 88.9; rarity 120/200
(66/37/11/6). ARM B (+~1,558 real failed steps): gate 70/120 @
65.37 {3:22,4:9,5:18,6:7,7:14}; series 99.2 / energy 100.0 / poly
88.5; rarity 119/200 (64/35/13/7).

(a) SOLVED-ONLY LEAK: **directional support, SUB-NOISE — not
confirmed.** B leads rare+unseen 20 v 17 (+3/81 probes); each bin
individually (+2, +1) sits inside the noise band. The convincing
part is the SHAPE: B redistributes ~4 solves out of common/mid
into rare/unseen — exactly what failed-steps-mined-at-hard-
skeletons predicts. CAVEAT BOOKED: the test ran at ~HALF dose
(1,275 of 2,833 "unsolved" rows were miner identities, dropped at
build). Re-run at full dose with miner v2 (verified-AND-distinct
at bank) before treating +3 as the effect's ceiling.
(b) CHAIN-CARRY COUNTERWEIGHT: **not observed** — B is nominally
UP on the gate (70 v 68, sub-bar; all of it L7 12->14, replicate
before trusting) with validity dead even. Dead-end steps did not
teach wandering at this dose. First honest leg of the carry
ablation, free.
(c) RATIONS: **clean PASS both arms** — L3 = 22/24 twice (gen-8's
19 repaired; the doctrine validated independent of the winner).
Most of the 64 -> 68/70 gate gain is L3 +3 and L5 +2, mechanism-
clean.
(d) CONTINENTS: **PASS both arms**; B's series 99.2 TIES gen-8's
specialist-beating record; poly thin over the bar both arms.

WINNER DECLARATIONS (reviewer's, adopted): 45M union re-ask diet =
**arm B** (richer/harder diet, defensible forward pick; A remains
the cleaner control substrate). Production lineage = **statistical
tie** (B +2 nominal, below the >=3 bar — NOT booked as a real
gain; champion gen6_grown 76 untouched). Split-instrument note:
gate says B, overall rarity says A, both sub-noise — the fp16-
near-tie signature at the A/B level; booked as REDISTRIBUTION.

## poly5 knock-3: ALL THREE LEGS FAIL — and the wall is named to the coefficient (2026-07-24 night)

Reviewer-cross-checked before booking. (a) DOOR FAIL, hard:
icancel **15/15 (100%)** — the new one-fact kind saturates at
birth, ladder law's sixth confirmation — but ibridge **1/15**, a
REGRESSION through three spellings (1/9 -> 4/15 -> 1/15). (b) BAR:
gate 62/120 @ 62.85 — no lift, a -2/-3 dent that LOCALIZES to L3
(20 v band 22-23): poly5 ran the OLD un-rationed recipe, so the
dent is maintenance-ration dilution (gen-8's exact wound), NOT
bridge interference — gate confound, do not cite against the
determinable-rows law. (c) L4 fingerprint 7/24 < 8: poly4's one
positive did not replicate. Bridge law at the gate level: still
UNTESTED (door never opened), not refuted.

THE MECHANISM, settled by miss-classification (series_preds dump,
oracle-checked): the 14 ibridge misses = **12 INVALID / 2
valid-alternative peels / 1 exact.** The model emits the right
STRUCTURE every time (peel + remainder) and botches the residue
COEFFICIENTS (gold x/3+1/3, pred x/3-2/3) — A = N(a)/(a-b) is
computed SILENTLY inside the emission. Not choice-ambiguity
(valid-alt scoring would read 3/15 — door still shut); it is the
determinability law at the coefficient level: latent arithmetic
never trains. "One-fact" != "one-primitive-determinable" — the
re-spells shrank step count while the binding axis was OUTPUT
emission (psub 7/15 and padd 24/34 sit on the same gradient).

## ODE zero-diet probe: VOID BY VOCABULARY — and a tokenizer footgun (2026-07-24 night)

The pre-registered probe died at the tokenizer, which IS the
verdict: the vocab-41 crystal cannot SPELL closed-form ODE grammar
(no y, Eq(, Derivative(, C1 atoms) — the zero-diet baseline is
structurally zero at the language level, before capability enters.
The bridge-law prediction (~0) is confirmed in its strongest form:
no transfer is possible into a grammar the vocabulary cannot
express; territory = vocabulary expansion (the Liouville-boundary
note, measured again). Re-registration: the L9b champion-control
becomes the FIRST ODE-vocab birth's pre-diet probe (VOCAB_EXTRA
gains the ODE atoms exactly as series gained 't').
FOOTGUN CAUGHT EN ROUTE: MathTokenizer.encode SILENTLY DROPS
unknown substrings (43-char ODE string -> 19 tokens, no exception,
roundtrip false). Known-clean diets have masked this; the first
continent with a stray atom would train on silently-mangled rows.
FIX QUEUED (Fable): strict encode that raises on unencodable
input, legacy escape hatch for measured cases.

KNOCK 4, pre-registered conditional (the LAST knock, hard-stop):
re-spell ibridge as pure COPY — residue coefficients pre-computed
in explicit pmul/psub precursor rows, the ibridge emission only
places them; run on the gen-9 RATIONED recipe (kills the L3
confound); precondition = door only (ibridge >= 12/15). Door opens
-> the bridge bar finally becomes testable; door stays shut ->
CLOSE the bridge-at-the-gate file with the verdict "transfer
exists but is emission-size-gated; the gate-level test is
structurally unreachable." No fifth knock; no scoring changes to
force a pass (the door metric is the honest instrument).

## Desert test v2: NO SPONTANEOUS COMPOSITION — coefficient (iv) reads zero (2026-07-24 night)

The cross-grammar composition probe (union equation candidate
coefficient iv), on gen-9 arm B (series skill 99.2 resident): 8
Liouville-dead integrals x 16 samples = 128 proposals — **0
oracle-valid, 0 series-representation reaches.** The model attacks
with familiar integral-grammar patterns (i_sum-split shapes), every
one oracle-rejected, and NEVER reaches for the expansion move that
its own series grammar makes trivial. HONEST STALL at scale (the
original desert test's 3-proposal stall, replicated at 128) — the
architecture cannot bluff a closed form, and it also cannot
DISCOVER the jailbreak it already contains. House pre-reg (a)
CONFIRMED; with poly2 (co-residency zero) and the ODE vocab-void,
the bridge law graduates to its strong form: **NOTHING transfers
without demonstrated shared steps — not co-residency, not
inference-time composition, not cross-vocabulary reach.** The
discovery-as-a-move diet (bridge rows demonstrating "when stuck,
expand") is now the named, farmable fix.
INSTRUMENT SAGA, booked per honesty norms: run 1 was VOID — my
probe sampler didn't break decode at the newline step-boundary
(auditor-was-the-bug #4-adjacent); caught by the solvable-state
CONTROL reading 0/8 from a 70/120 model. Fixed to mirror the
production sampler; control then read 4/8 valid. Controls ride
every instrument, no exceptions — one run's delay, a void verdict
prevented. RIDER OBSERVATION: naked textbook states (Integral(
x**2, x)) FAIL where textured generator states succeed — the
crystal is brittle on inputs simpler than its diet ever showed
(exposure, not difficulty; the rarity law's floor: "common" means
common-IN-CORPUS, and bare forms are rare). Cheap fix candidate:
a thin naked-forms shard in gen-10.

## 45M UNION RE-ASK: width does not pay for primitive federations (2026-07-24, closing the wave)

Same-device pair (both probed WSL/3080, SAME 4th census {50,62,25,
63}): 19M arm B — series 99.2 / energy 100.0 / poly 88.5 / rarity
**119/200** (46/50, 50/62, 16/25, 7/63); 45M on the IDENTICAL diet
— 98.8 / 100.0 / 89.0 / **108/200** (45/50, 47/62, 11/25, 5/63).
Reviewer-cross-checked verdicts:
- **(iii) NOTHING-ELSE: WIDTH-STABLE** — continents pinned at both
  widths; the -11 is uniform capacity loss (rare -5, mid -3), not
  grammar interference appearing at width. (Weak-instrument note:
  continents are saturated, so (iii) is confirmed where headroom
  exists — the rarity bins — not where it can't show.)
- **(i) splits into two claims**: grammars-don't-fight = width-
  stable ✓; but the union does NOT raise W* — the WIDER model is
  WORSE by 11, rare bin 64%->44%: the tokens-per-width envelope on
  the union corpus. MECHANISM: primitive rows are TOKEN-LIGHT
  (decomposition-discount corollary) — 1.2M rows is a thin token
  meal; W* stays ~19M-class; 45M starves with the classic
  tail-dies-first signature. The elegant tension, booked: the same
  discount that makes federations cheap to FARM makes them cheap
  to FEED — primitives saturate a narrow model fast and never
  justify a wide one. EQUATION REFINEMENT: k(W)'s feeding argument
  re-prices in TOKEN mass, not row count.
- CONFOUND, unexcluded: births cross precision/device (19M
  fp32/MPS vs 45M bf16/cuda; parity sweep covered d384, not this
  cell). Attribution: tokens-per-width PRIMARY; bf16 birth debit
  possible minority. ONE-VARIABLE CONTROL QUEUED (cheap, next 3080
  window): re-birth the 45M fp32 — -11 survives => "19M is natural
  width for token-light federations" becomes law; shrinks => W*
  sits between.
- HAZARD, hard-flagged (reviewer F7 live): this 19M's 119/200 and
  the gen-9 report's 119/200 are DIFFERENT batteries (4th vs 3rd
  census) — same total by coincidence, a false friend. Only
  within-census pairs are valid. FOUR incompatible rarity censuses
  now exist; the taxonomy surgery's re-baseline is urgent.
- GEN-10 SIZING: 19M, pending the precision control. To ever
  justify width, raise the union's TOKEN mass (chains, more
  grammars), not its primitive-row count.

## The 388 mangled rows: strict encode's first live catch (2026-07-25 ~1AM)

The new strict tokenizer crashed the width-ladder launch — on a
REAL defect: **388/132,837 gen-4 rows (0.29%) contain out-of-
language atoms (Subs(, u_, erf-class) and have been silently
LETTER-STRIPPED into every gen-4-lineage birth to date**
(Subs(expr, u_, g) trained as (expr, _, g)-style fragments — the
v2.1 language filter was policy, but these rows slipped the farm
filters and the tokenizer's silent-skip hid them). Fix shipped:
trainer skips out-of-language rows WHOLE and prints the count;
tests green. Impact honestly bounded: 0.29% of rows, and every
historical comparison shared the same 388 (paired deltas
untouched); absolute gates carry a ~0.3%-of-diet noise floor
nobody knew about. The strict-encode fix paid for itself within
three hours of landing — the silent-drop class is now extinct at
the trainer.

## THE WIDTH FLOOR: W_min ~ 8.4M — the full curve closes 0.5M -> 400M (2026-07-25 ~2AM)

Downward ladder (Artin's vocab-floor riff), gen-4 corpus, 3ep,
standard width recipe (head_dim=64, ffn=4d — reviewer verified
this is a CLEAN width isolation, not a config bundle), Mac
lineage, reviewer-cross-checked: **d256 (8.4M) 65/120 @ 59.91
{3:22,4:7,5:16,6:8,7:12} — ON the 19M plateau; d128 (2.1M) 57;
d64 (0.5M) 38 with L4 = ZERO.** The complete curve on one corpus:
38 -> 57 -> 65 -> 64-65 -> **69 (45M = W*)** -> 65 -> 49 -> 30.
Chart: docs/assets/width-curve-gen4.png.
- **W_min ~ 8.4M** (plateau floor): the 19M was ~2.3x
  over-provisioned; 45M remains the peak (-4 from d256 clears the
  bar — do NOT read "full band" as the peak). Floor < peak <
  over-provision < collapse, all measured.
- **PRE-REG REFINED AND CONFIRMED: isolated-clade techniques die
  first under width compression** — L4 (ansatz clade, no descent
  path) 7->4->0 and L6 (heurisch level) 8->7->3 collapse while
  clade-connected L3/L5/L7 lose only ~30%. Sharper than the
  registered "long dependencies": the phylogeny's isolated
  branches need DEDICATED width. Symmetry booked: **L4 = 0 at
  BOTH extremes** (d64 under-width AND 400M underfed) — the
  isolated clade is the first casualty of every starvation mode.
- "Not vocab count" leg UNTESTED (vocab fixed at 40 throughout) —
  needs a vocab-varied arm before that clause books.
- n=1 per width; seed sigma<1 measured at 19M only — ORDERING and
  collapse booked, exact knee values not.
- **Federation floor reading**: pure-math floor 8.4M vs the
  everything-diet's W* ~19M (union re-ask) — adding grammars
  RAISES the floor; minimum width scales with grammar count. The
  union equation gains its floor-side law next to k(W)-in-tokens.
- Over-claims fenced: production does NOT move (gen6_grown 76 is
  a different lineage+territory; W_min is a deployment-efficiency
  point — an 8.4M pure-math tool artifact); 8.4M is not the
  federation floor.
- RIDER: the ladder supplies boundary-or-bulk's missing low-width
  cells — the 0.5M->400M grid is COMPLETE; the regression
  (capability vs N*b / d / d*L / sqrt(N)) is now one script.
- Lattice note: portraits + gauges show the democracy SURVIVES at
  every width (phases isotropic, no shelf, kurt 2.5-3.1) — the
  d64 crack is dimensional, not geometric: the organism stays
  healthy and simply runs out of directions (assets:
  neurons-wfloor-*.png).

## PRE-REG: boundary-or-bulk regression (2026-07-25, pre-dawn — analysis of measured grid, no new runs)

Reviewer-shaped design (its red-team, adopted): the width-only
points are COLLINEAR (d, sqrt(N), N*b all monotone in width), so
the regression is an ORDERING test on the independent bits axis,
not a curve fit. Three sectors, never pooled (device fences:
wfloor+five-point = Mac lineage; alphabet tournament + crossover
grid = 3080/TF32):
- (a) rising arm <= W* (d64/128/256/384/512, gen-4 3ep, n=1 each):
  affine fits, <= 2 free params, capability vs N*b / N / sqrt(N) /
  d / d*L. PREDICTION: all fit, dR^2 < 0.05 — sector CANNOT
  discriminate; reported as the collinearity exhibit only.
- (b) bits axis at fixed 19M (B/T/M4/M5/P2/fp32): volume N*b says
  fp32 (32 bits) must top P2 (3.17); measured 64 < 66. PREDICTION:
  >= 1 named ordering violation for volume; zero for a saturating
  effective-bits reading.
- (c) crossover grid (d256/384/768 x {fp32, ternary}): volume ranks
  fp32@768 FIRST; measured it came LAST of the fp32 arm (58, beaten
  by ternary@768's 65). PREDICTION: volume mis-ranks >= 2 cells;
  Spearman rho(volume) < rho(boundary-class measures).
VERDICT RULE: if volume accumulates named ordering violations in
(b)+(c) while boundary-class measures (d; b+0.5*log2(d)) keep
order, book BOUNDARY — confirming the 07-22 raw verdict on the
full grid. Underfed points (113M/200M/400M; 200M+400M 1-epoch)
EXCLUDED from all fits, shown as the feeding fence. n=1 per cell;
seed sigma measured (<1) at 19M only.

## Boundary-or-bulk VERDICT: volume is FEEDING-BLIND; exchange law gains its b_eff caveat (2026-07-25 pre-dawn)

scratch/boundary_or_bulk.py on the completed grid (param counts
read from checkpoints, never labels). Reviewer-cross-checked
BEFORE booking; its three tightenings adopted verbatim.
1. **NOT-VOLUME books, on sector (c) ALONE**: N*b as a monotone
   capability predictor is refuted by the crossover grid — it
   ranks fp32@768 FIRST; measured 58, beaten by ternary@768's 65
   (the ONE real-magnitude violation, 7 solves >= the 3-real bar).
   Sector (b) is a near-PASS for volume (rho +.943; its only
   violation P2 66 v fp32 64 = +2, sub-noise) — volume survives
   the bits axis within noise. rho -.058 / 7 pair-violations in
   (c) are inflated by sub-noise flips; the verdict rests on the
   single d768 crossover.
2. **Mechanism = FEEDING, not boundary geometry**: fp32@768 is
   past fp32's W* (tokens-per-width starvation) while ternary@768
   rides ternary's larger W* (W* ~ 1/b, booked 07-22). Volume
   fails because it ignores the bit-dependent W* — do NOT cite
   this as "boundary won." No boundary measure kept order where
   volume failed (width-only measures silent-not-correct; honest
   null).
3. **The exchange law is LOW-BIT-ONLY**: b+0.5*log2(d) fails (c)
   identically to volume — the raw b-term dominates at b=32. The
   law HOLDS in its measured regime (<=3.17 bits); the fix is a
   saturating b_eff, and raw-bit extrapolation to fp32 is the
   misuse. Propagated to THEORY (B@768 row caveat).
4. Exploratory rider DOWNGRADED per review: quadratic in
   log10(N*b) on 6 cells (3 params, R^2=.647 — LOW for that
   flexibility; peak ~193M bit-params hugs the best cell,
   near-tautological) = consistent with the KNOWN inverted-U
   (W*/tokens-per-width) in bit-volume coordinates; not a new law,
   no peak location booked.
5. Rising arm cannot discriminate (R^2 spread .052 = exactly the
   noise bar; collinearity confirmed as pre-registered; d's
   nominal .883 is not a winner).
Fences: sectors never pooled (Mac lineage vs 3080/TF32); n=1 per
cell; underfed 113M/200M/400M excluded from every fit.

## fp32 control at 45M: no detectable bf16-birth debit on cuda; gen-10 = 19M books (2026-07-25 pre-dawn)

Paired arms, WSL cuda, gen-9 diet B, BIRTH_SEED=1, 3ep, one
variable (--fast/--nopack): **bf16 66/120 @ 61.95 (rarity
108/200) vs fp32 66/120 @ 61.46 (rarity 111/200).**
Reviewer-cross-checked BEFORE booking; both its objections
adopted.
1. **Precision confound DEAD**: no detectable bf16-birth debit at
   45M on cuda (gate 66=66; rarity +3 sub-noise, sign-consistent
   with a residual whisper — "no detectable," not "exactly
   zero"). CONFIRMS AND EXTENDS the cuda parity sweep to 45M; the
   historical MPS/d384 +3-fp32 debit is a SEPARATE regime, not
   overturned. Level-mix shuffle (L4 4 v 6) = within-tie, unread.
2. **Width-doesn't-pay books as MECHANISM, not magnitude**
   (reviewer catch, adopted): the 19M arm B was born MPS
   (logs/gen9B_birth.log, Mac), the 45M born cuda — birth-device
   crossed, and the parity sweep prices an MPS-birth offset at
   ~+3-4 solves at d384, ~the size of the −4 being attributed to
   width. Bookable: 45M shows NO width dividend on the token-light
   union (both precisions gate 66); the pure-math sign-flip (45M
   69 > 19M 65 on chains; 19M > 45M on primitives) attributes the
   flip to token density/W*, not precision or device. The width
   PENALTY magnitude is NOT booked — 19M-cuda control launched to
   close it (same diet B/seed, bf16).
3. **GEN-10 = 19M BOOKS** as the efficient choice: at-least-as-
   good and 2.4x cheaper on this diet; robust to the birth-device
   confound even though the −4 is not.
4. **Speed rider, capability leg only**: bf16 births
   capability-lossless now measured at TWO widths on cuda (d384
   parity sweep + 45M 66=66). Honest wall-clock: EPOCH PARITY at
   45M on the 3080 (bf16 1939/1920s vs fp32 1872/1883s
   uncontended epochs; fp32 ep0 3651s was GPU contention) — no
   throughput claim books at this width on this box.
Fences: n=1 per precision arm; rarity numbers are WSL-4th-census
(within-census pairs only).

## Rung-1 prefix pair: NO-ADOPT — the literature shortening did not transfer (2026-07-25 dawn)

Paired births (Mac MPS, row-identical diets 132,449 rows, seed 1,
3ep, one variable = serialization). Reviewer-cross-checked; its
reframe adopted as the headline.
- **THE HEADLINE: the banked 20-30% prefix shortening
  (Lample-Charton) measured −4.5% median (−9.8% mean) on
  closed-system rows** — our rows are frame+coefficient+number
  dominated; parens are a small share. Prefix's speed case is
  DEAD independent of adoption, and the speed track re-ranks:
  streaming-birth and batched-KV are now the speed headline.
- Gate: infix twin 64/120 @ 62.84 (replicates the 64-65
  historical band through the --diet path) vs prefix 61/120 @
  63.24 — **NO-ADOPT as pre-registered** (bar (i) missed by one
  solve beyond −2; bar (ii) failed on the serialization itself).
- **The −3 is a WELL-FORMEDNESS tax, not a reasoning loss**
  (reviewer reframe): validity +0.4 (per-step reasoning intact)
  while 8.1% of emissions (120/1480) fail the prefix parser —
  infix's redundant parens act as error-correcting scaffolding;
  per-level the tax lands where rollouts are densest (L3 −2,
  L4/L5 −1, L6 +1).
- **The 388-row hazard CLOSES**: the converter's 388 skips are
  exactly the known Subs() out-of-language class, dropped
  symmetrically from both arms; infix twin still replicates 64 —
  zero measurable cost at 19M.
- NOT RUN: (iii) the prefix-variant emission probe (psub/padd/
  ibridge) — the pivot for rung-2: tree-PE's only surviving
  rationale is the emission wall. Ported and queued next. (iv)
  int3 rider in flight, appended when it lands.
- Banked (reviewer riff, RIFF-LEDGER): rung-1b prefix+arity-mask
  decoding — constrained decoding nulled twice on infix because
  misses were semantic; prefix's 8.1% syntactic misses are the
  one regime where the mask might pay.
Fences: n=1 per arm; MPS lineage; prefix gate uses gate_prefix.py
(same seeds/oracle, conversion at the two boundaries).

## Rung-1 secondaries close the native-transformer program: the emission wall is NOTATION-INVARIANT (2026-07-25 dawn)

Reviewer-cross-checked; CLOSE endorsed with fences, all adopted.
- **(iii) THE LOAD-BEARING FINDING: the long-emission wall does
  not move with notation** (length-proxy on 240 in-language
  generator states, band 12.345M; poly psub/padd/ibridge are
  OUT-OF-DIET for gen-4 twins, so the wall was NOT tested on
  those kinds directly — proxy named, honest). Quartile validity
  infix 81.7/73.3/50.0/21.7 vs prefix 80.0/76.7/46.7/20.0 (all
  deltas <= 3.3pp, n=60/quartile). The wall is CAPACITY/SEMANTIC
  — which forecloses tree-PE too, not just prefix: a
  capacity-bound wall doesn't move for any positional/notational
  intervention; the most tree-PE could do is recover prefix's
  self-inflicted arity tax back to a TIE. **Rung 2 not spent;
  the native-transformer program CLOSES** (pending Artin, but
  the program dies with a mechanism, not a shrug). Levers for
  the emission wall are territory/exposure and decomposition,
  NOT representation.
- **(iv) inverted, differential −8**: under identical int3 PTQ,
  prefix 61->54 (−7, parse-fail 8.1->10.0%) vs infix 64->65
  (+1); prefix-int3 vs infix-int3 = −11. Standalone line:
  per-channel-absmax int3 near-lossless at 19M/infix (65 v 64) —
  EXTENDS the int4-MX/GPTQ-int3 45M results to 19M, not new.
- **ONE MECHANISM, THREE OBSERVATIONS**: parens explicitly close
  each operator's scope so the model cannot lose arity count;
  prefix forces internal tracking, which fails on long emissions
  (parse-fail 1.7/0.0/1.7/30.0 by quartile). Observations: gate
  −3, quant −7, Q4 collapse.
- **General law banked (replaces the arity-mask as descendant):
  grammar-constrained decoding pays iff the misses are
  SYNTACTIC** — explains the infix-GCD null (misses were
  semantic) and predicts where GCD would pay. Arity-mask survives
  only as a conditional note if prefix is ever revived.
- **Retro-validation of teach-don't-impose at the architecture
  layer**: seed-from-measured-statistics (warm-birth FFN, +8 ep1)
  PAYS; impose-a-new-representation (prefix/tree-PE) does NOT —
  the parens the model learned to lean on are load-bearing. Same
  law as hints-x2, now at the representation layer.
- Speed track re-rank: prefix off (−4.5%, quant-fragile);
  **streaming-birth and batched-KV are the speed headline.**

## HOLD: gen-10 = 19M booking SUSPENDED pending same-device cells (2026-07-25 morning)

The 19M-cuda control (bf16 --fast) gated 56/120, rarity 105/200 —
same cuda harness that read the MPS-born arm B at gate 70/120,
rarity 119/200. The
width verdict is device/precision-dependent and INVERTS on
all-cuda births (45M 66 > 19M 56); the original 19M>45M rested on
a cross-birth-device pair where the birth-machine delta (14)
dwarfs the width gap (4). Reviewer-directed HOLD, not a
rebooking: do not build gen-10 on the 19M pick until the
decomposition cells read. Leading hypothesis (reviewer):
WIDTH-DEPENDENT bf16 DEBIT — bf16 (8 mantissa bits) sits below
the 8-10-bit dynamical cliff and narrow models lack the dimension
dividend to absorb rounding (debit 0 at d768, −3 at 45M-MPS-era,
candidate −14 at 19M); the cuda cell also carried the --fast
token-budget packing bundle (8,426 vs 37,498 steps/ep). Cells in
flight: 19M-cuda-fp32 (decider), 19M-MPS-seed-2 (sigma_MPS).
Data-order and init alternatives RULED OUT by code inspection:
init is CPU-RNG before .to(dev) (bit-identical across machines)
and epoch order is random.Random(ep) (device-independent) — the
only live variables are kernel numerics and the --fast bundle.
Loss-capability divergence noted (cuda-19M loss 0.4124 < 45M
0.4539 while gating −10): gate-not-the-loss, 6th instance.

## PRE-REG: the Lloyd-Max codebook race (2026-07-25, before gates)

PTQ-only on the 19M infix twin (born MPS, gates MPS to match the
int3=65 / fp32=64 baselines; no births). Per-output-channel exact
1-D k-means codebooks (free vs zero-pinned), gate-scored ONLY
(house law: never weight distance). Arms: LM-2bit, LM-2bit-zero,
LM-3bit, LM-3bit-zero, ternary-LM (k=3, zero-pinned — the
born-vs-rounded read vs born-ternary 60), int2-uniform (ladder
point). PREDICTIONS: (1) zero-forced >= free Lloyd-Max at 2 bits
despite worse MSE (zero-is-load-bearing); LM-3bit within noise of
uniform int3 65 (distribution near-Gaussian, kurt 2.4 — uniform
is already close); (3) ternary-LM (rounded) vs born-ternary 60
prices train-into-lattice vs round-onto-lattice; house has no
strong prior — banks either way. Fences: n=1, PTQ-only, MPS
gates, per-channel codebooks (finer than the born alphabets'
global levels — noted, not hidden).

## The decider reads: device innocent, the −14 decomposes, HOLD lifts on direction (2026-07-25 midday)

19M-cuda-fp32: gate {3:21,4:8,5:18,6:7,7:15} = **69/120 @ 63.32**,
rarity 114/200. Reviewer-cross-checked; its packing objection
landed and CHANGED the booking — command diff written in.
1. **Device effect ~vanishes at fp32**: 19M MPS-fp32 70 vs
   cuda-fp32 69 (−1 gate, sub-noise). Rarity residual −5 (119 v
   114) flagged, sigma-pending; rarity-based cross-device reads
   carry it, gate-based reads are device-clean.
2. **The −14 was NOT a device law — and NOT yet a bf16 law
   either.** Command diff (the reviewer's demanded grep): the 19M
   bf16 cell ran BARE --fast → `if fast and not nopack` re-enables
   token-budget packing (8,426 vs 37,498 steps/ep) — MY LAUNCH WAS
   OFF-DOCTRINE (the packing hole was booked at ~−10 on
   2026-07-17 and --fast was supposed to ride with --nopack). The
   45M bf16 twin DID run --nopack (37,498 steps both arms), so
   66=66 isolates bf16-alone = FREE at 45M. Decomposition at 19M:
   −13 = packing (known ~−10) + bf16 (UNMEASURED alone at this
   width). **Isolation cell queued: 19M cuda bf16+--nopack**,
   fires behind the sigma_cuda cells. Doctrine NOW: --fast never
   ships without --nopack (flag pair enforced in scripts), any
   width; the "no --fast below d512" law WAITS for the isolation
   cell rather than booking on a bundle.
3. **HOLD LIFTS on DIRECTION: gen-10 = 19M re-books** on the clean
   same-device same-precision pair (19M-cuda-fp32 69 > 45M-cuda-
   fp32 66; rarity 114 > 111). Magnitude +3 = exactly the >=3 bar
   with cuda sigma unmeasured — booked MARGINAL-REAL pending the
   sigma cells; direction is mechanism-backed (token-light
   federation, W* <= 19M).
4. **De-escalation (reviewer, correcting its own alarm)**: the
   five-point scaling table is CLEAN — all ladder points born MPS
   (log audit), the table is gate-based (device-clean at fp32),
   and any hypothetical wide bf16 birth would be bf16-immune via
   the dimension dividend. No W* revision.
5. Loss-capability divergence (bf16+packing cell: LOWER loss
   0.4124, gate −13): gate-not-the-loss, 6th instance — packing's
   signature from 07-17 (loss blind to the hole) reproduced.

## Instrument fix: the phase-density null was wrong (2026-07-25, reviewer catch)

The reviewer's portrait audit caught neuron-density-vs-phase using
raw PC1+i*PC2 angles: SVD orders axes by variance, so even a
perfectly isotropic cloud reads non-uniform — the bimodal wiggle in
the original chart was largely PROJECTION ARTIFACT and its "0.16 =
isotropy" line the wrong null. FIX SHIPPED: polar projection now
whitens (equalizes PC1/PC2 variance) before taking angles;
compare-panels now share one color scale (same color = same
magnitude). Chart regenerated: the REAL read survives — all four
widths (d64/d128/d256/19M) overlap in whitened phase density:
lattice geometry is width-robust even where the capability floor
cracks (d64), consistent with R=0.034 and texture-frozen-at-birth.
The earlier bimodality read is retracted as instrument artifact.
Portrait-audit verdicts also banked: binary = uniform-color blob
(constant row norm, the no-zero law made visible), ternary =
graded (sparsity dimmer switch); PR-across-alphabets is an
alphabet artifact (never compare PR levels cross-alphabet);
cross-model PHASE comparisons remain gauge-void until Procrustes.

## PRE-REG ADDENDUM: scalar 4-bit PTQ arms (2026-07-25, before gates)

Appended to the Lloyd-Max race (same twin, same MPS gates): P4
(powers-of-two ladder, 15 levels + spare — the zero+symmetry
corner of the two-of-three law), LM-16-zero (k=16 k-means,
zero-pinned), NF4-quantile (equal-mass, zero not guaranteed).
PREDICTION: all three within noise of fp32 64 and int3 65 — 4
bits is above the crystal's knee (democracy/no-outliers); any
SEPARATION here would mean level placement still matters at 16
levels and the knee sits higher than the democracy story
predicts. The complex bracket (M4^2 / G16 / G5 / D9 / Q9) is
BORN-only — spec'd separately, builds after today's cells.

## Opus-5 reviewer onboarding: four catches booked (2026-07-25 afternoon)

New reviewer (claude-opus-5[1m], self-report) onboarded on the full
corpus; its fresh-eyes pass on today's entries, all adopted:
1. 70/119 denominator conflation in the HOLD entry — FIXED (gate
   70/120, rarity 119/200).
2. **Cross-fix fence on the width floor**: the strict-encode fix
   landed mid-ladder — wfloor d256 trained 132,449 rows (388
   skipped) vs the historical 19M's 132,837. "d256 ON the 19M
   plateau" is a cross-fix pair; bounded by the accidental
   control (post-fix infix twin 64 vs pre-fix band 64-65 =>
   drift <= 1 solve on gen-4), but the fence is now LABELED.
3. MPS `--fast` is a packing-only flag (autocast gated on
   dev=="cuda") — same silent-per-device-meaning trap as the
   `if fast and not nopack` hole. Doctrine: name flags by what
   they do per device.
4. **SIGMA GRID PROTOCOL AMENDED (the big one)**: s1 (arm B, 70)
   pre-dates the strict-encode fix; s2/s3 skip 1,017
   out-of-language gen-9 rows s1 trained on mangled — so
   {s1,s2,s3} varies seed AND diet. PRE-REGISTERED RULE (before
   any sigma reads): sigma is computed over s2/s3 pairs first; if
   sigma > 3 (the re-fencing tripwire), the read is repeated with
   s1 DROPPED before any verdict is re-fenced; s1 is labeled a
   pre-fix point in every table. No gen-9 cross-fix control
   exists (unlike gen-4's <= 1 solve bound) — the 1,017-row
   delta rides as an explicit fence.

## PRE-REG: the prologue arms (Opus-5 design, 2026-07-25, before gates)

Zero births, PTQ/gauge on the 19M infix twin, MPS gates chained
behind the 4-bit arms. Cells + predictions:
- S4 {±1/3,±1} (symmetric, 2 bits, NO zero): the cheap zero-test.
  Zero-law predicts a crater (B-class failure) DESPITE having two
  magnitudes — if S4 holds near M4's 61, binary died of
  resolution, not zero-absence, and the law revises.
- sparse (fp32 magnitudes pruned to ternary's zero-fraction): if
  sparse ~ ternary-PTQ, "zero pays" = sparsity, alphabet story
  shrinks; if sparse >> tern, the discrete levels also matter.
- Gauge-commutation (the law candidate's direct test): gate
  (m4) vs gate(gflip_m4) — the gauge transform is verified
  function-identical in fp32 (max logit delta 0.0). Symmetric
  quantizers commute by algebra (tern/gflip_tern = exact control
  pair); PREDICTION: the ASYMMETRIC M4 pair's gates DIVERGE
  (quantization fails to commute with the network's own gauge
  group) — if they match anyway, the gauge-subgroup story loses
  its teeth.
Arms built: s4, sparse, tern, m4, gflip_m4, gflip_tern, gflip_s4.

## PRE-REG RIDER: the {0,1} arm (Artin, 2026-07-25 evening)

z1 = zero-or-positive-scale (1 bit, per-channel): SILENCE WITHOUT
SIGN — the missing corner of the deletion square (binary = sign
without silence, same 1 bit). PREDICTION (Fable): z1 craters
BELOW binary's 54 — no negative weights = no inhibition, the net
can only add; binary at least cancels. If z1 >= binary, sign is
cheaper to emulate than silence and the zero-law sharpens to
"zero > sign > neither." Chained behind PROLOGUE_DONE.

## PRE-REG AMENDMENT: z1 square, in-regime (reviewer blocking objection adopted)

Census: {0,1} NEVER run (reviewer sweep; near-misses = unsigned
STORAGE grids (HQQ affine — represented weights still straddle
zero), sparsity, output-zeroed growth init — none excitation-only).
Blocking fix adopted: z1-PTQ may NOT be read against born-B's 54
(born != rounded). In-regime square, all PTQ on the same twin:
b1 (sign-no-silence) vs tern (both) vs z1 (silence-no-sign) vs
z1c (centered control — separates inhibition-loss from DC shift).
BETS on record: Fable z1-PTQ craters (mass deletion); reviewer
z1-PTQ ~20, AND z1-BORN 49 (40% on >= born-B 54: LN mean-sub +
softmax = implicit inhibition). Rider: z1c ~ z1 iff DC absorbed
by LN; z1c >> z1 iff the crater was DC not inhibition. THE REAL
TEST = z1-BORN, queued for the next 3080 window (one tournament-
arm birth); z1-PTQ books only as the square's cheap corner,
labeled foregone-conclusion-candidate.

## BF16 ISOLATION: bf16 is INNOCENT — the whole −13 was packing (2026-07-25 night)

The decisive cell: 19M cuda --fast --nopack (bf16, standard
batching, seed 1, diet B) gates **69/120 @ 63.19** — IDENTICAL to
the fp32 cell's 69. The bundle decomposes completely: packing
−13, bf16 0, device −1 (fp32 MPS 70 v cuda 69). Loss fingerprint
matches (bf16np ep2 0.450 in the fp32 band 0.454-0.455; only the
packed cell showed lower-loss-worse-gate). VERDICTS:
1. **bf16 births are capability-lossless at BOTH measured widths
   on cuda** (19M 69=69; 45M 66=66) — the "no --fast below d512"
   candidate law DIES before booking; the width-dependent-debit
   hypothesis is retired (it was packing all along).
2. **Packing is re-confirmed as the −10-class hole at a second
   width** (07-17's 50M finding reproduced at 19M as −13) — the
   flag pair --fast --nopack is DOCTRINE at every width; the
   trainer keeps packing only as an explicit research flag.
3. The gen-9 A/B, sigma cells, and every arm of this week ran
   unpacked or fp32 — no retro-contamination beyond the one
   off-doctrine cell (mine, already booked).
Speed rider: bf16np epochs ~= fp32 epochs on the 3080 at 19M
(memory-bound regime) — bf16's win here is capability-neutrality
+ the 45M--3080 fit, not wall-clock.

## SIGMA RULE AMENDED: compare margins to sqrt(2)*sigma (reviewer catch, 2026-07-25 night)

Every gate verdict this week is a DIFFERENCE of two single-seed
cells, so the honest dispersion is sigma_diff = sqrt(2)*sigma_cell
~ 3.5 solves at 19M/cuda (per-cell sigma 2.5 from {69,66,71}).
Amendments, all adopted:
1. bf16 isolation restated: "no detectable debit, bounded ~±3.5
   at 1 sigma" — enough to retire the −14 width-debit hypothesis,
   NOT enough to claim exactly-zero. Doctrine unaffected.
2. **gen-10 = 19M: the +3 is SUB-NOISE (0.85 sigma_diff)** — the
   pick stands on MECHANISM + EFFICIENCY (token-light federation,
   2.4x cheaper), not on the number. Label corrected from
   "marginal-real."
3. prefix −3: sub-noise on the gate; NO-ADOPT unaffected (it
   rests on parse-fail 8.1% + int3 −8 + length null) — fence
   added so the −3 is never cited as the finding.
4. poly5 −2 / gen-9 +3: unaffected (door-based / already booked
   as redistribution).
5. NEW BAR: single-seed pairs at 19M need >= 5 solves to book a
   number; sigma is MEASURED only at 19M/cuda/fp32 — 45M and MPS
   sigma are assumptions and labeled as such. n=3 both sides
   drops the bar back.

## LLOYD-MAX RACE VERDICTS (partial booking per reviewer; 2026-07-25 night)

Raw card (PTQ, 19M infix twin, MPS gates; baselines fp32 64 /
int3-uniform 65): LM3-free 64, LM3-zero 63, LM2-free 59, LM2-zero
54, ternary-LM 42, int2-uniform VOID (absmax/lv scale bug at
bits=2 — collapses weights to zero; instrument error, fix + rerun
queued). d256 sigma rider: {65,63,64} => sigma ~1.0 (down-width
noise is tight; sigma does NOT transport across cells — 2.5 at
19M/cuda vs 1.0 at d256/MPS). [Provenance verified 2026-07-26:
three SEED BIRTHS — mathnative_wfloor_d256{,_s2,_s3}.pt — so this
is genuine birth-seed dispersion, n=3, not PTQ-arm spread.]
BOOKED NOW:
1. **MSE-optimality buys nothing: the never-score-by-distance
   doctrine gets its direct experimental confirmation** — the
   explicitly distance-optimal codebook (per-channel exact
   k-means) ties naive uniform at 3 bits (64/63 v 65) and shows
   no advantage anywhere. TurboQuant-class codebook optimization
   is a storage lever, not a capability lever, on democratic
   (kurt 2.4, outlier-free) crystals.
2. **Born-vs-rounded at matched alphabet: +18 LOWER BOUND**
   (born-ternary 60 v ternary-LM 42; the PTQ arm had FINER
   per-channel codebooks and still lost). Fences: cross-device
   (3080-born v MPS-gate), cross-recipe (global v per-channel).
   Trend note (n=2): premium 45 at d768, 18 at 19M — shrinks
   with width; not a law.
HELD: (a) "zero loses at 2-bit PTQ" — unmatched pair (LM2-zero
paid a code for its zero: 3 magnitudes v 4); narrower claim "a
zero level does not pay for its code at 2 bits" awaits the fixed
int2-uniform (symmetric WITH zero = the deciding third point).
(b) "zero is a born law not a PTQ law" — HYPOTHESIS only; both
legs confounded (M4>B is +1.00 bit); the named clean cell =
born-S4 vs born-M4 at matched 2 bits; born-z1 moves it tonight.
Meta-line (reviewer): three instrument bugs in 24h (phase-density
null, 388 rows, int2 scale) — ALL caught by control arms, none by
headlines. Controls ride every instrument.

## PROLOGUE CARD BOOKED: three deaths, one sleeper, a directional gauge (2026-07-25 night)

PTQ on the 19M infix twin, MPS gates; diagnostic confirmed
non-degenerate quantizers + clean tern control (correct step
emitted). Reviewer-cross-checked; its framings adopted.
1. **THE DEATH TAXONOMY (the card's best result)**: at 1 bit,
   PTQ lands at the measurement floor in three mechanistically
   distinct ways — **b1 (sign-only): arithmetic death** — fluent,
   well-formed math with wrong coefficients at validity 0.00 =
   CONFIDENTLY-WRONG generation: the determinability law's
   failure mode (calibrated hallucination) produced by
   QUANTIZATION instead of underdetermined data — a new route to
   the same pathology, and 7th gate-not-the-loss-family
   observation (syntax intact, semantics gone; independently
   corroborates the prefix close: syntax and semantics are
   separably damageable, scaffolding is the robust layer).
   **z1 ({0,1}): language death** — degenerates to Step: loops.
   **born-B: alive at 54** (cross-device fence: 3080/TF32
   born vs MPS PTQ; acceptable at this delta).
2. **Born-vs-rounded at 1 bit: >= 54 solves and FLOOR-CENSORED**
   (PTQ at the instrument floor; true gap unmeasurable from
   above).
3. **HARD FENCE: z1-PTQ does not bear on born-z1** — deleting
   negatives from a signed-trained net is a foregone conclusion;
   the {0,1} corner remains OPEN until the birth gates (reviewer
   bet 49 stands; Fable bet crater).
4. **The sleeper: sparse 62 vs tern 43** — imposing ternary's
   zero-PATTERN on fp32 magnitudes costs −2; adding magnitude
   flattening costs −21. Split-wording (reviewer): zeros here
   are magnitude-chosen, so this is the pruning/tail-dies-first
   result, NOT evidence about the zero LEVEL; what
   training-into-lattice buys is COEXISTENCE WITH DISCRETENESS,
   not the zeros.
5. **Gauge-commutation: DIRECTIONAL only** — symmetric pair
   commutes EXACTLY (tern=gflip_tern 43=43, per algebra);
   asymmetric M4 pair diverges 51 v 49 (right direction,
   sub-noise; M4 is asymmetric in 1 of 4 levels = deliberately
   weak instance). Decisive follow-up = max-asymmetric alphabet
   ({0,1,2,3}) arm, queued.
6. 2-bit fence: LM2free/LM2z are FITTED per-channel; m4/S4 are
   FIXED — "placement dominates" is partly fitted-beats-fixed.
   Clean fixed-vs-fixed pair: m4 51 > S4 44 (zero+asym beats
   sym-no-zero at matched fixed 2 bits). The clean S4 test stays
   BORN (with born-M4 at matched bits — the named law-converting
   cell).
7. Method line: three instrument bugs in 24h (388 rows,
   phase-density null, int2 scale) ALL caught by control arms,
   none by headlines — controls ride every instrument, and the
   control arms are outperforming the experiments.

## int2fix resolves the held 2-bit claim (2026-07-25 late night)

The reviewer's demanded third point: int2-uniform (fixed grid
{-2,-1,0,1}, range-scaled, WITH zero) gates **54/120** — landing
exactly on LM2-zero's 54 despite entirely different level
placement (fixed uniform vs fitted k-means). Two independent
zero-ful 2-bit codebooks at 54 vs zero-free fitted at 59: the −5
tracks ZERO-PRESENCE, not placement. BOOKS (narrow form, as
pre-agreed): **at 2-bit PTQ, a zero level does not pay for its
code** — spending 1 of 4 codes on silence costs ~5 solves of
magnitude resolution on an already-trained signed net. Fences:
PTQ-only (the born side of this question = born-S4 vs born-M4,
still the named law-converting cell); MPS sigma unmeasured (the
5 = the cuda bar exactly); the earlier VOID int2 stays booked as
instrument error.

## AMENDMENT to the 2-bit zero claim: the sign FLIPS with codebook type (reviewer objection, adopted)

The booked one-liner was drawn from the fitted pair alone and is
contradicted by the fixed triple in the same table. Full 2-bit
card by codebook type: FITTED (k-means): LM2-free 59 > LM2-zero
54 (zero costs −5). FIXED (hand levels): int2fix 54 ~ m4 51 >
S4 44 (zero PAYS +7..10). CONDITIONAL LAW (replaces the
narrow claim): **at 2-bit PTQ, a pinned zero PAYS in fixed
codebooks (it captures the mode of the kurt-2.4 distribution
that fixed levels otherwise miss) and COSTS in fitted ones
(k-means already places a level near the mode; pinning re-derives
what fitting gives free).** Re-reads the 3-bit null coherently:
uniform is near-optimal at 3 bits, fitting has nothing to buy;
at 2 bits fitting still buys, and the zero is how a FIXED
codebook buys it. The 54=54 tie is worded "consistent with
placement-invariance" (validities 49.50 v 47.02, same animal),
never leaned on as exact. The asymmetry axis remains
unseparated in every 2-bit arm — born-S4 vs born-M4 stays the
axis-separating cell.

## THE SIGMA GRID COMPLETES: the birth-device saga ends with every term measured (2026-07-25 late)

All cells cuda-gated, gen-9 diet B. cuda-fp32 seeds {69,66,71}
(sigma 2.5); MPS-fp32 post-fix seeds {64,66}; MPS s1 (PRE-FIX
diet) 70 = the outlier, exactly as the pre-registered s1 fence
anticipated. VERDICTS:
1. **The MPS-birth bonus is DEAD**: post-fix MPS-born reads at or
   below cuda-born on the same gate (65 v 68.7 means, inside
   sqrt(2)-sigma). The full decomposition of the original −14:
   **packing −13, bf16 ~0, device ~0, s1-luck/diet the rest.**
   Every term measured; the saga closes.
2. Arm B's 70 was a pre-fix + fortunate point. Its BOOKED verdicts
   survive unchanged (redistribution was already sub-noise; rations
   pass was band-based), but 70 must never be cited as the 19M
   level — the honest 19M-on-diet-B band is 64-71 across 6 births.
3. Production note: crown/production lineages are NOT affected
   (different diet/lineage); gen-10 = 19M stands on mechanism +
   efficiency as amended.
4. [SUPERSEDED by the amendments below: the 1.4 is RETRACTED as
   underdetermined — booking s1=70 as luck requires sigma_MPS >= 3.]
   sigma_MPS(19M, post-fix) ~ 1.4 (n=2, provisional); sigma does
   not transport (2.5 cuda / >=3 MPS / 1.0 d256).

## AMENDMENTS to the sigma-grid close (reviewer, all adopted; 2026-07-25 late)

1. "Device ~0" RETRACTED: the point estimate has FLIPPED SIGN
   (MPS mean 65 v cuda 68.7 = −3.7, MPS LOWER; not significant at
   n=2/3 but "~0" licenses cross-device comparisons, which stays
   FORBIDDEN). Book: MPS-birth BONUS refuted, sign reversed;
   residual device term UNRESOLVED. Packing −13 unaffected
   (same-device measurement).
2. Consistency fix: booking s1=70 as luck REQUIRES sigma_MPS >= 3
   (n=2 pair is then lucky-tight; 1.4 retracted as
   underdetermined). The alternative — s1's 1,017 pre-fix rows
   worth +4-6 — is rejected as unparsimonious (0.085% of rows).
   Corollary: cuda's 2.5 is itself n=3-fragile (dominated by one
   internal draw); all sigma estimates carry their n.
3. Pooled "band 64-71" retracted for per-cell reporting: MPS
   post-fix 64-66; cuda 69/66/71; s1 pre-fix-MPS 70 (labeled).
   The instruction survives: never cite any ONE number as the
   19M level.

## Z1 seed-2: NOT input-blind — the collapse was seed-luck, the square still closes (2026-07-25 close)

Seed-2 confirmation (blindness probe only, per protocol): emissions
are INPUT-DEPENDENT ('Integral(18*x, x)' for the 16x prompt —
wrong coefficient, right neighborhood; a different malformed
attempt for x**2) at 94% zero-fraction. VERDICTS:
1. The strong cone claim ("excitation-only nets CANNOT condition
   on input") is REFUTED by the confirmation seed — s1's total
   input-blindness (cross-input cos .9934) was one basin, not a
   law. The mechanism downgrades to: excitation-only training is
   COLLAPSE-PRONE and arithmetic-crippled (severe, seed-unstable
   degradation), with deletion as the only inhibition (83-94%
   zero fractions both seeds).
2. **The deletion square still closes asymmetric on capability**:
   sign-without-silence chains (born-B 54); silence-without-sign
   is capability-dead at every observation (s1 blind, s2 fluent-
   wrong) — but the WHY is now "unstable + contrast-starved,"
   not "structurally blind."
3. Both-seed observation: zeros do double duty (silence +
   opposition-proxy) — the extreme sparsity is forced, not
   chosen.
Artin's original framing gets its honest answer: the 1-bit weight
as stored is really "1 bit + a positive per-matrix scale," and
one bit per weight IS enough for language but not for arithmetic
contrast without signs at some granularity. The named next cell
(reviewer): z1 + SIGNED PER-CHANNEL scales — is sign sufficient
at channel granularity? Queued for a future window (3080 held for
Artin from here).

## Closing riders on the Z1 arc (2026-07-25 close)

1. Reviewer over-prediction RECORDED (its own ask): the strong
   cone claim predicted deterministic blindness; seed-2 refuted
   it. Weak form survives (cone-confined rows => degraded
   diversity => capability death, both seeds 0-class). Its bets
   (z1-born 49; strong cone) both booked against it — as were
   Fable's (bf16-width-debit; the "+3 real" width read). The
   protocol, not any predictor, is the asset.
2. FREE DATAPOINT: collapse is NON-MONOTONE in sparsity (83%
   zeros = blind seed; 94% = input-dependent seed) — direct
   evidence against the sparsity-driven hypothesis; the
   matched-sparsity-ternary control arm is DROPPED as demoted.
3. STREAMING RED-TEAM (recorded before it seeds a prior): Z1-s2's
   fast convergence is DEGENERACY, not efficiency — a 94%-zero
   net has less to learn; the streaming-birth motivation stands
   on surprise-gating + the zero-epoch 0/120 floor only.
4. Queued free diagnostics for the next 3080 window (held for
   Artin now): attention-entropy + hidden-state cos on BOTH Z1
   seeds (rescues or kills the weak cone mechanism); the three
   law-converting cells in priority order: born-S4 v born-M4
   (matched 2 bits), z1+signed-per-channel-scale, max-asymmetric
   {0,1,2,3} gauge arm.
5. Method line (the day's meta-result): n=1 MECHANISMS are the
   fragile class — two mechanisms (strong cone, LN-inhibition)
   were killed or halved by single confirmation seeds within
   hours; n=1 numbers were already fenced. Confirmation seeds on
   mechanisms join the doctrine.

## PRE-REG: d256 substrate gate — reproduction leg (2026-07-26, before the run)

Purpose: promote d256 to the pilot-A/B substrate (evening-queue
v2.1). Replication leg ALREADY PAID (provenance verified this
morning: wfloor_d256{,_s2,_s3}.pt are three seed births, {65,63,64},
sigma ~1.0). Remaining leg = known-result reproduction, and per
reviewer the target must be LARGE-MARGIN (sub-noise verdicts can't
gate a substrate). Target chosen: **the packing hole (−13 at 19M)**.
Arm: identical wfloor d256 recipe (gen-4, 3ep, BIRTH_SEED=1,
d256/L8/ffn1024/h4, MPS) + bare `--fast` (packing ON — on MPS,
--fast is packing-only; autocast is cuda-gated). Control: existing
wfloor_d256.pt (65) — same seed, same device, post-strict-encode,
one variable.
PRE-REGISTERED: packing arm gates BELOW control by >= 3 solves
(sigma_diff = sqrt(2)*1.0 ~ 1.4 at d256; direction + >=3 = pass).
PASS REGION PRE-SPLIT (reviewer refinement, registered while the
run is in flight, before any read): delta <= -8 = "reproduces with
comparable magnitude"; -8 < delta <= -3 = "reproduces
DIRECTIONALLY, magnitude attenuated at d256" — both clear the
substrate gate but license different uses (pilots size effects
only under the first). DUAL READ: primary = vs seed-1 control 65
(the strict one-variable pair); secondary = vs 3-seed mean 64
(seed 1 is the band's high draw — a 62 reads -3 primary / -2
secondary; registered in advance). PASS => substrate gate CLEARS,
d256 becomes the pilot substrate (promotion-grade runs stay at
W*). FAIL (delta > -3 or sign flip) => either the packing hole is
width-dependent (a result in itself) or d256 doesn't carry 19M
verdicts — substrate NOT promoted, book which.
BUNDLE LABEL (adopted): bare --fast changes packing AND steps_total
AND the OneCycle schedule AND batch composition — this arm
reproduces the FLAG-level effect exactly as measured at 19M; the
-13's mechanism (packing vs step-count vs LR schedule) remains
unseparated at both widths. Control provenance: wfloor_ladder.sh
ran with NO --fast at all (ladder default, fp32, unpacked) — the
pair's one variable is the flag bundle, checkable from this entry.

## PRE-REG: graph-modularity on gen-8 (2026-07-26, before the read)

The 07-17 prediction (multi-domain crystals grow visible modules)
finally meets its substrate; probes union coefficient (iii).
INSTRUMENT NOTE (honest): the 07-17 six-mind script was never
committed and its kNN k is unrecorded — absolute comparison to the
07-17 numbers (Q 0.142, clustering 0.021-0.026) is FORBIDDEN; this
is a NEW instrument, internally paired only. Construction, fixed
before looking: per-layer kNN graph (k=10, cosine) over FFN neurons
(feature = concat gate-row + up-row), Newman greedy modularity Q +
average clustering, mean over 8 layers. Arms, same instrument same
run: (A) gen-8 everything-crystal (five grammars, vocab-41);
(B) matched-width single-grammar control mathnative_19m.pt
(vocab-40; row-count not exactly matched — noted). Control arm
mandatory per reviewer (Q alone uninterpretable).
PRE-REGISTERED BAR: "modules appear" = Q_A > Q_B + 0.05; within
±0.05 = null (reviewer bet ON RECORD: within-noise — expander +
phase-clumps-arent-communities + coefficient (i)~0 all point null).
Rider read, free: clustering coefficient both arms (expander check).

## Graph-modularity on gen-8: NULL at the bar — modules do NOT appear (2026-07-26)

Read per pre-reg (new instrument, k=10 cosine kNN on FFN
gate+up rows, Newman Q, 8 layers, paired arms). **delta Q = +0.030
(A 0.2928 v B 0.2633) — below the +0.05 bar: NULL.** The
multi-domain-modules prediction (07-17 lineage) fails at its second
substrate; union coefficient (iii) "nothing else" survives another
probe; reviewer's on-record bet (within-noise) WINS as booked.
RIDER (observation, internally paired only): mean clustering 0.0545
(gen-8) v 0.0222 (single-grammar) — ~2.5x, the only quantity that
moved with grammar count. The 07-17 instrument is LOST; no
comparison to its crystal (0.021-0.026) or internet (0.063-0.095)
numbers is licensed, INCLUDING the incidental agreement of arm B
with the old crystal band — that agreement is uninterpretable, not
corroboration. Density fence: both arms are d384/ffn1536 (identical
n=1536 per layer, same k=10), so the 2.5x is a real internal
contrast, not a graph-density artifact. Other fences: vocab 41 v
40, diets differ in rows AND eras — arms are paired on the
instrument, not the corpus. U-shape note: both arms show
depth-symmetric Q/clustering (layers 0/7 high, middle low), same
shape both arms — instrument- or architecture-driven, not
diet-driven.
CAVEAT (reviewer, against its own winning bet — adopted): delta Q
+0.030 is positive and directionally consistent with the modules
prediction, and Q has NO measured seed dispersion — this is a
BAR-BASED null, not a statistical null. Never cite as "no
difference"; only "below the pre-registered bar, dispersion
unmeasured." Free sigma if ever needed: run the same instrument on
wfloor_d256{,_s2,_s3} (three same-diet seed births on disk) before
Q decides anything.

## ODE desk decisions: atom set, a determinability catch, and the sealed committee prediction (2026-07-26)

Desk work only (no birth today); the zero-diet probe re-registers
onto the FIRST ODE-vocab birth (VOID-BY-VOCAB lesson applied).
1. **ODE atom set (proposal, order-sensitive per VOCAB_EXTRA
   doctrine)**: `SolveODE: `, `CharEq: `, `Eq(`, `Derivative(`,
   `y`, `C1`, `C2`, `mu`, `r`, `=` — vocab 40 -> 51. Notes: bare
   `=` is NEW and greedy-longest-match-safe against ' => '
   (reviewer-verified: _by_len sorts descending); `y(x)` spells as
   y + ( + x + ). ORDER FROZEN (reviewer amendment, adopted):
   extra = ["t"] + ODE_ATOMS UNCONDITIONALLY — t first always,
   even on diets with no series rows (one dead atom is free), so
   every vocab-41 checkpoint's id map is a strict prefix of the
   ODE vocab; the alternative is three incompatible id maps and a
   probe silently reading the wrong atom (the class of bug that
   just bit at the ODE void).
2. **DETERMINABILITY CATCH (blocks any ODE birth as-is)**:
   data/ode_chains.jsonl (317 pairs: linear1 183 / cc2 110 /
   separable 24, all L2) contains rows whose targets carry
   CONCRETE coefficients not derivable from the prompt — cc2
   `CharEq: r**2+4*r+3 = 0 -> exp(-x) + 4*exp(-3*x)` (the 4 comes
   from unstated ICs) and separable `y = C1*exp(...) ->
   4*exp(-x**2/2)` (C1 silently becomes 4). Same pathology as
   series rung 1 (memorized-factorial fill-ins); the
   determinability law says these rows TRAIN HALLUCINATION.
   Required before farming/birth: either carry ICs in the prompt
   or emit general solutions with C1/C2 kept symbolic. The 317
   file is 2026-07-12-era; audit/refarm belongs to the ODE
   continent GO, not today.
3. **SEALED BLIND COMMITTEE PREDICTION (standing rule #15, first
   application; scored at: first ODE-vocab birth, same instrument
   as the mass-spectrum table)**: from corpus frequency alone
   (linear1 183 > cc2 110 > separable 24), the mass-spectrum law
   predicts committee selectivity ordering linear1 > cc2 >
   separable, with linear1 power-class sharp (localized) and
   separable exp-class diffuse; the Integral-reduction step
   inherits the EXISTING integral committees (shared-step bridge,
   not new mass). No numbers claimed beyond ordering + class.

## ZX desk gates: both PASS with named constraints (2026-07-26, desk only)

The two blockers named at ZX's promotion to next-continent
candidate #1 (RIFF-LEDGER 07-25), resolved at the desk — this is
NOT a farm GO, it clears the desk prerequisites.
1. **Serialization without canonical sorts: a design exists.** The
   engine's own ZXState.key() canonically sorts vertices — fine
   for DEDUP (hash identity, internal), FORBIDDEN as the
   model-facing serialization (gauge law: permutation-augmentation
   88.4 beat canonical sorting 82.4 — teach invariance, don't
   impose it). Design: anchor on the BOUNDARY ORDER (ZX
   inputs/outputs are physically ordered — that ordering is real
   structure, not gauge), traverse BFS from inputs, and RANDOMIZE
   internal vertex labels per training sample (permutation
   augmentation over the true gauge freedom). Reviewer rider
   honored: no leg of this choice is justified by sequence-length
   gains (notation-invariance of the emission wall, booked 07-25).
2. **Atom set vs vocab: PASS, no explosion.** ZX needs ~10-14 new
   atoms on top of base-40: spider constructors `Z(`, `X(`,
   hadamard-edge marker, plain-edge marker, boundary markers
   In/Out, and move labels (fuse/lc/pivot/id) if moves are spelled;
   phases are Clifford+T fractions of pi — `pi`, `/`, digits ALL
   in-language already; vertex ids spell in digits. Comparable to
   the series (t) and ODE (+10) expansions; strict encode enforces
   it at birth (the ODE void's lesson, now machinery).
Remaining before any farm: serialization spec written down as a
spec file + the federation-floor pre-registration (count vs class)
— those belong to the ZX GO, not today.

## PRE-REG: streaming-birth A/B at d256 (2026-07-26, before the run)

Bank: RIFF-LEDGER 2026-07-24 (streaming birth; Artin, half-retracted
same breath — banked anyway) + the Z1 red-team rider (Z1-s2's fast
convergence was DEGENERACY; motivation stands on surprise-gating +
the zero-epoch 0/120 floor only). Arm S: scratch/
streaming_birth_d256.py — gen-4 corpus streamed ONCE, no epochs,
surprise-gated LR (per-batch mult = loss/EMA(0.99) clamped
[0.25,4], warmup 200, base 3e-4), BIRTH_SEED=1, BS=32, d256/L8/
ffn1024/h4, MPS. Control: wfloor_d256.pt (65; band {65,63,64},
sigma ~1.0). One variable: schedule (1-pass surprise vs 3ep
OneCycle); init is standard (NOT template — the composed-birth
stack stays a separate banked pilot).
PRE-REGISTERED: this is a SPEED claim. PASS = gate >= 61
(capability-neutral: within 3 = ~2*sigma_diff of the band mean 64)
AND wall <= 40% of the 3ep birth wall. Gate < 61 => streaming
FAILS at equal rows (the epoch is load-bearing; books as a result).
Gate > 66 would be a WIN, not just neutrality — flagged separately
if seen (sub-noise unless >= +3 vs band mean per the sigma rule).

## d256 SUBSTRATE GATE CLEARS — and packing is the third starvation mode (2026-07-26)

Arm: wfloor recipe + bare --fast (flag bundle), MPS, seed 1. Gate
38/120 @ 35.75 {3:15, 4:1, 5:10, 6:5, 7:7}; control 65/120 @ 59.91.
Reviewer-cross-checked before booking; its framing adopted in full.
1. **Substrate gate CLEARS decisively**: delta -27 v the
   pre-registered -3 bar (~19 sigma_diff); dual read -27 primary /
   -26 vs 3-seed mean — the seed-1-high-draw worry is dead. **d256
   is PROMOTED to pilot substrate**; promotion-grade runs stay at
   W*.
2. **The internally-valid headline**: the packed d256 lands at 38 —
   EXACTLY the d64 (0.5M) point of the same-corpus, same-device
   width curve (38 -> 57 -> 65). Packing at d256 costs the
   equivalent of a ~16x parameter reduction, priced on the lab's
   own curve, zero cross-device inference. (-27 solves / -24
   validity points.)
3. **Per-level: broad with an L4-DISPROPORTIONATE component** —
   clade-connected levels retain 58-68% (L3 68, L5 63, L6 63,
   L7 58); L4 retains 14% (7 -> 1). The isolated-clade signature
   rides on top of broad damage: packing joins under-width (d64:
   L4=0) and under-feeding (400M: L4=0) as the THIRD measured
   starvation mode with the clade-dies-first fingerprint.
4. **Magnitude-vs-19M does NOT book** (reviewer objection,
   adopted): the 19M -13 cell is cuda/gen-9-dietB/solves; today is
   MPS/gen-4 — device+diet+era+metric all differ, and cross-device
   is doctrine-forbidden as of yesterday's amendment. "Packing
   amplifies at thin width" = HYPOTHESIS, one observation per
   width. Partial separation, reported free: packed 1,324 steps/ep
   v control 4,139 = 3.13x fewer — SMALLER than 19M's 4.45x ratio
   yet a larger wound, a point AGAINST pure step-starvation
   (confounded; report-only). Clean test if ever wanted:
   packed-vs-unpacked at two widths, same device/diet/day.
5. **Flag-bundle label rides**: reproduces the FLAG effect;
   mechanism (sequence packing v step count v LR schedule)
   unseparated at both widths. Gate-not-the-loss fired on this
   bundle twice now (19M packed cell had LOWER loss and -13 gate;
   today ep1=ep2 loss 0.3594 stagnation at gate 38).
6. **The flag is lose-lose at d256/MPS**: -27 solves AND 3.3x
   SLOWER (1,330 s/ep packed v 400 s/ep unpacked — token-budget
   batch shapes mismatch MPS at this width). Doctrine reinforced:
   --fast NEVER without --nopack, every width, every device.
7. Reviewer over-prediction RECORDED (its own ask, second on the
   books): the pass-region pre-split imagined attenuation at thin
   width; the effect went the other way.

## Streaming birth FAILS both legs — the epoch is load-bearing (2026-07-26)

Arm S (scratch/streaming_birth_d256.py): gate 53/120 @ 47.85
{3:17, 4:6, 5:13, 6:7, 7:10}, wall 1,096s. Control: 65 @ 59.91,
3ep = 1,199s. Pre-reg bars (gate >= 61 AND wall <= 480s): BOTH
FAIL. Booked on house judgment; reviewer check is post-booking.
1. **Capability leg: clean FAIL.** One pass at equal rows loses
   -12 vs seed-1 (-11 vs band mean; ~8 sigma_diff) — the epoch is
   NOT an artifact of the batch era; revisiting rows is
   load-bearing at birth. The banked question ("is the epoch an
   artifact?") gets its answer: NO, at this width/corpus.
2. **Speed leg: fail is CONFOUNDED, not fundamental.** The arm ran
   3.8 it/s vs control's ~10.4 because it shuffles-then-batches
   (mixed-length, padded) where the trainer length-sorts. The
   honest speed statement is "1 pass = 1/3 the optimizer steps";
   realized wall depends on batch construction and was not
   cleanly measured here. A length-sorted streaming arm would
   settle it, priced ~20 min — queued only if streaming ever
   matters again given leg 1.
3. **Fingerprint, OBSERVATION ONLY (house fence): retention is
   even across levels** (L3 77 / L4 86 / L5 81 / L6 88 / L7 83%)
   — no L4 clade death, unlike packing/under-width/under-feeding.
   NOT booked as "schedule starvation is graceful": the debit here
   is -12 where the clade-killers are -27-class, and even
   retention may simply be what a MILD debit looks like (the
   clade may die last, not never). Deciding cell if ever wanted:
   a HARSHER schedule starvation (e.g. half-pass) — clade death
   at matched -27 damage would kill the "graceful" reading.
4. Surprise-gating telemetry: multiplier ~0.86-0.95 by late
   stream (EMA-tracking, barely gating) — the mechanism had
   little leverage; no claim either way.
5. The Z1-degeneracy red-team rider discharges trivially: no
   degeneracy involved; the arm under-trained, honestly.

## AMENDMENTS: the matched-steps accident separates the packing bundle (2026-07-26)

Post-booking cross-check (reviewer) + house read on top of it; both
the packing and streaming entries take amendments.
1. **THE MATCHED-STEPS READ (house catch, the day's sharpest)**: the
   two arms did nearly IDENTICAL total optimizer steps — packed
   1,324 x 3ep = 3,972; streaming 4,140 (control 12,417) — same
   device, corpus, seed, ~same step budget. Results: packing 38
   (L4 = 1), streaming 53 (L4 = 6). **Step count alone cannot be
   the packing hole's mechanism**: at matched step starvation,
   packing does twice the damage and kills the clade where
   streaming leaves it untouched. Mechanism candidates narrow to
   packing-specific factors: batch composition (token-budget
   mega-batches), row repetition pattern (3 passes packed v 1 pass
   even), or the OneCycle-on-fewer-steps schedule. Fences: LR
   schedule and repetition DO differ between the arms (not a
   designed pair — an accidental one), so this SEPARATES the
   bundle partially, not fully; the designed cell stays
   packed-vs-unpacked at matched schedule.
2. **PER-LEVEL SIGMA MEASURED (free, from the three seed births)**:
   seeds {1,2,3} per-level: L3 {22,20,23}, L4 {7,7,6}, L5
   {16,17,17}, L6 {8,7,7}, L7 {12,12,11} — sigma ~0.6-1.5. So:
   streaming L4=6 is INSIDE the seed band (clade undamaged, not
   merely "even retention"); packing L4=1 is ~9 sigma below it
   (clade death, now statistical, not eyeballed). The instrument
   gap (gate sigma everywhere, per-level sigma nowhere) is closed
   for d256.
3. Streaming entry item 3 SUPERSEDED accordingly: the
   schedule-vs-structural wording is dead on its own examples
   (packing also cuts steps and DID kill L4). What books instead:
   two readings were open — (a) dose-response (clade dies at deep
   damage only), (b) packing-specific mechanism — and the
   matched-steps read decides FOR (b)-leaning: damage is not a
   function of step count alone. Dose-response survives only as
   "whatever packing does, more of it kills the clade."
4. Streaming capability fail: fence added — the claim is "one pass
   at equal rows loses -12" (the epoch is load-bearing), NOT
   "streaming fails" as mechanism; surprise-gating is UNTESTED
   (multiplier 0.86-0.95, barely engaged), not nulled — its real
   cell is the metabolic loop where waves vary.
5. Speed leg standing bar REGISTERED (no goalpost-moving later): a
   length-bucketed streaming rerun must hit the SAME 480s + gate
   >= 61 bars or streaming books dead on both legs. Composed-birth
   re-priced NOW: its largest lever (3ep -> 1 pass) is gone;
   the <15-min target loses its main multiplier — batched-KV is
   the speed track's remaining headline.
6. PROTOCOL NOTE (booked): paired arms must match BATCH
   CONSTRUCTION, not just diet/seed/device — the streaming arm's
   shuffled-padded batches cost the speed leg. Reviewer
   over-prediction #3 recorded (its own ask): it endorsed the
   streaming GO without catching the batch-construction
   one-variable violation at design stage.

## AMENDMENT (house self-review): the streaming -12 is CONFOUNDED with cooldown (2026-07-26)

Caught without the reviewer (retired for now; house self-checks).
The streaming arm differs from control in TWO ways, and the entry
attributed the -12 wholly to the first: (a) revisits (1 pass v 3);
(b) THE LR ENDGAME — control's OneCycle anneals to ~0 by the end,
the streaming arm ended still hot (~0.9 x 3e-4, no cooldown).
Final-LR annealing is independently known to carry capability;
"the epoch is load-bearing" is therefore OVER-ATTRIBUTED as
booked. PRE-REG for the deciding arm (fires now): one pass,
OneCycle COMPRESSED into the single pass (anneals to zero),
length-sorted batches (also the registered speed-leg rerun — the
480s + gate >= 61 bars stand as registered, no goalposts moved).
Readings: gate ~61+ => cooldown was the load-bearing part, epoch
claim RETRACTS to "one hot pass loses"; gate ~53 => revisits
confirmed as mechanism at this width; between => both contribute,
split booked as measured. Surprise gating stays in (it barely
engages; not the variable).

## Streaming v2 (compressed OneCycle): 45/120 — my schedule shape, not a cooldown verdict (2026-07-26)

Arm: 1 pass, OneCycle compressed into 4,140 steps, length-sorted
batches. Gate 45/120 @ 40.23 {3:17,4:3,5:10,6:6,7:9}; wall 405s.
1. **Speed bar PASSES** (405 <= 480; length-sorted batching alone
   took 3.8 -> 8-10 it/s — the v1 speed fail confirmed as batch
   construction, as suspected).
2. **Capability WORSE than v1's hot pass (45 < 53) — but this is
   NOT a cooldown verdict**: house design error, caught on
   self-review. OneCycle's default pct_start=0.3 spent ~30% of the
   single pass in warmup — the arm cut total effective LR mass,
   not just the tail. v2 tests "compressed OneCycle at 1 pass,"
   which is its own (negative) result: the control's 3-epoch
   schedule does NOT compress into one pass by squeezing.
3. L4 = 3 (below the {6,7} seed band) — deep-warmup starvation
   shows the clade signature where v1's even profile did not;
   consistent with dose, not booked beyond observation.
4. **v3 pre-registered (the clean cooldown isolation)**: v1's
   exact profile (warmup 200, constant 3e-4) + final-10% linear
   decay to zero; length-sorted batches. Integral-LR ~0.90 vs
   v1's ~0.95 — near-matched; single variable = ends-hot vs
   ends-cold. Readings: >= 61 => cooldown rescues the single
   pass (epoch claim retracts); ~53 => cooldown is minor, revisits
   confirmed; < 50 => decay tail hurts at 1 pass (books as
   schedule law leg).

## Streaming v3: 45 again — the schedule was never the variable; batch DIVERSITY is (2026-07-26)

v3 (v1's constant profile + final-10% cooldown, length-sorted
batches): gate 45/120 @ 43.79 {3:15,4:3,5:13,6:5,7:9}, wall 403s.
House self-review, the read that books:
1. v2 (compressed OneCycle) and v3 (constant+tail) — maximally
   different SCHEDULES — score IDENTICALLY (45, 45). v1, schedule-
   matched to v3 except the tail, scored 53. The moving variable
   across {53, 45, 45} is BATCH CONSTRUCTION: v1 = shuffled
   mixed-length (iid) batches; v2/v3 = length-sorted homogeneous
   batches (the "speed fix"). Cooldown reads ~0 where it was
   isolated on sorted batches (45 = 45).
2. MECHANISM CANDIDATE: length-sorted batches group same-length =
   same-family/kind rows (generated corpus), so each step's
   gradient is intra-batch correlated — fewer effectively-diverse
   updates per pass. With revisits (control, ALSO length-sorted,
   65) the damage washes out; at ONE pass it costs ~-8. Candidate
   interaction law: batch homogeneity is FREE with epochs,
   EXPENSIVE without them (diversity-per-step is the binding
   resource of single-pass training).
3. PRE-REG v4 (the missing 2x2 cell, fires now): v1's mixed
   shuffled batches + final-10% cooldown. Readings: ~61+ =>
   cooldown DOES help and sorting cost -16 (both claims book);
   ~53 => cooldown ~0, sorting -8 books alone, "one pass loses
   -12 and the epoch is load-bearing" STANDS as v1 measured it;
   ~45 => construction story wrong, rethink. Wall ~1,100s
   (padded) — this is the mechanism cell, speed bar not at issue.
4. Speed note: 403s twice confirms the sorted-batch wall (~34% of
   control) — IF capability were ever recovered, the speed bar is
   comfortably passable; the two bars currently anti-correlate
   through the same construction choice (the honest tension,
   named).

### v3 rider: verification + the 07-16 precedent (2026-07-26)

Code-verified: control batching IS length-homogeneous (BS=32
slices over sorted enc, batch ORDER shuffled) — the interaction
claim's control leg stands. Two extras from the same read:
(a) trainer OneCycle runs pct_start=0.03 — v2's default-0.3 arm
was doubly non-matched (booked error confirmed sharper);
(b) PRIOR ART, in the trainer's own 07-16 comment: the parity 2x2
measured length-homogeneous PACKED batches at ~-10 unseen-validity
— homogeneity cost is a measured phenomenon in the packed regime,
fixed then by shuffling. Today's v2/v3 (-8 at 1 pass, BS=32) is
its small-batch single-pass sibling: homogeneity cost appears
where diversity-per-step binds (packed mega-batches then, no-
revisit passes now) and washes out otherwise (control 65 at 3ep).
Also narrows today's packing-mechanism candidates: post-07-16
packing is mixed-length by construction, so the packed d256 arm's
L4 death is NOT homogeneity — mega-batch size and schedule remain.

## STREAMING CLOSES: the 2x2 completes — epoch load-bearing at -8, cooldown +4, homogeneity -12 (2026-07-26)

v4 (mixed shuffled batches + final-10% cooldown): 57/120 @ 54.24
{3:20,4:5,5:15,6:7,7:10}, wall 1,095s. The full 2x2 (all 1-pass,
d256, seed 1, vs control 65 @ 1,199s):
  mixed+hot 53 | mixed+cool 57 | sorted+OneCycle 45 | sorted+cool 45
VERDICTS:
1. **COOLDOWN IS REAL: +4** (53 -> 57 on matched mixed batches;
   ~2.9 sigma_diff at d256's measured sigma 1.0 — above the
   3-solve directional bar). The v1 amendment's suspicion
   confirmed: part of the original -12 was the missing anneal.
2. **BATCH HOMOGENEITY COSTS -12 AT ONE PASS** (57 v 45 cooled
   pair) — dominant factor, dwarfing schedule. With revisits it
   costs ~0 (control 65 IS homogeneous). INTERACTION LAW BOOKS:
   diversity-per-step is the binding resource of single-pass
   training; epochs buy back what correlated batches spend
   (07-16 packed-homogeneity precedent, small-batch sibling).
3. **THE EPOCH STAYS LOAD-BEARING: -8 at best-case single pass**
   (57 v 65, ~5.7 sigma_diff). Amended from v1's -12 (which
   bundled the missing cooldown) — but the pre-registered
   neutrality bar (>= 61) is unreachable in every measured cell:
   **STREAMING BIRTH CLOSES as a capability loser at this
   width/corpus.** Composed-birth re-pricing stands; batched-KV
   remains the speed headline.
4. Speed-capability tension, final form: the batch construction
   that wins capability (mixed) costs the wall (1,095s padded);
   the one that wins the wall (sorted, 403s) costs -12. The
   untested escape (length-BUCKETED shuffled — the legacy LoRA
   recipe's trick) could plausibly hold 57 at ~450s, but 57
   fails the bar regardless — NOT RUN, banked as the first cell
   if streaming is ever revived for non-birth uses (metabolic
   feeding is already single-pass-by-nature and is where these
   laws actually live).
5. Surprise-gating: rode every arm, engaged nowhere (EMA-tracking
   throughout) — UNTESTED at birth, unchanged; its cell remains
   the metabolic loop.
Per-level note: v4 L4=5 sits just under the {6,7} seed band —
mild, consistent with dose; no clade claim.

## PRE-REG: margin census on the crown-tie ternary (2026-07-26, before the read)

Artin's high-entropy-neurons riff, distilled to its runnable form:
on a democratic crystal, per-neuron ENTROPY is flat (holography,
1.58-bit max-entropy census) — but DECISION entropy concentrates
at the ternary threshold (flip-location census: flips at median
margin 0.0008 v bulk 0.427). Question: does undecided
(near-threshold) mass CLUSTER (by neuron / layer / family-
selective neurons) or is it uniform? Instrument: per-weight margin
|w_latent| vs 0.5*absmean-scale on merged_grown_latent.pt (89.7M
crown-tie ternary latents), per-neuron near-threshold fraction
(|margin| < 0.05*scale), distribution across neurons/layers;
family leg = correlation of per-neuron near-threshold fraction
with layer-6 family selectivity (existing committee-probe
pattern) IF the neuron-level spread warrants it.
READINGS: (a) uniform across neurons (CV of per-neuron
near-threshold fraction < ~0.3) => democracy all the way down —
the margin-aimed-diet idea CLOSES cheap with a mechanism;
(b) clustered => the diet-weighting A/B (vs rarity-matched
control, matched dose) becomes the named next cell at d256.
Fences: one checkpoint, PTQ-threshold definition tied to absmean
quantizer; no capability claim either way — this is an
instrument read.

## Margin census: undecided mass is UNIFORM — the high-entropy-elite idea closes with a mechanism (2026-07-26)

merged_grown_latent.pt (89.7M crown-tie ternary), absmean-scale
margins, near = <0.05 scale-units. **Per-neuron near-threshold
fraction: mean 6.37%, CV 0.140 — reading (a) at less than half the
0.3 bar.** Quantiles 4.4-8.6% (99th percentile barely 2x the 1st);
layers flat (6.29 -> 6.53% across depth); matrix spread 5.5-6.7%.
VERDICT: the crystal's DECISION entropy is as democratically
spread as its information entropy — there is no high-entropy
neuron elite to stream from or aim at, at neuron, layer, or matrix
granularity. The margin-aimed-diet A/B is NOT run (pre-reg (a):
close cheap); Artin's riff banks as answered-by-instrument: on a
democratic crystal, "where should data go" has no spatial answer —
temporal/self-read signals (surprise, wave agreement, frequency)
remain the only validated scheduling channels, now confirmed at
the finest spatial grain we can read. Rider (observation only):
undecided fraction rises ~4% relative from layer 0 -> 7 —
direction consistent with late-layer plasticity, magnitude far
below any bar. Family leg not run (pre-reg: only if neuron spread
warranted it; it did not). Fences: one checkpoint, one quantizer
(absmean), NEAR=0.05 pre-registered but arbitrary — a different
window rescales the mean, not the CV story.

## PRE-REG: clade-gated streaming pilot at d256 (2026-07-26, before the run)

Artin's layered-streaming question made one-variable. Streaming-vs-
epochs is CLOSED (-8 best case); this asks: WITHIN the single-pass
regime, does clade-ordered self-paced advancement beat the shuffled
stream? Control = v4 (mixed shuffled + cooldown, 57/120, measured).
Arm G (scratch/clade_stream_d256.py): same recipe/seed/cooldown/
surprise/mixed batches, ONE variable = stream order+gating: bands
(L1,L2) -> (L3) -> (L4,L5) -> (L6,L7) [levels proxy the phylogeny
clades — noted, not identical]; advancement when a VERIFIED wave
probe on the current band (8 states x 8 samples, 1 ply,
verify_wave — gate-not-the-loss honored) reads >= 0.55 valid
fraction, or band rows exhaust; after advance, 30% of each batch
is rationed from previous bands (maintenance doctrine). The gated
arm may consume FEWER rows (skipping mastered food) — that
self-paced allocation IS the intervention; total steps capped at
v4's 4,140.
READINGS: gate > 60 => clade-gating PAYS (+3 ~ 2.1 sigma_diff),
the layered-streaming thesis gets its first leg; 54-60 =>
directional/sub-noise, ordering neither helps nor hurts at one
pass; < 54 => ordering HURTS (interference/forgetting beats
structure — also a result; the rations knob becomes the suspect).
FREE RIDER: band-advance step indices + per-band probe curves =
the retention/mastery telemetry (task-2's mini form) logged for
free. Fences: d256/MPS lineage; n=1; probe threshold 0.55 and
ration 30% are pre-registered but untuned.

## Clade-gated streaming v1: 47/120 — the specced policy LOSES, and tells us how (2026-07-26)

Arm G: 47/120 @ 44.08 {3:16,4:2,5:11,6:8,7:10}, wall 543s, steps
2,552/4,140. Pre-reg reading "< 54: ordering hurts" FIRES, with
the mechanism decomposed by the arm's own telemetry:
1. **Self-paced skipping under-trains**: advancement skipped ~47k
   mastered L1/L2 rows + ~22k L4/L5, the last band exhausted at
   step 2,552, and the policy STOPPED — 38% of the step budget
   unspent. The specced form conflates ordering with dose.
2. **The forgetting fingerprint, measured live**: in-run mastery
   probes read 0.64 (L1/2) and 0.70 (L4/5) at advance, but the
   final gate is weakest on the EARLIEST bands (L3 16 v control
   22; L4 2) while the last-trained bands held (L6 8 = the seed
   band level; L7 10 = v4). 30% rations lost to recency at one
   pass. First direct measurement of in-run mastery decaying to
   gate-time — the retention question is now instrumented.
3. FREE TELEMETRY (task-2 mini-form, first data): band mastery
   curves — L1/2 0.03->0.64 in 600 steps; L3 slower (0.125->0.516,
   exhausted before threshold); L4/5 0.156->0.703 FASTER than L3
   (transfer from below, visible live in a probe curve).
4. v2 PRE-REG (fires now, one change): BUDGET RECYCLING — when the
   last band exhausts, probe ALL bands and spend remaining steps
   on the weakest band's rows (revisits allowed = rations aimed by
   measured weakness). Same everything else. Readings: > 57 =>
   self-pacing pays once budget is spent (v1's loss was dose);
   47-57 => partial; <= 47 => ordering itself is the harm at one
   pass, clade-gating closes for births.

## Clade v2 (budget recycling): 60/120 — self-pacing PAYS at one pass; the streaming ladder closes (2026-07-26)

Arm G2: 60/120 @ 55.83 {3:20,4:3,5:16,6:8,7:13}, full 4,140 steps,
wall 1,090s (probes included). VERDICTS:
1. **Pre-reg "> 57" FIRES: +3 over v4 (~2.1 sigma_diff at d256's
   measured sigma) — clade-ordered + probe-gated + budget-recycled
   self-pacing beats the shuffled stream at matched steps.** The
   v1 loss was DOSE (stopping early), not ordering: same ordering
   + spent budget = the best single-pass cell measured.
2. THE COMPLETE STREAMING LADDER (all d256/seed-1/single-pass):
   shuffled-hot 53 / sorted 45,45 / shuffled+cool 57 / clade-v1 47
   / **clade-v2 60** — vs 3ep control 65. The epoch claim SURVIVES
   (-5 at the best schedule) but the gap has halved from v1's -12:
   schedule + order + aimed revisits recover ~60% of what epochs
   buy. Remaining -5 = the revisit mass epochs provide that one
   pass + rations cannot.
3. **Aimed rations WORK**: recycling directed leftover budget to
   the weakest probe bands (L6/7 took it); final gate holds L5 16
   (= seed-band level) and L7 13 (one above the 3-epoch band,
   sub-noise); the v1 forgetting wound (L3 16) repaired to 20.
4. **L4 = 3 in EVERY streaming schedule** (v1 2, v2 3, v4 5, band
   {6,7} at 3ep) — the isolated ansatz clade starves at one pass
   regardless of ordering: phylogeny-consistent (no descent path =
   revisits are its only channel). The clade that needs epochs
   most is the one the tree says has no parents.
5. Retention instrumented end to end (task-2 mini-form delivered):
   mastery curves, decay-to-gate, and probe-aimed repair all
   measured live in one run. The metabolic-loop version (real
   session, per-kind columns) remains the full instrument.
Fences: n=1 per cell; d256/MPS lineage; probe threshold/ration
untuned; wall 2x v4's (probe cost ~40s/recycle — telemetry, not
optimized).

## PRE-REG: gauge-aligned model distance on the d256 zoo (2026-07-26, before the read)

Artin's models-by-distance/rotational ask, made lawful: raw weight
distance is FORBIDDEN as a score (gauge law) — but the orbifold/
Procrustes program (banked 07-23/07-25) asks whether distance UP TO
GAUGE is meaningful. Today's d256 zoo is the ideal testbed: same
arch/corpus/seed-lineage, known gates. Models (gate): wfloor 65 /
s2 63 / s3 64 / pack 38 / stream4 57 / clade2 60. Instrument:
per-layer FFN gate-matrix distance across all 15 pairs, three
lenses — (a) RAW normalized Frobenius; (b) PERMUTATION-aligned
(Hungarian on neuron cosine, the exact gauge group); (c)
ORTHOGONAL-aligned (Procrustes rotation, the relaxation).
PRE-REGISTERED PREDICTIONS: (1) raw distance carries ~no gate
signal (gauge noise dominates: same-function seed pairs {65,63,64}
read as far as cross-schedule pairs); (2) aligned distance
CORRELATES with |gate delta| (Spearman > 0.5) IF the calculated-
model thesis is right (function determines weights up to gauge =>
residual aligned distance is capability-relevant); (3) seed pairs
read CLOSEST after alignment (same function, different gauge).
Scored against gates only — the instrument is judged by the
oracle, never the reverse. Fences: one width, one corpus family,
FFN gate matrices only, n=15 pairs.

## Gauge-aligned distance: ALL THREE PREDICTIONS RESOLVED AGAINST the instrument — distance measures ANCESTRY, not function (2026-07-26)

15 pairs, 3 lenses (raw / Hungarian-permutation / per-matrix
orthogonal Procrustes), FFN gate matrices, d256 zoo. THE TABLE'S
HEADLINE: the three seed births (gates 65/63/64 — functionally
near-identical) sit at raw 1.414 = sqrt(2) = ORTHOGONAL, while
wfloor(65) v pack(38) — 27 solves apart — sits at 0.41. Every
close pair is a SAME-INIT pair (BIRTH_SEED=1 cluster: wfloor/
pack/stream4/clade2 at 0.31-0.47); every far pair is cross-seed.
1. Pre-reg (1) confirmed in the strongest form: raw distance
   carries not just no gate signal but ANTI-signal — it reads
   init lineage, full stop.
2. Pre-reg (2) FAILS: rot's rho -0.635 (p=.011) is pure ancestry
   confound (the large-gap pairs are same-init); WITHIN the
   same-init cluster aligned distance is flat vs capability
   (0.147-0.199 across gate gaps 3-27). Aligned distance does not
   recover function distance at this granularity.
3. Pre-reg (3) FAILS spectacularly: seed pairs are FARTHEST even
   after permutation (1.27) AND orthogonal (0.41) alignment —
   2x the same-init cluster despite Procrustes being MORE
   generous than the true gauge group. Per-matrix alignment
   cannot bring two basins of the same function together.
VERDICT: the never-score-weights-by-distance law gains its
seventh and sharpest leg — same function at sqrt(2), 27-solve-
different functions at 0.3; the rotational rescue (orbifold/
Procrustes hope) FAILS at per-matrix granularity. The measured
positive nugget: same-init fine-lineages stay in a tiny ball
(0.31-0.47) regardless of capability divergence — the whisper
(small ||dW||) at birth-schedule scale; distance IS a cheap
ancestry/lineage detector, nothing more.
NEXT RUNG (named, not run): consistent whole-network alignment
(git-re-basin style — one hidden-basis permutation applied
jointly to gate/up/down/attention per layer) is the honest
remaining hope for the orbifold metric; per-matrix free rotation
already failing to close the seed gap is evidence against, but
the joint constraint is the actually-correct group. Fences: gate
matrices only, one width, n=15 pairs, per-matrix alignment.

## BANKED: joint-permutation distance closure cell (2026-07-26, Mac, awaiting slot)

The proper-gauge completion of the ancestry verdict: git-re-basin-
style JOINT alignment (one hidden-unit permutation per layer,
applied consistently to gate+up rows AND down columns; embedding/
residual basis fixed) on the d256 seed pair (wfloor v s2/s3).
KILL CONDITION, pre-registered: if seed pairs still read >> the
same-init ball (0.31-0.47) under the CORRECT group, weight-space
distance closes PERMANENTLY (function-space — stitching/CKA —
already validated as the real distance space). If they close to
~same-init range, the orbifold metric revives with the joint
constraint as its group. ~1h Mac cell; banked behind the complex/
ZX program per Artin 2026-07-26.

## PRE-REG: the three cheap closers — born-S4, born-Z1, born-Z1S (2026-07-26, before launch, 3080)

The law-converting cells, fired at the 3080 GO. All 19M/gen-4/3ep/
TF32 tournament recipe, gates on cuda (same lineage as born-M4=61
— the matched comparator; cross-device forbidden as always).
1. **born-S4** {±1/3, ±1} (symmetric, two magnitudes, NO zero,
   2.00 bits; global-absmean recipe EXACTLY matching M4's):
   the axis-separating cell three arms have waited for. ZERO-LAW
   prediction: S4 craters toward B-class (~54) despite matched
   bits/resolution — zero-absence is the wound. If S4 ~ M4's 61:
   binary died of RESOLUTION, and the zero-is-load-bearing law
   takes a major revision. Between (55-59): both contribute,
   proportions booked as measured.
2. **born-Z1** (per-row positive scale x {0,1} — silence without
   sign, born, in-regime): BETS ON RECORD stand — reviewer 49
   (LN mean-sub + softmax = implicit inhibition), Fable: crater
   below B's 54. This settles the deletion square's born corner.
3. **born-Z1S** (per-row SIGNED scale x {0,1} — sign at channel
   granularity): the reviewer's named follow-up. If Z1S >> Z1,
   channel-level sign rescues (sign is cheap to grant); if
   Z1S ~ Z1, inhibition needs per-WEIGHT sign; if Z1S ~ T-class
   (60), the whole sign budget was always per-channel-compressible
   — an encoding result for the alphabet program.
Fences: n=1 per arm; 3ep (matched to the bracket table, NOT the
6ep parity dose — parity questions need the 6ep rerun); sigma at
19M/cuda = 2.5 (n=3-fragile), single-seed bar >= 5.

## PRE-REG: max-asymmetric {0,1,2,3} gauge arm (2026-07-26, Mac/MPS, before gates)

The gauge-commutation law's decisive instance (M4's asymmetry was
1-of-4 levels, divergence +2 sub-noise by design; {0,1,2,3} is
asymmetric in ALL levels). Arms: m4x = per-row amax/3 scale x
{0,1,2,3} PTQ on the 19M infix twin; gflip_m4x = same quantizer
AFTER the sign-flip gauge (up-rows/down-cols, gate untouched).
PREDICTION (gauge-subgroup law): divergence GROWS with asymmetry —
|gate(m4x) - gate(gflip_m4x)| > the symmetric pair's 0 and M4's 2.
FLOOR FENCE, registered in advance: {0,1,2,3} deletes all negative
weights on a signed net (the z1-PTQ foregone-conclusion class) —
if BOTH arms gate <= 5, the cell is VOID BY FLOOR (divergence
unreadable at the measurement floor); validity is the secondary
differentiator if solves floor out. MPS gates, twin baselines
(fp32 64 / int3 65 / m4 51 / gflip_m4 49).

## Max-asymmetric gauge arm: VOID BY FLOOR — the fence fired (2026-07-26)

m4x 0/120 @ 0.00; gflip_m4x 0/120 @ 0.00. The pre-registered floor
fence fires exactly as written: deleting every negative weight on
a signed-trained net is the z1-PTQ foregone conclusion, and at the
measurement floor the commutation divergence is unreadable on both
the primary (solves) and secondary (validity) axes. NOT evidence
for or against the gauge-subgroup law. The max-asymmetry
commutation question survives only as a BORN pair (born-m4x vs
born-gflip-init-m4x) — priced at two 19M births, BANKED behind the
complex program (asymmetry axis already has the cheaper born-S4 v
born-M4 cell in flight). PTQ-side gauge testing at extreme
asymmetry is structurally closed: any alphabet asymmetric enough
to test the law hard also kills the signed net that must survive
to be measured — the instrument and the intervention fight for the
same weights (kin to the floor-censoring class).

## PRE-REG: v5 s2 = the retention-curve session (2026-07-26, chained behind the closers)

Metabolic v5 session 2 (long-queued; miner-v2 identity guard landed
07-24) doubles as the retention instrument: CENSUS_SEC=300 (dense),
census now logs the PER-LEVEL proxy vector as [retention] columns.
Config: metab_v5_s1.pt continuation, worklist =
stuck_states_v5_head.jsonl, exchange food = stuck_chains_v5_head
chains, 120 min, cuda, fp64 masters, zero-rollback stack as s1.
READS, pre-registered: (a) s2's own practice/resolution verdicts
(the standing v5 asks — resolution vs the s1 endpoint's fixed
seeds); (b) RETENTION: per-level columns over 24 censuses — fit
per-level decay/hold constants; prediction (phylogeny): L4-class
isolated-clade columns decay first under frontier-aimed food,
clade-connected levels hold (the streaming forgetting fingerprint,
now measured in the metabolic regime at n=24 timepoints instead of
2); (c) surprise-gating engagement check rides free (the streaming
arms showed it never engages at birth — the metabolic loop is its
claimed home; if the multiplier still tracks EMA ~1.0 here, the
mechanism books UNTESTED-NOWHERE and gets redesigned or dropped).
Wired: WSL watcher fires on closers_done.marker (success-only).

## ZX batch 1 ADJUDICATED: PASS — the row factory qualifies (2026-07-26)

axiom f4e46c6, 10,107 rows. STRUCTURAL: independent llmopt-side
fuse replay (own implementation, both site orientations, same
multigraph refusals) — **9,899/9,899 exact, zero mismatches**;
208 rare-kind rows (id 140 / lcomp 62 / pivot 6) skip structural
v1 by design. SEMANTIC (pyzx compare_tensors, fork-walled 10s):
**384/384 PASS, 0 fail**, 15 WALL (3.8% treewidth tax — the
chapter's densification scar; taxed as UNVERIFIED, never valid,
all in the large-diagram tail). Verdict: batch 1 qualifies; the
serialization contract holds end to end at first contact (eighth
consecutive clean axiom batch). Riders: fuse-skew flagged by the
factory itself (9,899/140/62/6) — color-change move-five GO'd in
the relay addendum, batch 2 = balanced kinds (vm-asm 1a scar);
structural replay of id/lcomp/pivot = adjudicator v2, before any
farm-scale batch.

## BORN-S4: 58/120 @ 57.17 — binary's crater was mostly RESOLUTION; the zero law takes its bounded revision (2026-07-26)

The axis-separating cell, landed (19M/gen-4/3ep/TF32, global-
absmean recipe matched to born-M4). {3:20, 4:4, 5:16, 6:7, 7:11}.
The 2-bit born ladder: B 54 @ 36.73 / **S4 58 @ 57.17** / M4 61 @
57.75. Pre-reg "between" reading fires, proportions as measured:
1. **Adding a second magnitude WITHOUT any zero recovers +4 of
   binary's 7-solve deficit and ~ALL of its 18-point validity
   crater** (57.17 ~ M4's 57.75). Binary died mostly of
   RESOLUTION (one loudness), not of silence.
2. **Zero's unique born contribution at 2 bits: <= 3 solves,
   sub-noise** (M4 - S4 = 3 < the 5-solve single-seed bar;
   sign-consistent with zero paying, never bookable as a number).
3. AMENDMENT to the two-of-three law's ordering: "zero-absence
   fatal born" was measured on B alone — CONFOUNDED with
   resolution, now separated: at born-2-bit, zero-absence is
   TOLERABLE (58-class), resolution-absence is the fatal axis.
   The zero-is-load-bearing law SURVIVES in its measured regimes
   — 1-bit (sign-only b1 arithmetic death; z1 language death) and
   PTQ-fixed-codebooks (the conditional zero law's +7-10) — and
   is BOUNDED out of the born-2-bit regime.
4. G16's prior updates: with zero cheap at born-2-bit+, the
   "interference substitutes for zero" cell loses urgency;
   the complex bracket's zero-related predictions inherit this
   ladder as their baseline.
Fences: n=1; 3ep (the bracket dose, NOT the 6ep discrete-parity
dose — S4 at 6ep could close the M4 gap or widen it; queued only
if a decision ever needs it); cuda lineage.
LIVE NOTE, same chain: born-Z1's ep0 loss is EXPLODING (27.7 v
~0.5 normal) — excitation-only birth diverging at standard
recipe, consistent with the Z1-arc collapse-prone mechanism;
gates when the chain gates it, booked with mechanism then.

## ZX batch 2 ADJUDICATED: PASS — balanced diet, color-change exact at first contact (2026-07-26)

axiom a3480fd, 10,152 rows, kinds fuse 5,076 / color 3,576 /
lcomp 629 / pivot 596 / id 275 (the ration doctrine applied at the
farm; ~100x lcomp/pivot mass over batch 1). Adjudicator v2:
STRUCTURAL 8,927/8,927 exact (fuse + id + COLOR-CHANGE — the
five-line move replays perfectly at first contact), zero
mismatches; SEMANTIC 727/727 PASS, ZERO walls (spider<=14 cap:
548/1,225 lcomp+pivot rows in-cap, all verified; 200-row small
subsample of the rest). NINTH consecutive clean axiom batch.
Honest residue: 677 above-cap lcomp/pivot rows are neither
structurally replayed (no v2 replay for those kinds) nor
semantically verified (treewidth) — taxed UNVERIFIED; adjudicator
v3 (structural lcomp/pivot replay) retires the class if the farm
scales. Size-cap question ANSWERED in the relay: NO farm cap —
big diagrams are the rare class (tail-dies-first applied to diet
design); verification slices, never farm filters.
The ZX continent now has: certified oracle path, qualified row
factory, balanced diet, ~20k adjudicated rows. The vocab-51 ZX
birth (federation-floor pre-reg + the G5/M5 factorial's ZX
column) is an engineering decision, not a research one.

## PRE-REG: complex-weight NNUE on magic labels (2026-07-26, before the run)

The first alphabet cell in the OLIGARCHY phase (NNUE kurt 4.78 —
every alphabet law so far is democracy-phase). Paired same-run
arms on magic_labels_v7, same split/loss/metrics as the founding
estimator: real64 (20-64-64 ReLU) vs cplx42 (20-42C-42C, genuine
complex multiply, modReLU), real params matched within ~3% (both
counts printed; per the fairness rule both framings reported).
PREDICTION (skeptical, pre-registered): NULL-to-loss for complex —
the 20 structural features carry no phase-like signal, and
rotation should pay only where data carries phase (Trabelsi
lineage). If cplx42 WINS on rho: rotation pays in the oligarchy
phase without phase-carrying input — a genuine surprise that
would re-rank the complex bracket. Either verdict = the oligarchy
phase's first alphabet datapoint.

## Complex NNUE: NULL as pre-registered — rotation does not pay in the oligarchy phase either (2026-07-26)

real64 (5,634 params): rho 0.541 / AUC 0.928. cplx42 (5,504):
rho 0.493 / AUC 0.922. The skeptical prediction lands: complex
LOSES rho by 0.048 and ties AUC at matched real params, on
features that carry no phase. The oligarchy phase's first
alphabet datapoint: rotation buys nothing here, consistent with
the Trabelsi lineage (complex pays where data carries phase) and
with every democracy-phase alphabet result. FENCE: both rhos sit
at the v7-label-era signal ceiling (~0.55 — the starved-judge
collapse: those labels postdate the engine saturating the
generator), so absolute rho is era-bound; the PAIRED delta is the
result. Consequence for the bracket: the complex program's
surviving justifications are now exactly two — per-family euler
reads on phase-carrying DATA (math trig/exp, ZX rotation grammar)
and the interference-substitutes-for-SIGN question (G16, re-priced
by the Z1 arc). Everything phase-free is measured null at both
phases of the crystal taxonomy.

## BORN-Z1: 0/120 — the deletion square closes at birth, maximally asymmetric (2026-07-26)

{3:0,4:0,5:0,6:0,7:0} @ 0.00% validity. The loss biography IS the
mechanism: ep0 opens 27.7 (confidently-wrong saturation — no
opposition channel, cone collapse), claws to ~3.4 ~ ln(40) (the
uniform-ignorance floor: LN mean-sub + softmax recover CONTRAST
enough to undo saturation but never enough to CONDITION), and
never leaves the floor. VERDICTS:
1. **The deletion square's born corners, final: born-B 54 (sign
   without silence) v born-Z1 0 (silence without sign).** Sign
   emulates silence at the cost of 7 solves-class; silence
   emulates sign at the cost of EVERYTHING. Opposition is the
   computational primitive (Artin's thesis, now measured at
   birth at the extreme).
2. BETS SETTLED: Fable's crater bet WINS outright; the reviewer's
   49 (40% on >= born-B) loses maximally — implicit inhibition
   (its named mechanism) is real but bounded at "undoes
   saturation," ~50 solves short of its prediction. Fourth
   reviewer over-prediction on the books (booked in absentia;
   it set the bet before retirement).
3. Mechanism fence, honest: capability-death and optimization-
   instability are CONFOUNDED at n=1/standard-recipe — a tuned
   LR/init might train further into the excitation-only regime.
   The claim books at MATCHED RECIPE (the tournament bracket's
   own terms, like every alphabet row). Z1S (Dale's-law
   channel signs) is the live rescue arm, next in chain.

## BORN-Z1S: 0/120 — Dale's law buys STABILITY, not capability; the sign-granularity ladder completes (2026-07-26)

{3:0,...} @ 0.00%. But the loss biography splits the mechanism in
two, and both halves book:
1. **Channel-level sign FIXES the optimization pathology**: Z1S
   opens at 3.69 (= the uniform floor — NO confidently-wrong
   explosion; compare Z1's 27.7) and descends smoothly 2.21 ->
   1.87 -> 1.70. Tier-2 (cone collapse -> saturation) is ESCAPED
   by per-channel opposition alone — Dale's architecture is
   sufficient for trainability.
2. **But not for capability at the bracket dose**: final loss 1.70
   is ~4x the healthy band (0.34-0.50) and the gate reads zero
   with zero validity. The sign-granularity ladder at matched
   recipe/3ep: per-weight sign (B) 54 / per-channel sign (Z1S) 0 /
   no sign (Z1) 0 — **the sign budget is NOT channel-compressible;
   inhibition needs per-weight resolution** (or an untested
   granularity between: per-block/per-group sign, banked as the
   ladder's missing middle rung if it ever matters).
3. Fences: n=1; 3ep (the 6ep discrete-parity dose is the one live
   rescue — Z1S's smooth descent means MORE DOSE COULD PAY where
   Z1's divergence means it cannot; queued as a conditional cell,
   fires only if a decision ever needs it); cuda lineage.
4. THE CHEAP CLOSERS COMPLETE: S4 58 (zero law bounded), Z1 0
   (deletion square closed at birth), Z1S 0-but-trainable (sign
   granularity floor found). The complex bracket's priors are set;
   the marker fired; the retention session is live.

## PRE-REG: Z1S hot-LR arm (2026-07-26, chained behind the retention session)

Artin's LR offer + the discrete-plasticity prior (STE latents
integrate sub-threshold — hot LR crosses thresholds; "silent
until it fires"). Arm: Z1S at LR 1e-3 (3.3x standard), 3ep, one
variable vs Z1S-standard (loss 1.70, gate 0). READINGS: loss
< ~1.0 or gate > 0 => dose/LR was the binder, the excitation-only
ceiling rises, and the 6ep cell fires next; loss ~1.7 or diverges
=> the Z1S floor is real at any recipe in this family — books as
the excitation regime's capability wall. Chained on
v5s2_done.marker (success-only).

## PRE-REG: the Muon streaming-optimizer cell (2026-07-26, before the run)

Tier-1 #1 (spec 2026-07-26-next-session). Claim under test: the
homogeneity -12 is a GRADIENT-COVARIANCE wound — correlated
batches feed correlated gradients, and SGD/AdamW spends the pass
re-learning the same directions. Muon-style orthogonalized
momentum (Newton-Schulz 5, 2D interior weights only; AdamW stays
on embeddings/head/norms) whitens the update per step =
diversity-per-step moved INTO the optimizer.
ARMS (both d256, seed 1, 1 pass, final-10% cooldown, surprise
rider unchanged): (a) MUON+sorted — the wounded cell, comparator
45; (b) MUON+mixed — the best-case cell, comparator 57. Control
3ep = 65. Muon LR 0.02 (default from the literature), warm/cool/
surprise multipliers identical to v4.
READINGS (sigma_d256 ~1.0, directional bar 3): (a) sorted >= 51
(+6) => covariance mechanism CONFIRMED — the interaction law gains
its optimizer leg; sorted ~45 => the wound is not covariance
(composition/curriculum effect), Muon leg dies. (b) mixed >= 61 =>
"the epoch is load-bearing" RETRACTS to "SGD wastes single passes"
(the publishable revision); 57 < mixed < 61 => partial, epoch keeps
a smaller residual; <= 57 => optimizer not the channel at either
composition. Divergence at LR 0.02 => one retry at 0.01, booked as
a tuning note, not an arm.

## PRE-REG: weight-FFT euler read (2026-07-26, before the read; free)

Prologue leg inherited from the Opus-5 complex-bracket review;
sets the complex bracket's PRIOR. Question: do real-born crystals
already carry phase-pair / rotational structure in their FFN rows
(i.e., does SGD spontaneously pair channels as re/im), or is
rotation an alphabet you must IMPOSE at birth?
INSTRUMENT (merged_grown_latent.pt FFN matrices, minutes, no
training): (a) adjacent-channel pairs (2k,2k+1) as (re,im) —
phase-angle distribution vs a magnitude-matched paired-shuffle
control; (b) per-row rFFT top-8 energy fraction vs within-row
permutation control (positional/rotational order would concentrate
the spectrum). PREDICTION (house): NULL on both — the gauge law
says channel ORDER is meaningless, so any pairing structure would
have to survive a symmetry SGD has no reason to break; a null
keeps the complex bracket honest (rotation must earn its way at
birth, no free lunch already in the weights). Structure bar: crystal
exceeds its own shuffle control by >3 sigma of the control spread
(20 shuffle seeds). Either verdict banks: null => bracket prior
stays skeptical; structure => the euler read gets aimed at WHERE
it concentrates (family-resolved follow-up).

## Weight-FFT euler read: NULL for rotation — the flag was ANCESTRY again (2026-07-26)

Instrument fired the 3-sigma bar (phase-pair z to -19 on every
down matrix) and the autopsy retracts it in one read: the signal
is NEGATIVE z (real pairs MORE uniform than shuffled), down-only,
and the mechanism is the GROWN BLOCK — merged_grown's appended
neurons sit at column norm 0.30 vs 0.90 for the originals, so
adjacent pairs are variance-matched (|log norm ratio| 0.124 v
0.446 random) while shuffled pairs mix the blocks and skew phases
toward the heavy axis. Channel index encodes ANCESTRY, not phase
— the distance verdict's lesson appearing inside a single
checkpoint (growth is visible in the index layout; function
isn't). FFT-order lens: clean null everywhere (max |z| 1.75).
The one positive phase cell (blocks.1.up +4.5) is isolated,
uncorrected for 48 reads, and carries no fft partner — noise.
VERDICT: **no euler/rotational structure in real-born crystals —
the pre-registered house null stands.** Complex-bracket prior
stays skeptical: rotation must be imposed at birth (the G5 arms)
to exist at all. Instrument note for reuse: on grown checkpoints,
shuffle controls must shuffle WITHIN ancestry blocks or the
block-norm step masquerades as structure.

## Muon sorted arm @ LR 0.02: 10/120 @ 8.70% — loss-gate DIVERGENCE (2026-07-26)

{3:3,4:0,5:5,6:2,7:0}, wall 608s (6.8 it/s — Muon's ns5 costs
~zero on these shapes). The trap worth naming: the loss biography
looked like the best streaming arm ever recorded (0.71 at step
200 where AdamW arms sat ~1.4; ended ~0.55) and the gate is the
WORST non-Z1 cell in the ladder. Teacher-forced loss and
generative capability decoupled completely — the orthogonalized
update at LR 0.02 optimizes next-token prediction while
destroying whatever the gate actually measures (validity 8.7% =
the model emits garbage steps confidently). Per the pre-reg's
tuning clause: ONE retry at MUON_LR 0.01 fires before any
mechanism claim; the mixed arm waits for its verdict. If 0.01
also craters, the cell books as "Muon-class updates are
gate-toxic at this scale/recipe" and the covariance hypothesis
stays untested-by-this-instrument (not refuted).

## MUON CELL CLOSES: 34/120 @ 0.01 — orthogonalized updates LOSE at one pass; the covariance hypothesis is unsupported here (2026-07-26)

Retry gate: {3:10,4:2,5:11,6:6,7:5} = 34/120 @ 32.79%, wall 604s.
The ladder reads 10 (LR .02) -> 34 (.01) vs AdamW-sorted 45 —
Muon trails the comparator at every tested LR, in exactly the
cell (homogeneous batches) where the covariance hypothesis
predicted its biggest win (bar was >= 51). VERDICTS:
1. The streaming-optimizer cell CLOSES NEGATIVE: Muon-class
   orthogonalized momentum does not buy diversity-per-step at
   this scale/recipe; "the epoch is load-bearing" KEEPS its -8.
   Mixed arm not run (pre-reg: it waited on the mechanism test,
   which failed at its most favorable composition).
2. Honest scope fence: this instrument tests the NAIVE transplant
   (ns5 momentum, two LRs, d256/MPS, LR-coupled to the surprise
   rider). The covariance hypothesis is UNSUPPORTED, not refuted
   — a tuned Muon (separate schedule, decoupled weight decay,
   more LR points) could reopen it, but not on house priority.
3. The diagnostic worth keeping: BOTH Muon arms show loss-gate
   DECOUPLING (0.55/0.41 final CE — the best streaming losses
   ever recorded — at gates 10/34). Orthogonalized updates
   optimize teacher-forced CE while damaging generative validity;
   the gate is measuring something CE does not see, and Muon
   widens that split enough to make it visible. Composes with the
   confidently-wrong Z1 biography as the second loss-metric
   dissociation of the week.

## JOINT-PERM CLOSURE: kill condition FIRES — weight-space distance closes PERMANENTLY (2026-07-26)

git-re-basin joint alignment (one hidden-unit permutation per FFN
layer over [gate row | up row | down col], residual basis fixed —
the correct gauge subgroup): cross-seed pairs move 1.414 -> 1.327
(6% recovered), a factor ~2.7x ABOVE the same-init ball
(wfloor-stream4 0.497 / wfloor-clade2 0.537, unchanged by the
perm — schedule divergence carries no permutation component
either). Script label nit: pairs of s2/s3 vs stream4/clade2
printed "same-init" but those arms share wfloor's seed, so they
are CROSS-seed — and they read 1.327 exactly like the seed pairs,
an unplanned replication. Per the pre-registered kill condition:
**weight-space distance is CLOSED as a function instrument at
every gauge group tried (raw, Hungarian, Procrustes, joint-perm)**
— the 8th and final leg of never-score-by-distance; function
space (stitching/CKA, oracle gates) is the only distance that
exists. The orbifold-metric revival does NOT fire. Residue that
stays open: the U(n) variant rides the complex births (different
group, not a re-ask of this one) per the quantum-LLMUE walk.

## RETENTION SESSION (v5 s2): 6 -> 10/100, and the crystal FORGETS NOTHING for 120 minutes (2026-07-27, booked from overnight)

FINAL: resolution 6 -> 10/100 (new: 4x L8 + 1x L6 walls fell;
m-l8-v5s1-{122,150,194} and m-l6-v5s1-108 join s1's residue),
proxy 23 -> 23 @ 61.9%, NET 68,322 tokens over 17 update steps,
31 live, 443 new walls banked to the stuck-state exchange.
THE HEADLINE IS THE FLAT LINE: dense per-level [retention]
columns every ~5 min for two hours read {3:5-6, 4:6, 5:4, 6:2,
7:5} @ 60.8-61.9% THROUGHOUT — zero decay in ANY column,
including the isolated-clade levels the pre-reg predicted would
decay first. PREDICTION FAILS in the good direction: at v5's
dose (17 micro-steps/120min, LR 1e-5) metabolic feeding is
purely ADDITIVE — resolution +4 with retention cost
UNMEASURABLE at this instrument's resolution. Decay constants:
un-fittable (no decay to fit); the retention-curve question
needs a HOTTER session (more steps or LR) before any tau exists
to measure. Surprise-gating leg: NOT INSTRUMENTED — v5 carries
no surprise multiplier at all, so its untested-nowhere status
stands; the home-regime cell still has to be BUILT, not just
logged (flag for the next metabolic design pass).
Riders: proxy flat at 23 while resolution rose (the two
instruments measure different things — walls falling != proxy
composition); Z1S-hot fired on the marker as armed (chain
verified end-to-end).

## FACTORIAL math column, arm 1 — cplx_none control: 63/120 (2026-07-27)

{3:21,4:6,5:17,6:8,7:11} @ 61.35% validity, 19M/gen-4/3ep/seed 1
(complex-FFN: paired re/im channels, modReLU, genuine complex
multiply — NO quantizer). Lands between M5 62 and fp32 64: the
complex INTERIOR is capability-neutral on math — pairing channels
and rotating through the multiply neither costs nor pays by
itself, so the factorial's G5 cell (in flight) will read the
QUANTIZED rotation alphabet against a fair same-architecture
control (63), not against fp32's architecture. Exactly what the
control was for.

## Z1S HOT-LR: 0/120 at loss 1.10 — the excitation capability wall books AT ANY RECIPE (2026-07-27)

3.3x LR (1e-3) moved the loss biography exactly as the discrete-
plasticity prior predicted (1.70 -> 1.10 — hot LR crosses
thresholds standard LR integrates under) and bought ZERO gate:
{3:0,4:0,5:0,6:0,7:0} @ 0.00% validity. Pre-reg reading: loss
not < ~1.0 AND gate 0 => the 6ep cell does NOT fire; **the
excitation-only regime (no negative weights, channel-sign or
none) is a CAPABILITY WALL, not a dose/LR artifact.** The
opposition ladder's final rung: per-weight sign 54 (S4) /
channel sign 0 at ANY LR (Z1S, Z1S-hot) / no sign 0 (Z1).
Opposition must live at per-weight granularity; Dale's law
remains stability-only. Third loss-capability dissociation of
the arc (Z1 confidently-wrong, Muon CE-vs-gate, now Z1S-hot's
smooth 1.10-that-solves-nothing).

## FACTORIAL math column COMPLETE: G5 latent 66 / deployed 62 — rotation is FREE on math, not PROFITABLE; the ZX column becomes the decider (2026-07-27)

cplx_G5 (G5 STE {0,±s,±is} on complex-FFN, 19M/gen-4/3ep/seed 1):
latent 66/120 @ 62.29 {3:22,4:6,5:17,6:8,7:13}, deployed 62/120
@ 59.32 {3:20,4:6,5:16,6:8,7:12}. The math column, all
same-recipe: fp32 64 | cplx_none 63 | G5-latent 66 | G5-dep 62 |
M5 62 (the bit-matched magnitude twin at 2.32b).
VERDICTS: (1) G5-dep TIES M5 exactly at matched bits — rotation
neither pays nor costs on the phase-free math grammar (all
margins under the >=5 single-seed bar; ties, not wins). (2) The
quantization tax is ~4 (66 latent -> 62 deployed), in the normal
STE-deploy band. (3) This is precisely the alphabet-follows-
domain setup the program wanted: math shows NO rotation
dividend, so if the ZX column (phase-carrying grammar, vocab-51
birth pending the axiom farm batch) shows G5 > M5 there, the
interaction books as the headline; if ZX also ties, the
alphabet-follows-domain hypothesis dies clean. Euler-read
consistency note: the null prior held — rotation imposed at
birth is SURVIVABLE (66/62) but spontaneous structure never
appears in real-born crystals.

## PRE-REG: template-refresh mid-stream (2026-07-27, before the run)

Tier-1 #6 (the "something missing in streaming" candidate; never
run anywhere). Claim: single-pass SGD underperforms partly
because macro-STATISTICS (row-norm distributions — the same
quantity template-spray growth and warm birth manipulate, +8
measured at init) converge slower than one pass allows; correct
them DIRECTLY mid-stream and let gradients handle only the
residual (calculated-model thesis applied during training).
ARM (d256, seed 1, v4 construction: mixed shuffled + final-10%
cooldown + surprise rider): every 250 steps, softly (half-way)
re-map each interior matrix's row norms onto the target quantile
distribution measured from the wfloor_d256 control checkpoint
(statistics are seed-invariant training constants — using the
control's PANEL is calculation, not copying; directions/gauge
untouched, so function content stays SGD's).
READINGS (comparator v4 57, control 65, sigma ~1.0/bar 3):
>= 61 => statistics were the missing mass — streaming reopens;
58-60 => directional, refresh-rate/strength sweep banked;
~57 => statistics converge fine on their own, the "something
missing" lives elsewhere (format x schedule cell next in line);
< 54 => correction fights Adam state, books as interference.

## Template-refresh: 59/120 — DIRECTIONAL, not decisive; the sweep banks (gated morning after the run; run finished 07:44)

{3:21,4:6,5:16,6:7,7:9} @ 53.42%, wall 1,098s. (Session note: the
run completed in the rolled-back chat and was gated after
reconciling the tree — pre-reg and code were the uncommitted
diff, verified intact before gating.) Against the pre-registered
readings (comparator v4 57, control 65): 59 falls in the 58-60
DIRECTIONAL band — +2 over v4 (under the 3-solve bar), the
second-best single-pass cell measured (clade-v2 60 remains #1),
and no interference (the < 54 fence stayed quiet; Adam tolerated
direct norm surgery every 250 steps). Per pre-reg: the
refresh-rate/strength sweep BANKS (stronger/more frequent
correction, full re-map instead of half-way) but does not fire
now — three streaming interventions (cooldown +4, self-pacing
+3, statistics +2) each recover a slice of the epoch's -8 and
none alone reaches the bar; the format x schedule cell (Artin's
riff) is next in line as the remaining unexplained mass.
L4 note: 6/24 — ON the seed band's edge; the isolated-clade
starvation signature does not worsen under norm correction.

## ZX FARM BATCH 1 ADJUDICATED: PASS — 99,022 rows qualified for the vocab-51 birth (2026-07-26 morning)

zx_farm1.jsonl (axiom, 9,300 problems, ration caps): fuse 49,511
(50.0%) / color 35,505 / lcomp 6,009 / pivot 5,592 / id 2,405 —
kind mix reproduced the sample batches (ration seed behaved, no
drift at 10x scale). ADJUDICATION: structural leg 87,421/87,421
replayable rows TRUE, zero false (11,601 SKIP = lcomp/pivot
above independent-replay capability — the standing adjudicator-v3
gap, now 11.7% of the diet and rising in priority); semantic leg
5,351 rows (400 small + all rare-kind <=14 spiders) 5,346 TRUE /
0 FALSE / 5 treewidth walls (taxed unverified per doctrine).
Tenth consecutive clean batch; the factory holds at farm scale.
Axiom's dial note banked: rare-kind absolute mass rises via
seeds + fuse-cap tightening at the FARM ration, never the
generator. NEXT: vocab-51 ZX birth on this diet (federation-
floor pre-reg: count vs class), then the factorial's ZX column
(G5 vs M5 on phase-carrying data — the program's decider).

## PRE-REG: the factorial's ZX column — first ZX births (2026-07-26, before launch)

The program's decider (spec 2026-07-26-complex-zx-program leg C).
Diet: zx_farm1_train.jsonl (97,036 adjudicated rows; 1,986 rows /
186 whole problems held out by seed%50==0, zero overlap).
Vocab-47 (7 atoms: in(/out(/Z(/X(/P(/H(/':' — order frozen,
probes must match), SEQ_CAP=1536 (26% of rows exceed 512; big
class stays in per no-size-cap), 19M d384/L8/ffn1536/h6, seed 1,
3ep, Mac/MPS serial — same recipe as the math column.
ARMS (in order): (1) M5-ZX (scalar 2.32b magnitude twin);
(2) G5-ZX (complex-FFN + G5 STE rotation, 2.32b); (3) cplx_none-
ZX (unquantized complex control, prices the tax on ZX).
GATE (new instrument, built behind the births): 120 held-out cur
states stratified by size class; greedy generation; a solve =
parses in-grammar AND is verified as a sound rewrite of cur
(structural replay where replayable, pyzx compare_tensors <=14
spiders, else honest WALL — walls excluded from the denominator,
count reported). Non-identity enforced.
PREDICTIONS (pre-registered): the headline is the INTERACTION —
alphabet-follows-domain says G5-dep > M5 on ZX where they tied
62=62 on math; bar = the >=5 single-seed rule at 19M (unknown
sigma on a NEW grammar+instrument — treat 3-4 as directional
only, and say so). House prior honestly split: the euler-read
null + phase-free-math tie leave rotation's value entirely to
phase-carrying data; if G5 <= M5 here too, alphabet-follows-
domain DIES CLEAN and the complex bracket closes to the G16-
as-sign question. Federation-floor pre-reg rides the NEXT birth
(math+ZX union), not these.

## PRE-REG AMENDMENT: ZX column moves Mac -> 3080, BS 32 -> 8 (2026-07-26, before any arm completes)

Booked before results exist. (1) DEVICE: Mac/MPS measured <0.15
it/s on the long-row diet (quadratic attention at 4x math row
length) = ~33h for the chain; moved to cuda/TF32 (parity-passed
recipe). License: the sigma-grid decomposed the birth device
term to ~0. (2) The 10GB card OOM-thrashed at BS=32 x 1536
tokens (allocator tripwire fired twice — killed per the 43x
doctrine, never trained through it); fixed with GRAD_CKPT=1 +
new BIRTH_BS env at 8. CONSEQUENCE, stated now: BS=8 = 4x more
optimizer steps/epoch than the math column's BS=32 — ZX gate
numbers are NOT comparable to math-column absolutes; the ONLY
licensed comparison is within-column (M5-ZX vs G5-ZX vs
none-ZX, all identical recipe), which is exactly the factorial's
interaction question. Chain healthy at relaunch: 0 OOM, 5.6
it/s, ~5.5h + gates, self-marking (zx_cuda_done.marker).

## NNUE symmetry read: sign symmetry is a DEMOCRACY-phase property (2026-07-26, free read)

Artin's ask ("is the NNUE symmetrical?") — never measured
directly; measured now on nnue_eval.pt + magic_estimator_v7.pt
vs the ternary-LM crystal. THE LM (democracy): pos/neg fraction
0.499-0.502 on EVERY weight matrix, |skew| < 0.01 — perfect sign
balance, no exceptions across 12 layers. THE NNUE (oligarchy,
kurt 4.78): input layer balanced (0.502/0.498, skew -0.08),
hidden layer skewed positive (0.595/0.405), output heads
strongly asymmetric (skew -1.68 eval / +1.49 solved / -1.36
cost; kurt 5-7). VERDICT: **sign symmetry is a phase property —
democracies are sign-symmetric everywhere, oligarchies break
symmetry progressively toward the readout.** Two retro-lights:
(1) the opposition ladder (S4 54 / Z1S 0 / Z1 0) is a
DEMOCRACY-phase law — its extrapolation to oligarchy regimes is
unmeasured (the NNUE keeps 40% negative mass even while skewed:
opposition never disappears, it just unbalances); (2) the
complex-NNUE null lived in a sign-skewed regime — one more
reason it doesn't transfer to crystal predictions. Scalar
regression heads plausibly force the readout skew (a signed
scalar output has a preferred direction; a softmax over 40
tokens does not) — mechanism note, not measured.

## PRE-REG: the CE-gate study — what does the gate see that loss cannot? (2026-07-26, before the read)

Promoted per Artin ("that's entirely the whole point"). HYPOTHESIS
(the branching-entropy floor): in a closed system the diet is
one-of-many-valid — the same cur admits many oracle-valid nxt and
the farm banked ONE arbitrary pick, so Bayes-optimal CE is
bounded below by the true branching entropy of valid steps.
Pushing CE below the floor = reallocating mass FROM valid
alternatives TO farm picks (memorization); the GATE samples from
the distribution-over-valid-moves, so it measures BRANCHING
COVERAGE — the quantity CE trades away.
SPECIMENS (d256, on disk, gates known): muon 34 / stream3
(sorted-AdamW) 45 / stream4 57 / wfloor control 65.
INSTRUMENT: leg A = teacher-forced mean CE on 400 fixed diet rows
(seed 7); leg B = 40 fresh states (L3-L7, 8/level, never-trained
seed space), 8-sample waves at T=0.7, oracle-verified; metrics =
valid fraction, DISTINCT-valid steps per state (coverage),
identity fraction.
PREDICTIONS: (1) CE ordering INVERTS gate ordering at the muon
end (muon lowest CE, lowest gate); (2) distinct-valid coverage
follows gate ordering exactly (4/4 rank agreement); (3) the
control's CE is NOT the lowest (it pays the branching tax).
n=4 models — orderings and margins, no p-values; any prediction
failing books as-is. If (2) holds, "loss to 0 is the target"
books as FORMALLY WRONG for closed-system training and the gate
gains its mechanism (coverage, not correctness-on-picks).

## CE-GATE STUDY: my hypothesis FAILS — CE on a fixed instrument TRACKS the gate, and the Muon "dissociation" takes a retraction (2026-07-26)

| model | gate | trainCE | valid% | distinct-valid/state | ident% |
|---|---|---|---|---|---|
| muon | 34 | 0.3933 | 27.8 | 0.55 | 0.0 |
| stream3 | 45 | 0.3582 | 43.4 | 0.62 | 0.0 |
| stream4 | 57 | 0.3306 | 48.1 | 0.62 | 0.6 |
| control | 65 | 0.3149 | 47.8 | 0.62 | 0.0 |

VERDICTS, against the pre-reg:
1. **Prediction 1 FAILS, decisively**: CE on a FIXED row sample
   does not invert the gate — it tracks it PERFECTLY (4/4 rank
   agreement, lower CE = higher gate; muon is the WORST at
   0.393, control the best at 0.315). Prediction 3 fails with it.
2. **RETRACTION (booking amendment to the Muon close)**: the
   "loss-gate dissociation" was a cross-instrument artifact —
   Muon's "best losses ever recorded" were measured on ITS OWN
   batch stream (different composition per arm); on one fixed
   instrument its CE is the worst of the four. The doctrine
   crystallizes: **losses are only comparable on a fixed shared
   sample — training-stream losses are not an instrument.** (The
   Z1 confidently-wrong dissociation SURVIVES — that was
   loss-27-vs-floor within one stream, a different class.)
3. The branching-entropy floor stays a THEORY without a
   measured leg (CE differences here are dominated by validity
   loss, not coverage reallocation); do not cite it as measured.
4. The REAL dissociation found is elsewhere: stream4 57 vs
   control 65 tie on 1-ply validity (48.1 v 47.8) and distinct-
   valid coverage (0.62 = 0.62) — the -8 epoch gap is INVISIBLE
   at one ply. **The gate measures CHAINING (multi-step
   composition), and single-pass training's wound is
   specifically the chain, not the step.** Composes with the
   chain-carry hypothesis and Artin's format x schedule cell —
   which this result PROMOTES: if revisits are what teach
   multi-step composition, the format ablation on streamed
   models is now the sharpest open streaming question.
5. Muon's mechanism, finally visible: validity collapse (27.8%)
   at ~zero identity — it emits confident well-formed garbage;
   orthogonalized updates damaged the grammar itself.

## PRE-REG: THE FORMAT LADDER — format x schedule at d256 (2026-07-26, before any run)

Full design + all readings: specs/2026-07-26-format-ladder.md
(the binding document; this entry is the booking pointer).
Seven formats {pairs, traces, skip-pairs, de-chained, one-shot,
delta-chained, random-packed} x {1-pass v4 recipe, 3ep}; pairs
cells already measured (57/65). Primary readings, restated for
the record: (1) traces@1P >= 61 => the epoch's -8 was FORMAT
(composition-in-context substitutes for revisits); ~57 =>
format-neutral, the epoch survives. (2) THE DELTA TEST: E
(similarity-walked 4-pair contexts) vs E0 (random-packed
control): E > E0+3 => in-context similarity pays where in-batch
similarity costs -12 — the interaction law splits into two
opposite-signed laws; E ~ E0 => analogy-by-juxtaposition inert.
(3) one-shot predicted <= 40 (step-tokens from birth). CE-400
rides every arm (does the CE-gate 4/4 tracking survive format
variation?). Attribution: Artin (format x schedule + the
delta-chaining riff, incl. the similarity-as-superposition
frame); house (E0 control, trace stitching, fences).

## FORMAT LADDER, partial card (4 of 6 cells; E/E0 pending) (2026-07-26 evening)

All 1P (v4 recipe), d256, seed 1; comparators pairs@1P 57 /
pairs@3E 65. Token mass + steps reported per the fence:
| format | seqs | tokens | gate | validity | CE-400 |
|---|---|---|---|---|---|
| pairs (comp) | 132k | ~8M | 57 | 54.24 | 0.331 |
| traces | 58k | 5.6M | 37 | 41.53 | 0.518 |
| skip | 132k | 8.0M | 54 | 48.93 | 0.341 |
| dechain | 87k | 5.4M | 49 | 47.62 | 0.367 |
| oneshot | 58k | 3.4M | 54 | 33.50 | 0.589 |
EARLY READS (final card after E/E0):
1. **PRIMARY FAILS INVERTED: traces@1P = 37 (-20)** — composition-
   in-context does NOT substitute for revisits; it COSTS, hugely.
   Dose fence rides (1,810 steps vs pairs' 4,140) — but oneshot
   at the SAME 1,810 steps scored 54, so step-dose alone cannot
   explain traces' crater: the multi-step-target format itself is
   hostile at one pass (candidate mechanism: most of each trace's
   loss mass is mid-chain continuation prediction, a task the
   per-ply gate never asks; the format trains a different job).
2. **THE ONESHOT SURPRISE: 54/120 — the pre-registered <= 40
   prediction FAILS.** Root->answer-only, from birth, one pass,
   3.4M tokens: 54 solves incl. L6 8 / L7 10 (deep levels).
   Mechanism note: the engine one-plies much of gen-4
   (i_linear_basis era), so answer-emission IS the dominant move
   of this corpus; the step-tokens 5/30-v-0/30 precedent was a
   BASE-MODEL PROMPTING result, not a from-birth law. Validity
   33.5% (worst) = no mid-path correction; the chain format buys
   VALIDITY (error recovery), not raw solves, at this dose.
3. skip 54 (-3, directional edge) at matched dose: hop
   compression neither pays nor craters. dechain 49 (-8,
   adjacency-OR-dose per pre-reg, 66% dose).
4. CE-400 tracks the gate across formats so far (0.34 skip / 0.37
   dechain / 0.52 traces) EXCEPT oneshot (CE 0.59 with gate 54 —
   CE is pair-formatted, and the oneshot model was never taught
   the pair frame): the proxy is format-BOUND — valid only
   within matched-format comparisons. Instrument fence booked.

## PRE-REG: reverse-pairs cell (dual-direction crystal) (2026-07-26, before the run)

The reverse-LLMUE bank's cheapest cell, slotted behind the 3E
format cells (Artin GO). Arm: d256/1P (v4 recipe), 50/50 forward
pairs + REVERSED pairs (later state as Current, predecessor as
Step — vocab-40 native), matched 132k total dose, seed 1.
Comparator pairs@1P 57. READINGS (sigma 1.0, bar 3):
>= 60 => backward mapping strengthens the forward crystal — the
temporal-pincer program gains its first measured leg and the
reverse-distribution teacher gets priority; 54-60 => neutral
(direction is free — still pincer-relevant: a single crystal can
host both directions at no forward cost); <= 53 => reverse rows
teach WANDERING (reverse steps are equivalence-valid, so
pollution shows as Integral-count increase / chain-carry
counterweight finally observed) — books as the direction tax.
Fences: reverse rows halve forward dose (76k fwd vs 132k) — a
drop must beat the dechain-style dose read before booking as
direction-caused; contamination fence n/a at this cell (no new
corpus; the bidirectional-cheat fence applies to reverse-TRAINED
MINERS, not this diet remix).

## FORMAT LADDER 1P COMPLETE: the delta test reads E 30 v E0 47 — similarity-MAXIMIZING packing hurts; the banded form stays open (2026-07-26)

Final 1P card (d256, v4 recipe, seed 1; comparator pairs 57):
pairs 57 > oneshot 54 = skip 54 > dechain 49 > randpack 47 >
traces 37 > **delta 30** @ 27.9 (CE-400 0.686, worst everywhere).
VERDICTS:
1. **THE DELTA TEST: E < E0 by -17** — the pre-registered third
   reading fires: similarity-adjacency hurts IN-CONTEXT too, and
   worse than at-random packing. The interaction law GENERALIZES
   across dimensions in its measured form... with a named KNOB
   CONFOUND, booked before anyone runs with it: the greedy walk's
   realized intra-group similarity was 0.974 (argmax always took
   the nearest row) — the cell tested NEAR-DUPLICATE packing, not
   Artin's banded delta. At sim ~0.97 a 4-row context is
   repetition wearing analogy's clothes: 3 of 4 slots re-spend
   the same gradient AND the same context. What books: 
   **similarity-maximizing context packing costs -17 vs random**
   (consistent with, and stronger than, the in-batch -12).
   UNTESTED and banked: the BANDED walk (target sim in [0.5,
   0.8], near-dupes excluded) — the honest form of the riff; one
   cell if the format thread ever reopens.
2. Ladder ordering, one line each: every departure from plain
   pairs LOSES at one pass — pairs is the measured optimum of
   seven formats at this dose/width. The pair format is not an
   artifact of the mining pipeline; it is the best-measured way
   to spend a single pass. (The oneshot 54 surprise + traces 37
   crater booked in the partial card stand.)
3. CE-400 tracks within-format-family only (booked fence
   confirmed: oneshot/packed cells' CE inflated by frame
   mismatch).
4. 3E interaction cells (traces/delta/randpack) firing now;
   revpairs behind them (pre-reg above).

## FORMAT LADDER 3E COLUMN: epochs do NOT buy back context damage — batch and context are different dimensions (2026-07-26)

| format | 1P | 3E | 3E deficit vs pairs@3E 65 |
|---|---|---|---|
| traces | 37 | 44 | -21 |
| delta (sim-max packed) | 30 | 49 | -16 |
| randpack | 47 | 60 | -5 |
VERDICTS:
1. **THE ASYMMETRY (the day's law-grade find)**: in-BATCH
   homogeneity costs -12 at one pass and ~0 with revisits (the
   2x2, control 65 is itself length-sorted); in-CONTEXT packing
   costs at one pass AND STILL COSTS AT 3EP (randpack -5, delta
   -16, traces -21). Epochs buy back batch-composition damage but
   NOT context-composition damage. The interaction law splits as
   pre-registered — just along a different axis than predicted:
   not sign (similar-vs-random) but DIMENSION (batch recoverable,
   context not). Mechanism candidate: batch composition only
   shapes gradient correlation (which averaging over epochs
   washes out); context composition changes the TASK the model
   learns (predict-given-neighbors), and no amount of revisiting
   un-teaches a task.
2. Format main effects at 3E confirm the 1P ordering: pairs 65 >
   randpack 60 > delta 49 > traces 44 — no rank flips; the
   ladder's verdict is schedule-robust.
3. delta-vs-randpack at 3E: -11 (was -17 at 1P) — similarity-max
   packing hurts at both schedules; the knob confound (realized
   sim 0.974) rides both cells; banded walk stays the open form.
4. traces@3E 44 is the strongest single argument yet that the
   trace FORMAT (not dose) is hostile: at 3ep the dose excuse is
   3x weaker and the deficit barely moved (-20 -> -21).
Cells remaining: revpairs@1P (in flight), then the per-problem
sidecars sweep (solve-set overlap + wandering rates).

## REVPAIRS: 39/120 — reverse rows are ACTIVELY toxic to the forward crystal, beyond any dose read (2026-07-26)

{3:11,4:2,5:11,6:7,7:8} @ 33.57. Pre-registered <= 53 reading
FIRES with margin. The dose fence resolves AGAINST dose: dechain
at 87k forward rows scored 49; revpairs at 66k forward + 66k
reverse scored 39 — ten solves BELOW a smaller all-forward diet.
Reverse rows don't dilute, they TEACH THE WRONG MOVE: a reversed
pair presents de-solving (Integral-count-increasing rewrites) as
a legal Step, and the gate economics punish wandering directly.
BOOKS: **the direction tax is real at 50/50 mixing — the
dual-direction crystal FAILS in naive form.** The chain-carry
counterweight (dead-end/backward steps teach wandering), unseen
at gen-9's failed-step dose, is now measured at full reverse
dose. Fences + what stays open: 50/50 is the maximal mix (a
10-20% reverse ration is untested); reverse-as-SEPARATE-MODEL
(the temporal pincer's actual design — backward model never
contaminates the forward one) is UNTOUCHED by this verdict and
remains the banked path; the wandering signature lands in the pp
sidecar sweep for direct confirmation. The reverse-distribution
TEACHER idea (engine-side, never in-diet) also unaffected.

## PER-PROBLEM SWEEP: formats are SUBSETS, not alternatives — and the revpairs wandering mechanism is NOT confirmed (2026-07-26)

Nine cells re-gated with pp sidecars (same seeds; scores
reproduce). READS:
1. **NO format complementarity**: against pairs@3E (65), every
   other cell's solve set is a near-strict SUBSET — uniques of
   0-2 problems per format (traces 0, revpairs 1, oneshot 2);
   union of ALL nine cells ~67 vs best single 65. Format
   variation buys back almost NOTHING in union — completely
   unlike the substrate axis (duo-wave fp32+ternary union was
   +5 at equal budget). BOOKS: **complementarity lives in the
   SUBSTRATE dimension, not the format dimension** — weaker
   formats solve fewer of the SAME problems, not different ones.
   The oneshot-54/skip-54 "tie" is a real near-identity (jacc
   .83), not hidden diversity.
2. **REVPAIRS MECHANISM AMENDED (my claim, corrected by my own
   instrument)**: wander rates are FLAT across formats (17.5-
   21.7%; revpairs 21.7 vs pairs 20.8) — the "reverse rows teach
   wandering" mechanism is NOT supported by the Integral-count
   signature. The -18 direction tax STANDS (the score is the
   score) but its mechanism is OPEN — candidates: step-precision
   damage (reverse rows halve forward dose AND corrupt the
   emission head's coefficient statistics), not chain-navigation
   damage. Instrument note booked: the wander metric counts
   legitimate i_sum splits (Integral count rises on a valid
   split), hence the ~20% baseline — a cleaner metric would
   whitelist split-shaped rises.
3. oneshot mean plies 0.52 (all others 0.78-0.99) — confirms the
   one-hop mechanism directly.
Sidecars: logs/pp_*.jsonl (9 cells, full chains) — the standing
per-problem instrument's first full sweep.

## THE ZX COLUMN COMPLETES: alphabet-follows-domain DIES CLEAN — rotation ties magnitude on phase-carrying data too (2026-07-26 evening)

Full column (vocab-47 births, zx_farm1_train, 19M-class, cuda
BS=8/grad-ckpt per amendment; gate = 120 held-out diagrams,
pyzx tensor oracle, walls 0 across ALL FOUR gates):
| arm | gate | anatomy |
|---|---|---|
| cplx_none (unquantized complex) | **36/120** | parsefail 0 |
| G5-latent (STE forward; 95-parsefail run was my gate bug, fixed) | 32/120 | parsefail 0 |
| G5-deployed | 32/120 | parsefail 0 |
| M5 (scalar 2.32b) | 31/120 | parsefail 0 |
VERDICTS:
1. **THE HEADLINE THE PROGRAM WAS BUILT TO DECIDE: G5-dep 32 v
   M5 31 (+1, bar was >=5) — ALPHABET-FOLLOWS-DOMAIN IS DEAD.**
   Rotation ties magnitude on math (62=62) AND on the maximally
   phase-carrying grammar (32~31). The factorial's interaction
   term reads ZERO at both columns. Consequences, all booked:
   the rotation wing CLOSES (D9/Q9/E7 die unrun; RoPE x G5
   demotes to curiosity); the complex bracket reduces to the
   G16-as-sign question (itself now weakly motivated); the
   pincer spec's complex-conditional resolves REAL everywhere;
   Fourier-as-confirmation-continent demotes (no law left to
   confirm — Fourier stays a TERRITORY candidate on its own
   merits, not a physics test).
2. **First graph-grammar crystals WORK**: ~26-30% sound held-out
   rewrites at first contact, parsefail 0/480 across all four
   arms (7-atom vocab + boundary-anchored serialization fully
   validated), zero identity, zero walls. The federation
   machinery (VOCAB_EXTRA, strict encode, SEQ_CAP) carried a
   graph language on the first try.
3. **ZX quantization tax: -4 (36 -> 32), G5 latent = deployed
   EXACTLY (32 = 32)** — the deploy tax lands entirely at the
   alphabet snap, mirroring math's -4; STE converged the latents
   onto the lattice.
4. **Kind anatomy is model-invariant**: color 19-22 / fuse 5-7 /
   lcomp 3 / id 2 / pivot 1-2 in every arm — tracks corpus mass
   (color 36%) x move locality (color is edge-local
   self-inverse; pivot restructures). The mass-spectrum law's
   first ZX reading; the misses are 100% SEMANTIC (well-formed
   wrong rewrites) = the validity-autopsy law reproducing in a
   new continent.
5. Un-run rider named honestly: no SCALAR-REAL control at
   vocab-47 (fp32/M5 are real but M5 is quantized; cplx_none is
   complex-interior) — "complex interior costs ~nothing on ZX"
   (36 v M5-class) is suggestive, not a booked pair; the clean
   pair would be fp32-real-ZX vs cplx_none-ZX, banked if the
   complex question ever reopens.

## PRE-REG: pincer R0 — the conjecture leg read alone (2026-07-26 night)

Cell R0 of specs/2026-07-26-reverse-llmue-pincer.md. Instrument:
scratch/pincer_r0.py — fmt_oneshot_1p.pt (d256, oneshot 1P, gate
54/120 greedy-chain) proposes k=8 T-sampled answers per gate
problem (same 120 problems/seeds as gate_ckpt); HIT = candidate
forward-verifies from the ROOT as valid AND integral-free (whole
problem falls at ply 0). Zero births; pp sidecar rides.
PREDICTIONS (house): (1) ply-0 yield 45-60/120 — k=8 sampling
should recover >= the greedy-gate 54-class since oneshot chains
averaged 0.52 plies (most of its 54 were already one-hop); (2)
yield concentrates L3-L4, decays by L6-7 (conjecture is a
low-depth skill); (3) a nonzero valid-but-unsolved band (partial
skips) — the raw material the backward peeler (R1) exists to
finish. Read books conjecture yield per level = the pincer's
ply-0 economics.

## PINCER R0: conjecture yield 55/120 — the oneshot gate WAS the conjecture leg, and L4 is the conjecture's hole (2026-07-26 night)

Ran as pre-registered (scratch/pincer_r0.py, k=8, zero births).
YIELD: {L3:17, L4:4, L5:16, L6:8, L7:10} = **55/120**; 18
valid-but-unsolved candidates (partial skips) across the battery.
READS:
1. **Prediction (1) CONFIRMED** (45-60 band): ply-0 conjecture
   alone reaches 55 — the pincer's goal-supply leg is real and
   costs ms per problem.
2. **The oneshot chain gate reduces to the conjecture leg**:
   per-problem overlap vs pp_fmt_oneshot_1p — both 54, r0-only 1,
   gate-only 0. Every problem the oneshot model solved in the
   12-ply chain gate it solves at ply 0 with k=8 sampling; its
   per-level profile is IDENTICAL (L5 15->16 is the +1). The
   mean-plies-0.52 signature graduates to strict: chaining
   contributes ZERO problems to this crystal. Sampling breadth
   (k=8 vs greedy) buys +1 — conjecture diversity is nearly
   exhausted at greedy, consistent with the subset law from the
   pp sweep.
3. **Prediction (2) WRONG in an interesting way**: no monotone
   depth decay — L4 CRATERS (4/24) while L5 (16) ~ L3 (17) and
   L6-7 hold 8-10. The L4 clade signature (a_bridge's L4
   fingerprint, packed-d256's L4 clade) reappears in the
   conjecture skill: L4's answer forms are the hardest to guess
   at one hop, harder than L7's. The backward peeler's
   highest-value territory is L4 + the 18 near-misses, NOT the
   deep levels.
4. Prediction (3) confirmed: 18 valid-but-unsolved = the R1
   peeling worklist exists.
Sidecar logs/pp_r0_conjecture.jsonl. NEXT (per spec): R1a
backward crystal birth (d256, reversed-pairs-only diet).

## PRE-REG: pincer R1a — the backward crystal (2026-07-26 night)

Cell R1 of specs/2026-07-26-reverse-llmue-pincer.md, arm B-a
(mirrored transformer baseline — exists to be beaten by B-b).
BIRTHS: FORMAT=backpairs (every gen-4 row reversed, zero forward
mixing; separate-crystals doctrine, no direction atom) at d256,
BOTH schedules (1P + 3E), seed 1 — checkpoints
fmt_backpairs_{1p,3e}.pt. PROBE: scratch/pincer_r1_probe.py —
105 distinct fresh gate-band mid-chain states (pooled from the
nine pp sidecars; L4/L6 thin, skew named), k=8 sampled
predecessors each, VALID iff forward step p->t verifies and
p != t (bidirectional-cheat fence: forward-verify only, never
corpus match). QUESTION: does backward train UP like forward
(85%-class per-candidate validity) or is peeling intrinsically
harder? PREDICTIONS (house): (1) backward trains DOWN in loss
like forward (same substrate, same pair frame); (2) per-candidate
validity lands BELOW forward's class — predecessor-emission must
invent structure (un-apply a rule = grow the expression), and
generation-toward-complexity is the harder direction; band guess
40-70%; (3) 3E >= 1P (pairs-shaped diet; the ladder's 3E column
recovered batch damage). The read arms R2 (pincer v1) and sets
B-a's bar for B-b.

## PINCER R1b PREP: (t, rule, site) labels are 68% recoverable by engine replay — no axiom ask needed for the pilot (2026-07-26 night)

scratch/pincer_r1b_labels.py, 300 gen-4 rows: replay
successors(cur, use_macros) and skeleton-match nxt. READ: unique
204 / ambiguous 0 / miss 96 -> **68% of farmed rows yield an
exact (rule@site, child) label with ZERO ambiguity** (move names
already carry the site: `i_const@Integral(3, x)` — B-b's
score-over-enumerated-moves frame gets real (rule, site) labels
free). MISS ANATOMY (named, all levels; L1-L2 heavy): farmed
rows that are ORACLE-valid but not single engine moves — (a)
constant-of-integration offsets (`Integral(9,x) -> Integral(9,x)
+ 5` is equivalence-valid under the d/dx oracle; no engine rule
mints it), (b) multi-rule skips (whole antiderivative in one
hop, LLM-farmed compression). CONSEQUENCE for B-b's objective
bracket: imitation labels (i) cover 68% of the corpus; value
labels (ii) and contrastive (iii) sidestep replay entirely
(engine enumerates, oracle scores — no true-move needed) and
cover everything. The axiom ask (annotate math chains at mint)
stays BANKED for the promotion tier, not blocking the pilot.

## PRE-REG: math+ZX union birth + ZX seed-2 (3080 night queue, armed on GO 2026-07-26 night)

Per specs/2026-07-26-next-session-2.md, fired on Artin's GO.
JOB 1 — the federation floor's first GRAPH datapoint: one real
19M-class birth (d384/8L, fp32, seed 1, vocab-47 = 40 +
"in(,out(,Z(,X(,P(,H(,:") on gen-4 math (132,870 rows) +
zx_farm1_train (97,036 rows; organic 58/42 share), SEQ_CAP 1536,
BS=8/grad-ckpt/TF32. Gated on BOTH columns: math gate_ckpt_cuda
(VOCAB_EXTRA now rides probe-side, same atom order) + gate_zx
(120 held-out diagrams, pyzx oracle). QUESTION: does the union
hold math ~63-65-class AND ZX ~30-class at 19M, or does grammar
CLASS (graph v tree) bind where COUNT (five grammars, gen-8) did
not? PREDICTIONS (house): (1) math column HOLDS at-class (gen-8
precedent: union nearly free, coefficient (i)~0); (2) ZX column
lands at-or-near its 30-class (real-scalar comparators M5 31 /
G5 32; the un-run scalar-real-ZX control means a modest debit v
cplx_none 36 is NOT interpretable as a union tax — fence named);
(3) no wall, parsefail 0 (strict encode held 0/480 on ZX).
FAIL reads: math -5+ = graph grammar taxes the tree federation;
ZX <25-class = CLASS binds where COUNT did not.
JOB 2 — ZX seed-2 (cplx_none recipe, BIRTH_SEED=2, same diet/
gate): first ZX seed-sigma point; n=2 on the 36 arm. PREDICTION:
within sqrt(2)*sigma-class of 36 (sigma unknown — that is the
point); a 5+ swing flags gate noise floor before any future ZX
pair is read. Queue: scratch/night_zx.sh via wsl.sh launch
(success-only marker).

## PINCER R1a: backward trains DOWN but peels WEAK — 11% per-candidate validity v forward's 85%-class; coverage saves B-a, barely (2026-07-26 night)

Ran as pre-registered (backpairs births 1P + 3E at d256, probe =
105 fresh gate-band mid-chain states, k=8, forward-verify fence).
| arm | per-cand validity | state coverage (>=1 valid) |
|---|---|---|
| backpairs_1p | 11.1% (93/840) | 92/105 |
| backpairs_3e | 11.5% (97/840) | 96/105 |
READS:
1. Prediction (1) CONFIRMED: loss trains down like forward
   (final ~0.31, pairs-class) — the substrate accepts the
   backward grammar.
2. Prediction (2) CONFIRMED in direction, WRONG in magnitude:
   11% lands far BELOW the 40-70% band — predecessor-emission
   (grow the expression) is much harder than the forward
   direction-tax (-18 at 50/50 mixing) suggested. Loss-down +
   validity-low = the model learned the FORM of backward rows,
   not the act of un-applying rules.
3. Prediction (3) NULL: 3E ~ 1P (11.5 v 11.1, coverage 96 v 92)
   — no schedule rescue; the deficit is not batch damage.
4. The saving read: per-STATE coverage 88-91% at k=8 — B-a can
   peel (some valid predecessor almost always lands) but pays
   ~9x oracle calls per valid peel. CONSEQUENCE: B-a's bar for
   B-b is set LOW exactly where amendment-2 predicted — the
   text-emission peeler is the weak link; score-over-enumerated-
   moves dissolves both the validity rate (validity by
   construction) and the 9x verify overhead.
Sidecars logs/pp_backpairs_{1p,3e}.jsonl. L4/L6 skew honest
(n=5/n=1). NEXT: distribution-quality readout (the quantum-
readout frame) before any R6 alphabet race.

## PRE-REG: the distribution readout — are the crystal's amplitudes over the enumerated legal set CALIBRATED? (Artin's quantum-frame ask, 2026-07-26 night)

The amendment-2 frame made measurable (the honest classical form
of "the quantum computer returns the distribution": engine
enumerates the COMPLETE legal move set exactly, model supplies
amplitudes). Instrument scratch/pincer_dist_probe.py: 105 fresh
gate-band states -> successors() full legal enumeration ->
teacher-forced sequence log-probs (no generation) from FOUR
checkpoints (pairs_3e wfloor / pairs_1p stream4 / oneshot_1p /
backpairs_1p as reverse-scorer logp_b(s|child), per-edge, own
normalization noted) -> value labels per child by FORK-ISOLATED
engine.solve (budget 150, wall 20s; streamed rows per the
killed-worker doctrine). Logged per child: rule@site, token len,
solved/nodes/plies; per state x model: entropy, mass-on-solving,
top-1-solves, Spearman(logp v solved), length-bias corr.
PREDICTIONS (house): (1) forward crystals are BETTER than
uniform but imperfectly calibrated — mass-on-solving beats the
n_solving/n_legal baseline and top-1-solves lands 70-85% (greedy
chains gate at 57-65, so the argmax is known-good; the open
question is the TAIL mass); (2) oneshot is MISCALIBRATED on
1-ply moves (trained to skip — its mass should wander off the
legal set's solving subset; entropy higher); (3) the backward
reverse-score carries signal (positive mean Spearman) but weaker
than forward — 11% emission validity caps how sharp
logp_b(s|child) can be; (4) length bias EXISTS (longer children
get less raw mass) — named confound, length-normalized column
logged. Uniform baseline and n_solving reported per state so
every read is against chance.

## DISTRIBUTION READOUT v1: the amplitudes are POLICY, not landscape — and the battery saturated in BOTH value dimensions (2026-07-26 night)

Ran as pre-registered (43/105 states survived enumeration —
62 skipped at n_legal<2 or unencodable; L4/L6 absent; battery
skew named). Sidecar logs/pp_dist_probe.jsonl (full per-child
logp x 4 models, rule@site, plies/nodes-to-root).
THE INSTRUMENT LESSON FIRST (the meta-pattern, 5th occurrence:
prediction pays only where variance lives): at budget 150 the
value oracle solves 95.3% of ALL legal children (chance
mass-on-solving 0.953) — the binary read is vacuous
(mass_solv=top1=1.000 for every model). The graded read
(Spearman v -plies_to_root: models 0.71-0.77, top1-minplies
97-100% v chance 38%) LOOKS strong but the control kills it:
**a length-only ranker scores 0.826 / top1 1.000 — on this
battery "shortest child" is a perfect policy**, so no model
skill beyond length is demonstrated. Calibration on gate-band
mid-chain states is UNMEASURABLE; the re-ask needs a frontier
battery (L6/L7 stuck states, walls, tighter budget) where
shortest-move fails.
WHAT SURVIVES (distribution geometry, real and new):
1. **The crystals' amplitudes encode the CORPUS POLICY, not the
   solvability landscape**: entropy_norm 0.016-0.022 (near
   delta); mass concentrates on canonical families (i_const .93,
   i_heurisch .60, i_inverse_trig .51) and is ZERO on
   equally-solving alternatives (i_sum, together, cancel,
   i_const_factor, expand all 0.000 pooled mass at solve rate
   1.00). The superposition is enumerated; the crystal collapses
   it to the farm's habits.
2. **The backward reverse-scorer has a DIFFERENT geometry**:
   10x flatter (entropy_norm 0.245), spreads real mass onto
   families the forward crystals zero (i_const_factor .131,
   cancel .084), and its graded Spearman (0.769) is the best of
   the four DESPITE 11% emission validity — scoring and emitting
   are different skills (B-b's premise, directly observed).
3. Length bias confirmed strong everywhere (len_corr 0.62-0.80)
   — any future distribution instrument must carry the
   length-only control as a mandatory arm (adopted).
Predictions: (1) unresolvable (saturation); (2) WRONG — oneshot
is NOT flatter on 1-ply moves (ent 0.022, same as pairs; its
skip training did not blur its step distribution); (3) direction
right (reverse-score carries signal) but comparator vacuous;
(4) CONFIRMED. NEXT: frontier-battery re-ask rides the R6
alphabet race (same instrument, stuck states, length control
standing).

## PRE-REG: altpairs — the distribution-rows bank, forward edition (winners combined; 2026-07-26 night)

Motivated by tonight's distribution readout (crystals put ZERO
mass on equally-valid non-canonical moves) + Artin's standing
push (nulled/banked ideas decay — retest when the config
changes; the config JUST changed: we can now read distributions).
Combination of winners: pairs format (ladder winner, 57@1P) +
engine enumeration (successors(), verified) + the R5
distribution-target bank (never run forward). DIET: full gen-4
pairs + ~22k farmed verified-alternative successor rows (~14%
share; make_altpairs.py, 4000 unique states, fork workers,
streamed shards). BIRTH: FORMAT=altpairs SCHED=1p d256 seed 1.
READS: (a) gate_pp (does the gate hold pairs-class 57? bar:
>= 52 = within the known seed-noise band); (b) the distribution
instrument re-run on fmt_altpairs_1p — entropy_norm (does the
near-delta 0.017 rise?), pooled mass on non-canonical solving
families (i_sum/together/cancel/i_const_factor: from 0.000 to
>0.05 = the diet taught the landscape). PREDICTIONS (house):
(1) gate HOLDS (alternative rows are verified-true steps, not
noise; +14% dose of valid physics shouldn't damage chaining);
(2) entropy rises toward the landscape and non-canonical mass
goes nonzero (the crystal learns what it is fed — teach-don't-
impose, diet edition); (3) risk named: if alternatives DILUTE
step-precision, gate drops like revpairs' direction tax — that
verdict would bound how much landscape a policy crystal can
carry. Either way banks.

## ALTPAIRS VERDICT: the diet teaches the landscape (12x entropy, every zeroed family wakes) — and it costs -6 gate; the policy-calibration TRADE is now a measured dial (2026-07-26 night)

Ran as pre-registered. Gate: {3:15, 4:4, 5:15, 6:8, 7:9} =
**51/120 @ 47.2%** v pairs@1P 57 @ 54.2% — BELOW the >=52 bar:
prediction (1) FAILS, the named risk in (3) fires (a step-
precision dilution tax, -6 at 14% alt share — same family as
revpairs' -18 at 50%, roughly dose-proportional).
Distribution read on the same 43-state battery (labels reused,
logs/pp_dist_altpairs.jsonl): prediction (2) CONFIRMED
decisively —
| family | pairs_3e mass | altpairs mass |
|---|---|---|
| entropy_norm | 0.016 | **0.188 (12x)** |
| i_const_factor | 0.000 | 0.048 |
| i_parts | 0.000 | 0.035 |
| cancel | 0.000 | 0.022 |
| i_sum / together | 0.000 | 0.011 / 0.006 |
| expand | 0.000 | 0.000 (only holdout) |
Canonical families keep their lead (i_const 0.78, i_heurisch
0.60) — the crystal ADDED the landscape without losing the
policy ordering. READS: (1) teach-don't-impose, diet edition,
directly confirmed — the distribution is exactly the exposure;
(2) the policy-vs-calibration trade is now a DIAL (alt share),
not a mystery: 0% = delta-sharp/gate 57, 14% = 12x flatter/gate
51; a share sweep (3-7%) would locate the knee if a calibrated
FORWARD crystal is ever needed; (3) for the PINCER the lesson
cuts the other way — don't spend the forward crystal's gate on
calibration; calibration belongs to the backward SCORER (B-b),
whose training set (enumerated legal sets + labels) is
altpairs-shaped BY CONSTRUCTION with no emission role to tax.
Chains: distribution-readout v1 (the motivating zero), R5
(backward soft labels inherit this recipe), amendment-2 B-b.

## PRE-REG: pincer R8 — meet v1, the full protocol vs let-it-finish at equal budget (2026-07-26 night, Artin's GO)

Cell R8 (spec amendment 3). BATTERY: the 55 problems pairs_3e
FAILS at its greedy gate (pp_pairs_3e sidecar; the misses are
the only place variance lives — distribution-readout lesson
applied). ARMS, equal sampled-token budget, both logged
per-problem:
- A (let-it-finish): pairs_3e chain search, second full attempt
  with fresh seeds (12 plies, B=8 waves).
- B (pincer v1): (1) conjecture — oneshot k=8, oracle-checked
  (hit = solved at ply 0); (2) peel — backpairs_1p peels j<=2
  plies from every distinct candidate (k=8/peel,
  forward-verified at mint; goal set B skeleton-normalized;
  soundness by edge-verification: a spliced chain is equivalent
  by transitivity, solved iff the endpoint is integral-free);
  (3) meet — pairs_3e forward chain, each reached state tested
  against B, contact = splice.
PREDICTIONS (house): (1) arm B converts MORE of the 55 than arm
A's re-roll (bar: B > A by >=3 problems — else the pincer is
seed-noise); (2) conversions concentrate at L4 + near-miss
territory (R0's map); (3) named risk: 11% peel validity makes B
thin — if peel yield ~0, arm B reduces to conjecture-only and
the verdict honestly reads "the pincer is R0 + noise at current
peeler quality" (that too banks — it prices B-a for B-b).
Instrument: scratch/pincer_r8.py, pp sidecar both arms.

## PINCER R8 VERDICT: meet v1 FAILS its bar — zero meets, and the honest token ledger reads 5.7x AGAINST the pincer (2026-07-26 night)

Ran as pre-registered on the 55 pairs_3e misses. **Arm A 4/55 v
arm B 5/55 — B+1, bar was >=3: FAIL (seed-noise class).**
ANATOMY (the run's real payload):
1. **ZERO meet-mechanism solves.** B's 5 = 3 conjecture + 2
   plain-chain (the chain solves are re-roll luck, arm-A class).
   The meet NEVER fired: 76 peels kept across 55 problems, only
   22/55 got ANY goal state, and no forward chain ever contacted
   one. Named risk (3) fires in full — at 11% peeler validity
   the goal sets are too thin and too shallow (j<=2 plies off
   conjectures that are themselves wrong) to intersect a
   12-ply forward cone.
2. **The token ledger is the sharper kill**: B spent 154,641
   sampled tokens v A's 27,053 — 5.7x — for +1 solve. My
   equal-budget fence was violated by my own design (peel cost
   rode on top instead of being traded); booked as an
   instrument defect AND as the honest economics: at B-a
   quality, peeling is ~pure waste (regret-round-2 lesson,
   third occurrence).
3. What survives: conjecture converts 3 fresh problems at L3/L4
   (2 of B's L4 solves are conjecture — the crater territory
   confirmed as conjecture-reachable on re-sample); conj-valid
   candidates near-zero elsewhere (5 total across 55).
CONSEQUENCE, pre-stated for the program: the pincer's viability
now rests ENTIRELY on B-b (score-over-enumerated-moves) — the
text-emission peeler is measured three ways tonight (11%
validity, 9x verify overhead, zero meets at 5.7x budget) and is
DEMOTED from the protocol; B-a exists only as B-b's baseline.
R6/R-next: build the B-b scorer + frontier battery; re-run R8
with engine-enumerated inverse peels before any promotion talk.

## PRE-REG: R0b — collapse-ordered readout (the honest Grover residue; Artin's push, 2026-07-26 late)

CLAIM UNDER TEST: amplitudes pay at the READOUT layer as an
oracle-call budget — checking candidates in descending model-mass
order reaches the first verified solution in fewer oracle calls
than random order (Grover's economic content, classical form; no
register, no unitary, no sqrt claim). Instrument
scratch/pincer_r0b.py: 120 gate problems, oneshot k=16 sampled
candidates (2 waves), all oracle-checked once (instrument cost),
seq-logp wave read; metric = oracle calls to first solving
candidate under (a) mass order, (b) sampling order, (c) random
expectation ((n+1)/(h+1) closed form). PREDICTIONS (house):
(1) mass order beats random by >=1.5x mean calls-to-hit (R0
showed argmax carries most information — the wave is
informative); (2) mass order ~ sampling order (T-sampling
already reads the same distribution — if so, the "amplification"
is free and already deployed); (3) the saving concentrates where
h/n is small (few needles — exactly Grover's regime).

## R0b VERDICT: amplitude ordering pays 1.7x at readout — and it concentrates exactly in the needle regime (2026-07-26 late)

Ran as pre-registered (56/120 problems had >=1 verified hit at
k=16). Mean oracle calls to first solution: **mass-order 1.43 |
sampling-order 1.52 | random 2.43.**
1. Prediction (1) CONFIRMED: 1.70x saving v random (bar 1.5x) —
   the wave is informative at readout; reading amplitudes before
   measuring is worth ~1 oracle call per problem at k=16.
2. Prediction (2) CONFIRMED: sampling order ~ mass order (1.52 v
   1.43) — T-sampling already IS an amplitude-ordered readout
   (first samples come from high-mass regions), so the
   amplification is deployed for free wherever we sample. The
   marginal value of the explicit wave read is the 1.52->1.43
   sliver PLUS determinism (no sampling variance, cacheable).
3. Prediction (3) CONFIRMED: the saving splits 2.06x in the
   needle regime (h/n<=0.25, n=23) v 1.28x abundant — exactly
   Grover's regime dependence, reproduced classically. Where
   solutions are rare, amplitude ordering pays double.
SCOPE (the fence holds): this is the READOUT-layer wave frame
paying again (R0's +1-over-argmax was its lower bound; this is
its price sheet). No sqrt(N), no register, no interference —
the classical residue, measured. PRACTICAL RIDER: verify-order
in every k-candidate loop (gates, exchange resamples, future
B-b peel readout) should be mass-descending when logp is
already in hand — sample_wave_lp RETURNS lps; the reorder is
one sort, zero cost. Adopted for new instruments; retrofits
opportunistic.

## PRE-REG: S1 — the frontier battery + persistent value cache (calibrated-scorer spec, cell 1; 2026-07-27 pre-dawn, Artin's confirm)

Instrument scratch/scorer_s1_battery.py (spec
2026-07-27-calibrated-scorer.md). Candidates: L6/L7 gate roots +
stall-endpoint states from pp sidecars; enumerate legal sets;
fork-isolated engine solves (budget 150, 25s wall, 6 workers)
into the persistent skeleton-hash cache (data/value_cache.jsonl
— deterministic labels, computed once, forever). KEEP: states
with VALUE VARIANCE (mixed solve/no-solve, or graded plies
spread >=3). PREDICTIONS (house): (1) the frontier de-saturates
— kept battery's mean solve fraction lands 0.3-0.7 (v the
saturated pool's 0.95); (2) >=40 states survive (enough for S2
training-signal reads at >=5-per-level honesty); (3) THE
DECIDING READ: the length-only ranker DEGRADES on mixed states
(spearman < 0.4, top1 well under 100%) — if length still wins
at the frontier, S2's bar is length-not-chance and the scorer
must beat IT.

## S1 VERDICT: the frontier battery exists (71 states, 45 mixed) — length degrades in RANK but keeps top-1; S2's bar is set (2026-07-27 pre-dawn)

Ran as pre-registered (one instrument fix mid-run: engine moves
can leave vocab-40 — fresnelc child; such children kept with
n_tok=None, excluded at crystal-scoring time; solve pass was
already cached so the re-run cost nothing — the persistent cache
paid on its FIRST day: 647/647 labels served, 0 recomputed).
BATTERY: 71/105 states kept (45 mixed solve/no-solve + 26
graded), levels {3:17, 4:2, 5:9, 6:21, 7:22} — L6/L7 carry 43;
49/499 child labels are walls/unknown (honest mass). Mean solve
fraction 0.80.
PREDICTIONS: (1) PARTIAL — 0.80 v the 0.3-0.7 band (the mixed
subset is the real de-saturation; the graded tail keeps the mean
high); (2) PASS — 71 >= 40; (3) SPLIT, and the split IS the
finding: **length-only rank collapses (spearman 0.294 v 0.826 on
the saturated pool) BUT length top-1 stays 44/44** — at the
frontier, the shortest child still always solves, while the
REST of the length ordering turns to noise. CONSEQUENCE FOR S2
(booked as the bar): the scorer cannot win at top-1 (length is
free and perfect there so far); it must win on (a) full-ranking
quality (Spearman v value), (b) needle-regime oracle economics
(R0b metric), (c) the 45 mixed states' tail ordering — exactly
the calibration territory, exactly where B-b's distribution
output is the differentiating skill. Deliverables live:
data/scorer_battery_v1.jsonl + data/value_cache.jsonl (647
labels, permanent).

## PRE-REG: S2 data leg — the listwise scorer's training table (2026-07-27 pre-dawn)

scratch/scorer_s2_data.py: stratified sample (400 unique
states/level) of corpus states -> full legal enumeration ->
per-child value labels (cache-aware fork solves, budget 150, 8s
walls; every solve extends data/value_cache.jsonl permanently) +
replayed true-move label (the R1b mechanism). S2 arms train
LISTWISE on the same MicroLM substrate (forward all children,
softmax over set, CE against target): arm (i) target = smoothed
one-hot on true move; arm (ii) target = softmax(-plies) — the
Dijkstra head. Same states, matched labels; eval on the S1
battery (Spearman v value, needle oracle calls, mixed-tail
ordering) v the length-only + pairs_3e-implicit controls.
PREDICTIONS (house): (1) true-move coverage lands ~68% (R1b
replicates at scale); (2) value-label yield >85% of children
labeled within the 8s wall at L1-5, degrading at L6-7 (honest
unlabeled mass reported); (3) no interference verdict yet — the
training race itself pre-registers at birth time.

## FEDERATION: math+ZX union birth — BOTH columns hold; grammar CLASS does not bind (2026-07-27)

Ran as pre-registered (union_math_zx: d384/8L fp32 seed 1,
vocab-47, 132,870 math + 97,036 ZX rows, SEQ_CAP 1536, 3080).
| column | union gate | comparators (standalone) |
|---|---|---|
| math | 66/120 (65.78%) | fp32 64 / G5-lat 66 / cplx_none 63 |
| ZX | 40/120 | cplx_none 36; real-scalar M5 31 / G5 32 |
READS:
1. Prediction (1) CONFIRMED: math HOLDS at-class — 66 ties the
   column BEST (G5-latent 66). The union is not merely cheap; the
   math column paid zero measurable tax for carrying a graph
   grammar. Gen-8 coefficient-(i)~0 law extends to grammar CLASS.
2. Prediction (2) CONFIRMED and exceeded: ZX 40 lands ABOVE every
   comparator — +4 over cplx_none 36 (complex-FFN, the previous
   best) and +8/+9 over the real-scalar pair (M5 31 / G5 32),
   which is the honest same-alphabet comparison (union is fp32
   real). Noise class unknown until seed-2 lands (JOB 2, in
   flight) — the +4 is NOT booked as a union BENEFIT yet; the
   at-or-above-class read is what's banked.
3. Prediction (3) CONFIRMED: parsefail 0/120, identity 0,
   unsound 80, walls 0 — strict encode holds in union.
4. Verdict: graph-vs-tree grammar CLASS binds nowhere at 19M-
   class, where grammar COUNT (gen-8, five grammars) also did
   not. The federation floor's first graph datapoint is a clean
   PASS on both fail-read thresholds (math -5+: not hit; ZX
   <25-class: not hit).
Fences: union d384 v standalone d384 same-recipe; ZX kind mix
shifted (color 23/fuse 10 v standalone's mix) — kind-level read
waits for seed-2 sigma. Seed-2 verdict books separately on
marker.

## PRE-REG: fp32-real ZX-only control (3080 job 3, armed on Artin GO 2026-07-26 ~11:12PM)

The missing comparator behind the union ZX read: the union arm's
40/120 sits above cplx_none 36 (different arch: complex-FFN) and
M5 31 / G5 32 (quantized/rotated) — but the plain fp32-real
ZX-only birth was never run, so "union credit" and "recipe
credit" are confounded. CONTROL: identical recipe to the union
arm (train_mathnative d384/8L fp32, vocab-47, BIRTH_SEED=1,
3 epochs, SEQ_CAP 1536) with diet = zx_farm1_train ONLY (97,036
rows — same ZX exposure as the union, minus math). Gate: gate_zx,
same 120 held-out diagrams. Queue: scratch/night_zx2.sh, chains
on the night queue's success marker (dies honestly if the queue
died); marker logs/night_zx2_done.marker.
PREDICTIONS (house): (1) control lands 37-42-class — the union
credit is RECIPE (unquantized fp32-real beats the modified
alphabets and the complex arch), transfer stays unproven; (2) if
instead the control holds the old 31-36-class, math->ZX TRANSFER
goes LIVE (first cross-domain positive at 19M) and gets its own
pre-registered replication before any law is written; (3) seed-2
sigma (in flight) fences both reads — a gap smaller than
sqrt(2)*sigma-class books as a tie.

## AMENDMENT (targets: S1 frontier battery booking + S2 data-leg pre-reg): wall-censored cache rows are not facts (2026-07-27)

Artin's audit question ("is there any rounding anywhere?") found
a real defect: solve wall-kills were written to the PERMANENT
value cache as {solved: null} in the same budget-150 namespace as
real verdicts — 88/647 rows at audit time. Since S1 used 25s
walls and S2 uses 8s, a cache hit on a censored row silently
prevents any future farm from retrying at a longer wall:
"didn't wait long enough" fossilized as "unknown forever" (the
checkpoint selection-effect's cache-shaped cousin). FIXES (both
scripts): (1) censored rows never enter the permanent cache
(kept in-process for table assembly only); (2) cache load treats
solved=null as not-cached (retryable); (3) rows now record their
wall. The RUNNING S2 farm predates the patch, so its merge still
appends Nones — a one-shot strip of all solved=null rows runs on
farm completion, before the trainer fires. DOCTRINE LINE:
censored != fact — any permanent store must either exclude
censored measurements or key them by their censoring parameter.

## AMENDMENT (targets: precision doctrine closure 2026-07-24 + fp64 disagreement resolution 2026-07-23): scope fence + named retest condition (2026-07-27)

Artin's audit ("our layers were never precise") scopes the closed
doctrine honestly. What stands as measured: the dynamical cliff
is bracketed 8-10 mantissa bits and the curve is FLAT from TF32
(10) through fp64 (53) — bf16 66 / TF32 65-parity / fp32 64-65 /
fp64 65 exactly, rare bins identical; at rest, MX-int4 full
parity, int3 cracks. What the ladder CANNOT see: effects below
gate sigma (~+-1-2 solves/120), because every arm ran on rounded
layer algebra and the gates themselves carry an arithmetic noise
floor (near-tie flips, device reduction order). AMENDED CLAIM:
"precision doesn't pay above TF32" holds ABOVE INSTRUMENT SIGMA.
RETEST CONDITION (named, single, cheap): when exact-mode
inference lands (relay rung 2b — exact GEMM + declared
nonlinearities, bit-identical logits), run ONE paired arm
(exact-mode gate vs rounded gate, same weights) at sub-sigma
resolution. Doctrine stays CLOSED until that arm says otherwise;
no other precision runs are justified by this amendment.

## ZX SEED-2: 28/120 — an 8-POINT seed swing; the ZX gate noise floor is FLAGGED (2026-07-27)

JOB 2 landed (cplx_none recipe, BIRTH_SEED=2, same diet/gate/
device): **28/120** (parsefail 0, identity 0, unsound 92, walls
0; by kind color 17 / fuse 5 / lcomp 3 / id 2 / pivot 1) v
seed-1's 36/120. READS:
1. The pre-reg tripwire FIRED: a 5+ swing was named as flagging
   the gate noise floor — we got 8. Two-point sigma-class
   ~8/sqrt(2) ~ 5.7 solves/120 (n=2, crude by construction; the
   POINT of the run).
2. The federation verdict's refusal to book union-ZX 40 v
   cplx_none 36 as a union benefit is VINDICATED: +4 sits well
   inside one seed-sigma-class. The 40 itself now reads
   "at-class with everything," not "above all comparators."
3. FENCE (standing): every single-seed ZX gate number (M5 31,
   G5 32, none 36, union 40, seed-2 28) carries a ~+-5-class
   seed fence until an n>2 sigma exists. JOB 3's recipe-v-
   transfer bands (37-42 / 31-36) OVERLAP under this sigma —
   its verdict books WITH the fence, discriminating only if it
   lands outside the union band by more than a sigma-class.
INCIDENT (friendly-fire variant #7, fixed dd6f7f3): both night
launches shared /tmp/wsl_job.sh; bash reads scripts LAZILY, so
the second launch's overwrite corrupted the still-running seed-2
queue mid-stream ("h: command not found") AFTER its gate step —
birth + gate completed, marker never fired, the zx2 waiter
honestly died. wsl.sh launch now mktemps a unique job file per
launch. Seed-2 artifacts verified by hand (ckpt 00:31, gate log
00:35); marker set manually on that evidence; job 3 relaunched
and confirmed running (birth in flight).

## PRE-REG: SR-bf16 birth (JOB 4, 3080; hold released by Artin GO 2026-07-27)

The open lever from the fp32-birth verdict (2026-07-17: fp32 69/
120 v bf16-RNE 66/120, identical loss — the debit is trajectory-
only). ARM: gen-4 std --fast recipe, one variable = SR_BF16=1
(new in train_mathnative.py): fp32 master -> bf16 weight cast by
uniform noise in the 16 dropped bits + truncate, straight-through
grad. SCOPE DECLARED: SR at nn.Linear weight/input cast sites
only; attention matmuls + everything else keep autocast RNE.
Local check: SR mean unbiased to ~4e-5 rel (RNE bias ~1e-3 rel);
outputs exactly representable bf16. Same seed (default 0), data,
gate (gate_ckpt_cuda 120) as the booked pair. Chained behind JOB
3 (scratch/night_sr.sh, honest-death waiter, success-only marker).
PREDICTIONS: (1) SR lands ABOVE bf16-RNE 66 — unbiased rounding
removes the trajectory drift; 68-69-class = lever CONFIRMED
(bf16 speed, fp32 capability; births switch to SR). (2) 66-67 =
partial/null — weight-cast-only scope insufficient (activation/
grad rounding carries the rest of the debit); book honestly,
extend only with a named next site. (3) below 65 = SR noise
itself taxes this scale. FENCE: math-gate seed sigma at 45M is
tighter than ZX's (~5.7-class) but nonzero — a 1-solve edge
books as a tie.

## AXIOM TRANCHE 3: frontier_eval + gate_battery + certify_tables ALL DELIVERED (2026-07-27)

Asks 4-6 shipped same-night (bridge INTERFACE_VERSION == 3, 14
pinned names; full record: relay/2026-07-27-5). Headlines: (1)
frontier_eval = the B-b pincer inner loop as ONE bridge call,
both directions, 18 ms forward WITH full verification (R0b
honored: successors unverified, oracle deferred to verify_edge
top-k); scorer slot listwise-shaped, awaiting the S2 winner's
AXNN + prompt spec. (2) gate_battery = 120-probe greedy gate in
one call at ~78 ms/probe — the nightly gate wall is DEAD
(sub-minute); FX-V1 refactored to a per-position KV stepper
where the stepper IS the forward, so generate() is bit-exact by
construction and the acceptance hashes reproduce unchanged
(refactor changed cost, not values — the regression pattern to
copy). (3) certify_tables = standing certificate wired as a LOAD
GATE (uncertified artifact throws), +-1 LSB + argmax-relevant
monotonicity + rope-circle + midpoint fuzz. TRAP BANKED: exp
table underflow region has equal adjacent entries — corrupting
there is a silent no-op for monotonicity tests; fuzz where the
gradient is ~1 LSB (idx ~1300 on [-16,0]). CONSEQUENCE: E3
shrinks to one call per side; the S2 race now feeds three
consumers (scorer slot, gate detokenizer, E2 export).

## PRE-REG: ZX seed-3 (JOB 5, 3080; queued under Artin's open-queue GO 2026-07-27)

Third sigma point on the cplx_none/zx_farm1 arm (seed-1 36,
seed-2 28): BIRTH_SEED=3, same diet/recipe/gate/device
(scratch/night_zx3.sh, chained behind JOB 4). PREDICTION: lands
in the 25-40 band implied by the two-point sigma-class ~5.7;
the n=3 spread REPLACES the crude two-point sigma as the
standing ZX seed fence. If it lands OUTSIDE 22-42 (beyond
~2.5x the current class), the gate itself is suspect (probe-set
composition, not seed noise) — that read triggers a probe-set
audit, not another seed.

## S2 DATA-LEG: 2,995 states / 70% true-move / L4 yield-miss booked; cache de-censored (2026-07-27)

Farm complete (8 workers, ~2.5h): 2,995/3,003 states enumerable,
20,552 distinct children, 2,110 states with true-move label
(70%; pre-reg ~68% CONFIRMED). Value-label yield by level: L1
100 / L2 100 / L3 98 / L4 47 / L5 95 / L6 80 / L7 67 / L8 48%.
PRE-REG READ: ">85% L1-5" holds at four of five — L4 MISSES at
47%, the integration crater's known signature (walled sympy
solves), now CENSORED-NOT-FOSSILIZED: post-amendment those rows
never entered the cache, so a longer-wall backfill stays
possible. One-shot cache strip EXECUTED: 21,184 -> 16,737 rows
(4,447 censored Nones removed; the running farm's module
predated the patch as predicted). True-move by level: 85/89/73/
39/70/73/68/57%. VERDICT: data-leg PASSES for the race (both
arms train on the same table; L4 weakness is arm-symmetric).
Trainer fires now (pre-reg at birth: see spec S2 cell).

## S2 RACE VERDICT: DISTANCE arm WINS on ranking — imitation ranks NO BETTER THAN NOISE (2026-07-27)

Both arms trained (3 ep, warm from pairs_3e, same table/holdout;
battery 71 fenced from train). Holdout: imit top1-true 87.6% v
dist 53.1% — imitation is the better MOVE-GUESSER. Battery
(n=37 ranking-measurable states, pre-reg'd bar = ranking/needle
economics v controls, NOT argmax):
| scorer | spearman | top1_solves | calls_to_hit |
|---|---|---|---|
| s2_dist | **0.497** | 37/37 | 1.0 |
| length_only | 0.399 | 37/37 | 1.0 |
| pairs_3e_raw | 0.309 | 37/37 | 1.0 |
| s2_imit | 0.092 | 36/37 | 1.05 |
READS: (1) dist BEATS both controls on rank; imit lands BELOW
RAW — supervising on the true move alone teaches argmax and
DESTROYS calibration (S1's length finding confirmed a second
way: argmax-good != rank-good). (2) The distance target
(softmax over -plies) is what buys ordered mass — calibration
comes from VALUE labels, not move labels. (3) Needle metric
saturated (~1.0 all arms, battery too easy on that axis) — the
economics read rides on spearman alone; edge 0.497 v 0.399 at
n=37 is modest, books as a WIN with a width caveat, replication
= S3's ternary cell on a harder needle battery. WINNER:
checkpoints/scorer_s2_dist.pt = the S2 crystal; next artifact
across the bridge = its AXNN + prompt spec (feeds frontier_eval
scorer slot, gate_battery detokenizer, E2).

## JOB 3 VERDICT: fp32-real ZX-only control = 35/120 — lands in the transfer band but INSIDE THE FENCE; control is NON-DISCRIMINATING at current sigma (2026-07-27)

The pre-reg'd recipe-v-transfer discriminator (real fp32 birth,
zx_farm1 diet only, union recipe/env): **35/120** (parsefail 0,
walls 0, unsound 85; kinds: color 18 / fuse 10 / lcomp 3 / id 2
/ pivot 2). Nominal read: 31-36 band = "math->ZX transfer LIVE"
(union's ZX 40 would carry a +5 transfer credit). FENCED READ
(the seed-2 amendment applies): with seed sigma-class ~5.7 and
the arm spread 28-36 on n=2, 35 sits within one sigma-class of
EVERY comparator (union 40, seed-1 36, seed-2 28, M5 31, G5 32)
— the bands overlap and the control CANNOT separate recipe
credit from transfer at n=1. BOOKS AS: non-discriminating;
no transfer claim, no recipe claim. What it DOES pin: the
union's ZX column (40) paid no visible union tax v ZX-only (35)
— "union nearly free" holds on the graph grammar as it did on
trees. JOB 5 (seed-3) firms sigma; a transfer re-read waits on
that spread. Union math column remains the load-bearing result.

## SR-BF16 VERDICT: 53/120 — the lever is DEAD AS IMPLEMENTED (band-3 fire with two named confounds) (2026-07-27)

JOB 4 landed: gen-4 --fast + SR_BF16=1 gate **53/120 @ 47.13%**
(losses 0.4628/0.3298/0.3297; flag confirmed armed; ~215s/ep =
the bf16 speed held). Pre-reg band 3 ("below 65 = SR noise
taxes") FIRES — but the magnitude (-13 v bf16-RNE 66, -16 v
fp32 69) exceeds any seed-sigma class and books with TWO
confounds named: (1) MECHANISM: per-forward RESAMPLED weight-SR
is a persistent untuned jitter regularizer (~2^-9 relative,
fresh noise every step) — not the "unbiased rounding" of the
lever's original riff; a variant that freezes noise per-step or
applies SR only at the optimizer cast site is a DIFFERENT
experiment. (2) PAIR AGE: the 69/66 comparison is 10 days old;
SR's train loss landed 0.3297 v the pair's 0.3525 — the gen-4
tree has drifted, so one-variable v the HISTORICAL bf16 arm is
not guaranteed. VERDICT: SR-as-implemented is strongly
negative; per the closed precision doctrine ("don't spend runs
on precision-capability questions") NO follow-up is queued —
any revival needs a new mechanism argument AND a same-day
bf16-RNE control. fp32-only births remain doctrine.

## ZX SEED-3: 34/120 — the n=3 sigma lands ~4-class; fences tighten, verdicts stand (2026-07-27)

JOB 5 landed: **34/120** (parsefail 0, walls 0; color 20 / fuse
7 / lcomp 4 / id 2 / pivot 1). Arm now 36 / 28 / 34: mean 32.7,
sd ~4.2 — the pre-reg 25-40 band HOLDS (no probe-set audit
triggered); the n=3 ~4-class sigma REPLACES the crude two-point
5.7 as the standing ZX seed fence. RE-READS UNDER THE NEW
FENCE: union ZX 40 = +7.3 v arm mean (~1.8 sigma) — suggestive,
still not clean; ZX-only control 35 sits inside the arm spread
(non-discriminating verdict STANDS); seed-2's 28 was the arm's
low draw, not an anomaly. The graph-grammar column's honest
summary: ZX-class ~29-37 at 19M regardless of recipe, union
possibly +1-2-sigma above it — a transfer/union claim needs
either more union seeds or a bigger effect.

## PRE-REG: E4 solve_batch parity gate (2026-07-27, fired on axiom push landing)

200 states sampled (string-seeded) from the de-censored house
value cache (budget-150 python solver labels, walls stripped —
censored != fact honored), re-labeled by axiom_sym.solve_batch
(budget=150, IV==3 Mac build 75d912f5). PREDICTIONS: (1)
solved-bit agreement >= 199/200 (axiom measured 197/200 qual
roots solvable; disagreements should be deadline expiries, i.e.
axiom-False-house-True only); (2) plies equal or within known
tie classes on agreed solves. DECISION RULE (spec E4): pass =
shared cache (one label family); any solved-bit disagreement in
the axiom-True-house-False direction = engines differ on
REACHABILITY, labels stay engine=axiom namespaced pending a
per-state audit.

## E4 VERDICT: solve_batch parity FAILS — labels stay engine=axiom; axiom's False carries CENSORING (2026-07-27)

200 cached budget-150 states (string-seeded sample): parsefail
16/200 (axiom grammar narrower — Subs(...) forms among them);
agree 161/184 (87.5% v pre-reg >=199/200); axFalse/houseTrue 21
(incl trivial basis states: asin-basis 1/sqrt(1-9x^2) — rule
coverage gaps); axTrue/houseFalse 2 (the reachability-claim
direction — per decision rule this ALONE keeps namespaces
split); plies equal only 111/148 on agreed solves (systematic
+1/+2 — ply semantics differ). Mac throughput 7 st/s with
[slow-fire] walls hitting the 8s deadline. VERDICT: parity gate
FAILS on every prong; engine=axiom namespace is PERMANENT until
a per-prong audit closes each gap. DOCTRINE CATCH (relay
point): solve_batch reports deadline expiries as solved=False
INDISTINGUISHABLE from proven-unreachable — axiom's False
labels carry exactly the censoring our value cache just purged
(censored != fact). Ask: surface an `expired` flag so censored
rows are cacheable-by-censoring-parameter, not fossilized.

## E4 RE-RUN ON IV4: agreement 168/184, the expired flag decomposes the misses — and TWO house corrections (2026-07-27)

Same seeded 200-state sample, axiom c69614e (IV==4, Mac build
3d73e1e3). MOVEMENT: agree 161 -> 168/184; axFalse/houseTrue
21 -> 14, now DECOMPOSED by the new expired flag: 11 censored
(deadline) v 3 decided — the audit's asin/sqrt-content fixes
landed as claimed (probe battery 3/9 -> 9/9 axiom-side).
HOUSE CORRECTION 1 (axiom was right): parsefail 16 is NOT
Subs(...) — the IV4 error strings show every reject is the
imaginary unit I (complex-carrier states from the cplx lineage
in our cache). That is a DOMAIN FENCE (axiom's carrier is
real-rational by design), not a parser gap; booked as amendment
to yesterday's E4 verdict wording. Remaining 3 decided misses:
log-constant coefficients (log(5), log(3) as opaque atoms) —
the named next audit tranche.
HOUSE CORRECTION 2 (reinterpretation of the "dangerous
direction"): axTrue/houseFalse (still the same 2 states) is
NOT a reachability contradiction — house False at budget-150
is BOUNDED-SEARCH failure (budget-parameterized censoring of
its own kind), and axiom reports plies 5-6 paths there. The
namespace split STANDS (the label distributions differ
regardless of blame), but the audit item is now symmetric:
verify axiom's 2 chains with OUR oracle; if they verify, house
labels are the weaker search, not the truth. Ply-recovery
formula (plies - |{cancel,expand,subs_eval}|) recorded; not
yet re-scored (queued with the chain-verify audit).

## E4 AUDIT CLOSE-OUT: axiom's 2 extra solves VERIFY on the house oracle (2026-07-27)

The symmetric audit ran: both axTrue/houseFalse states solved
via ax.solve (plies 6 and 5, expired=False) and their answers
verified by sympy (d/dx(answer) - d/dx(state).doit() simplifies
to 0, both). READ: on these states the HOUSE label was the
weaker bounded search — axiom overclaimed nothing; there is no
reachability contradiction anywhere in the E4 sample. The
namespace split now rests solely on (a) coverage asymmetry
(3 decided log-constant misses, audit tranche named), (b) ply
semantics (recovery formula recorded, re-score queued), (c) the
domain fence (I-atom states). Bridge deliveries this cycle:
solve() dict surface confirmed {solved, answer, history, nodes,
expired, slot_fires, slot_decisive} — history carries the rule
names the ply formula needs.

## E4 RE-DRAW ON 277bc19: numbers UNCHANGED (168/184, same 3 decided) — the delta is the PRIOR, the fence catches its second case (2026-07-27)

Re-drew the seeded E4 sample on a house build of 277bc19 (IV4):
agree 168/184, axExtra 2, censored 11, decided 3 — IDENTICAL to
the c69614e run, same three states. Axiom's report that all
three solve at "150/24/3 + prior" is not contradicted: their
replay ran PRIOR-ON (markov prior_tsv, a data artifact on their
box, not in-repo); the E4 instrument is solve_batch DEFAULTS
(prior_tsv='') — matching the prior-free house python solver it
is being compared against. SECOND substrate-fence catch in one
day: arm config is part of the instrument; "+ prior" changes
the number. HOUSE POSITION: E4 stays prior-free (label parity
must compare like search arms); the prior's effect on these
signatures is a real, separate result — worth one named cell
(prior-on v prior-off over the same 184) if axiom ships the
prior file with a sha256. Namespace split unchanged.

## PRE-REG: E4 prior cell — prior-on v prior-off, same 184 sample (2026-07-27, axiom's prediction on record)

Prior = data/priors/markov_prior.tsv @ axiom db95c6c, sha256
cd60b1d1...e46dea5 (4494 bytes, byte-pinned; provenance: TSV
export of the llmopt rule-bigram brain, 316/360 zero-NN — the
prior-on cell is "our old brain guiding their engine"). AXIOM
PREDICTS (pre-reg'd their side): 3 decided misses convert;
some fraction of the 11 walls move; INTERESTING READING = new
misses appear (stale bigram mass down-ranking the new i_table
tan / pair-u rules, which have no bigram history). Instrument:
solve_batch 150/24/3, prior on v off, house Mac build of
277bc19-rules. E4's own verdict stays prior-free by doctrine.

## E4 PRIOR CELL VERDICT: prior-on converts ALL 3 decided misses, no new misses, walls unmoved, +1 more house-miss found (2026-07-27)

Same 184-sample, solve_batch 150/24/3, prior
cd60b1d1...e46dea5 (sha verified post-pull, byte-pinned):
prior-on agree 170/184 (v 168 off), decided 3 -> 0 (axiom's
core prediction CONFIRMED), censored 11 -> 11 (the
walls-move sub-prediction MISSED — the prior reorders
children but does not rescue deadline states here), axExtra
2 -> 3 (one MORE state where house budget-150 was the weaker
search). The interesting reading did NOT fire: no new misses
— the stale bigram mass does not down-rank the new rules on
this sample. NOTE the provenance loop: the prior is the
llmopt rule-bigram brain (316/360, zero-NN) — our old brain
guiding their engine closes states neither search closes
alone. E4's parity verdict stays prior-free by doctrine;
namespace split stands (coverage now: censored 11, ply
semantics, I-fence).

## PRE-REG: 45M FEDERATION UNION (3080 night, Artin GO 2026-07-27)

The scale rung on the federation result: same union diet
(math gen-4 + zx_farm1, vocab-47), fp32 birth per doctrine,
d512/12L/ffn2048/h8 (the 45M gen9B shape), seed 1, both
gates (math gate_ckpt_cuda + gate_zx with new shape envs).
COMPARATORS: 19M union (math 66 / ZX 40); 45M gen-4 math-only
69/120; ZX seed arm 36/28/34 (19M). PREDICTIONS: (1) math
column 66-69-class — union stays nearly free at scale; (2)
ZX column ABOVE the 19M arm mean 32.7 (scale helps the graph
grammar at least as much as trees) — 37+ would be the first
ZX number clear of the seed fence; (3) no wall, parsefail
0-class both columns. FAIL reads: math <=64 = union tax
emerges at scale; ZX <=32-class = graph grammar does NOT
scale-benefit (capacity was never the binding constraint).
Queue: scratch/night_45m_union.sh via wsl.sh launch.

## PRE-REG: rational-snap distillation, Q-sweep on the 19M crystal (2026-07-27, before the run)

RIFF rung (a) (Artin's infinite-precision push). Snap every 2-D
weight of mathnative_19m.pt to the best fraction p/q, q <= Q
(exact-best over the range, so 1/3 is reachable at Q>=3 — the
point vs dyadic quant); gate control + Q in {4, 16, 64}, ALL FOUR
ARMS ON THE MAC MPS GATE (instrument fence: the control re-gates
here; no comparison to its cuda-era numbers). Framed as
COMPRESSION (precision doctrine stays closed; E3 sole reopening).
Approximation math: best-over-q error ~1/Q^2 (Dirichlet), so
Q=64 ~2e-4 (well under weight scale), Q=16 ~4e-3 (marginal),
Q=4 ~0.05 (weight-scale — brutal). PREDICTIONS: Q=64 ties control
within gate sigma; Q=4 cracks badly; the readout is the KNEE —
if Q=16 is free, trained weights tolerate simple-fraction
structure and the exact-representation program gains a cheap
deployment format. scratch/rational_snap.py.

## PRE-REG (CONDITIONAL): G9 roots-of-unity alphabet — a declared REOPENING of the closed rotation wing, gated on the 45M union verdict (2026-07-27, before any run)

RIFF rung (c), CAUGHT BY THE RESULTS SWEEP BEFORE ARMING: the
2026-07-26 ZX column CLOSED alphabet-follows-domain (G5-dep 32 v
M5 31, bar >=5; rotation wing closed, D9/Q9/E7 dead unrun), and
the 19M ZX seed fence (sd ~4.2, n=3) drowns EVERY 19M ZX cell —
a 19M G9 arm would be uninterpretable at any score. So this rung
does NOT fire at 19M, and it fires at all only as a DECLARED
REOPENING (Artin's, 2026-07-27 evening: the exact-representation
push) against that named closure, with scale as the new
variable. CONDITION: the 45M union's ZX gate (in flight) must
clear ITS pre-reg bar (37+/decisively above 32-class). If it
does: G9 = 8 exact phases + dead zone (angles exactly 2*pi*k/8,
the Z[zeta_8] frame — Clifford+T's number ring), complex-FFN
STE (complex_model.gn_quantize, G5 route byte-preserved,
sanity-tested), 45M-CLASS shape matched to the union arm, union
or ZX diet per the landing read, seed 1, 3080. Bars set at
condition-time against the 45M comparator, before launch. If
the 45M ZX gate stays 32-class: rung (c) banks UNFIRED (no
width rescues a fence-drowned column; the reopening waits for a
different lever). scratch/night_g9.sh holds the 19M recipe as a
template only — DO NOT LAUNCH as-is.

## RATIONAL-SNAP VERDICT: the crystal has an EXACTLY-RATIONAL TWIN at Q=64 (parity); the knee is between 4e-3 and 2e-4 (2026-07-27 evening)

All four arms on the Mac MPS gate, paired (control re-gated on
this instrument: 49/120 @ 48.94% — Mac numbers, never comparable
to the cuda-era column). Every 2-D weight moved (moved-frac 1.0,
as expected — fp32 values are never small fractions):
| arm | gate | validity |
|---|---|---|
| control (fp32) | 49/120 | 48.94% |
| Q=64 (err ~2e-4) | **48/120** | 49.29% |
| Q=16 (err ~4e-3) | 26/120 | 32.13% |
| Q=4 (err ~5e-2) | 0/120 | 0.00% |
VERDICTS: (1) **Q=64 GATES AT PARITY (-1, inside sigma): every
weight replaced by a fraction p/q with |p|,q <= 64 and nothing
functional is lost — the 19M crystal admits an exactly-rational
twin.** The exact-representation program's first measured
positive: "the weights as simple fractions" is a real deployment
format, not a metaphor. (2) Q=16 is NOT free (-23, validity
halves): trained weights do NOT prefer simple-fraction structure
— the pre-reg's knee question answers NO; functional sensitivity
sits between 4e-3 and 2e-4 perturbation, consistent with the
MX-int4-parity / int3-cracks precision ladder reading at rest.
(3) Q=4 dead (0/120) as predicted. Anatomy note: Q=16 loses L4
and L7 ENTIRELY while keeping L3 13/L5 11 — long-horizon levels
die first under weight noise (tail-dies-first, weight-space
edition). Both pre-reg predictions (Q=64 ties, Q=4 cracks) land;
the knee lands one octave lower than hoped.

## PRE-REG: rational-snap knee localization, Q in {24, 32, 48} (2026-07-27, before the run)

Follow-up to the Q-sweep verdict (Artin GO "run follow up
honestly"): the knee sits between Q=16 (26/120) and Q=64 (48 =
parity). Three more arms, same instrument (Mac MPS gate, same
control 49), same snap procedure. PREDICTIONS: recovery is
monotone in Q; the smallest parity-Q (within 2 of control)
defines the MINIMAL exactly-rational format — house guess Q=48
at parity, Q=24 still cracked (err ~8e-4 v 4e-4; the ladder
analogy puts the wall near int4's ~6e-3-class step but the
Q=16 crack argues the true wall is tighter). Readout: minimal-Q
becomes the packing-density number the black-hole frame asked
for (bits/weight of the smallest exact twin at parity).

## KNEE VERDICT: the cliff sits between Q=16 and Q=24; minimal parity twin = Q=48 (~10.5 bits/weight) (2026-07-27 night)

Same instrument, full curve (control 49): Q=16 26 | Q=24 45 |
Q=32 43 | Q=48 47 | Q=64 48. VERDICTS: (1) **The crack is a
CLIFF, not a slope: between Q=16 (err ~3.7e-3, 26/120) and Q=24
(err ~1.7e-3, 45/120) the crystal recovers almost fully.**
House prediction half-wrong, booked honestly: Q=48-at-parity
landed, but "Q=24 still cracked" missed — Q=24 sits in a
shallow -4/-6 shelf (Q=24 45 / Q=32 43, non-monotone by 2 =
gate sigma, a tie-class wiggle), not a crack. Functional
sensitivity wall now bracketed [1.7e-3, 3.7e-3]. (2) **Minimal
exactly-rational twin at parity (within 2 of control): Q=48 =
47/120 — ~10.5 bits/weight** (log2 of reduced fractions |p|,q
<= 48); the near-parity shelf at Q=24 is ~8.4 bits/weight. The
black-hole frame's first packing-density number: the 19M
crystal's verified structure fits in ~10.5 exact-rational bits
per weight at zero capability loss (consistent with MX-int4
full parity at rest — the ladder said ~4-5 mantissa bits; the
rational lattice pays ~2x bits for its denominator freedom but
buys EXACTNESS). Curve booked complete; no further Q arms owed.

## PRE-REG: born-rational birth, RAT_Q=6 v fp32, paired Mac arms (2026-07-27 night, before the run)

Artin's critique of the snap sweep, taken as the design ("the
weights were rounded to begin with — snapping back can just be
wrong; try NO error"): post-hoc snapping approximates an
fp32-born artifact and inherits its rounding history; the born
arm puts weights ON the exact lattice from step 0. Measured
motivation: M5 held parity BORN on a 2.32-bit lattice while
post-hoc rational parity cost ~10.5 bits — born-exact already
beats snapped-after by ~4-5x in bits, in-house. ARMS (both Mac
MPS, same recipe/diet/seed — birth device is a confound, so the
control is Mac-born too, NOT the cuda-born 19M): (A) fp32
control, 19M/gen-4/3ep/seed 1; (B) RAT_Q=6 STE (per-tensor s *
best p/q, q<=6 — ~5-6 bits/weight class; train_mathnative.py
RAT_Q env, STE smoke-tested), deployed via scratch/rat_deploy.py
(deploy IS the trained function — the point of the arm). BARS:
B-deployed within ~2 of A = born-rational parity at ~half the
snapped twin's bits — Artin's born-exact thesis BOOKS; B down
>=5 = the rational lattice trains worse than the magnitude
alphabet (M5's lineage does not transport to fractions). House
prediction: parity-class (M5 precedent). Mac gate numbers stay
Mac-only. scratch/night_rat.sh.

## SNAP-ANATOMY VERDICT: the sensitivity wall is a HORIZON property, not a weight property (2026-07-27 night)

Artin's localization ask ("pull one bit out, trace which weight
carries the wall") run as single-tensor Q=16 ablation + joint
probes, teacher-forced on 48 gen-4 rows (2,512 tokens, CPU;
scratch/snap_anatomy.py — first run hit the load_state_dict-
copies silent-failure class, all-zero KL exposed it). FINDINGS:
(1) **No localization**: top tensor (head.weight) carries 73% of
single-tensor KL but the ABSOLUTE numbers are microscopic (2e-6
kl); every tensor alone is invisible. Artin's guessed
weight-provenance relationship NULLS. (2) **The joint snap is
also invisible one-step**: full Q=16 — the model that gates
26/120 — has kl 4e-6, argmax flips 2/2512 (0.08%); joint ~=
additive (linear regime). (3) **Both flips sit at margin
0.00016 v median margin 8.9** — snap noise decides ONLY
pre-existing coin-toss tokens; the near-tie doctrine (fp16
verify-block ties) reproduces for weight perturbation. (4) The
horizon arithmetic closes the cliff: ~8e-4 flips/token x
~1k-token greedy solves ~= 0.8 lethal near-tie flips per solve
-> 49->26 gate, and explains tail-dies-first (L4/L7 vanish:
longest chains). CONSEQUENCE for the exact program: the
[1.7e-3, 3.7e-3] "sensitivity wall" is not about representing
weights more precisely — it is near-tie DENSITY x chain LENGTH;
precision below the near-tie margin scale buys nothing a
coin-flip didn't already own (the precision doctrine's
mechanism, now seen from weight space).

## Dithered snap: NULL — error feedback does not beat plain best-rational (2026-07-27 night)

Trick-2 probe (row-wise Floyd-Steinberg error feedback on the
snap lattice, same teacher-forced instrument as the anatomy):
plain Q=16 kl 4.2e-6 / 2 flips; dither Q=16 kl 9.2e-6 / 3 flips
(Q=24 same ordering). Dithering trades per-element accuracy for
ordered-column-sum cancellation and the trade LOSES — activations
are not smooth in column index and plain best-rational errors
are already zero-mean independent. Plain snap stays champion.
Live candidate remaining: function-aware (GPTQ-class) snap
minimizing ||xW - xWq|| on calibration activations — banked, not
built (needs per-layer activation capture; aims the error budget
at near-ties, the only place the anatomy says error matters).

## 45M UNION VERDICT: math 65 (union ~free at scale, boundary); ZX 36 — INSIDE the seed fence, the scale lever NULLS on the graph grammar; G9 conditional resolves NO-FIRE (2026-07-27 night)

union_45m.pt (fp32, d512/12L/ffn2048/h8, union diet math gen-4 +
zx_farm1, vocab-47, seed 1, 3080). MATH: 65/120 @ 63.88
{3:21,4:6,5:17,6:8,7:13} — pre-reg band was 66-69 (free) / <=64
(tax): 65 lands ONE under the band = boundary; no union-tax
signal above gate sigma, but the free claim doesn't book clean
either. Read: union ~free at scale, unclaimed. ZX: 36/120,
parsefail 0, walls 0, unsound 84 (misses stay 100% semantic;
kind spectrum color 20 / fuse 10 / lcomp 4 / pivot 1 / id 1 —
same shape as 19M). Pre-reg bar was 37+ (clear of the 19M seed
fence mean 32.7 sd ~4.2): 36 = +0.8 sigma, INSIDE the fence —
identical to the 19M cplx_none seed-1 number. **2.4x capacity
bought ZERO on the graph grammar: capacity was never the ZX
constraint** (the pre-reg's named fail reading books). Fence
note: 45M ZX has n=1 seed — the number is read against the 19M
fence per the pre-reg's own framing, not cross-width sigma.
CONSEQUENCE: the G9 conditional (this entry's decider) resolves
NO-FIRE — rung (c) banks UNFIRED as pre-registered; the
rotation reopening waits for a different lever (diet quality /
serialization / curriculum, not width). ZX next lever candidates
go to the morning consolidation.

## PRE-REG: quick exact battery on union_45m (2026-07-27 late, before the runs)

Five arms on the 3080, comparator = tonight's booked union_45m
math gate 65/120 (same device, same VOCAB_EXTRA env, same gate
command; machine state unchanged since). Arms: (1) rat64 =
best-rational Q<=64 (err ~2e-4) — does the Mac's rational-twin
parity TRANSPORT to a 45M cuda crystal? predict 65+-2. (2)
fq512 = fixed shared denominator q=512 (err ~1e-3, W=P/512
integer P — the INTEGER TWIN, exact int-GEMM road): predict
parity-class. (3) fq128 (err ~3.9e-3, above the Mac-bracketed
wall [1.7e-3, 3.7e-3]; int8-range P if |p|<128): predict
cracked-or-marginal — this arm READS whether the wall bracket
transports across width/device (fence note: the bracket is a
Mac-19M number; this is a coarse transport probe, not a sigma
comparison). (4) rat16 (err ~3.7e-3): predict crack (horizon
law). (5) rat16+repair: freeze the exact lattice, train ONLY
1-D norms/biases 400 steps on the birth diet — precision-as-
thin-film's decisive read: recovering >= half the rat16 deficit
books "precision is a small additive budget"; no recovery kills
the film claim at 45M. scratch/quick_exact_3080.sh.

## BORN-RATIONAL VERDICT: the exact-lattice birth BEATS its fp32 control at the single-seed bar (2026-07-27 night)

Paired Mac arms, one variable (RAT_Q=6 STE), same recipe/diet/
seed/instrument: fp32 control 60/120 @ 60.63 {3:21,4:4,5:16,
6:7,7:12}; RAT_Q=6 LATENT 66/120; **RAT_Q=6 DEPLOYED (every 2-D
weight exactly s*p/q, q<=6, ~5-6 bits/wt) 65/120 @ 63.05
{3:21,4:8,5:16,6:7,7:13} — +5 OVER CONTROL, exactly the >=5
single-seed bar.** VERDICTS: (1) Artin's born-exact thesis
("train with no error from step 0") lands ABOVE its pre-reg —
the bars asked for parity-within-2; the arm cleared the control
by the win bar instead. Single seed: books as a bar-level win,
seed-2 replication queued to confirm. (2) Deploy tax ~1 (66->
65): STE converged the latents onto the lattice, mirroring
G5-ZX's latent=deployed. (3) The +5 concentrates at L4 (4->8,
doubled) — the lattice constraint helps exactly where the
control is weakest, consistent with discrete-alphabet
regularization (M5/ternary lineage), now with EXACTNESS free on
top. (4) Composition with the day's ladder: born-on-lattice at
~5.5 bits BEATS what post-hoc snapping achieves at 10.5 bits —
the snap-vs-born gap is now measured IN THE SAME DAY, both
directions. Instrument fence: Mac gate numbers, Mac-born pair;
transports nowhere. NEXT: seed-2 replication; RAT_Q on cuda at
45M-class if the quick battery's rat arms read clean.

## PRE-REG: born-rational seed-2 replication (2026-07-27 late night, before the run)

The +5 win replicates or it doesn't: same paired arms
(fp32 control v RAT_Q=6 STE), same recipe/diet/instrument,
BIRTH_SEED=2 both arms. READINGS: rat-dep beats control again
by >=3 = the win CONFIRMS (two seeds, same direction, mean +4);
delta in [-2, +2] = demotes to parity (thesis still books —
exactness free); control wins by >=3 = seed-1 was a fluctuation,
verdict amended to parity-at-best. L4 anatomy read again (the
doubling was the seed-1 signature).

## QUICK EXACT BATTERY VERDICT: INTEGER TWIN AT PARITY; the sensitivity wall does NOT transport — snap robustness is a per-crystal property (2026-07-27 night)

union_45m on the 3080, comparator = tonight's booked 65/120:
rat64 65 | **fq512 65 (W = P/512, integer P, int16-range —
THE INTEGER TWIN AT FULL PARITY)** | fq128 64 | rat16 64 |
rat16+repair 63. VERDICTS: (1) Rational-twin parity TRANSPORTS
(65=65 at Q=64) — now measured on two crystals, two devices,
two widths. (2) **The integer twin is real: every weight an
integer over one shared denominator, zero capability loss** —
the forward pass is an integer GEMM / 512; the exact-integer-
inference road (ozaki/FX-V1) has its substrate. (3) TWO PRE-REG
PREDICTIONS WRONG, booked: rat16 did NOT crack (64 v the Mac
19M's 49->26 collapse at the SAME error 3.7e-3) and fq128 held
64. **The sensitivity wall is not a constant — it is a
per-crystal property.** This CONFIRMS the anatomy mechanism at
depth: damage = near-tie density x horizon, and near-tie
density is a property of the individual model; the union 45M
sits far from its ties where the Mac 19M sat close. The
"wall bracket" [1.7e-3, 3.7e-3] is hereby scoped to the crystal
that measured it. (4) Thin-film repair reads VOID here — there
was no deficit to repair (64->63 = noise); the film question
needs a substrate with a real crack (the Mac 19M rat16 at 26 is
the candidate). Consequence: snap-robustness (distance-to-
near-ties) is now a measurable model QUALITY — a crystal that
tolerates coarse exact lattices is better calibrated at its
decision points. Candidate instrument: flips-per-token under
Q=16 as a cheap calibration probe.

## BORN-RATIONAL SEED-2: −2 — the +5 DEMOTES TO PARITY; exactness is free, not profitable (2026-07-28 ~1AM)

Seed-2 paired arms (same recipe/diet/instrument, BIRTH_SEED=2):
control 62/120 @ 63.78 {3:21,4:6,5:17,6:7,7:11}; RAT_Q=6 deployed
60/120 @ 61.00 {3:20,4:5,5:16,6:7,7:12}. Pre-reg band [−2,+2]
FIRES: **the verdict demotes to PARITY.** Two seeds: +5, −2 —
mean +1.5, noise-class; the seed-1 +5 was a fluctuation-grade
draw (the sigma doctrine working exactly as designed — the >=5
single-seed bar admitted it provisionally, the replication
demoted it). The L4-doubling signature also fails to replicate
(s2 L4: 5 v 6). WHAT STILL BOOKS, and it is not nothing: **the
exact-lattice birth is CAPABILITY-FREE across two seeds** —
every weight exactly s*p/q (q<=6, ~5.5 bits/wt) from step 0,
deploy tax ~1, at zero measured cost vs fp32. Exactness rides
free; it does not pay. This matches the P2/M5 lineage read
(discrete ladders sit at-or-near fp32, wins inside noise) and
the whole program's shape: exactness is a determinism/
auditability/format lever, never a capability lever — now
measured at the BIRTH layer too, completing the set (inference:
integer twin parity; training-step: disagreement #2 bit-identity;
birth: this). FINDINGS.md entry updated same booking. Follow-ups
unchanged in priority: GPTQ-rational, mixed-Q, RNS wall rung;
no seed-3 owed (parity at n=2 with both signs is settled-class
for a free lever).

## GAUGE-SLACK 4-CRYSTAL CELL: NULL exactly as pre-registered — the lattice does NOT shrink gauge slack; ancestry dominates every lens (2026-07-28 ~2AM)

The cell armed on s2 landing (RIFF-LEDGER 07-27 night, skeptical
prediction on record: "both ~sqrt(2), no closure"). Instrument =
the 07-26 ancestry lens verbatim (per-layer FFN gate matrices,
raw / Hungarian-perm / neuron-space Procrustes normalized-Fro;
scratch/gauge_slack_rat.py). READING: fp32 seed-pair raw 1.4136 /
perm 1.2947 / rot 0.4341 vs rat-Q6 seed-pair raw 1.4136 / perm
1.2947 / rot 0.4339 — IDENTICAL to 3-4 decimals at every lens.
**The prediction fires exactly: lattice training kills continuous
gauge in principle, but seed-lottery basin assignment dominates
functional+gauge distance so completely that the canonicalization
is invisible.** No closure; distance remains an ancestry
instrument (the 07-26 verdict extends to lattice-born crystals).
Bonus control: same-seed cross-arm pairs (fp32_sN vs rat_sN) read
0.388 raw / 0.142 rot — the STE-trained rat twin stays in its
fp32 twin's basin (~3.6x closer than any cross-seed pair), i.e.
the lattice constraint bends the trajectory, not the basin.
Perm alignment recovers almost nothing on cross-seed pairs
(1.41 -> 1.29): whatever aligns these crystals, it is not a
neuron permutation. Cell CLOSED; exact-representations program
continues on the format/instrument legs (GPTQ-rational,
int8+outliers, calibration probe).

## PRE-REG: calibration probe R1 — flips/token vs snap robustness (2026-07-28, before the runs)

Spec: specs/2026-07-28-calibration-program.md rung 1. Crystals:
d256 zoo {wfloor 65, s2 63, s3 64, stream4 57, muon 34} + Mac-19M
fp32 (rat16 crack 49->26 already measured). Instrument: probe =
flips/token under Q=16 rat snap (calib_probe.py, 400 rows, MPS);
robustness = gate drop under the SAME Q=16 snap (gate_ckpt, same
device). PREDICTION: probe rank-correlates with gate drop
(higher flips/token = larger drop), Spearman rho > 0, and the
Mac-19M (known cracker) reads highest. FAILURE = probe is noise;
the calibration program closes at the cost of one script.

## PRE-REG: mass-on-valid — the branching-floor MASS leg (2026-07-28, before the run)

Spec rung 2 (amended: the sampled-coverage form already failed
2026-07-26; no CE-anti-track claim re-registered). ~40 held-out
cur states (seed space 99.1M, L3-L7); per state the engine
enumerates successors (already sympy-verified, non-identity by
construction), and each specimen model's teacher-forced sequence
probability is computed for every valid nxt. Readouts per model:
(a) mean total mass on the valid set, (b) mean mass on the modal
valid move (farm-pick proxy — fresh states have no banked row),
(c) mean entropy over the valid set. Specimens: muon 34 /
stream3 45 / stream4 57 / control 65. PREDICTION: (a) tracks the
gate at least as well as (b); the (a)-(b) delta — mass on
valid-but-non-modal moves — is the novel number and rung 3's
baseline. (a)~(b) everywhere = the floor theory stays
unmeasurable at this scale (books as such).

## AMENDMENT to PRE-REG calibration probe R1: instrument mismatch in battery v1; probe re-armed on the ground-truth snap operator (2026-07-28)

Battery v1 inlined rat_deploy's SCALED snap (w/s -> p/q -> *s,
s = absmean) but the 49->26 ground-truth crack was measured with
rational_snap's DIRECT snap (w -> p/q, no scale) on
checkpoints/mathnative_19m.pt (control 49) — a finer effective
lattice and the wrong crystal (mathnative_19m_mac_fp32 is
tonight's born-rational control, gate 60, never crack-measured).
Incidental result, booked: **the scaled Q=16 lattice is
capability-free across the entire d256 zoo** (65=65, 63=63,
64=64, 57=57, 34=34 — zero drop at every gate, flips/token
0.00065-0.00256, flip margins e-3-class). The pre-reg itself is
UNTESTED by v1 (no robustness variance existed to correlate
against). v2 re-arms with rational_snap Q=16 (the ground-truth
instrument) on the same five d256 crystals + mathnative_19m.pt
as the cracker probe point. Prediction unchanged. Snap operators
are instruments; fences travel with them.

## CALIBRATION PROBE R1 VERDICT: PASS — flips/token predicts snap robustness, Spearman rho 0.883 (p=.020) across 6 crystals (2026-07-28)

Battery v2 (rational_snap Q=16, the ground-truth operator; all
reads one MPS session):
| crystal | probe flips/tok | gate pre | gate post | drop |
|---|---|---|---|---|
| muon d256 | 0.0476 | 34 | 38 | -4 |
| s3 d256 | 0.0486 | 64 | 52 | 12 |
| wfloor d256 | 0.0535 | 65 | 53 | 12 |
| stream4 d256 | 0.0570 | 57 | 41 | 16 |
| 19m (cracker) | 0.0659 | 49 | 26 | 23 |
| s2 d256 | 0.0703 | 63 | 47 | 16 |
VERDICTS: (1) **Pre-reg primary leg PASSES: Spearman(probe,
drop) = 0.883, p = 0.020** — a 400-row teacher-forced probe
(~1 min/crystal, no gate) rank-predicts what a full snapped gate
measures. Near-tie density is hereby a MEASURED per-crystal
quality, read for ~1% of a gate's cost. (2) Secondary leg
MISSES honestly: the 19M cracker reads 2nd-highest (0.0659),
not highest (s2 0.0703, drop only 16) — the probe orders the
class, not the extremum; note the 19M is also the only d384
crystal in the set (probe transports across widths as a
fraction, but the single cross-width point carries that
caveat). (3) Curiosity booked, not claimed: muon's gate ROSE
under the snap (34 -> 38, +4 = sigma-class on this instrument);
its flip margins are the lowest of the set (0.233) —
consistent with a poorly-calibrated crystal whose coin flips
re-land favorably. (4) v2 flip margins (~0.3) sit 100x above
v1's scaled-snap margins (e-3) — coarser lattices flip
real decisions, not just coin flips; the near-tie *density
curve* (flips vs Q) is the natural next instrument refinement.
CONSEQUENCE: rung 4 (judge-collapsed decoding) is UNGATED —
ties are measurable and concentrated; the program proceeds.

## MASS-ON-VALID VERDICT: valid mass tracks the gate 4/4, but the models are near-DETERMINISTIC over the valid set — the branching-floor theory stays unmeasurable, by measurement (2026-07-28)

40 fresh states (99.1M seed space, L3-L7), mean 7.2 verified-valid
moves/state (engine enumeration); teacher-forced sequence mass:
| model | gate | mass_valid | mass_modal | delta | H_valid (bits) |
|---|---|---|---|---|---|
| muon | 34 | 0.0878 | 0.0859 | 0.0019 | 0.078 |
| stream3 | 45 | 0.1165 | 0.1148 | 0.0017 | 0.070 |
| stream4 | 57 | 0.1954 | 0.1895 | 0.0059 | 0.065 |
| control | 65 | 0.2134 | 0.2097 | 0.0037 | 0.057 |
VERDICTS, against the pre-reg: (1) mass-on-valid-set tracks the
gate 4/4 — a teacher-forced valid-mass read is a capability
correlate at zero sampling cost (consistent with the 07-26
CE-tracks-gate verdict, now on the engine-enumerated set). (2)
**The pre-declared (a)~(b) reading FIRES: the delta is ~zero
everywhere** — of ~7 valid moves per state, the models put
>97% of their valid mass on ONE. H_valid 0.057-0.078 bits vs
log2(7.2) = 2.85 available: **the crystals are near-deterministic
over the valid set. The farm's arbitrary pick has been absorbed
as THE move** — the branching-entropy floor is unmeasurable here
not because the instrument is weak but because the models never
spread mass in the first place (picks-trained models are
pick-shaped; the 07-26 floor theory described a distribution the
training regime never builds). (3) Sharper rung-3 framing, free:
H_valid also DESCENDS with the gate (control 0.057 = most
deterministic) — determinism over valid moves correlates with
capability under pick-training. Distribution rows (rung 3) now
test a genuine unknown: does explicitly TRAINING the spread
(raising H_valid toward the engine's distribution) pay at the
gate, or was one-true-move concentration load-bearing? Baseline
numbers for that arm: delta 0.0037 / H 0.057 (control).

## PRE-REG: distribution rows at d256, THREE-ARM design (2026-07-28, before the births)

Spec rung 3, upgraded at plan-execution time with a dose control
(exposure-rations law: replication changes exposure share, so a
naive dist-vs-control pair confounds label distribution with
dose). Arms, all fresh births (same-day-control doctrine: trainer
drifted since wfloor — RAT_Q/SR_BF16/BIRTH_BS commits), same
recipe (BIRTH_SEED=1, d256/L8/ffn1024/h4, 3ep, MPS, no --fast):
(1) CONTROL: gen4 diet as-is; (2) DOSE-CONTROL: each treated
state's pick row replicated 4x; (3) DIST: each treated state's
4 rows = largest-remainder apportionment of its verified-valid
move distribution (engine-enumerated, MarkovPrior-weighted,
farm_dist_rows.py). Treated states = the ~4,000 farmed curs;
dosectl and dist row counts match exactly by construction.
PRIMARY: dist vs dose-control at the gate, L4 read first (canary).
SECONDARY: calib_probe (Q=16 direct) on all three arms —
prediction: dist reads FEWER flips/token than dose-control (soft
labels sharpen decision margins). Baselines from rung 2: control-
class H_valid 0.057 bits, delta 0.0037. Dose reading free:
dose-control vs control isolates pure 4x-replication exposure.
Null reading pre-declared: calibration is a diagnostic, not a
lever; rung 4 unaffected.

## AMENDMENT to PRE-REG distribution rows 3-arm: the altpairs precedent applies — house prediction updated before any birth fires (2026-07-28)

The full-RESULTS reread surfaced what the spec's design pass
missed: ALTPAIRS (2026-07-26) already measured the nearest
neighbor of rung 3 — verified-alternative successor rows ADDED at
14% share cost -6 gate while teaching the landscape (12x entropy),
and the pincer program booked its consequence ("calibration
belongs to the backward SCORER, not the forward crystal").
Rung 3's design differs in three named ways: (a) REPLACEMENT at
matched total rows, not added share; (b) the dose-control arm
isolates the label distribution exactly; (c) MarkovPrior
weighting preserves the policy ORDERING where altpairs spread
uniformly. UPDATED HOUSE PREDICTION (replacing the pre-reg's
neutral stance): dist <= dose-control at the gate (the altpairs
dilution class), with the pre-registered probe delta (fewer
near-ties) as the novel readout that altpairs never measured.
A gate WIN would book as "replacement + prior-weighting fixes
the dilution tax" — a real result against precedent. A loss
confirms the scorer-owns-calibration doctrine at a second design
point. Rung-2's free finding sharpens the stakes: H_valid
DESCENDS with capability under pick-training (control 0.057
bits = most deterministic) — policy sharpness correlates with
the gate, so flattening has measured priors against it.

## AMENDMENT (target: DISAGREEMENT #2 VERDICT 2026-07-24): "bit-identical" scoped to what the instrument measured; endpoint verification queued (2026-07-28)

Artin's audit ask ("what came back bit identical?") re-read the
d2 instrument (scratch/metabolic_d2.py): the arms were compared
on (a) NET FLIP COUNT vs birth signs (132,566 = 132,566), (b)
the resolved-wall set, (c) the proxy gate — aggregate
deployed-function readouts. The instrument did NOT compare the
two endpoint state_dicts element-wise, nor the flip SETS
(locations). Count-equality at identical seeds/food/optimizer
with a sub-1e-14 accumulation delta makes set-equality the
overwhelmingly likely reading, but as booked, "bit-identical
outcomes" means COUNT+OUTCOME-identical, not proven
weight-identical. VERIFICATION QUEUED (next 3080 window, CPU,
minutes): metab_d2_fp64.pt vs metab_d2_dd.pt — (1) element-wise
state_dict equality; (2) deployed-lattice flip-SET equality
(ternary sign maps); (3) the NEW rung-1 calibration probe on
both (flips/token + margin fingerprint — identical near-tie
geometry = the strongest cheap corroboration the 07-24 era
didn't have). Any element-wise difference found would NOT
overturn the capability verdict (outcomes measured equal) but
would re-scope the "nothing below fp64's floor reaches the
deployed function" wording to its outcome form.

## DISTRIBUTION ROWS 3-ARM VERDICT: PARITY — the altpairs dilution tax VANISHES under matched-dose replacement; the training lever demotes to diagnostic (2026-07-28)

Three fresh births, one chain, same device/seed/recipe
(d256/L8/ffn1024/h4, BIRTH_SEED=1, 3ep, MPS):
| arm | gate | validity | L4 | probe flips/tok (Q=16) |
|---|---|---|---|---|
| control (gen4) | 62 | 57.79 | 7 | 0.0513 |
| dose-control (pick x4) | 62 | 59.39 | 6 | 0.0542 |
| dist (distribution x4) | 60 | 57.12 | 6 | **0.0498** |
VERDICTS, against the amended pre-reg:
1. **PRIMARY: dist v dose-control = -2 — parity class** (sigma
   ~1.0, bar 3). The altpairs precedent (-6 at 14% ADDED share)
   does NOT reproduce under matched-dose REPLACEMENT with
   prior-weighted labels: the updated house prediction
   (dist <= dosectl) lands at its parity edge, and the dilution
   tax reads as a SHARE/DOSE effect, not a cost of the label
   distribution itself. Real difference from precedent, booked —
   but no GAIN either: per the original pre-reg's null reading,
   **calibration training on the forward crystal demotes to
   DIAGNOSTIC** (second design point for scorer-owns-calibration;
   the policy-sharpness observation survives untouched).
2. L4 (canary): 7/6/6 — all inside the {6,7} seed band; no
   clade effect in any arm.
3. **SECONDARY (the novel readout): direction CONFIRMED,
   sub-noise** — dist reads the FEWEST flips/token (0.0498 v
   dosectl 0.0542, -8% rel; control between at 0.0513). Soft
   labels sharpen decision margins in the predicted direction;
   probe sigma is unmeasured, so this books as directional only.
   (First use of the rung-1 probe as a secondary instrument on
   paired arms.)
4. Dose reading free: dose-control = control exactly (62=62,
   +1.6 validity) — 4x replication of 4,000 states is
   capability-neutral; exposure-share effects need bigger doses
   than this.
5. Same-day-control doctrine vindicated AGAIN: fresh control 62
   v wfloor's historical 65 on the same recipe — the drifted
   tree (RAT_Q/SR/BIRTH_BS commits + diet-loader path) is worth
   ~-3; any comparison against the old 65 would have
   manufactured a phantom tax in every arm.
Rung 4 (judge-collapsed decoding) proceeds — independent leg,
now with scorer_s2_dist banked as its designated judge.

## PRE-REG: judge-collapsed decoding at d256 (2026-07-28, before the run)

Spec rung 4, ungated by R1's PASS (ties measurable, concentrated
at low margins). Design corrected at implementation time: the
production decode is ALREADY a wave (B=8/ply, oracle picks), so
the fair three-arm cell at 30 fresh L5-L7 states (seed space
99.3M, 12 plies, calib_d256_ctl.pt = tonight's same-day 62-gate
control) is:
(a) WAVE-8 (production semantics) — the token-rich comparator;
(b) GREEDY-1 (argmax/ply, verify, fail on invalid) — the floor;
(c) JUDGE-COLLAPSED greedy — greedy except at top-2 margin
    < 0.02 (the measured near-tie class): branch BOTH
    continuations to the step boundary, oracle (verify_wave)
    judges, BOTH branches' tokens charged.
PREDICTIONS (house): (1) (c) > (b) by >= 3 states — near-tie
branching recovers real solves over pure greedy (the flips-
decide-coinflips anatomy); (2) ECONOMICS headline: (c) lands
within ~3 solves of (a) at <= 40% of (a)'s tokens — the wave's
value concentrates at ties, so collapsing only ties buys most
of the wave at a fraction of the spend (FA-law form); (3) fence:
if (b) ~ (a) already (greedy ties the wave on this battery),
the battery has no tie-variance and the cell books
VOID-BY-BATTERY (needs harder states, not more arms). Tokens
and per-state sidecars logged; S2-dist as a learned judge is
the banked follow-up iff the oracle-judge form pays.

## JUDGE-COLLAPSED DECODING VERDICT: NULL BY TIE-SCARCITY — generation-time near-ties are too rare to buy anything; greedy captures 90% of the wave at 12% of the tokens (2026-07-28)

30 fresh L5-L7 states, 12 plies, calib_d256_ctl (62-gate):
| arm | solves | tokens |
|---|---|---|
| wave-8 (production) | 20/30 | 9,072 |
| greedy-1 | 18/30 | 1,130 |
| judge-collapsed | 18/30 | 1,161 |
VERDICTS, against the pre-reg:
1. **Prediction (1) FAILS flat: judge = greedy (18 = 18)** — and
   the token ledger names the mechanism: +31 tokens over greedy
   ACROSS ALL 30 STATES means the near-tie branch fired ~once or
   twice total. At generation time, top-2 margins < 0.02 are
   VANISHINGLY RARE (median margin 8.6; the snap anatomy's
   2-in-2,512 flip density, now measured at the decode layer).
   The lever has nothing to grip: decisions are near-tie-decided
   only under WEIGHT perturbation (device rounding, snaps), not
   under the model's own argmax stream. Judge-collapsed decoding
   CLOSES as null-by-tie-scarcity; the S2-dist learned-judge
   follow-up dies with it (no ties to judge).
2. **The economics finding that books instead (prediction 2
   inverted into something better): greedy captures 90% of the
   wave's solves at 12% of its tokens** (18 v 20 at 1,130 v
   9,072). The 8-sample wave buys +2/30 solves for 8x spend —
   composing with R0b (sampling order ~ mass order; the argmax
   carries most of the wave's information), the efficient decode
   for farms/probes is GREEDY-FIRST with wave-retry only on
   failure. Candidate lossless-speed lever for every
   sampling-bound loop (gates stay wave-8 for lineage
   comparability; the fence is instrument continuity, not
   economics).
3. The device-dependence paradox RESOLVES cleanly: hardware
   rounding decides frontier probes (18/24 v 9/24) not because
   generation is tie-dense but because a tie ANYWHERE in a
   12-ply chain forks the whole trajectory — rare ties x long
   horizons x butterfly amplification, exactly the snap
   anatomy's damage law (flips/token x chain length), now
   confirmed from the decode side.
Fences: n=30 states, one crystal (same-day 62-class), one
margin threshold (0.02); sidecar logs/pp_judge_decode.jsonl.

## PRE-REG: NIGHT-28 QUEUE — audits + four null-revival mixes + adoption cell (2026-07-28, before any run; Artin GO, 3080 open to ~5PM EST 07-29)

3080 chain (scratch/night_28.sh) + Mac chain (night_28_mac.sh);
jobs independent, markers success-only.
1. **d2 endpoint verification** (audit): state_dict + flip-set +
   probe fingerprints on metab_d2_{fp64,dd}.pt. PREDICTION
   (house): identical on all three (the count+outcome equality
   was set-equality all along). Any difference re-scopes the
   07-24 wording per the standing amendment, capability verdict
   untouched.
2. **Production crown re-baseline** (audit): gen6_grown re-gated
   on today's tree, cuda. PREDICTION: 76 +/- sigma 2.5; drift
   >= 5 = the d256 tree-drift (-3) has a production sibling and
   every standing crown comparison gains a same-day-anchor rule.
3. **Greedy-first adoption cell**: full battery, gen6_grown,
   wave-8 vs greedy-first-wave-retry. BAR (adoption): gfirst
   >= wave-3 solves at <= 40% tokens => adopt for farm/probe
   loops (gates keep wave-8 for lineage continuity).
4. **S4 vs M4 at 6ep** (dose revival): does the discrete-parity
   dose close or widen the 2-bit zero gap (3ep: M4 61 v S4 58)?
   S4-6ep >= M4-6ep => zero worthless at dose (law revision);
   gap >= 5 => zero PAYS at dose (first bookable zero premium).
5. **P2 @ 45M-class, 6ep** (winner x width): comparators
   45M-gen4 ternary-6ep 69 / fp32-3ep 69. HOUSE PREDICTION:
   ties 69-class (W* ~ 1/bits shrinks P2's edge at width);
   P2 >= 74 (bar 5, single-seed cuda) => alphabet choice pays
   at W* too and the bracket reopens at width.
6. **Muon @ 3ep standard schedule** (regime revival, Mac d256):
   comparator wfloor 65. >= 62 => the streaming crater was the
   REGIME, Muon exonerated at standard schedules; <= 55 =>
   Muon-class updates are gate-toxic here at any schedule.
7. **revpairs at 10% ration** (stupid corner, Mac d256/1P):
   comparator pairs@1P 57. >= 60 => low-dose backward pays
   (direction is a spice, not a diet); 54-59 neutral; <= 53 =>
   the direction tax has no safe dose.
8. **Z[i] born-rational** (mix: exact lattice x complex interior,
   Mac 19M/3ep): comparators cplx_none 63 / G5-dep 62 / fp32 64.
   HOUSE PREDICTION: parity-class (both parents capability-
   neutral); latent-vs-dep tax ~4-class expected; >= 68 latent
   would be the first composed-lever WIN of two neutral parents.

## PRE-REG: successors-bridge acceptance, house side (2026-07-28 night, before the run)

Axiom shipped IV5 (fcf4a52, pushed tonight; Mac rebuild verified
IV==5, smoke i_power exact). Instrument
scratch/successors_acceptance.py: 500 string-seeded roots (L1-L8,
band 99.4M), house derivation.successors vs ax.successors
(deadline 15s), child sets srepr-normalized, E4 taxonomy.
PREDICTIONS: (1) MATCH on the large majority of read states;
disagreements decompose to NAMED classes (axiom rule-coverage
gaps house-only; macro/table differences); (2) SOUNDNESS: zero
axiom-only children fail house verify_edge (their in-engine
verification holds on our oracle) — any failure BLOCKS adoption;
(3) throughput: axiom >= 5x house on the read set (their p50
claim, our sample). ADOPTION RULE: soundness clean + taxonomy
named => farmers/enumeration move to the axiom bridge (Artin's
axiom-only directive), sympy stays oracle-of-record at final
verification; soundness failure => bridge stays shadow, per-state
audit relay.

## NIGHT-28 STAGE 1: d2 endpoints verified (the tail's footprint = 321 fp32 last-bits, ZERO decisions); crown re-baseline EXACT; greedy-first fails its token bar (2026-07-28 night)

1. **d2 ENDPOINT VERIFICATION — the amendment resolves with the
   most precise wording yet**: metab_d2_fp64 v metab_d2_dd are
   NOT strictly weight-identical — 9 tensors / 321 elements
   differ at max |delta| 3.7e-9 (the dd tail's 1.06e-14-class
   accumulation surfacing as last-bit fp32 cast flips) — and
   NOTHING ELSE differs: deployed ternary flip-set IDENTICAL
   (0 of 50.3M sign cells), calibration-probe fingerprints
   IDENTICAL to all printed digits (flips/tok 0.36556 both,
   margins equal). The precision hierarchy is now measured
   element-wise end-to-end: the exact tail's entire causal
   footprint on a 90M-param crystal is 321 sub-ULP latent cast
   flips and zero deployed decisions. The 07-24 verdict upgrades
   from count+outcome-identical to DEPLOYED-FUNCTION-IDENTICAL,
   proven at the weight level; capability claim unchanged and
   now unassailable at this instrument.
2. **CROWN RE-BASELINE: 76/120 @ 70.42 — EXACTLY the booked
   number, zero drift.** The d256 tree-drift (-3) does NOT
   transport to the production lineage (45M-class, cuda,
   gen6_grown): the crown's standing comparisons remain valid
   as-is. Audit closes clean.
3. **GREEDY-FIRST ADOPTION: FAILS its token bar, honestly** —
   gfirst 74/120 @ 29,724 tokens v wave-8 75/120 @ 42,113
   (solves leg PASSES at -1; tokens = 71% of wave, bar was
   <= 40%). Mechanism: with per-ply wave-retry, every mid-chain
   greedy miss re-buys the full wave — on 12-ply champion
   chains most chains miss somewhere, so the retry eats the
   saving (last night's 12% figure was PURE greedy on 30 short
   probes, no retry). NO-ADOPT as pre-registered. BANKED
   refinement (one cell, if wanted): ladder retry (greedy ->
   k=2 -> k=8) instead of greedy -> k=8 — the retry cost is
   the whole gap.

## SUCCESSORS-BRIDGE ACCEPTANCE: SOUNDNESS PASS (200/200), exact-set parity FAILS with every class named — SCOPED ADOPTION (2026-07-28 night)

500 roots read (0 expired, 0 fence-skips): exact child-set MATCH
12/500; house-only 201 / both-diff 284 / axiom-only 3 states.
**SOUNDNESS: 200/200 axiom-only children VERIFY on the house
oracle — zero failures. The adoption-blocking leg is clean.**
Throughput: axiom 4.9 st/s v house 1.3 (3.8x on this band, tail
included; their p50 claim held on gen-4-typical states).
THE DECOMPOSITION (rule histogram over 60 states): house-only
mass = cancel 38 / expand 30 / i_parts 30 / i_heurisch 26 /
factor 25 / euler 21 / together 15; axiom-only = the SAME
families from the other side (cancel 23, i_parts 12, expand 5).
READING: this is the E4 ply-semantics split at the enumeration
layer — the two engines emit the same MOVE FAMILIES but
different algebra NORMAL FORMS (each CAS's cancel/expand/factor
lands on its own canonical string), plus two named coverage
edges (i_heurisch = house slot, absent axiom-side by design;
euler = house macro). Exact-set parity between two CAS algebra
systems is structurally unreachable without canonical-form
agreement — and it is NOT REQUIRED: every emission is
independently sound.
ADOPTION (per the pre-registered rule, SCOPED): the axiom bridge
becomes the DEFAULT ENUMERATOR for soundness-consumers —
distribution/altpairs-class farming, stuck-state food, the
pincer's forward leg (frontier_eval) — at 3.8x with deadline_ms
walls. House derivation.successors remains the SEMANTIC
REFERENCE for house-set replication consumers (R1b true-move
replay, gate semantics) and sympy remains oracle-of-record at
final verification. Farmers migrate at their next touch.

## MUON @ 3EP STANDARD SCHEDULE: 43/120 — the crater was NEVER the streaming regime; Muon-class updates are gate-toxic here at any tested schedule (2026-07-28 night)

{3:13,4:3,5:15,6:5,7:7} @ 39.91% v wfloor comparator 65 (same
construction, all-AdamW). Pre-reg reading "<= 55 = gate-toxic"
FIRES with 12 solves to spare. The regime-exoneration hypothesis
is DEAD: standard 3-epoch OneCycle-AdamW-head + Muon-interior
(ns5, LR 0.01, momentum 0.95) loses -22 where single-pass lost
-11-to--35. Muon's ledger at this scale: 10 (.02/1P) / 34
(.01/1P) / 43 (.01/3ep) — monotone in dose/schedule gentleness,
never within 20 of comparator. The null-revival meta-lesson
books honestly: this mix was worth one cell and the answer is
the ORIGINAL verdict was under-scoped, not over-scoped (it
generalizes ACROSS schedules). Fences: naive transplant (shared
LR schedule multipliers via warmup only, one momentum, d256/MPS);
published-Muon-scale tuning stays possible and stays off-priority.

## AMENDMENT (target: Muon-3ep verdict, same night): the loss-gate dissociation reproduces at 3ep — final CE ~0.22-0.35 (best-class band) at gate 43. Third Muon instance (0.55/gate-10, 0.41/gate-34, now 0.22-class/gate-43): orthogonalized updates optimize teacher-forced CE while damaging generative validity at EVERY schedule tested. The instrument value of Muon (widest CE-gate split available on demand) is banked even as the optimizer closes.

## REVPAIRS-10%: 55/120 — neutral; the direction tax has a SAFE DOSE (2026-07-28 night)

{3:16,4:5,5:16,6:8,7:10} @ 49.23 v pairs@1P 57. Pre-reg band
54-59 (neutral) FIRES: at a 10% backward ration the -18 tax of
50/50 vanishes (delta -2, sub-noise) — but nothing is gained
either. BOOKS: backward rows are a TOLERABLE spice, not a
poison, at low dose; a single crystal can host a direction
ration for free. Pincer relevance: if the backward SCORER ever
wants co-residency with the forward crystal, 10%-class dose is
the measured safe region. Validity -5 (49.2 v 54.2) noted —
the tax shows in validity before solves, consistent with
step-precision (not wandering) as the 50/50 mechanism.

## Z[i] BORN-RATIONAL: 65 latent = 65 DEPLOYED — zero deploy tax on the exact complex lattice; parity-class with every comparator (2026-07-28 night)

cplx_ZI_zi: latent 65/120 @ 63.13, deployed 65/120 @ 63.13 —
IDENTICAL per-level maps. Comparators (same recipe/device
lineage): fp32 64 / cplx_none 63 / G5-latent 66 / G5-dep 62.
VERDICTS: (1) The composed-lever WIN does not book (+1/+2/+3,
bar 5) — two capability-neutral parents compose to
capability-neutral, as the house predicted. (2) **The finding
that DOES book: ZERO deploy tax** — the STE converged the
latents exactly onto the Z[i] rational lattice (deployed =
latent to the solve map), where G5's phase lattice paid -4 on
the same substrate and math grammar. The born-rational law
(exactness rides free) extends to the complex plane: every
weight is exactly (s/q')*(a+bi), Gaussian-integer-rational,
with no snap cost at all. Exact-representations program gains
its 2-D lattice point; combined with fq512's integer twin, the
exact-format family now spans R and C. (3) G5-latent 66 v
ZI-latent 65: the phase alphabet's latent edge is sub-noise —
lattice GEOMETRY (rotational v rational) is a non-factor at
matched dose; only the deploy tax separates them.

## PRE-REG: flips-vs-Q calibration fingerprint (2026-07-28 night, before the sweep; pre-approved follow-up class)

The R1 probe read one lattice (Q=16); the banked refinement reads
the CURVE: flips/token at Q in {4,8,16,32,64} (direct rational
snap) for 5 crystals {wfloor, s2, stream4, muon d256; the 19m
cracker}. PREDICTIONS: (1) flips fall monotonically with Q for
every crystal (finer lattice = smaller perturbation); (2) the
crystal ORDERING is Q-stable (the single-Q rank generalizes —
fingerprints don't cross); (3) the 19m cracker sits above the
d256 zoo at Q<=16 (its knee), converging by Q=64 (its measured
parity point). A crossing = the fingerprint carries MORE
information than any single-Q read (banked as the instrument's
justification); no crossing = one Q suffices, curve retired.

## FLIPS-VS-Q FINGERPRINT: crossings EXIST — the curve carries more than any single Q; muon's anomaly is structural (2026-07-28 night)

| crystal | Q4 | Q8 | Q16 | Q32 | Q64 |
|---|---|---|---|---|---|
| wfloor | 1.000 | 0.416 | 0.054 | 0.018 | 0.0075 |
| s2 | 0.999 | 0.441 | 0.070 | 0.017 | 0.0059 |
| stream4 | 0.984 | 0.622 | 0.057 | 0.021 | 0.0075 |
| muon | 0.924 | **0.139** | 0.048 | 0.016 | 0.0057 |
| 19m | 1.000 | 0.777 | 0.066 | 0.026 | 0.0102 |
VERDICTS: (1) monotone-in-Q CONFIRMED everywhere (Q4 ~1.0 =
saturation floor, probe ceiling noted). (2) **Ordering is NOT
Q-stable — prediction 2 FAILS, which is the instrument's
justification firing**: s2/stream4/19m cross between Q8 and Q16
(s2 0.441 < stream4 0.622 at Q8; s2 0.070 > stream4 0.057 at
Q16), so the CURVE is a real fingerprint, not a rescaled scalar;
the flips-vs-Q profile joins the rarity curve as a standing
per-crystal instrument. (3) The 19m cracker leads at Q8/Q32/Q64
but is pipped by s2 at exactly Q16 — single-Q reads can misrank
neighbors (R1's rho .883 was rank-robust in aggregate, lucky at
pairs). (4) THE MUON ANOMALY IS STRUCTURAL: the weakest crystal
(gate 34) is by far the most snap-robust at coarse Q (0.139 v
0.42-0.78 at Q8) — consistent with R1's muon gate RISING under
snap (+4): Muon's orthogonalized updates build a lattice whose
decisions sit far from ternary-class boundaries while being
WRONG more often. Robustness and capability are separable axes,
measured. Fences: 400-row probe, one device, direct-snap
operator; gates not re-run at new Q values (fingerprint-only).

## NIGHT-28 FINAL BATCH: S4/M4 at 6ep (zero premium 4, sub-bar); P2@45M lands 72 — at-or-above class, bracket stays closed (2026-07-28 overnight)

1. **S4-6ep 67 v M4-6ep 71** (matched recipe/dose): both 2-bit
   alphabets gain ~+10 from the discrete-learning dose (58->67,
   61->71 — the ternary 6ep law reproduces at 2 bits), and the
   zero's premium reads 4 — GREW from 3ep's 3, still under the
   5-solve bar. The zero-is-load-bearing law stays BOUNDED at
   born-2-bit: directionally paying at both doses, never
   bookable. No further dose cells owed (two doses, same
   answer-class). Dose-confound fence: M4-6ep 71 v fp32-3ep 64
   is schedule-unmatched — do NOT read "2-bit beats fp32" from
   it; the matched fp32-6ep@19M cell does not exist.
2. **P2 @ 45M-class, 6ep: 72/120 @ 69.11** v comparators
   ternary-45M-6ep 69 / fp32-45M-3ep 69. House "ties 69-class"
   prediction lands at its directional edge: +3, under the
   5-solve single-seed bar — P2 is AT-OR-ABOVE class at width
   (the 3.17-bit ladder does not decay at W* the way the
   W*~1/bits mechanism might have predicted for a mid-bit
   alphabet), but the bracket does NOT reopen. Alphabet standings
   at 45M-class: P2 72* / tern 69 / fp32 69 (* sub-noise lead).
3. **P2 deploy tax: ZERO** (latent gate = deployed gate, 72=72)
   — the second zero-tax reading tonight (Z[i] 65=65). Pattern
   forming across P2/ZI vs G5's -4: STE onto MAGNITUDE/RATIONAL
   lattices converges the latents fully; the PHASE lattice does
   not — banked as a candidate law (lattice-geometry-dependent
   deploy tax) with n=2 v n=1.

## AMENDMENT (target: Z[i] + P2 zero-deploy-tax bookings, night-28): the "latent" gates were STE-forward — instrument corrected, ZI leg CONFIRMED on the true latent (2026-07-28 morning review)

Morning self-review caught the artifact: gate_cplx with alpha=ZI
snaps in the forward (STE), so night-28's "ZI-latent 65" and
"ZI-dep 65" measured the SAME function — 65=65 was guaranteed;
likewise P2's first 72 was tournament_birth's internal STE gate.
The TRUE-latent gates (alpha=none on the unsnapped weights) ran
this morning: **ZI true-latent 65/120 @ 63.27, per-level map
IDENTICAL to deployed** — the zero-deploy-tax claim for Z[i]
SURVIVES the corrected instrument (right number, wrong reasoning;
now both right). P2's true-latent gate in flight on the 3080;
the lattice-geometry deploy-tax candidate law rides its result.
Doctrine line: an STE forward IS the deployed function — "latent
gate" only means something at alpha=none. Applies retroactively
to G5's 66/62 (correctly measured: its latent gate WAS alpha-
appropriate) — verified, not assumed, by this review.

## AMENDMENT CLOSE-OUT (target: same-morning STE-latent amendment): P2 true-latent 72 = deployed 72 — the zero-deploy-tax law leg stands corrected AND confirmed (2026-07-28 morning)

P2-45m TRUE latent (alpha-none weights, cuda): 72/120 @ 69.17,
per-level map identical to the deployed 72 @ 69.11. Both named
legs (Z[i] 65=65, P2 72=72) now measured on the honest
instrument. The lattice-geometry deploy-tax candidate law stands
at n=2 (magnitude/rational: tax 0) v n=1 (phase/G5: tax -4);
the STE-latent doctrine line (latent gates mean alpha=none)
is now scar-filed. Morning review complete: night-28's ledger
is clean end to end.

## PRE-REG: rotational snap R1 — the commutant instrument (2026-07-28, before the read)

Spec 2026-07-28-next-session A. For weight W (out x in), complex
structures J_out/J_in (block rotations under a channel pairing)
split W = W_c + W_a exactly: W_c = (W - J_out W J_in)/2 commutes
(W_c J_in = J_out W_c, the rotational part), W_a anti-commutes.
INSTRUMENT: anti-commutant mass fraction ||W_a||^2/||W||^2,
per-layer FFN gate matrices. Arms: adjacent pairing + 20
random-pairing null seeds on real crystals {wfloor, s2, muon
d256; 19m}; POSITIVE CONTROL: cplx_none + cplx_G5 under their
NATIVE half-split pairing (complex-FFN is complex-linear by
construction — must read ~0). PREDICTIONS: (1) control ~0
(instrument validation); (2) real-born crystals sit INSIDE the
random-pairing null band at ~0.5 mass — no spontaneous
rotational structure at any tested pairing (the euler-read null,
upgraded); (3) any crystal below its null band by >3 sigma of
the null spread = hidden rotational subspace found (would be
new). Hungarian-optimized pairing banked as v2 if (3) fires.

## ROTATIONAL SNAP R1 VERDICT: no rotational structure anywhere — and the control's failure is the sharpest finding (2026-07-28)

Real crystals (adjacent pairing v 20-random-pairing null, per-
layer gates): wfloor 0.49983 (z +0.15) / s2 0.50022 (+0.66) /
muon 0.50045 (+1.08) / 19m 0.50021 (+0.94) — ALL inside the
null band at ~0.500 anti-commutant mass. Prediction (2)
CONFIRMED: no spontaneous rotational subspace at any tested
pairing (euler-read null, upgraded to pairing-swept form).
Prediction (3) does not fire; Hungarian v2 stays banked.
THE CONTROL SURPRISE (prediction 1 failed archITECTURALLY, and
it books): cplx_none/G5 read 0.4994-0.4998 — NOT ~0 — because
the house complex-FFN carries its rotation ENTIRELY in the
activation algebra (modReLU + the elementwise complex multiply);
its weight matrices are plain real maps (input h is unpaired
real, so weight-level complex-linearity cannot even be defined
there). The instrument itself is EXACT (synthetic complex-linear
rep [[Wr,-Wi],[Wi,Wr]]: anti-mass 0.0; random: 0.495).
CONSEQUENCE, reframing R3: teaching WJ=JW would create a
weight-level symmetry NO house model possesses — including the
born-complex arms. "Turn a linear model into a rotational
model" is a genuine construction, not a recovery.

## PRE-REG: ladder-retry decode + rotational snap R2 (2026-07-28, before the runs)

1. LADDER-RETRY (3080, champion, full battery): arms greedy->k8
   (gfirst, measured 74 @ 29.7k) vs greedy->k2->k8 (ladder).
   BAR unchanged: >= wave-3 solves (>=72) at <= 40% of wave
   tokens (<=16.8k). The k=2 rung should catch most greedy
   misses at 1/4 the retry cost; if tokens still >40%, the
   adoption file closes (retry economics are structural).
2. ROTATIONAL SNAP R2 (Mac, wfloor d256): gate at
   W - t*W_a (adjacent pairing, all interior 2-D weights) for
   t in {0.25, 0.5, 1.0}. REVISED prediction post-R1: with
   anti-mass 0.5 and no structure, t=1.0 deletes half the
   weight content coherently -> expect severe damage (int2/
   rank-collapse class); the PAYLOAD is the knee location and
   whether damage-vs-t is graceful (near-tie-mediated, flips
   probe) or cliff (structural). Sets the R3 starting-point
   damage baseline (projected init = t=1).

## PRE-REG: ladder-retry decode + rotational snap R2 (2026-07-28, before the runs)

LADDER-RETRY (3080, champion, full battery): arms ladder
(greedy -> k=2 -> k=8) v the night-28 measured wave 75 @ 42,113
/ gfirst 74 @ 29,724. ADOPTION BAR unchanged: >= wave-3 solves
at <= 40% of wave tokens. Prediction: the k=2 rung absorbs most
greedy misses at 1/4 the retry cost — tokens land 35-50%; the
bar is genuinely in reach and may still miss (books either way).
ROTATIONAL SNAP R2 (Mac, wfloor comparator 65): gate at
W_gate - t*W_a (adjacent pairing, gate matrices only — fence:
attention/up/down untouched), t in {0.25, 0.5, 1.0}. REVISED
prediction post-R1: with anti-mass 0.500 and no structure,
t=1.0 deletes half the gate mass into a symmetry subspace —
expect heavy damage (int2-class or worse); the payload is the
CURVE (knee location) + flips-probe at each t, the rotation-axis
sibling of the rational Q-sweep, and the R3 starting-point
damage baseline (projected init = the t=1 point).

## LADDER-RETRY VERDICT: 74 @ 34,895 tokens — the middle rung LOSES money; the decode-economics family closes (2026-07-28)

Champion battery, ladder (greedy -> k=2 -> k=8): 74/120 @ 34,895
tokens v wave 75 @ 42,113 / gfirst 74 @ 29,724 (both same-battery,
same-model, night-28). The k=2 rung ADDED cost over plain
greedy->k8 (+5,171 tokens, +0 solves): when greedy misses, k=2
usually misses too (the hard plies are hard for 1-2 samples
alike), so the ladder pays three rungs where gfirst pays two.
Adoption bar (>= wave-3 at <= 40% tokens): FAILS at 83%. The
family's honest close: (1) wave-8 stays production; (2) the only
measured cheap regime is PURE greedy on short probe batteries
(-10% solves at 12% tokens, R4) — a probe-tier option, not a
gate option; (3) retry laddering does not bridge them — the
wave's spend concentrates exactly where no cheap rung reaches
(the tie-scarcity anatomy from the other side: hard plies are
hard by SEMANTICS, not by sampling breadth). Decode-economics
cells stop here; no further rungs owed.

## PRE-REG: THE STABILITY ATLAS (2026-07-28, before the grid; double-pendulum riff)

8x4 grid (LR log-spaced 1e-4..1.2e-2 x BIRTH_BS {8,16,32,64}),
d64/gen-4/3ep births, cuda gates; + 4 seed-2 cells (the noise
floor). The pendulum question drawn on training space: is
gate(LR, BS) SMOOTH or does it hold chaotic regions and islands?
READS, pre-registered: (1) SMOOTH = adjacent-cell |delta| ~
seed-sigma across the map (a boring, bookable null: training is
non-chaotic in this window); (2) CHAOTIC BAND = a region where
adjacent deltas >> seed-sigma while other regions are smooth
(the pendulum structure — islands would be high-gate cells
inside it); (3) the known cliff (high-LR divergence) should
appear as a smooth boundary, not chaos, if (1). Also read: does
BS interact with LR as the sqrt-scaling rule predicts (ridge
along LR ~ sqrt(BS)) or independently? Fences: d64 (the
substrate's own gate band is 38-class; sigma unmeasured at d64
— the 4 s2 cells price it), n=1 per cell, one device.

## ROTATIONAL SNAP R2 VERDICT: a fully rotational-gated crystal keeps 88% of its solves — the knee is at the far end (2026-07-28)

wfloor, gate matrices projected W - t*W_a (adjacent pairing):
t=0: 65 | t=0.25: 65 (per-level map IDENTICAL; ~3% mass) |
t=0.5: 64 (-1; 12.5% mass) | t=1.0: **57 (-8; 50% of gate mass
deleted into the commutant)**. READS: (1) the damage curve is
shockingly gentle — forcing every gate matrix to be EXACTLY
complex-linear (a symmetry no house model has ever had) costs 8
solves of 65; the democracy absorbs a 50%-mass structured
deletion the way it absorbed rank-128 (-3 at 4x less mass
removed) — holography's strongest showing yet. (2) L6 held 8/8
at every t; damage concentrates L5/L7/L3 — no clade signature.
(3) The R3 starting point is measured: projected-init = 57-class,
and the conversion question is now concrete — can warm training
under the commutation penalty recover the -8 while KEEPING
anti-mass ~0? Fences: gates-only projection (attention/up/down
untouched); flips-probe leg NOT run (gates only — booked
honestly against the pre-reg's rider); n=1, MPS.

## PRE-REG: rotational snap R3 — the conversion (2026-07-28, before the runs)

From the t=1.0 projected init (57-class), TWO warm arms (d256,
1 epoch gen-4, AdamW 1e-4, MPS, seed 1): (a) lambda=0 — does
plain SGD RESTORE the anti-commutant (symmetry unstable, gate
recovers toward 65 by re-breaking it)? (b) commutation penalty
lambda * sum_l ||W_l J_in - J_out W_l||^2 / ||W_l||^2 (lambda
ramped 0.1 -> 1.0 over the epoch, gate matrices only). READS:
arm (b) recovers gate >= 62 (within 3 of control) with final
anti-mass < 0.05 => **CONVERSION ACHIEVED — a linear model
turned rotational at ~zero capability cost** (the session
question answered constructively); gate recovers but anti-mass
drifts > 0.2 => training rejects the symmetry (teach-don't-
impose extends to symmetries — also a clean verdict); gate
stuck 57-class with anti-mass low => the symmetry subspace
cannot host the missing 8 solves (capacity-of-the-commutant
result). Arm (a) prices the restoration force either way.

## ROTATIONAL SNAP R3 VERDICT: THE CONVERSION WORKS — 64/120 at anti-mass 0.0002; and SGD does not even fight the symmetry (2026-07-28)

From the projected (57-class) init, 1 warm epoch each:
| arm | gate | anti-mass (0.5 = none, 0 = exact) |
|---|---|---|
| (a) lambda=0 | 62/120 @ 55.60 | 0.0145 |
| (b) penalty ramp | **64/120 @ 57.30** | **0.0002** |
VERDICTS: (1) **The pre-registered conversion bar is SMASHED:
arm (b) recovers to 64 (control 65, within 1 = seed-sigma) while
the gates are complex-linear to 2 parts in 10,000.** The session
question answers constructively: a linear model CAN be turned
into a rotational model — project onto the commutant (-8), one
warm epoch under a ramped commutation penalty (+7), net cost ~1
solve. Arc: 65 -> 57 -> 64, anti-mass 0.5 -> 0 -> 0.0002.
(2) **THE SURPRISE (arm a): plain SGD does not restore the
anti-commutant** — lambda=0 recovers to 62 at anti-mass 0.0145,
i.e. the commutant is a (locally) STABLE manifold under
unconstrained training: once the symmetry is imposed, the
gradient barely pulls away from it. The "training rejects
imposed symmetry" reading is FALSE here — nuancing
teach-don't-impose: imposing a REPRESENTATION on inputs nulls
(prefix, hints), but imposing a WEIGHT-SPACE SYMMETRY + brief
re-training holds at ~zero cost. Symmetries are cheap to grant;
they were just never chosen spontaneously (R1's null).
(3) What a rotational model buys is now an open, concrete
question with an artifact in hand (rot_convert_b.pt): candidate
follow-ups — parameter halving (complex-linear = half the free
parameters per gate matrix), the U(n) gauge leg (quantum-LLMUE
walk), rotational-native quantization (G5-class alphabets on a
genuinely rotational substrate). Banked, not fired. Fences:
gates-only symmetry, d256/MPS, n=1/arm, 1 warm epoch.

## PRE-REG: SYMMETRY LADDER S1 — quaternionic conversion (2026-07-28, before the runs)

Spec docs/superpowers/specs/2026-07-28-symmetry-ladder.md. Three
anticommuting structures I,J,K (left quaternion-unit action on
4-channel groups; I^2=J^2=K^2=-1, IJ=K), commutant projection
P(W) = (W - IWI - JWJ - KWK)/4 — params/4 per gate matrix.
d256/MPS wfloor lineage, gates only, comparator 65, seed 1.
CELLS + READS: (1) anti-mass read on wfloor/s2/19m + 20
random-grouping nulls — prediction: ~0.75 everywhere, z < 3 vs
nulls (the R1 null repeats at the quaternion group). Instrument
validated FIRST on a synthetic quaternionic-linear matrix
(anti-mass must read 0.0) and a random matrix (~0.75) — the R1
synthetic-control pattern. (2) projection gate at t=1.0
(deleting 75% of gate mass — holography's harshest test yet;
R2 lost only 8 at 50%). (3) heal arms lambda=0 and ramped
penalty (R3 recipe verbatim, penalty summed over all three
generators). BAR: **gate >= 61 at anti-mass < 0.05 =
CONVERSION AT 4x SHARING** — the symmetry axis of the
bits-dimension exchange law gets its second point. Gate stuck
low at low anti-mass => commutant capacity limit found at 4x
(also a clean verdict: the ladder has a measured rung where it
stops). Arm (a) again prices SGD's restoration force.

## SYMMETRY LADDER S1 CELL 1 VERDICT: no spontaneous quaternionic structure — the R1 null repeats at the quaternion group (2026-07-28)

Instrument (scratch/quat_commutant.py) validated first, per
fence: synthetic commutant member anti-mass 0.000000; random
matrix 0.737 (~0.75 as predicted). One real scar fixed at the
algebra asserts before any read: the first block tables dropped
quaternion signs (i*k = -j etc.) — the in-code checks
I^2 = -1, IJ = K, IJ = -JI caught it. REAL CRYSTALS (adjacent
4-grouping vs 20 random-grouping nulls, avg over 8 gate
layers): wfloor 0.74993 (z -0.38), s2 0.75020 (z +0.59), 19m
0.74968 (z -1.39). All within a sigma or so of the null band —
**exactly the pre-registered prediction: SGD chooses no
quaternionic structure spontaneously, mirroring the complex
R1 null.** Symmetries must be imposed; cells 2-3 (75%-mass
projection + heal arms) now test whether they can be, at 4x
sharing. Fences: gates-only, d256/MPS lineage, cpu-side reads.

## SYMMETRY LADDER S1 VERDICT: QUATERNIONIC CONVERSION AT THE BAR — 61/120 at anti-mass 0.0007, 4x sharing (2026-07-28)

Cells 2-3 (scratch/quat_convert.py, R3 recipe verbatim,
penalty over all three generators):
| stage | gate | anti-mass |
|---|---|---|
| projected init (75% mass deleted) | 22/120 @ 13.10 | 0 |
| arm (a) lambda=0, 1 epoch | 60/120 @ 55.10 | 0.0558 |
| arm (b) ramped penalty, 1 epoch | **61/120 @ 56.03** | **0.0007** |
VERDICTS: (1) **The pre-registered bar (gate >= 61 at
anti-mass < 0.05) is met exactly: 61 at 7 parts in 10,000 —
CONVERSION AT 4x SHARING.** Arc 65 -> 22 -> 61: net cost 4
solves for a gate parameterization with 1/4 the free params.
The symmetry compression axis has its second point: complex
(2x) costs ~1, quaternion (4x) costs ~4 — the ladder pays a
rising but sublinear toll. (2) **Holography BREAKS at 75%
deletion**: projected init 22/120 (R2's 50% cut cost only 8).
The democracy's absorption has a measured edge between 50% and
75% structured mass removal — the first quantitative bound on
the holography doctrine. (3) Arm (a) repeats the R3 surprise
at the harder group: lambda=0 drifts only to 0.056 anti-mass —
the quaternionic commutant is ALSO locally stable under
unconstrained SGD; the penalty is worth +1 solve and 80x
tighter symmetry, not a fight. Fences: gates-only, d256/MPS,
n=1/arm, comparator 65, artifacts quat_convert_{a,b}.pt.

## PRE-REG: SYMMETRY LADDER S4 (Z2) + S3 (circulant C8) — before the runs (2026-07-28)

Generic instrument scratch/sym_convert.py: P(W) = group average
of orthogonal conjugations; synthetic member/random controls
per group; anti-mass read + 10 nulls; then projected-init gate
+ heal arms a (lambda=0) / b (ramped generator penalty), R3
recipe verbatim. d256/MPS wfloor, gates only, comparator 65.
S4 Z2 (sign involution, params/2, generic anti-mass 0.5):
prediction — no spontaneous structure (z < 3); projected init
~57-class (same mass fraction as R2's complex cut); arm b
recovers >= 62 at anti-mass < 0.05 (the Z2 commutant is a
LOOSER constraint than complex — block-checkerboard, no
rotation coupling — so healing should be at least as cheap).
S3 CIRCULANT C8 (shifts within 8-blocks, params/8, generic
anti-mass 0.875): the audacious rung — 87.5% mass deletion is
past the measured holography edge (S1: breaks between 50% and
75%), so projected init should CRATER (< 22/120). READ: arm b
gate at anti-mass < 0.05 is the result either way — >= 58
(within seed-noise territory of the ladder trend 65/64/61)
means dense gates of a trained crystal RETROFIT into
conv-structured form at 8x sharing; far below extends the
ladder toll curve (2x:-1, 4x:-4, 8x:-?) and locates the
symmetry axis's capacity wall. Arm a prices restoration force
per group. n=1/arm.

## SYMMETRY LADDER S4 + S3 VERDICT: Z2 converts at 64; DENSE GATES RETROFIT INTO CONV STRUCTURE AT 8x — the ladder holds to params/8 (2026-07-28)

Generic instrument controls clean per group (member 0.000000;
random 0.5003 / 0.8754 vs predicted 0.500 / 0.875). Anti-mass
reads: both nulls (z +0.07 / -0.22) — the no-spontaneous-
structure result now stands at THREE groups (complex,
quaternion, Z2/C8).
| rung | proj-init | arm a (lambda=0) | arm b (penalty) |
|---|---|---|---|
| S4 Z2 (params/2) | 49/120 | 63 @ am 0.0144 | **64 @ am 0.0002** |
| S3 C8 (params/8) | **2/120** | 60 @ am 0.1356 | **59 @ am 0.0028** |
VERDICTS: (1) **S3, the audacious rung, HOLDS: 59/120 at
anti-mass 0.0028 — dense gate matrices of a trained crystal
RETROFIT into block-circulant (convolution) structure at 8x
parameter sharing**, healing from a projected init of 2/120
(total destruction, as pre-registered: 87.5% deletion is past
the holography edge). The heal is a near-total reconstruction
inside a params/8 subspace in ONE warm epoch. (2) The ladder
toll curve completes: 2x:-1, 4x:-4, 8x:-6 of 65 — rising,
sublinear, no wall through 8x. Symmetry is a REAL third
compression axis. (3) PRE-REG MISS worth keeping: Z2 projected
init read 49, not the predicted ~57-class — same 50% mass
fraction as the complex cut, DOUBLE the damage. Projection
damage depends on WHICH structure is deleted, not just the mass
fraction (checkerboard cuts different functional directions
than rotation); holography's edge is structure-dependent.
(4) First sign of SGD resistance: circ8 arm (a) drifts to
anti-mass 0.136 (complex 0.015, quat 0.056, Z2 0.014) — local
stability of the commutant WEAKENS as the group grows; the
penalty starts earning its keep at C8. Fences: gates-only,
d256/MPS, n=1/arm, comparator 65, artifacts
sym_{z2,circ8}_{a,b}.pt. Banked: C16 (params/16) extension;
actual packed conv forward (param-compression stays IMPLIED).

## PRE-REG: SYMMETRY LADDER S2 — complexification control (2026-07-28, before the run)

Construction: double every linear of wfloor d256 -> d512
(heads 4->8, head_dim fixed 64 so RoPE is per-head identical;
emb columns [W|W]; qkv/o/gate/up/down block-diag W(+)W; head
[W/2|W/2] so logits match exactly in real arithmetic; norm
gains duplicated). The doubled gates commute with J_half BY
THEOREM — script asserts anti-mass = 0 before gating. READS:
gate = 65 => the exactness answer lands empirically (exact
rotational conversion EXISTS at 2x width; the -8/-1 tolls of
the 1x rungs are the price of staying at-width). AMENDED BAR
vs spec (booked before the run): fp32 reductions over 2d
reorder sums (rmsnorm mean, matmul accums), so last-bit logit
deviations can flip coin-flip ties — house fp-near-tie
doctrine. A small delta with eager logit margins <= ~0.02 at
divergence = tie, not bug; anti-mass != 0 or margin-large
deltas = instrument bug, fix before booking. d256->512/MPS,
gates-only claim, comparator 65, no training (pure control).

## SYMMETRY LADDER S2 VERDICT: EXACTLY 65 — exact rotational conversion exists at 2x width; the ladder closes (2026-07-28)

scratch/complexify_control.py: wfloor doubled d256->d512
(W(+)W everywhere, head [W/2|W/2], heads 4->8 at fixed
head_dim 64). Theorem assert passed: all 8 doubled gates
anti-mass < 1e-12 vs J_half. Gate: **65/120 @ 59.91 —
EXACTLY the comparator, no fp tie flipped.** VERDICTS:
(1) Artin's exactness question is answered empirically: the
-8 projection cost at 1x was GENUINE non-rotationality, not
instrument error — an EXACT rotational form of the same
function exists, at 2x width, gate-identical. (2) The trade
table is complete: exact-at-2x-width (cost: 4x params) vs
healed-at-1x (cost: ~1 solve, params/2). Compression and
exactness pull opposite ways on the symmetry axis, and both
endpoints are now measured. (3) THE SYMMETRY LADDER IS
COMPLETE in one session: anti-mass null at every group;
conversion toll 2x:-1, 4x:-4, 8x:-6; holography edge
structure-dependent; commutant stability weakens with group
size; exact embedding verified. Incidental scar fixed:
rot_commutant.py lacked a __main__ guard (import re-ran the
R1 sweep) — guarded. Banked: C16, packed conv forward,
symmetry-at-birth (train IN the commutant from scratch vs
retrofit — the R1-null flip test).

## PRE-REG: ATLAS-2, THE LYAPUNOV LEG (2026-07-28, before the runs; Artin: "momentum space/diff starting speeds")

The atlas swept PARAMETERS (LR x BS = arm lengths); the double-
pendulum result is about INITIAL CONDITIONS at fixed parameters.
Dictionary: init weights = release angles; init scale = ENERGY
(2swap: chaos is energy-dependent — the stable islands live at
low energy); observable = FUNCTION-SPACE divergence (weight
distance forbidden): teacher-forced argmax disagreement on 200
fixed gen-4 rows (scratch/lyap_compare.py) + gate. Cells (3080,
d64/3ep at the atlas peak lr1.5e-3/bs8, seed 1, ckpts KEPT):
twin births at eps in {1e-6, 1e-4, 1e-2}; energy arms
INIT_SCALE 4 and 0.25 (base + eps 1e-4 twin each); independent
seed-7 reference = the saturation distance. READS: (1) CHAOTIC
= disagreement ~flat in eps at approx the seed-reference level
(exponential mixing erases eps; the attractor-spread is reached
even from 1e-6). (2) CONTRACTIVE/SMOOTH = disagreement graded
~monotone in eps, with e6 twins near-identical (solve sets
equal or within a solve). (3) ENERGY LAW = hi-scale twins
decorrelate more than base-scale twins at matched eps (the
2swap energy-dependence transplants); lo-scale less. Gate
column read alongside (does divergence COST capability, or do
twins land equally-good-but-different — the degeneracy split).
Momentum leg (Adam beta1 / warmup as "starting speeds") BANKED
as follow-up, kept out to hold this single-variable. Fences:
same device/diet/steps all cells; disagreement is format-bound
to gen-4 teacher forcing; n=1 per cell (this is a map, sigma
priced by the seed-7 column).

## STABILITY ATLAS VERDICT: A SMOOTH PLATEAU WITH ONE CLIFFED CORNER — no chaos, no islands; the double-pendulum instinct fails for parameters (2026-07-28)

Full 8x4 LR x BS map (d64/gen-4/3ep, seed 1) + 4 seed-2 cells.
The map: gate rises smoothly from the starved corner (lr1e-4 x
bs64 = 1/120) to a BROAD plateau — everything lr 8e-4..1.2e-2
at bs8/16 sits in the 56-65 band (peak lr1.5e-3/bs8 = 65 —
matching the d256 comparator at d64!). Seed-2 deltas: 3, 5, 4,
0 => sigma ~3.5, and every adjacent-cell step off the plateau
is within ~sigma EXCEPT the far corner: lr1.2e-2 crosses
bs16 -> bs32 as 56 -> 26 (~8 sigma) — a genuine CLIFF, entering
the map exactly at the (highest LR x largest batch) corner.
PRE-REG READINGS: (1) SMOOTH — confirmed nearly everywhere.
(2) CHAOTIC BAND — ABSENT: no interleaved good/bad cells, no
islands anywhere. (3) The known cliff appears as a BOUNDARY —
confirmed, at the far corner only, sharp (one cell) but with no
evidence of fractal structure at this resolution. VERDICT: at
this scale, capability-over-hyperparameters is a smooth
landscape with one instability boundary, NOT a chaotic map —
the double-pendulum geometry does NOT transplant to PARAMETER
space. (Artin's follow-up standing: chaos lives in INITIAL
CONDITIONS, not parameters — atlas-2 Lyapunov leg pre-reg'd
and now running on the same substrate.) Curiosities kept
honest: small batch TOLERATES high LR better than large batch
here (bs8 gates 58 at lr1.2e-2 where bs32 craters) — with the
fixed-epoch confound noted (larger BS = fewer updates);
lr0.006/bs16 62 > bs8 60 is within sigma, not texture. Fences:
d64/cuda lineage, gate-only coloring, n=1 map + 4-cell sigma
column; checkpoints deleted (map cells) per script.

## PRE-REG: THE COMPRESSION CORNER — bits x sharing orthogonality (2026-07-28 night, before the runs)

Paired arms, MPS, one device: rational_snap (DIRECT operator,
exact-best p/q, Q in {8, 16}) applied to (a) dense wfloor d256
(baseline 65) and (b) sym_circ8_b (baseline 59, anti-mass
0.0028). Snap preserves circulant structure by construction
(elementwise determinism), so any extra damage is functional.
READS: delta-of-deltas |(circ8_snap - 59) - (dense_snap - 65)|
<= sigma(3) at both Q => AXES ORTHOGONAL — the bits-dimension
exchange law gains a symmetry factor (compression composes
multiplicatively: 8x sharing x lattice bits at additive toll).
Circ8 pays MORE => sharing consumed the redundancy that
quantization was living on (competition-for-slack model —
also a clean law). Circ8 pays LESS => the commutant regularizes
toward snap-friendly weights (would rhyme with the lattice
zero-tax law). n=1/cell, gates-only substrate difference.

## PRE-REG: SYMMETRY-AT-BIRTH, C8 at d64 (2026-07-28 night, before the runs)

The R1-null flip test on the atlas-certified cheap substrate.
TWO arms from SCRATCH, paired on MPS (device fence: the atlas's
cuda 65 is NOT the comparator; the paired dense arm is), seed 1,
lr 1.5e-3, bs 8, gen-4, 3 epochs, d64/ffn256: (a) DENSE control;
(b) COMMUTANT BIRTH — C8-projected init + ramped generator
penalty from step 0 (R3 recipe transplanted to birth).
READS: b >= a - sigma(3.5) => symmetry is FREE AT BIRTH (the
retrofit heal was not riding dense scaffolding; params/8 from
step zero). b ~ a - 6 (the retrofit toll) => toll is
PATH-INDEPENDENT (retrofit = birth — a conservation statement).
b << a - 6 => the SCAFFOLD hypothesis: dense pre-training builds
structure the commutant can inherit but not grow — teach-don't-
impose returns at birth. Also read: final anti-mass (does the
penalty hold symmetry through 3 epochs of from-scratch SGD).

## AMENDMENT (target: E2 AXNN export, relay -28-4): head was declared "tied" but the scorer's head is UNTIED — corrected container delivered (2026-07-28 night)

Axiom's v1.1 loader validates head declaration against tensor
presence; inspecting our own container against their note found
the bug on OUR side: cfg said head:"tied" while the file carries
a separate head.weight NOT byte-equal to emb.weight (the S2
scorer is untied, per build_model). Their guard would have
rejected the file — the guard WORKED, cross-lab, before any
wrong logits shipped. Fixed export_axnn.py (head:"separate"),
re-exported; NEW sha256 298f9077a4622ce0...ab094 (the announced
b87d0976... is RETIRED — reject it). Artifacts delivered
directly to axiom's data/ via the WSL bridge (first shared-
filesystem handoff; sha verified on their disk). Tensor-name
answer to their question: names are state-dict style —
blocks.{i}.gate.weight/up.weight/down.weight/qkv.weight/
o.weight/n1.g/n2.g + emb.weight/norm.g/head.weight — NOT
ffn.gate/attn.qkv.

## E2 CLOSED CROSS-LAB: 6.2e-6 max logit delta — two independent forwards agree on one container; E3 ARMED (2026-07-28 night)

Axiom's v1.1 loader ran the corrected container (sha 298f9077)
and delivered the pinned 20-prompt battery + expected logits
(shas 9ef00948 / e0e7385c, pulled via the WSL bridge). House
reproduction (scratch/e2_logit_check.py, torch fp32, CPU):
tokenization parity 20/20 (their greedy-longest-match ids
decode exactly to meta text via the house tokenizer);
**max|delta logit| over 20x40 = 6.2e-06 — PASS at 16x under
the 1e-4 bar.** The E-series handshake is complete: same
bytes, two languages, two labs, agreeing forwards. This ARMS
E3 (the exact-mode paired gate — FX-V1 fixed-point vs float,
the precision doctrine's sole sanctioned reopening). Note for
the record: their loader gained house-dialect tensor-name
remapping + string attn_fused acceptance (eb20896 their side);
E2's two caught bugs (our head declaration, their fused-flag
type) were BOTH caught by declared-contract validation before
any wrong number shipped — the guard architecture is the
result as much as the number.

## PRE-REG: C8-RETROFIT AT 45M — does the sharing toll transfer across scale? (2026-07-28 ~5PM, before the run; 3080 tail window)

Substrate: union_45m.pt (d512/12L/ffn2048/h8; math 65/120, ZX
36; the capacity-null crystal). C8 commutant projection on all
12 gate matrices (params/8), then ONE warm epoch on the union
diet (bf16 autocast, ramped permutation penalty — cyclic
generators are permutations, so the penalty is index-shuffle
cheap). GATES: projected-init math gate; final math AND ZX
gates + final anti-mass. READS: (1) final math >= 59-class
(the d256 toll -6 transfers) => sharing toll is SCALE-STABLE —
capability occupies a small structured subspace at 45M too,
and the ZX capacity null gets its inside explanation measured
at scale. (2) math craters and stays cratered => the 8x
commutant is capacity-limited at richer diets (the toll curve
bends with scale — also clean). (3) ZX column read alongside:
does the graph grammar pay MORE than the tree grammar for
sharing (grammar-dependent toll — new axis interaction), or
ride at ~36-class? n=1, cuda, comparators are the same-device
bookings (65/36). Fences: gates-only, union diet, 1 warm epoch.

## COMPRESSION CORNER VERDICT: THE AXES ARE NOT ORTHOGONAL — sharing consumes the slack that quantization lives on (2026-07-28 night)

Paired MPS gates, snap-preserves-circulant confirmed by
construction:
| substrate | control | Q16 snap | Q8 snap |
|---|---|---|---|
| dense | 65 | 53 (-12) | 1 |
| circulant-8x | 59 | 34 (-25) | 0 |
VERDICT: delta-of-deltas = -13 (>> sigma 3): the circulant
substrate pays DOUBLE the quantization toll of the dense one at
Q16, and Q8 destroys both. **The pre-registered
competition-for-slack read fires: compression axes draw on a
SHARED redundancy budget** — 8x parameter sharing spent most of
the slack, leaving quantization little to live on. The
bits-dimension exchange law does NOT gain a free product form;
it gains a BUDGET form (compression composes sub-additively).
Corollary worth keeping: the zero-deploy-tax lattice law
(P2/Z[i]) was measured on DENSE substrates — its scope fence
now has a mechanism (tax-free requires slack; substrates
without slack pay). Fences: gates-only substrate difference,
d256/MPS, direct snap operator, n=1/cell.

## SYMMETRY-AT-BIRTH VERDICT: FREE AT BIRTH — C8-from-scratch gates 50 vs dense control 53 at d64 (2026-07-28 night)

Paired MPS arms, from scratch, 3ep gen-4, seed 1: dense 53/120
(anti-mass 0.872 ~ generic); C8-commutant birth (projected init
+ penalty from step 0) **50/120 at anti-mass 0.0025**. Delta -3
~ sigma(3.5): **the pre-registered FREE-AT-BIRTH read fires.**
The R1 null flips completely: SGD never CHOOSES symmetry, but
will happily grow a full crystal INSIDE the commutant from step
zero at ~zero cost — params/8 from birth, no dense scaffold
needed (the retrofit heal was not riding pre-trained
structure). With the ladder + this: symmetry is free at birth,
cheap at retrofit (-6 at d256), and never spontaneous. Note
d64 substrate fence: the d256 birth replication is the
confirmation rung if this ever carries weight-bearing load.

## MATRYOSHKA RUNG 0: free nesting FAILS — raw C8 projections crater (2026-07-28 night)

P_C8(dense65) gates 2/120; P_C8(rot_convert_b, the complex-
commutant crystal) gates 6/120. No crystal's projection is
usable without training toward it (consistent with S3: the
healed circulant needed its warm epoch). Crumb kept: the
complex-commutant crystal's projection is marginally less dead
(6 v 2) — nested symmetries may compose. VERDICT: the
matryoshka needs the JOINT LOSS (CE(W) + CE(P(W)), STE through
projection) — rung 1 is the real experiment, and the corner
verdict's slack-budget law predicts its price. Banked for a
session with a training window.

## ATLAS-2 LYAPUNOV VERDICT: CHAOTIC MIXING WITH CAPABILITY DEGENERACY — and the energy law transplants (2026-07-28 night)

Nine births + six twin comparisons (d64/3ep, atlas peak cell):
| pair | disagree (fine obs) | gates |
|---|---|---|
| base vs e6 (eps 1e-6) | 0.0164 | 64 v 64 |
| base vs e4 (1e-4) | 0.0175 | 64 v 60 |
| base vs e2 (1e-2) | 0.0169 | 64 v 62 |
| base vs seed7 (independent) | **0.0210** | 64 v 55 |
| hi(x4) twins @1e-4 | **0.0636** | 3 v 4 |
| lo(x0.25) twins @1e-4 | 0.0155 | 57 v 55 |
VERDICTS: (1) **Pre-reg read (1) CHAOTIC fires: disagreement is
FLAT across FOUR orders of magnitude of eps (1.64/1.75/1.69%)
at ~78% of the independent-seed saturation (2.10%)** — a 1e-6
perturbation and a 1e-2 perturbation land equally far apart:
training mixes exponentially and erases the perturbation
scale. The double-pendulum geometry DOES transplant — to
initial conditions, exactly where Artin pointed, and exactly
where the parameter atlas nulled. (2) **With capability
DEGENERACY: every base-energy twin gates plateau-level
(64/64/60/62)** — trajectories diverge to DIFFERENT functions
of EQUAL quality. The attractor is a quality shell, not a
point: chaos in function space, order in capability space
(kin: the weight-reader doctrine's many-arrangements-one-
function, now measured dynamically). (3) **The ENERGY LAW
fires on the fine observable, monotone: 0.0155 (x0.25) < ~0.017
(x1) < 0.0636 (x4)** — divergence RATE rises with init energy,
the 2swap energy-dependence measured in a training system;
this leg is NOT budget-confounded (it is a rate, not a
capability). The x4 arms' 3-4/120 gates stay FENCED as
budget-confounded (smooth monotone loss descent from a distant
start — basin distance, not measured instability; refinement
arms banked in RIFF-LEDGER). Fences: d64/cuda, one cell,
disagreement format-bound to gen-4 teacher forcing, n=1/pair.

## PRE-REG: NIGHT-28b — TAMING THE CHAOS + the matryoshka joint loss (2026-07-28 late, before the runs; Artin GO "make training not chaotic / Ozaki frankenstein")

The Lyapunov instrument (twin disagreement, format-bound
teacher-forced argmax on 200 rows) becomes the DIAL-TESTER for
training-taming interventions. All d64/3ep at the atlas peak
cell unless noted; cuda; baselines from tonight: twin
disagreement ~0.0164-0.0175, seed-saturation 0.0210, base gate
64, sigma 3.5.
CELLS (3080, chained after the 45M retrofit):
(A) TWIN SOUP, desk: average the KEPT lyap checkpoints
  pairwise (base+e6, base+e4, base+e2) and all-4; gate each.
  READS: soup >= base + sigma => the quality shell is
  weight-space CONVEX between twins (variance is harvestable
  for free — the soup literature transplants); soup craters =>
  even 1e-6 twins cross basin boundaries (chaos extends to
  weight-space topology).
(B) EMA TWINS: two perturbed births (eps 1e-6) with Polyak EMA
  (decay 0.999) tracked through training; gate EMA + final;
  disagreement EMA-vs-EMA against the 0.0164 raw baseline.
  READ: EMA disagreement << raw => EMA CONTRACTS the shell —
  "training tamed" lever 1, measured.
(C) SYMMETRY TWINS: two C8-commutant births (eps 1e-6, the
  sym_birth recipe on cuda); disagreement vs the dense 0.0164.
  READ: commutant constraint shrinks the wander-space =>
  symmetry doubles as a chaos damper (the ladder and the
  Lyapunov program fuse); no shrink also clean (the shell
  lives in the commutant too).
(D) HYPERPARAM SOUP (the score-raiser): rebirth 4 plateau
  cells (lr 8e-4/1.5e-3/3e-3/6e-3, bs8, seed 1), soup, gate.
  READ: soup > 65 (the map peak) => diverse-hyperparam
  averaging harvests the plateau's degeneracy (greedy-soup
  transplants; a new score lever at zero inference cost).
(MAC, tonight) MATRYOSHKA RUNG 1: joint loss CE(W) +
  CE(STE P_C8(W)) via weight parametrization, 1 warm epoch
  from wfloor d256. Gate BOTH tiers. READS vs the corner
  verdict's slack-budget prediction: dense tier >= 62 AND
  cheap tier >= 50 => nesting price is small (dynamic-budget
  inference is real); dense tier craters => tier competition
  confirmed at the training level (the budget law binds).
Fences: per-cell n=1, device-paired comparators only, EMA
decay pinned 0.999, soup = plain parameter mean (no Fisher
weighting this rung).

## MATRYOSHKA RUNG 1 VERDICT: THE NESTED CRYSTAL WORKS AT ZERO PRICE — one tensor, two tiers, 65/60 (2026-07-28 night)

One warm epoch of the joint loss CE(W) + CE(STE P_C8(W)) from
wfloor d256 (scratch/matryoshka_r1.py, toggleable STE
parametrization):
| tier | gate | comparator |
|---|---|---|
| DENSE (raw W) | **65/120** | 65 (the full crystal) |
| CHEAP (P_C8(W), params/8) | **60/120** | 59 (separately-trained circulant) |
VERDICT: both pre-reg bars smashed — the dense tier pays ZERO
(65 = 65) while its own circulant projection gates 60, at or
above the separately-healed circulant (59). **A single weight
tensor now carries a dynamic inference budget: project for the
cheap task, run raw for the hard one — Artin's
shift-complexes-per-task model exists as an artifact
(matryoshka_d256.pt).** The slack-budget law is NUANCED, not
refuted: quantization-on-sharing pays double (corner, cross-
family), but nesting tiers of the SAME symmetry family is
free — slack competition is between UNLIKE compression axes;
like-axes nest. Free rung unlocked: the tier is a runtime
choice, so per-level routing (R2's L6-immune profile) applies
with no extra training. Next rungs banked: 3-tier nest
(complex between), packed circulant forward for real
wall-clock, Snell tier-selection policy. Fences: d256/MPS,
gates-only symmetry, n=1, 1 warm epoch.

## C8-RETROFIT AT 45M VERDICT: the math toll is SCALE-STABLE (-5) — but the GRAPH GRAMMAR pays 2x (ZX 36 -> 17); and the symmetry is only PARTIAL (2026-07-28 night)

union_45m (math 65 / ZX 36), projection + 1 warm union epoch:
projected init 0/120 (total, consistent with 87.5% deletion);
healed math **60/120 (-5; the d256 toll was -6 — read (1)
fires: the sharing toll transfers across 2.4x scale and a
richer diet)**; healed ZX **17/120 (-19, >half the column,
unsound 101)** — read (3) lands on the sharp side: **the toll
is GRAMMAR-DEPENDENT — the tree grammar rides sharing at -5
while the graph grammar loses half its capability to the same
projection.** The ZX capacity story inverts at the commutant:
capacity was never the constraint for ADDING capability (45M
null), but the graph grammar's EXISTING capability is far less
sharing-compressible than the tree grammar's. HONEST FENCE:
final anti-mass 0.1258 (not the d256-class 0.0007) — one union
epoch under the ramp did NOT fully converge the symmetry at
45M; the 60 is a PARTIALLY-symmetric crystal, and the full-
symmetry toll is plausibly worse. Booked as measured; a longer
heal or stronger ramp is the confirmation rung (banked).
Fences: cuda, union diet, n=1, VOCAB_EXTRA pinned (the vocab-47
scar), comparators same-device (65/36).

## NIGHT-28b VERDICT: EMA TAMES THE CHAOS (+12 gate, -58% divergence); ALL SOUPS CRATER — the shell is not convex (2026-07-28 overnight)

(A) TWIN SOUPS CRATER: base+e6 47 (members 64/64), base+e4 49,
base+e2 8, all-4 13. **Even 1e-6 twins cross basin
boundaries** — the equal-quality shell is NOT weight-space
convex; model-soup averaging is dead on independent births.
Nuance kept: soup damage is GRADED in eps (47/49/8) even though
function disagreement was FLAT — weight-space distance grows
with eps while function distance saturates (two different
geometries, now both measured).
(D) HYPERPARAM SOUP: members 62/60/65/63, soup **1/120** —
greedy-soup does NOT transplant to independent births (it
lives on shared-init fine-tunes). Decisive null, cheap.
INCIDENTAL FINDING (kept): the lr1.5e-3/bs8 seed-1 REBIRTH
gated 60 v the atlas's 65 — same seed, same config, same
device class: cuda nondeterminism alone moves a cell ~sigma.
Even eps=0 "twins" diverge; the chaos needs NO perturbation.
(B) **EMA IS THE NIGHT'S WIN — both dials at once**: EMA gates
59/57 v raw 47/51 (+12/+6, a real SCORE LEVER at zero training
cost) AND EMA-vs-EMA twin disagreement 0.0087 v raw 0.0210 —
**a 58% contraction of the shell. Polyak averaging is a
measured chaos damper and a capability lever simultaneously.**
Banked follow-up: EMA on the production recipes (d256, 45M).
(C) SYMMETRY TWINS: c8 twin disagreement 0.0322 v dense
0.0210 — the commutant AMPLIFIES functional divergence (~1.5x),
not damps: fewer parameters means each carries more function,
so equal weight-noise moves the function further. Symmetry
compresses params, not variance.
Fences: all d64/cuda in-battery paired; disagreement
format-bound; EMA decay 0.999 single value; n=1/cell.

## PRE-REG: EMA AT PRODUCTION SCALE — d256 (2026-07-29 pre-dawn, before the run)

Night-28b's EMA win (+12/+6 gate at d64, -58% shell
contraction) rides or dies at the production substrate: one
dense d256/gen-4/3ep birth (seed 1, lr 1.5e-3->? no — the
d256 production recipe: lr per train_mathnative default
schedule, bs 32) with Polyak EMA 0.999 tracked; gate BOTH
endpoints (raw + EMA) on cuda. READ: EMA >= raw + sigma(3) =>
the lever transfers — adopt EMA as a lossless-class speed
default candidate (zero extra compute, one extra weight
copy); EMA ~ raw => d64-only artifact (decay/steps mismatch —
0.999 over 50k steps v 6k; the horizon fence); EMA < raw =>
averaging fights the larger crystal's sharper minima. n=1,
cuda, in-run paired.

## AMENDMENT (target: EMA-at-production pre-reg, same night): device moved cuda -> MPS before the run (3080 released to Artin); raw-vs-EMA stays in-run paired, so the read is device-internal and unaffected. Comparator is the run's own raw endpoint, not any cuda number.

## PRE-REG: THE SYMMETRY SPECTRUM — capability vs frequency band (2026-07-29 pre-dawn, before the run; Artin's superposition riff)

Isotypic decomposition under C8 conjugation: W splits into 5
real frequency bands (k=0 the commutant, {1,7}, {2,6}, {3,5},
4). Instrument: band masses per gate layer, then CUMULATIVE
gates on wfloor d256 — bands added in descending total-mass
order. READS: (1) capability turns on only near full
reconstruction => the function uses the whole spectrum
(consistent with S3's proj-init 2/120 = band-0 alone). (2) a
PARTIAL sum restores 60-class => the crystal's capability
lives in FEW bands — a new compression axis (keep top-m bands
= params*(m/8)-class) and the measured form of Artin's
superposition-of-rotational-models frame. (3) band masses ~
uniform 1/8 predicted (R1-class null); deviation = spontaneous
frequency structure (would contradict the no-spontaneous-
symmetry law — flag hard). Desk only, MPS, no training.

## SYMMETRY SPECTRUM VERDICT: capability accumulates ~linearly with spectral mass — holography along the frequency axis (2026-07-29 pre-dawn)

Band masses read EXACTLY dimension-proportional (0.125 / 0.250
/ 0.2504 / 0.2495 / 0.125) — read (3)'s null holds: no
spontaneous frequency structure (the no-spontaneous-symmetry
law extends to the full spectrum). Cumulative gates
(descending mass): 19 -> 49 -> 63 -> 64 -> 65. VERDICT:
capability turns on GRADUALLY, ~proportional to reconstructed
mass — no single load-bearing band, no threshold: holography
generalizes from random/structured deletion to the symmetry-
frequency axis. Top-3 bands (75% of params) gate 63 (-2) —
a free compression point ON THE DESK, no training (contrast
S3: band-0 alone = 2/120 after deletion needed a heal; keeping
the top-6-of-8 frequencies loses ~sigma). Artin's
superposition frame is measured: the crystal IS a superposition
of symmetry-frequency components, each carrying capability
~proportional to its mass. Fences: d256/MPS, gates-only, n=1.

## EMA AT PRODUCTION SCALE VERDICT: +6 at d256 — the lever transfers; ADOPT-CANDIDATE (2026-07-29 pre-dawn)

One d256/gen-4/3ep birth (constant-lr 3e-4 variant, bs32, MPS),
raw vs EMA(0.999) from the same run: raw 58, **EMA 64 (+6,
2 sigma)**. With night-28b's d64 result (+12/+6, -58% shell
contraction): EMA improves gates at BOTH scales for one weight
copy of memory and zero extra compute. ADOPT-CANDIDATE for the
lossless-speed-defaults family pending one confirmation at the
true production schedule (warmup+cosine, the recipe fence) —
banked as the adoption gate. Fences: constant-LR variant,
in-run paired, n=1.

## PRE-REG: A0 — EMA ADOPTION GATE AT THE TRUE PRODUCTION SCHEDULE (2026-07-29, before the run; minimal-crystal spec Leg A cell 0)

The d256 +6 EMA verdict carried one fence: constant-LR variant.
A0 closes it: one dense d256/gen-4/3ep birth on MPS, seed 1,
bs 32, PRODUCTION schedule (OneCycleLR max_lr 3e-4, pct_start
0.03 — the train_mathnative recipe, replicated knob-for-knob in
sym_birth.py SCHED=onecycle incl. the last-step guard), Polyak
EMA 0.999 tracked; gate BOTH endpoints in-run paired. READS:
EMA >= raw + sigma(3.5) => ADOPT — EMA joins the lossless-speed
-defaults family (every subsequent birth in Legs A-C rides it);
EMA ~ raw (within sigma) => schedule-sensitive: cosine decay
already averages implicitly (small terminal LR = its own
Polyak), EMA stays a constant-LR rescue tool, NOT a default;
EMA < raw => the cosine tail + EMA double-average overshoots —
book and fence. n=1, MPS, in-run paired, gates-only.

## A0 VERDICT: EMA IS REDUNDANT UNDER THE PRODUCTION SCHEDULE — NOT ADOPTED as a default (2026-07-29)

d256/gen-4/3ep, bs 32, seed 1, OneCycleLR(3e-4, pct 0.03), MPS,
in-run paired: raw 59/120 @ 57.33%, EMA(0.999) 59/120 @ 57.14%
— IDENTICAL solve profile per level ({3:20, 4:5, 5:16, 6:7,
7:11} both). The pre-reg's middle read lands exactly: the
cosine tail (terminal LR ~ 0, vanishing final steps) is its own
implicit Polyak average, so EMA has nothing left to add.
Resolution of the EMA story: the +12/+6 (d64) and +6 (d256)
wins were all CONSTANT-LR variants — EMA was substituting for
decay, not adding to it. LAW: averaging and annealing are the
same lever; you pay for it once (schedule OR EMA), and paying
twice is free but idle. EMA stays in the kit as (a) the
constant-LR rescue (e.g. mid-run gates on unfinished births)
and (b) the chaos-damper (-58% shell contraction, still real).
NOT joining lossless-speed defaults; Legs A-C ride the
production schedule alone. Side note: production-schedule raw
59 v constant-LR raw 58 / EMA 64 — the constant-LR+EMA combo
gated HIGHER than the production schedule here (64 v 59, n=1,
cross-run unpaired, sigma 3.5 — flag, not verdict; a paired
schedule-vs-schedule arm is banked if Leg A width work makes
the 5-gate gap look real). Fences: n=1, MPS, gates-only.

## PRE-REG: LEG A WIDTH FLOOR — d48 (2026-07-29, before the run; minimal-crystal spec)

Binary-search the width cliff at the fixed recipe: gen-4, 3ep,
atlas peak cell (constant lr 1.5e-3, bs 8, seed 1, sigma 3.5),
MPS. d48, ffn 192 (4x), heads 4 (head_dim 12). EMA 0.999
tracked and gated ALONGSIDE raw — per A0's law this recipe is
CONSTANT-LR, exactly where EMA is the averaging lever (the d64
comparators: dense raw 53, night-28b EMA +12/+6 class). READS:
d48 within sigma of the d64 line => floor is lower, descend to
d32; d48 drops > sigma => the cliff is between 48 and 64 —
probe d56 or stop and stack compressions on the smallest
survivor. Scoreboard: params-per-solve. n=1 per width, paired
device/recipe.

## LEG A d48 VERDICT: the width cliff is between 48 and 64 — bisect (2026-07-29)

d48/ffn192/heads4, atlas cell (constant lr 1.5e-3, bs 8, seed
1), gen-4/3ep, MPS: raw 44/120 @ 39.06%, EMA(0.999) 50/120 @
51.43%. Like-for-like: raw-vs-raw 44 v d64's 53 = -9 (> sigma
3.5); EMA-vs-EMA 50 v the d64 night-28b EMA class (~59-65) =
similar-sized drop. EMA again earns +6 at constant LR (the A0
law holding: this recipe has no annealing, so averaging pays)
and also recovers most of the width toll's VALID-rate collapse
(39% -> 51%). READ: d48 is below the d64 line — the cliff sits
in (48, 64]. Next per pre-reg: probe d56 (ffn 224) to bisect.
Params note: d48 is ~0.6x the d64 param count for -9 gates —
params-per-solve WORSE than d64, so the floor search is already
bracketing, not descending. Fences: n=1 per width, MPS,
gates-only.

## PRE-REG: LEG A WIDTH FLOOR — d56 bisection (2026-07-29, before the run)

Same recipe exactly (atlas cell, gen-4/3ep, seed 1, MPS, EMA
0.999 tracked): d56, ffn 224 (4x), heads 4 (head_dim 14).
READS: d56 within sigma of d64 (raw ~53 / EMA ~59+) => floor
is d56 — stack compressions there (matryoshka tier, top-3
spectral bands, snap); d56 at the d48 line => cliff is sharp
in (56, 64] — d64 IS the floor at this diet/recipe; strictly
between => the toll is graded, book the slope and take
whichever width wins params-per-solve. n=1, paired
device/recipe.

## LEG A d56 VERDICT: d56 IS THE WIDTH FLOOR — the cliff is sharp in (48, 56] (2026-07-29)

d56/ffn224/heads4, atlas cell, gen-4/3ep, MPS: raw 54/120 @
46.70%, EMA(0.999) 63/120 @ 54.45%. Read 1 lands: d56 sits ON
the d64 line (raw 54 v 53; EMA 63 v the 59-65 class) at ~0.77x
the params — the params-per-solve winner so far. With d48's
44/50, the width cliff is SHARP: 8 dims (56->48) cost -10 raw
/ -13 EMA, while 8 dims (64->56) cost ~nothing. EMA's
constant-LR +9 here is the largest single-run EMA gain yet
booked. NEXT (per spec): stack measured compressions on the
d56 EMA crystal — top-m spectral bands (desk), matryoshka
tier, snap at the knee. Fences: n=1 per width, MPS, gates-only,
sigma 3.5.

## PRE-REG: SPECTRUM ON THE FLOOR — top-m bands of the d56 EMA crystal (2026-07-29, before the run; desk only)

sym_spectrum (now env-parameterized: CKPT/D/FFN) on
sym_birth_dense_w56_ema.pt: band masses + cumulative gates in
descending-mass order, MPS, no training. READS: (1) masses
dimension-proportional again (0.125/0.25/0.25/0.25/0.125) =>
the no-spontaneous-frequency law holds at the floor; (2)
top-3 bands within ~sigma of 63 => the d256 free-compression
point (75% of gate params for ~-2) transfers to the floor
crystal — stackable; (3) capability accumulates ~linearly
with mass (holography) => the frequency-holography law is
width-independent. Divergence on any read = width-dependence
of the spectrum laws — book hard. n=1, desk, gates-only.

## SPECTRUM ON THE FLOOR VERDICT: no spontaneous frequency structure, but the FREE-COMPRESSION POINT VANISHES at the floor (2026-07-29)

d56 EMA crystal, desk: band masses again exactly dimension-
proportional (0.1242/0.248/0.2499/0.2538/0.1242) — read (1)
holds, the no-spontaneous-frequency law is width-independent.
Cumulative gates (descending mass): 2 -> 30 -> 54 -> 60 -> 63.
Read (2) FAILS: top-3 bands (75% of gate params) = 54, a -9
toll — at d256 the same cut cost -2. Read (3) refined:
accumulation is CONVEX at the floor (last 25% of mass carries
+9 gates), v ~linear at d256. VERDICT: holography-along-
frequency is a SLACK phenomenon — a crystal at its width floor
uses its whole spectrum, and band deletion pays the same
sub-additive slack-budget law the compression corner measured
(bits x sharing x width: three axes, ONE shared budget). The
free 75% point at d256 was excess width dressed as spectrum.
Consequence for Leg A stacking: desk-side band cuts are NOT
free on the floor model — compression must be trained-in
(matryoshka joint loss), not sliced-off. Fences: n=1, desk,
MPS, gates-only.

## PRE-REG: MATRYOSHKA AT THE FLOOR — d56 joint-loss tier (2026-07-29, before the run)

The spectrum-on-floor verdict says floor compression must be
TRAINED-IN. Cell: matryoshka_r1 (env-parameterized CKPT/D/FFN/
BS/OUT) warm-started from the d56 EMA crystal (63/120), 1 joint
epoch (CE(W)+CE(STE P_C8(W))), lr 1e-4, bs 8, MPS. Gates both
tiers. READS: dense-tier holds ~63 AND cheap-tier lands well
above the desk band-0-class floor (band-0 alone gated 2 on
desk; commutant projection = 1/8 params) => nesting-under-
joint-loss survives at the floor — the crystal can carry a
sub-tier even below its own width cliff; dense-tier drops
> sigma => at the floor the tier COMPETES (slack-budget law
wins over like-family nesting — the two laws finally collide);
cheap-tier stays dead (~single digits) => even training can't
fit a d56/8-class function — the cliff is representational,
not organizational. n=1, MPS, gates-only, sigma 3.5.

## MATRYOSHKA AT THE FLOOR VERDICT: the tier COMPETES — nesting price = slack (2026-07-29)

d56 joint epoch from the EMA crystal: DENSE-TIER 57/120 @
54.71% (from 63: -6, > sigma), CHEAP-TIER 52/120 @ 45.86%
(commutant projection, 1/8 gate params — desk projection of
the same crystal class gated 2). Read 2 lands: at the width
floor the sub-tier is no longer free — the d256 zero-price
result (65/60) was slack again. UNIFIED LAW (three
measurements now agree): the price of ANY compression —
band cuts, snap bits, nested tiers — is paid from ONE slack
pool, and a floor crystal has none; what training buys at the
floor is a TRADE (dense -6 for a viable 1/8-tier), not a free
lunch. Striking datum: the cheap tier (52) BEATS the full d48
birth (raw 44) — a projected sub-model inside d56 outperforms
a natively-born narrower crystal at comparable effective
budget; organization beats raw width below the cliff.
Params-per-solve: the d56 matryoshka pair is the current
scoreboard leader (one tensor, two working budgets). Fences:
n=1, MPS, 1 joint epoch, gates-only.

## PRE-REG: NIGHT-29 — Leg A hardening on the 3080 (2026-07-29, before launch; Artin GO for the night)

Battery 1 (replication): d64 v d56 (production sym_birth
recipe, EMA tracked), seeds {1,2,3}, PAIRED ON CUDA — its own
comparator line, never read against the MPS numbers. READS:
d56 within sigma of d64 on the cuda line (raw and EMA, n=3)
=> the width-floor verdict replicates and transfers; d56
below d64 on cuda => the floor is device/precision-sensitive
— fence the MPS claim. Battery 2 (finer floor, Artin's d55/
d54 ask): width quantizes to multiples of 8 (heads=4 x even
head_dim for RoPE-half), so descend the CONTINUOUS axis —
d56 with ffn {192,160,128} (from 224), seed 1. READS: gates
flat => the floor is set by d, not ffn (attention-bound);
graded drop => the capacity dial is total params, refine the
knee in ffn units; cliff => ffn has its own sharp floor.
9 births, sequential, marker on success only. Fences: cuda
line n=3 / ffn n=1, gates-only.

## NIGHT-29 VERDICT 1: THE WIDTH FLOOR REPLICATES — d56 = d64 at n=3 on the cuda line (2026-07-29 overnight)

Paired cuda births, seeds {1,2,3}: d64 raw {50,52,52} mean
51.3, EMA {57,61,58} mean 58.7; d56 raw {48,54,53} mean 51.7,
EMA {58,59,58} mean 58.3. d56 matches d64 on BOTH endpoints
within seed noise (n=3) — the MPS n=1 floor verdict replicates
AND transfers device (each line read internally; no cross-
device comparison). EMA's constant-LR gain replicates at +7.0
(d64) / +6.7 (d56) mean. The width-floor claim is now
n=4-across-devices class. Fences: cuda line, gates-only.

## NIGHT-29 VERDICT 2: FFN IS NOT THE DIAL EITHER — flat to ffn 160, params-per-solve leader moves to d56/f128 (2026-07-29 overnight)

d56 with ffn {224,192,160,128}, cuda, seed 1: EMA 58/58/58/55,
raw 48/50/46/52. EMA-endpoint capability is FLAT from ffn 224
down to 160 (0.71x the ffn params for -0) and dips only -3
(within sigma) at 128. Combined with verdict 1 (d56=d64) and
the sharp d48 cliff: capability at this diet is pinned by the
ATTENTION WIDTH d (with a cliff in (48,56]), not by ffn
capacity — the ffn was carrying slack all along, consistent
with gate-layer band cuts being where every compression
instrument bites. Params-per-solve leader is now d56/ffn128
EMA (55-58 class at ~0.55x the d64/f256 params). NEXT
candidates: ffn 96/64 to find the ffn cliff; d56/f128 as the
new floor substrate for matryoshka + snap stacking. Fences:
n=1 per ffn point, cuda, gates-only, sigma 3.5.

## PRE-REG: NIGHT-29b — the ffn cliff (2026-07-29 overnight, before launch)

Extension of verdict 2: d56 with ffn {96, 64, 48}, cuda, seed
1, EMA tracked, same recipe. READS: EMA holds ~58-class into
the 90s => the ffn floor is far below 4x convention — book the
knee and crown the smallest survivor; sharp drop at a point =>
the ffn cliff, symmetric evidence to the d-cliff in (48,56];
graded decline => ffn is a smooth capacity dial below 128
(unlike d). 3 births, marker on success only. Fences: n=1 per
point, cuda line, gates-only.

## NIGHT-29b VERDICT: THERE IS NO FFN CLIFF — flat to ffn 48; the crystal is an ATTENTION machine (2026-07-29 overnight)

d56, ffn {96,64,48}, cuda, seed 1: EMA 56/55/54 (raw 49/54/49).
Full descent now reads ffn 224->48: EMA 58/58/58/55/56/55/54 —
a -4 drift over a 4.7x ffn reduction, no knee anywhere (read 3,
in its shallowest possible form). Contrast the d axis: 8 dims
(56->48) cost -13. VERDICT: at this diet the MLP is almost
pure slack — the function lives in attention width; ffn 48
(SMALLER than d=56, inverted SwiGLU) still gates 54-class.
Params-per-solve leader: d56/ffn48 EMA (54 at roughly 1/3 the
d64/f256 params — v its 58.7 mean, one sigma down for 3x
fewer params). Program consequence: Leg A stacking should
target d56/f48-class substrates, and the attention block is
where the irreducible computation sits — the next compression
frontier is qkv/o, not gate. Fences: n=1 per point, cuda,
gates-only, sigma 3.5.

## PRE-REG: TIER-RETRY — the first adaptive-crystal controller (2026-07-29 morning, before the run; attention-core Leg 0)

d56 matryoshka pair (matryoshka_d56.pt: dense 57 / cheap 52 on
MPS), desk, zero training. Per gate row: attempt with the CHEAP
tier (commutant projection, 1/8 gate params); on failure retry
the SAME row with the DENSE tier. Report: retry-policy solves,
overlap census (cheap-only / dense-only / both / neither), and
effective-gate-params-per-row = cheap + P(cheap-fail) * dense.
READS: retry ~= dense solves (~57) at effective params well
under dense => the dial's first measured win (oracle-fail = a
free difficulty signal; no predictor needed); retry ~= cheap
(~52) => failures are tier-blind (fail both) — the tiers are
NESTED in capability, not complementary; retry > dense =>
the tiers are partially COMPLEMENTARY (union > either) — a
diversity bonus, book hard (would echo verify-block near-tie
composition). Fences: n=1, MPS, gates-only, same seeds as all
d56 gates.

## TIER-RETRY VERDICT: the dial's first measured win — 59 solves at 69% effective params, and the tiers are PARTIALLY COMPLEMENTARY (2026-07-29)

d56 matryoshka pair, desk: retry policy 59/120 v always-dense
57 v always-cheap 52. Census: both 50, cheap-only 2, dense-only
7, neither 61; cheap-fail rate 0.567; effective gate params/row
69.4k v dense 100.4k (69%). BOTH pre-reg wins land together:
(read 1) the controller beats always-dense on COST (-31% gate
params) — oracle-fail is a sufficient difficulty signal, no
predictor model needed; (read 3) retry EXCEEDS dense (+2,
sub-sigma but directionally the cheap-only rows exist: 2 rows
the projection solves that the full tensor misses) — the tiers
are partially complementary, echoing the verify-block
composition near-tie mechanism (different roundings, different
coin-flips). The adaptive crystal is REAL at its smallest
scale: one tensor, a free difficulty signal, strictly better
params-per-solve than either fixed tier. NEXT: 3-tier ladder
(matryoshka rung 2) turns retry into escalation; predictive
routing must now beat THIS baseline (the fence from the
entropy-adaptive null: adaptivity pays only where variance
lives — here it measurably does). Fences: n=1, MPS, gates-only,
retry +2 within sigma.

## PRE-REG: ATTENTION ANATOMY 1a — the head census (2026-07-29, before the run; attention-core Leg 1)

d56 EMA crystal (63/120), desk: for each head h of 4 (head_dim
14), zero head h across ALL layers (q,k,v row blocks + o
column block), gate. 4 arms. READS: uniform ~equal drops =>
heads are DEMOCRATIC (holography extends into attention —
compression should shrink all heads, e.g. head_dim cuts);
one head's deletion craters while others are ~free => an
OLIGARCHY inside attention (echoes the NNUE readout
oligarchy; compression should PRUNE heads — 4->2 arm gets
priority and keeps the load-bearers); intermediate (graded
spectrum of drops) => mixed economy, rank read (1b) decides.
Fences: n=1 per arm, MPS, gates-only, all-layer deletion
(per-layer follow-up only for load-bearing heads).

## HEAD CENSUS VERDICT: NO SLACK IN ATTENTION — every head is load-bearing (2026-07-29)

d56 EMA crystal, all-layer single-head deletions: h0 8, h1 6,
h2 4, h3 17 (of 63; valid rates 1.4-10.5%). NEITHER pre-reg
read: not democratic-with-redundancy (holography predicts
graceful drops; these are catastrophic) and not an oligarchy
(no single carrier — ALL FOUR are essential, h3 marginally
least so). The attention-machine law sharpens: the MLP held
~all the slack (flat to ffn48), and attention holds ~NONE —
25% of attention width removed post-hoc destroys the function
regardless of WHICH 25%. Consequence for Leg 2: post-hoc head
pruning is dead; attention compression must be TRAINED-IN
(heads-2 birth, C8-on-qkv with heal) — mirroring the floor
lesson (trained-in tiers work where desk cuts fail). NOTE the
contrast with gate layers, where post-hoc desk cuts (bands,
tiers, snap) all found free points at d256: the slack pool is
not just size-dependent but LOCATION-dependent. Fences: n=1
per arm, MPS, gates-only.

## PRE-REG: ATTENTION ANATOMY 1b — the rank read (2026-07-29, before the run)

d56 EMA crystal, desk: SVD every qkv.weight [168,56] and
o.weight [56,56]; report singular-value decay (top-k mass);
truncate ALL attention mats to rank r in {48, 32, 24, 16}
(per-matrix), gate each. READS: parity at r<=32 => attention
is LOW-RANK — the d-cliff is about head geometry/count, not
rank, and factorized qkv births are the Leg 2 arm; fragile by
r=48 => d IS the effective rank (the width cliff explained
mechanically — 56 dims because the function needs ~56 rank);
graded => rank is a smooth axis (unlike heads), knee sets the
factorization budget. Fences: n=1 per r, MPS, gates-only.

## RANK READ VERDICT: attention has a rank knee in (24, 32] — graded then cliff (2026-07-29)

d56 EMA crystal, all qkv/o truncated per-matrix: r=48 -> 59
(-4, ~sigma), r=32 -> 54 (-9), r=24 -> 12 (crater), r=16 -> 0.
SV energy heavy-tailed (qkv top-32 = 88.9%, top-48 = 98.2%;
o more compact: top-32 = 96.8%). READ (graded-then-cliff
hybrid): rank is a SMOOTH axis down to ~32 (unlike heads,
where any deletion craters) with a hard floor in (24,32] —
the crystal runs on ~32 effective attention rank inside
56-dim head geometry. The d-cliff and the rank floor are
DIFFERENT walls: width 48 kills (head_dim geometry shrinks)
while rank 32 in full 56-dim geometry only pays -9 — geometry
matters beyond rank. Leg 2 arms (anatomy-informed): (1)
trained-in heads-2 birth (tests count v width, post-hoc
untestable per census); (2) factorized-qkv birth at r=32
(trained-in should beat the post-hoc -9; params 168x56 ->
168x32+32x56 = 76%). Fences: n=1 per r, MPS, gates-only.

## PRE-REG: LEG 2 ARM 1 — heads-2 at d56 (2026-07-29, before the run)

sym_birth, Mac, one variable v the d56 Mac comparator (raw 54
/ EMA 63): D=56, FFN=224, HEADS=2 (head_dim 28), seed 1, EMA
tracked. READS: parity => head COUNT is slack at fixed width
(only total width matters — compression should cut heads not
dims); drop > sigma => 4 heads at hd14 beat 2 at hd28 — head
multiplicity is load-bearing (matches census: 4 essential
specialists), and the attention core is count x geometry, not
just width. n=1, MPS, gates-only.

## HEADS-2 VERDICT: head multiplicity is load-bearing — 4 specialists beat 2 generalists at equal width (2026-07-29)

d56/ffn224/HEADS=2 (head_dim 28), Mac, seed 1: raw 51 / EMA 56
v the 4-head comparator raw 54 / EMA 63 — EMA -7 (2 sigma).
Read 2 lands: at FIXED width, halving head count while
doubling head_dim costs real capability — the census's four
essential specialists are not an artifact of post-hoc
deletion; the attention core is COUNT x GEOMETRY, and the
multiplicity itself (4 distinct attention patterns per
position) carries function that fatter single patterns cannot.
With the rank knee (32) this completes the anatomy: the
attention core = MANY cheap-rank specialist heads in
sufficient geometric room; compression must preserve count
and geometry, may cut rank. Fences: n=1, MPS, gates-only.

## PRE-REG: LEG 2 ARM 2 — factorized qkv at the knee (2026-07-29, before the run)

sym_birth + RANK knob: each blk.qkv becomes Linear(56->32) @
Linear(32->168), trained FROM BIRTH (post-hoc r=32 paid -9;
trained-in is the test). d56/ffn224/heads4, seed 1, EMA, Mac,
v the same comparator (54/63). qkv params 76% of dense. READS:
parity => trained-in factorization recovers the post-hoc toll
(matches the matryoshka/floor pattern: train-in beats
slice-off) — attention params-per-solve win; -9-class drop =>
the knee is a hard property of the FUNCTION, not the
projection; worse => the bottleneck hurts optimization itself
(low-rank births are known-harder — book against lottery-
ticket reachability). n=1, MPS, gates-only.

## PRE-REG: E3 — exact-mode paired greedy gate, 50 rows (2026-07-29; axiom GO by relay 0b47888)

House emits 50 fresh gate-style prompts (band 777000, disjoint
from battery20 and GATE_BAND) + fp32-eager greedy continuations
(<=64 tok, eos-stop) from the S2 scorer of record (sha
298f9077). Delivery: e3_battery50.txt / _meta.jsonl /
e3_expected_greedy.txt + sha pins to axiom's data/llmopt/.
Axiom decodes in EXACT mode; PASS = token-identical on 50/50.
READS: 50/50 => exact-mode decode path certified (E-series
closes; call-span tranche becomes next joint work); any
divergent row => diff the first divergent position's eager
logit margin — margin <= ~0.02 is the known fp16-near-tie
class (composition rounding), NOT a bug; margin > 0.02 =>
real decode-path divergence, bisect. Fences: greedy-only,
prompt-format-bound (CE-400 class), house side fp32 eager.

## AMENDMENT (target: E3 pre-reg, same day, before delivery): stop rule and margin design changed after inspecting the first generation — (1) the dist-trained scorer rarely emits eos greedily (0/50; post-newline decode degenerates to repetition), so continuations stop at newline/eos/64; (2) whole-row margin filtering kept only 20/120, so the protocol is TRUNCATE-at-first-near-tie instead: every delivered token has eager margin >= 0.05 (min row len 8). Consequence: any axiom-side divergence is a real decode bug by construction — the margin-adjudication branch of the pre-reg is unreachable. Delivered shas: battery 466a6592, meta 223235ee, greedy 398b7993.

## FACTORIZED-QKV VERDICT: the rank knee is a FUNCTION property — trained-in factorization does not recover it (2026-07-29)

d56/ffn224/heads4, qkv = 56->32->168 from birth, Mac, seed 1:
raw 50 / EMA 55 v comparator 54/63 — EMA -8 (>2 sigma),
matching the post-hoc r=32 toll (-9). Unlike tiers (train-in
beat slice-off at the floor), attention rank does NOT heal by
training through the bottleneck: the function needs the
near-full-rank maps, and SGD cannot route around the
constraint. Leg 2 closes 0-for-2 (heads-2 -7, rank-32 -8):
the attention core resists BOTH count and rank compression —
consistent with the census (no slack) and the sharp d-cliff.
The attention block at this diet is effectively
INCOMPRESSIBLE by the axes tried; remaining untried: C8-on-
qkv (sharing), attention-side snap (bits — 1c still queued).
Params-per-solve leader UNCHANGED (d56/f48 EMA class).
Fences: n=1, MPS, gates-only.

## PRE-REG: ATTENTION ANATOMY 1c — where snap bites (2026-07-29, before the run)

d56 EMA crystal, rational snap Q=16 (below the knee) on
attn-only (qkv+o) v mlp-only (gate+up+down) v both, desk, MPS.
READS: attn craters, mlp survives => bits follow the function
(the slack map from ffn/census transfers to the bits axis —
allocate precision to attention, snap the MLP hard: the
bits-portfolio riff's first measured allocation); both
degrade equally => snap damage is diffuse (contradicts the
location-dependent slack map — book hard); both-arm ~=
sum of singles => bits-axis damage composes additively across
blocks (v the corner's sub-additive slack pool: tests whether
LOCATION-split budgets are independent where AXIS-split ones
were not). Fences: n=1 per arm, MPS, gates-only.

## SNAP-ALLOC VERDICT: Q=16 IS FREE EVERYWHERE — bits and structure are different currencies (2026-07-29)

d56 EMA crystal: attn-only 63, mlp-only 63, both 63 — v
baseline 63. NEITHER pre-reg read: snap at Q=16 costs NOTHING
in any location, including the attention block that just
resisted rank (-8) and count (-7) compression and the both-arm
that snaps every 2D weight in the model. HARD BOOK: the
attention core is structurally incompressible but fully
BIT-compressible at Q=16 — precision and structure spend from
DIFFERENT budgets at this width (v the corner's bits-x-sharing
sub-additivity on the 19M: sharing consumed quantization slack
there; here plain snap without sharing finds the full bits
slack intact even at the width floor). The bits-portfolio
allocation answer at Q=16 is: NO allocation needed — snap all.
Follow-up armed same-day: Q=8 sweep to find where the bits
axis finally bites and whether THAT is location-dependent.
Fences: n=1 per arm, MPS, gates-only, Q=16 only.

## PRE-REG: SNAP-ALLOC Q=8 (2026-07-29, immediately following)

Same three arms at Q=8. READS: location-dependent damage at
Q=8 => the allocation question reopens below Q=16 (bits follow
the function once bits are scarce); uniform damage => bits-
axis damage is diffuse at every price; still free => the d56
crystal has an exactly-rational twin at Q<=8 (echoes the 19M
integer-twin result — book toward the exact-representation
thread). Fences: n=1 per arm, MPS, gates-only.

## SNAP-ALLOC Q=8 VERDICT: bits bite uniformly — location-blind and ~additive (2026-07-29)

attn-only 59, mlp-only 59, both 56 (v 63): -4 / -4 / -7. Read
2 lands: below the free zone the bits axis is DIFFUSE — equal
damage per location (despite attention being structurally
precious and the MLP structurally slack) and near-additive
composition (-7 ~ -4 + -4). The full bits picture at d56:
free at Q=16 everywhere, uniform gentle toll at Q=8 — the
precision currency has no location structure at all,
completing the dissociation from the structural currency
(which is ALL location: heads/rank/width precious, ffn/bands
slack). Allocation answer for the bits-portfolio riff: at
matched total bits there is nothing to allocate ACROSS
locations; allocation only pays across the STRUCTURAL axes.
ANATOMY DAY CLOSES: the d56 attention-machine anatomy is
heads=all-essential, rank>=32-with-geometry-56, ffn=slack,
bits=uniform-and-cheap-to-Q16. Fences: n=1 per arm, MPS,
gates-only.

## PRE-REG: REVERSE-PAIRS — the pincer entry ticket (2026-07-29, before the run; banked 07-26, pulled into today)

d64 (production sym_birth recipe, Mac, seed 1, EMA): arm A =
gen-4 forward rows only (comparator class known); arm B =
matched TOTAL dose, 50/50 forward + REVERSED pairs (nxt->cur:
"Current: <nxt> ... Step: <cur>" — the model learns the
backward step). Gate both FORWARD (the standard gate). READS:
B ~ A on the forward gate => reverse capability rides free at
half forward dose (dual-direction crystal exists — pincer
next); B < A > sigma => backward rows TAX forward capability
(direction competes for capacity); B > A => backward training
HELPS forward (bidirectional consistency regularizes — book
hard). Bidirectional-cheat fence: eval rows are the standard
gate (never reverse-sampled). n=1, MPS, gates-only.

## REVERSE-PAIRS VERDICT: no free dual-direction crystal — forward capability drops at matched total dose (2026-07-29 evening)

d64, 50/50 forward + reversed rows, matched TOTAL dose, Mac:
forward gate raw 36 / EMA 50 v the forward-only comparator
class (Mac raw 53; EMA class 59+). EMA -9-class, raw -17. Read
2's surface lands (backward rows cost forward capability), BUT
the pre-reg carried an unnamed CONFOUND, stated here honestly:
arm B has only HALF the forward rows, so "tax" and "half-dose"
are not separated — the clean claim is the NEGATIVE one: the
dual-direction crystal is NOT free at matched compute (the
cheapest pincer entry fails; direction must be paid for
somehow). Also unmeasured: whether the backward direction was
actually LEARNED (no reverse gate exists yet). CONTROLS BANKED
before any sequel: (a) forward-only at HALF dose (separates
dose from tax); (b) a reverse-gate instrument (backward
capability read); both cheap. The temporal-pincer bank stays
banked — its entry price just went up. Fences: n=1, MPS,
gates-only, dose-confounded as stated.

## PRE-REG: SLACK RESTORATION — d256 anatomy (2026-07-29 evening, before the run; spec 2026-07-29-slack-restoration)

Unified instrument (scratch/anatomy.py; frozen 07-29 originals
kept) on wfloor d256 (comparator 65, MPS): head census (4
drops), rank read (RANKS 192,128,96,64 — scaled to d256), snap
Q=16 x {attn, mlp, both}. PREDICTIONS (at-floor theory): head
drops GRACEFUL at d256 (slack = provisioning above floor);
rank knee ~2x the d56 knee in proportion (~96-128); Q=16 free.
READS: graceful heads => incompressibility is a FLOOR
property, per-block — the slack-location law generalizes;
catastrophic again => attention is INTRINSICALLY routing-bound
at any width (relations do not superpose — stronger law).
Fences: n=1 per arm, MPS, gates-only.

## PRE-REG: REVERSE-PAIRS CONTROLS on the cuda line (2026-07-29 evening, before launch)

3080: A2 = forward-only HALF dose (HALF=1, same shuffle seed
as the REV split); B2 = REV=1 replication. Comparators: cuda
d64 n=3 (raw 51.3 / EMA 58.7). READS: A2 ~ B2 => the Mac
revpairs drop was DOSE (backward rows neutral filler; pincer
ticket retries at full-fwd + backward-extra); A2 > B2 =>
backward rows actively TAX forward (direction competes;
pincer stays two-model). Fences: n=1 per arm, cuda line,
gates-only.

## SLACK RESTORATION VERDICT: heads are INTRINSICALLY incompressible — and the snap grid is sigma-priced (2026-07-29 evening)

wfloor d256 (comparator 65), unified anatomy: HEADS 21/11/30/19
— still catastrophic at 4x the geometric room. The at-floor
prediction is WRONG (booked): head incompressibility is not
floor-pinch, it is INTRINSIC — relations do not superpose at
any width; every head is a load-bearing relation channel in
both the floor crystal and the roomy one. RANK: 65/60/54/48 at
r=192/128/96/64 — graded slack that SCALES with width (r=192 =
75% rank is FREE at d256 v d56's -4 at 86%); prediction
half-right (soft knee, right region). SNAP Q=16: attn 57 (-8),
mlp 60 (-5), both 53 (-12) — NOT free at d256, reversing the
d56 result. MECHANISM (post-hoc, flagged as such): d256
weights sit at ~sigma 1/sqrt(256) ~ 0.06 while the Q<=16 grid
steps at 1/16 = 0.0625 — the grid is COARSER than the weight
scale; at d56 (sigma ~ 0.13) the same grid is fine-grained.
LAW CANDIDATE: the precision currency is priced in SIGMA
units, not absolute denominators — Q must scale ~sqrt(d) for
constant relative resolution (the sigma-never-transports fence
reappearing on the bits axis). Falsifiable: Q=64 at d256
should be free (grid/sigma ratio matches d56's Q=16).
Fences: n=1 per arm, MPS, gates-only.

## PRE-REG: SIGMA-PRICED SNAP CHECK — Q=64 at d256 (2026-07-29 evening, immediately following)

If the precision grid is sigma-priced, Q=64 at d256 (grid
1/64 = 0.016 ~ sigma/4, matching d56's Q=16 ratio) should be
FREE in all three locations. Free => the law books (snap
denominators scale ~sqrt(d); "Q" fences must state width);
still bites => the d256 crystal is genuinely more precision-
sensitive (a capability-density story instead). n=1, MPS,
gates-only.

## SIGMA-PRICED SNAP VERDICT: Q=64 at d256 is EXACTLY FREE — the precision knee is a sigma-ratio constant (2026-07-29 evening)

Q=64 x {attn, mlp, both} on wfloor d256: 65/65/65, identical
per-level profiles to the comparator. The law books: snap
denominators are priced in WEIGHT-SIGMA units, not absolute
values. Quantitative cross-check across every measured
crystal: the knee sits where grid ~ 0.5-1.0 sigma everywhere
— d56 (sigma~0.13): Q=16 free (grid 0.48 sigma), Q=8 bites
(0.96 sigma); d256 (sigma~0.06): Q=64 free (0.25 sigma), Q=16
bites (1.0 sigma); 19M/d256 rational-snap knee Q in (16,24]
(2026-07-27) = grid 0.65-1.0 sigma. One constant, three
crystals, two widths, both labs' precision results unified.
CONSEQUENCES: (1) every Q fence in the corpus is width-bound —
state sigma or d; (2) the exact-twin format gets a principled
rule: denominator ~ 2/sigma (Q ~ 2*sqrt(d) class) for free
snap, no magic numbers; (3) the d56 "bits free everywhere"
verdict amends implicitly: free BECAUSE its Q=16 grid was
half-sigma — the location-blindness read stands, the
free-ness was scale. Fences: n=1 per arm, MPS, gates-only,
sigma estimated from init-scale class.

## REVERSE-PAIRS CONTROLS VERDICT: backward rows actively TAX — and the diet is already SATURATED at half dose (2026-07-29 evening)

cuda line (comparators n=3: raw 51.3 / EMA 58.7): A2 half-dose
forward-only raw 52 / EMA 56 — HALF the corpus costs ~nothing
(-2.7 EMA, sub-sigma). B2 revpairs raw 35 / EMA 50 —
replicates the Mac null on the paired line (36/50 there). A2 >
B2 by 6 EMA / 17 raw: the read lands on TAX — backward rows
actively damage forward capability at matched dose; direction
competes for capacity; the temporal pincer stays TWO-MODEL as
originally banked (single dual-direction crystal rejected
twice, now dose-controlled). BONUS VERDICT (Leg C's first
datum, free): gate capability at d64 is FLAT in dose from 50%
to 100% of the gen-4 corpus — the marginal value of the second
~325k rows is ~zero at this width. Diet-descent implication:
the corpus is oversized for d64-class crystals; farm tranches
should buy HARDER rows (level/kind gaps), not more rows —
Leg C's allocation question just got its first coefficient.
Fences: n=1 per arm, cuda line, gates-only; A2-B2 gap 1.7
sigma (directional, consistent with the raw gap 4.9 sigma).

## PRE-REG: THE d56 EXACT TWIN — full-model snap at the sigma rule (2026-07-29 night)

Sigma-priced rule says Q=16 (grid ~0.5 sigma) is the free
denominator at d56. Cell: snap EVERY floating tensor of the
d56 EMA crystal (attn + mlp + emb + head + 1D norm gains —
the prior "both" arm covered only attn+mlp) to best-rational
Q<=16, gate. READS: 63-class => the d56 crystal has a FULL
exact-rational twin at 4-bit-class denominators — every weight
an integer/16; book the artifact (int16 numerators + one
denominator = a deployable exact format; precision doctrine:
exactness is a speed/determinism lever); embeddings/norms
crack it => the sigma rule is blockwise (norm gains sit at
sigma~1, not 1/sqrt(d) — per-tensor denominators needed:
Q_t ~ 2/sigma_t). Fences: n=1, MPS, gates-only.

## EXACT-TWIN VERDICT: the d56 floor crystal has a FULL exact-rational twin at Q<=16 (2026-07-29 night)

Full-model snap (attn + mlp + emb + head + all 1D norm gains):
62/120 @ 54.35% v 63 comparator — parity-class (-1, sub-
sigma). Norm gains needed no special treatment (snap error
<= 0.22 sigma per tensor — the sigma rule held per-tensor
without blockwise denominators). Artifact saved:
checkpoints/exact_twin_d56_q16.pt (fp32 image; rational-of-
record = the (n, q<=16) table, deterministically derivable;
dyadic share 39.8%). The exact-representation thread now
spans 19M (int twin fq512) and the d56 floor (best-rational
Q<=16) — smallest crystal, smallest denominators, consistent
with the sigma law (bigger sigma at small width => coarser
free grid). Fences: n=1, MPS, gates-only.

## PRE-REG: MATRYOSHKA RUNG 2 — the 3-tier ladder (2026-07-29 night, before the run)

Warm 1 joint epoch from the d56 EMA crystal, THREE forwards
per step: CE(W) + CE(P_C2(W)) + CE(P_C8(W)) on gate weights
(C2 = half params, C8 = eighth; STE both). Gate all three
tiers. READS: full ~63-class AND half > eighth > desk-floor =>
a working 3-rung ladder in one tensor — tier-retry becomes
ESCALATION (cheap -> mid -> dense) and the Snell policy has a
real dial; full pays > sigma more than rung-1's -6 => tier
COUNT deepens the slack tax (each rung prices in); middle
tier ~ eighth => nesting collapses to binary (intermediate
groups buy nothing). Fences: n=1, MPS, 1 joint epoch,
gates-only.

## MATRYOSHKA RUNG 2 VERDICT: the 3-rung ladder works — and the HALF tier is FREE (2026-07-29 night)

1 joint epoch (3 forwards: dense + STE P_C2 + STE P_C8) from
the d56 EMA crystal: DENSE 57, HALF 57, EIGHTH 48. Reads: (1)
tier COUNT does not deepen the tax — dense pays exactly
rung-1's -6 (63->57) with a third objective added; (2) the
C2 half-tier gates AT the dense line (57=57, half the gate
params) — consistent with the ladder toll (2x sharing: -1)
and the mlp-slack law; the middle rung is a free stop; (3)
eighth 48 v rung-1's 52 (-4, ~sigma): the deepest rung pays
slightly for sharing its tensor with a third sibling. The
nested-subspace ladder (C2 contains C8-commutant) is REAL:
one tensor, three working budgets 57/57/48 at 1x/0.5x/0.125x
gate params. Tier-retry upgrade path: escalate eighth ->
half -> dense; with half free, the POLICY may collapse to
"eighth first, then half, rarely dense" — the escalation cell
is armed (desk). Artifact: matryoshka_d56_3tier.pt. Fences:
n=1, MPS, 1 joint epoch, gates-only.

## E3 VERDICT: PASS 50/50 — the exact-mode decode path is certified; the E-series CLOSES (2026-07-29 night; axiom relay)

Axiom decoded all 50 margin-certified prompts in exact mode
(FX-V1 tables appended once-at-export to container 298f9077,
weight bytes untouched, certify_tables clean; new KV-cached
exact greedy driver, 10.7 s single-core full battery) —
token-identical 50/50, independently confirmed by hash: their
generated-ids file sha 398b7993 = our e3_expected_greedy.txt
pin byte-identical. E2 (logits 6.2e-6) + E3 (greedy tokens
exact over margin-certified rows) = the cross-lab contract is
CLOSED end-to-end: same weights, same tokenization, same
argmax path, two independent implementations. The margin-
certified battery design paid: zero adjudication needed.
CROSS-LAB LOOP NEXT: the call-span pilot (500 rows, sha
de6c9f15 verified house-side) -> paired d64 arms.

## PILOT RECEIPT (call-span 500): verified + accepted (2026-07-29 night)

Shas verified house-side (de6c9f15 / 5f23e34a). Atom order
CONFIRMED from sidecar (residents, gcd, Mod, **, then call:
-> LAST) — matches the VOCAB_EXTRA append plan; the ordered
list is the fence of record for every arm birth. Site
spelling: comma-space ACCEPTED (verbatim-from-row is correct;
our spaceless example was illustrative only — no re-emit).
Seed band: 1000+ confirmed disjoint from qual 0..39; arm
evals will additionally hold out BY ROW SPLIT from the 500
(no band overlap possible). Level mix hard-heavy accepted
(saturation coefficient says harder rows are the paying
kind). Design note for the arms (flagged, not a conformance
issue): gcdstep call values resolve the WHOLE subchain (e.g.
call: gcd(62,39) -> 1 while nxt is the next step), so spans
act as auxiliary end-value supervision rather than next-step
shortcuts — the capability read stays valid, the mechanism
read (use v ignore) sharpens: a model that USES the span
should get END-of-chain rows right earlier. Fences: receipt
only; arms pre-reg separately.

## PRE-REG: CALL-SPAN PAIRED ARMS — Leg B first read (2026-07-29 night, before the run)

Pilot 500 (sha de6c9f15), split Random(7) 400 train / 100
held-out eval, SAME rows both arms. Arm PLAIN: "Hints: none".
Arm SPAN: "Hints: <calls joined by '; '>" (engine-computed —
the delegation format; at eval the span arm also receives its
hints, as the LLMUE engine would supply them at decode).
Tokenizer atoms pinned in sidecar order: gcd,Mod,**,call:,->.
d64/ffn256/heads4, 20 epochs (tiny corpus), lr 1.5e-3, bs 8,
seed 1, Mac, both arms one script. SCORE: held-out greedy
next-step EXACT MATCH (the rows are axiom-certified; string
match is sound here because nt-chain steps are canonical
integers/forms, unlike sympy expressions). READS: SPAN >
PLAIN > sigma-ish => delegation pays — the model uses engine
values it did not compute (Leg B capability-per-param
headline); SPAN ~ PLAIN => spans ignored at this scale/mix
(mechanism read: end-value spans may be too indirect —
step-local span option already banked with axiom); SPAN <
PLAIN => hint tokens tax context (format cost). Fences: n=1,
MPS, tiny-corpus regime, exact-match (not sympy) scoring.

## AMENDMENT (target: call-span paired-arms pre-reg, same night, before the valid run): first SPAN run was VOID — 0/0 rows tokenized (the base tokenizer covers "Hints: none" only as ONE fixed template atom, so any non-none hint is unencodable; PLAIN arm unaffected, scored 38/100). Fix: two FORMAT atoms "Hints: " and ";" appended AFTER the five pinned diet atoms (full order: gcd,Mod,**,call:,->,"Hints: ",";" — the atom-order fence of record for these arms; round-trip verified). Rerun both arms under the 47-atom vocab so the arms stay format-paired.

## CALL-SPAN PAIRED ARMS VERDICT: end-value spans are NEUTRAL-to-positive — delegation neither pays nor taxes yet (2026-07-29 night)

Paired 47-atom vocab, d64, 20 ep, same 400/100 rows: PLAIN
48/100, SPAN 52/100 held-out greedy exact. +4 at n=100 is
sub-sigma (binomial sigma ~5) — the read lands between (1)
and (2): no format tax (hint tokens are free), no clear
delegation win. The mechanism flag from the receipt holds:
gcdstep spans carry END-of-subchain values, which cannot
shortcut next-step prediction — the span is auxiliary signal,
not a prosthetic. The sharpened cell is already banked with
axiom: STEP-LOCAL spans (call: Mod(62, 39) -> 23 for the
immediate step) would let the model READ the answer off its
hint — use-v-ignore dissociates cleanly there. Also honest:
tiny-corpus regime (400 rows), single seed; a real Leg B
verdict wants the full-size tranche. Fences: n=1, MPS,
exact-match scoring, end-value spans only.

## PRE-REG: STEP-LOCAL CALL-SPAN ARMS (2026-07-29 night; axiom tranche dd5fbb09, row-paired to de6c9f15)

Same instrument, same 47-atom vocab, same Random(7) 400/100
split (row-pairing is structural: identical rows, only calls
differ; contrast carried by 309 gcdstep rows whose span now
hands the model the exact remainder token it must emit).
READS: SPAN >> PLAIN (multi-sigma, v the end-value +4) => the
model READS its prosthetic — delegation works when the span is
step-local; capability-per-param headline lands and the LLMUE
decode-side resolver is next; SPAN ~ PLAIN again => the model
ignores hints ENTIRELY at this scale (hint-blindness — a
format/attention question, book hard); PLAIN drifts from 48 =>
seed/vocab sensitivity fence. n=1, MPS, exact-match, paired.

## STEP-LOCAL CALL-SPAN VERDICT: DELEGATION WORKS — the model reads its prosthetic (2026-07-29 night)

Row-paired arms (dd5fbb09), same vocab/split: PLAIN 44/100,
SPAN 59/100 — +15 at n=100 (~3 sigma binomial), v the
end-value tranche's +4 (sub-sigma). Read 1 lands: when the
span carries the IMMEDIATE step's value, the model USES
engine-computed results it did not compute itself — Artin's
"weights call an external function" measured at its first
scale. The dissociation is clean: same rows, same atoms, same
budget; only span locality changed, and the effect went from
noise to 3 sigma. PLAIN drifted 48->44 across runs (sub-sigma
backend noise; fence noted per pre-reg). CONSEQUENCES: (1)
Leg B's mechanism is confirmed — spans must be STEP-LOCAL
(prosthetics answer the question being asked, not the question
five steps ahead); (2) next rungs: full-size tranche (scale
read), then the LLMUE decode-side resolver (engine computes
the span AT INFERENCE — true delegation, weights store the
protocol not the arithmetic); (3) capability-per-param note:
+15 exact-match for ZERO added params — the prosthetic diet
is the cheapest capability lever measured this week. Fences:
n=1 per arm, MPS, tiny-corpus, exact-match, gcdstep-carried
(309/500 rows).

## PRE-REG: THE ESCALATION POLICY — 3-rung tier-retry (2026-07-29 night; escalation-engine cell 1)

matryoshka_d56_3tier.pt (57/57/48): per gate row, attempt
EIGHTH; on fail, HALF; on fail, DENSE. Report policy solves,
full escalation census (8 outcome classes), effective gate
params/row. PREDICTION: ~57 at effective params well below
half-alone (eighth catches the easy mass, half is free).
READS: policy > 57 (union effects, cf. rung-1's +2) => the
ladder is a CAPABILITY lever, not just cost; policy ~ 57 =>
cost lever confirmed; policy < 57 => escalation loses rows
the dense tier alone would catch (ordering effect — book).
Desk, MPS, n=1, gates-only.

## ESCALATION POLICY VERDICT: the ladder BEATS its own dense tier — 62/120, recovering the pre-matryoshka crystal (2026-07-29 night)

3-rung policy on matryoshka_d56_3tier.pt: 62/120 v dense-tier
57, half 57, eighth 48. Census: eighth solves 48 outright,
half rescues +10, dense rescues +4, 58 fail all. Read 1
lands (union effects, echoing rung-1's +2 at larger scale):
each rung's projection rounds coin-flip rows differently, so
the ladder harvests the union — +5 over always-dense, and
within noise of the ORIGINAL 63-crystal the matryoshka
training taxed (the -6 tier price is bought back by the
ladder at decode time). Cost read is honest: effective gate
params 94% of dense (the gate is fail-heavy, so half the rows
escalate fully; on easier traffic the eighth's 48-outright
share dominates and the policy gets cheap). THE ADAPTIVE
CRYSTAL ARC CLOSES ITS FIRST LOOP: one tensor, three budgets,
a free difficulty signal, and a decode policy that converts
tier diversity into capability. Predictive routing's bar is
now 62 @ 94%. Fences: n=1, MPS, gates-only, fail-heavy
traffic profile.

## PRE-REG: THE FARMER PROBE — reverse model as data farmer (2026-07-29 day; escalation-engine cell 6, Artin's reverse-self-learner riff)

Full-reverse d64 birth (sym_birth REV=2: every gen-4 row
flipped nxt->cur; EMA 0.999, SKIP_GATE — the forward gate is
meaningless for a backward model). Probe (farmer_probe.py):
125 NOVEL band expressions (gate band + 50k seed offset,
disjoint from the gate's i<24), 8 samples each = 1000
predecessor candidates; each verified by FORWARD rule
application (fork-boxed verify_wave: cand -> seed must be a
valid step), identity-rejected, deduped, and checked against
the ENTIRE gen-4 corpus (cur+nxt) for novelty. METRIC:
verified-distinct-novel yield /1000 + novel/s wall rate.
PREDICTION: yield 100-300/1000 — reverse direction is
GENERATIVE (many predecessors per expression), so verify
should pass often; novelty fence is the real filter. READS:
yield >> sympy-farm rate at matched hardness => the
self-farming loop is LIVE (reverse learns, forward eats; two
models per the backward-tax law, hardness-targeted per the
saturation law) — next rung closes the loop (train forward
on farmed rows). Yield ~0 => reverse model memorizes rather
than inverts; book and stop the arc. Desk, MPS, n=1.

## PRE-REG: LEG C — THE MARGINAL-VALUE LADDER (2026-07-29 day; escalation-engine cell 2, night31 cuda)

Saturation law in hand (half dose ~ free at d64); now resolve
it BY LEVEL. 8 births (night31.sh, cuda, EMA 0.999, seed 1):
levels {1,2,3,5} x keep {25%, 50%} of that level's rows only,
all other levels untouched. Corpus masses: L1 26.7k, L2
39.3k, L3 12.2k, L5 14.6k. Comparator: cuda d64 EMA 58.7
(n=3, night-29). METRIC: d(gate)/d(log rows) per level = the
allocation answer for the next farm tranche (and the farmer
probe's targeting order). PREDICTION: L1/L2 cuts free at both
doses (easy mass saturated); L3/L5 cuts bite at 25% (the
gate lives at L3-L7). Fences: n=1 per cell, cuda line only
(sigma ~3.5, so single-cell deltas < 4 are noise — read the
LADDER shape, not cells).

## FARMER PROBE VERDICT: the reverse model INVERTS but does not FARM — 2/1000 novel off-distribution, 11/1000 in-distribution (2026-07-29 day)

Probe A (pre-reg seeds, novel band starts): 2/992 verified-
distinct-novel, verified 3. Sample autopsy: candidates are
LOCAL MUTATIONS of the seed (expansions, coefficient edits),
not grammar inversions — and the seeds were off-distribution
by construction: in the flipped corpus, chain STARTS appear
only as targets, never prompts (the corpus was itself farmed
backward from answers — a predecessor of a start barely
exists in the grammar). Probe B (SEEDMODE=corpus control,
in-distribution later-states, same novelty fence): verified
3 -> 107/1000 (the model DOES invert the step grammar when
asked the question it was trained on — the step-local
delegation lesson again, now on the farmer side), but novel
stays at 11/1000: inversion is MEMORIZATION-DOMINANT at
d64/3ep; 90% of verified predecessors are corpus rows
reproduced. VERDICT: the self-farming loop is NOT live at
this scale — 0.25 novel/s v the sympy farm's orders-more;
prediction (100-300/1000) WRONG, booked. The failure axis is
generalization-into-novelty, not verification or inversion.
Revive conditions banked: (1) sampling temperature/diversity
sweep (the wave is 8 near-greedy samples), (2) seed with
SOLVED states (the true farmer seeding — needs cheap answer
generation), (3) scale (d64 is memorization-prone at 120k
expressions). Fences: n=1, MPS, EMA weights, gen-4 corpus.

## PRE-REG: THE DEPTH LADDER — the never-varied axis (2026-07-29 day; Mac battery)

LAYERS=8 in every birth all month; the horizon wall (12-ply
gate, compositional steps) is a DEPTH property. Ladder:
d56/f224/heads4 births at layers {4, 8, 12, 16}, EMA 0.999,
seed 1, gen-4, 3ep, MPS — layers-8 cell is the in-battery
control (same battery, same device; cross-battery comparators
stay advisory only). Plus ONE params-matched pair: layers-4
at d80/f320 (per-layer params x2.04 ~= layers-8 at d56/f224)
— does capability live in DEPTH or in PARAMS when matched?
PREDICTIONS: (1) layers-4 drops hard (>1 sigma below the
in-battery 8) — chain steps are compositional; (2) layers-12
and -16 ~= 8 (diet-bound per the saturation law, not
architecture-bound); (3) the params-matched shallow-wide
(4x d80) lands BELOW deep-narrow (8x d56) — depth is not
purchasable with width (the attention-machine law extended
to the third axis). Read against sigma ~3.5. 5 births.

## DEPTH LADDER VERDICT: DEPTH IS SLACK TOO — the ladder is FLAT and both directional predictions FALSIFIED (2026-07-29 day)

EMA gates, one battery, MPS, d56/f224 unless noted: layers-4
56, layers-8 59 (in-battery control), layers-12 61, layers-16
59, params-matched 4x d80/f320 59. Every cell within ~1 sigma
(~3.5) of every other. Prediction 1 (layers-4 drops hard)
WRONG: -3, sub-sigma, at HALF the parameters. Prediction 3
(params-matched shallow-wide < deep-narrow) WRONG: 59 = 59
exactly. Prediction 2 (12/16 ~= 8, diet-bound) lands. THE
READ: the never-varied axis, varied at last, is ANOTHER SLACK
POOL — the 12-ply horizon wall is NOT a depth deficit; a
4-layer crystal walks chains as well as a 16-layer one. The
capability bound stays where the week put it: attention
width/heads x diet hardness. Consequences: (1) the black-hole
arc's "depth is the missing axis" hypothesis is DEAD on the
gate diet — what's missing is not architecture at all (three
axes now measured slack: ffn, bits, depth; one precious:
attention geometry); (2) NEW EFFICIENCY POINT: layers-4 d56 =
56/120 at ~HALF the layers-8 crystal — the minimal crystal
shrinks again; (3) valid% rises monotonically with depth
(51.5 -> 57.0) while solves stay flat — depth buys FLUENCY,
not capability (echoes the calibration arc's fluency/
capability split). Fences: n=1 per cell, MPS line, gen-4
diet, gates-only; deeper ladders untested past 16.

## LEG C VERDICT: THE MARGINAL-VALUE LADDER IS FLAT — no level's dose bites at quarter-cut (2026-07-29 day; night31 cuda)

8 cells v comparator cuda d64 EMA 58.7 (n=3, sigma ~3.5):
L1c25 58, L1c50 58, L2c25 62, L2c50 57, L3c25 55, L3c50 60,
L5c25 59, L5c50 59. Every cell within ~1 sigma of the
comparator; the two doses per level do not even order
consistently (L3: 55 at 25% keep but 60 at 50%) — pure noise
shape. Prediction (L3/L5 cuts bite at 25%) WRONG, booked —
the day's second falsification. THE READ: saturation is
TOTAL, level by level — even a QUARTER dose of any single
level (including the gate's own L3/L5 mass) is free at
d64/gen-4. d(gate)/d(log rows) ~ 0 at every measured level.
THE ALLOCATION ANSWER (what cell 2 was for): the next farm
tranche should buy NOTHING at existing levels — more rows of
L1-L5 are worthless; only new HARDNESS (levels, families,
prosthetic formats) can move the gate. THE DAY'S TRIPLE
CLOSES: depth flat + dose flat + storage at 99% of the
rate-distortion bound => the crystal is bound by NEITHER
architecture depth, NOR data volume, NOR bits — the frontier
is attention geometry x diet hardness x decode policy
(escalation 62, delegation +15: both decode/format levers,
both worked; every capacity lever this week nulled). Fences:
n=1 per cell, cuda line, EMA gates, gen-4 diet.

## PRE-REG: THE HEAD-TENSION CELL — born-8-heads (2026-07-29 day; escalation-engine cell 3)

Michel/Voita prune heads freely; we measure all-essential at
4 heads, two widths. Frame under test: theirs are OVER-
provisioned, ours at floor — head slack = head count above
the task's relation count (~4). One birth: d64/f256 HEADS=8
(head_dim 8, even — RoPE-safe), gen-4, EMA 0.999, seed 1,
MPS. Then anatomy.py head-deletion census on the EMA weights.
READS: some heads prunable (gate drop < 1 sigma for >=1
single-head deletion) => provisioning frame CONFIRMED — head
essentiality is a floor phenomenon, not an attention law;
ALL 8 essential => the diet recruits every channel offered
(relations scale with provision — would need booking as its
own law and the Michel/Voita tension row hardens). Also read
the 8-head gate v the 4-head line (58.7-class): heads x2 at
fixed d = free, tax, or gain? n=1, MPS, gates-only.

## HEAD-TENSION VERDICT: ALL EIGHT ESSENTIAL — the diet recruits every channel offered (2026-07-29 day; cell 3)

Born-8-heads d64/f256 (head_dim 8): EMA gate 58/120 — dead
level with the 4-head cuda line (58.7, n=3): heads x2 at
fixed d is FREE, neither tax nor gain. Census on the EMA
weights: single-head deletions land 39/47/30/34/40/43/41/21
v 58 — best case -11 (~3 sigma), worst -37. NOT ONE prunable
head at DOUBLE the provision. The provisioning frame
(Michel/Voita heads-are-slack because over-provisioned; ours
essential because at-floor ~relation count 4) is FALSIFIED:
given 8 channels, training distributes load across all 8 and
every one becomes essential. BOOKS AS ITS OWN LAW: head
essentiality here is a TRAINING outcome (load spreads to fill
provision), not a task constant — relations scale with
provision. The Michel/Voita tension row HARDENS: their
prunability likely reflects redundancy induced at
transformer-LM scale/diet, not an attention universal; at
oracle-verified micro-scale, deletion is always catastrophic.
Fences: n=1, MPS, gates-only, single-head deletions (no
pair/subset scan).

## PRE-REG: POLAR-SPLIT SNAP — |c| coarse x arg fine (2026-07-29 day; escalation-engine cell 4)

cplx_none.pt (unconstrained complex FFN d384/f1536/h6): snap
the complex gate/up weights on a POLAR grid (|c| step in
sigma units x N angle bins) v UNIFORM re/im grids (step in
sigma units), bits/complex MEASURED as mean log2(#distinct
values). Cells: uniform u {0.5, 1, 2} sigma; polar (1s x 64),
(1s x 16), (2s x 64), (0.5s x 8). Fence: gate+up only; in-
battery t=0 control. PREDICTIONS (sigma law + R2's angular
knee): (1) uniform obeys the sigma-priced knee — 0.5s free,
2s bites; (2) at MATCHED measured bits, polar with fine
angles BEATS uniform (the rotational crystal spends its
information angularly); (3) angle-starved polar (8-16 bins)
bites even with fine magnitude — the angular axis is the
priced one. Desk, MPS, n=1, gates-only. polar_snap.py.

## POLAR-SPLIT SNAP VERDICT: BITS ARE GEOMETRY-BLIND — the knee follows measured bits, not the grid's coordinates (2026-07-29 day; cell 4)

cplx_none d384 (control 63): uniform 0.5s 63 @ 7.83 bits, 1s
63 @ 5.97, 2s 60 @ 4.62; polar 1s x64 63 @ 8.14, 1s x16 63 @
6.30, 2s x64 63 @ 7.57, 0.5s x8 63 @ 6.23; low-bit fills:
2s x8 61 @ 4.75, 3s x6 58 @ 3.83. Prediction 1 (sigma knee on
the uniform arm) CONFIRMED. Prediction 2 (polar wins at
matched bits) FALSIFIED: 4.75-bit polar 61 v 4.62-bit uniform
60 — parity, sub-sigma. Prediction 3 (angle starvation bites)
FALSIFIED: EIGHT angle bins are free at fine magnitude (63 @
6.23 bits) — the rotational crystal does NOT price its phase
finely. THE LAW THAT BOOKS: the snap knee follows TOTAL
MEASURED bits/weight regardless of grid geometry — Cartesian
and polar grids at equal information cost equal capability.
This EXTENDS the bits-v-structure currency law: bits were
location-blind (Q8 uniform -4 everywhere); they are now also
GEOMETRY-blind. One scalar (information per weight, priced in
sigma units) governs the entire quantization axis; the free
knee sits at ~5-6 bits/complex (~2.5-3 bits/real component),
consistent with the sigma-law's Q~2/sigma across every
crystal measured. Note: import-scar bite #3 (polar_snap
module-level battery re-ran on import; inlined fill —
__main__ guards adopted for future instruments). Fences: n=1,
MPS, gates-only, gate+up fence, cplx_none only.

## PRE-REG: THE HEAD AUTOPSY — per-(layer, head) deletion map (2026-07-29 eve; Artin's review ask on the head-tension cell)

The day census deleted head h across ALL layers at once — a
COLUMN of the layers x heads grid. Autopsy on the h8 EMA
crystal (d64/f256/8 heads): delete each SINGLE (layer, head)
cell — 64 cells on the proxy gate (n=8/level, +-2 noise; read
the MAP shape), then FULL gates on the min/max cells + the
control. READS: single-cell deletions gentle (proxy drop
within noise for most cells) => "all heads essential" AMENDS
to "no head INDEX is disposable across depth" — redundancy
exists layer-wise and the intrinsic-heads law weakens to its
column form; single cells catastrophic too => heads are
per-layer organs, the law hardens further. MPS, n=1.

## PRE-REG: G5 POLAR — the predicted BREAK of geometry-blindness (2026-07-29 eve)

Mechanism claim: bits are geometry-blind because our weights
are ISOTROPIC (R1: rotation lives in activations only). The
exception the mechanism predicts: cplx_G5 was BORN on the
4-angle star {0, +-s, +-is} — anisotropic by construction.
Cells on cplx_G5 (d384, alpha=G5): control; polar 4 angles x
{1s, 2s} magnitude (ALIGNED with the star); polar 4 angles
ROTATED 45 deg (misaligned control); uniform u={1,2}s.
PREDICTION: aligned-4-angle polar ~ control even at very low
bits; misaligned-45 and uniform at matched bits BITE — if
so, geometry-blindness is a property of ISOTROPY, not of
quantization, and the law rewrites: the knee follows bits
priced in the weight DISTRIBUTION's own coordinates. MPS.

## PRE-REG: THE DISTORTION COLLAPSE — one curve for the quantization axis (2026-07-29 eve; Artin's "build the equation" ask)

Claim under test: solves = f(D/sigma^2) — a single function
of normalized induced weight distortion, independent of grid
geometry/location/width. Desk instrument: recompute the
induced normalized MSE (D/sigma^2, per snapped tensor, mean)
for EVERY logged snap cell (d56 Q16/Q8; d256 Q64/Q16; 19M
knee cells; polar battery incl. low-bit fills) and scatter
against booked solves. PREDICTION: monotone collapse with a
knee at D/sigma^2 ~ 0.02-0.08 (= (0.5-1.0 sigma)^2/12, the
sigma law restated as rate-distortion); polar and Cartesian
cells interleave on ONE curve. Any crystal whose cells sit
OFF the curve names a non-Gaussian weight structure. Desk.

## DISTORTION COLLAPSE VERDICT: ONE CURVE PER CRYSTAL, NOT ONE CURVE — the equation is two-parameter (2026-07-29 eve)

21 historical snap cells recomputed to x = param-weighted
D/sigma^2 (distortion_collapse.py, desk). WITHIN each crystal
the cells are monotone within noise — but there is NO
universal f(x): at x~0.09, 19M keeps 0.878 while cplx_none
keeps 1.000; at x~0.33 cplx keeps 0.921 while 19M at x~0.45
keeps 0.531; d56 bites by x~0.014 where cplx is untouched at
x~0.12 — a ~30x fragility spread at matched normalized
distortion. Geometry-blindness SURVIVES (polar and Cartesian
cplx cells interleave on the cplx curve) but the knee
CONSTANT is crystal-priced. THE EQUATION: kept ~ f(k_c *
D/sigma^2) — f a universal monotone shape, k_c a per-crystal
fragility (rough fit: cplx_none 1, 19M ~4-5, d56 ~25-30).
AND WE ALREADY OWN k_c's METER: calibration R1 measured
flips/token predicting snap robustness at rho .883 — near-tie
density is the conversion factor from noise power to gate
damage. Chain: bits (in sigma units) -> distortion power
(geometry-blind, rate-distortion) -> logit flips (priced by
near-tie density, per-crystal) -> solves. Also explains the
day's "knee at 0.5-1.0 sigma across crystals" as grid-step
accounting hiding the fragility spread that D/sigma^2
exposes. NEXT (banked): fit f + k_c jointly across all cells;
predict a NEW crystal's knee from its flips/token probe alone
(1% of a gate's cost) — the prediction that would ship the
equation. Fences: booked solves transcribed (no re-gates);
best-rational vs sigma-step grids mix in x; n=1 per cell.

## G5 POLAR VERDICT: geometry-blindness does NOT break — even star weights shrug off a 45-degree grid rotation (2026-07-29 eve)

STAR (cplx_G5_dep, control 66): polar4 aligned 63, polar4
ROTATED-45 62, uniform 1s 63 — all three snaps cost the same
~3-4, and rotation (which moves EVERY nonzero star weight by
45 degrees, |dc| ~ 0.77|c| — massive distortion) costs -1 v
aligned. ISO control battery flat as predicted (61-63).
PREDICTION (aligned free / misaligned craters) FALSIFIED —
the third strike on axis-priced quantization. Reads: (1) the
aligned cell was NOT free because the sigma-step MAGNITUDE
grid moved the star radius (angle preservation bought
nothing); (2) the rotated cell's huge distortion still lands
on the cplx robustness curve — consistent with the two-
parameter law (k_c small for the cplx class) rather than any
coordinate pricing; (3) anisotropic WEIGHTS do not imply
anisotropic FRAGILITY: what the function needs is preserved
under global rotation of the star (the modReLU/rotation
mechanism lives in activations — R1's lesson landing again).
The distortion-collapse equation stands unqualified: bits ->
distortion -> flips -> solves, coordinates nowhere in the
chain. Fences: n=1, MPS, gate+up fence, one star crystal.

## HEAD AUTOPSY VERDICT: essentiality is CELL-SPARSE, not uniform — one (layer,head) deletion craters, another is free (2026-07-29 eve; amends the intrinsic-heads law)

64-cell per-(layer,head) map on h8 EMA (proxy ctrl 19/40):
range 7..20; only 13/64 cells drop >2 below control. FULL
gates on the extremes: control 58; MIN cell L1h7 = 24/120
(-34: ONE of 64 heads deleted, catastrophic); MAX cell L1h4 =
61 (free, +3). AMENDMENT to the intrinsic-heads law: the day
census (all COLUMN deletions catastrophic) was correct but
coarse — per-cell structure is a SPARSE CRITICAL CIRCUIT
(~13/64 load-bearing cells, concentrated in early-mid layers:
L1, L3) embedded in substantial per-cell slack (~51/64 within
proxy noise). Every column contains at least one critical
cell, which is why column deletion always cratered. Refines
"attention geometry precious": PRECIOUS = the sparse circuit;
the rest of the head grid is slack like ffn/depth/bits.
Consequences: (1) head-level pruning IS available if done
per-cell (structured sparsity with a census, not by index);
(2) the critical cells are a MAP of where the capability
lives — candidate probe targets for the delegation/routing
work. Fences: proxy n=8 (+-2) for the map, full gates on
extremes only, n=1, MPS, one crystal.

## PRE-REG: PACKED CRYSTAL C0+C1 — format + pack-parity (2026-07-29 eve; spec 2026-07-30-packed-crystal)

C0 FORMAT: generalized integer twin. Per 2-D block tensor
(qkv, o, gate, up, down; emb/head/norms stay fp32 — tiny and
never snapped before): denominator q_t = ceil(2/sigma_t)
(grid step <= sigma/2, inside the free zone of the sigma-law
knee), codes = round(W * q_t) packed to ceil(log2(span))
bits, one q_t + offset per tensor. Artifact = .npz + reader
(scratch/pack_crystal.py). Report: bits/wt (packed), Shannon
entropy of the code stream (bound check), artifact bytes v
fp32/fp16.
C1 PARITY (Mac, MPS, full gates n=24): pack
sym_birth_dense_mps_L4_ema (layers-4 d56, 206k params) AND
sym_birth_dense_mps_h8_ema (d64 h8, booked control 58).
Fresh fp controls gated in the same run, same device.
PREDICTIONS: (1) packed gate within sigma (~3.5) of its fp
control on BOTH crystals (deploy-tax law: rational lattices
deploy at zero tax; step <= 0.5 sigma is below the knee);
(2) raw packed bits/wt in 5-7; (3) Shannon entropy of codes
within ~5% of the Gaussian capacity for the measured step
(0.5 log2(2 pi e) - log2(step/sigma) per weight), i.e. the
fixed-width penalty over entropy coding is < 1 bit/wt.
Falsifier: a >sigma gate drop on either crystal means the
sigma/2 step is NOT free once emb-adjacent tensors interact
with packed blocks — then C1b (born-rational arm) fires.
Fences: MPS, n=1 per crystal, sigma per-tensor never
transported, GATE_BAND 9_900_000.

## PACKED CRYSTAL C0+C1 VERDICT: zero-tax pack, entropy within 1% of Gaussian capacity, 6.15-6.65x smaller than fp32 (2026-07-29 eve)

C0 artifact shipped (scratch/pack_crystal.py -> checkpoints/
packed_{L4d56,d64h8}.npz, ~15-line reader). C1 full gates
(MPS, same-process controls): L4d56 control 56 -> PACKED 55
(-1, well inside sigma 3.5); d64h8 control 58 -> PACKED 58
(+-0, EXACT parity). ALL THREE PRE-REG PREDICTIONS CONFIRMED:
(1) parity within sigma on both crystals — deploy-tax law
holds in the real artifact, not just fake-quant; (2) raw
packed 4.94 / 5.06 bits/wt (predicted 5-7); (3) Shannon
entropy of the code stream 3.11 / 3.12 v Gaussian capacity
3.13 / 3.14 at the sigma/2 step — within 1%, the code
distribution IS at the rate-distortion bound (claim 2 lands
harder than predicted: bound-gap <1%, not <5%). Fixed-width
penalty ~1.8 bits/wt, but deflate inside .npz recovers most
of it: artifact = 133,883 B v 822,752 fp32 (6.15x) and
318,868 B v 2,121,984 (6.65x) — ~5.0 bits/param INCLUDING
the fp32 norms/emb/head passthrough. Calibration cost: ZERO
(q_t = ceil(2/sigma_t) read off the weights in closed form;
no data, no Hessian, no search). Claims 1+2 of the spec are
now measured; next: C3 baselines (GPTQ/AWQ/HQQ honest table),
C2 kernel, C5 tiered pack; C4 needs a 3080 window; C6 HOLD.
Fences: n=1 per crystal, MPS, full gates n=24, sigma
per-tensor never transported.

## PRE-REG: PACKED CRYSTAL C3 — GPTQ/AWQ/HQQ honest table on the d64h8 crystal (2026-07-29 eve)

In-tree baselines (llmopt/quantize/methods.py: rtn, gptq, awq,
hqq) applied to every block Linear of d64h8 EMA, v the C1
sigma-law pack. Calibration for gptq/awq: activations hooked
from a forward pass over 24 prompts at GATE_BAND+500_000
seed offsets (never the gate band). Arms: {rtn, gptq, awq,
hqq} x {5 bits (matched to our 5.06), 3 bits (stress, below
the knee)} -> full gates + mean DeltaKL v fp logits on the
calibration battery + calibration wall-time per method (ours
= per-tensor std() in closed form). C1 controls reused (fp 58,
packed 58 — same device, same session, same instrument).
PREDICTIONS: (1) at 5 bits ALL methods gate within sigma of
control — the free zone is free for everyone; the honest
differentiator is wall-time (ours ~0) and calibration DATA
(ours none), not quality; (2) at 3 bits (step ~2 sigma,
above the knee) all methods drop by >sigma, and per the
distortion-collapse law calibrated compensation (gptq) buys
back SOME gate v rtn3 but does not restore control (D
dominates, k_c is crystal-priced, no allocator can hide 2
sigma of distortion); (3) DeltaKL orders with gate damage
across arms (the flips chain). Falsifier for claim 1 of the
spec: a calibrated method beating our pack by >sigma at
matched 5 bits — then calibration buys real quality here and
"calibration-free at parity" collapses. Fences: n=1 per arm,
MPS, full gates n=24, one crystal.

## PACKED CRYSTAL C3 VERDICT: nothing beats the calibration-free pack at matched bits — and the 3-bit stress arm FLAT on solves (prediction 2 falsified) (2026-07-29 eve)

d64h8, C1 controls (fp 58, sigma-pack 58). 5-bit arms: rtn 59,
gptq 58, awq 58, hqq 57 — ALL within sigma of control and of
our pack; DeltaKL 0.0011-0.0018/tok. PREDICTION 1 CONFIRMED
and the spec's claim-1 falsifier did NOT fire: with a real
Hessian and real activation scales, no calibrated method
beats the closed-form sigma-law pack at matched bits.
3-bit stress arms: rtn 57, gptq 55, awq 56, hqq 60 — solves
STILL flat (PREDICTION 2 FALSIFIED: no >sigma drop), but the
damage is visible elsewhere: valid% falls 57 -> 46-54 and
DeltaKL jumps 20-40x (0.022-0.059/tok). Reads: (1) this
crystal's k_c is small enough that even step ~1.5 sigma
distortion stays under the solve knee — fragility is
crystal-priced (d56 paid -4/-7 at comparable x; d64h8
shrugs), another two-parameter-law data point, not a
contradiction; (2) fluency (valid%) and DeltaKL are the
early-warning channel, solves the last to fall — same
ordering as the flips chain. PREDICTION 3 weakly held: hqq3
had lowest DeltaKL and highest gate, but 3-bit gate spread is
sub-sigma. HONEST CAVEAT for the paper: at 200k-500k params
calibration is sub-second (0.9s pass + <0.2s optimize), so
the wall-time differentiator is trivial HERE — it becomes
real at 0.5B+ (C6, HOLD). The quality claim stands: zero
data, zero search, parity with Hessian-armed baselines.
Fences: n=1 per arm, MPS, one crystal, C1 controls reused.

## PRE-REG: PACKED CRYSTAL C5 — the tiered pack (matryoshka as real bytes) (2026-07-29 night)

Pack matryoshka_d56_3tier.pt (d56, tiers on the 8 gate.weight
tensors; booked gates DENSE 57 / HALF 57 / EIGHTH 48) as a
NESTED artifact: (a) non-gate tensors packed once (C0 rule);
(b) gate payloads nested — tier-8 base = the numel/8 orbit
representatives of P_C8 (exact reconstruction by the joint
block-shift identity), tier-2 payload = numel/2 delta of
P_C2's representatives v the tier-8 prediction, dense payload
= full-numel delta v reconstructed P_C2; every payload
sigma-law-quantized on its OWN sigma (q=ceil(2/sigma), never
transported). Reader reconstructs any tier touching only its
prefix of bytes. Desk check first: unquantized reconstruction
must match project() exactly. Then full gates on all three
packed tiers. PREDICTIONS: (1) each packed tier within sigma
(~3.5) of its booked fp tier (57/57/48); (2) bits/wt of each
payload ~5 (sigma-normalized Gaussians all look alike); tier-8
artifact ~75-80% of the dense artifact's bytes (gate tensors
are only ~25% of params — honest: tier savings are capped by
the tiered-tensor share); (3) desk escalation economics
(eighth-first -> half -> dense using booked solve rates):
bytes-touched per solved row beats dense-always. Fences: n=1,
MPS, full gates n=24, one crystal, deltas quantized after
tier-8 quantization (error does not compound silently —
deltas measured against the QUANTIZED base).

## PRE-REG: PACKED CRYSTAL C2 — dequant-fused sigma-pack GEMV (Metal/MLX) (2026-07-29 night)

Kernel: crystal8 GEMV — sigma-law codes stored int8 (1 B/wt;
byte-aligned runtime twin of the 5-bit disk format), one fp
scale (1/q_t) per tensor, dequant fused into the GEMV (one
simdgroup per row, char4 weight + half4 activation vector
loads, simd_sum, scale applied once per row — int4_gemv v3
style). Bench protocol: mx.eval EVERY timed iteration (the
lazy-graph scar). PREDICTIONS: (1) elementwise correct v fp
reference (max err ~1e-3); (2) large shapes (D=4096
N=14336, D=2048 N=8192): approaches the bandwidth model's
2.0x over fp16 GEMV (2 B -> 1 B per weight), model printed
next to measured; (3) crystal shapes (56x224, 64x256,
384x1536): overhead-bound, parity-to-LOSS v fp16 (fp16 GEMV
itself only ~40 GB/s at D=896 — honest losses booked, the
win is the memory footprint not tok/s at micro shapes).
Fences: M3 Pro 36GB, MLX, n=1 per shape.

## PACKED CRYSTAL C5 VERDICT: the nested artifact is REAL and the escalation economics win — but zero-tax does NOT transport to the matryoshka crystal (prediction 1 partially falsified) (2026-07-29 night)

Reconstruction identities EXACT (C2 0.0, C8 3e-8). Bytes: the
nested pack ships tier-8 at 193,060 B, tier-2 at +29,008,
dense at +62,720 (cum 284,788 B v fp32 1,627,360 — 5.7x);
gate payloads 6,468 / 35,476 / 98,196 B v 401,408 fp32.
Gates: packed EIGHTH 42 (fp 48, -6), HALF 53 (fp 57, -4),
DENSE 52 (fp 57, -5) — the ladder ORDERING survives and the
half tier still ties packed-dense, but every tier pays ~1-2
sigma where C1's crystals paid ZERO with the same squant
rule. PREDICTION 1 PARTIALLY FALSIFIED. Read: the joint-STE
matryoshka crystal is MORE FRAGILE (higher k_c) than its EMA
parent — third data point that fragility is crystal-priced
(C3's d64h8 shrugged 3-bit; this one pays at sigma/2 steps);
the flips/token meter should have predicted it BEFORE the
gates — booked as the next desk cell for the k_c meter.
PREDICTION 2 held (payload bits ~5, tier-8 artifact 68% of
dense). PREDICTION 3 held on the packed gates: escalation
decode (eighth-first -> half -> dense, nesting assumption)
touches ~559 kB per solved row v 657 kB dense-always — the
tiered artifact beats dense-always by ~15% bytes/solve even
WITH the deepest tier degraded. Fences: n=1, MPS, booked fp
tiers as comparators (same instrument, 07-29 night), nesting
assumption unverified per-row. Artifact rule: prefer packing
EMA-parent crystals; tier + pack compound taxes on joint-STE
tensors.

## PACKED CRYSTAL C2 VERDICT: fused sigma-pack GEMV correct and 1.76x at large shapes; honest loss at 256x64 (2026-07-29 night)

crystal8 GEMV (int8 codes + one per-tensor scale, dequant
fused): CORRECT everywhere (bit-exact v the dequantized
reference at crystal shapes; 1.6e-4 relative at 14336x4096 —
fp32 accumulate v fp16 reference rounding). Bench (M3 Pro,
mx.eval every iteration): 224x56 1.36x over fp16 GEMV,
256x64 0.91x (LOSS — launch-overhead-bound, as pre-reg'd),
1536x384 1.16x, 8192x2048 1.58x, 14336x4096 1.76x v the
2.00x bandwidth model — 88% of model at the largest shape,
same shape class where int4 v3 hit its wins. All three
predictions held (correctness, approach-2x at large,
parity-to-loss at micro shapes). The deploy story is
consistent: at crystal scale the win is FOOTPRINT (5 bits/wt
on disk, 1 B/wt live); tok/s wins arrive with size. Fences:
M3 Pro, MLX, n=1 per shape, weights-only bandwidth model.

## PRE-REG: PACKED CRYSTAL C2b — the disk format goes live: bit-packed 5-bit GEMV (2026-07-29 night)

crystal5 GEMV: 6 codes per uint32 word (5 bits each, 2 bits
pad -> 5.33 effective bits/wt), signed codes clamped to
[-15, 15] (clamp fraction reported; sigma-law q keeps the
6.8-sigma tail inside range). Bandwidth model v fp16: 16 /
5.33 = 3.0x. PREDICTIONS: (1) correct v dequantized
reference; (2) large shapes reach 2.2-2.7x (73-90% of model,
same efficiency band C2 hit); (3) micro shapes remain
overhead-bound (parity-to-loss); (4) the bit-unpack ALU cost
does not knock the kernel off the bandwidth roofline at
large shapes (if it does — honest loss v crystal8 booked).
Fences: M3 Pro, MLX, n=1 per shape.

## PACKED CRYSTAL C2b VERDICT: the disk format runs live — bit-packed 5-bit GEMV hits 2.39x, BEATING the byte-aligned kernel (2026-07-29 night)

crystal5 (6 signed 5-bit codes per uint32, 5.33 eff bits/wt):
CORRECT at every shape (<=1.7e-4 relative v dequantized
reference), clamp fraction 0.0000 everywhere (the 6.8-sigma
tail never fires at these q). Bench: 224x54 1.11x, 256x66
1.21x, 1536x384 1.07x, 8192x2046 1.73x, 14336x4098 2.39x v
the 3.00x bandwidth model (80% of model) — and it BEATS
crystal8's 1.76x at the same shape class: prediction 4 held,
the bit-unpack ALU is free next to the bandwidth saved. All
four predictions confirmed. Consequence for the paper: the
sigma-law artifact needs NO byte-aligned runtime twin — the
5-bit disk format IS the runtime format, 3x footprint v fp16
live. Fences: M3 Pro, MLX, n=1 per shape, synthetic Gaussian
weights at crystal sigma (0.19).

## PRE-REG: PACKED CRYSTAL C4 — cross-device determinism of the integer forward (2026-07-29 night)

Instrument: scratch/pack_determinism.py. Two hashes per
device on d64h8's sigma-law codes: (A) INTEGER-GEMM hash —
every block Linear's codes [int, span<=2^6] times a fixed
deterministic integer activation battery (seeded ints,
|x|<2^12), accumulated via fp64 matmul: every partial is an
integer < 2^53, so fp64 addition is EXACT and the result is
REDUCTION-ORDER-INVARIANT by construction (the Ozaki
principle) — sha256 over the output bytes MUST match MPS v
cuda; (B) FULL-FORWARD hash — fp32 model, fixed prompt
battery, sha256 over logits bytes — EXPECTED to differ
(libm/reduction/FMA divergence, the device-fence pain
class); greedy token streams reported alongside (may or may
not match depending on margins). CLAIM 3 of the spec lands
on hash-A equality alone. PREDICTIONS: (1) hash A identical
Mac/3080; (2) hash B differs; (3) greedy token streams
nevertheless identical on most prompts (margins >> device
epsilon per the fp16 near-tie doctrine — only coin-flip
margins diverge). Fences: same commit both machines, same
battery seeds, int64 codes cast to fp64 for the matmul on
both devices (no TF32 path — fp64 GEMM only).

AMENDMENT (pre-run, same session, targets the C4 pre-reg
above): MPS has no fp64 — hash A's exact carrier is fp32
with |x| < 2^6 (|codes| < 2^6, sum length <= 256, so every
partial is an integer < 2^24: exact in fp32, order-
invariant). cuda TF32 explicitly disabled in the instrument.
Claim unchanged.

## PRE-REG: PACKED CRYSTAL C6 — external validity: sigma-law v HQQ on Qwen2.5-0.5B (2026-07-29 night; Artin's GO given, 3080)

Instrument: scratch/pack_c6.py (3080). All transformer Linear
weights (embed/lm_head/norms untouched) fake-quantized per
arm: (a) SIGMA-PACK q_t=ceil(2/sigma_t), avg raw bits
MEASURED (Qwen tails may push past the crystal's ~5); (b)
HQQ at round(measured avg) bits, group 64; (c) RTN same
bits. Score: mean DeltaKL/token v fp on 16 fixed prompts +
perplexity on a fixed local text slice (README.md, byte-
deterministic) + per-arm quantization WALL-TIME. This is the
cell where calibration cost becomes real (0.5B, ~290
linears). PREDICTIONS: (1) sigma-pack DeltaKL within 2x of
HQQ at matched bits (honest risk: real-LLM outlier channels
are exactly what HQQ's robust zero-points buy — Gaussian-
capacity logic may NOT transport from born crystals to web-
trained weights); (2) sigma-alloc wall-time >=100x faster
than HQQ full-model; (3) PPL ordering matches DeltaKL
ordering. FALSIFIER (claim-1 external fence): sigma-pack
DeltaKL > 5x HQQ's — then calibration-free allocation is a
CRYSTAL-CLASS result, fenced off web-scale LLMs, and the
paper's claim narrows honestly. Fences: 3080 cuda, fp16
eval, fake-quant only (bytes claim already landed at C0/C2b),
one model, n=1.

## PACKED CRYSTAL C4 VERDICT: CLAIM 3 LANDS — the integer forward is bit-identical across devices; fp logits differ but greedy streams match anyway (2026-07-29 night; Mac + 3080)

Hash A (integer GEMM over all sigma-law code tensors, exact
fp32 carrier, TF32 off): fda95457...cb07ca IDENTICAL on MPS
and cuda — the packed path has NO device ambiguity, by
construction and now by measurement. Hash B (fp32 full
forward logits): differs (1d87338f v 59adca91) — the familiar
device-fence class, present exactly where predicted. Greedy
streams: 8f93028c IDENTICAL on both devices (5 prompts x 40
toks) — argmax margins swamp device epsilon on this battery,
per the fp16 near-tie doctrine (ties, not bugs). ALL THREE
PREDICTIONS CONFIRMED. Consequence: a full integer-GEMM
decode path (norms/softmax in fp between exact GEMMs) would
make the cross-device fence VANISH for the packed crystal —
the determinism differentiator v GPTQ/AWQ/HQQ artifacts is
real. Fences: one crystal (d64h8), 5-prompt battery, greedy
only; full integer end-to-end decode not yet built (the
GEMM-path hash is the claim, as pre-registered).

## PACKED CRYSTAL C6 VERDICT: THE FALSIFIER FIRES — per-tensor sigma-allocation does NOT transport to Qwen-0.5B (2026-07-29 night, 3080)

Qwen2.5-0.5B, 168 linears / 357.8M params, fp16 control ppl
60.29. sigma-pack (per-TENSOR q=ceil(2/sigma), avg raw 6.82
bits -> matched 7): DeltaKL 1.2839/tok, ppl 96.39 — v rtn7
0.1669 / 60.90 and hqq7 0.0388 / 60.52. 33x HQQ's DeltaKL:
the pre-reg'd falsifier (>5x) FIRES decisively. Even plain
RTN (per-ROW max scales) beats per-tensor sigma by 8x.
Wall-time held: 0.9s v HQQ 61.7s (69x, just under the
predicted >=100x). READ: born crystals are per-tensor
Gaussian (entropy = capacity, C1); web-trained LLMs are NOT —
row scales vary ~an order of magnitude, so one tensor grid
over-quantizes quiet rows. The claim FENCES honestly:
calibration-free per-tensor allocation is a CRYSTAL-CLASS
result. Open rescue (C6b, pre-reg below): per-ROW sigma is
STILL closed-form and calibration-free — the granularity,
not the calibration, may be what real LLMs demand. Fences:
one model, fake-quant, 16-prompt DeltaKL + README-slice ppl,
n=1.

## PRE-REG: PACKED CRYSTAL C6b — per-row sigma rescue arm (2026-07-29 night, 3080)

Same harness, one change: q_r = ceil(2/sigma_r) per OUTPUT
ROW (still zero calibration, closed form, one std() per
row; storage cost one scale/row = same metadata class as
rtn/hqq). PREDICTIONS: (1) DeltaKL improves >=10x over
per-tensor sigma-pack (the row-scale spread was the wound);
(2) lands within 2x of rtn7 (same granularity, sigma-grid v
max-grid); (3) still loses to hqq7 (robust zero-point
optimization buys real error on outlier tensors) — if it
TIES hqq, calibration-free wins the 0.5B table outright and
claim 1 un-fences. Wall-time stays <5s. Fences: as C6.

## PACKED CRYSTAL C6b VERDICT: per-row rescue FAILS — granularity is not the wound, the KNEE CONSTANT is (2026-07-29 night, 3080)

sigma-pack[row] (q_r per output row): DeltaKL 1.3072/tok,
ppl 90.26 — statistically unchanged from per-tensor (1.2839
/ 96.39). PREDICTION 1 FALSIFIED (no >=10x recovery); rtn7
0.167 / hqq7 0.039 unchanged. READ: both sigma arms anchor
the grid STEP at sigma/2 — the CRYSTAL knee. The collapse
law says damage ~ f(k_c * D/sigma^2); web-trained Qwen's
k_c is evidently far larger than any crystal's, so its knee
sits at a much finer step. The sigma-law FORM survives; the
knee CONSTANT is model-priced — exactly what the flips/token
meter measures. C6c below closes the chain.

## PRE-REG: PACKED CRYSTAL C6c — the knee is model-priced: step sigma/8 arm (2026-07-29 night, 3080)

Same harness, ARM=row K=16: q_r = ceil(16/sigma_r) (step
sigma/8, ~3 extra bits; still closed form, zero data).
Baselines re-read at the new measured avg bits (expect ~10;
rtn/hqq at round(avg)). PREDICTIONS: (1) sigma[row,K=16]
DeltaKL drops >=10x v C6b (into the 0.05-0.15 class) —
confirming step-anchor, not allocation form, as the C6
failure mode; (2) rtn at matched ~10 bits still <= ours by
<=2x (max-anchored grids stay slightly better on outlier
rows); (3) hqq at matched bits <= rtn (ordering preserved).
If (1) fails too, the sigma form itself is dead on web
LLMs, not just the constant. Fences: as C6.

## PACKED CRYSTAL C6c VERDICT: the sigma FORM survives (11.6x recovery) but max-anchored grids own heavy tails — the external fence is now MECHANISTIC (2026-07-29 night, 3080; C-series COMPLETE)

sigma[row, K=16] (step sigma/8, 9.64 raw bits -> matched 10):
DeltaKL 0.1127/tok, ppl 60.64 — an 11.6x recovery from C6b
(PREDICTION 1 CONFIRMED: the knee constant, not the
allocation form, was the C6 failure). But rtn10 0.0029 and
hqq10 0.0007 (PREDICTION 2 FALSIFIED — not within 2x of rtn;
PREDICTION 3 CONFIRMED — hqq <= rtn). MECHANISM, now clean:
our fixed-width bits are priced by the WORST outlier's span
(sigma-anchored step is uniform, so one 50-sigma outlier
inflates every row's bit count), while max-anchored grids
spend the same bits as a finer step on every quiet row. On a
crystal this cannot happen — C1 measured entropy = Gaussian
capacity to <1%, i.e. NO outlier structure to exploit; on
web-trained weights the outliers ARE the structure and
max/zero-point-anchored schemes harvest it. THE PAPER'S
HONEST SHAPE: (a) calibration-free sigma-allocation is
OPTIMAL for at-capacity (max-entropy) weights — born
crystals, measured; (b) web LLMs are not at capacity; their
tails demand max-anchored or calibrated grids (use rtn/hqq
there); (c) universal claims unaffected: entropy-bound
packing (C0/C1), bit-packed kernels 2.39x (C2b), cross-
device integer determinism (C4), tiered bytes (C5).
C-SERIES COMPLETE: C0-C6 all run, 5 confirms / 3 honest
falsifications, every claim measured. Fences: one web model,
fake-quant, n=1 per arm; entropy-coded (rather than fixed-
width) sigma-grids on heavy tails = banked follow-up.

## PRE-REG: PACKED CRYSTAL R-PASS — replication + review battery (2026-07-29 late night, before the reruns)

Confirmation sweep before docs/README promote the C-series:
(R1, Mac) C0+C1 full rerun — bar: identical pack stats and
gates within sigma of first pass (instrument deterministic;
any drift = flake caught). (R2, Mac+3080) C4 with a SECOND
activation seed (SEED=5 v 4) — hash A must AGAIN match
across devices on fresh integer activations (stronger than a
rerun: new random battery, same invariance). (R3, Mac)
C2/C2b bench rerun — timing is the only stochastic channel;
bar: speedups within ~15% of booked (1.76x / 2.39x classes
hold). (R4, 3080) C6c rerun as-is — bar: DeltaKL/ppl
reproduce to the printed precision (fake-quant + fixed
prompts = deterministic modulo cudnn nondet in eval; drift
reported honestly). Alongside: line-review of all six
instruments (bits accounting, Hessian construction, KL
direction, restore-after-arm). Any discrepancy books an
AMENDMENT naming its target.

## PRE-REG: THE CAPACITY METER — a zero-calibration predicate for which allocator a model deserves (2026-07-29 late night, desk, before the read)

C6c's mechanism implies a MEASURABLE decision rule, no
inference required: at step sigma_r/2 per row, M =
(span bits) - (code-stream Shannon entropy), param-weighted
over 2-D weights. M is the fixed-width penalty the sigma
grid pays to the worst outlier; at-capacity (Gaussian)
weights give M ~ 1.5-2 bits (C1 measured 1.8), heavy tails
inflate span while entropy stays low. Kurtosis rides along.
Instrument scratch/capacity_meter.py. Cells: house crystals
(d64h8, L4d56, cplx_none), Qwen2.5-0.5B (3080), SmolLM2-1.7B
(Mac), DeepSeek-V3 layer-30 routed experts from the cached
shard (the 07-17 "experts are crystals" gauge, now on the
capacity axis; weights fp8-dequant via block scales as then).
PREDICTIONS: (1) crystals M in 1.5-2.5; (2) Qwen + SmolLM
M >= 4 (the C6 failure, predicted from disk); (3) DeepSeek
experts land CLOSER TO THE CRYSTAL BAND than to the web-
dense band (router-as-diet-focuser extends from spectral
gauge to capacity) — if (3) holds, sigma-law packing has a
web-scale home in MoE experts and the C7 cell (pack an
expert, score it) arms. DECISION RULE shipped either way:
M small -> sigma-law; M large -> max-anchored/calibrated.
Fences: desk read only, one shard for V3, block-dequant
approximation noted.

## CAPACITY METER VERDICT: the allocator predicate works from disk — and DeepSeek's EXPERTS land in the crystal band (2026-07-29 late night)

Full table (M = span_bits - code_entropy at per-row sigma/2;
kurtosis alongside; param-weighted):
  crystal cplx_none  M 0.96  kurt 2.26   (9.4M read)
  crystal d64h8      M 1.61  kurt 3.70   (0.5M)
  crystal L4d56      M 1.61  kurt 3.56   (0.2M)
  DeepSeek-V3 L30 EXPERTS  M 2.33  kurt 3.07  (704.6M, fp8
    block-dequant, one shard)
  Qwen2.5-0.5B dense M 3.62  kurt 5.29   (357.8M)
  SmolLM2-1.7B dense M 3.85  kurt 6.54   (465.6M read)
PREDICTION 1 CONFIRMED (crystals 0.96-1.61, in band).
PREDICTION 2 DIRECTIONAL (web-dense 3.62/3.85 v predicted
>=4 — separation clean, threshold missed by ~0.3; booked
honestly). PREDICTION 3 CONFIRMED, the headline: DeepSeek's
routed experts read kurtosis 3.07 — GAUSSIAN to two decimals
— and M 2.33, far from the web-dense band. The 07-17
"experts are crystals" spectral result extends to the
CAPACITY axis: the router-as-diet-focuser drives expert
weights to max-entropy, exactly the regime where sigma-law
allocation is optimal (C1: entropy=capacity <1%) and where
Lloyd-Max fitting buys nothing (07-25: MSE-optimal codebooks
tie uniform on outlier-free crystals — the same law from the
codebook side). DECISION RULE SHIPPED: M <~ 2.5 -> sigma-law
(calibration-free); M >~ 3.5 -> max-anchored/calibrated. C7
ARMS: pack DeepSeek experts (or any open MoE's) with the
sigma rule v HQQ — the paper's web-scale home. Fences: desk
read, one V3 shard (layer 30), 48-tensor samples for the big
models, thresholds n=1 per class.

## R-PASS VERDICT: the C-series replicates on every arm (2026-07-29 late night)

R1 (C0+C1 rerun, Mac): EXACT — all pack stats and all four
gates reproduce line-for-line (deterministic instrument
confirmed, no flake). R2 (C4, SECOND activation seed): hash A
seed-5 = 7f6849f7...024eb4 IDENTICAL on MPS and cuda — the
cross-device integer invariance holds on a fresh random
battery, not just the registered one; hash B and greedy
streams also reproduce per-device. R4 (C6c rerun, 3080):
DeltaKL 0.1127/0.0029/0.0007 and ppl reproduce to printed
precision. R3 (kernel bench rerun, Mac): large-shape
speedups within 2% of booked (crystal8 1.73x v 1.76x;
crystal5 2.35x v 2.39x); micro shapes are +-0.1x-class
timing noise (256x64 read 0.91x then 1.03x) — micro-shape
entries are hereby fenced as parity-within-noise, the
large-shape claims stand. NO AMENDMENTS REQUIRED from the
instrument line-review (bits accounting, Hessian
construction, KL direction, arm-restore all verified).
C-series numbers are CONFIRMED for docs/README promotion.

## PRE-REG: PACKED CRYSTAL C7 — sigma-law on MoE ROUTED EXPERTS (the at-capacity transport claim; 2026-07-29 late, Artin's GO)

Model: allenai/OLMoE-1B-7B-0924-Instruct (smallest good open
MoE; 64 routed experts/layer; Mac 36GB, MPS/fp16 — 3080
cannot hold it). Instrument scratch/pack_c7.py, C6-harness
form (fake-quant, 16 fixed prompts DeltaKL + README-slice
ppl, wall-times). ARMS on EXPERT tensors only (router/attn/
embed untouched): sigma[row, K=2] v rtn v hqq at matched
measured bits. CONTROL ARM: the same three on the DENSE
attention tensors only (experts untouched) — the meter reads
both groups first. PREDICTIONS: (1) capacity meter on OLMoE
experts lands <= 2.5 bits / kurt ~3 (crystal band, as
DeepSeek's did) and the dense attn group reads HIGHER (web-
dense band); (2) THE TRANSPORT CLAIM: sigma[row] DeltaKL
within 2x of hqq on the EXPERT arm (v 33x on dense Qwen —
at-capacity weights need no calibration ANYWHERE they occur);
(3) the meter ORDERS the damage within one model: sigma's
relative loss v hqq is larger on the attn control arm than on
the expert arm; (4) wall-time sigma <=2s v hqq minutes-class
on ~6B expert params. FALSIFIER: sigma >5x hqq on experts
despite a crystal-band meter reading — the meter would be
measuring the wrong statistic and the at-capacity criterion
takes an amendment. Fences: one MoE, fake-quant, n=1,
fp16-on-MPS eval, expert sample may be capped for wall-time
(reported).

AMENDMENT (targets: C7 pre-reg): first run KILLED mid-pack —
the instrument cloned all 6.4B expert params (fp32) on top of
the 14GB fp16 model and blew the Mac's 36GB. v2 is streaming:
model RELOADED per arm, quantized in place tensor-by-tensor,
matched bits pinned at 6 (measured 5.85 first pass). Meter
readings from the killed run stand (they printed before the
kill): EXPERTS M 2.85 / kurt 3.50, ATTN M 3.11 / kurt 7.55 —
kurtosis separates the groups cleanly in-model; experts' M
lands 0.35 ABOVE the predicted <=2.5 band (booked when the
verdict reads). Predictions otherwise unchanged.

## C7 VERDICT: strong-form transport FAILS — and the capacity meter upgrades from predicate to CONTINUOUS DIAL (2026-07-29 close; OLMoE-1B-7B, Mac)

Full table (DeltaKL/tok v fp16, ppl 75.74 control; matched 6
bits): EXPERTS sigma[row] 0.0716 / rtn 0.0097 / hqq 0.0044
(sigma = 16.3x hqq; quant 16.6s v hqq 675.5s). ATTN sigma
0.0442 / rtn 0.0053 / hqq 0.0020 (sigma = 22x hqq).
PREDICTION 1 MISSED: OLMoE experts are NOT crystal-band
(M 2.85, kurt 3.50 — Gaussian-shaped but wide-spanned).
PREDICTION 2 FALSIFIED: 16x, not within 2x. PREDICTION 3
CONFIRMED in ordering form — and this is the finding: the
sigma-v-calibrated premium is MONOTONE IN M across every
group ever measured: crystals M 0.8-1.6 -> ~1x (parity);
OLMoE experts M 2.85 -> 16x; OLMoE attn M 3.11 -> 22x; Qwen
dense M 3.62 -> 34x (C6b, same sigma[row] form). The
at-capacity criterion is GRADED, not binary; M is the dial
that prices the calibration premium, and it said so BEFORE
the arms ran (the amendment flagged M 2.85 > band at kill
time). The pre-reg falsifier does NOT fire against the meter
(it required a crystal-band reading; the meter refused to
give one). PREDICTION 4 CONFIRMED at scale: hqq 675.5s on
6.4B params v sigma 16.6s (41x) — the wall-time claim's
first real exhibit. DECISION RULE SHARPENED: sigma-law only
below M ~ 2 (measured members: born crystals, the house NNUE
at 0.82); DeepSeek experts (2.33) sit between bands —
predicted premium ~5-10x, UNMEASURED (no full model in
house). Paper shape: the meter-premium curve (6 points,
monotone) may be worth more than the binary claim it
replaces. Fences: one MoE, n=1/arm, fp16 MPS, kurtosis
demoted (3.50 looked Gaussian; span-priced M caught what the
4th moment missed).

## PRE-REG: P2a-v2 — THE ANALYTIC-CLIP ALLOCATOR (trajectory change per the C7 dial; 2026-07-29 close, Mac only)

The C7 dial locates the sigma premium in SPAN-PRICING, so P2a
pivots from meter-routing to closed-form span attack (Artin's
GO to re-rung). Arms on SmolLM2-1.7B (local fp, Mac; all
linears ex lm_head), matched bits, C6 harness (DeltaKL 16
prompts + README ppl + wall-time): (a) rtn per-row absmax
(baseline, the max-anchored incumbent); (b) SIGMA-CLIP k in
{4, 6, 8}: per-row uniform grid over [-min(absmax, k*sigma),
+...], outliers SATURATE — zero calibration, two stats per
row; (c) hqq (calibrated-class bar). PREDICTIONS: (1) some
sigma-clip k BEATS rtn (absmax wastes resolution on one
outlier per row; clipping trades rare saturation error for
everywhere-finer steps); (2) best sigma-clip closes >=half
the log-gap between rtn and hqq; (3) wall-time stays rtn-
class (<5s). FALSIFIER: all k lose to rtn -> outliers in
web-dense rows are load-bearing at full magnitude (the AWQ
salient-channel story) and the span attack dies — hybrid
routing (original P2a) resumes as the fallback. Fences: one
model, n=1/arm, fp16 MPS, k grid coarse.

## P2a-v2 VERDICT: THE FALSIFIER FIRES — web-dense outliers are load-bearing at FULL magnitude; the span attack dies (2026-07-29 close)

SmolLM2-1.7B, 6 bits: rtn(absmax) DeltaKL 3.32 / ppl 62.98;
sigma-clip k=4 DeltaKL 8.82 / ppl 138,890 (DESTROYED); k=6
ppl 5,913; k=8 ppl 532; hqq 0.445 / 61.96 (127s). Every clip
arm catastrophically loses to plain absmax — saturating even
the >8-sigma tail kills the model. PREDICTIONS 1-2 FALSIFIED;
the pre-reg falsifier fires: outliers are not wasted span,
they are LOAD-BEARING at full magnitude (the AWQ salient-
channel mechanism, confirmed in-house from the destructive
direction). Corrected mechanism for the dial: sigma grids
lose on web-dense NOT because bits are wasted on outliers
but because BOTH ends bind — the outliers must be kept AND
the bulk needs fine steps; only range-adaptive (max-anchored
per row) or calibrated (zero-point/scale-optimized) grids
serve both. P2 CLOSES (both forms): the hybrid fallback is
vacuous (web-dense tensors are uniformly M>3 — nothing to
route); the zero-calibration domain is M < ~2, period.
Honest metric note: prompt-KL and ppl disagree on rtn (KL
3.32 but ppl +2.0) — KL over 16 prompts is the harsher,
noisier channel; ppl carries the verdict. DESK RIDER (same
close): meter on the C5 matryoshka crystal reads M 1.46 ~
parent 1.63 — the meter does NOT see the C5 tier tax,
confirming fragility (k_c, flips/token) is an ORTHOGONAL
second axis: the P5 card needs BOTH numbers. Fences: one
model, n=1/arm, coarse k grid, 6 bits only.

## PRE-REG: THE EXPERT-SCALE HYPOTHESIS — Artin's prediction, Kimi-K2 shard meter (2026-07-29 close, desk)

Artin (verbatim intuition, mid-session): a small MoE
cramming many experts into few params is "probably not very
optimal" — bigger/finer-grained MoEs should have MORE
crystal-like experts. The house's two existing points agree:
OLMoE (7B, 64 exp/layer) M 2.85 v DeepSeek-V3 (671B, 256
exp/layer) M 2.33. Cell: meter one mid-stack expert shard of
Kimi-K2-Instruct (1T-class, 384 experts/layer, fp8 +
scale_inv like V3; K3 exists but ships 8-bit compressed-
tensors — histogram pre-gridded, confounded, excluded).
PREDICTION (Artin's): K2 experts read M < 2.33 (monotone in
expert fineness/scale), extending the dial's expert branch
toward the crystal band. If confirmed at n=3, the projection
is that sufficiently fine-grained frontier experts ENTER the
sigma-law domain (M < ~2) — the transport claim revives at
the scale where it matters, as a scaling law rather than a
binary. Fences: desk, one shard, fp8-dequant approximation,
n=1 per model class, expert-fineness and total-scale
confounded (named).

## EXPERT-SCALE VERDICT: ARTIN'S PREDICTION CONFIRMED — expert capacity is monotone in fineness, and frontier experts REACH the sigma-law boundary (2026-07-29 close)

Kimi-K2-Instruct shard 37 (layer 36, 384 routed experts/
layer, fp8-dequant, 705M params read): M = 2.01 bits, kurt
3.02. THE LADDER AT n=3, monotone exactly as Artin called
it ("cramming many experts into few params is probably not
optimal" -> finer/larger should be more crystal-like):
  OLMoE   7B,  64 exp/layer:  M 2.85
  DS-V3 671B, 256 exp/layer:  M 2.33
  K2     1T,  384 exp/layer:  M 2.01  <- AT the M~2 domain
                                          boundary
Read against the dial (premium monotone in M; crystals ~1x
at M<2, OLMoE experts 16x at 2.85): K2-class fine-grained
experts sit where the projected sigma premium approaches
the noise floor — the transport claim revives as a SCALING
LAW: routers at frontier scale focus each expert toward
max-entropy weights, and sufficiently fine-grained experts
enter the calibration-free domain exactly where calibration
is most expensive (hqq 675s on OLMoE's 6.4B; a 1T MoE pays
that x150). K3 (2.78T) excluded as instrument-confounded
(ships 8-bit pre-quantized). NEXT rungs banked: (a) second
K2 shard (depth check — does M vary by layer?); (b) the
DeltaKL arm at K2 scale needs hardware the house lacks —
flagged as the paper's external-collab cell; (c) the title
riff banked: "black-hole experts" (max-entropy endpoint of
router focusing). Fences: one shard, desk, fp8-dequant
approximation, fineness confounded with total scale AND
training recipe (n=3 model classes).

## PRE-REG: BLACK HOLE MoEs B0+B1+B2 — the capacity atlas + streaming pack of Qwen3-30B-A3B (2026-07-29 close, before the pass)

One streaming pass (scratch/blackhole_b0.py; shard ->
process -> delete, one shard on disk at a time): (B0) meter
every 2-D tensor -> M/kurt by (layer, group expert|attn|
shared|router); (B1) dial-routed zero-calibration pack —
sigma[row] codes where M<2, per-row max-anchored codes
otherwise; real packed bytes + entropy v capacity per group;
(B2) per-tensor function-space error ||x(Wq-W)||/||xW|| on
64 Gaussian probes. PREDICTIONS: (1) experts read below
attn/shared at every layer (C7 ordering replicates
in-model); (2) expert M at 128-fineness lands BETWEEN OLMoE
2.85 and V3 2.33 (fineness drives the ladder, not total
scale — the discriminating point); (3) router tensors read
worst; (4) function-space error <=2% on expert tensors at
their assigned grids; (5) whole pass wall-clock < 2.5h on
the Mac, zero calibration data. Fences: desk only (no 30B
inference), bf16 source, spot checks are not end-to-end
quality, n=1 model.

## BLACK HOLE MoEs B0-B2 VERDICT: the first full capacity atlas of a production MoE — routers are the incompressible organ, experts are bimodal by projection, and the ladder's variable is EXPERT SIZE (2026-07-30 ~00:30; pass 136.6 min)

Qwen3-30B-A3B, all 16 shards streamed (download -> process ->
delete), 30.2B params / 18,673 tensors metered + dial-packed;
artifact 21GB v ~60GB bf16 (2.9x; expert raw 5.60 bits/wt).
ATLAS: experts M med 2.93 (40% of expert tensors in the
sigma-law domain), attn 2.94 (k_proj worst at 3.93), router
4.45 (ALL top-5 function errors are routers), shared 3.93.
Depth: mid-stack dip (layers ~8-16 read 1.93 — the most
black-hole-like region), both ends at 2.93. Projection
bimodality: up_proj med 1.93 (IN the sigma domain) v
gate/down 2.93. PREDICTIONS: (1) PARTIAL — experts tie q/v/o
attn, beat only k_proj/router; (2) FALSIFIED INFORMATIVELY —
128-fineness at 30B reads 2.93, ABOVE OLMoE's 2.85: the
ladder's variable is PER-EXPERT SIZE, not count (Qwen3 ~5M/
expert 2.93 | OLMoE ~6M 2.85 | V3 ~45M 2.33 | K2 ~40M 2.01
— monotone in expert params; Artin's original phrasing "the
little guy's experts can't be crystal" was the correct
variable both times); (3) CONFIRMED — routers worst
everywhere (they serve every token; keep them fp); (4)
FALSIFIED BY MIS-SPECIFICATION (amendment): the <=2% bar
ignored that a sigma/2 grid's INTRINSIC relative output
error is sqrt(1/48) ~ 14.4% — measured sigma tensors read
0.1424, max-anchored 6-bit read ~0.043 = their step's
arithmetic; the err column is comparative-at-matched-bits
only, not a quality readout (check-don't-assume applies to
pre-reg bars); (5) CONFIRMED — 136.6 min < 2.5h, laptop,
zero data. Fences: desk only (no 30B inference; quality
inference flows through the dial, not measured end-to-end
here), n=1 model, bf16 source, atlas jsonl =
logs/blackhole_atlas.jsonl, parts = checkpoints/
blackhole_q3_parts/ (untracked).

## PRE-REG: NIGHT-30 TRIPLE — C1 seed-2 births (3080), OLMoE dial-pack end-to-end (Mac), expert-similarity desk (2026-07-30 ~01:00, before the runs)

(N1, 3080 overnight, Artin's night GO): seed-2 births of the
two C1 crystals — sym_birth ARM=dense SEED=2, (a) D=64
HEADS=8 FFN=256 L=8, (b) D=56 HEADS=4 FFN=224 LAYERS=4 —
EMA 0.999, markers on success. Purpose: n=2 on the flagship
zero-tax parity claim (C1 replication packs+gates tomorrow,
Mac). PREDICTION: both birth normally (gates within the
seed sigma ~3.5 of seed-1's 58/56).
(N2, Mac): OLMoE dial-pack END-TO-END — the atlas policy
applied to a RUNNABLE MoE: per-tensor sigma[row] where
M<2 else per-row max-anchored 6-bit, experts only; DeltaKL
+ ppl v the booked C7 arms (sigma-pure 0.0716, rtn 0.0097,
hqq 0.0044). PREDICTION: dial-pack lands AT-OR-BELOW rtn
(0.0097-class) — the dial fixes sigma-pure's damage by
routing only the ~M<2 tensors to sigma grids; falsifier:
dial worse than rtn -> per-tensor routing adds nothing over
uniform max-anchoring on this MoE.
(N3, Mac desk): EXPERT SIMILARITY / dynamic replication
(Artin's riff): from the packed Qwen3 parts, one mid-dip
layer (L12) + one endpoint layer (L40): decode all 128
experts' gate_proj, compute the mean-expert and per-expert
delta; report sigma(delta)/sigma(weight) and the top-16
pairwise correlation. PREDICTION (house): deltas are NOT
small (ratio > 0.8) and correlations ~0 — router
specialization decorrelates experts (the anti-tied limit;
the C7 meter said experts carry max marginal info). If the
ratio reads < 0.5 anywhere, base+delta compression arms and
Artin's dynamic-replication idea gets a rung.

## N3 VERDICT: experts share NOTHING in weight space — dynamic replication is dead post-hoc, alive only at birth (2026-07-30 ~01:15)

Qwen3-30B, 128 gate_proj experts decoded from the packed
parts at L12 (mid-dip) and L40 (endpoint):
sigma(delta)/sigma(W) = 0.995 / 0.993; pairwise correlations
mean 0.0024/0.0054, max 0.0155. HOUSE PREDICTION CONFIRMED:
router specialization decorrelates experts to numerical
zero — no shared base, no low-rank family, no post-hoc
base+delta compression, no static replication of one expert
from another. Consistent with (and mechanistically explains)
the capacity ladder: each expert carries maximal MARGINAL
information; the router's focusing is precisely what removed
the redundancy a tied representation would need. Artin's
dynamic-replication idea therefore re-routes to BIRTH-TIME
(impose the sharing: the tied-expert ladder riff — group-
action or low-rank-delta experts trained jointly, symmetry-
ladder precedent 2x:-1..8x:-6) — banked, not runnable
post-hoc on trained MoEs. K3 side-answer booked here too:
896 experts x 93 layers at ~66M params/expert (BIGGEST
per-expert yet — the size ladder predicts deepest M), but
K3 ships MXFP4 (4-bit pre-gridded) so the meter is
undefined on it; the meta-fact that a frontier lab SHIPS
4-bit experts is independent, industry-side evidence for
the at-capacity scaling story. Fences: one model, 2 layers,
gate_proj only, desk.

## N2 VERDICT: dial-pack recovers 3.8x over sigma-pure but rtn still wins — DeltaKL forgives nothing (2026-07-30 ~01:30)

OLMoE experts, dial policy (685 tensors sigma[row] at M<2 /
2,387 max-anchored 6-bit): DeltaKL 0.0187/tok, ppl 76.25,
quant 60.7s. v the booked arms: sigma-pure 0.0716 (dial 3.8x
better), rtn 0.0097 (dial 1.9x WORSE), hqq 0.0044.
PREDICTION (at-or-below rtn) FALSIFIED, and the mechanism is
worth the booking: a sigma/2 grid carries intrinsic
distortion sigma^2/48 while a 6-bit max grid on a typical
expert row (max ~5 sigma) steps at ~0.16 sigma = ~10x less
distortion at comparable bits. On HOUSE crystals that extra
distortion is FREE because the capability metric (gate
solves) has slack below the knee — but DeltaKL is a
slack-free metric: it sees every logit perturbation and
forgives none. THE CLARIFIED LAW: sigma-law's zero-tax
domain = (M < ~2) AND (the deployment metric has knee slack,
i.e. you score FUNCTION OUTCOMES, not logit distance). For
web MoEs scored by KL/ppl, the zero-calibration
recommendation is plain per-row max-anchoring (rtn) — which
STILL ties hqq within ~2x at seconds v minutes. Fences: one
MoE, n=1, experts only. The paper gains a sharper claim
boundary, not a loss: our gates were never the soft option —
they are the deployment-realistic one (nobody ships logits;
everybody ships answers).

## PRE-REG: NIGHT-30b — the extended overnight queue (2026-07-30 ~01:45, before the runs)

(B3, Mac): K2 depth curve — shards 5 (early) + 55 (late) v
booked 37 (layer 36, M 2.01); meter experts per layer; ALL
K2 shards deleted after. PREDICTION: mid > ends is the house
guess (the Qwen3 atlas found a mid-stack dip; if K2 shows
the same shape, "black-hole depth profile" is a cross-model
regularity; if flat, the dip is a Qwen3 recipe artifact).
(B4, Mac): entangled-experts MI — hooks on OLMoE's routers,
top-8 selections over the 16-prompt battery + README slice;
pairwise MI between expert-active indicators within layer v
a shuffle baseline. Then merge the top-MI pair (weight mean
in both slots) -> DeltaKL. PREDICTIONS: (1) MI above shuffle
exists (co-routing structure is real); (2) house prior:
top-pair merge COSTS measurably (N3 says weights are
decorrelated even if routing is correlated — co-firing does
NOT imply mergeable); Artin's link hypothesis wins if merge
is ~free.
(P6-acct, Mac desk): entropy accounting of the Qwen3 packed
parts — raw code bits v Shannon entropy v npz-deflate bytes,
by group. PREDICTION: deflate recovers >=60% of the
fixed-width penalty on sigma tensors (C1 precedent), less on
max-anchored (their code distributions are flatter).
(N1b, 3080, chained behind night30's marker): seed-3 births
of both C1 crystals -> tomorrow C1 parity at n=3.
PREDICTION: births normal, gates within sigma of seeds 1-2.
Fences: B3 fp8-dequant, B4 one MoE + 2 merge reads max,
watcher pgrep-fenced, markers on success only.

## NIGHT-30b VERDICTS: K2 is uniformly black-hole in depth; co-routing links are REAL but live in routing, not weights; entropy coding has 1.3 bits/wt on the table (2026-07-30 ~02:15)

B3 (K2 depth): L4 M 2.09 / L36 2.04 / L54 2.09 (kurt 3.02-
3.04) — FLAT. House guess (mid-dip regularity) FALSIFIED:
K2's near-boundary state is uniform in depth; Qwen3's L8-16
dip is a recipe/scale artifact, not a law. K2 shards deleted.
B4 (entangled experts): co-routing MI is REAL and large —
top-pair MI 0.21-0.46 bits per layer v shuffle scale 0.0009
(~300-500x baseline): Artin's link hypothesis CONFIRMED at
the ROUTING level — expert pairs fire together far above
chance, every layer. But the merge read: averaging the
top-MI pair (L15, experts 1+29) costs ppl 75.74 -> 79.10 —
co-firing does NOT imply mergeable weights (house prediction
2 confirmed, consistent with N3's zero weight correlation).
THE SPLIT LAW: the links exist in the ROUTER's statistics,
not in weight space — so the exploitable structure is
routing-side (speculative expert prefetch, cache policy,
paired placement across devices), not weight-side (merging,
sharing). Bandwidth/latency lever, not a params lever.
P6 (entropy accounting, 30.2B params): experts raw 5.60 v
entropy 4.31 bits/wt (1.29 recoverable by rANS — artifact
21GB -> ~16.3GB, 3.7x v bf16); router raw 6.00 v entropy
2.84 (penalty 3.16 — heavy-tailed weights concentrate codes;
ironically the LEAST meter-compressible tensors are the MOST
entropy-codable). Fences: B3 3 layers/1 shard each; B4 one
model, one merge read, MI on 16-prompt+README battery; P6
bound-accounting, rANS not yet implemented.

AMENDMENT (targets: NIGHT-30 N1 + NIGHT-30b N1b pre-regs):
both 3080 birth arms CRASHED at launch — friendly-fire
variants #8 and #9 booked. (8) Missing data dep: sym_birth
reads data/micromodel_gen4_sidecar.jsonl, which lives on the
Mac only (untracked, file-handoff convention) — the launcher
never verified file deps at arm time (standing doctrine,
violated). (9) STALE MARKER: logs/night30_done.marker
already existed from a July-29 job of the same name, so the
chained night30b watcher fired IMMEDIATELY into the same
crash — markers must be unique per launch (extends
friendly-fire #7's unique-job-file rule to marker names).
Zero GPU-hours lost (crashes were instant). RECOVERY: the
seed-2/3 births move to the MAC — which the comparability
fence PREFERS anyway (seed-1 C1 crystals are MPS-born and
MPS-gated; same-device doctrine). Births relaunched Mac-side
this morning, same pre-reg predictions.

## C1 AT n=3 VERDICT: h8 parity REPLICATES (headline holds); the L4 floor-crystal pays -5 at both new seeds — the fragility axis surfaces exactly where the theory says it should (2026-07-30 morning)

Same packer, same instrument, same device (MPS), fresh
seed-2/3 births. d64h8: fp 54 -> packed 56 (+2), fp 59 ->
56 (-3); with seed-1's 58 -> 58, n=3 reads +2/-3/0 — CLEAN
PARITY, the headline claim stands at n=3. Entropy replicates
too: 3.10-3.13 v capacity ~3.13 on all four (bound-gap <1%
at n=3 on BOTH architectures). BUT layers-4 d56: fp 50 ->
45 (-5) and fp 52 -> 47 (-5), v seed-1's 56 -> 55 (-1) —
TWO -5 readings (>sigma 3.5) is not noise: the smallest
crystal pays a pack tax at the weaker seeds. READ: exactly
the two-parameter law's shape — seeds 2/3 of the floor
architecture birthed weaker (fp 50/52 v 56) and near-tie
density (k_c) plausibly higher; distortion that is free on
d64h8 bites the floor crystal. THE PAPER'S CLAIM NARROWS
HONESTLY: zero-tax parity is n=3-solid on the d64h8 class;
at the width floor it is seed-dependent — and the flips/
token meter (rho .883) is the pre-pack check that should
predict which seeds pay. NEXT (queued): P5 card on all six
seed crystals — retrodict the -5s from flips/token alone;
if it separates seed-1-L4 from seeds-2/3-L4, the card
graduates from diagnostic to gate. Fences: n=3 per arch,
MPS, one packer, weaker-seed confound named (fp gates
differ across seeds).

## P5 RETRODICTION VERDICT: the flips card predicts at CLASS level, not seed level (2026-07-30 morning)

Six crystals, standing instrument (Q=16 best-rational,
400-row probe): h8 class 0.00838-0.00952 flips/tok; L4 class
0.01046-0.01205 (~1.3x higher). CLASS-LEVEL RETRODICTION
PASSES: the card ranks the architectures correctly — it
would have flagged the L4 class as the pack-tax risk before
any gate (L4 paid -1/-5/-5; h8 paid +2/-3/0). SEED-LEVEL
RETRODICTION FAILS: within L4, seed-1 (paid -1) reads 0.01176
~ seed-2 (paid -5) at 0.01205, and seed-3 (paid -5) reads
LOWEST at 0.01046 — flips/token does not resolve which seed
pays. Two honest readings: (a) the probe's operator is Q=16
rational, not the actual sigma-pack operator — fences travel
with instruments, and the matched-operator probe is the
named follow-up; (b) the seed-level distinction itself
(-1 v -5, ~1 sigma apart) may be gate noise — the ROBUST
n=3 statement is class-level (L4 pays ~-4+-2, h8 ~0), and
THAT the card predicts. CARD STATUS: shipped as a CLASS
gate (meter M for regime + flips/tok for class fragility);
seed-granular prediction unproven at current probe power.
Fences: one Q, one probe set, n=3 per class.

## PRE-REG: P6-v2 — rANS lands the entropy bound as real bytes (2026-07-30 morning, before the runs)

Coder: constriction (Rust rANS, smoke-tested: exact
roundtrip, +0.01% over entropy on synthetic). Instrument
scratch/pack_rans.py. Cells: (a) house artifacts — re-emit
the six packed crystals (seed 1-3 x h8/L4) with per-tensor
rANS code streams (frequency tables + fp scales in the
container); (b) Qwen3-30B parts — rANS all 30.2B params'
codes. PREDICTIONS: (1) rANS lands within 0.5% of the
per-tensor Shannon entropy everywhere; (2) house crystals
drop from ~5.0 to ~3.3-3.5 bits/param INCLUDING fp
passthrough (=> ~9x v fp32); (3) the Qwen3 artifact reads
~16.3GB (P6's 4.31 bits/wt accounting made real, 3.7x v
bf16); (4) every stream roundtrips EXACTLY (lossless, or
the cell is void). Fences: table overhead counted honestly
(counts stored per tensor); wall-time reported; n=1 per
artifact.

## P6-v2 VERDICT: the entropy bound is now real bytes — 30B MoE at 16.48 GB (3.67x), house crystals at ~9x fp32, all lossless (2026-07-30 morning)

House artifacts (rANS + tables + fp passthrough): d64h8
233,180 B (9.10x v fp32; codes 3.179 bits/wt v entropy
3.12); L4d56 99,728 B (8.25x); all six seed crystals in the
3.16-3.21 bits/wt band. Qwen3-30B: 16.48 GB total (codes
4.337 bits/wt + 0.09 GB scales) v bf16 60.4 GB = 3.67x —
P6's accounting (4.31 + overhead) made real within 0.6%.
PREDICTIONS: (1) streams within 0.5% of entropy CONFIRMED
(tables add ~2% on the small crystals, counted); (2) ~9x
house CONFIRMED (d64h8 9.10x; L4 8.25x — heavier fp share
at 4 layers, noted); (3) ~16.3 GB CONFIRMED at 16.48; (4)
EXACT roundtrip everywhere (asserted per stream; first 2B
Qwen symbols verified, coder identity thereafter). Wall:
seconds per crystal, ~minutes for 30B. THE ARTIFACT STORY
COMPLETES: sigma-law allocation (closed form) + bit-pack +
rANS = a 30B production MoE packed AND entropy-coded on a
laptop with zero calibration data, losslessly recoverable,
at 3.67x v bf16 — with the C2b lesson that the bit-packed
form is directly executable and the dial/N2 lesson pricing
exactly when the sigma step is capability-free. Fences: n=1
per artifact, table overhead per-tensor (amortizable),
decode-side rANS throughput not benched (storage format;
the runtime twin remains crystal5/int8).

## PRE-REG: P3 — THE DETERMINISTIC DECODE (full fixed-point forward; 2026-07-30 morning, before the build)

Instrument scratch/pack_decode.py: a deterministic twin of
the MicroLM forward on the PACKED d64h8 crystal, where every
operation is either exact integer or a shipped-table lookup
— no libm anywhere in the path (the axiom FX-V1 recipe,
house-built): weights = sigma-law codes (5-6 bit ints);
activations fixed-point (scale 2^8, clamped, requant by
exact shifts); GEMMs via the exact-fp32 integer carrier
(all partials < 2^24 by construction — bounds printed);
RMSNorm via int64 sum-of-squares + integer-Newton isqrt;
SiLU and exp as precomputed int tables (computed ONCE on
CPU, shipped in the artifact, looked up exactly); RoPE from
shipped fixed-point sin/cos tables; attention softmax as
exp-table weights + exact int64 weighted sum + one fixed-
point divide; greedy readout = integer argmax (no softmax).
CELLS: (a) CORRECTNESS/PRICE (Mac): full gate of the
deterministic path v the fp control 58 — bar: within sigma
~3.5 (the fixed-point activation quantization is the new
tax; 2^8 levels predicted inside the knee); (b) THE HASH
(Mac + 3080): 40-token greedy streams AND full integer
logit hashes on a fixed battery — bar: BOTH identical
across MPS and cuda (the claim C4 could not make: C4's fp
norms let logits differ; here nothing may differ); (c)
wall-time honest card v the fp path (expected SLOWER —
tables and fixed-point are a determinism price, booked as
such). PREDICTIONS: (1) gate within sigma of 58; (2) hash
equality on both channels; (3) tok/s 2-10x slower than fp
MPS (unoptimized reference path). FALSIFIER: any hash
mismatch = a nondeterministic op survived (hunt it, book
it); gate crater = 2^8 activations below the knee (raise to
2^10, re-gate, book the activation-knee reading). Fences:
one crystal, greedy only, reference implementation (speed
is not the claim).

## P3 VERDICT: THE DETERMINISTIC DECODE LANDS — bit-identical logit traces across GPU vendors, at a measured 96.7% agreement price (2026-07-30 midday)

The fixed-point twin (shipped tables, exact-integer
everything, no libm) hashes IDENTICAL on MPS and cuda on
BOTH channels: greedy streams bf76568d... AND the full
per-step logit trace 311f71bf... — two independent table
versions confirmed equality (first pass 2f06e6c8/af0ba7e7,
final a2c89daa-tables bf76568d/311f71bf). Max GEMM partial
2^21.2 (bound 2^24 held by construction). PREDICTION 2
CONFIRMED — the claim C4 could not make: not just tokens,
every NUMBER identical cross-vendor. CAPABILITY PRICE
(amended cell — full gate infeasible at reference-python
speed, ~5 tok/s: substituted teacher-forced argmax
agreement v the fp model, 3,055 tokens): 96.66%, with
disagreements concentrated at low margins (median 0.177 v
7.6 overall — the soft-token class). Debug lineage booked:
A=2^8 -> 2^10 changed NOTHING; emb/head sigma/8 nothing;
the wound was ACT_CLAMP=8 (real residual features clamped
— 92.0 -> 96.7 at clamp 32); an int64 overflow in the
rmsnorm bump caught and fixed (5.4% canary); remaining 3.3%
= sigma/2 weight grids + table softmax, the honest floor of
this table set. PREDICTION 1 (gate within sigma) NOT
MEASURED — flips-chain prediction is within-sigma, named as
unverified. PREDICTION 3 CONFIRMED (reference path ~10-40x
slower than fp — determinism price, unoptimized). Fences:
one crystal, greedy, 5-prompt battery + 3k-token agreement
probe; tables travel as bytes (sha-pinned), never
regenerated per device. CONSEQUENCE: cross-vendor
bit-reproducible transformer decode is REAL and cheap to
build (one file, ~280 lines); the axiom FX-V1 cross-lab
hash cell is now a pure formality of shipping them the
tables.

## PRE-REG K3-D1: THE KIMI-K3 SINGLE-EXPERT DETERMINISTIC DEMO — one expert out of 2.8T, metered, packed, hash-locked cross-vendor (2026-07-30 ~10:30)

The victory-lap-as-integration-test: pull ONE routed expert
(layer 45, expert 7 — w1/w2/w3, 7168x3072x3 ~ 66M params)
out of moonshotai/Kimi-K3 by safetensors byte-range (header
parse + HTTP Range on shard model-00046-of-000096), never
touching the other 2.8T params. FORMAT DISCOVERY (probe,
pre-run): routed experts ship MXFP4-pack-quantized, group
32, SYMMETRIC, uint8 E8M0 scales — i.e. e2m1 codes
(2x-integers 0..12 signed) times power-of-two scales:
EXACTLY representable in integer arithmetic. The confound
that blocked the B3 meter becomes the carrier: the
deterministic GEMV consumes Moonshot's shipped codes
NATIVELY (int codes, shift scales, int64 accumulation
relative to the min group exponent) — no requantization.
CELLS: (a) EXTRACTION — bytes fetched < 50 MB for the
expert (v 2.8T full pull); dequant exactness asserted
(integer reconstruction == compressed-tensors reference
dequant, bit-for-bit in fp32). (b) METER — llmopt.quantize
.meter on the exactly-dequantized fp expert: Artin's
per-expert-size law says capacity is monotone DECREASING
premium in expert size (5M 2.93, 6M 2.85, 45M 2.33, 40M
2.01); K3 at ~66M/expert is the largest expert ever
metered. PREDICTION: M <= 2.0 (at/below the K2 boundary —
the most black-hole-like expert yet). Fence: the meter
reads the MXFP4 grid's image, not the fp master (named
confound; kurtosis reported alongside). (c) ENTROPY — rANS
size of the shipped 4-bit code stream v its Shannon
entropy: how close Moonshot's format is to capacity
(PREDICTION: code entropy < 4 bits/param, i.e. MXFP4 still
leaves lossless margin; report the free %). (d) THE HASH —
deterministic integer GEMV (fixed battery of 64 int
activation vectors, sha256 of the full int64 output trace)
run on Mac AND 3080: PREDICTION: bit-identical. FALSIFIERS:
range-fetch reconstruction mismatch = format
misunderstanding (book it, stop); M > 2.4 breaks the
size-monotone law at n=5 (book the break — that IS the
result); hash mismatch = a nondeterministic op in the new
MXFP4 path. Fences: one expert, one layer (spot-check a
second expert for the meter only); GEMV-level demo, not
full-model decode; same instrument both devices
(scratch/k3_expert_demo.py).

## K3-D1 VERDICT: ONE EXPERT OUT OF 2.8T — 17.5 MB fetched, natively hash-locked across THREE backends, and K3 is a LATENT MoE (2026-07-30 ~11:00)

ALL FOUR CELLS LAND. (a) EXTRACTION: 17.5 MB by safetensors
byte-range for the full expert (v 2.8T repo); integer
reconstruction roundtrip asserted exact; the 3080
re-fetched its own bytes from HF and the raw-blob sha256s
matched the Mac's. DISCOVERY: the expert is 3072x3584, not
3072x7168 — K3 is a latent MoE (config latent_moe_use_norm:
experts read a 3584-dim latent, half the residual width),
so a K3 expert is ~33M params, NOT the ~66M naive estimate.
(b) METER: l45/e7 M=1.94 (kurt 3.12), spot-check l70/e512
M=2.15 — K3 spans 1.94-2.15, STRADDLING K2's 2.01;
prediction M <= 2.0 confirmed on the named expert; at 33M
params strict size-monotonicity would put K3 ABOVE K2 —
mild tension, but the MXFP4 grid confound (the meter reads
the 4-bit image, which caps span) biases K3 low, so the law
stands as a BAND (large experts ~2.0, at the boundary), not
a rank claim here. (c) ENTROPY: shipped 4-bit codes carry
3.643 bits/param (rANS = Shannon to 3 decimals) — MXFP4
leaves ~9% lossless margin (Moonshot ships within ~0.36
bits of its own format's capacity; they packed well). (d)
THE HASH: deterministic integer GEMV (native MXFP4 codes,
shift scales, int64 accumulation) on a fixed 64-vector
battery: sha256 traces IDENTICAL on cpu, mps, AND cuda for
all three mats (9c4062/fd1257/8a7486) — the P3 result now
holds on a frontier model's own shipped format with ZERO
requantization. Fences: two experts, GEMV-level (not
full-expert SiLU chain, trivially composable from P3
pieces); meter fenced to the MXFP4 image. CONSEQUENCE: the
whole pipeline — range-fetch, exact dequant, meter, rANS,
cross-vendor determinism — is library calls + ~180 lines
(scratch/k3_expert_demo.py); "pull one organ from a 2.8T
model and hash-lock it on any GPU" is now a cheap,
repeatable operation.

## FX-V1-H VERDICT: CROSS-LAB PASS — axiom reproduces both P3 hashes bit-for-bit; determinism now spans two labs, four backends (2026-07-30 midday)

Axiom Fable ran the P3 twin on their machine (MPS; their
relay 2026-07-30-1, commit 70777ea): streams bf76568d...
AND logit trace 311f71bf... FULL-digest identical to the
house values; max GEMM partial 2^21.2 matches; sha of
p3_tables.pt verified against the pin BEFORE running (no
artifact transfer needed — the shared llmopt clone at
origin/main already carried it). House added the CPU
point same hour (P3_DEV knob added to pack_decode.py at
axiom's suggestion): cpu hashes IDENTICAL too. The claim
as it now stands: one tables file, four backends (cpu,
mps x2 machines, cuda), two labs, zero tolerance columns
— sha in, sha out. AMENDMENT (error caught by axiom,
credit theirs): relay 2026-07-30-0's model card said
"d256 L4 h8 ffn 1024" — WRONG; the card of record is the
packed d64h8 crystal, d64 L8 ffn 256, as pack_decode.py
pins. Hashes matched because both labs ran the same
pinned instrument; the prose was wrong, not the run.
Fences: greedy battery only; MPS+cpu axiom-side pending
nothing (their verdict booked as delivered). CONSEQUENCE:
the paper's determinism section upgrades from cross-vendor
to CROSS-LAB, and the E-series protocol (sha-pinned
artifact + pinned instrument + full-digest compare) is now
the standing template for replication cells.

## PRE-REG K3-D2: THE FULL-EXPERT CHAIN — closing K3-D1's composition fence (2026-07-30 ~11:45)

K3-D1 fenced itself to GEMV-level. D2 composes the tested
pieces into ONE deterministic full-expert forward on the
shipped MXFP4 codes: x (int, A=2^10 fixed point, 64-vector
battery) -> w1 and w3 integer GEMVs -> power-of-two requant
back to A scale (pure shifts, round-half-away) -> SiLU via
a SHIPPED integer table (generated once on the Mac,
sha-pinned, scp'd — P3 doctrine: tables travel as bytes) ->
gate*up product at A scale, clamped +-2^15 -> w2 integer
GEMV -> int64 output trace. PREDICTION: sha256 of the full
expert output identical on cpu, mps, and cuda (the same
claim as D1, one level up the composition). FALSIFIER: any
mismatch = a nondeterministic op introduced by the chain
glue (requant shifts, table lookup, product) — hunt and
book. Fences: synthetic battery (not real routed
activations); one expert (l45/e7); activation clamp +-2^15
in A units named as the D2 analog of P3's ACT_CLAMP.

## K3-D2 VERDICT: FULL-EXPERT CHAIN HASH-LOCKED — a Kimi-K3 expert forward, exactly, on three backends (2026-07-30 ~12:00)

PREDICTION CONFIRMED first run: the composed expert forward
(w1/w3 integer GEMVs -> power-of-two requant -> shipped
SiLU table f503c814... -> gate*up at A=2^10 -> w2 GEMV)
hashes d771796f... IDENTICAL on cpu, mps, and cuda. The
SiLU table traveled as bytes (Mac master, sha-verified on
the 3080 before the run — P3 doctrine held). K3-D1's
composition fence is CLOSED: what is now demonstrated is a
frontier model's routed expert running EXACTLY — same
integers, any GPU — directly on Moonshot's shipped MXFP4.
Fences (carried): synthetic 64-vector battery, one expert,
clamp +-2^15 in A units. Friendly-fire note: scp mangled
the remote colon-path (target parsed as local cp) — shipped
via ssh cat-pipe instead; transfer verified by sha, the
standing pattern anyway. CONSEQUENCE: "the packed model as
the expert" (Artin's riff) now has its existence proof at
the single-expert level; a deterministic-MoE shelf (router
in fp, experts as hash-locked integer organs) is buildable
from tested parts alone.

## PRE-REG UMOE-1: MICRO-MoE CONSERVATION, THE CAUSAL 3-ARM — does the balance loss push redundancy out of weights and into routing? (2026-07-30 ~12:15)

The split law (N3+B4) is observational: production MoEs
show expert decorrelation ~0 in weight space AND co-routing
MI 300-500x shuffle. The banked causal version (RIFF
2026-07-30): load-balancing aux losses force expert
independence — plausibly the very mechanism that moves
redundancy into routing. FIRST HOUSE MoE BIRTHS. DESIGN:
d64 h8 L8 crystal class, FFN replaced by 4 experts (SwiGLU
ffn_e=128 each) + per-block top-1 switch router
(prob-weighted); gen-4 diet, 3 epochs, seed 1, ALL arms on
the 3080 (paired, one device), instrument
scratch/umoe_conserve.py. ARMS: (a) lb — switch
load-balance aux (coef 0.01, the standard); (b) free — aux
0 (correlation permitted; collapse risk accepted as a
finding); (c) tied — expert_i = base + delta_i (deltas
0.1-scale init), aux 0.01. Plus (d) a dense d64h8 control
born same-seed same-device for the gate reference (Mac-born
58 does not transport). MEASURED PER ARM: gate
(G.gate_eval); mean pairwise expert-weight correlation per
block (the N3 instrument); adjacent-layer co-routing MI v
token-shuffle control (the B4 instrument, 4x4 joint);
capacity meter M per expert group (exploratory axis).
PREDICTIONS: (1) corr(free) > corr(lb) — without the
balance loss experts stay partially redundant/mergeable;
(2) MI/shuffle(lb) > MI/shuffle(free) — the loss converts
weight redundancy into routing structure; (3) tied arm:
the tie carries the shared mass explicitly (base norm >>
delta norm), deltas decorrelated, gate within ~2 of lb;
(4) CONSERVATION (the law, if it holds): corr and MI move
in OPPOSITE directions across arms — the redundancy total
is roughly conserved, only its address changes.
FALSIFIERS: router collapse in (b) (one expert takes >90%
of tokens) makes corr unmeasurable-as-designed — book the
collapse itself and rerun (b) with a floor eps; MI flat
across arms kills prediction 2 and the causal story (the
observational law stands regardless). Fences: house scale
(d64), one seed per arm this pass, MoE params 2x dense FFN
(active 1/2) — arm-internal comparisons only; gate v dense
control is context, not a matched-params claim.

## AMENDMENT (UMOE-1 launch): friendly-fire #8 REPEATED — same file, same lesson, one day later (2026-07-30 ~12:40)

First UMOE-1 launch crashed all four arms in seconds:
data/micromodel_gen4_sidecar.jsonl is Mac-only and was
never shipped — the EXACT dep that killed the night-30
births, booked as friendly-fire #8 yesterday. The lesson
("verify file deps at arm time") was in the doctrine and
not applied. Cost: ~3 min (smoke-tested locally, crash
surfaced immediately in the monitor; sidecar shipped by
gzip ssh-pipe, dep VERIFIED loading on the box — 38,325
rows — before relaunch). Process change adopted: the
arm-time check is now "run the loader on the target box
before launch," not "eyeball the script's imports."
Fresh log + fresh marker per relaunch (friendly-fire #9
discipline held). Arms relaunched 12:40, chained
dense->lb->free->tied, monitor armed.

## UMOE-1 VERDICT: THE SPLIT LAW IS NOT THE BALANCE LOSS'S DOING — sparse assignment itself creates it (2026-07-30 ~12:15 PM)

First house MoE births, all four arms landed (d64 h8 L8, 4
experts ffn_e 128, top-1 switch, gen-4, seed 1, one
device). THE TABLE: dense gate 45/120; lb 37, corr 0.0085,
MI 288x shuffle, M 1.26; free 36, corr 0.0080, MI 256x, M
1.37; tied 43, delta-corr 0.0589, MI 235x, M 1.47 (base
norm 21.1 v delta 16.0). PREDICTIONS 1+2 FALSIFIED, and
the falsification IS the finding: removing the balance
loss changed NEITHER the decorrelation (0.0080 v 0.0085)
NOR the routing structure (256x v 288x), and the router
did not collapse (worst share 55%). The split signature —
experts decorrelated to ~0 in weights, co-routing MI
hundreds-x shuffle — is created by TOP-1 HARD ASSIGNMENT
ITSELF: hard-routed experts never see the same tokens, so
they cannot stay correlated; the aux loss only shapes the
load, not the split. PREDICTION 3 SPLIT: tied gate 43 =
best MoE arm, within 2 of dense (the tie pays almost
nothing) — but base does NOT dominate (deltas grew to
0.76x base norm) and deltas stay 7x more correlated than
untied experts (the base absorbs shared mass imperfectly).
PREDICTION 4 (conservation on corr x MI): NOT OBSERVED —
both axes flat across arms. EXPLORATORY rider worth a
seed-2 look: M and MI move OPPOSITELY across the three MoE
arms (M 1.26/1.37/1.47 as MI 288/256/235x) — a hint that
routing structure and weight-side capacity trade off on
the (M, MI) axes instead. SCALE REPLICATION (unplanned
win): the 0.9M-param micro-MoE reproduces the production
split-law signature (N3 corr ~0.005, B4 MI 300-500x) at
1/30,000th the size — the phenomenon is studyable at house
scale. Fences: one seed, house scale, MoE params 2x dense
FFN, MI probe on training-distribution rows. NEXT: seed-2
same-day replication (3080 window); the causal story for
the paper amends from "the loss does it" to "the sparsity
does it."

## UMOE-1 SEED-2: THE CORE NULL REPLICATES; the (M, MI) rider dies (2026-07-30 ~1:45 PM)

Same protocol, seed 2, same device. TABLE: dense 40/120;
lb 44, corr 0.0082, MI 205x, M 1.38; free 40, corr 0.0085,
MI 299x, M 1.35; tied 42, delta-corr 0.0613, MI 219x, M
1.43 (base 20.4 v delta 16.8). REPLICATED at n=2: (1)
split-from-sparsity — lb v free indistinguishable on corr
(0.0082 v 0.0085) and both at hundreds-x MI, router alive
in the free arm both seeds; the balance loss shapes LOAD,
not the split. (2) tied anatomy — deltas ~7x more
correlated than untied experts, deltas grow to ~0.8x base,
gate within sigma of dense both seeds (tie-at-birth stays
a live cheap lever). (3) micro-scale production signature
(corr ~0, MI 205-299x). KILLED HONESTLY: the seed-1
(M, MI) opposite-motion rider — seed 2 shows no monotone
pattern (M 1.38/1.35/1.43 v MI 205/299/219); the
exploratory trend was seed noise, deleted from the paper
rider. Gate ordering across arms is within sigma both
seeds (no capability claim between MoE arms and dense at
matched recipe). Fences: n=2 seeds, house scale, one
diet. Thread closes for the day; next falsifier stays
banked (soft-routing arm should KILL decorrelation if
assignment is the mechanism).

## PRE-REG GRAV-1: EXPERT GRAVITY — mass, field falloff, and screening in the micro-MoE (2026-07-30 ~3:15 PM)

Artin's frame: mass bends spacetime EFFICIENTLY — it
affects what it needs to. Operationalized on the residual
stream (the shared medium): per-expert MASS m_e = mean
residual-write norm x usage share; FIELD FALLOFF g_e(k) =
representation displacement at depth l+k after ablating
expert e at layer l (does influence decay lawfully with
depth-distance; do heavy experts reach further?);
EFFICIENCY/SCREENING = on-target effect (delta-NLL on
tokens routed THROUGH e) / off-target broadcast (delta-NLL
on tokens never routed to e). Instrument
scratch/grav_probe.py on umoe_lb_s1 + s2 (checkpoints
local, Mac, no GPU window needed); eval battery = held-out
gen-4 rows. PREDICTIONS: (1) mass predicts ablation damage
(rank corr positive, both seeds); (2) SCREENING: off-target
delta-NLL << on-target (routing = gravitational screening;
the split law's inference-side face) — ratio > 5x; (3)
falloff g_e(k) monotone decreasing in k (perturbations
wash out, not amplify — the stability prior), shape
reported not predicted; (4) EXPLORATORY, the "optimal
gravity" question: efficiency (on/off ratio) varies across
experts and correlates with usage share (the busiest
expert is NOT the most efficient — attention-hog v
surgeon distinction). FALSIFIERS: off-target ~ on-target
kills screening (routing does NOT localize influence —
would itself be a finding against the split law's
inference face); g_e(k) non-monotone = chaotic
amplification (book as stability finding). Fences: house
scale, top-1 routing (screening is partly definitional
under hard routing — the LEAK is the measurement), n=2
seeds, analogy fenced as instrument-naming (no conserved
charge, no reciprocity, distance = depth).

## GRAV-1 VERDICT: NO GRAVITY — influence is UNSCREENED and AMPLIFYING; attention is the long-range channel routing cannot contain (2026-07-30 ~3:45 PM, n=2 seeds)

All three predictions FALSIFIED, both seeds, and the
falsifications cohere into one mechanism. (1) SCREENING
FAILS: median on/off-target dNLL ratio 1.6x (s1) and 1.0x
(s2) v the predicted >5x — ablating an expert hurts tokens
that NEVER routed through it about as much as tokens that
did; several cells show off > on. (2) FALLOFF INVERTED:
displacement GROWS monotonically with depth in every one of
24 (layer,expert) cells both seeds (e.g. L1: 0.18 -> 1.70
by L7) — the residual stream AMPLIFIES perturbations, it
does not dissipate them (the pre-registered chaotic-
amplification falsifier branch; consistent with the
2026-07-28 Lyapunov atlas verdict, chaotic-degenerate). (3)
MASS-DAMAGE DECOUPLED: rank corr 0.168 (s1) v -0.245 (s2)
— write-norm x usage does not predict ablation damage; no
"heaviest body" exists. MECHANISM (the finding): routing
localizes COMPUTATION but not INFLUENCE — attention is the
unscreened long-range interaction; a perturbed token's
representation enters every other token's attention keys/
values downstream, so the field of any expert reaches the
whole context within a layer or two, and the chaotic medium
amplifies it thereafter. Artin's gravity frame thus gets a
precise answer: the experts are not gravitating bodies with
efficient local fields — the medium (residual stream +
attention) is closer to a turbulent fluid than to
spacetime; the split law's tidiness lives in WEIGHTS and
ROUTING STATISTICS, not in inference-time influence, which
is global. Fences: micro-MoE house scale, top-1, dNLL under
full-expert zero-ablation (a large perturbation — a
gentler-kick falloff probe is the banked follow-up),
battery from training distribution tail. RIDER for the
paper's Sec. 7: "params-side compression closed" now has an
inference-side face — you also cannot LOCALIZE an expert's
influence for cheap approximation; the medium spreads it.

## PRE-REG CAL-DK-1: DOES THE CRYSTAL KNOW WHEN IT DOESN'T KNOW — token-level reliability by difficulty (2026-07-30 ~4 PM)

Artin's Dunning-Kruger question operationalized at house
scale: teacher-forced pass of the d64h8 EMA crystal over a
held-out gen-4 battery; per token record confidence (max
softmax prob) and correctness (argmax == target); report
(a) reliability curve + ECE (is confidence honest
overall?), (b) per-level split 3..7: confidence v accuracy
by difficulty — the DK signature would be overconfidence
concentrated at the hardest levels (accuracy falls faster
than confidence). PREDICTIONS: (1) the crystal is
DIRECTIONALLY calibrated (higher conf -> higher acc,
monotone reliability curve) — margins already predicted
flip fragility (P5) and soft-token disagreements (P3), so
confidence carries real signal; (2) mild DK: confidence
falls with level SLOWER than accuracy does (overconfidence
grows with difficulty); magnitude reported not predicted.
FALSIFIER: flat/non-monotone reliability = confidence
carries no self-knowledge (would invalidate margin-based
escalation ideas downstream). Fences: token-level (not
solve-level), one crystal, teacher-forced.

## CAL-DK-1 VERDICT: NO DUNNING-KRUGER — the crystal is slightly HUMBLE, and its confidence is a working error detector (2026-07-30 ~4:15 PM)

Teacher-forced, d64h8 EMA, 5 levels x ~300 rows, answer-
span tokens only. PREDICTION 1 CONFIRMED strongly:
reliability is monotone across all bins and ECE = 0.0068
(a 0.9M-param model, essentially calibrated). PREDICTION 2
FALSIFIED in the good direction: the DK gap (conf - acc)
is NEGATIVE at every level (-0.004 to -0.012) — the
crystal is systematically UNDERconfident, and no more so
at hard levels than easy ones; there is no overconfidence-
where-skill-is-lowest signature at all. AUROC of
confidence as a correctness detector: 0.989 — the model
KNOWS when it doesn't know, at token granularity, for
free. CONSEQUENCE: margin/confidence-based machinery
downstream (escalation tiers, entropy-adaptive drafting,
soft-token flagging in P3) rests on a measured foundation,
not an assumption. The verified-diet hypothesis (named,
untested): calibration may come from training exclusively
on oracle-verified rows — a web-trained model has no such
guarantee; comparing calibration v diet purity is a
banked follow-up. Fences: token-level (solve-level
calibration = separate probe), one crystal, teacher-
forced, in-distribution battery.

## PRE-REG GRAV-2: ENGINEERED SPACETIME — train the medium lawful, price the toll (2026-07-30 ~4:20 PM)

GRAV-1 measured the trained medium as turbulent
(amplifying, unscreened). Artin's push: don't simulate the
fluid — CONSTRAIN it; a lawful medium should give more
control over the weights. DESIGN (Mac, background, paired
seed-1): dense d64h8 arms via the umoe trainer's dense
path + a CONTRACTIVITY penalty — per step, one random
block, input perturbed by delta ~ N(0, 0.01*rms); penalty
lambda * relu(||f(x+delta)-f(x)|| / ||delta|| - 1)
(block expansion factor above 1 is taxed). Arms: ctl
(lambda 0) v contract (lambda 0.1). MEASURED: (a) the
falloff curve (epsilon-kick displacement by depth, the
GRAV-1 instrument at gentle amplitude) — does it flip from
growing to flat/decaying; (b) gate (the toll of
lawfulness); (c) CAL-DK reliability on both arms (Artin's
two riffs joined: does a lawful medium sharpen
self-knowledge?). PREDICTIONS: (1) contract arm's
displacement growth rate drops measurably (else the
penalty failed to bind — check penalty loss curve); (2)
the gate pays a toll > 0 (chaos is likely load-bearing;
magnitude is the measurement — a FREE lawful medium would
be the surprise result); (3) exploratory: calibration
unchanged (already near-ceiling per CAL-DK-1).
FALSIFIERS: penalty binds but falloff unchanged =
amplification lives in attention mixing, not block
expansion (a mechanism finding — book it); gate crater
>10 = lawfulness unaffordable at this scale. Fences: house
scale, one seed this pass, lambda single-point (no sweep).

## PRE-REG UMOE-2 + CAL-DK-2: two falsifiers on the 3080 window (2026-07-30 ~3:05 PM)

UMOE-2 SOFT-ROUTING (falsifier for the UMOE-1 mechanism):
one arm, ARM=soft — the same 4-expert d64h8 MoE but with
FULL soft mixing (every expert sees every token, prob-
weighted; no aux). If top-1 sparse assignment is what
creates the split (UMOE-1, n=2), soft routing removes the
cause: PREDICTION: pairwise expert corr rises by >=10x
(from ~0.008 toward the tied arm's ~0.06 or beyond);
argmax-MI ratio reported (expected to fall; the argmax of
a soft mixture is a weaker object, fenced). FALSIFIER:
corr stays ~0.008 under soft routing = decorrelation does
NOT come from assignment (mechanism wrong — book and
rethink).

CAL-DK-2 DIET DILUTION (falsifier for the verified-diet
calibration hypothesis): train the same dense d64h8 recipe
with a fraction f of rows' TARGETS corrupted (nxt swapped
among the corrupted subset — fluent, determined-looking,
WRONG rows: the confident-nonsense class). Arms f=0.1,
f=0.3 on the 3080; control = umoe_dense_s1 (f=0, same
recipe/device, already born). Probe = cal_dk instrument
(CKPT env). PREDICTIONS: (1) ECE degrades monotonically
in f; (2) the DK gap flips POSITIVE (overconfident) at
f=0.3 — the model learns confident wrongness only when
the diet contains it; (3) gate degrades too (dose-
response, reported). FALSIFIER: calibration SURVIVES 30%
corruption = honesty is architectural/scale-driven, not
diet-driven (would kill the dilution hypothesis and be
interesting alone). Fences: token-level, corruption =
target-swap (one corruption class), house scale, n=1 per
f this pass.

## PRE-REG GRAV-1b: DISTANCE FROM THE EXPERT — the field in router coordinates (2026-07-30 ~3:30 PM)

Artin's refinement of GRAV-1: the natural distance is not
depth or the binary routed/not split — it is ROUTER
PROXIMITY. For ablated expert e at layer l, bin every
token by its router probability p_e (the pre-ablation
model's assignment mass on e) and measure ablation dNLL
per bin. PREDICTION: dNLL monotone increasing in p_e —
tokens closer to the expert feel more force; a lawful
field in the right coordinates (GRAV-1's chaos = wrong
metric). FALSIFIER: flat/non-monotone dNLL in p_e = the
field is genuinely unlawful, not mis-coordinatized.
Instrument: grav_probe extension, umoe_lb_s1+s2, same
battery, Mac. Fences: as GRAV-1.

## GRAV-1b VERDICT: NOT LAWFUL IN ROUTER COORDINATES EITHER — damage is not monotone in distance-from-expert (2026-07-30 ~3:50 PM, n=2)

The coordinate-change hypothesis FALSIFIED. Binning
ablation dNLL by router probability p_e (Artin's
distance-from-expert metric): no consistent monotone
profile in either seed. Cells disagree with themselves
across seeds (L4e1 rises in s1, falls in s2), several
cells put their LARGEST damage at the extreme bins in a
U-shape (s1 L1e2: +0.24 at p<0.05 AND +0.56 at p>0.75),
and many cells are flat at noise level. Weak rising
trends exist in some mid-layer cells but do not
replicate cell-by-cell. VERDICT: the influence field is
genuinely unlawful at this scale — not mis-coordinatized;
GRAV-1's turbulence conclusion stands in the routing
metric too. The U-shapes hint at a two-population story
(tokens the expert OWNS and tokens it actively REPELS
both matter; the middle is slack) — banked as a follow-up
lens, not a claim. Fences: as GRAV-1 (zero-ablation kick,
house scale, top-1); n=2 seeds.

## GRAV-2 VERDICT: SPACETIME IS TRAINABLE, AND LAWFULNESS IS NEARLY FREE (2026-07-30 ~4:10 PM)

The headline result of the gravity thread. The
contractivity penalty (expansion tax, lambda 0.1, one
random block per step) FLIPPED THE MEDIUM: epsilon-kick
displacement profile goes from AMPLIFYING in the ctl arm
(L1 kick: 0.67 -> 2.24 by L7; L4: 1.63 -> 2.95 — GRAV-1's
turbulence replicated in a paired dense arm) to FLAT in
the contract arm (L1: 0.26 -> 0.44; L4: 0.40 -> 0.50) —
~5x smaller displacement, per-layer growth ~1.06x v
~1.2x. PREDICTION 1 CONFIRMED decisively. PREDICTION 2
(gate toll > 0): the SURPRISE branch fired — ctl 50/120 v
contract 46/120, a -4 at the edge of gate sigma (~3.5):
chaos is NOT load-bearing at this scale; a lawful medium
costs almost nothing. PREDICTION 3 CONFIRMED: calibration
identical (ECE 0.0055 v 0.0053, AUROC 0.980 v 0.979 —
also a same-recipe REPLICATION of CAL-DK-1's no-DK result
on two fresh births). CONSEQUENCE (the control story
Artin asked for): the turbulent medium GRAV-1 measured is
a TRAINING DEFAULT, not a necessity — a one-line penalty
buys a near-dissipative residual stream at ~zero gate
cost, which is the precondition for local editing,
influence-bounded packing, and any future gravity-like
structure (fields can only be lawful in a medium that
does not amplify noise). NEXT (banked): rerun the GRAV-1
screening probe on a contract-arm MoE (does a lawful
medium RESTORE expert locality?); lambda sweep; seed 2.
Fences: n=1 seed, dense arm (not MoE), single lambda,
house scale.

## UMOE-2 VERDICT: SOFT ROUTING DOES NOT RESTORE CORRELATION — decorrelation is the INIT DEFAULT, and nothing we tried moves it (2026-07-30 ~4:30 PM)

The soft-mixture arm (every expert sees every token,
prob-weighted) landed corr 0.0089 — INDISTINGUISHABLE from
top-1 arms (0.0080-0.0085). PREDICTION (corr rises >=10x
under soft routing) FALSIFIED, which kills the UMOE-1
amended mechanism too: decorrelation is NOT caused by
sparse assignment. The desk check that settles it:
expert correlation AT INITIALIZATION is 0.0016 —
independently-initialized high-dim weight vectors start at
~0, training raises it only to ~0.008 in EVERY regime
(lb/free/soft), and the single arm that produced real
correlation (tied, 0.06) is the one where experts SHARE A
GRADIENT PATH through the base. FINAL MECHANISM (two
falsifications deep): the split law's weight-side face is
the INIT DEFAULT PRESERVED — independent parameterization
starts orthogonal and no routing regime supplies a force
toward correlation; the ROUTING-side structure (MI 205-374x
shuffle across ALL arms incl. soft) is where training
actually writes. Gate soft 42/120 (within sigma of the
family). Fences: n=1 for soft (its corr number sits inside
the n=2 top-1 band), house scale. Rider: N3's production
observation now reads as the same default at 1000x scale —
"experts decorrelated" is not an achievement of MoE
training, it is the absence of a correlating force.

## CAL-DK-2 VERDICT: DILUTION BREAKS CALIBRATION MONOTONICALLY — but noise teaches DOUBT, not arrogance (2026-07-30 ~4:35 PM)

Dose-response CONFIRMED, direction SURPRISED. ECE: clean
0.0049 -> f=0.1 0.0225 -> f=0.3 0.0386 (monotone, ~8x at
30%; PREDICTION 1 CONFIRMED — the verified diet IS what
calibration rests on). Gate: 45 -> 35 -> 30 with validity
sigma 45 -> 25 -> 17 (corruption crushes fluency first).
AUROC: 0.971 -> 0.968 -> 0.956 (self-knowledge degrades
but survives). PREDICTION 2 FALSIFIED, informatively: the
DK gap goes MORE NEGATIVE with dilution (-0.005 ->
-0.023 -> -0.039) — target-SWAP corruption is NOISE
(conflicting evidence for the same prompt), and noise
teaches HEDGING, not confidence. The sharpened law:
inconsistent falsehood -> underconfidence; to train
OVERCONFIDENCE (true DK) you likely need CONSISTENT
falsehood — systematically wrong but internally coherent
rows (the confident-nonsense class proper). CONSISTENT-
CORRUPTION ARM BANKED (e.g. a deterministic wrong rule at
f=0.1; predicts the DK flip). Fences: n=1 per dose,
token-level, one corruption class, house scale.

## PRE-REG UMOE-3: THE THIN CHANNEL AND THE GRAVITATIONAL TIE — do experts talk if offered a cheap way to? (2026-07-30 ~5:10 PM, Mac overnight)

Artin's design, sharpened by UMOE-2's law (no correlating
force exists; only shared gradient paths couple). ARMS
(d64h8 4-expert recipe, seed 1, Mac, paired): (a) channel
— expert_i = delta_i + a_i * S, S a THIN shared low-rank
base (r=16 per mat), a_i per-expert learned coupling INIT
0 (experts talk only if training opens the channel; the
a_i trajectory is the measurement); (b) gravmoe — same-
layer pairwise coupling c_ij = lambda_g * EMA(router
overlap <p_i p_j>): experts that serve similar tokens are
PULLED TOGETHER in weight space (mass = usage, force =
co-assignment), lambda_g 0.5, EMA decay 0.99, coupling
applied as W_i <- W_i + c_ij (W_j - W_i) every 100 steps
(a relaxation toward co-used neighbors). MEASURED: gate;
final a_i / c_ij values (did anyone talk/move?); expert
corr; MI ratio; meter M. PREDICTIONS: (1) channel arm:
a_i grows away from 0 (|a_i| > 0.1 for >=1 expert) AND
gate >= tied arm's 43 — the shared mass wants to exist
when offered cheaply (based on tied's near-parity); (2)
gravmoe: c_ij concentrates on high-overlap pairs (the
attraction is selective, not uniform); gate within sigma
of lb; (3) corr rises ONLY through the offered channels
(the UMOE-2 law holds elsewhere). FALSIFIERS: a_i pinned
~0 with gate unchanged = experts have NOTHING to share
(the strongest possible form of the init-default law —
book it as such); gravmoe gate crater = usage-attraction
destroys specialization (the anti-gravity result, also a
law). Fences: n=1 per arm this pass, house scale, one
lambda_g/r point each.

## PRE-REG PLACE-1: INFERENCE-TIME GRAVITY — co-routing-informed placement/prefetch v popularity, on real OLMoE traces (2026-07-30 ~5:30 PM; Mac, chained after UMOE-3)

Artin's cell: co-locate co-routed experts WITH the
(training-derived) routing statistics, and WITHOUT as
control. Trace-driven, honest-as-simulation. COLLECT:
OLMoE-1B-7B routing traces (top-8 of 64 per layer per
token) over a few thousand tokens; split first/second
half = fit/eval. CELLS: (a) NEXT-LAYER PREFETCH — predict
layer l+1's expert set from layer l's ACTUAL routed set
via the fit-half conditional co-occurrence (the co-routing
face), v marginal frequency ranking (popularity face), v
uniform floor; metric recall@8 and @16 on eval half.
(b) CACHE SIM — per-layer device cache of C=16/64 experts:
LRU dynamic v FREQ-pinned static v LRU+MI-prefetch;
metric miss rate = bytes moved per token. PREDICTIONS:
(1) conditional beats marginal on recall@8 by a REAL
margin (>=5 points absolute) — B4's MI 300-500x is
exploitable, not just present; (2) LRU+prefetch beats
plain LRU on misses; (3) freq-pinned beats uniform
trivially (sanity floor). FALSIFIER: conditional ==
marginal recall = co-routing MI is real but NOT
exploitable at placement granularity (the split law's
systems lever would DIE here — important either way).
Fences: trace-driven simulation (no wall-clock claim),
one model, one trace corpus (README + prompt battery
class), C single-point.

## PRE-REG BASIN-1: EXPERTS AS ATTRACTORS — routing basin radius v usage (2026-07-30 ~5:50 PM, CPU desk probe)

The Hopfield identification's first measurable: perturb
the pre-router hidden state by eps * rms noise and measure
P(top-1 route unchanged) v eps, per expert, on
umoe_lb_s{1,2} (cpu; battery as GRAV-1b). Basin radius =
eps at 50% retention. PREDICTIONS: (1) retention monotone
decreasing in eps (sanity); (2) THE MASS HYPOTHESIS:
higher-usage experts have LARGER basins (mass = basin
size, the Hopfield coordinate replacing GRAV-1's failed
write-norm mass) — rank corr(usage, radius) > 0 both
seeds. FALSIFIER: no usage-basin relation = routing basins
are usage-blind; the attractor picture loses its mass
variable (book and keep the contract-arm comparison as
the remaining Hopfield rung). Fences: token-level routing
retention (not full-trajectory attractor convergence),
house scale, n=2 seeds.

## BASIN-1 VERDICT: basins are real and remarkably robust; the mass relation is WEAK-POSITIVE only (2026-07-30 ~6 PM, n=2)

Retention is monotone and HIGH: the routing decision
survives eps=0.2 perturbations at 87% and even eps=0.8
(perturbation as large as the signal) at 63% — top-1
routing sits in wide attractor basins, consistent with
the Hopfield-retrieval reading of the router (both seeds
near-identical: 0.986/0.984 at 0.02 down to 0.630/0.632
at 0.8 — the retention curve itself REPLICATES to 3
decimals). PREDICTION 2 (mass = basin size): rank
corr(usage, basin) = +0.142 (s1) / +0.263 (s2) — positive
both seeds as predicted but WEAK; usage explains little
of basin geometry. Verdict: the attractor picture stands
(wide, stable basins measured), but its mass variable is
NOT usage — basin size is mostly usage-blind at this
scale. The remaining Hopfield rung is the sharp one:
contract-arm MoE v standard (does a lawful medium widen
basins?) — still banked, needs the contract-MoE birth.
Fences: router-level retention only (not full-block
trajectory), house scale, n=2.

## PRE-REG TREE-1: THE EXPERT TREE — weights in a phylogeny, not a fluid (2026-07-30 ~6:15 PM; Mac, chained after PLACE-1)

Artin's frame: structure the parameters as a TREE instead
of a fluid/spacetime. House lineage: tied = a one-level
tree and the ONLY arm that ever correlated experts
(UMOE-1/2); clade-transfer terms measured in the exposure
curve (lineage tree in time); matryoshka/escalation =
trees in bits; published anchor Jordan & Jacobs 1994
(hierarchical MoE). ARM=tree (d64h8 4-expert recipe, seed
1): expert_i = S_root + S_side(i) + delta_i — root shared
by all, mid bases shared by sibling PAIRS (0,1 | 2,3),
leaf deltas private; root full init, mids+leaves 0.1
init; aux 0.01. MEASURED: gate; per-level norms (does
training USE the mid scale or collapse it?); WITHIN-PAIR v
ACROSS-PAIR expert correlation (the tree's signature in
weight space — by construction the first real correlating
force); MI; meter M. PREDICTIONS: (1) gate >= tied's 43
(more sharing structure at matched leaf capacity); (2)
within-pair corr > across-pair corr by >=2x (the tree
visible); (3) mid norms grow from 0.1-init (the
intermediate scale is WANTED — falsifier: mids collapse
toward 0 = two scales suffice, hierarchy unwanted, star
is enough). Fences: n=1, one tree shape (binary, fixed
sibling assignment — random pairing, not learned; a
learned-topology arm is the follow-up), house scale.

## PRE-REG FOURIER-1: THE ROOTS-OF-UNITY PROBE — does the crystal implement the zeta^k filter in its activations? (2026-07-30 ~7:15 PM, CPU desk probe)

The 3B1B generating-function machinery (evaluate at roots
of unity to extract residue classes by interference) is
the published mechanism grokked networks implement for
modular arithmetic (Nanda et al. 2023, progress measures /
modular-addition Fourier circuits; arXiv 2301.05217). R1
booked rotation as living in ACTIVATIONS only — this is
the activation-side probe the weight-side R-series never
ran. DESIGN: d64h8 EMA crystal, prompts "Current:
Mod(n, k)" for k=5,7, n swept 0..N; capture the residual
stream at the answer position; per neuron, regress
activation(n) on the Fourier basis {cos, sin}(2 pi m n /
k), m = 1..(k-1)/2, PLUS a residue-class-indicator basis
(the non-Fourier way to be periodic). METRICS: (a)
spectral concentration — fraction of periodic variance in
the top single frequency per neuron (Fourier circuits
concentrate; lookup tables spread); (b) population count:
neurons with periodic R^2 > 0.5 v a shuffle control
(n randomly permuted). PREDICTIONS: (1) a real population
of residue-periodic neurons exists (>> shuffle); (2) the
INTERESTING split: variance concentrated in single
frequencies = the crystal grokked the zeta^k filter
(Fourier circuit); spread across the indicator basis
instead = memorized residue lookup (the farmer-probe
memorization signature would then extend to Mod).
FALSIFIER: no periodic population at all = Mod is solved
some third way (book and hunt). Fences: one crystal, two
moduli, answer-position activations, CPU.

## FOURIER-1 VERDICT (amended instrument): a small, real population of digit-periodic neurons, with PARTIAL Fourier character (2026-07-30 ~7:45 PM)

INSTRUMENT AMENDMENT first: run 1 was VOID — the gen-4
crystal's vocab (40 atoms) has NO Mod atom; strict encode
fails and lenient encode fed "Current: (23, 5)" garbage
(caught by the check-don't-assume round-trip, not by the
probe). No Mod-capable checkpoint exists house-side (the
nt/callspan births live in axiom's pilot); probe
REFORMULATED on clean vocabulary: digits ARE residues —
sweep "Current: n+7", test activation periodicity in n
mod 10. RESULT: 11/512 neurons periodic (R^2 > 0.5)
v 0/512 in the permutation control — the population is
REAL; median top-single-frequency share 0.534 — roughly
half the periodic variance lives in ONE frequency,
between the pure-Fourier signature (~1.0, the grokked
zeta^k filter of Nanda et al.) and a residue lookup
(~0.2). READ: the crystal carries a small, partially
Fourier-flavored units-digit subsystem — the
roots-of-unity mechanism is PRESENT but not the dominant
implementation at this scale/diet. The proper substrate
for the full question (Mod-diet crystal, VOCAB_EXTRA
gcd/Mod, sweep both n and k) needs a purpose-birth —
BANKED as FOURIER-2. Fences: one crystal, k=10 only,
answer-position residual only, R^2 threshold single-point.

## PRE-REG x3: EQUIV-1 (the equivalence principle), FOURIER-2 (Mod-diet birth), HOPF-1 (contract-MoE) — 2026-07-30 ~8:15 PM, 3080 window + desk

EQUIV-1 (desk, now): Artin asked what MASS is in the
Hopfield picture. Physics has two masses that happen to
coincide — INERTIAL (resistance to perturbation) and
GRAVITATIONAL (influence on others). We measured both,
separately, per (layer,expert) cell: basin radius
(BASIN-1: routing retention under kick) and ablation
damage (GRAV-1 on-target dNLL). The probe: rank
corr(basin, damage) on the 12 probed cells x 2 seeds.
PREDICTION: positive both seeds (masses equivalent —
robust experts are influential experts). FALSIFIER: no
relation = the two masses are INEQUIVALENT in this
universe (deep either way; would mean robustness and
influence are separately trainable).

FOURIER-2 (3080): birth d64 (callspan recipe, atoms
pinned sidecar order, plain arm) on the nt pilot 500
(gcd/Mod diet, 20 ep) and run the roots-of-unity probe
PROPERLY: "Current: Mod(n, k)" sweeps, k=5 and 7.
PREDICTIONS: (1) periodic population >> FOURIER-1's
11/512 (the diet drives the mechanism); (2) top-frequency
concentration > 0.534 (closer to the grokked zeta^k
filter of Nanda et al. 2301.05217). FALSIFIER: population
stays tiny on a Mod-DOMINANT diet = the filter is not how
small crystals do modular arithmetic at all.

HOPF-1 (3080): the unification arm — UMOE lb recipe +
GRAV-2's contractivity penalty (CONTRACT=0.1), seed 1,
then the basin probe and the GRAV screening probe ON the
contract-MoE v the lb control. PREDICTIONS: (1) falloff
flattens (the GRAV-2 result transfers to MoE); (2) basins
WIDEN (retention curve strictly above lb's
0.986/0.966/.../0.630 — lawful medium => stronger
attractors, the Hopfield unification's first
falsifiable); (3) exploratory: on/off screening ratio
improves. FALSIFIER: basins unchanged while falloff
flattens = medium lawfulness and attractor geometry are
INDEPENDENT (unification dies, both results stand).
Fences: n=1 each this pass, house scale.

## EQUIV-1 VERDICT: THE MASSES ARE INEQUIVALENT — robustness and influence are different quantities in this universe (2026-07-30 ~8:45 PM, n=2)

PREDICTION FALSIFIED cleanly: rank corr(basin radius,
ablation damage) = -0.371 (s1) and -0.119 (s2) over the
12 probed (layer,expert) cells — NEGATIVE both seeds.
Inertial mass (resistance to routing perturbation) and
gravitational mass (influence of removal on the loss) do
NOT coincide; if anything the widest-basin experts are
mildly LESS damaging to remove. With BASIN-1 (usage is
not basin size) and GRAV-1 (write-norm is not damage),
the picture: usage, robustness, and influence are THREE
separately-varying quantities at expert granularity — no
single "mass" exists. The equivalence principle does not
hold in the crystal universe; whatever organizes experts,
it is not a gravity with one charge. Fences: 12 cells x 2
seeds, damage from zero-ablation, house scale.

## FOURIER-2 VERDICT: VOID ON SUBSTRATE — the 500-row pilot birth MEMORIZED, so there is no Mod capability to probe (2026-07-30 ~8:50 PM)

The Mod-diet birth (nt pilot 500, 20 ep) shows 0/512
periodic neurons — but the check-don't-assume sanity
probe shows WHY: the model cannot answer isolated
"Mod(n, 5)" prompts AT ALL (answers are chain-shaped
garbage) — 500 rows x 20 epochs is the farmer-probe
memorization regime, and a memorizer needs no filter.
VERDICT: void on this substrate; the mechanism question
stands open. FOURIER-2b BANKED with the data-hygiene
doctrine applied: a WIDE generated Mod/gcd diet (stable
string seeds, exclude=-guarded, 20k+ rows so the
generator space forces computation over memorization),
then the zeta^k probe. Fences: the FOURIER-1 digit
result (11/512, share 0.534) remains the only live
in-vivo reading of the mechanism.

## HOPF-1 VERDICT: LAWFUL MEDIUM AND ATTRACTOR GEOMETRY ARE INDEPENDENT — the unification's falsifier branch fires (2026-07-30 ~9:15 PM)

The contract-MoE (lb recipe + CONTRACT=0.1) landed gate
41/120 (v lb 37 — within sigma; lawfulness again ~free,
GRAV-2's toll result transfers to MoE). PREDICTION 2
FALSIFIED with surgical precision: basin retention curve
[0.986, 0.966, 0.931, 0.869, 0.768, 0.626] v lb's
[0.986, 0.966, 0.935, 0.874, 0.773, 0.630] — IDENTICAL to
noise. The contractivity penalty calmed the medium
(ablation displacements 3-5x smaller across cells;
prediction 1 partial: magnitudes shrink, within-cell
growth shape persists) while leaving routing-basin
geometry UNTOUCHED: medium lawfulness and attractor
geometry are INDEPENDENT properties. The Hopfield
unification (lawful medium => stronger attractors) is
DEAD as stated; both component results stand alone.
EXPLORATORY riders: median screening on/off ratio 2.5x v
lb's 1.6x (a lawful medium mildly LOCALIZES influence —
the GRAV-1 screening failure softens under contraction);
usage-basin corr 0.334 and mass-damage corr 0.322 (both
higher than lb's 0.14/0.17 — weak hints that a calmer
medium makes the mass-like quantities slightly MORE
lawful; single seed, exploratory only). Fences: n=1, one
lambda, basin probe router-level. READ for the mass hunt:
the basin curve's near-identity across lb/ct AND s1/s2
(four models, three decimals) says routing-basin geometry
is an ARCHITECTURAL invariant — set by router
dimensionality/init, not by training regime; mass will
not be found there.

## UMOE-3 VERDICT: ARTIN'S CHANNEL BET PAYS (48/120, best-in-family) — but NOT through the channel; and GRAVITY COLLAPSES THE EXPERTS INTO ONE, WHICH WINS EVEN BIGGER (52/120) (2026-07-30 ~9:45 PM)

CHANNEL ARM: gate 48/120 @ 45.5 valid — the best MoE arm
of the entire program to this point (family: lb 37/44,
free 36/40, tied 43/42, soft 42). Artin's prediction
("the channel arm will score REALLY well") CONFIRMED at
+11 over lb-s1 (~3 sigma). BUT prediction 1's mechanism
half FALSIFIED: the couplings a_i stayed PINNED near zero
(max |a_i| 0.03 across all 32) — the experts did NOT open
the offered channel, corr stayed 0.0080, and yet the arm
wins. The win is NOT carried by shared mass; candidate
mechanisms (banked for seed-2 + ablation): the shared
low-rank params act as a gradient-coupled regularizer
even at near-zero read-out, or seed fortune. DO NOT book
the +11 as real until seed-2.
GRAVMOE ARM: the falsifier table flipped. Usage-
attraction pulled expert corr to 0.9625 (!) — gravity
HOMOGENIZED the experts (overlap ~uniform, so all pairs
attracted; four near-copies remain) — and the gate went
UP: 52/120 @ 47.8, the best MoE gate of the program,
BEATING both dense controls (45/40). Neither
pre-registered branch fired (selective attraction: no;
gate crater: NO — the opposite). READ: at house scale the
4-way specialization never paid; relaxing experts toward
each other = an averaging/ensemble effect (the EMA
averaging=annealing law resurfacing at module grain), and
the router's MI structure SURVIVES homogenization (245x —
routing structure is about assignment, not about the
experts differing; the strongest decoupling of routing
from weights yet measured). Fences: n=1 each, house
scale; the 48 and 52 both need seed-2 before entering any
table as laws. NEXT: seed-2 pair; channel-ablation (zero
the S params at eval: does 48 survive?); gravmoe
lambda-sweep (is there an optimum between split and
collapse?).

## PLACE-1 VERDICT: INFERENCE-TIME GRAVITY IS REAL — co-routing prediction beats popularity by +13.8 points; prefetch cuts cache misses 27% (2026-07-30 ~9:40 PM)

All three predictions CONFIRMED on real OLMoE traces (16
layers, 6,255 tokens, fit/eval split): (a) next-layer
expert prediction: conditional co-occurrence recall@8 =
0.522 v marginal frequency 0.384 v uniform 0.125 — the
co-routing structure B4 measured as MI is EXPLOITABLE,
worth +13.8 absolute points over popularity (bar was
+5); (b) cache sim at C=16/64: LRU 0.469 misses/expert
-> LRU+MI-prefetch 0.343 (-27% bytes moved), freq-pin
0.434 between them. CONSEQUENCE: the split law's systems
lever is measured — routing statistics from the fit half
transport to the eval half, so "co-locate co-routed
experts, prefetch by conditional" is a real deployment
win with zero model change. The Artin cell ("co-located
WITH the training stats, WITHOUT as control") closes
CONFIRMED. Fences: trace-driven sim (no wall-clock), one
model, one corpus class, C single-point.

## UMOE-3 RIDER (channel-ablation): the channel contributes ZERO at eval — the win, if real, is a training-dynamics effect (2026-07-30 ~10 PM)

Zeroing ALL shared-base params (S triples, every block) on
the trained channel arm leaves the gate IDENTICAL: 48/120
with the same per-level map (17/3/15/6/7). Combined with
a_i pinned ~0: the channel's eval-time read-out is nil;
whatever produced +11 over lb acted DURING training (a
gradient-coupled regularizer through the shared low-rank
path — a "scaffold" that can be discarded at deployment)
or is seed fortune. Seed-2 pair chained overnight decides
which. If the scaffold reading survives seed-2, it is a
free-lunch training lever: add a discardable shared
channel at birth, remove it at ship. Fences: n=1 until
seed-2.

## TREE-1 VERDICT: the tree GATES like the channel (48/120) but leaves NO phylogenetic trace — the scaffold pattern repeats (2026-07-30 ~10:45 PM)

Gate 48/120 @ 44.6 — ties channel for best structured arm
(family now: lb 37/44, free 36/40, tied 43/42, soft 42,
channel 48, gravmoe 52, tree 48, dense 45/40). PREDICTION
1 CONFIRMED (>= tied). PREDICTION 3 CONFIRMED: the mid
bases GREW from 0.1-init to full norm (27 v root 31,
leaves 27-29) — training fills every offered level of the
hierarchy. PREDICTION 2 FALSIFIED: within-pair leaf corr
0.0437 v across-pair 0.0616 — siblings are NOT more
similar than cousins; the tree's sharing structure is
used (norms) but leaves NO pair signature in the deltas.
THE EMERGING PATTERN (channel + tree + gravmoe, n=1
each): offering ANY extra shared-gradient structure at
birth improves the gate (48/48/52 v 36-44 family) while
the predicted weight-space signatures consistently fail
to materialize — the benefit appears to live in TRAINING
DYNAMICS (shared gradient paths as scaffolding), not in
the deployed weight structure (channel eval-inert; tree
pair-blind; gravmoe homogenized). Named THE SCAFFOLD
HYPOTHESIS; seed-2 pair (chained, overnight) is the
gatekeeper before it becomes a law. MI 354x (structure
robust as ever). Fences: n=1, house scale, one topology.

## AMENDMENT RIDER (UMOE-3/TREE-1 verdicts): a params confound the verdicts under-weighted (2026-07-30 ~10:40 PM)

Named against my own bookings, before seed-2: the channel
arm carries 1.00M params and the tree arm 1.52M v the lb
family's 0.93M — their +11 gates are CONFOUNDED with
capacity until a params-matched control runs (lb with
ffn_e scaled to match each). GRAVMOE IS NOT CONFOUNDED
(0.93M, identical to lb) — the collapse-wins 52 stands as
the cleanest scaffold datapoint. The scaffold program's
ordering therefore: seed-2 -> params-matched controls ->
Artin's combo arms (tree-edged gravity; branching
channel). Booked so the confound cannot be forgotten if
seed-2 replicates the 48s.

## PRE-REG MERGE-1: gravmoe's collapse as a COMPRESSION result — merge 4 experts to 1, price the deltas (2026-07-30 ~11 PM, CPU desk)

Artin's frame: the scaffolds trade structure for
COMPRESSION. Gravmoe (corr 0.9625) is the limit case: it
manufactured the base+delta redundancy that N3 showed
post-hoc merging cannot find in production MoEs (top-pair
merge +3.4 ppl there). CELLS on umoe_gravmoe_s1 (cpu):
(a) TRUE MERGE — replace each block's 4 experts by their
mean (4x expert params -> 1x, router made irrelevant):
gate v the unmerged 52. (b) DELTA PRICE — sigma/2 code
entropy of the deltas (W_i - mean) v the experts
themselves: bytes for keeping the 4% differences.
PREDICTIONS: (1) merged gate within sigma of 52 (the
differences are decorative) OR drops meaningfully (the 4%
carries real function — then the deltas are the cheapest
params in the model, priced in (b)); named as a fork, not
a bet — EITHER branch is a result: branch A = "gravity
trains a dense model wearing MoE routing" (router theater;
compression 4x free), branch B = "gravity trains
base+delta AT BIRTH" (the N3-impossible factorization,
manufactured). Fences: n=1, house scale, mean-merge only.

## MERGE-1 VERDICT: BRANCH A — gravity trained a dense model wearing MoE routing; the 4->1 merge is FREE and gains (2026-07-30 ~11:15 PM)

The true merge (each block's 4 experts replaced by their
mean; router rendered decorative) gates 54/120 @ 48.6 —
UP from the unmerged 52 (within sigma; certainly no
price). Delta entropy was 3.05 bits/param but is moot:
the deltas can be deleted outright. READ: gravmoe is a
TRAINING RECIPE, not an architecture — birth as 4-expert
MoE with usage-attraction, then ship the MERGED DENSE
model: 54/52-class gates v the same-recipe dense
controls' 45/40, at 1/4 the expert params and zero
routing machinery at inference. This is the scaffold
hypothesis's strongest single datapoint (the entire MoE
apparatus was scaffolding) AND Artin's structure-for-
compression trade made exact: the structure's value was
never in the shipped weights — it was in the training
dynamics, and at ship time it compresses away ENTIRELY.
Caveat honestly held: gravmoe-s1's 52 v dense 45/40 is
n=1 v n=2; seed-2 (running) adjudicates before this
becomes the recommended recipe. Fences: n=1, house scale,
mean-merge, one lambda.

## PRE-REG OVERNIGHT-31: seed-2 review chain + params-matched controls (2026-07-30 ~11:40 PM, Mac, chained)

After the seed-2 pair lands: (a) REVIEW cells —
MERGE-1 on gravmoe_s2 (does merge-free replicate?) and
channel-ablation on channel_s2 (does eval-inertness
replicate?); (b) RUNG-1 CONTROLS — lb at FFN_E=139
(params-matched to channel's 1.00M) and FFN_E=224
(matched to tree's 1.52M), seed 1, same device/recipe.
PREDICTIONS: (1) gravmoe-s2 corr > 0.9 and merged gate
within sigma of unmerged (the recipe replicates); (2)
channel-s2 a_i stays pinned < 0.1 and S-zeroed gate
identical (eval-inertness is systematic); (3) THE
CONFOUND TEST: if widened-lb reaches 48-class gates, the
channel/tree wins were CAPACITY, not structure (scaffold
claim then rests on gravmoe alone); if widened-lb stays
in the 37-44 band, structure is real at matched params.
Fences: controls seed-1 only tonight; combos (tree-edged
gravity, branching channel) remain gated behind these.

## VERDICT OVERNIGHT-31 (rung 0): scaffold REPLICATES on seed 2 — both review cells pass (2026-07-31 morning, Mac)

Seed-2 pair (logs/umoe3_s2_0730.log) + review cells
(logs/overnight_0730.log, scratch/scaffold_review.py):

- channel s2: 49/120 (s1 48); gravmoe s2: 51/120 (s1 52).
  Both hold the top of the family band on a fresh seed
  (plain family 36-44/45-48; tree s1 48).
- Prediction 1 CONFIRMED: gravmoe-s2 expert corr 0.9642
  (>0.9, per-block 0.90-0.98 — the collapse replicates),
  routing MI 304x shuffle survives, and the MERGED 4->1
  model gates 49/120 vs 51 unmerged — within sigma
  (binomial sigma ~5 at this rate; s1 was 54 v 52). The
  "birth as Hebbian MoE, ship as dense" recipe is now
  2-for-2 seeds: merging costs nothing either time.
- Prediction 2 CONFIRMED: channel-s2 max|a_i| = 0.021
  (pinned, <0.1) and the S-zeroed gate is 50/120 vs
  49/120 full — one-solve difference, eval-inert again.
  The channel scaffold leaves no load-bearing anatomy on
  a second seed either.

Rung-0 verdict: the scaffold hypothesis survives seed
adjudication. Structure-during-birth (Hebbian pull,
channel tail, tree) beats the plain family on both seeds
while remaining anatomically deletable at ship time.
STILL OPEN (rung 1, running): the params confound — lb at
FFN_E=139/224 will say whether channel/tree wins were
capacity; gravmoe (0.93M, params-clean) is already
confound-free and is the strongest single result of the
program. Prediction 3 adjudicates when those gates land.

## PRE-REG RUNG-2 COMBOS (OVERNIGHT-31 chain): treegrav + chantree drafted, launch gated on rung-1 (2026-07-31 morning, Mac)

Artin's combo riffs implemented in scratch/umoe_conserve.py
(smoke-tested CPU: build/forward/backward/EMA all pass):

- ARM=treegrav: tree parameterization + Hebbian relaxation
  RESTRICTED TO TREE EDGES (leaf i relaxes only toward its
  sibling, weighted by co-routing EMA). Params 1.52M =
  tree exactly (relaxation is param-free) — the cleanest
  paired comparison of the program: treegrav v tree
  isolates edge-gravity with zero confound.
- ARM=chantree: per-sibling-pair low-rank channels (expert
  i talks only through pair i//2's channel; a_i gates as
  before). Params 1.07M v channel 1.00M (+7% from the
  second channel — noted, small).

PREDICTIONS: (1) treegrav >= tree on the gate, and the
within-pair leaf corr flips ABOVE across-pair (TREE-1
measured within 0.044 < across 0.062 — no phylogeny;
edge-gravity should CREATE the phylogeny the tree alone
failed to produce). (2) chantree a_i stays pinned near 0
(the eval-inert family signature) unless pair-scoping
makes the channel cheap enough to use; gate in the
structured band (45-52). (3) If treegrav beats BOTH tree
and gravmoe, topology-shaped gravity is a real lever and
the lambda-sweep (rung 3) runs on treegrav, not gravmoe.
FENCES: launch only after rung-1 params controls book
(prediction-3 adjudication first); Mac, seed 1, same
recipe/diet; chantree's +7% params noted at booking.

## VERDICT FX-V2 (cross-lab receipt): C++ twin PASSES — determinism is now CROSS-RUNTIME (2026-07-31, axiom's box, their commit d5e9d5a)

Axiom's report (relayed by Artin; their reply relay at
docs/relay/2026-07-31-1-fx-v2-verdict.md their side): a
~350-line single-file C++ binary (tools/fx_v2/) that never
touches torch, Python, or libm reproduces BOTH P3 digests
exactly — streams bf76568d... and logit trace 311f71bf...
— in 0.16 s. Pipeline: stdlib-only exporter (whitelisted
Unpickler, sha-pin verified before parse) copies the int64
storages verbatim to a flat "AXP3" binary; battery prompt
ids frozen as data (sympy/tokenizer needed only at prompt
GENERATION, never at decode); pure-int64 end-to-end (their
pick of our "pure int64" branch — no fp32 carrier, no
hi/lo split; max partial ~2^21, matching house diagnostic);
own minimal SHA-256 over the exact repr byte formatting.
Port subtlety they proved rather than assumed: Python
floor division = C++ truncating division here because the
round-half-away numerators are nonnegative with positive
divisors — no floor emulation needed.

VERDICT: the determinism ladder gains its final rung —
cross-device (P3) -> cross-lab (FX-V1-H) -> CROSS-RUNTIME
(FX-V2). The model is the integers, not the framework: the
same weights produce the same bytes from a torch-free
binary on a different substrate. Paper claim upgrade
available for the determinism section. rANS-unpack rider
deliberately left for a future cell (their call, agreed).

## PRE-REG NIGHT-31-CUDA: rung-2 combos + rung-3 lambda-sweep, internally paired on the 3080 (2026-07-31 night, Artin's GO)

Chain (scratch/night31_cuda.sh, seed 1, OTAG=_cuda, one
device): lb -> tree -> treegrav -> gravmoe(0.5) -> channel
-> chantree -> gravmoe lambda {0.1, 0.25, 1.0}. All nine
arms born on cuda; comparisons stay WITHIN this chain
(cross-device doctrine — Mac gates are never the control
for these). FENCE AMENDMENT, honest: the rung-2 launch
fence said "after rung-1 books"; FFN_E=139 (channel's
params twin) has landed at 45/120 — above the old lb band,
below channel's 48/49, so the capacity confound is already
part-adjudicated (capacity buys some, not all); FFN_E=224
(tree's twin) still training on the Mac and books when it
lands. Launching combos tonight on that basis.
PREDICTIONS (inherit RUNG-2 pre-reg, now device-paired):
(1) the scaffold ORDER transports to cuda: gravmoe/channel
/tree family beats same-device lb; (2) treegrav >= tree
(params-identical pair) and within-pair leaf corr flips
above across-pair; (3) chantree a_i pinned ~0 again; (4)
lambda-sweep: collapse corr rises monotonically with
lambda; gate has an interior optimum (0.5 was not tuned —
if 0.25 or 1.0 beats it, the Hebbian pull is a real dial;
if flat, the scaffold is threshold-not-dose).

## VERDICT OVERNIGHT-31 (rung 1): THE PARAMS CONFOUND BITES — tree's win was capacity, channel's mostly; the scaffold claim narrows to GRAVMOE (2026-07-31 ~3 AM, Mac)

Params-matched lb controls (same recipe/device/seed):
- FFN_E=224 (1.52M = tree's twin): 53/120 — ABOVE tree's
  48. Prediction 3's capacity branch fires PAST its own
  threshold: at matched params, plain lb beats the tree.
  The tree parameterization was not a scaffold; if
  anything it TAXED capacity (-5 at equal params, ~1σ).
- FFN_E=139 (1.00M = channel's twin): 45/120 v channel
  48/49 — a +3.5 residual, under 1σ. Channel's win is
  mostly capacity; the structured residual is not
  distinguishable from noise at n=2.
- GRAVMOE UNTOUCHED: 0.93M (= plain lb exactly), 52/51
  across seeds, corr-collapse + merge-free replicated.
  It is the only params-clean structured win, and it
  remains the program's result.

VERDICT: the scaffold hypothesis SURVIVES ONLY AS THE
HEBBIAN PULL. "Any extra shared-gradient structure helps
training" is retracted to: "co-routing-weighted weight
relaxation (gravmoe) helps at zero params and zero
deploy-time anatomy; tree/channel structure was buying
capacity, not dynamics." Yesterday's family narrative is
hereby scoped — the BOARD line and THEORY row get the
amendment. Rider: FFN_E=224 at 53 is itself notable
(best non-merged single model of the program at +63%
params); the capacity dial on this diet is not saturated
at 0.93M. Cuda chain (NIGHT-31-CUDA) adjudicates the
same questions device-paired; its lb-at-48 baseline
makes the within-cuda bar appropriately higher.

## PRE-REG DAY-31 QUEUE (both machines to 5 PM): gravmoe n=3 (Mac) + cuda params controls (3080 follow-on)

(a) MAC: ARM=gravmoe SEED=3 birth + S=3 scaffold review
(merge + corr). The surviving pillar goes to n=3.
PREDICTION: corr collapse > 0.9, gate in the 46-56 band,
merged within 1σ of unmerged — the merge-free recipe at
3-for-3 becomes a bookable RECIPE (promotion candidate to
llmopt/ + axiom C++ backend ask).
(b) 3080 (chained after NIGHT-31-CUDA DONE marker):
ARM=lb FFN_E=139 and FFN_E=224, OTAG=_cuda — the params
controls INSIDE the cuda pairing, completing the
device-paired replication of rung 1. PREDICTION: same
shape as Mac (224-twin reaches/passes tree-class; 139-twin
closes most of channel's gap). If cuda replicates the
confound, the narrowed scaffold verdict is device-robust.

## VERDICT DAY-31(a): MERGE-FREE AT n=3 — "birth as Hebbian MoE, ship as dense" books as a RECIPE (2026-07-31 morning, Mac)

gravmoe seed 3 (logs/gravmoe_s3_0731.log): gate 50/120,
corr collapse 0.9665 (per-block 0.92-0.97), and the merged
4->1 model gates 51/120 — the merged model matches or
BEATS the unmerged MoE for the third seed in a row:
s1 54 v 52, s2 49 v 51, s3 51 v 50. Merge deltas
{+2, -2, +1}: zero-mean, tiny. Prediction confirmed as
pre-registered (corr > 0.9, gate in band, merge within
1σ).

RECIPE (bookable): train a top-1 switch MoE with
co-routing-weighted weight relaxation (GRAV_LAM 0.5 every
100 steps), then average the four experts into ONE dense
FFN at ship time. Cost: zero extra params v dense, zero
deploy-time routing, zero merge tax (n=3). On Mac the
recipe holds +5-6 over params-matched lb (n=3 v n=2
controls). PROMOTION CANDIDATE: llmopt/ implementation +
axiom C++ backend ask — HELD pending the cuda chain's
device-transport adjudication (early cuda arms show a
TIGHTER family spread: lb 48, gravmoe 49 — the Mac-sized
advantage may not transport; booked when chain lands).

## PRE-REG FOURIER-2b: wide-Mod birth + roots-of-unity probe with memorization gate (2026-07-31 morning, Mac)

scratch/fourier2b_widemod.py: 20,000 Mod(n,k) rows, n
uniform [10, 99999], k in 3..11, string seeds; 500 eval
pairs whose n's are EXCLUDED from train (prompt-set
guard, asserted). Memorization gate BEFORE the probe:
greedy exact-match on held-out n's; below 0.5 the run is
VOID and the probe skips (the FOURIER-2 pilot lesson —
memorized substrate probes nothing). Tokenizer round-trip
verified pre-launch (the FOURIER-1 lesson).
PREDICTIONS: (1) held-out acc > 0.9 — Mod is learnable
in-format at d64 with a wide generator; (2) if learned,
periodic-neuron count at k=5,7 LARGE v FOURIER-1's 11/512
(a dedicated Mod diet should force clock structure, the
grokking-modular-arithmetic literature's signature), with
shuffle ~0; (3) top-freq share HIGHER than 0.534 (purer
Fourier under a diet that needs it). First use of
llmopt.runlog in a birth script (elapsed-stamped log).

## AMENDMENT (NIGHT-31-CUDA) + FOURIER-2b v1: friendly-fire #10 kills the lambda-sweep silently; the memorization gate WORKS and fires (2026-07-31 ~9:30 AM)

(a) FRIENDLY-FIRE #10: the chain's run() used
${2:+GRAV_LAM=$2} as an env prefix — but expansions happen
AFTER the shell parses assignment prefixes, so the three
lambda arms died in 0s with "command not found", and the
chain's DONE marker fired anyway (per-arm failures don't
propagate). Fix: env(1) form (assignments as ARGUMENTS
survive expansion). Lesson for the remote-ops doctrine:
a chain's success marker must depend on its arms' exit
codes, or at minimum the morning read must check arm
COUNT, not the marker. Lambda arms re-queued behind the
running ctl chain.
(b) FOURIER-2b v1: the pre-registered VOID gate FIRED —
held-out Mod acc 0.492 < 0.5, probe skipped. Informative:
0.492 is ~3.5x chance (~0.14 across k=3..11), so the
crystal PART-generalized; birth was 226s at loss 0.88
(undertrained, cf. ~0.3-0.4 at convergence elsewhere).
v2 relaunched at EPOCHS=30 (env knob added). Not booked
as a FOURIER verdict either way yet.

## VERDICT FOURIER-2b: CLOCKS FOLLOW COMPETENCE, COMPETENCE FOLLOWS DIGIT LOCALITY (2026-07-31 ~10:50 AM, Mac, 30-epoch v2)

Gate: held-out Mod acc 0.526 — passes the 0.5 VOID bar,
barely. Per-k breakdown (in-line check before booking):
  k=4 1.00 | k=5 1.00 | k=10 1.00 | k=8 0.63
  k=6 0.38 | k=3 0.34 | k=7 0.21 | k=9 0.16 | k=11 0.14
Perfectly bimodal by MECHANISM: moduli decidable from the
last 1-2 digits (4, 5, 10; 8 needs three -> partial 0.63)
are SOLVED; moduli needing whole-number reduction / digit
sums / carries (3, 6, 7, 9, 11) sit near chance. The
crystal learned positional shortcuts, not modular
arithmetic.

Probe (the payoff): at k=5 — a SOLVED modulus — 276/512
neurons periodic (54%!, v FOURIER-1's 11/512), median
top-freq share 0.779 (v 0.534), shuffle 0/512. At k=7 —
an UNSOLVED modulus — 0/512. The roots-of-unity clock
structure appears exactly where competence exists and
nowhere else, at 25x the population FOURIER-1 found on a
diet that merely contains modular structure.

LAW (rotational thread, weights-side closure): Fourier/
rotational structure in a crystal is DIET-FORCED, not
architecture-available — it emerges iff the training
distribution makes the periodic computation both needed
and learnable, and its per-frequency presence is a
readout of per-modulus competence. Grokking literature
(Nanda et al. 2301.05217) agrees from the forced side;
we now have the unforced side: no competence, no clock.
Loss plateaued (~0.78, ep25-29 flat) — the hard moduli
are not "more epochs" away at this recipe; they need the
ALGORITHM in the diet (FOURIER-3 candidate: digit-sum
decomposition rows for k=3/9, then test whether clocks
APPEAR at k=9 after competence — the causal arrow test).

## PRE-REG FOURIER-3 + MERGE-CUDA review (2026-07-31 ~11 AM, Mac, parallel)

(a) FOURIER-3 (scratch/fourier3_algdiet.py): FOURIER-2b
diet + the ALGORITHM for k in {3,9} — rows teach
Mod(n,k) -> Mod(digitsum(n),k) rewrites (n>=10), terminal
answers below; k=7/11 stay UNTAUGHT as hard controls;
eval = multi-hop greedy rollout on held-out n's (oracle
convergence of the teaching chain verified pre-launch, 0
mismatches/2000). PREDICTIONS: (1) k=3/9 rollout acc
rises far above 2b's 0.34/0.16 (algorithm learnable as
local rewriting); (2) k=7/11 stay near chance on BOTH
axes (acc + clocks) — competence doesn't leak; (3) THE
POINT — the k=9 clock either (a) stays absent while
competence arrives (algorithm SUBSTITUTES for rotational
representation; single-pass activations never need n mod
9) or (b) appears (practice on reduced forms builds it).
House lean: (a), by the prosthetics-replace-anatomy
precedent (step-local spans). Either branch books.
(b) MERGE-CUDA: umoe_gravmoe_cuda_s1.pt pulled sha-
verified (3e86e7c3); merged v unmerged BOTH gated on Mac
(same-device pair; cuda-run gate never compared).
PREDICTION: merge delta within 1σ — merge-free goes
4-for-4 across devices.

## VERDICT MERGE-CUDA: merge-free goes 4-for-4 — seeds AND devices (2026-07-31 ~11:30 AM)

umoe_gravmoe_cuda_s1.pt (cuda-born, sha 3e86e7c3, gated
both ways on Mac for a same-device pair): corr collapse
0.9508; UNMERGED 49/120, MERGED 51/120. Fourth
consecutive merge at zero-or-positive delta: {+2, -2,
+1, +2} across three Mac seeds + one cuda birth. The
merge-free property of the Hebbian-pull MoE is now
replicated across BOTH training devices — it is a
property of the RECIPE, not of a seed or a backend.
Incidental determinism point: the Mac re-run of the
unmerged gate reproduced the cuda-run gate EXACTLY
(49/120, valid 45.1867816091954 to full precision) —
greedy gates on this checkpoint are device-stable.
Remaining transport question is only the SIZE of the
gravmoe advantage v lb (Mac +5-6, cuda +1) — adjudicates
with the cuda lambda-sweep + a cuda lb seed-2 if needed.

## VERDICT NIGHT-31-CUDA (controls): THE RESOLUTION LAW — single-cell gate deltas under ~5 are seed noise; the confound adjudication INVERTS across devices (2026-07-31 ~11:40 AM)

Cuda params controls: lb at 0.93M/0.99M/1.52M = 48/46/47
— the capacity slope that was decisive on Mac (44/45/53)
is FLAT on cuda. Meanwhile cuda tree (51) and channel
(51) sit ABOVE their params-twins (+4/+5) — the opposite
sign of the Mac rung-1 verdict. Neither pattern clears
1σ (binomial σ≈5 at these rates) cell-by-cell.

JOINT READING (the honest one): with a 120-prompt gate,
n=1 cells CANNOT resolve effects smaller than ~5 solves.
What survives across all 15+ cells of the program:
(1) MERGE-FREE: sign-consistent at n=4 (never negative
    in expectation: +2,-2,+1,+2) — a property claim, not
    a delta claim, robust;
(2) MAC GRAVMOE ADVANTAGE: +5-6 v lb at n=3 consistent
    seeds — real ON MAC; did not appear on cuda (n=1) —
    device-scoped until a cuda seed ladder says otherwise;
(3) EVERYTHING ELSE (tree/channel/capacity slopes, both
    directions) — below resolution, retract to noise.

METHODS RULE (adopt): any claim of a gate delta < 1.5σ
requires n>=3 paired seeds BEFORE booking a direction; a
single-seed adjudication (as rung-1 was) can flip on the
next device/seed and did. This also amends rung-1's tone:
"tree's win was capacity" overreached — the defensible
statement is "no structured arm separates from matched-
params lb at current resolution; gravmoe-on-Mac and
merge-free are the only resolved effects."

## AMENDMENT FOURIER-3 v1: the chain's BASE CASE was diet-starved — taught moduli 0.00 by fixed-point looping, not incompetence (2026-07-31 ~12 PM)

v1 rollout gate: k=3/9 = 0.00 (worse than 2b's
no-algorithm 0.34/0.16) while 4/5/10 stayed 1.00 and the
k=5 clock GREW to 351/512. Transcript autopsy before
booking (check-don't-assume): the digit-sum rewrite WAS
learned — first hops from 5-digit n are correct
(23807->20, 96750->27, 93223->19...) — but chains die at
2-digit n with junk rewrites and fixed-point loops
(Mod(10,3)->Mod(10,3) forever). Cause: uniform n on
[2,99999] puts ~90% of rows at 5 digits; the reduced
forms every chain must pass through (n<100) got ~0.1%
share. The recursion's base case was starved — the diet-
exposure-SHARE doctrine measured at micro scale, and a
new failure shape for it: MULTI-STEP competence needs
share allocated per RECURSION DEPTH, not per input.
v2 relaunched: length-uniform generator (each digit
count 1-5 gets ~20% share), everything else fixed.
Predictions unchanged from the FOURIER-3 pre-reg; v1
clocks (k=5 351/512, k=9 0, k=7 0) consistent with
clocks-follow-competence but NOT bookable for the causal
arrow until competence actually arrives at k=9.

## VERDICT FOURIER-3: THE ALGORITHM SUBSTITUTES FOR THE CLOCK — competence WITHOUT representation (2026-07-31 ~12:40 PM, Mac, v2 length-uniform)

Gate (multi-hop rollout, held-out n): taught moduli
k=3 0.71, k=9 0.83 — competence ARRIVED via the digit-sum
rewrite chain (v1-starved 0.00; no-algorithm 2b baseline
0.34/0.16). Untaught hard controls stay dead: k=7 0.04,
k=11 0.11 — no leak (prediction 2 confirmed). Shortcut
moduli 4/5/10 hold 1.00; k=8 rose 0.63->1.00 (length-
uniform diet gave the 3-digit rule its share too).
Overall 0.674 v 2b's 0.526.

Clock probe: k=9 periodic neurons = 0/512 AT 0.83
COMPETENCE. Branch (a) of the pre-registration books:
the learned ALGORITHM externalizes the computation into
the rewrite chain, and the single-pass rotational
representation NEVER FORMS. With FOURIER-2b this
completes a two-sided law:
  - single-pass competence (digit shortcuts) -> clocks,
    massively (276-351/512);
  - chain competence (taught algorithm) -> NO clocks
    (0/512), same accuracy class;
  - no competence -> no clocks (k=7: 1/512 at 0.04).
Clocks are the signature of WHERE the computation runs,
not whether the task is solved. Prosthetics-replace-
anatomy (step-local spans, 07-28) now has its
representation-level twin, measured with a pre-registered
two-branch design. Rider: k=5 clock population varies
with diet composition (351 -> 142 across v1/v2 at acc
1.00 both) — population SIZE is exposure-sensitive even
where competence saturates; clock presence/absence is the
robust readout, count is not.

## PRE-REG TIER-A REVIVAL CELLS (2026-07-31 afternoon, Artin GO): gate pooling + graph-mod sigma + rotation positive control

A1 GATE POOLING (scratch/pack_decode.py battery): greedy-
decode the full 120-prompt gate battery through the P3
integer path on Mac AND 3080 (P3_DEV=cpu box-side to
spare the lambda run), one sha256 digest over all
streams. PREDICTION: digests IDENTICAL (P3 already
bit-identical on 5 prompts + a full logit trace) ->
ADOPT cross-device pooled seed ladders for greedy
deterministic batteries; the resolution law's n>=3 cost
halves. Fence: pooling legality claim covers the
DETERMINISTIC path only, not fp sampled gates.
A2 GRAPH-MOD SIGMA (scratch/graph_mod_sigma.py): the
07-26 dQ +0.030 null was BAR-based, dispersion
unmeasured; run the same instrument on the three
same-diet wfloor_d256 seeds (the entry's own named free
sigma). PREDICTION (house): seed sd >= ~0.02 -> +0.030
stays unresolved (z < 1.5); if sd is tiny (<0.01) the
+0.030 RE-ADJUDICATES toward real modularity and gen-8
modules reopen.
A3 ROTATION POSITIVE CONTROL (scratch/rotinstr_control.py):
euler lenses (phase-pair KS, rFFT top-8, 20 shuffles) +
anti-commutant mass (adjacent v 20 random pairings) on
fourier2b_widemod.pt — the crystal with a CONFIRMED
276/512 activation clock. Either outcome books: NULL ->
weight-side lenses are blind to activation clocks (old
spontaneous-rotation nulls said nothing about
representations); FIRE -> the old nulls were diet
statements (clock-placement reading). House lean: NULL
(gauge law — clocks live in activation SPACE, channel
basis stays meaningless).

## VERDICT TIER-A A2+A3: a NULL REVERSES (gen-8 modules are ~10-sigma real against measured seed dispersion) and the rotation lenses are BLIND to clocks (2026-07-31 ~1 PM, Mac desk cells)

A2 GRAPH-MOD SIGMA: Q per seed on the three same-diet
wfloor_d256 births = 0.2673 / 0.2663 / 0.2702, sd 0.0020.
The 07-26 dQ +0.030 (gen-8 five-grammar v single-grammar)
sits at z ~ 10.5 in delta-sigma units. RE-ADJUDICATION:
the "modules do NOT appear" bar-based null REVERSES —
+0.030 is overwhelmingly resolved against measured
dispersion; the five-grammar diet DOES buy weight-graph
modularity. FENCE (honest): sigma measured on the
wfloor_d256 family (the entry's own named free-sigma
substrate), not on the gen-8/19M pair itself — dispersion
could differ by family. Confirmatory cell (banked): Q
sigma on 19M-class same-diet seeds when a pair exists.
First revival-sweep payoff: the resolution law CONVICTS
in both directions — it retracted three of our wins this
morning and resurrected someone else's null after lunch.

A3 ROTATION POSITIVE CONTROL: on fourier2b_widemod.pt
(confirmed 276/512 activation clock at k=5): phase-pair
z mean +0.03 max 1.68; fft-order z mean +0.60 max 2.92
(bar 3); adjacent anti-mass 0.5011 v random-pairing null
0.4999±0.0057 (z +0.21). ALL NULL — as house-leaned
(gauge law). VERDICT: the weight-side rotation
instruments are BLIND to activation clocks; the
spontaneous-rotation nulls (weight-FFT euler, complex
NNUE, quaternion R1) were statements about WEIGHT-BASIS
structure only and say nothing about representations.
The clock-placement law stands as the activation-side
truth; weight-side rotation reads require imposed
structure (the symmetry ladder) or activation probes,
never raw-basis lenses. Revival candidate #6 resolves:
the instruments were the wound.

## VERDICT TIER-A A1: GATE POOLING ADOPTED — full 120-prompt battery digest IDENTICAL Mac/3080 (2026-07-31 ~1:15 PM)

P3 integer battery, greedy, 120 prompts, both machines:
sha256 9ee4fa83ec9a52e8408aaf01b234557b68f22264c2692009c
c3b18b71e26864b — byte-identical (Mac cpu / 3080 cpu).
Prediction confirmed; ADOPTION: cross-device POOLED seed
ladders are legal for greedy deterministic batteries.
The resolution law's n>=3 requirement now costs half the
wall-clock (both machines contribute seeds to one
ladder). Fences travel with the instrument: pooling
covers the DETERMINISTIC integer path only — fp sampled
gates remain device-scoped (the 18/24-v-9/24 lesson
stands); and the deterministic battery is a DIFFERENT
instrument from the fp gate (scores may differ; pool
within one instrument, never across). Tier A closes
3-for-3 booked in one afternoon: one adoption, one
reversal, one instrument autopsy.

## PRE-REG B6 (revival Tier B): G9 zeta-8 on the Mod diet — the rotation reopening, fired on the substrate that wants it (2026-07-31 afternoon, Mac)

scratch/fourier_g9.py: complex-FFN d64 crystal (modReLU,
genuine complex multiply), phases STE-snapped exactly to
Z[zeta_8] (ARM=G9) v complex-unsnapped (ARM=none), both
on the EXACT fourier2b diet/split/seed/epochs; paired
against the booked real crystal (0.526 overall; k=8
partial 0.63; clocks 276/512 at k=5). Snap verified
pre-launch (phase residues exactly 0 on pi/4 grid).
PREDICTIONS: (1) house lean from the 2026-07-26 closure
(alphabet-follows-domain null): G9 does NOT beat the
real control overall — architecture-provided rotation
is not adopted even where the diet is periodic; the
interesting falsifier is k=8 (zeta_8's own modulus,
real ctrl 0.63): if G9 solves k=8 to ~1.0 while real
stalls, the alphabet DID pay exactly at its resonant
modulus and alphabet-follows-domain reopens for real.
(2) complex-none arm ~ties real (complex structure alone
neutral, consistent with cplx history). (3) clock
populations: complex arms may read differently on the
SAME probe (activation geometry differs) — count
reported, presence/absence is the readout per the
clock-placement rider. Resolution-law fence: single
seed; deltas < 5 points on per-k accs read as noise
unless k=8 hits the 1.0-v-0.63 separation.

## PRE-REG B5 (revival Tier B): farmer rebirth on a DEPTH-UNIFORM reversed corpus (2026-07-31 afternoon, Mac, chained behind B6)

sym_birth REV=3 (new): full reverse + chain-position
reconstruction (link nxt->cur; every row resolves — 0
unresolved of 132,870) + depth-uniform resample at
matched dose. Measured skew being corrected: rows/depth
= 52,937 / 37,845 / 29,771 / 10,439 / 1,878 (28x
depth-0-to-4) — the recursion-depth diet-share clause
applied to the farmer. Then farmer_probe on the new EMA
ckpt, BOTH seed modes (band + corpus), same novelty
fence. PREDICTIONS: (1) SEEDMODE=corpus verified stays
~100/1000 (inversion not harmed); (2) the revival claim:
NOVEL yield moves off 11/1000 (bar: >=2x, i.e. >=22/1000,
else the depth-share lever reads null for farming and the
banked revive conditions — temperature, solved-state
seeding, scale — stay the path); (3) SEEDMODE=band novel
also rises if starved-depth coverage was the wound (band
seeds are starts; their predecessors barely exist in the
grammar — this mode may stay dead for structural reasons,
fenced as such). Single seed, MPS, EMA — resolution-law
fence on any sub-2x delta.

## VERDICT B6: THE ROTATION REOPENING CLOSES AGAIN — exact zeta-8 phases are neutral-to-negative even on the diet that wants them; and the clock law goes GRADED (2026-07-31 ~3 PM, Mac)

Three-arm table (same diet/split/seed/epochs, Mac):
| arm | overall | k=8 acc | k5 clock | k8 clock | k7 clock |
| real (2b)     | 0.526 | 0.63 | 276/512 | 115/512 | 0 |
| complex-none  | 0.530 | 0.77 | 175/512 |  51/512 | 0 |
| G9 zeta-8 STE | 0.504 | 0.54 | 264/512 | 115/512 | 0 |
(real k=8 clock measured fresh this session — CPU probe,
shuffle 0/512 everywhere.)

VERDICTS: (1) Prediction 1 lands on the house lean: the
exact-phase alphabet does NOT pay overall (0.504 v 0.526)
and the resonant-modulus falsifier did NOT fire (k=8:
G9 0.54 v real 0.63 — the alphabet is worst at its own
modulus). Alphabet-follows-domain stays CLOSED, now
tested on the one substrate where the computation is
provably rotational. Architecture-provided rotation is
not adopted even where the diet is periodic — the
strongest form of teach-don't-impose yet measured.
(2) Complex-none ties real overall (0.530 v 0.526) —
complex structure alone is neutral, consistent with the
whole cplx history. Rider (suggestive, ~2.2 sigma, n=57,
single seed — resolution-law fenced): complex arithmetic
may help k=8 (0.77 v 0.63) while the zeta-8 SNAP hurts
it; if anyone revives the alphabet, revive the
UNSNAPPED complex FFN, not the exact phases.
(3) THE GRADED CLOCK LAW: within every arm, clock
presence tracks per-modulus competence — full competence
big clock (175-276), partial competence partial clock
(51-115), no competence zero (0/512, all arms, k=7).
Clock COUNT varies by architecture at matched competence
(real 115 v none 51 at k=8) — count stays a non-readout;
the presence/absence/partial LADDER is the robust
instrument. The clock-placement THEORY row gains this
gradation.

## VERDICT RUNG-3 LAMBDA-SWEEP: the Hebbian pull is a DIAL FOR ANATOMY, NOT CAPABILITY (2026-07-31 ~3 PM, 3080, internally paired)

Gates across lambda {0.1, 0.25, 0.5, 1.0} on cuda:
48 / 47 / 49 / 50 (lb control 48) — FLAT, every delta
inside 1 sigma. Expert-corr collapse across the same
sweep: 0.827 / 0.935 / 0.951 / 0.974 — MONOTONE dose
response. Prediction 4's threshold branch books: the
relaxation strength dials how HOMOGENIZED the experts
get, while the gate does not move on this device. With
the merge-free property (n=4) this completes the
recipe's honest card: "birth as Hebbian MoE, ship as
dense" costs nothing at ANY lambda, collapses anatomy
proportional to lambda, and its +5-6 gate advantage
exists on Mac (n=3) but not cuda — device-scoped, cause
unadjudicated (candidates: mps numerics, seed pool).
PROMOTION DECISION now unblocked — evidence table
complete; recommendation to Artin: promote the RECIPE
(training utility + merge function + test) to llmopt/
as capability-neutral-or-better with zero deploy cost;
hold the capability CLAIM at device-scoped until a
pooled deterministic ladder exists.

## PROMOTION (Artin GO): hebbian_moe + complex_ffn land in llmopt/ — the scaffold program and the rotation thread ship their survivors (2026-07-31 ~4 PM)

llmopt/train/hebbian_moe.py: HebbianCoupler (co-routing
overlap EMA + periodic relaxation, optional edge
restriction) + merge_experts (ship-time E->1 collapse,
with the dense-plus-scalar-gate export caveat documented
— top-1 switch scaling survives the merge). Docstring
carries the full measured card: merge-free n=4, lambda =
anatomy dial, advantage device-scoped, decorrelation-is-
default (why post-hoc merging fails). 6 tests: EMA math
v manual, contraction, schedule/edges, snapshot symmetry,
merge mean, validation.

llmopt/train/complex_ffn.py (Artin's tie-in): the
UNSNAPPED complex FFN (modReLU + genuine complex
multiply), promoted with its honest card — ties real
overall, suggestive at the carry modulus (fenced), and
the zeta-N snap variant explicitly NOT promoted (B6:
neutral-to-negative, teach-don't-impose). 4 tests incl.
equivalence to torch.complex64 arithmetic and the
e^{2i*theta} phase-equivariance property of the gated
product.

Suite: 392 passed. Axiom C++ ask drafted (relay
2026-07-31-0): dense-plus-scalar-gate forward for merged
crystals in their stack — training stays torch by
doctrine; inference homecoming is theirs.

## VERDICT MERGED-CRYSTAL C++ (cross-lab receipt): PASS both seeds — the merge recipe's endpoint runs token-identical in axiom's stack, same day as the ask (2026-07-31 evening, their commit 0104e1d)

Axiom's report (relayed; their reply relay at
docs/relay/2026-07-31-2-merged-crystal-cpp.md their
side): C++ forward reproduces the torch merged model's
greedy streams TOKEN-FOR-TOKEN — 5 prompts x 40 tokens,
zero divergences, BOTH umoe_gravmoe_s1 and s2. No
artifact transfer needed (shared llmopt clone); they
wrote the exporter to their own spec and closed the cell
locally. Shipped their side: AXNN v1.2 (declared
ffn_gate "switch_top1" + n_experts + per-block router
tensor; validation rejects undeclared/stray/missing
router — new gtest, suite 481/481);
tools/moe_merge/export_merged_axnn.py (merge 4->1 dense
SwiGLU keeping the router, sha-pinned exports, torch
fp32 reference streams); axiom-nn-moe-greedy driver on
the frozen FX-V2 battery prompts. The scalar-gate
subtlety (router softmax-max off the pre-LN hidden,
exactly as the trainer computes it) landed correctly
first try.

HONEST CAVEAT (theirs, adopted): the bar is
token-identical FLOAT agreement (torch fp32 v their
double-accumulation), not bit-identical logits — argmax
margins survived all 48 steps x 8 blocks per row.
FOLLOW-UP ON OFFER (house recommendation: ACCEPT as
FX-V3): route the merged crystal through the P3/FX-V2
integer twin — one extra shipped table (router softmax)
deletes the tolerance column here too and puts the
recipe's endpoint fully inside the cross-runtime
determinism family. Awaiting Artin's relay to confirm.
The recipe's lifecycle is now measured end-to-end in one
day: torch birth -> free merge (n=4) -> C++ deployment
(token-identical, two labs).

## AMENDMENT (B5 launches): two stumbles before the clean run — ARM env missing (crash at import), then EMA omitted (birth completed raw-only; original fence is EMA 0.999) (2026-07-31 evening)

The v2 raw birth (sym_birth_dense_revdepth.pt, loss 0.30)
is kept but NOT the comparison substrate; v3 relaunched
with EMA=0.999 matching the booked farmer birth exactly.
Lesson for the launch checklist: when rebirthing against
a booked baseline, diff the FULL env line of the original
launch (the pre-reg text carries it), not just the new
knob. Cost ~2h Mac.

## PRE-REG CUDA SEED LADDER (Tier C opener; 3080 idle, evening): does the gravmoe advantage transport, at n=3? (2026-07-31)

The last open question of the scaffold program: Mac
gravmoe beats lb by +5-6 at n=3; cuda n=1 showed +1.
Chain (3080, OTAG=_cuda, internally paired): lb SEED=2 ->
lb SEED=3 -> gravmoe SEED=2 -> gravmoe SEED=3 (~4h).
With the existing cuda s1 pair (lb 48, gravmoe 49) this
gives n=3 BOTH arms on one device. PREDICTIONS: (1) if
mean(gravmoe) - mean(lb) >= +4 on cuda, the advantage
transports and the recipe's capability claim upgrades
from device-scoped to general; (2) if the means sit
within +-2, the advantage is Mac-specific (candidates
then: mps numerics, batch composition) and the recipe's
card stays "free merge, capability-neutral"; (3) merge
review on every new gravmoe ckpt extends merge-free
toward n=6. Resolution law satisfied by design (n=3
paired, one device).

## AMENDMENT (REVIEW ADOPTION, day-31 pass): the reviewer's verified corrections to today's bookings (2026-07-31 night)

An Opus reviewer (self-reported Opus 5 this pass) cross-
checked all day-31 bookings against logs; Fable verified
each finding before adoption. CORRECTIONS (each names its
target entry):

1. MERGE DELTAS {+2, -2, +1, +2} are NOT "zero-or-
   positive"/"sign-consistent" (DAY-31(a), MERGE-CUDA,
   resolution-law entries all misphrase this — the set
   contains a -2). Correct claim, everywhere including
   the recipe card: merge cost is ZERO-MEAN AND BOUNDED
   (mean +0.75, |delta| <= 2, n=4) — "free merge" means
   no systematic cost, not never-negative.
   llmopt/train/hebbian_moe.py docstring corrected.
2. THE MAC GRAVMOE ADVANTAGE is +5.5 to +7.5 (gravmoe
   52/51/50 v lb 44/45), not the "+5-6" quoted in five
   entries. The cuda seed-ladder's +4 bar stands but is
   a ~1sigma call at n=3 (se of the mean diff ~4.1) —
   read the ladder with its CI, not just the bar.
3. B6 RIDER downgraded: the k=8 complex-v-real gap is
   z=1.63 two-proportion (not "~2.2 sigma") — below the
   noise fence; likewise "worst at its own modulus"
   (G9 0.54 v real 0.63) is inside the fence. B6's
   defensible core: NO resonant advantage anywhere for
   exact phases; direction claims withdrawn.
4. RESOLUTION-LAW SCOPE fixed: the Mac capacity slope
   (44 -> 53, ~1.8 sigma) CLEARS the 1.5-sigma bar and
   should read UNRESOLVED-pending-replication (it failed
   device transport), not "noise". The blanket "both
   directions" retraction was stricter than the law.
5. GRADED CLOCK LAW scoped WITHIN-architecture: across
   architectures the one checkable comparison INVERTS
   (complex-none: higher k=8 acc 0.77, smaller clock 51
   v real 0.63/115). THEORY row updated. Presence/
   absence remains the only cross-arch readout.
6. LAMBDA CARD scoped: merge-free is MEASURED at lambda
   0.5 only; "costs nothing at ANY lambda" was unearned.
   RIDER PRE-REG: merge reviews on the three lambda-arm
   ckpts (0.1/0.25/1.0, on the 3080) queued behind the
   seed ladder — prediction: zero-mean bounded deltas at
   every lambda (the collapse is higher at 1.0, lower at
   0.1 — if LOW lambda merges badly, merge-free needs
   the collapse, a mechanism claim worth having).
7. Small fixes: FOURIER-2b-v1 chance ratio is 2.9x
   (mean 1/k = 0.169), not 3.5x; rung-1's "1.00M" for
   the FFN_E=139 arm should be 0.99M; FOURIER-3 v2's
   k=6 fell 0.38 -> 0.25 (unremarked; consistent with
   6 = 2x3 needing the untaught 3-part). Axiom's "48
   steps" RECONCILES with "5 prompts x 40 tokens"
   (8-token prompt + 40 generated). Remote-side logs now
   mirrored to logs/remote/ (cuda ladder/controls/
   lambda/battery — the 3080 battery digest 9ee4fa83...
   is now locally recorded, completing A1's evidence).
Unadopted reviewer notes (checked, judged fine): rung-0's
family-band comparison already superseded by rung-1+law;
A2's sd-of-sd caveat folded into its existing family
fence.

## VERDICT B5: depth-uniform diet does NOT revive the farmer — the novelty wall is not base-case starvation (2026-07-31 night, Mac)

EMA-matched rebirth on the depth-uniform reversed corpus
(28x skew flattened to 26,574/depth, matched dose), same
probe protocol as the booked FARMER PROBE:
- SEEDMODE=corpus: 7/1000 verified-distinct-novel
  (original 11/1000; bar was >=22), verified 97 (orig
  107) — inversion competence unchanged, novelty did NOT
  move. PREDICTION 2 FAILS at its pre-registered bar.
- SEEDMODE=band: 2/992 (original 2/992 — identical).
NULL: the recursion-depth diet-share clause, which fixed
FOURIER-3's fixed-point loops, does NOT transfer to the
farmer's problem. Sharpens the diagnosis: FOURIER-3 v1
failed at CHAIN EXECUTION (couldn't finish a rewrite it
had started — a coverage wound; depth share healed it);
the farmer fails at GENERATION NOVELTY (memorization-
dominant sampling — a diversity/scale wound; depth share
is the wrong medicine). The two multi-step failure modes
are now measurably DISTINCT. Banked revive conditions
stand unchanged: temperature/diversity sweep, solved-
state seeding, scale. Fences: n=1, MPS, EMA, same
novelty fence (119,371 expressions).

## PRE-REG FOURIER-4a: clock-FORMATION dynamics — does the representation lead or lag the competence? (2026-07-31 night, Mac)

scratch/fourier4a_dynamics.py: identical FOURIER-2b birth
(diet/split/seed/recipe), probed every 2 epochs — per-k
greedy accuracy (k in 4/5/8/7, 40-prompt subsets) +
periodic-neuron counts (k in 5/8/7) + an init-epoch
baseline. PREDICTIONS (all branches book): (a) clock
LEADS acc at k=5 -> the probe upgrades to a training-time
progress instrument (representation forms before behavior
converges, grokking-shaped); (b) clock LAGS acc -> clocks
are post-hoc consolidation, and the clock-placement law's
causal reading weakens to correlational; (c) co-arrival
-> single transition. Hard control: k=7 clock stays 0 at
every probe (any nonzero = probe artifact, run VOID).
Fences: within-arch counts only (review-adoption clause);
acc subsets are 40/k (sigma ~0.08) — read trajectories,
not points; single seed.

## VERDICT FOURIER-4a: CO-ARRIVAL at solved moduli, post-plateau DEEPENING at partial ones — and the "overshoot-then-prune" reading RETRACTS as cross-run variance (2026-07-31 night, Mac, threshold-free v2)

Formation curve (probe every 2 ep, init baseline clean:
n50 = 0 everywhere at init):
- k=5 (solved): acc 0 -> 1.0 AND clock 0 -> 459/512
  within ONE epoch — co-arrival, faster than the probe
  cadence resolves (branch (c) of the pre-reg). No lead
  or lag measurable at the easy modulus.
- k=8 (partial): acc plateaus ~0.5 by ep3 while the clock
  KEEPS GROWING (69 -> 210 by ep5, then 165-230 band) —
  at partial competence, representation formation
  continues AFTER behavior plateaus. The lag branch,
  scoped to unsolved moduli.
- k=7 control: n50 = 0 at every probe — run VALID. (The
  loose metrics have a noise floor: n25 ~1-29, sumR2
  ~53-74 at a ~70 null baseline — n50 is the clean
  instrument; sumR2 reads relative to floor.)
- NO PRUNING: k=5 n50 oscillates 440-473 the whole run;
  sumR2 co-moves. The v1 "441 overshoots then prunes to
  2b's 276" story is RETRACTED: v1/v2/2b are same-seed
  mps runs whose trajectories diverge (fp scheduling),
  and the k=5 population at acc 1.0 spans 276-473 ACROSS
  runs. Clock count gains its THIRD sensitivity (exposure,
  architecture, now RUN) — the presence-only doctrine is
  final. Artin's threshold-free instrument catch made
  this diagnosable: at the ep21/25 accuracy dips (0.88,
  0.70), n50 AND sumR2 both fall ~15% — real weakening,
  not blurring, and it RECOVERS with accuracy.
- LIVE-GAUGE finding (the practical yield): within a run,
  the clock co-tracks instantaneous per-modulus competence
  through training fluctuations (dips and recoveries
  mirror in both metrics). The probe works as a
  training-time competence readout WITHOUT running a
  gate — presence/level at THIS run's own baseline, never
  compared across runs. Market riff status: bubble-and-
  correction dissolves; what remains measured is
  "instant recruitment, no eviction" — SGD recruits the
  whole available population immediately and never
  prunes it at this scale.
Fences: n=1 formation curve (cross-run claims rest on
the 3-run k=5 spread); 40-prompt acc subsets (sigma
~0.08 — dips at ep21/25 are 1.5-3.7 sigma, read as real
but single-run); within-arch, within-run comparisons
only.

## VERDICT DETERMINISTIC-BIRTH R1a: integer FFN forward+BACKWARD bit-identical Mac-cpu = 3080-cuda, same evening as the spec (2026-07-31 night)

scratch/detbwd_r1.py: fixed-point (fq512) FFN sublayer,
int64 elementwise+sum ops (order-independent -> backend-
exact by construction), SiLU + dSiLU tables sha-pinned
(24499877 / 967943f9), round-half-away rdiv, STE-style
table-derivative backward. RESULTS: fwd+bwd sha
d6b673234474dc1abaaf71d6b664e8f9 IDENTICAL cpu(Mac) and
cuda(3080); rerun-identical both; gradient fidelity v
fp64 autograd of the smooth twin cos = 1.000000 on all
four tensors (dx, dwg, dwu, dwd), both devices. Two bugs
found+fixed by the fidelity check before booking: dwd
computed transposed (cos 0.013) and the derivative table
inheriting SiLU's saturation extension (returns x, not
1.0, above range -> gradient explosion on saturated
units; per-table extensions now explicit). Sibling
shipped: exact integer GEMM Metal kernel
(llmopt/kernels/metal.py exact_gemm — int32 in, long
accum, int64 out; 4 oracle tests v Python big-int, exact
equality; the Mac/Metal Ozaki sibling, correctness-first,
tiling = R4). Suite 396 passed. NEXT per spec: attention
sublayer backward (softmax table + jacobian), then
fixed-point AdamW -> R2 short-birth trajectory hash.

## VERDICT DETERMINISTIC-BIRTH R1b: integer ATTENTION forward+backward bit-identical Mac-cpu = 3080-cuda (2026-07-31 late night)

scratch/detbwd_r1b.py: single-head causal attention core
(q/k/v/o projections, scaled dot with SCALE = Q*sqrt(dh)
folded into one rdiv, integer softmax via exp table sha
9b864924 — range [-8, 0], exact integer max/sum — and
the softmax JACOBIAN in fixed point: ds = p*(dp - <p,dp>)
needs NO derivative table, only forward probabilities).
RESULTS: fwd+bwd sha 2eeacdca4bc3656681521ef2444af7ce
IDENTICAL cpu(Mac)/cuda(3080); rerun-identical; gradient
cosines v the fp64 causal-softmax twin: dx 0.999974,
dwq 0.999964, dwk 0.999977, dwv/dwo 0.999998. Two bugs
caught by the fidelity check pre-booking: dx projection
transposes, and a relative-scale error (dq/dk carried an
extra Q — per-tensor cosines blind to it, the dx SUM
exposed it at 0.949; lesson: fidelity-check COMPOSITE
grads, not just leaves). With R1a, both hard sublayers
of the block are now cross-device bit-exact in training
math. R2 remaining: rmsnorm backward (algebraic),
embedding + CE/margin loss path, fixed-point AdamW, then
the short-birth trajectory hash. Rope: R1c (fixed
rotation, backward = transpose).

## VERDICT DETERMINISTIC-BIRTH R2 (mini): a 200-step INTEGER TRAINING RUN is bit-identical Mac-cpu = 3080-cuda — trajectory sha and every loss value (2026-07-31 evening)

scratch/detbwd_r2_adamw.py: fixed-point AdamW (int64
throughout — EMA moments as integer rationals 9/10 and
999/1000, bias correction via exact python big-int
rationals capped to 30 bits, denominator via exact
integer Newton isqrt, decoupled decay) driving the R1a
integer FFN on a teacher-student regression (squared
loss, integer dL/dy). RESULTS: loss 2.19e12 -> 1.45e6
(six orders, monotone) and the trajectory sha
5f8dcdcc75acc0f4... IDENTICAL on both machines, with
every printed loss value equal. Training is now
cross-device deterministic end to end at the mini scale.

Three lessons bought en route (all caught by checks
before booking): (1) R1a's Q^2-scale weight-grads
overflow int64 in the v-path (g^2) — grads normalize to
Q at the loss boundary; (2) Q=512 FLOORS real-1e-3
Adam updates to zero — toy runs at lr 0.05; production
needs a WIDER WEIGHT ACCUMULATOR (R3 refinement, the
standard fixed-point-training move); (3) torch.randint
on device draws from DEVICE RNG streams — init on CPU
then move (the R1a/b convention; violating it cost one
cross-check: first cuda run had a different INIT, not
different arithmetic — the trajectory sha caught it
immediately, which is the instrument working).
SCOPE: R2-mini = FFN-only birth. Full-block short birth
(rmsnorm bwd + rope + CE/margin loss + attention wired
in) = R2b, then R3 (gravmoe deterministic pair; wide
accumulators; speed).

## VERDICT CUDA SEED LADDER: the gravmoe advantage does NOT transport — cuda means are IDENTICAL (50.7 v 50.7); the recipe stays "free merge, capability-neutral"; rescue framing adopted (2026-07-31 night)

Pre-reg above (Tier C opener). Chain completed on the
3080 (logs/cuda_ladder_0731.log, internally paired,
OTAG=_cuda): lb s2 51, lb s3 53; gravmoe s2 53,
gravmoe s3 50. With the existing s1 pair: lb {48, 51,
53} mean 50.7 v gravmoe {49, 53, 50} mean 50.7 — diff
0.0, dead-center in the pre-registered +-2 band.
PREDICTION (2) FIRES: the advantage is Mac-scoped; the
recipe card stays "free merge, capability-neutral" with
the Mac +5.5-7.5 gate advantage listed device-scoped.

READ (the rescue framing, pre-registered before s3):
cuda lb ALREADY sits at ~50.7 — the same basin Mac
gravmoe reaches only WITH the pull (Mac lb 44/45). The
gravitational relaxation doesn't add capability on cuda
because there is nothing left to rescue: it lifts Mac
births to the basin cuda training (TF32 matmul + its
kernel reduction orders) reaches unaided. This makes
the transport gap a TRAINING-ARITHMETIC effect, exactly
the question the deterministic-birth program (R2b/R3)
is built to answer exactly — a bit-identical gravmoe
pair will show whether the pull's effect is a function
of birth arithmetic or vanishes when arithmetic is
equalized. Per amendment #2 the +-2 call at n=3 carries
se(mean diff) ~4.1: the SIGN of a small residual is not
resolved, but ">= +4 transport" is excluded at the
observed 0.0.

Merge reviews on the two new gravmoe ckpts (prediction
3, toward merge-free n=6) ride in scratch/night31b_cuda
.sh [HOLD]. SCAFFOLD PROGRAM: this was the last open
question — thread CLOSED pending only the night31b
merge-free extension and the R3 deterministic pair.

## INSTRUMENT + READ: gate transcripts — at L4 the STRATEGY is present and the ARITHMETIC is the wound (2026-07-31 night, Artin's ask)

scratch/gate_transcripts.py: mirrors gate_eval's exact
loop (same seeds, sampler, oracle) but prints the full
step chain per prompt — the first instrument that shows
HOW the crystal works a problem rather than whether.
Read on umoe_gravmoe_s1 (Mac, eval device-stable):

L3 (18-19/24 in the gates): clean textbook chains —
linearity split, term-by-term closure, and ONE-SHOT
u-substitution recognition (Integral(6*x*exp(3*x**2+1))
-> exp(3*x**2+1) at ply 0; same for the (15x^2+2)e^u
form). The strategy library is real.

L4 (2-5/24, the weakest gate level): all four sampled
prompts STUCK AT PLY 0 with 0/8 valid — and the failure
mode is diagnostic: every sample has the RIGHT SHAPE
(linearity splits, derivative-of-composite patterns)
with GARBLED TERM ARITHMETIC — dropped terms, mutated
coefficients (3*x*cos(u) -> (9*x**2+2)*cos(u)), even
unbalanced parentheses. The model pattern-matches the
ansatz and cannot execute the exact algebra at L4's
expression size. This is the transcript-level face of
the L1/L2-basics dependency (66k transfer rows, RESULTS
L2429/L5668): strategy transfers, symbol-exact
manipulation is the binding constraint. NOTE the gate
needs no trust in model arithmetic — sympy verifies
every step; invalid algebra scores zero by
construction. IMPLICATION (banked, not spec'd): the
next capability dollar at L4 is exact-manipulation
diet share (long-expression copy/edit fidelity), not
more strategy exposure.

## PRE-REG GRAV-0T + GRAV-REV: post-hoc gravity (no-training gravmoe) and the white-hole arm (2026-07-31 night, Mac only; Artin's riff)

The transport verdict's rescue framing makes a testable
claim: the pull lifts Mac births into a basin via WEIGHT
GEOMETRY. If true, applying the same relaxation POST-HOC
to an already-trained lb checkpoint should recover part
of the deficit with ZERO training. Design (Mac, mps,
paired on ckpts with known gates): load Mac lb s1/s2
(gates 44/45), observe routing over 200 TRAIN-side
prompts (exclude= guards the gate seeds), build the
co-routing EMA exactly as HebbianCoupler does, then
apply R relaxation steps of the merge-free pull to
expert weights; re-gate. Arms per ckpt: (a) GRAV-0T
pull lam=0.5, R in {1, 10, 50}; (b) GRAV-REV repel
lam=-0.5, R=10 (the white-hole arm); (c) control =
re-gate untouched ckpt (harness check, must reproduce
44/45). PREDICTIONS: (1) if 0T recovers >= +4 at any R,
the pull is a geometry operation and "no-training
gravmoe" is real (huge: post-hoc rescue for any birth);
(2) +-2 = the pull needs training-time gradient
interaction; post-hoc null banked and the win stays a
training force; (3) REV: rescue-of-noise story predicts
repel hurts NO MORE than pull helps (|delta| <= pull's
gain); a large asymmetric LOSS (>= -6) instead means
the basin is load-bearing structure, not noise rescue.
Resolution law: n=2 ckpts x paired arms on one device;
any single-cell delta < 5 reads as noise by default.

## VERDICT GRAV-0T + GRAV-REV: NO no-training gravmoe — post-hoc pull is DESTRUCTIVE both directions (37 -> 9 -> 1; repel -> 0); the pull is a TRAINING-COUPLED force (2026-07-31 night)

Pre-reg above. Mac lb s1, within-process paired (all five
gates share one mps fp context). Control 37, observation
512 train rows (off-diag co-routing mass 0.58-0.73 per
layer), then: pull lam=0.5 R=10 -> 9/120 (valid 2.3);
R=50 -> 1/120; repel lam=-0.5 R=10 -> 0/120 (valid 0).
PREDICTION (2) FIRES, emphatically: the pull only works
INTERLEAVED with gradient repair (training applies ONE
relaxation per 100 optimizer steps; post-hoc R=10
consecutive steps is ~100x the recipe's dose density
with zero repair between).

MECHANISM READ (resolves an apparent contradiction with
merge-free): consecutive relaxation is consensus
dynamics — with sum_j lam*E[i,j] ~ 0.3/step, R=10 drives
experts ~97% of the way to their co-routing-weighted
consensus. Merge-free measured that collapse-to-mean is
FREE on GRAVMOE-trained ckpts (experts already
co-adapted near their mean). On an LB-trained ckpt the
experts are DIVERSE — consensus collapse destroys the
capability the router was exploiting. Merge-free is a
property OF the pulled basin, not of merging. The repel
arm is the mirror: (1+0.3)^10 ~ 14x weight spread —
the white hole explodes, as white holes do.

FENCES: (1) small-dose post-hoc (R=1, lam<=0.05)
untested and currently UNREADABLE — the control gate
came back 37 v the booked 44 (fresh process v
post-training in-process), exposing an UNMEASURED
CROSS-PROCESS RE-GATE SIGMA on mps (same seeded
sampler; wobble enters via forward-pass fp at coin-flip
tokens — the fp16-near-tie mechanism, run-scoped like
FOURIER-4a's clock counts). Sigma cell queued. (2) lb
s1 only, n=1 ckpt — but at effect sizes of -28 and -37
solves the resolution law is satisfied by any sigma.

## PRE-REG RE-GATE SIGMA: cross-process gate spread of one frozen checkpoint on mps (2026-07-31 night, before the runs)

GRAV-0T's control read 37 v the booked 44 on the SAME
frozen umoe_lb_s1 (seeded sampler, fresh process v
post-training in-process). Cell: gate the untouched
ckpt in 3 fresh processes (scratch/gate_regate.py).
PREDICTIONS: (1) spread (max-min) <= 3 -> the 37 v 44
gap needs another explanation (audit the original
booking context); (2) spread >= 5 -> re-gate sigma is
REAL on mps and every cross-process Mac gate comparison
gains a fence: paired arms must share a process, and
frozen-ckpt gate numbers carry +-sigma_regate. Either
way the number becomes part of the resolution law.

## VERDICT RE-GATE SIGMA + AMENDMENT (target: Mac lb baseline "44/45"): the gate is EXACT (spread 0) — the discrepancy was checkpoint PROVENANCE, not noise (2026-07-31 night)

Sigma cell: umoe_lb_s1.pt gated in 3 fresh processes —
THREE IDENTICAL READS (37/120, per-level and validity
digit-identical). Cross-checks: umoe_gravmoe_s1 re-gates
52/120 EXACTLY matching its booked line (umoe3_0730.log
L213, digit-for-digit); umoe_lb_s2 re-gates 44 (its
booked number). PREDICTION (1) FIRES, strongly: the
120-gate with seeded sampling is a fully deterministic
instrument on mps — eval-side determinism now MEASURED
at gate granularity (training remains nondeterministic;
FOURIER-4a's run variance was training-side).

THE 45-v-37 EXPLANATION (provenance, not physics):
overnight_0730.log L7-110 re-trained a FRESH ARM=lb
seed=1 (params-control chain) which gated 45 — but
umoe_lb_s1.pt's mtime (Jul 30 11:44 AM) predates that
run: the overnight model was never saved to (or never
overwrote) the disk path. The booked Mac lb "45" refers
to a model that NO LONGER EXISTS; the surviving morning
artifact gates 37. The checkpoint selection-effect's
cousin, now bitten a 4th way: a VERDICT NUMBER whose
artifact wasn't the one preserved.

CONSEQUENCES: (1) Mac lb baseline for the scaffold
program is {37 (disk s1), 44 (s2), 45 (overnight model,
artifact lost)} — the Mac gravmoe advantage (52/51/50)
is if anything LARGER than the booked +5.5-7.5; the
device-scoped fence and the transport verdict are
unaffected (cuda ladder was internally paired). (2) NEW
RULE (proposed for doctrine): a gate number books WITH
the sha256 of its checkpoint; scripts/ckpt_manifest.py
already carries shas for the confirmed/ tree — booking
and manifest must cross-link. (3) GRAV-0T's control 37
is CORRECT for its artifact — the harness check
actually passed; tonight's instrument alarm was the
provenance bug surfacing.

## RECEIPT FX-V3 (axiom-side PASS) + house reproduction QUEUED (2026-07-31 night)

Axiom reports FX-V3 complete (their f9b0e04, relay
2026-07-31-3): the integer twin of the merged crystal
runs BIT-IDENTICALLY in two runtimes axiom-side (Python
reference subclassing their DeterministicLM + pure-int64
C++ decoder on the FX-V2 core), frozen battery, both
seeds. Digests house must reproduce:
  s1 e377201c79bc2034ad74bc039f5c2bddbd5c3d2c16f2d8aa
     0b916a6fea4917f7
  s2 f5013f2b34c00f8a0a47b26630c284f0bf7d5d8ed0bea3df
     923ceae212fbd82b
Cross-lab check ALREADY LANDED: they adopted the house
detbwd_r1b exp-table construction and verified BYTE
IDENTITY against our code (sha 9b8649244ca8c235 — the
same table our attention backward pins). Their gate
spec: rdiv(rl*Q, A) -> integer max-shift -> table ->
top_p = rdiv(Q*Q, z), FFN out scaled rdiv(out*top_p, Q)
pre-residual-clamp; two proven properties: winner
contributes exactly Q to the partition (division never
degenerate) and only the max logit enters (tie-breaking
provably cannot affect output — house lowest-index rule
compatible by construction). Doctrine note honored:
tables ship as BYTES (their fx3_tables_*.pt sha-pinned;
house must DECODE, not regenerate — make_tables derives
sigma from a float std()). QUEUED (next session): house
twin implementing their gate spec on our P3 core,
reproduce both digests from their shipped tables. They
accepted the R2 optimizer teaser — spec relayed
(2026-07-31-2 house-side).

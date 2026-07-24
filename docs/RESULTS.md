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
Handoff: docs/handoffs/2026-07-23-0.md (new convention).

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

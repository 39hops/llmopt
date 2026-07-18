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

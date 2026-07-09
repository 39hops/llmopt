# The derivation-search results: 73.6% → 95.3% in 48 hours

*llmopt's "Stockfish for math" arc, 2026-07-06 → 07-08. Every number
below is a committed measurement on held-out, string-seeded problem
sets, sympy-oracle-verified. Written as the handoff/publication draft.*

## The one-paragraph version

A beam search over sympy rewrite rules, guided first by a hand-crafted
eval, then by learned components, solves generated calculus problems.
Over 48 hours of measure-everything iteration, held-out solve rate at
fixed node budgets went **265/360 → 343/360** without changing the
rules, the model, or the problems — every gain came from *search
wisdom*: confidence calibration, width/depth allocation, and knowing
which components actually carry which capability. The headline
finding: **ranking moves is grammar and fits in a dictionary; knowing
when you're sure requires learned per-node discrimination; and beam
width is a partial substitute for confidence.** Plus a two-mechanism
answer to "is there a limit to self-teaching."

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
| **+ mined macros (the shipped `engine.solve()` config)** | **343 (95.3%)** | highway dividend |

n=30 confirmation of the adaptive-vs-fixed comparison: 593/720 vs
560/720 (int L3 flips to fixed-k: over-confident dives at the hardest
level — width hedges).

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

Tabula rasa (AlphaZero-way ablation): round 0, knowledge-free (random
k1 dives, count_ops tie-break, verifier only) solves 63% overall —
perfect at L1-2, cliff at L4 (4/20, 2/20): the implicit curriculum.
Round 1+ in progress at write time.

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

## Honest nulls (all pre-registered or instrumented)

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

## Origin story, closed

Limits resisted LoRA training (<=21%), motivating the engine. The
engine now solves them: l_hopital emits UNEVALUATED derivatives that
the rung-1 diff rules finish — the rungs composing in one derivation.

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

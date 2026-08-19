# MODEL-2 design spec (design phase only — registration and launch go through /rung on Artin's GO)

Written 2026-08-18 evening on Artin's design GO. Evidence base:
MODEL-1 tree (INSTRUMENT-ALARM), ATTN-ATTRIB-1, IO-ATTRIB-1,
LBAND-1 (+alarm diagnosis closure), RK-CENSUS-0, CAPACITY-METER-1
r2/r3, EFFORT-QUANT-0. Everything below cites booked numbers only.

## 1. What MODEL-2 is

MODEL-1 asked "how damaged is the 2-bit artifact, and does repair
attribute?" — answered: badly damaged (X_A 1.061, alarm), repair
attributes with strong structure (io splits by end and metric;
attention redundant across families; linear repair early-heavy;
one interference cell). MODEL-2 is the ALLOCATION instrument: given
a byte budget over repair primitives, does a REGISTERED allocation
policy beat the flat/aggregate policies at matched spend?

## 2. Frozen inputs (already booked, reused verbatim)

- Teacher lock 0ca4151 (v2d), corpus/prefix token lists, X/K
  scorer (Mac CPU, fp16 record, sensitivity floors), the
  small-n=30 fence, margin bins.
- Repair primitives with measured per-byte value (frozen
  receipts): io pair D/E (0.5920 GiB), full-attn F (0.3905), BLe
  (0.4296), BLm/BLl, L (1.2887), C-payload family.
- Standing reference arm: BLe (Artin ruling 2026-08-18; NOT a
  deployment default — free-gen screen required first).

## 3. Design requirements (all earned this week, each cites its
   incident)

R1. TWO-GATE SPLIT (banked at MODEL-1): separate
    INSTRUMENT-ALARM (data-integrity/assumption breach:
    bracket, teacher identity, traversal) from
    LOW-RATE-OUT-OF-RANGE (a registered quantity landing outside
    its predicted band). MODEL-1 conflated them; LBAND's
    monotonicity assumption became an alarm that suppressed a
    clean refutation. MODEL-2 bars carry a "gate_class" note
    field: sanity gates may suppress refutation; range gates
    never do.
R2. refutation_precedence REQUIRED in the machine JSON
    (engine + CLI shipped a2899e5; five library fixtures + one
    end-to-end CLI fixture). The precedence names ONLY
    sanity-class bars.
R3. REGISTERED OBJECTIVE for any combined ordering: X/GiB and
    K/GiB orderings DISAGREE on io's rank (X: early-linear 0.933
    > full-attn 0.802 > io 0.384 > late-linear 0.064; K: io
    0.227 > early-linear 0.209 > full-attn 0.189 > late-linear
    0.114). MODEL-2 must register the objective BEFORE pricing
    arms. Candidate: report X-primary and K-primary allocations
    as SEPARATE registered arms rather than a weighted scalar —
    a weighting constant would be a free parameter with no
    booked justification.
R4. NO MONOTONICITY ASSUMPTION in sanity brackets for
    conditional arms: LBAND's FLl cell measured negative
    marginal X on top of F (bit-identical on pair rescore).
    Sanity brackets widen to [min(C, base) - 5f, max(base, A) +
    5f] or drop the upper edge for stacked arms; interference is
    a RESULT, not an instrument failure. (Forward-only; LBAND
    stands.)
R5. Launch invariants (earned 2026-08-18): full-path markers;
    interpreter pinned in every remote command; compose gate
    (admissibility minus score-chain) runs pre-score; derived
    byte spend from manifests, committed before receipts.
R6. Auditor pair pre-booking, claim_lint with prereg+obs, all
    receipts force-added at booking, code_commit pinned. Scorer
    RESCORE mode exists for repeatability checks.

## 4. Candidate arm sets (desk-priced, nothing composed yet)

Budget anchor: B's io spend class (~0.59 GiB) and the ~1.0-1.3
GiB class where BLe+io combinations live.

ARM SET ALPHA ("policy v aggregate", X-primary, ~1.02 GiB):
- P_X = BLe + io pair (0.4296 + 0.5920 = 1.0216 GiB): the
  X-primary policy stacks the two highest X/GiB primitives.
- AGG = L-aggregate truncated? NO — no 1.02 GiB aggregate
  exists; the honest aggregate control is B+F (existing
  0.9825 GiB class, X 0.520 measured as F) — near-iso, disclose
  the 4% byte gap, or compose a fresh matched control.
- Predicted (from additivity found in IO-ATTRIB, near-additive
  bars): X(P_X) if independent ~ B - 0.401 - 0.227 = 0.206
  v C 0.249 — a POLICY ARM PRICED TO BEAT C's X at 40% of C's
  spend. That is the headline question: does allocation beat
  uniform upgrade.
- Interference risk is the science: LBAND says stacking is not
  free (FLl cell). The near-additivity bars from IO-ATTRIB are
  the registered prediction; their failure is the finding.

ARM SET BETA ("K-primary", ~0.59-1.02 GiB): io pair alone is
already measured (that IS B). K-primary adds BLe to io: same
composite as ALPHA — the two objectives CONVERGE on the same
first two picks, differing only in order. This collapses ALPHA/
BETA into ONE composite arm + the registered per-metric
predictions. Cheaper design: 1 new compose (BLe+io = "P1"), one
score, against frozen B, C, F, BLe receipts.

ARM SET GAMMA (optional second rung): P1 + F (~1.41 GiB) — does
the third pick still pay after redundancy? Predicted X ~ 0.206 -
(F's conditional value | early-linear repaired). The I_X(e) =
-0.229 redundancy says F's value conditional on BLe is ~0.17,
not 0.28 — a REGISTERED interaction prediction, first real test
of transporting the conditioning table.

## 5. Bars sketch (to be frozen at /rung registration)

- SANITY (gate_class: sanity, in precedence list): teacher
  identity, traversal 48/16, R4-widened bracket, compose
  admissibility (base/donor/mark/keys/derived bytes).
- PRIMARY (gate_class: range): X(P1) below X(C) (allocation
  beats uniform at 40% spend) — FIRE = the headline.
- ADDITIVITY (range): |X(P1) - (X_B - dX_BLe - dX_io)| within a
  registered band (5f floor multiple for the numerical read + a
  0.05-nat interference band for the science read, SEPARATE
  bars — floors are never significance).
- K analogues as separate bars; per-metric, no combined scalar.
- refuted_if: the allocation prior is refuted if X(P1) reads
  ABOVE B - 0.5*(dX_BLe + dX_io) (less than half the independent
  prediction materializes — interference dominates allocation).
  Predicate on the P1 receipt; precedence: sanity bars only.

## 6. Costs (desk)

1 compose on 3080 (~2 min + transfer), 1 score on Mac (~8 min),
GAMMA doubles it. Auditor pair per booking. Total wall well under
an hour per rung. No GPU-heavy or long jobs; fits any window.

## 7. Open decisions for Artin (before /rung)

1. ALPHA/BETA collapse to composite P1 (BLe + io): approve the
   single-composite design v separate per-objective arms?
2. GAMMA (P1 + F) same night or second rung?
3. Control choice: frozen B+F-class receipts as near-iso control
   (disclose 4% byte gap) v composing a fresh matched control.
4. The free-gen screen for BLe deployment promotion — separate
   observation, or fold as a rider on the P1 rung (one mlx/CUDA
   generation smoke, EFFORT-QUANT harness exists)?

## 8. What this spec is NOT

Not a registration (no bars frozen here), not a launch, no
composes armed. The kurtosis/meter thread stays out of MODEL-2's
gates (diagnostic, never allocator — booked twice).

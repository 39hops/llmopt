# Handoff 2026-08-11-2: the merge-space arc (5 rungs, one mechanism) + floor ladder closed; GROW-DECOMP live

Session: Fable 5 seat, afternoon block. Previous handoff:
2026-08-11-1 (morning sweep + refactor infra). Resume chain: this
file -> RESULTS tail -> BOARD.

## THE DAY'S ARC — merge space went from one null to a mechanism

Five rungs on the 3080, all pre-registered, all booked same-day:

1. MERGE-SPACE-1 (26770): six independent-init pairwise averages
   ALL gate 0/120 (not degraded — dead); shared-init fork merge 14
   in the 12-15 parent band; task vector 1/120 (first booked TV
   result, negative). Rider: d64 birth-seed variance 12-30/120.
2. MERGE-SPACE-2 (26866): P-INIT-INSUFFICIENT REFUTED — same init
   with ZERO shared optimizer steps merges in-band (avg_e0 11 v
   parents 10/12). Fence: pair shared epoch-seeded data order.
3. MERGE-SPACE-3 (26947): the deconfound. ORDER_SEED knob added to
   train_mathnative (default stream byte-identical); same-init
   INDEPENDENT-order merge = 13, top of band. Init is the address.
4. MERGE-SPACE-4 (27013): 4-way order-twin soup = 14 = family max
   exactly. Soup is FREE, not PROFITABLE (flatten-onto-best).
   Bonus: byte-identical merge rerun (pair_a sha == avg_ord sha) —
   merge path deterministic end-to-end.
5. MERGE-SPACE-5 (27158): n=3 paired-seed replication — every
   same-init merge in-band (13/25/30 v pairs 12,11/23,25/30,31),
   zero craters. REPLICATED tag earned.

Doctrine consequence (booked in 3): merge.py's shared_lineage
now MEANS shared birth init; a BIRTH_SEED-keyed merge gate is a
legal catalog rule. Soup-of-N stays capability-neutral at d64.

## FLOOR-HK-1 CLOSED (27055) — both bars fire

Fresh ladder, warm diet, fp32/mps: d64 0.4364 > d128 0.3815 >
d256 0.3566 > d512 0.3478 (monotone, bar 1); d512 never
approaches H_32 0.187 (bar 2) — 8x width buys ~1 token of
effective context past the 16-gram wall. Gates 43/65/67/72.
Observations booked: fresh d512 floor sits ON the grown-crown
0.348 reference (provenance does not move the floor); fresh
d512/L8 gates 72 v crown band 73-75 -> priced GROW-DECOMP-1.

## LIVE RIGHT NOW

- GROW-DECOMP-1 cell A (pre-reg 27101-ish, grep GROW-DECOMP):
  fresh birth at the crown's EXACT arch d512/L12/ffn2304/h8, warm
  diet, fp32/mps, rjob growdecomp1 — the growth-v-width
  decomposition. ~7 h wall (2.1 it/s, 5565 steps/ep x3 + gate).
  Bars: fresh <= 67 = growth premium real; within 2 of 74 =
  premium demotes to schedule-savings + amendment-grade re-read
  of the +10.7 lever. Books tonight.
- 3080: IDLE, window closed ~17:00 — next launch needs Artin GO.

## INFRA SHIPPED THIS BLOCK

- lab/gate.py (verbatim sample_wave_lp + gate_eval, source-identity
  guarded, GateSpec + gate_checkpoint API), lab/hash.py (one digest
  semantics; fixed runfiles' un-anchored git_sha), lab/jsonl.py
  (read/write/append semantics). All exported, INDEX/CODEMAP regen.
- ORDER_SEED knob (train_mathnative): default byte-identical,
  string-seeded fork otherwise. The merge-space instrument.
- 33 gate rows in the lake (device + n_seeds + weights_sha, all
  dict-sum verified).

## HONEST FAILURES BOOKED IN-SESSION

- Committed on red TWICE via `pytest | tail` swallowing rc (the
  2026-08-07 hazard, verbatim recurrence). Both healed same-hour;
  guards that caught it were the ritual tests themselves. Standing
  fix for the session: rc read directly, never through a pipe.
  This is the live argument for queue item: CI block.

## QUEUE (post-GROW-DECOMP)

1. Book GROW-DECOMP-1 when it lands (bars above).
2. Mechanize-the-ritual block (opus-chat cross-check, RIFF-banked):
   .github CI (pytest + INDEX/CODEMAP/results-index staleness),
   doctrine-lint tests, GENERATED/HAND headers, verdict tags,
   rjob prereg_ref refusal.
3. R4 curriculum-funnel micro (needs trainer curriculum knob —
   design first), R6 uncited-checkpoint revival (Mac, START LAST),
   R7 fingerprint dedup, R8 exact-mode gate v rounded (the named
   precision retest slot).
4. README headline refresh + THEORY rows (merge-mechanism row is
   now REPLICATED-grade; loss-floor row Shannon-cited) — next
   freeze point.
5. Corpus-manifest overlay (grok D).

## STANDING FENCES THAT TRAVEL

- msearch family (3080) and floorhk family (Mac) NEVER compare.
- All merge-space deltas except the n=3 binary pattern are
  single-seed; magnitudes not claimed.
- 0.348 stays a d512-grown reference line; the ladder owns its own
  floors.
- 3080 post-17:00 = HOLD for Artin GO. R6 starts LAST.
- wsl.sh verb split is NOT a security thread (2026-08-11-0 note).

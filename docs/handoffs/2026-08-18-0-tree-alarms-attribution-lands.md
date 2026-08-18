# Handoff 2026-08-18-0: the tree alarms honestly, attribution lands the same night

Seat: Fable (main session model), Mac. HEAD at close: 118450e.
3080: idle (F/L/Q artifacts + A/B/C still resident at
~/qwen_whole0t/, checkout at origin/main). Mac: idle, all local
27B artifact copies deleted after scoring (chain files are the
byte-identity record). Resume = this file -> BOARD -> RESULTS
tail.

## What landed (one night, scorer to two verdicts)

- **VERDICT QWEN-MODEL1-TREE** (effbea3): the frozen tree's first
  firing books INSTRUMENT-ALARM — X_A = 1.061 nats sits 6% over
  the pre-registered 1.0 uniform-damage gate, fail-closed, no
  allocation branch; every trigger underneath fired monotone
  (B-over-A 21.4% X, C-over-B 70.2%). T1 io prior UNADJUDICATED
  under the clean-gates clause (prereg-auditor blocker caught the
  walker mapping any non-T1 branch to REFUTED; fixed pre-booking).
  Scorer + mechanical walker + projection JSON all shipped and
  committed BEFORE receipts (docs/preregs/qwen-model1-tree.json).
- **OBSERVATION QWEN-MODEL1-POSTHOC-DIAGNOSIS** (9cce242): damage
  is real precision loss, not instrument (monotone repair ladder,
  shared decode fixtures); A "outside the registered fidelity
  regime at 2.07 bpw", not dead; the per-byte inversion (io ~10%
  better X/GiB, 2.2x K/GiB); B's flips repair LARGE-margin teacher
  decisions; gate lesson (INSTRUMENT-ALARM v LOW-RATE-OUT-OF-RANGE)
  banked for FUTURE trees — the 1.0 gate itself immutable (Artin
  ruling; GPT concurrence).
- **PRE-REG QWEN-ATTN-ATTRIB-1** (9cce242) then **VERDICT** same
  night (35a87a8): L-DOMINANT (R_X(L)=0.949 v F 0.536) but the two
  attention families are heavily REDUNDANT (recoveries sum 1.49,
  both near-additive bars miss) — next-grain singleton split
  BLOCKED-BY-INTERACTIVE; per-byte inverts (F 0.802 nat X/GiB =
  2x everything; io keeps the K crown); the discretion-free
  iso-rate arm Q (A + in_proj_qkv, matched to io's budget within
  1.05%) confirms io beats attention at matched bytes on BOTH
  metrics. Priors: (i) held, (ii) held, (iii) MISSED (honest),
  (iv) held with X decisive.
- Instruments shipped: scratch/qwen_model1_score.py (X/K scorer,
  full refuse-list), scratch/qwen_tree_adjudicate.py,
  scratch/qwen_recompose.py (key-level byte recomposition),
  scratch/qwen_attrib_adjudicate.py; 32 new test fixtures across
  four test files. Both auditor pairs ran pre-booking on both
  verdicts; every should-fix adopted in code or disclosed in the
  entries.
- Two riff banks (per-byte allocation lens; future-tree gate
  split). MTP confirm done: exclusion set = exactly 333 vision +
  15 mtp keys, zero leakage either direction (desk, from B's
  manifest). External-runtime check: mlx-lm 0.31.3 ships
  qwen3_5.py and vendor model_type matches — conversion smoke
  itself DEFERRED (q8 write ~28GB v ~29GB free while the Mac was
  scoring).

## Conditions that bite next session

- Artifacts F/L/Q exist ONLY on the 3080; their chain files
  (logs/qwenwhole/artifact_digest_{F,L,Q}.txt, force-added) are
  the sole byte-identity evidence on the Mac side.
- The adjudicator floor fix (per-contrast max, 8c3042b) postdates
  the booked attrib receipts — receipts stand as emitted with the
  deviation disclosed in the verdict; a re-run would produce
  slightly different floor multiples (stronger), same outcomes.
- Scorer now admits arms A/B/C/F/L/Q; new arms need the allow-list
  + a chain file + a compose receipt.
- No sampling-uncertainty fence exists anywhere in the X/K chain
  (355/92 positions, point readings) — any future claim leaning on
  a small step must carry that sentence or register a fence.
- README front door: Artin ruled WAIT (no churn until the
  attention-attribution story settles; it half-settled tonight —
  his call whether the L/F/iso result is enough).

## Next session opens with (recommended)

1. The B+F artifact class question (7.48 GiB, X 0.52, 54% of the
   recovery for 23% of the attention bytes) — cheap recomposition,
   already-shipped machinery; needs its own pre-reg (it is NOT the
   blocked singleton split; it is an arm the redundancy result
   motivates).
2. D/E io attribution (embed v head of the byte-efficient 0.59
   GiB) — queued behind attention per Artin, now unblocked since
   attention attribution booked.
3. Depth-band split (early/mid/late thirds of L) — the one
   next-grain branch the interactive clause arguably permits;
   needs registration.
4. Q8/mlx conversion smoke when disk allows (~28GB; delete-after).

## Open decisions for Artin

1. B+F v D/E v depth-band: which registers first (all three are
   one-night rungs with existing machinery).
2. README Qwen paragraph: the story now has a stable shape
   (2-bit out of regime; io cheap and K-efficient; attention
   redundant pair carrying the bulk; F the X-per-byte king) —
   minimal status paragraph or keep waiting.
3. MODEL-2-class tree registration (the two-gate design from the
   diagnosis) — when the next whole-model compile happens.
4. 3080 overnight allocation (box idle, artifacts resident).

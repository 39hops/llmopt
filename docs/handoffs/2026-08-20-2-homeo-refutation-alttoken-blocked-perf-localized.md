# Handoff 2026-08-20-2: HOMEO books its registered refutation, ALTTOKEN blocked by its own gate (a finding), the 7x localized

Seat: Fable 5 on the Mac; HEAD when this file was authored 897985f
(final close HEAD = the handoff commit itself, 082aaa5; correction
noted 2026-08-20 post-review), both checkouts
lockstep; 3080 idle, nothing armed, no watchers live. Artin opened
both machines through the evening; nothing runs overnight.

## What landed (all same day, after handoff -1)

- PRE-REG QWEN-HOMEO-ACTUATOR-0 (7d30ccb) + pre-run AMENDMENTS
  -REFRESH (254e756: same-prefix reconstruction — the registered
  REFRESH was ALWAYS-HIGH by another name; unified 300-token escape
  criterion; refutation_precedence encoded) and -BOUNDARY
  (3c5d202: detector constants WIN=32/LAG_MAX=512/T_MIN=544 +
  1-based index semantics pinned; per-item boundary fixture).
- BLem composed (48 keys, +0.4296 GiB exactly, self-qualified).
- VERDICT QWEN-HOMEO-ACTUATOR-0 (7312097, RESULTS L37432): THE
  REGISTERED REFUTATION — sanity 3/3 exact through the treatment
  reconstruction path, bars 2-5 zero, 0/6 high-arm escapes, every
  branch's first escalated token IS the loop-continuing token
  (divergence only 14-157 tokens later); precision actuator
  DEMOTES at the BLem dose; controller stage does not launch on
  it. Offline consumer authoritative (own extraction + per-family
  sympy), zero producer mismatches, both chains 18/18, both
  auditors pre-booking (receipt-audit S1-S4 adopted + re-run;
  prereg-audit prior-scoring + criterion-change disclosures in the
  booked text). Post-booking wording chain: -RUNTIME-WORDING ->
  -DIAGNOSIS-SCOPE -> -BAR4-SCOPE (each narrowing an overclaim).
- PRE-REG QWEN-ALTTOKEN-CONTROL-0 (e6def15) + AMENDMENT
  -FREEZE-PROTOCOL (immutable params, phase-1 receipt =
  control_table.json with exact-bytes-in-HEAD required for phase
  2, 0.05-logit match gate, argmax preconditions). Phase-1
  derivation ran derivation-only and the GATE FIRED: VERDICT
  CONTROL-MATCH-FAILED (10a7386, L37747) — no admissible
  gap-matched control outside the vendor top-256 at ANY locus
  (best 12.18-18.61 logits, 244-372x the gate; 0-based BLe ranks
  204-223). Bars NOT-RUN. The blocking fact is a finding: the BLe
  near-top candidate region at these five near-tie h is covered by
  the exclusion union, dominated by vendor-top-256 (-CONTAINMENT-
  WORDING amendment carries the exact narrowed form).
- OBSERVATION QWEN-BLEM-DECODE-PERF-0 (2de2967, L37825): the HOMEO
  7x decode slowdown LOCALIZED by a five-phase elimination ladder —
  restored-cache x second-tower x cache-position>~1785-1849
  conjunction; cross-tower values, serializer FUNCTION (outputs
  correct; restored provenance itself remains condition (a) of
  the trigger — wording tightened 2026-08-20), contiguity, s16
  kernel (~1.5x w4 per route), RAM ballast, first-tower restores
  all measured out; native-prefill immune through 3641; identical
  onset for cross and roundtripped-native states. Resolves the
  diagnosis-scope fork to the BENIGN branch (HOT tokens computed
  correctly, slowly). Phase-5 allocator-counter slot banked.
- RECEIPT-LOCK OVERHAUL: brace expansion (both citation forms,
  annotation-safe), honest counter (sha-locked /
  present-but-pending / cited-but-absent / prereg-awaiting-run),
  present-pending ratchet (13 legacy pinned), parser-independent
  BOOKING-TIME INVARIANT (booked prereg receipts must lock;
  NOT-RUN absences pinned by exact identity, shrink-only), negative
  brace fixture. HOMEO + historic brace-cited receipts all
  individually sha-locked (191 -> 240).
- Banks: TOPSET-OVERLAP CENSUS (observation-only BLe v vendor
  top-set geometry at the five frozen h — run BEFORE any redesign
  pick); redesign triage (widen-tolerance DEMOTED, rank-match
  likely same incompatibility, drop-top256 framed narrowly as
  exact-vendor-token v equally-BLe-plausible candidate).

## Conditions that bite next session

- HOMEO run receipts frozen under logs/qwenhomeo/ (sha manifest
  inside); BLem artifact lives on the 3080 at ~/qwen_whole0t/BLem.
- Restored-state drivers on SECOND tower builds hit the perf
  conjunction — restart the process per tower or pre-grow the
  cache until phase-5 lands.
- The three ALTTOKEN treatment receipt paths are pinned as
  NOT-RUN absences in test_frozen_receipt_mutation.py; a redesign
  registration should declare NEW paths.

## Next session

Resume: this handoff -> BOARD -> RESULTS tail. Recommended order:
1. TOPSET-OVERLAP census (cheap, 3080 CPU minutes, banked spec) —
   its geometry chooses the ALTTOKEN redesign.
2. Artin picks the redesign (drop-top256 is the live candidate,
   narrow framing) -> new registration.
3. Vendor-body leg 3 pricing; homeostatic non-precision actuators
   (routing / representation restart / drift amplification) each
   need a fresh registration.
4. Phase-5 allocator counters when runtime work resumes.

## Open decisions for Artin

1. ALTTOKEN redesign pick (after the census).
2. Whether phase-5 perf forensics is worth a slot soon or parks.

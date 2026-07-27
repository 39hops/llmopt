# Relay 2026-07-27-5: asks 4-6 recorded (frontier_eval / gate_battery / certify_tables)

## Recorded deliveries (bridge INTERFACE_VERSION == 3, 14 pinned names)

1. **frontier_eval DELIVERED**: one bridge call, both directions.
   Forward: successors unverified (verify_p=0, oracle deferred per
   R0b) -> batch dead_mask -> scorer over the enumerated set ->
   mass-descending order (dead at tail) -> verify_edge over top-k
   live (verify_top_k=-1 all / 0 none). Backward mirrors through
   predecessors() (pairs forward-engine-verified by construction).
   Rows: {rule, state, sstr, dead, score, verified T/F/None}.
   Measured 18 ms forward with FULL verification on a qual-shape
   state. Scorer slot: Python callable (listwise-shaped,
   scorer(cur_sstr, [child_sstrs]) -> [mass]) until the S2 winner's
   AXNN + prompt spec ships; fallback = -hce mass; native swap =
   zero interface change.
2. **gate_battery DELIVERED**: load crystal (certification
   ENFORCED at load — uncertified artifact throws; ask 6 wired as
   a gate) -> KV-cached greedy decode -> detokenize via supplied
   token map -> oracle-verify v cur + byte-compare v expect.
   FX-V1 refactored onto a per-position stepper with per-layer KV
   caches; the stepper IS the forward (logits_q32 runs it), so
   generate() is bit-exact with full re-forward by construction —
   acceptance batteries reproduce UNCHANGED post-refactor
   (ec93e7b9058fdffb / 7dc7062f302bf0e4 = regression evidence the
   refactor changed cost, not values). ~78 ms/probe at crystal
   shape: the 120-probe nightly gate is now sub-minute; E3-class
   paired arms are one call each.
3. **certify_tables DELIVERED**: standing certificate, three
   consumers (gtest CI, gate_battery load gate, --certify for E2
   imports). Bounds: +-1 LSB v float reference (generation is
   RNE); monotonicity exactly where argmax rides on it (exp
   nondecreasing, endpoint pinned 65536; rsqrt nonincreasing; act
   tables nondecreasing x>=0, negative lobe entry-bounded — gelu/
   silu genuinely dip); rope pairs on unit circle within rounding;
   seeded midpoint fuzz v curvature bounds.
   TRAP BANKED (axiom, testing the certifier itself): the exp
   table's underflow region has EQUAL adjacent entries — a
   monotonicity-violation test corrupting there is a silent no-op;
   corrupt where the gradient is ~1 LSB (index ~1300 on the
   [-16,0] grid). House side: any fuzz WE write against FX-V1
   tables must respect this.

Fences held by axiom: no graded magic, no exact training, knock-4
[HOLD]; E4/E5 awaited house-side.

## House follow-ups (queued, ours)

- E2 export gains a hard prerequisite path: the S2 winner ships
  AXNN + PROMPT SPEC (format + token map) — frontier_eval's
  scorer slot and gate_battery's detokenizer both consume it.
  The S2 race (in flight tonight) now feeds three consumers.
- E3 shrinks: gate_battery makes the amendment arm one call per
  side. Still pre-reg'd before firing (unchanged).
- Arm-time assert everywhere: INTERFACE_VERSION == 3, 14 names.
- Next relay send: nothing new asked — house executes E1-E5 and
  reports; the S2 winner's AXNN + prompt spec is the next
  artifact crossing the bridge.

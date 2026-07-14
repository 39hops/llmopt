# GRPO v2 (verify levers) + the unified climb (syndrome-in-RL)

Provenance: 2026-07-14 evening, after run 1's cycle-2 gate went
green on every level ({13,9,5,5}@1.38 -> {15,10,6,8}@1.90 — first
uniform improvement after six reallocating SFT retrains). Artin's
GOs: verify levers 1-3; unified climb gated on the syndrome-head
A/B; the lossy-verify lever explicitly NOT pulled.

## A. Verify levers for run 2 — lossless by construction

Cycle economics (run 1): collection 48-67 min/cycle, dominated by
~1,800 oracle forks/cycle. The FA Law literal: pipeline throughput
IS verification throughput.

1. **Verify cache** by (cur, nxt): deterministic oracle => cache
   forever (the hint-memoization move). Chains revisit states; at
   ~1% validity the model re-mints popular wrong steps constantly.
2. **One fork per wave** (not per candidate): batch the deduped
   candidates into a single worker that STREAMS verdicts (magic-
   bucket pattern — a wedged candidate loses itself, not its
   wave-mates). ~6x fewer forks.
3. **Numeric-first rejection**: diff the difference, evaluate at 3
   generic points (~ms) BEFORE expand/symbolic. Asymmetric and
   sound: a valid step cannot be numerically nonzero, so the screen
   only rejects; survivors pay the full oracle before acceptance.
   Zero loss on accepts; the 99%-garbage common case gets ~50x
   cheaper.

Ship bar: parity bench — identical verdicts on a battery of known
valid/invalid pairs (corpus rows + perturbations), or no ship.

**Deliberately NOT pulled: lossy verification.** In RL the verifier
is the reward; reward noise doesn't average out, it gets optimized
TOWARD (false-accepts become the policy's favorite exploit).
Training-side oracle stays exact; the magic-style speed/accuracy
trade belongs to search-time pruning only.

## B. The unified climb (run 3, gated)

Artin: "can the syndrome training also look like our GRPO sustained
verification?" Yes — fold representation shaping into the RL update:

- The collector already visits thousands of on-policy states/cycle.
- Label their syndromes in the background (cached _hints_isolated,
  ~200ms first-look; the verify-lever cache infrastructure shares).
- GRPO update gains + lam * BCE(head(hidden_15 at state), bits) —
  same layer-15 head as the probe work (mid-network plateau,
  RESULTS layer sweep).
- Result: one sustained climb, two supervision signals — verifier
  teaches WHAT TO DO, syndromes teach WHAT TO SEE, both on the
  exact distribution the policy visits (not a frozen corpus).

GATE: the Mac syndrome-head A/B (running) isolates the aux-loss
effect in the clean SFT setting. Helps -> fold into run 3.
Nulls -> complexity saved, unified climb closed.

## C. Run-2 knobs (morning menu, with run-1 data in hand)

- cycle size (64 groups) and GATE_EVERY (2) — retune once verify
  levers change cycle wall.
- finishing bonus (+0.5 reward for Integral-free nxt) — if L2 or
  finishing sags across run 1's gates.
- lr schedule beyond halve-on-rollback; consider warmup.
- curriculum drift: collection band level mix as mix-rate shifts
  (watch: all-pass 1 -> 24 after one update; if all-pass dominates,
  levels must move up).
- collection/verify overlap (pipeline sampling of wave N+1 during
  verify of wave N) — banked, bigger refactor.

## Tracking

- Run 1 results -> RESULTS + LOOP-LOG (morning).
- Verify-lever parity bench -> scripts/bench_verify_fast.py.
- This spec supersedes nothing; extends 2026-07-14-step-grpo-design.

# Handoff 2026-08-01-3 — P4 device leg PASS + the brute leg closes

Artin's GOs this leg: "go parallel" (P4 on the 3080). The brute
leg (queue item 2, Mac) ran on the standing Mac authorization.

## Verdicts (RESULTS tail, in order)

1. **AMENDMENT PARAMS-GRAVMOE**: battery model is 208,192 params,
   not "60k" (logs govern; readings unchanged).
2. **VERDICT GRAVMOE-P4-DEVICE**: ALL 16 pinned arms sha-identical
   on the 3080 (parallel, 6m28s wall, rc=0, self-verifying table,
   remote HEAD hash-asserted). Every gravmoe verdict this week is
   now device-free. Axiom lab leg unblocked: artifacts at
   scratch/detbwd_gmoe_ref/ (4 contracts: init+windows+contract
   JSON with per-family draw bounds, + pins.json all 16 arms);
   relay -5 staged with postscript — Artin carries it.
3. **VERDICT GRAVMOE-BRUTE (leg 1)**: steps arm diverges at const
   lr (MB-S12 shape at 8k steps); width arm voided by its own
   clamp fence (0.231).
4. **VERDICT GRAVMOE-BRUTE-B/C (partial)**: decay legalizes the
   steps axis (S4D: loss tamed, 2/8 flat, tok 112/140 — +18 tok,
   zero new solves); width found an INSTRUMENT LAW — ACT_CLAMP is
   rms_fwd's int64 overflow guard (m40 = (s2//D)<<32 overflows at
   mean-sq > 2^31, rms ~90 Q-units; W4c crashed rdiv-by-zero at
   ACLAMP=49152). Safe bound ~80 Q-units; clamp demand is
   MEASURED (STEPS=1 init sweep), never scaled.
5. **VERDICT GRAVMOE-BRUTE (closing)**: brute does NOT convert —
   the legal width arm (W4d, healthy, decay, overflow-safe) lands
   0/8 tok 56/140, UNDER the 208k baseline. 8 rows x 2000 steps
   is DATA-BOUND: 4x params memorize slower. G-RB1 stands as the
   battery optimum; SS revives only where the data bound lifts.

## Files/knobs

- detbwd_r2b.py: DIM/DHEAD/FFN + ACLAMP env knobs (defaults
  unchanged; G-RB1 sha 1fcfd187 reproduced 3x as regression).
- detbwd_gravmoe.py: SCHED=1 (mb integer lr decay at quarter
  points), EXPORT mode (P4 artifacts), D/F inherited from r2b.
- scratch/p4_arms_0801.sh (16-arm parallel, self-verifying),
  brute_arms/brute_b/brute_c runners; logs/brute_*_0801.log,
  3080: logs/p4_arms_0801.log + logs/p4/.

## Queue (next session)

1. Axiom lab leg receipt (their reproduction of the gmoe pins);
   window-cycling engine surface is the ask in relay -5.
2. COND+QK graduation: seeds/windows ladder + WIDER DIET (the
   heldout-solve rung) — the closing verdict says data, not
   compute, is the binder.
3. rms_fwd headroom rung (pre-shift s2 before <<32, or 128-bit)
   — spec first; only if width ever graduates.
4. Carried: diet meter, ckpt forensics [Artin GO], FOURIER-4,
   exact-manipulation diet share.

## Fences

- All brute numbers n=1; W4b/W4c form an accidental clamp-dose
  pair, not a controlled one.
- Width arms change the draw stream (protocol-paired only).
- Gate cells still not loss-comparable to truncated cells.

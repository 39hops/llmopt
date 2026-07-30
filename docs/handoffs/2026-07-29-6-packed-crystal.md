# Handoff 2026-07-29-6 — THE PACKED CRYSTAL: C-series complete in one evening

Spec 2026-07-30-packed-crystal ran END TO END (C0-C6 + two
rescue arms), Mac + 3080 (Artin's GO; box revived after
restart — WSL mirrored-mode Hyper-V firewall was the block;
`Set-NetFirewallHyperVVMSetting ... -DefaultInboundAction
Allow` + profile Private is the recipe, memory-worthy).

## Verdicts (all in RESULTS, newest ~8 entries)
- C0+C1: zero-tax pack (L4d56 -1, d64h8 EXACT parity);
  entropy within 1% of Gaussian capacity; 6.15-6.65x v fp32.
- C3: honest table — GPTQ/AWQ/HQQ with real Hessian gain
  NOTHING over closed form at matched 5 bits; 3-bit solves
  flat (falsified — d64h8 k_c small), damage in valid%/KL.
- C5: nested tiered artifact real (5.7x; escalation -15%
  bytes/solve) BUT zero-tax fails on the joint-STE matryoshka
  crystal (~1-2 sigma per tier) — pack EMA parents.
- C2/C2b: fused GEMV 1.76x; bit-packed 5-bit GEMV 2.39x,
  BEATS byte-aligned — the disk format IS the runtime format.
- C4: integer-GEMM hash IDENTICAL Mac/cuda (fda95457); fp
  logits differ; greedy streams match anyway. Claim 3 lands.
- C6/C6b/C6c: per-tensor sigma 33x worse than HQQ on Qwen
  (falsifier fired) -> per-row null -> step sigma/8 recovers
  11.6x. Fence is MECHANISTIC: sigma-grids optimal for
  at-capacity weights (crystals); web tails reward
  max-anchored grids (fixed-width bits priced by worst
  outlier).

## Paper shape (publication candidate, honest)
1. Calibration-free sigma-allocation optimal for at-capacity
   weights; born crystals measured AT capacity (<1% gap).
2. Web LLMs are not at capacity -> use rtn/hqq there (our own
   honest table says so). 3. Universal: entropy-bound pack,
   bit-packed kernel wins, cross-device integer determinism,
   tiered bytes. 4. Wall-time: 0.9s v 61.7s HQQ at 0.5B.

## Banked follow-ups
- Entropy-coded sigma-grids on heavy tails (the fixed-width
  penalty was the C6c killer; C1 showed deflate recovers it).
- k_c flips-meter as pre-pack check (C5's miss was
  predictable); fit f + k_c jointly across all snap cells.
- Full integer end-to-end decode (norms/softmax fp between
  exact GEMMs) -> device-fence-free packed inference.
- Training lens rider still unfired (optional on any birth).

## State
- Instruments: scratch/pack_crystal.py, pack_baselines.py,
  pack_tiered.py, pack_gemv.py (+crystal5), pack_determinism.py,
  pack_c6.py (ARM/K knobs). All __main__-guarded.
- Artifacts: checkpoints/packed_{L4d56,d64h8}.npz (untracked).
- 3080: has h8_ema ckpt (scp'd); stash@{0} preserved
  (pre-existing WIP, NOT dropped — inspect before next sync).
- Queue next: diet-evolution spec; multi-ply farmer tree;
  axiom step tranche -> LLMUE resolver.

## Addendum 1 (late night): R-pass + capacity meter + C7
- R-pass: every C-series arm replicated (C1 exact; C4 hash-A
  invariance on a SECOND activation seed, Mac=cuda 7f6849f7;
  C6c to printed precision; kernels within 2%, micro shapes
  fenced as parity-within-noise). No amendments.
- Corpus re-sweep tie-ins: Lloyd-Max race = at-capacity law
  from the codebook side; experts-are-crystals extends to
  capacity axis; "quantize the notches" framing.
- CAPACITY METER shipped (scratch/capacity_meter.py): crystals
  0.96-1.61 bits / DeepSeek-V3 experts 2.33 @ kurt 3.07 /
  Qwen 3.62 @ 5.29 / SmolLM 3.85 @ 6.54. Decision rule booked.
- C7 ARMED (not yet pre-reg'd): sigma-pack MoE routed experts
  v HQQ per-expert — candidate: a small open MoE end-to-end
  (Qwen1.5-MoE / OLMoE class) or V3 shard desk-DeltaKL.
- README Highlights + design note updated; THEORY row added;
  RIFF banked. All pushed.

## Addendum 2 (close): C7 verdict + the dial
- C7 (OLMoE): strong transport FAILS (16x hqq on experts) but
  the meter becomes a CONTINUOUS DIAL — premium monotone in M
  (6 points, crystals 1x -> Qwen 34x); meter warned pre-run.
  hqq 675.5s v sigma 16.6s on 6.4B (the wall-time exhibit).
- Sharpened rule: sigma-law below M~2. NNUE (M 0.82) is a
  member; DeepSeek experts (2.33) between bands, unmeasured.
- Banked this evening: entangled-experts MI/merge cell,
  tied-expert ladder, superposition frame (already measured:
  R0b + duo-wave + dist readout), NNUE pack (deterministic
  search heuristic), area-law probe sketch.
- Next: P2a hybrid allocator (meter-routed, zero calibration)
  -> P5 retrodiction card -> P3; P4 on nightly GO.

## Addendum 3 (close): the dial day ends in BLACK HOLE MoEs
- P2a-v2: falsifier fired (clip k=4 ppl 138,890) — outliers
  load-bearing at full magnitude; P2 CLOSED; domain = M<~2.
- EXPERT-SCALE LAW (Artin's prediction, confirmed n=3):
  M monotone in expert fineness — OLMoE 64exp 2.85 / V3
  256exp 2.33 / Kimi-K2 384exp 2.01 (AT the boundary).
  Transport claim revives as a scaling law.
- NNUE metered: M 0.82 (deepest crystal-band); per-layer
  oligarchy gradient visible from disk (2.19/4.78/7.03).
- NEW SPEC: 2026-07-30-blackhole-moes.md (B0 atlas + B1
  streaming dial-pack of Qwen3-30B-A3B + B2 function checks
  + B3 K2 depth + B4 entangled-experts MI on OLMoE + B5
  paper fold-in). B0-B2 pass LAUNCHED at close (bg, ~2h,
  logs/blackhole_b0.log + atlas jsonl).
- Banked riffs this evening: black-hole model title;
  superposition frame (already measured x3); tied-expert
  ladder; cosmology bank with literal fences.

## Addendum 4 (07-30 ~01:30): black-hole MoEs night
- B0-B2 booked: capacity atlas of Qwen3-30B (30.2B, 136.6
  min, 21GB artifact); routers incompressible (M 4.45);
  up_proj IN sigma domain (1.93); mid-stack dip L8-16;
  LADDER VARIABLE = per-expert SIZE (Artin, twice).
- N3: experts decorrelated to zero — replication only at
  birth. N2: dial-pack 3.8x over sigma-pure, rtn still wins
  on KL (clarified: zero-tax needs knee-slack metric).
- N1 IN FLIGHT (3080): seed-2 births of both C1 crystals
  (marker logs/night30_done.marker). Morning: scp EMAs to
  Mac, pack_crystal on both -> C1 at n=2 (paper cell).
- Queue: B3 (K2 depth, 2 shards, then DELETE K2 shards),
  B4 (entangled-experts MI on OLMoE), B5/P7 paper skeleton,
  P5 pre-deploy card. K3 excluded (MXFP4). Disk: 21GB parts
  + 17GB K2 shard — clean after B3.

## Addendum 5 (07-30 ~02:15): night-30b mac chain booked
- B3: K2 depth FLAT (2.04-2.09) — black-hole state uniform;
  Qwen3 dip is recipe-local. K2 shards deleted (disk freed).
- B4: THE SPLIT LAW — co-routing MI 0.21-0.46 bits (~300-500x
  shuffle; Artin's links confirmed ROUTING-side) but top-pair
  merge costs +3.4 ppl (not weight-side). Lever = prefetch/
  placement/cache, not merge.
- P6: entropy coding worth 1.29 bits/wt on experts (21GB ->
  ~16.3GB, 3.7x v bf16); routers ironically most codable.
- 3080 overnight: night30 (seed-2) -> night30b (seed-3)
  markers pending. MORNING: scp EMAs, pack_crystal both,
  C1 at n=3 -> paper skeleton (P7/B5) -> P5 -> P3.

## Addendum 6 (07-30 morning): n=3 + P5 card
- C1 n=3: h8 parity REPLICATES (+2/-3/0 — headline solid);
  L4 floor pays -5 at seeds 2/3 (-1 at seed 1) — fragility
  axis surfaced. Entropy 3.10-3.13 v cap ~3.13 at n=3 both
  archs.
- P5 retrodiction: flips card predicts at CLASS level (L4
  1.3x h8, correctly ranked) but not seed level; card ships
  as class gate. Follow-up named: matched-operator probe.
- 3080 overnight arms crashed (friendly-fire #8 missing dep,
  #9 stale marker) — births recovered on Mac (better fence).
- Paper skeleton drafted: docs/paper-draft-entropy-bound.md.
  Checklist remaining: rANS coder, P3 deterministic decode,
  related work. practice_14 left for Artin (rank/nullspace).

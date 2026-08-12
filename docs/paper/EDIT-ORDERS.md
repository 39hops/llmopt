# Edit orders for the LaTeX assembly (verified 2026-08-11 night)
Source: 5-seat claims verification vs RESULTS, spot-checked by session model.
Apply ALL to the prose of docs/paper-prose-v1.md as it is converted.

## DEAD (rewrite required)
- L321-326: "sparse assignment creates the split" RETRACTED by UMOE-2 +
  P-INIT-DEFAULT-HOLDS (n=3, RESULTS L11896/L22732). Live mechanism:
  INIT DEFAULT PRESERVED (independent parameterization starts orthogonal,
  corr 0.0016; no routing regime supplies a force toward correlation).
- L259-261: C4 books ONE hash, one seed; "two independent seeds" false —
  two hash CHANNELS (A integer / B fp). L10657-74.
- L96-98: sigma/2 is NOT "coarsest step keeping parity" — knee is
  0.5-1.0 sigma; 3-bit arms flat on solves. L10465/L9809.
- L21: Lloyd-Max clause has NO arm receipt (pre-reg + retrospective
  one-liner only, L4715/L10829). CUT from abstract.

## FIX (value corrections)
- L191: 41x -> 69x for 0.5B (0.9s v 61.7s, L10684); 41x is the 6.4B pair.
- L189-193: timings come from C6/C7 where transport FAILED; where parity
  holds, wall-time is trivial. Rewrite honestly (wall-time = calibration
  time, L10477-80).
- L151: M 2.78 -> 2.85 (L10890).
- L303: 0.995/corr is Qwen3-30B not OLMoE; corr 0.0024/0.0054 (L11109).
- L305: merged pair = top-MI co-routed, not "most similar" (L11197).
- L307: MI is WITHIN-layer, not adjacent-layer (L11168).
- L186-188: emb/head stay fp32 in the booked artifact; sigma/8 knob
  changed nothing (L10381/L11373). Remove.
- L216-218: 0.91x retired as noise (re-read 1.03x, micro fenced
  parity-within-noise, L10850); crystal5 1.07-1.21x. "first attempts
  lost" belongs to int4_gemv only (L1193).
- L75/L195: "matches exactly" -> exact at seed 1; n=3 = +2/-3/0, all
  sub-sigma; say "parity within sigma 3.5" (L11234).
- L234-235: streams within 0.5% of entropy, tables add ~2%; d64h8
  3.179 v 3.12 = +1.9% (L11300). No 0.1% claim.
- L163-166: expert-capacity NOT monotone (45M->2.33 but 40M->2.01,
  L11058/L11445); state the tension.
- L167: 896 experts PER LAYER x 93 layers (L11125).
- L170-172: MXFP4 leaves ~9% lossless margin (inverted as written, L11450).
- L157-159: RTN wins among ZERO-CALIBRATION methods; HQQ 0.0044 beats
  RTN 0.0097 (L11136).
- L383-385: two experts METERED, one hash-locked at GEMV+chain (L11443).
- L377: 2x device dependence is the L9 PROBE; gates measured -1
  (L3213/L4740). Fix attribution.
- L86-88: N3 was a CONFIRMED prediction, not a falsification (L11114).
- L26/139: one web-dense premium measured (Qwen 3.62 -> 34x); SmolLM2
  3.85 has NO measured premium; range 3.62-3.85.
- L63-65: FIVE production families metered (add Qwen3-30B).
- L136-137: crystals measured 0.96-1.61; "NNUE 0.82" is an in-passing
  C7 assertion — keep only with that caveat (L10917).
- L29: 2.39x is best-of-five shapes on SYNTHETIC weights (L10593/L10850).
- L31-34/275: cross-LAB covers P3 only; K3/MXFP4 is house-only three
  backends, axiom leg MPS-only (L11480/L11515). "Decode" -> expert
  forward (L11433).
- L35/392: "all claims pre-registered" -> "every experiment
  pre-registered; exploratory reads labelled" (L11556).
- L374-376: DeepSeek-V3 is ALSO a dequantized read — add to fence (L10819).
- L402-405: ONE booked contamination incident in RESULTS (Holdout v1,
  281 collisions, L3274); the other two live in CLAUDE.md — say so.
- L237: Qwen3-30B roundtrip verified on first 2B symbols (L11309).
- L110-114: symbol-level only; entropy coding took 1.3 bits/wt after
  (L11297).
- L330-331: tied-at-birth booked SPLIT (L11608).
- L318-320: seed-2 MI ordering FLIPS (205x/299x, L11630); drop the
  seed-1-only contrast or show both.
- L344: give SmolLM2-1.7B 6-bit baseline (rtn 62.98) beside 138,890.
- L340/363: cut unverifiable literature negatives ("no prior...") or
  mark as "we are not aware of".
- L358-359: REAP = bibliography-only; merge probe n=1 — soften.

## Statistical register (global)
"tie"/"exactly" only where receipts support it; sub-sigma deltas are
underpowered nulls. Single-seed claims carry their fence inline.

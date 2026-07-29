# Handoff 2026-07-29-1 — the width-floor arc (minimal-crystal Leg A)

Resume: BOARD -> this handoff -> spec 2026-07-29-attention-core
-> RESULTS tail (L9230+). Nothing in flight; 3080 quiet, Mac open.

## The arc's verdicts (all pre-registered, all booked)

1. A0: EMA redundant under production warmup+cosine (59=59) —
   LAW: averaging and annealing are ONE lever; the EMA wins
   (+7-12) are constant-LR phenomena (substitution, not addition).
2. WIDTH FLOOR at d56: raw 54/EMA 63 = d64 line at 0.77x params;
   cliff SHARP in (48,56] (d48: 44/50). REPLICATED n=3 on cuda
   (d56 58.3 v d64 58.7 EMA means; paired, device-internal).
3. SPECTRUM ON FLOOR: band masses dimension-proportional (law
   holds); free-compression point VANISHES (top-3 = -9 v d256's
   -2) — frequency-holography is a SLACK phenomenon.
4. MATRYOSHKA AT FLOOR: tier competes (63->57 buys a 52/120
   eighth-tier); d256 zero-price was slack. UNIFIED LAW: bits x
   sharing x width x tiers draw on ONE slack pool.
   Striking: the projected eighth-tier (52) BEATS native d48 (44).
5. NO FFN CLIFF: ffn 224->48 EMA 58...54 (-4 over 4.7x; even
   inverted SwiGLU ffn48<d56 works) v -13 for 8 attention dims.
   THE CRYSTAL IS AN ATTENTION MACHINE; MLP ~ pure slack.
   Params-per-solve leader: d56/f48 EMA (~1/3 d64 params, -1 sigma).

## Instruments
sym_birth.py grew SCHED=onecycle (production-faithful), SEED,
env dims; sym_spectrum.py + matryoshka_r1.py env-parameterized
(CKPT/D/FFN/BS/OUT). night29.sh / night29b.sh = the cuda
batteries. Checkpoints: sym_birth_dense_w56{,_ema}.pt,
matryoshka_d56.pt (Mac); cu-series on WSL.

## Banked riffs (RIFF-LEDGER tail, Artin 2026-07-29)
Runtime compression dial (3-axis; difficulty-gated tier cell);
bits-as-portfolio (polar-split snap cell); multi-axis
quantization inventory; THE ADAPTIVE CRYSTAL synthesis
(tier-retry = cheapest controller, oracle-fail as free
difficulty signal); reverse-LLM alphabet (finite rule set);
infinite-density (Q-dense snap / delegation escape).

## Next (spec 2026-07-29-attention-core)
Tier-retry desk cell -> attention anatomy (desk) -> attention
compression arms -> axiom-on-Mac onboarding (Fable-5 subagent
sanctioned by Artin for ~/code/axiom). Then Leg B call-spans,
Leg C diet descent.

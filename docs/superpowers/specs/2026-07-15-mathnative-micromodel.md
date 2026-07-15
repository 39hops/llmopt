# Math-native micro-model — from-scratch, closed-system only

Small decoder-only LMs trained exclusively on oracle-verified
derivation steps. No pretraining. Provenance: Artin's
purely-on-rule-bits push + the bad-habits mechanism (2026-07-15):
the anatomy showed RL cannot uproot incumbent preferences (CKA
0.9998; the heartbreaker sign was a measured habit-fight), so a
model with NO incumbents trains on undisputed territory. Tests
priors-vs-drag falsifiably and, if it works, gives the climb a
20-100x cheaper native engine.

## Design

**Tokenizer** (~400 tokens, hand-built, deterministic): digits,
operator/paren/space/newline chars, variable x, multi-char atoms
(sin cos tan exp log atan asin sqrt pi E Integral), the scaffold
tokens (Current: / Hints: / none / Step: / =>), integers beyond one
digit as digit sequences. Built from the corpus charset — the
charset mask IS the vocabulary, made honest.

**Architecture**: standard decoder (RMSNorm, rotary, SwiGLU,
untied head): d=384, 8 layers, 6 heads, ffn 1536, ctx 512 ->
~19M params. Torch; trains on MPS, samples anywhere.

**Phase 0 — farm** (Mac CPU, parallel): engine-minted chains at
L1-5, forward + reverse, target 100k+ verified pairs (current
corpus 10.2k is a seed, not a diet — from-scratch must learn
arithmetic itself: 18/2=9 arrives only through examples).
Fresh seed band 50M+. Streamed, fork-walled, as always.

**Phase 1 — LM pretraining on chains**: full-sequence loss on the
step format (this text is its entire world). Gate: >=1% step
validity at L2-3, temp 0.7 — the 0.5B's historical starting point.
If unreachable at 100k pairs: the priors were load-bearing; book
the price tag and the null.

**Phase 2 — GRPO from birth**: the existing driver verbatim
(FAST_VERIFY on), frontier band starting L1-3. Gate ladder
unchanged. Bars: two consecutive green gates = the climb works on
a from-scratch substrate; sampling throughput >=10x the 0.5B
(measured, it should be ~30x from params + vocab).

**Kill switch**: phase 1 validity <0.1% after the full farm, or
2k GRPO groups with no green gate -> HALT, book, and the 0.5B
keeps the crown with a measured moat.

## What each outcome buys

- WORKS: cheaper climbing (params AND vocab), no habit tax, a
  clean substrate for the anatomy work (crystal formation from
  RANDOM INIT — the evolution movie from frame zero), and the
  micro-model becomes the test-bed for ES-LoRA and rank-matched
  climbing at 30x iteration speed.
- FAILS: the first quantitative price of pretraining priors for
  symbolic manipulation at this scale — publishable-grade either
  way, and it sharpens what the 0.5B is actually contributing.

## Non-goals

General language. Chat. Anything outside the closed system. These
are small special-purpose math models — nothing more.

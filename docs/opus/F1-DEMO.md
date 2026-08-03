# V4-Flash on a MacBook — the F1 demo sheet (2026-08-03)

**DeepSeek-V4-Flash — a 304-billion-parameter, 166.9 GB frontier MoE —
generated text on a 36 GB Mac this morning**, running the vendor's own
unmodified `model.py` over a ~450-line pure-torch kernel twin (half of it
acceptance bars), with 7.1% of its routed experts resident.

## The numbers

One config per column (an earlier revision mixed three runs — reviewer
catch, AMENDMENT F1-REVIEW):

| | first demo (V4-F1d) | fastest (V4-F1e arm 5) |
|---|---|---|
| dense path | 8.85 GB packed fp8 | dequanted bf16 (~15 GB) |
| experts resident | 785 (10.5 GB packed fp4) | same |
| decode | 0.100 tok/s | **0.268 tok/s** |
| prefill (9 tok) | 30.6 s | 58.8 s |
| Metal / RSS | ~20 GB / 5.3 GB | ~26 GB / 6.2 GB |
| output | prompt-echo loop | bit-identical to the first |

Both greedy, batch 1, same prompt; masking = **one write into the
aux-loss-free bias**; ~19 GB of byte-range fetches, never the 166.9 GB
artifact.

## The text (verbatim, prompt: "The three most important ideas in computer science are")

> the three most important ideas in computer science are the three most
> important ideas in computer science are the three most important ideas
> in computer science are …

This is the **registered prediction firing**, not a surprise: 16-of-256
experts per layer is far below the pruning cliff, and the pre-registration
said "degraded text is the EXPECTED outcome." What's scientifically fun is
*how* it degrades: the model doesn't emit noise — it **copies its context,
fluently**. The attention stack and dense path (kept whole) carry the
copy/induction machinery; content generation lives in the routed experts
we amputated. The output is the architecture showing which organs it
still has.

## How it was done (four pre-registered rungs, one morning)

1. **F1a — the twin.** The vendor's only GPU dependency is six tilelang
   kernels. Pure-torch twins, 22/22 acceptance bars: fp4 decode
   bit-identical to the lab's certified decoder, every gemm within 1/128
   of an exact fp64 reference, RNE rounding matched against torch's own
   e4m3 cast over all 256 values including every tie.
2. **F1b — the boot.** Vendor model.py runs unmodified (cpu + mps) on a
   truncated random-weight config exercising every path: hash routing,
   score routing, windowed + compressed attention, Sinkhorn
   hyper-connections. Cross-device divergence at random weights was
   measured (~3×/layer growth) and correctly predicted to vanish with…
3. **F1c — real weights.** Layers 0–2 with exact hash-routed expert
   demand: all bars pass, the cross-device gap collapses 56–297×
   depth-matched (~2.5 orders of magnitude), expert output within
   1.5e-6 of the exact reference.
4. **F1d — generate.** Everything above, at full depth, on real weights.

## Why this was possible at all

- The **format work**: the lab had already proven the shipped fp4 bytes
  decode exactly (byte-identical to its own K3 format) and runs an
  expert exactly in integers on three backends — so the twin had an
  oracle, not a hope.
- The **census**: a header-only sweep found the "27 GB dense path" was
  actually 8.85 GB (19 B of the difference was mislabeled MTP experts) —
  which is the fact that made residency feasible.
- The **bias inversion work**: DeepSeek's aux-loss-free balancing
  separates selection from output weighting, which makes the trained
  bias both a *readable load record* (used to pick residents) and a
  *writable routing mask* (used to fence them) — zero vendor-code
  changes.

## F1e, run (same day)

Four optimization arms, honestly scored: bf16-dequanting the dense path
bought **nothing** (it was never the wall — the profile said 84% of
decode was the experts' per-use fp4 unpack); a bounded expert cache made
things **5× worse** (774-tensor per-token working set vs a 180 cap =
pure thrash); a **pair-LUT unpack** bought **2.2×** for zero memory
(0.100 → 0.219 tok/s, output bit-identical); bit-exact **batched expert
calls** (equivalence gate 0.00e+00) added 1.22× more to **0.268 tok/s**;
and at K=24 — *in the bf16-dense config; the fp8-dense cell is unrun* —
the repetition attractor **held** while Metal crossed its ceiling and
throughput collapsed ~8× (paging inferred from decay, not instrumented).
The localized remaining wall is unpack VOLUME per token, which neither
launches, caching, nor batching reduce — banked next: prompt-lookup
speculative decoding (already in-tree, and an echo loop accepts ~every
draft), plus the unmasked-router "expert recall" instrument that
separates amputation from mask bugs. No capability claims anywhere —
this sheet describes a measured system and its honest output.

*Everything above is booked in `docs/RESULTS.md` (PRE-REG V4-F1 →
VERDICTs V4-F1a/b/c/d → PRE-REG V4-F1e + riders → VERDICTs V4-F1e +
ARM5 → AMENDMENT F1-REVIEW, which corrects this sheet's own first
revision) with scripts in `scratch/v4flash_*.py` and raw rows in
`logs/opus/v4_f1d.jsonl`.*

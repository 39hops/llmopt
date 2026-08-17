# QWEN-MODEL-1 functional evaluation — FROZEN PRE-ARTIFACT

Committed 2026-08-17, BEFORE any compressed Qwen artifact exists
(the compressor has not been pre-registered yet). Freezing the
test before the model exists is the point: tensor-rate decisions
in QWEN-WHOLE-0T may never be tuned against these prompts'
observed outputs.

Subject pair: vendor Qwen3.8-27B at revision 1d4bf0f2 (text tower)
v the compressed artifact. Same tokenizer, same chat template,
greedy decoding unless a metric says otherwise.

## Quantitative core (teacher-forced — robust to early divergence)
1. Teacher-forced cross-entropy / perplexity on corpus.txt
   (fixed token corpus, this directory), per-position mean and
   distribution.
2. Next-token top-1 agreement and full-softmax KL at every
   position of the prefixes in prefixes.jsonl.
3. Per-layer-family attribution when damage appears: re-run 1-2
   with ONE family decompressed to native at a time (the
   functional analogue of the ladder).

## Behavioral checks (free generation — "does it talk")
4. Greedy generation on prompts.jsonl (all categories), 256
   tokens: exact-prefix agreement length v vendor, then
   side-by-side human-readable pairs booked verbatim:
       PROMPT / vendor: / compressed: / top1-agreement / KL
5. Degeneration battery: repetition (max n-gram loop length),
   EOS behavior (stops when vendor stops +-32 tokens),
   non-language garbage (non-UTF or vocabulary-tail flooding).

## Rules
- The quantitative CORE is 1-2; free generation is the sanity
  read, not the score.
- No prompt may be added, removed, or reworded after the first
  compressed artifact exists; extensions form a SECOND versioned
  set (prompts_v2) that books separately.
- Every reported number carries the artifact hash and the eval
  commit.

## Teacher-baseline procedure (frozen with the eval, added 2026-08-17
## pre-artifact — the vendor model cannot run in the 10GB target env)

Reference logits are produced ONCE by a deliberately slow
layer-streaming CPU pass of the VENDOR artifact (revision 1d4bf0f2,
bf16 decoded to fp32 per layer, bounded residency — the same
streaming machinery as the compressor, run uncompressed), over
exactly: corpus.txt token positions, every prefixes.jsonl position,
and the prompts.jsonl greedy rollouts. The resulting logit/token
records are hashed and LOCKED as the immutable teacher; every later
comparison (compressed CPU reference, Metal W4, CUDA W4) scores
against these frozen records, never against a re-run vendor pass.
One baseline, computed once, at whatever wall-clock it costs.

## Runtime ladder the scores attach to (registered order)
  QWEN-RUNTIME-0R  portable slow CPU decode reference (mmap the
                   artifact, decode per-op, release) — the
                   "does it talk" oracle
  Metal direct-W4  primary Mac performance leg (unified memory;
                   llmopt/kernels/metal.py lineage, NOT a new
                   MLX/torch-mps prototype stack)
  CUDA direct-W4   primary 3080 performance leg (hard 10GB
                   residency)
Per-backend reporting: tok/s, peak RSS/VRAM, and EFFECTIVE
COMPRESSED-WEIGHT BANDWIDTH = compressed bytes touched per token x
tokens/s — separates memory-bound from lookup-bound from
implementation-bound.
Numerical-backend rule: default arithmetic is W4 + ordinary fp32
accumulation everywhere; the compression error floor (~0.34
relative weight-space) dwarfs rounding. Precision escalation only
if a backend's KL materially diverges from the CPU reference —
then the EXISTING exactness lineage applies (higher accumulator ->
exact integer carrier -> tiled exact_gemm / fp32-limb Metal), as
oracle and repair, never as baseline. No new precision science
rung: Ozaki-class exactness is CLOSED on CUDA; its Metal leg
(tiling/fp32-limb) is the one legitimate build, and only when this
runtime gives it a real consumer.

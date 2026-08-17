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

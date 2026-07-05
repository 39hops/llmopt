"""llmopt: inference + training optimization library.

Subpackages:
  decoding/  - generation-time: prompt-lookup, speculative; roadmap: sampler
               pipeline, constrained/FSM decoding, search, Medusa/tree verify,
               chunked prefill + continuous batching
  cache/     - KV cache: radix prefix tree; roadmap: paged blocks, KV quant,
               sink/H2O/SnapKV eviction, sliding window
  quantize/  - per-layer sensitivity + bit allocator; roadmap: GPTQ/AWQ/HQQ,
               pruning, 2:4 sparsity, low-rank SVD
  train/     - batched ref-logprob precompute; roadmap: LoRA family, sequence
               packing, DPO/IPO/KTO/ORPO/SimPO/GRPO objectives, optimizers
  eval/      - perplexity, tokens/sec bench; roadmap: pass@k, calibration,
               bootstrap CIs, numerical-equivalence harness
  context/   - roadmap: RoPE scaling (PI/NTK/YaRN), attention sinks, RULER eval
  internals/ - roadmap: logit lens, attention entropy, activation stats, CKA
"""

from llmopt.cache.radix import RadixCache
from llmopt.decoding.prompt_lookup import find_ngram_continuation
from llmopt.quantize.allocator import allocate_bits, pareto_front

__all__ = [
    "find_ngram_continuation",
    "RadixCache",
    "allocate_bits",
    "pareto_front",
]

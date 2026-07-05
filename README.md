# llmopt

Inference + training optimization library.

| Subpackage | Implemented | Roadmap |
|---|---|---|
| `decoding/` | prompt-lookup, speculative (greedy + rejection sampling) | sampler pipeline (top-k/p, min-p, DRY, mirostat), constrained/FSM decoding, beam/best-of-N/self-consistency, Medusa/tree verify, chunked prefill + continuous batching |
| `cache/` | radix prefix KV tree w/ LRU | paged blocks, KV int8/int4 quant, sinks/H2O/SnapKV eviction, sliding window |
| `quantize/` | per-layer ΔKL sensitivity (fake-quant), min-memory bit allocator, Pareto sweep | GPTQ/AWQ/HQQ, pruning, 2:4 sparsity, low-rank SVD |
| `train/` | batched ref-logprob precompute + disk cache | LoRA family, sequence packing, DPO/IPO/KTO/ORPO/SimPO/GRPO, optimizers/schedules |
| `eval/` | perplexity, tokens/sec bench | pass@k, calibration (ECE), bootstrap CIs, numerical-equivalence harness, TTFT/TPOT |
| `context/` | — | RoPE scaling (PI/NTK/YaRN), attention sinks, RULER-style eval |
| `internals/` | — | logit lens, attention entropy, activation stats, CKA |

Ref logprobs stored as top-k + tail mass (full-vocab for 164×1k×150k would be ~100 GB).

## Install

```bash
pip install -e ".[dev]"
# GPU torch (CPU wheel installed by default):
# pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.float16).cuda()

# prompt-lookup decoding
from llmopt.decoding.prompt_lookup import generate_with_prompt_lookup
ids = tok("Summarize: ...", return_tensors="pt").input_ids
tokens, stats = generate_with_prompt_lookup(model, ids, max_new_tokens=128)
print(stats)  # forward_passes << 128 on input-grounded tasks

# ref logprobs (e.g. HumanEval's 164 prompts) cached to disk
from llmopt.train.ref_logprobs import precompute_ref_logprobs
seqs = [tok(p).input_ids for p in prompts]
refs = precompute_ref_logprobs(model, seqs, model_name="qwen2.5-0.5b",
                               cache_dir=".refcache", batch_size=8)

# sensitivity -> allocator -> Pareto
from llmopt.quantize.sensitivity import measure_sensitivity
from llmopt.quantize.allocator import allocate_bits, pareto_front
sens = measure_sensitivity(model, seqs, refs, bit_widths=(2, 4))
cfg = allocate_bits(sens, kl_budget=0.05)
print(cfg.avg_bits, cfg.bits_by_layer)
```

## Benchmarks

Qwen2.5-0.5B fp16, single prompt, greedy, RTX GPU (`scripts/bench_lookup_static.py`):

| Config | tok/s | Speedup |
|---|---|---|
| static cache, eager | 23.3 | 1.0x |
| vanilla decode + CUDA graphs | 71.2 | 3.1x |
| prompt-lookup + CUDA graphs (fixed [1,11] verify blocks) | 141.3 | 6.1x |

Lookup run: 64 forward passes, 86/382 draft tokens accepted. All configs greedy-equivalent to eager baseline.

## Tests

```bash
pytest  # pure-Python parts (lookup, radix, allocator) run without GPU
```

## Design notes

- Speculative/lookup loops recompute KV each step: correct and simple; KV reuse is the radix module's job, integration is a later phase.
- Allocator assumes ΔKL additivity across layers — approximate; always re-measure the final mixed config end-to-end.
- Fake-quant measures quality only. Real memory savings need bitpacked kernels (HQQ/GPTQ) at deploy time.

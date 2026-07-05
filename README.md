# llmopt

LLM inference + training optimization lab. Small, readable implementations of the tricks that make local inference fast — each benchmarked and verified greedy-equivalent against the eager baseline.

## Highlights

Qwen2.5-3B-Instruct fp16, single prompt, greedy, 150 new tokens (`scripts/bench_lookup_static.py`):

| Config | tok/s | Speedup |
|---|---|---|
| static KV cache, eager | 23.3 | 1.0x |
| + CUDA graphs (`torch.compile` reduce-overhead) | 71.2 | 3.1x |
| + prompt-lookup w/ fixed-shape verify blocks | 150.1 | 6.4x |
| + longer draft (num_draft=16, max_ngram=4) | 156.7 | 6.7x |

All configs produce token-identical output to eager greedy decoding.

Hyperparameter sweep (`scripts/sweep_lookup.py`): draft length dominates, ngram size barely matters. Under CUDA graphs a rejected draft token costs almost nothing, so longer verify blocks win even at ~15% accept rates.

MLX (Apple silicon), Qwen2.5-3B-Instruct 4-bit, same prompt (`scripts/sweep_lookup_mlx.py`): greedy 56.9 tok/s → prompt-lookup 63.1 tok/s (1.1x, num_draft=5). The same generic loop runs unchanged and stays token-identical; the gain is smaller because MLX has no per-step launch overhead for drafting to amortize, so shorter drafts win.

## What's inside

| Subpackage | Implemented | Roadmap |
|---|---|---|
| `decoding/` | prompt-lookup, speculative (greedy + rejection sampling), backend-agnostic lookup loop, sampler pipeline (top-k/p, min-p, DRY, mirostat v2), regex-constrained FSM decoding, tree verify (multi-candidate lookup drafts, tree attention), Medusa heads (Medusa-1 training + tree-verified decode), chunked prefill + continuous batching engine | sampler-aware speculative verify (filtered sampling + rejection scheme) |
| `backends/` | `DecodeBackend` protocol, torch StaticCache + CUDA graphs, MLX (Apple silicon) | — |
| `cache/` | radix prefix KV tree w/ LRU | paged blocks, KV int8/int4 quant, sinks/H2O/SnapKV eviction, sliding window |
| `quantize/` | per-layer ΔKL sensitivity (fake-quant), min-memory bit allocator, Pareto sweep | GPTQ/AWQ/HQQ, pruning, 2:4 sparsity, low-rank SVD |
| `train/` | batched ref-logprob precompute + disk cache | LoRA family, sequence packing, DPO/IPO/KTO/ORPO/SimPO/GRPO |
| `eval/` | perplexity, tokens/sec bench, pass@k, bootstrap CIs, equivalence harness | calibration (ECE), TTFT/TPOT |
| `context/` | — | RoPE scaling (PI/NTK/YaRN), attention sinks, RULER-style eval |
| `internals/` | — | logit lens, attention entropy, activation stats, CKA |

## How the fast path works

Prompt-lookup decoding drafts continuations by matching n-grams against the prompt (no draft model needed), then verifies the whole draft in one forward pass. Making that CUDA-graph compatible requires fixed shapes:

- Every decode step feeds exactly `num_draft + 1` tokens: `[last_token, draft..., pads]`. One shape → one captured CUDA graph, replayed every step.
- Pads sit at the highest positions; causal masking means no real query attends them. Pad logits are ignored.
- After acceptance, the StaticCache write pointer rewinds to the true sequence length so the next block overwrites stale slots.

The decode loop itself is framework-agnostic (`decoding/lookup_generic.py` + `backends/base.py`): only Python ints cross the backend boundary. The torch backend lives in `backends/torch_static.py`; `backends/mlx_backend.py` implements the same three methods (`begin` / `step_argmax` / `rewind`) over mlx-lm's trimmable KV cache.

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

# fast path: lookup decoding over the backend protocol
from llmopt.backends.torch_static import TorchStaticBackend
from llmopt.decoding.lookup_generic import generate_lookup

ids = tok("Summarize: ...").input_ids
compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
backend = TorchStaticBackend(model, compiled_step=compiled)
tokens, stats = generate_lookup(backend, ids, max_new_tokens=128, num_draft=16)
print(stats)  # forward_passes << 128 on input-grounded tasks

# quantization: sensitivity -> bit allocator -> Pareto
from llmopt.quantize.sensitivity import measure_sensitivity
from llmopt.quantize.allocator import allocate_bits

seqs = [tok(p).input_ids for p in prompts]
from llmopt.train.ref_logprobs import precompute_ref_logprobs
refs = precompute_ref_logprobs(model, seqs, model_name="qwen2.5-0.5b",
                               cache_dir=".refcache", batch_size=8)
sens = measure_sensitivity(model, seqs, refs, bit_widths=(2, 4))
cfg = allocate_bits(sens, kl_budget=0.05)
print(cfg.avg_bits, cfg.bits_by_layer)
```

## Benchmarks & tests

```bash
pytest                                  # pure-Python parts run without GPU
python scripts/bench_lookup_static.py   # full stacked benchmark (GPU)
python scripts/sweep_lookup.py          # ngram/draft hyperparameter sweep (GPU)
python scripts/sweep_lookup_mlx.py      # same sweep on MLX (Apple silicon)
```

On Windows, `torch.compile` needs MSVC — run benchmark scripts inside a vcvars64 environment (see `scripts/bench_compile.py` docstring).

## Design notes

- Ref logprobs stored as top-k + tail mass (full-vocab for 164×1k×150k would be ~100 GB).
- transformers 5.x StaticCache appends at an internal `cumulative_length` counter; rewinding resets that tensor in-place (`fill_`), CUDA-graph-safe because HF marks it as a static address.
- Allocator assumes ΔKL additivity across layers — approximate; always re-measure the final mixed config end-to-end.
- Fake-quant measures quality only. Real memory savings need bitpacked kernels (HQQ/GPTQ) at deploy time.

## License

MIT

"""llmopt — an oracle-verified mathematics and physics ML lab.

Every claim this package supports is checked by running something, not
by inspection: decoding is proved token-identical to eager greedy,
mathematics answers are accepted by sympy equivalence rather than
string match, generated assembly is assembled and executed, and weights
are scored by function rather than by distance to other weights. The
verdicts those instruments produced — wins, nulls, and retractions
alike — live in `docs/RESULTS.md`, curated by evidence maturity in
`docs/FINDINGS.md`.

Subpackages
-----------
Research instruments

  search/       symbolic derivation search: rewrite rules, structural
                and learned evaluators, proposal policies, transposition
                memory, and a verified ZX-graph path for circuit
                reduction. `search.engine` is the one-import facade.
  mathgen/      seeded generators for calculus, linear algebra, ODEs,
                mechanics, and proofs, with symbolic checks built into
                generation. Stable string seeds only.
  quantum/      model-Hamiltonian ground-state instruments.
  moe/          mixture-of-experts routing anatomy: demand ranking,
                keep-set construction, router masking, expert surgery.
  weightspace/  weight-reading subjects and readers — predicting what a
                network computes from its parameters.
  lab/          the adopted instrument layer: the standard gate, the
                fork-isolated oracle, verified wave checking, run
                receipts and markers, the checkpoint catalog, merge
                operations, and the Parquet result lake. Modules copied
                verbatim from frozen experiment scripts are guarded by
                source-identity tests.

Training and numerics

  train/        closed-system model births, controlled diets, LoRA,
                sequence packing, preference objectives (DPO/IPO/KTO/
                ORPO/SimPO/GRPO), and interventions comparable
                trajectory by trajectory.
  intmath       exact integer primitives — the arithmetic core behind
                bit-identical, cross-machine training replay.
  distill/      logit-KD and generalized knowledge distillation.
  quantize/     weight diagnostics, sensitivity probes, closed-form bit
                allocation, GPTQ/AWQ/HQQ, sparsity, low-rank, and packed
                integer artifacts.

Inference and systems

  decoding/     speculative and prompt-lookup decoding, sampler
                pipelines, constrained/FSM decoding, tree verification,
                chunked prefill, continuous batching, deterministic
                paths.
  cache/        KV cache: radix prefix tree, paged blocks, quantization,
                eviction policies.
  context/      RoPE scaling (PI/NTK/YaRN), attention sinks, RULER.
  kernels/      hand-written Metal and Triton kernels with honest
                benchmarks — including the ones that lost.
  backends/     torch static-cache, MLX, and a native exact-integer
                backend with a pure-Python fallback.
  codegen/      LLVM toolchain scoring: assemble the prediction, run the
                program.

Measurement

  eval/         equivalence, calibration, pass@k, latency, bootstrap
                confidence intervals — supporting readouts, never
                substitutes for a capability gate.
  internals/    logit lens, attention entropy, activation statistics,
                CKA.
  reproduce     pinned one-command replays; `python -m llmopt.reproduce
                --list` shows the registry.

Dependencies
------------
Core is torch, numpy, and sympy. Optional extras keep the heavy paths
out of a default install: `[hf]` for Hugging Face models, `[mlx]` for
Apple-silicon kernels, `[lake]` for Parquet/DuckDB queries, `[triton]`
for the CUDA kernels. Torch is imported lazily inside functions
throughout, so importing this package is cheap.
"""
_LAZY = {
    "RadixCache": "llmopt.cache.radix",
    "find_ngram_continuation": "llmopt.decoding.prompt_lookup",
    "allocate_bits": "llmopt.quantize.allocator",
    "pareto_front": "llmopt.quantize.allocator",
}

__all__ = sorted(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib
        mod = importlib.import_module(_LAZY[name])
        return getattr(mod, name)
    raise AttributeError(f"module 'llmopt' has no attribute {name!r}")

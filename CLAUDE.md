# llmopt — working notes for Claude sessions

LLM inference/training optimization lab. Small, readable, oracle-verified
implementations. See README for the full inventory and measured numbers.

## Non-negotiable conventions

- **Oracle-verified everything.** Decoding must be token-identical to eager
  greedy (`eval/equivalence.py`); math answers checked by sympy symbolic
  equivalence, never string match; asm/code scored by the toolchain
  (assemble the prediction, run the program) — `codegen/llvm.py`.
- **fp16 near-ties are a known non-bug**: different verify-block
  compositions round coin-flip logits differently. Diagnose with the eager
  logit margin at the divergence point (see `scripts/bench_stacked.py`);
  margins ≤ ~0.02 are ties, not bugs.
- **Generated datasets**: stable *string* seeds only (`random.Random(f"kind-{level}-{seed}")`)
  — tuple `__hash__` is per-process randomized and killed reproducibility
  once. Guard train/eval splits with `exclude=` (prompt sets), never seed
  offsets alone: small problem spaces collide (two real contamination
  incidents: mathgen L1/L2 43% eval-in-train; ladder `pick()` had only 4
  possible bodies). Widen the generator space before trusting a split.
- Benchmarks report honest losses too (Metal attention_decode losing to
  GEMV, first paged-attention cut losing to gather+SDPA). Keep that.

## Machine-specific setup

**Windows box (RTX 3080 10GB)**: `torch.compile` needs MSVC — run GPU
benches via
`cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && python scripts/..."`.
MSYS LLVM toolchain (clang/llvm-mc/objdump) at `C:\msys64\mingw64\bin`,
not on PATH — `codegen/llvm.py` finds it. transformers 5.12 quirks already
handled in-tree: no `from_legacy_cache`, `apply_chat_template` returns an
Encoding (go through `tokenize=False`), `cumulative_length` fills need
`inference_mode`. StaticCache max_len is bucketed to 512 under compiled
steps — every distinct length re-captures the CUDA graph (~12 s).

**Mac (36GB, Apple silicon)**: MLX backend in `backends/mlx_backend.py`,
Metal kernels in `kernels/metal.py` (v1, educational). Queued work:
port the Triton optimization lessons to Metal — group-shared K/V reads
for GQA, exp2-domain softmax, boundary-split masking (interior tiles
mask-free), autotuned tiles. 36GB fits larger teachers for `distill/`
(logit-KD + GKD ready) with 0.5B–3B students.

## Active research threads

- Capability ladder (`codegen/ladder.py` + `scripts/bench_ladder.py`):
  which rung a small model climbs — encode/decode (learned mapping) train
  up; output/o2_asm (simulation) resist. Every rung toolchain-scored.
- mathgen calculus: 0.5B LoRA 15.7% → 65.7% symbolic accuracy on
  provably-unseen problems; limits need step traces (`make_limit_traced`).
- Expert iteration ("chess-engine framing"): generator = self-play
  opponent, oracle = game rules, train on verifier-approved step chains,
  raise difficulty at the model's frontier. The step-level search version
  (model scores candidate rewrites) is the long-term goal.

## Practical

- `pytest` — pure-Python parts run anywhere; GPU/toolchain tests skip
  cleanly when hardware/tools are missing.
- Training scripts (`scripts/train_calculus.py`, `scripts/bench_ladder.py`)
  follow one recipe: LoRA r=16 on all proj linears via `train/lora.py`,
  loss on answer tokens only, length-bucketed batches with per-epoch order
  shuffling (pure length-sorted order regressed accuracy once — keep the
  shuffle).

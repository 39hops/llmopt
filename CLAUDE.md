# llmopt — working notes for Claude sessions

LLM inference/training optimization lab. Small, readable, oracle-verified
implementations. See README for the full inventory and measured numbers.

## Lab charter — domains (non-negotiable)

**We build engines for MATHEMATICS and PHYSICS. Only.**
- **No chemistry engines, no biology engines — ever.** No molecule
  generators, no reaction/pathway oracles, no protein anything, no
  wet-lab-relevant capability. This holds regardless of how
  tractable or interesting the domain looks ("methods, not
  molecules" — and now: methods, not organisms).
- **Concepts and frames from any science are welcome as METHODS**
  when they carry zero harmful applicability: quantum-chemistry
  math (basis sets, orbitals, overlap matrices), neuroscience
  structure (efficient coding, wiring economy, human-brain
  analogies for weight geometry). Borrowing the mathematics of a
  field is fine; building capability IN chemistry/biology is not.
- Benign human-brain/neuroscience links (as analogy or analysis
  frame for our models' weights/representations) are explicitly
  fine. Anything that starts to look like capability toward
  molecules, pathogens, or organisms gets refused and flagged,
  full stop.

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
- **NO sympy call is safely boxed by SIGALRM — fork is the only real
  timebox** (fork, join with deadline, SIGKILL: the
  `gen_magic_labels.solve_isolated` pattern). Generalized 2026-07-12
  from pathology #7 (`make_integrate` on L4+/L8 seeds — FIVE call
  sites bitten, including the farm loops themselves) after the
  alarm-boxed oracle live-locked anyway (pathology #10). Applies to
  generation, rules, routing probes, verifiers, and any
  oracle-on-model-text. Corollary: workers killed by an outer wall
  must STREAM their rows out incrementally, or the killed class is
  invisible to whatever trains on the data (the checkpoint
  selection-effect; bit three times).
- **Never score weights by weight distance.** The same function lives at
  many weight arrangements (neuron permutations, rescalings), so
  matching numbers is the wrong target for anything that predicts,
  generates, or compares weights (weightspace/ rungs, task vectors,
  distill). Score by *running* the weights against the oracle
  (function MSE, symbolic accuracy, toolchain). Measured basis: the
  2026-07-06 weight-reader ablation — raw weights already readable at
  80.8%, permutation-augmentation (88.4%) beat canonical sorting
  (82.4%); teach invariance, don't impose it.

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
WSL venv has NO C compiler: torch's `_native` eager router JITs triton
kernels for aten ops (Qwen RoPE) even WITHOUT torch.compile —
`TORCH_COMPILE_DISABLE`/`TORCHDYNAMO_DISABLE` don't stop it; set
`TORCH_DISABLE_NATIVE_JIT=1` (knob lives in `torch/_native/common_utils.py`).

**Mac (36GB, Apple silicon)**: MLX backend in `backends/mlx_backend.py`,
Metal kernels in `kernels/metal.py`. Split-K decode (single-head +
GQA, exp2-domain softmax) landed 2026-07-05 — ties mx.fast sdpa at
T=32k; see docstring for honest numbers. NOTE: the old bench harness
timed lazy graph construction (MLX skips dropped unevaluated arrays);
mx.eval every timed iteration. Still queued: flash prefill port
(boundary-split masking, autotuned tiles), wiring kernels into the MLX
backend. 36GB fits larger teachers for `distill/` (logit-KD + GKD
ready) with 0.5B–3B students.

## Active research threads

**`docs/BOARD.md` is the live status board** — every thread as
LIVE/BANKED/CLOSED with pointers; check it before starting work.
`docs/RESULTS.md` holds verdicts, `docs/RIFF-LEDGER.md` holds
idea provenance, `docs/LOOP-LOG.md` tracks expert-iteration rounds.

- **Step-level expert iteration** (the founding goal, LIVE since
  2026-07-12): model emits one verified rewrite per call, trained on
  oracle-approved chains (engine-replay seed + on-policy), frontier-
  adaptive loop with tripwires — `scripts/expert_loop.py`, spec in
  docs/superpowers/specs/. Judges beware the meta-pattern:
  **prediction pays only where variance lives** (four judges starved
  in one day when the engine improved under them).
- Capability ladder (`codegen/ladder.py` + `scripts/bench_ladder.py`):
  which rung a small model climbs — encode/decode (learned mapping) train
  up; output/o2_asm (simulation) resist. Every rung toolchain-scored.
- mathgen calculus: the engine now leads sympy.integrate at every
  level (L5 100%, L8 37/40); the open residue is simplify-fused
  multi-family quotients.

## Practical

- `pytest` — pure-Python parts run anywhere; GPU/toolchain tests skip
  cleanly when hardware/tools are missing.
- Training scripts (`scripts/train_calculus.py`, `scripts/bench_ladder.py`)
  follow one recipe: LoRA r=16 on all proj linears via `train/lora.py`,
  loss on answer tokens only, length-bucketed batches with per-epoch order
  shuffling (pure length-sorted order regressed accuracy once — keep the
  shuffle).

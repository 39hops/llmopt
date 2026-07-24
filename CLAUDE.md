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

- **Code changes are Fable 5's job** (Artin's standing policy,
  2026-07-16). If you are any other model or agent working in this
  repo: do not change code. Found a bug or stale reference? MENTION
  it (file, line, what's wrong) — Fable handles the fix. Mechanical
  work (file moves, reference updates) only under explicit
  supervision, and Fable verifies the pass afterward.
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

## Navigation — READ THESE BEFORE WORKING (in this order)

1. **`docs/BOARD.md`** — the live status board: every thread
   LIVE/BANKED/CLOSED, one line each. Never start work without it.
2. **`docs/handoffs/`** — dated, 0-indexed session handoffs (the
   repo-side resume artifacts; multiple per day = -0, -1, ...).
   Read the newest first after any compaction/clear.
3. **`docs/RESULTS.md`** — every verdict, win/null/retraction alike,
   newest at the bottom. Before proposing ANY experiment, grep it:
   the idea has often been run, nulled, or pre-registered already.
4. **`docs/RIFF-LEDGER.md`** — idea provenance. EVERY riff Artin or
   the house proposes gets banked here with attribution, even
   half-retracted ones ("bank everything" is standing policy).
5. **`docs/THEORY.md`** — the grounding map: house laws x published
   lineage. No row without a measured result AND a real citation.
6. **`scripts/INDEX.md`** — signature/docstring index of scripts/.
   Grep it before writing anything (don't rewrite existing code).
   **Regenerate after adding/changing scripts:**
   `.venv/bin/python scripts/gen_index.py`.
7. `docs/superpowers/relay/` — the axiom-Fable exchange (Artin
   relays manually); `docs/superpowers/specs/` — pre-run specs;
   `docs/LOOP-LOG.md` — expert-iteration rounds.

**Habits that keep these useful**: pre-register predictions in
RESULTS before a run fires; book verdicts (including honest
failures) the moment they land; consolidate BOARD + a new handoff
at natural stopping points, not mid-sprint.

## Doctrine (distilled; full text in RESULTS/handoffs)

- **Pre-registration + paired arms, always**: same device, same
  seeds, one variable. Never compare probes/gates across devices
  (measured 2x device dependence at the frontier).
- **Verified AND distinct, at every learning layer**: the oracle
  accepts X=>X as true; reward, gate candidates, AND miners must
  all reject identity rewrites (bit three times: GRPO reward hack,
  gate candidates, miner v5's bank).
- **Precision doctrine (CLOSED 2026-07-24)**: birth precision is a
  non-factor above TF32; fp64 masters are the FINAL capability rung
  for online learning (exact-vs-fp64 measured bit-identical);
  exact arithmetic is a SPEED/DETERMINISM lever (int8-sliced beats
  native fp64 — scratch/ozaki_*). Don't spend runs on
  precision-capability questions.
- **Speed defaults (lossless, always on)**: KV-cached sampling;
  bf16 births (--fast) on cuda / fp32 on Mac; GRAD_CKPT=1 for
  d768+ on 10GB; PYTORCH_CUDA_ALLOC_CONF set in-tree. A CUDA
  allocator OOM warning in a log is a TRIPWIRE (the 43x), not noise.
- **Remote ops (friendly-fire, 6 variants deep)**: kill/write/
  launch = separate ssh calls; a watcher's pgrep must never match
  a string its own launcher carries; verify file deps at arm time;
  completion markers fire on SUCCESS only; remote host/key live in
  gitignored `scratch/remote.env.sh` (never commit them); sync =
  stash -> pull -> VERIFY -> drop (never drop-on-abort). Both
  machine checkouts stay at origin/main (`git pull --ff-only`).
- **Data hygiene**: exclude=-guarded splits; underdetermined rows
  train hallucination (audit for determinability, not just
  correctness); diet exposure SHARE matters (rations for resident
  grammars when the corpus grows).

## Practical

- `pytest` (scoped to tests/ via pyproject; scratch/*_test.py are
  scripts, not tests). Pure-Python runs anywhere; GPU/toolchain
  tests skip cleanly.
- Math-native training: `scripts/train_mathnative.py` (--diet,
  --fast, VOCAB_EXTRA/BIRTH_SEED/GRAD_CKPT envs; probe scripts
  take VOCAB_EXTRA too — atom ORDER must match the birth env).
  Legacy LoRA recipe (`train/lora.py`, r=16, answer-only loss,
  length-bucketed + shuffled) still serves the 0.5B-era scripts.
- Scratch experiments live in `scratch/` (committed — they are the
  lab notebook); stray root artifacts go to `logs/archive/`; big
  jsonl/checkpoints stay untracked (file-handoff convention).

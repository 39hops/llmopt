# Engineering

What is actually built, with the code and tests to read first. Every
measured number on this page is one of two kinds and is labeled as
such: a **booked measurement**, anchored to the entry in
[RESULTS.md](RESULTS.md) or the paper that books it, carrying that
entry's device and scope fence; or a **development benchmark**,
recorded in the source of the kernel it describes, unbooked, and not a
research verdict or a CI performance guarantee. Tests verify behavior;
none of them asserts speed. Counts are given as of commit `f1f3a3ef`
(2026-09-29).

## 1. Exact integer training and cross-machine replay

A training run whose every step is integer arithmetic under one
rounding rule reproduces bit for bit on a different machine, and the
repository ships the pins to check it.

**Look at the code**

- [`llmopt/intmath.py`](../llmopt/intmath.py): the integer core. Q = 512 fixed point, one round-half-away division used everywhere, sha-pinned lookup tables.
- [`llmopt/reproduce.py`](../llmopt/reproduce.py): the one-command replay; scrubs experiment environment variables before running so an ambient knob cannot poison a pin.
- [`scratch/detbwd_gmoe_ref/pins.json`](../scratch/detbwd_gmoe_ref/pins.json): the 16 pinned arms and their trajectory digests.
- [`tests/test_intmath.py`](../tests/test_intmath.py), [`tests/test_reproduce.py`](../tests/test_reproduce.py), [`tests/test_gravmoe_artifact_input.py`](../tests/test_gravmoe_artifact_input.py): the contract, the environment scrub, the artifact refusal boundaries.
- [`llmopt/backends/intbirth_native.py`](../llmopt/backends/intbirth_native.py) with [`tests/test_intbirth_native.py`](../tests/test_intbirth_native.py): a native exact-integer backend with a pure-Python fallback.

**Booked measurements**

- A 1000-step multi-block run and a real-diet bridge replay bit-identically on a second machine; both legs execute on CPU. [VERDICT LOCKSTEP-A1/A2](RESULTS.md#L21460 "id:2026-08-06-verdict-lockstep-a1-a2-pass-on") with its device-scope correction [AMENDMENT LOCKSTEP-DEVICE-SCOPE](RESULTS.md#L22056 "id:2026-08-07-amendment-moe-gt-7-range-lockstep"); scope `[REGIME-SCOPED: deterministic integer battery]`.
- A 200-step integer run (mini, FFN-only) is trajectory-identical across Apple CPU, NVIDIA CUDA and a C++ implementation. [VERDICT DETERMINISTIC-BIRTH R2](RESULTS.md#L13255 "id:2026-07-31-verdict-deterministic-birth-r2-mini-a") and [RECEIPT R2 C++ LEG](RESULTS.md#L13502 "id:2026-07-31-receipt-r2-c-leg-axiom-the").
- All 16 arms are SHA-identical on a second CPU machine; the 10 arms the C++ engine implements are all identical there (the other six exercise house-side knobs the engine does not carry). [VERDICT GRAVMOE-P4-DEVICE](RESULTS.md#L14889 "id:2026-08-01-verdict-gravmoe-p4-device-the-entire"), [VERDICT GRAVMOE-P4-LAB](RESULTS.md#L15015 "id:2026-08-01-verdict-gravmoe-p4-lab-the-gravmoe").
- The environment scrub was hardened after an audit measured a seed variable leaking past it and changing a replay digest. [AMENDMENT AUDIT-0802](RESULTS.md#L16271 "id:2026-08-03-amendment-audit-0802-amends-verdict-v4").

**Caveats, stated once.** The C++ path ([39hops/axiom](https://github.com/39hops/axiom)) was developed independently of this code but under the same operator; it is an independent implementation, not an independent investigator. Trajectory equality certifies the pinned weight path and teacher-forced readouts, not symbolic correctness ([REPRODUCE](REPRODUCE.md)). CI runs the contract tests with a fake runtime; it replays no arm. `tests/test_vendor_axiom.py` checks the vendored C++ sources byte for byte and skips when the sibling checkout is absent.

## 2. Decoding, cache and context

Every optimized decoding path is checked token-identical against eager
greedy, and the harness reports the first position where two paths
diverge.

**Look at the code**

- [`llmopt/eval/equivalence.py`](../llmopt/eval/equivalence.py): the equivalence harness; `assert_tokens_equal` names the first divergent position.
- [`llmopt/decoding/`](../llmopt/decoding/): speculative, prompt-lookup, lookahead, EAGLE, Medusa, tree verification, FSM-constrained decoding, continuous batching, a deterministic path (20 modules).
- [`llmopt/cache/`](../llmopt/cache/): radix prefix tree, paged blocks, KV quantization, eviction, prefix reuse. [`llmopt/context/`](../llmopt/context/): RoPE scaling (PI, NTK, YaRN), RULER, gist, compression.
- Tests: [`tests/test_decoding_torch.py`](../tests/test_decoding_torch.py) and [`tests/test_tree_verify.py`](../tests/test_tree_verify.py) for token identity, [`tests/test_paged.py`](../tests/test_paged.py) and [`tests/test_context.py`](../tests/test_context.py) for the cache and context layers; the rest of the family is `tests/test_decoding_*.py`, `test_stacked.py`, `test_batching.py`, `test_prefix_reuse.py`, `test_medusa.py`, `test_fsm.py`, `test_kv_cache_policies.py`, `test_context_extras.py`.
- [`scripts/bench_stacked.py`](../scripts/bench_stacked.py): the diagnostic for fp16 near-ties, where two verify-block compositions round a coin-flip logit differently; a margin under about 0.02 is a tie, not a bug.

**Booked measurements**

- Entropy-adaptive draft length never beat a fixed draft of three on the Qwen2.5 1.5B / 0.5B pair: a null with its cost recorded. [Entropy-adaptive speculative decoding](RESULTS.md#L762 "id:2026-07-10-entropy-adaptive-speculative-decoding-2026-07"), RTX 3080.
- A fused cross-entropy (a training-side op, listed here with the other MLX systems work) at 16k tokens: 13.5 GB against 38 GB peak memory and 3203 against 2008 tokens per second, with the naive path faster below about 8k. [Fused cross-entropy (MLX)](RESULTS.md#L1387 "id:2026-07-13-fused-cross-entropy-mlx-liger-style"), Mac.

**Coverage note.** CI builds a small Llama to run the decoding tests ([ci.yml](../.github/workflows/ci.yml)). `decoding/kv.py` and `decoding/speculative_adaptive.py` have no direct test of their own.

## 3. Kernels, including the ones that lost

Hand-written Metal and Triton kernels, each verified elementwise
against a pure reference, with the losing versions kept in the source
beside the winning ones.

**Look at the code**

- [`llmopt/kernels/metal.py`](../llmopt/kernels/metal.py): rmsnorm, swiglu, rope, attention decode (v1 and split-K), GQA decode, int4 GEMV v2 and v3 (v1 survives as a comment), the 5-bit and 8-bit packed-crystal GEMVs, an exact integer GEMM, flash prefill v1 and v2, with the measurement history in the module docstring.
- [`llmopt/kernels/triton_kernels.py`](../llmopt/kernels/triton_kernels.py), [`llmopt/kernels/mlx_integration.py`](../llmopt/kernels/mlx_integration.py).
- [`tests/test_metal_kernels.py`](../tests/test_metal_kernels.py), [`tests/test_triton_kernels.py`](../tests/test_triton_kernels.py), [`tests/test_exact_gemm.py`](../tests/test_exact_gemm.py): correctness against references; they skip without the hardware.
- [`scripts/bench_metal_kernels.py`](../scripts/bench_metal_kernels.py): the benchmark harness.

**Booked measurements**

| kernel | result | device | entry |
|---|---|---|---|
| int4 dequant GEMV v3 | 1.11x over `mx.quantized_matmul` at D=4096 and 2.80x over fp16; 0.94x at D=2048; 0.72x at D=896 (shape N=14336 and the M3 Pro device are recorded in the kernel source, not in the entry) | Metal | [Fused int4 dequant-GEMV](RESULTS.md#L1185 "id:2026-07-11-fused-int4-dequant-gemv-metal-kernel") |
| 5-bit packed GEMV | 2.39x over fp16 on the packed-crystal disk format, 14336x4098, synthetic Gaussian weights, n=1 per shape | Metal, MLX, M3 Pro | [PACKED CRYSTAL C2b](RESULTS.md#L10587 "id:2026-07-29-packed-crystal-c2b-verdict-the-disk"), paper [main.tex](paper/main.tex) |
| lossless rANS (a storage format, not a runtime kernel) | Qwen3-30B at 16.48 GB, 3.67x over bf16; decode-side rANS throughput not benchmarked | Mac | [P6-v2 VERDICT](RESULTS.md#L11297 "id:2026-07-30-p6-v2-verdict-the-entropy-bound"), paper |

**Development benchmarks** (source-recorded, unbooked; from the
`metal.py` docstring, M3 Pro 36 GB, July 2026; not research-verdict
claims and not CI performance guarantees)

| kernel | observation |
|---|---|
| rmsnorm 4096x4096 | 3.0x over the unfused path, on par with `mx.fast.rms_norm` |
| swiglu | 2.2x over unfused |
| attention decode v1 (32 threads), dim 128 | loses to naive softmax@V everywhere, 10x at T=32768 |
| attention decode split-K (block_t 256) | loses at T=2048 (250 vs 197 us, launch overhead), wins from about T=8192, at T=32768 1.8x over naive and ties `mx.fast.scaled_dot_product_attention` (440 vs 443 us) |
| attention decode GQA (H=32, KVH=8) | about 3x over a per-head naive loop at every T (T=32768: 2322 vs 6910 us); the grouped variant that shares K/V loads across a group is about 10% slower |
| flash prefill v1 | correct (max error 1.2e-4 against the MLX attention) and loses 3 to 11x on every prefill shape (0 of 6) |
| flash prefill v2 (simdgroup MMA) | correct (9.8e-4) and slower than v1: 0.6 to 0.7x, 0.2x at D=128 |

**Correctness coverage.** `test_metal_kernels.py` and
`test_triton_kernels.py` check outputs against references only. No
test asserts a speed.

## 4. Symbolic oracles, generators, search, codegen and quantization

An answer counts when a computer algebra system says it is equivalent
to the reference; a predicted program counts when the toolchain
assembles and runs it.

**Look at the code**

- [`llmopt/mathgen/`](../llmopt/mathgen/): seeded generators for calculus, linear algebra, ODEs, mechanics and proofs with symbolic checks at generation time; stable string seeds only.
- [`llmopt/lab/oracle_worker.py`](../llmopt/lab/oracle_worker.py) and [`llmopt/lab/oracle.py`](../llmopt/lab/oracle.py): the process-isolated sympy oracle. A subprocess line server with a parent-owned deadline and a memory limit; a timeout or crash is a typed, conservative reject, never a silent pass.
- [`llmopt/search/`](../llmopt/search/): derivation search with explicit rewrite rules, learned evaluators and transposition memory; [`zx_engine.py`](../llmopt/search/zx_engine.py) for ZX-calculus T-count reduction, the physics leg; [`benchkit.py`](../llmopt/search/benchkit.py).
- [`llmopt/codegen/llvm.py`](../llmopt/codegen/llvm.py): `assemble()` turns predicted x86-64 into bytes with `llvm-mc`; `compile_c` and `compile_cpp(run=True)` compile and run, cross-compiling on arm64 Macs.
- [`llmopt/quantize/`](../llmopt/quantize/): sensitivity probes, closed-form bit allocation, packed integer artifacts, rotation, GPTQ, AWQ and HQQ paths.
- [`tests/test_lab_oracle.py`](../tests/test_lab_oracle.py), [`tests/test_gate_battery.py`](../tests/test_gate_battery.py), [`tests/test_lab_verify_gen_battery.py`](../tests/test_lab_verify_gen_battery.py), [`tests/test_zx_engine.py`](../tests/test_zx_engine.py), [`tests/test_derivation_search.py`](../tests/test_derivation_search.py), [`tests/test_ladder.py`](../tests/test_ladder.py), [`tests/test_quantize_methods.py`](../tests/test_quantize_methods.py), [`tests/test_pack_format.py`](../tests/test_pack_format.py).

**Booked measurements**

- The 167-row Phase D battery from the C++ implementation's emission replays 167 of 167 through the house verifier. [Phase D adjudicated](RESULTS.md#L2866 "id:2026-07-20-phase-d-adjudicated-the-c-engine"); `test_lab_verify_gen_battery.py` replays it against the tracked `data/axiom_phaseD_167.jsonl`.
- The certificate tier: 443 of 443 kernel-checked in Lean. [VERDICT LEAN-TIER-1](RESULTS.md#L18619 "id:2026-08-04-verdict-lean-tier-1-the-certificate").

**Caveats.** `sympy` calls cannot be boxed by a signal alarm inside the
calling process; only a separate worker process with a parent-owned
deadline can end one, which is why the oracle is a spawned subprocess
rather than in-process timeboxing. `test_ladder.py` and `test_codegen.py` skip
without an LLVM or MSVC toolchain. [`llmopt/quantum/`](../llmopt/quantum/)
and [`llmopt/weightspace/`](../llmopt/weightspace/) are smaller
instruments with one or two test files each.

## 5. Evidence and provenance infrastructure

A claim here is checked by programs before a person reads it: the bar
was written first, the instrument is frozen, the receipts are hashed,
and the generated documents refuse to drift.

**Look at the code**

- [`llmopt/lab/prereg.py`](../llmopt/lab/prereg.py), [`scripts/adjudicate.py`](../scripts/adjudicate.py), [`llmopt/lab/metrics.py`](../llmopt/lab/metrics.py): machine-readable pre-registration and its adjudicator; a measurement that is not the registered quantity raises `MetricContractError`. [`preregs/`](preregs/) holds the registered bars.
- [`llmopt/lab/claimlint.py`](../llmopt/lab/claimlint.py), [`scripts/claim_lint.py`](../scripts/claim_lint.py), [`claims.deny.json`](claims.deny.json): verdict prose linted against the adjudicator's objects and a list of superseded readings.
- [`scripts/gen_receipt_lock.py`](../scripts/gen_receipt_lock.py), [`receipts.lock.json`](receipts.lock.json), [`tests/science_incidents/test_frozen_receipt_mutation.py`](../tests/science_incidents/test_frozen_receipt_mutation.py): every cited receipt is content-addressed; a changed or vanished receipt fails the suite.
- [`tests/science_incidents/`](../tests/science_incidents/): the rule that every blocking finding by an auditor becomes an executable invariant plus a regression fixture built from the real incident's numbers; its status table records which classes have both and which are still pending ([README](../tests/science_incidents/README.md)).
- [`scripts/liverun.py`](../scripts/liverun.py) and [`llmopt/lab/locator.py`](../llmopt/lab/locator.py): the live-run sentinel with its pre-commit interlock, and logical artifact locators in place of machine paths ([`tests/test_path_hygiene.py`](../tests/test_path_hygiene.py)).
- [`scripts/gen_codemap.py`](../scripts/gen_codemap.py) and [`scripts/check_source.sh`](../scripts/check_source.sh): the generated documents (CODEMAP, results index, script index, README block), each a fixed point over its own output with a `--check` mode that never writes, and the single script that defines green. [`tests/test_docs_integrity.py`](../tests/test_docs_integrity.py) checks every ledger link lands on an entry heading with a matching id and ratchets the FINDINGS backlog.
- [`llmopt/runs/`](../llmopt/runs/): receipts, completion markers, trajectory instruments and the Parquet result lake, whose schema requires device, seed count and weights hash ([`tests/test_lake.py`](../tests/test_lake.py)).

**Implementation facts, not measurements.** The claim linter is
unit-tested and run by the booking ritual; it is not a CI gate on
prose. The liverun interlock has no dedicated test. The smoke-mode
path isolation is a convention backed by an advisory editor hook, not
a runtime guard. The science_incidents status table is maintained by
hand. Machine-adjudicated pre-registration arrived in mid-August 2026
and registered rungs increasingly carry it; bars before that, and many
since, were recorded in prose inside RESULTS.

## Package, tests and CI

- Lazy-import root with documented API stability tiers: [`llmopt/__init__.py`](../llmopt/__init__.py); core dependencies are `torch`, `numpy` and `sympy` (the oracle is a core dependency); optional extras for figures and the lake. Count the tests with `pytest --collect-only -q`; GPU, MLX and toolchain tests skip cleanly when the hardware or toolchain is absent.
- Four CI jobs ([ci.yml](../.github/workflows/ci.yml)): the source tree via `check_source.sh` (generator checks, pytest, ruff, README sync); a wheel built and imported from outside a checkout, with an honest error for the subsystems that need the checkout; a core-dependencies import guard; the generated-document check. [`tests/test_layering.py`](../tests/test_layering.py), [`tests/test_public_imports.py`](../tests/test_public_imports.py), [`tests/test_lazy_root.py`](../tests/test_lazy_root.py).
- Python: the package declares 3.11 or newer and its modules parse on 3.11; CI tests 3.12 only, and imports on 3.11 are not exercised there. Two frozen `scratch/` drivers, and the tests that import them, use 3.12 syntax (nested quotes inside f-strings). They are evidence records kept byte-identical and are not ported.
- Figures: every published number is read from [`figures.json`](figures.json) by both renderers in [`llmopt/figures/`](../llmopt/figures/) ([`tests/test_figure_export.py`](../tests/test_figure_export.py)); the README's maturity counts are regenerated from FINDINGS by `scripts/gen_readme.py`.

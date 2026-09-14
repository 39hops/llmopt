---
name: machines
description: Windows 3080/WSL and Mac MLX toolchain quirks (MSVC for torch.compile, MSYS LLVM path, transformers 5.12 handling, TORCH_DISABLE_NATIVE_JIT, MLX bench eval rule). Read before any GPU bench, remote 3080 job, or Mac kernel work.
---

# Machine quirks

Moved from CLAUDE.md (Machine-specific setup) so the text loads only when
needed. The two-computers framing stays in CLAUDE.md.

**Windows box (RTX 3080 10GB)**: `torch.compile` needs MSVC — run GPU
benches via
`cmd /c "call \"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && python scripts/..."`.
MSYS LLVM toolchain (clang/llvm-mc/objdump) at `C:\msys64\mingw64\bin`,
not on PATH — `llmopt/codegen/llvm.py` finds it. transformers 5.12 quirks already
handled in-tree: no `from_legacy_cache`, `apply_chat_template` returns an
Encoding (go through `tokenize=False`), `cumulative_length` fills need
`inference_mode`. StaticCache max_len is bucketed to 512 under compiled
steps — every distinct length re-captures the CUDA graph (~12 s).
WSL venv has NO C compiler: torch's `_native` eager router JITs triton
kernels for aten ops (Qwen RoPE) even WITHOUT torch.compile —
`TORCH_COMPILE_DISABLE`/`TORCHDYNAMO_DISABLE` don't stop it; set
`TORCH_DISABLE_NATIVE_JIT=1` (knob lives in `torch/_native/common_utils.py`).

**Mac (36GB, Apple silicon)**: MLX backend in `llmopt/backends/mlx_backend.py`,
Metal kernels in `llmopt/kernels/metal.py`. Split-K decode (single-head +
GQA, exp2-domain softmax) landed 2026-07-05 — ties mx.fast sdpa at
T=32k; see docstring for honest numbers. NOTE: the old bench harness
timed lazy graph construction (MLX skips dropped unevaluated arrays);
mx.eval every timed iteration. Flash prefill + MLX kernel
wiring both SHIPPED (kernels/metal.py + kernels/mlx_integration.py
docstrings carry the honest numbers). 36GB fits larger teachers for `llmopt/distill/` (logit-KD + GKD
ready) with 0.5B–3B students.

# Rotate-then-quantize: numerically better arrangements of the same weights

**Date:** 2026-07-06
**Status:** Approved

## Goal

Measure whether an orthogonal rotation — which changes every numeric
value of W while preserving the function exactly (store W' = QW, apply
Q^T at runtime, or absorb into adjacent layers) — reduces quantization
error (QuaRot-style incoherence processing). This is the testable core
of "is there a mathematically better arrangement of the same weights."

## Scope (part a only — reconstruction error, CPU)

Part (b) — end-to-end decode accuracy of a rotated-quantized model —
is deferred until part (a) shows a signal and the GPU queue clears.

## Components

### `llmopt/quantize/rotate.py`

- `hadamard(n)`: normalized Hadamard matrix, n a power of 2
  (Sylvester construction). Orthogonal: H @ H.T == I.
- `random_orthogonal(n, seed)`: QR of a seeded Gaussian.
- `rotation_error(w, bits, rotation)`: relative Frobenius error
  ||W - Q^T q(QW)||_F / ||W||_F where q = existing methods.rtn.
  Baseline is rotation=None: ||W - q(W)||_F / ||W||_F.

### `scripts/bench_rotate_quantize.py`

- Matrices: (1) real Linear weights from the cached torch
  Qwen2.5-0.5B-Instruct (a few layers spanning depth: q_proj, down_proj
  — CPU load, no GPU); (2) synthetic controls — iid Gaussian (rotation
  should NOT help: already incoherent) and Gaussian + planted outlier
  columns (rotation SHOULD help: outliers get smeared).
- Grid: bits in {4, 3, 2} x rotation in {none, hadamard, random}.
- Report per (matrix, bits): relative error for each rotation. Honest
  kill-switch from the roadmap: int8 is skipped (no measurable gap
  expected anywhere); if rotation doesn't beat none at 4 bits on real
  weights, that's the reported result.

## Hypotheses on record

1. iid Gaussian control: rotation ~ no effect (sanity for the harness).
2. Outlier-planted control: rotation clearly better (the mechanism).
3. Real Qwen weights: rotation helps, more at fewer bits — real
   transformer weights have outlier structure. If they don't (MLX-era
   models may be trained flatter), the negative is the finding.

## Testing

- hadamard/random_orthogonal are orthogonal (Q @ Q.T ~ I, atol 1e-5).
- Function preservation: (Q.T @ (Q @ w)) reconstructs w exactly
  (before quantization) — the rotation itself is lossless.
- rotation_error(w, bits, None) matches direct rtn error.
- Outlier matrix: hadamard error < none error at 4 bits (the mechanism
  test — deterministic seed).

## Non-goals

- Absorbing rotations into adjacent layers (needed for deployment,
  not for measuring the effect).
- GPTQ/AWQ/HQQ interactions — RTN isolates the arrangement effect;
  smarter quantizers partially compensate for bad arrangements and
  would muddy attribution.

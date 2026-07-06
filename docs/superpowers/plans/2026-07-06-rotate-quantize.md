# Rotate-Then-Quantize Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether orthogonal rotations reduce RTN quantization error on real and synthetic weight matrices.

**Architecture:** One module (`llmopt/quantize/rotate.py`), one bench script, tests. Reuses `quantize/methods.rtn`.

**Tech Stack:** torch (CPU), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-06-rotate-quantize-design.md`.
- Quantizer is `methods.rtn` only. CPU only. Hypotheses in the spec.

---

### Task 1: rotate.py + tests

**Files:**
- Create: `llmopt/quantize/rotate.py`
- Test: `tests/test_rotate_quantize.py`

**Interfaces (produces):**
- `hadamard(n) -> torch.Tensor` (n power of 2, orthonormal)
- `random_orthogonal(n, seed=0) -> torch.Tensor`
- `rotation_error(w, bits, rotation=None) -> float` (relative Frobenius
  error of round-trip through rtn in the rotated basis)

Steps: failing tests (orthogonality both constructions; losslessness of
rotate/unrotate; rotation_error(None) == direct rtn error; hadamard
beats none on an outlier-planted matrix at 4 bits) → run → implement →
run → commit.

### Task 2: bench script + record results

**Files:**
- Create: `scripts/bench_rotate_quantize.py`
- Modify: `llmopt/quantize/rotate.py` (docstring: measured numbers)

Steps: implement grid (real Qwen 0.5B layers q_proj/down_proj at
layers 0/12/23 + two synthetic controls; bits 4/3/2; rotations
none/hadamard/random) → run → record table in docstring → commit.

# Weight-Space Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Classify which function family a tiny MLP was trained on, from its weights alone, with a permutation-symmetry ablation (raw / canonical / augmented).

**Architecture:** `llmopt/weightspace/` package: `subjects.py` (generate + train subject MLPs, canonicalize/permute transforms), `reader.py` (neuron-token transformer classifier). `scripts/train_weight_reader.py` runs the three-arm experiment.

**Tech Stack:** torch (CPU/MPS), pytest. No GPU-queue dependency.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-06-weightspace-reader-design.md`.
- String seeds only: `random.Random(f"{family}-{i}-{seed}")`.
- Subject: MLP 1→16→16→1 tanh; reject fits with MSE > 0.01 and resample.
- 6 families; eval seed namespace disjoint from train; assert no
  coefficient-tuple collisions.
- Canonicalization must satisfy `canon(permute(net)) == canon(net)`
  exactly and preserve the computed function.

---

### Task 1: subjects.py — families, subject training, transforms

**Files:**
- Create: `llmopt/weightspace/__init__.py`, `llmopt/weightspace/subjects.py`
- Test: `tests/test_weightspace_subjects.py`

**Interfaces (produces):**
- `FAMILIES: tuple[str, ...]` (6 names)
- `Subject` dataclass: `family: str`, `coeffs: tuple`, `weights: list[torch.Tensor]` (the 6 tensors: W1,b1,W2,b2,W3,b3), `fit_mse: float`
- `make_subject(family: str, i: int, seed: int) -> Subject`
- `make_dataset(n: int, seed: int, exclude: frozenset = frozenset()) -> list[Subject]` (round-robin over families)
- `permute_hidden(weights, perm1, perm2) -> weights` (function-preserving)
- `canonicalize(weights) -> weights` (sort hidden neurons by incoming L2)

Steps: write failing tests (determinism, fit quality, transform
properties below), run, implement, run, commit.

Key test content:

```python
def test_subject_deterministic():
    a = make_subject("sin", 3, seed=0)
    b = make_subject("sin", 3, seed=0)
    assert a.coeffs == b.coeffs
    assert all(torch.equal(x, y) for x, y in zip(a.weights, b.weights))

def test_subject_fits_its_function():
    s = make_subject("poly2", 1, seed=0)
    assert s.fit_mse < 0.01

def test_permute_preserves_function():
    s = make_subject("tanh", 0, seed=0)
    x = torch.linspace(-2, 2, 64)[:, None]
    p = permute_hidden(s.weights, torch.randperm(16), torch.randperm(16))
    assert torch.allclose(forward(s.weights, x), forward(p, x), atol=1e-5)

def test_canonicalize_collapses_permutations():
    s = make_subject("gauss", 2, seed=0)
    p = permute_hidden(s.weights, torch.randperm(16), torch.randperm(16))
    ca, cb = canonicalize(s.weights), canonicalize(p)
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(ca, cb))

def test_dataset_family_balance_and_exclude():
    ds = make_dataset(12, seed=0)
    assert len({s.family for s in ds}) == 6
    banned = frozenset(s.coeffs for s in ds)
    ds2 = make_dataset(12, seed=1, exclude=banned)
    assert not banned & {s.coeffs for s in ds2}
```

### Task 2: reader.py — neuron-token transformer

**Files:**
- Create: `llmopt/weightspace/reader.py`
- Test: `tests/test_weightspace_reader.py`

**Interfaces (produces):**
- `tokenize(weights) -> torch.Tensor` `[33, FEAT]`: one row per neuron
  (incoming weights + bias, zero-padded to FEAT=17), layer index
  appended by the model's embedding, not the tokens.
- `WeightReader(n_families=6, d_model=128, n_layers=4, n_heads=8)`;
  `forward(tokens [B, 33, FEAT]) -> logits [B, 6]`
- `train_reader(subjects, labels, *, epochs, augment: bool, canonical: bool, seed) -> WeightReader`
- `evaluate_reader(model, subjects, labels) -> float`

Tests: tokenize shape `[33, 17]`; forward shape; overfit 32 subjects to
>= 0.95 train accuracy in <= 200 epochs (sanity that gradients flow).

### Task 3: scripts/train_weight_reader.py — the three-arm experiment

**Files:**
- Create: `scripts/train_weight_reader.py`

Generate 4000 train / 500 eval (process pool, `exclude=` guard),
run arms raw / canonical / augmented with the same reader config and
seed, print accuracy per arm vs 16.7% chance floor. Save the summary to
`checkpoints/weight_reader_results.json`.

Steps: implement, run end-to-end, record numbers in module docstring of
`llmopt/weightspace/__init__.py`, commit.

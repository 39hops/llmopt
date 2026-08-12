# Phase 4: llmopt/common layer + duplication harvest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spec 2026-08-12 §Phase 4 (4.1-4.3): create `llmopt/common/`
(device/seed/ckpt), give the four copy-pasted helpers a package home,
migrate FREE-file call sites, delete dead `sys.path` bootstraps in free
files.

**Architecture:** Decision #1 resolved (Artin GO 2026-08-12):
`pick_device(override) -> str`, torch-like precedence
`explicit arg > LLMOPT_DEVICE env > cuda > mps > cpu`. Placement stays
torch-native `.to(dev)`; deliberate CPU islands (seeded samplers,
oracle paths) are correctness PINS, stay hardcoded `"cpu"`, never
migrate. No ambient default device (sigma never transports across
devices; device stays visible per call). Decision #2 resolved: new
`llmopt/common/`, not `lab/` (these are utilities, not instruments).

**Tech Stack:** pytest, torch (lazy-imported), sympy.

## Global Constraints

- FREE files only: a file's CODEMAP class must be UNCITED (or the file
  lives in `llmopt/`/`tests/`, which CODEMAP does not govern) before
  any edit. results-cited / spec-cited / reproduce-pinned / library
  files keep their inline idioms and copies.
- CPU pins are exempt from migration: any `"cpu"` chosen for seeded
  sampling or oracle work stays. Migration only touches the
  `X if <backend>.is_available() else ...` DEFAULT-accelerator idiom.
- `import torch` stays lazy inside `llmopt/common/*` functions (Phase 5
  wants `import llmopt` torch-free).
- One task per commit; pytest rc captured unpiped; ruff 0; CODEMAP +
  INDEX regen in any commit touching scripts/scratch.
- Public repo commit trailer: `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>`; never a session URL.

---

### Task 1: `llmopt/common/` — device, seed, ckpt

**Files:**
- Create: `llmopt/common/__init__.py`, `llmopt/common/device.py`,
  `llmopt/common/seed.py`, `llmopt/common/ckpt.py`
- Test: `tests/test_common.py`

**Interfaces:**
- Produces: `pick_device(override: str | None = None) -> str`;
  `srng(*parts) -> random.Random`; `load_ckpt(path, map_location="cpu",
  weights_only=True)`.

- [ ] **Step 1: failing tests.** `tests/test_common.py`:

```python
"""llmopt.common — device/seed/ckpt utilities (spec 2026-08-12 §4.1).

pick_device precedence is torch-like (Artin GO 2026-08-12):
explicit arg > LLMOPT_DEVICE env > cuda > mps > cpu. Placement is
NOT this module's business: .to(dev) stays per-object, and seeded
CPU islands are pins that never route through pick_device.
"""
import random

import pytest

from llmopt.common.device import pick_device
from llmopt.common.seed import srng


def test_pick_device_override_wins(monkeypatch):
    monkeypatch.setenv("LLMOPT_DEVICE", "cpu")
    assert pick_device("mps") == "mps"


def test_pick_device_env_beats_detection(monkeypatch):
    monkeypatch.setenv("LLMOPT_DEVICE", "cpu")
    assert pick_device() == "cpu"


def test_pick_device_detection_order(monkeypatch):
    monkeypatch.delenv("LLMOPT_DEVICE", raising=False)
    torch = pytest.importorskip("torch")
    want = ("cuda" if torch.cuda.is_available() else
            "mps" if torch.backends.mps.is_available() else "cpu")
    assert pick_device() == want


def test_srng_string_seed_law():
    # stable STRING seeds only (house law: tuple hash is
    # per-process randomized)
    a = srng("mathgen", 3, 7000)
    b = srng("mathgen", 3, 7000)
    assert isinstance(a, random.Random)
    assert [a.random() for _ in range(4)] == \
        [b.random() for _ in range(4)]
    assert srng("mathgen", 3, 7001).random() != \
        srng("mathgen", 3, 7000).random()


def test_load_ckpt_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from llmopt.common.ckpt import load_ckpt
    p = tmp_path / "w.pt"
    torch.save({"w": torch.ones(3)}, p)
    ck = load_ckpt(p)
    assert torch.equal(ck["w"], torch.ones(3))
```

- [ ] **Step 2: run, expect FAIL** (`ModuleNotFoundError: llmopt.common`).
- [ ] **Step 3: implement.**

`llmopt/common/__init__.py`:

```python
"""llmopt.common — shared utilities (spec 2026-08-12 §Phase 4).

Not instruments (those live in llmopt.lab): device resolution,
string-seeded RNG, checkpoint IO. torch imports stay lazy so
`import llmopt` never drags torch in.
"""
```

`llmopt/common/device.py`:

```python
"""Default-accelerator resolution. PLACEMENT is torch-native
(.to(dev)); deliberate CPU islands (seeded samplers, oracles) are
correctness pins and never route through here. No ambient default:
device stays an explicit per-call value because instrument sigma
never transports across devices."""
from __future__ import annotations

import os


def pick_device(override: str | None = None) -> str:
    """explicit arg > LLMOPT_DEVICE env > cuda > mps > cpu."""
    if override:
        return override
    env = os.environ.get("LLMOPT_DEVICE")
    if env:
        return env
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

`llmopt/common/seed.py`:

```python
"""String-seed law (CLAUDE.md data-hygiene): stable STRING seeds
only — tuple __hash__ is per-process randomized and killed
reproducibility once."""
from __future__ import annotations

import random


def srng(*parts) -> random.Random:
    """random.Random seeded by the dash-joined string of parts."""
    return random.Random("-".join(str(p) for p in parts))
```

`llmopt/common/ckpt.py`:

```python
"""Checkpoint IO. weights_only=True is the default and the audit
point; the few legitimate weights_only=False loads stay explicit
torch.load calls at their sites."""
from __future__ import annotations


def load_ckpt(path, map_location="cpu", weights_only=True):
    import torch
    return torch.load(path, map_location=map_location,
                      weights_only=weights_only)
```

- [ ] **Step 4: run tests, expect 5 passed.**
- [ ] **Step 5: commit** `feat: llmopt/common — pick_device/srng/load_ckpt (phase 4.1)`.

### Task 2: migrate FREE-file device sites

**Files:**
- Create: `scratch/phase4_sites.py` (survey generator, one-shot)
- Modify: free files it lists (llmopt/, tests/, UNCITED scripts/scratch)

**Interfaces:**
- Consumes: `pick_device` from Task 1; CODEMAP classes.

- [ ] **Step 1: survey.** `scratch/phase4_sites.py` prints, per file
  containing a device-idiom line
  (`re.search(r'"(cuda|mps)" if torch\.', line)`), the file, its
  CODEMAP class (parsed from `docs/CODEMAP.md`; files under `llmopt/`
  and `tests/` report class `package`), and the line numbers:

```python
"""One-shot Phase 4 survey: device-idiom sites x CODEMAP class.
Output drives the migration list; only class UNCITED or package
rows are migrated."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
classes = {}
for row in (ROOT / "docs" / "CODEMAP.md").read_text().splitlines():
    m = re.match(r"\| \S+ \| (\S+) \| (\S+) \|", row)
    if m:
        classes[m.group(1)] = m.group(2)
pat = re.compile(r'"(cuda|mps)" if torch\.')
for d in ("llmopt", "tests", "scripts", "scratch"):
    for f in sorted((ROOT / d).rglob("*.py")):
        hits = [i + 1 for i, l in
                enumerate(f.read_text().splitlines()) if pat.search(l)]
        if hits:
            cls = ("package" if d in ("llmopt", "tests")
                   else classes.get(f.name, "?"))
            print(f"{cls:>16} {f.relative_to(ROOT)} {hits}")
```

- [ ] **Step 2: migrate.** For every `package`/`UNCITED` row: replace
  the idiom expression with `pick_device()` (or
  `pick_device(args.device)` where a CLI flag exists), add
  `from llmopt.common.device import pick_device`. NEVER touch a line
  whose device expression omits the accelerator check entirely
  (`"cpu"` literals = pins). `mps`-first or `cuda`-first both collapse
  to `pick_device()` — behavior identical on both lab machines (each
  has exactly one backend).
- [ ] **Step 3: verify** — full pytest rc 0; ruff 0; re-run survey:
  zero `package`/`UNCITED` rows remain.
- [ ] **Step 4: regen CODEMAP/INDEX, commit**
  `refactor: free-file device sites -> pick_device (phase 4.1)`.

### Task 3: helper adoption — `_root`/`_check`/`load_nnue` -> `llmopt/search/benchkit.py`, `ternary` -> `llmopt/common/quant.py`

**Files:**
- Create: `llmopt/search/benchkit.py`, `llmopt/common/quant.py`
- Test: `tests/test_benchkit.py` (extend `tests/test_common.py` for
  ternary)
- Modify: UNCITED host files only (survey step decides the list)

- [ ] **Step 1: identity diff.** For each helper, diff every copy
  (survey with `grep -A12 "^def _root"` etc. across hosts). Copies
  that differ from the majority body are DIVERGENT: leave them and
  their host untouched, list them in the commit message. Expect
  `_root` variants with/without `kind` param — treat the
  `(rng, level, kind)` form as canonical; `(rng, level)`-form hosts
  (bench_opcap) count as divergent unless the body is a strict
  specialization.
- [ ] **Step 2: canonical copies.** `llmopt/search/benchkit.py` takes
  the majority `_root`/`_check`/`load_nnue` bodies verbatim, with the
  module deps they need imported from their package homes
  (`_expression`/`X` from the mathgen module the hosts use;
  `NnueEval`/`featurize`/`State` from `llmopt.search`). Public names:
  `make_root = _root`, `check_answer = _check`, `load_nnue` as-is.
  `llmopt/common/quant.py`:

```python
"""Weight quantization transforms."""
from __future__ import annotations


def ternary(w):
    """Per-tensor ternary quantization: sign(w) * mean(|w|) over
    the nonzero band (the train_ternary.py lineage body, adopted
    verbatim — verify against the host copies in Task 3 step 1)."""
```

  (body copied verbatim from `scripts/train_ternary.py:26` after the
  identity diff confirms the scratch copies match; if they do not
  match, adopt the `train_ternary.py` body and leave divergent hosts
  untouched).
- [ ] **Step 3: tests.** `tests/test_benchkit.py`: `_root` returns an
  (unevaluated, truth) pair whose `_check` round-trips True for both
  kinds at level 1 with `srng("benchkit-test", 1)`; `load_nnue`
  smoke-skips without a checkpoint on disk. `ternary` test: output
  values in `{-m, 0, +m}` and sign pattern matches input signs above
  threshold.
- [ ] **Step 4: migrate call sites** in UNCITED hosts only: delete the
  local copy, import from the package home. Frozen hosts keep copies
  (no dual-copy guard needed — Phase 3's source-identity era is
  closed; behavior tests own correctness now).
- [ ] **Step 5: pytest rc 0, ruff 0, regen, commit**
  `refactor: helper adoption — benchkit + quant.ternary (phase 4.2)`.

### Task 4: delete dead `sys.path` bootstraps in free files

- [ ] **Step 1: enumerate** — extend `scratch/phase4_sites.py` (or a
  second one-shot) listing files with `sys.path.insert` x CODEMAP
  class. Only `package`/`UNCITED` rows proceed; scripts that are
  ENTRY POINTS importing sibling scripts by bare name (the
  `from bench_step_tokens import ...` pattern) KEEP the bootstrap —
  deletion only where every import already resolves through the
  installed `llmopt` package.
- [ ] **Step 2: delete, then prove** — for each edited file:
  `.venv/bin/python <file> --help` (or plain import via pytest for
  non-CLI modules) exits 0.
- [ ] **Step 3: pytest rc 0, ruff 0, regen, commit**
  `chore: drop dead sys.path bootstraps in free files (phase 4.3)`.

### Task 5: close

- [ ] Full suite fresh (`rc` captured), push, 3080 ff-only sync with
  HEAD assert, exit-criteria check against spec §Phase 4: `common/`
  exists with tests; duplicate helper count in free files zero;
  `sys.path.insert` count in free files zero (minus the entry-point
  exemption, which gets named in the commit if nonzero).

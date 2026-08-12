# Phase 5: package coherence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spec 2026-08-12 §Phase 5: torch-free `import llmopt` via lazy
`__getattr__`; split `llmopt/figures/` and `llmopt/runs/` out of
`lab/`; route/document the scattered env reads incl. `FRAC = 0.453`
provenance; dead-code sweep + import-every-module smoke; >400-line
module split only at natural seams.

**Architecture:** Moves use full-fidelity module aliasing for
back-compat: the old `llmopt/lab/<name>.py` file becomes
`import llmopt.figures.<name> as _m; sys.modules[__name__] = _m` (alias
shim — private names survive, unlike star re-exports). Free importers
migrate to the new paths; anything frozen keeps working through the
shim. Artin GO 2026-08-12 (housekeeping gate green: suite fresh, smokes
rc 0, both checkouts lockstep at 1c8b5cf).

**Tech Stack:** pytest, PEP 562 module `__getattr__`.

## Global Constraints

- Canonical lab bodies with batteries (gate, verify, gen, keepsets,
  oracle_worker, shards) DO NOT MOVE — only figures/runs families.
- Batteries must stay green after every task (they pin booked numbers).
- One task per commit; unpiped pytest rc; ruff 0; CODEMAP/INDEX regen
  when scripts/scratch change.
- Public repo trailer: `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>`; never a session URL.

---

### Task 1: lazy root `__init__` + torch-free import proof

**Files:**
- Modify: `llmopt/__init__.py` (imports + `__all__` block only;
  docstring untouched)
- Test: `tests/test_lazy_root.py`

- [ ] **Step 1: failing test.**

```python
"""Root llmopt import stays cheap and torch-free (spec §Phase 5)."""
import subprocess
import sys


def test_import_llmopt_is_torch_free():
    code = ("import sys; import llmopt; "
            "assert 'torch' not in sys.modules, 'torch leaked'; "
            "assert 'sympy' not in sys.modules, 'sympy leaked'; "
            "print('CLEAN')")
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout


def test_lazy_attrs_resolve():
    import llmopt
    assert llmopt.RadixCache.__name__ == "RadixCache"
    assert callable(llmopt.find_ngram_continuation)
    assert callable(llmopt.allocate_bits)
    assert callable(llmopt.pareto_front)
```

- [ ] **Step 2: implement** — replace the three eager imports with:

```python
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
```

- [ ] **Step 3: run both tests** (if the torch-free assert fails
  because cache.radix already imports torch at module top, that is the
  finding — the lazy map is still right; adjust the test to assert
  torch-free BEFORE attribute access and document which attribute pulls
  torch).
- [ ] **Step 4: commit** `feat: lazy root __getattr__ — import llmopt torch-free (phase 5)`.

### Task 2: split `llmopt/figures/` and `llmopt/runs/` out of `lab/`

**Files:**
- Move: `lab/figstyle.py`, `lab/figsvg.py`, `lab/figures.py`,
  `lab/anatomy.py` -> `llmopt/figures/`; `lab/runlog.py`, `lab/lake.py`,
  `lab/traj.py`, `lab/runfiles.py` -> `llmopt/runs/`
- Create: alias shims at each old path:

```python
"""Moved to llmopt.<figures|runs>.<name> (Phase 5, 2026-08-12).
This alias keeps old imports working with full fidelity."""
import sys

import llmopt.figures.figstyle as _m

sys.modules[__name__] = _m
```

- New `__init__.py` for both packages (docstring naming the family and
  the move date).

- [ ] **Step 1: git mv + shims + inits.** (`runfiles` rides with runs/:
  `lab/runlog.py`'s own docstring says it composes with runfiles —
  they are one family.)
- [ ] **Step 2: migrate importers** in llmopt/, tests/, and free
  scripts/scratch (survey: `grep -rln "lab.figstyle\|lab import
  figstyle\|..."` — 21 files at plan time; frozen files keep the shim
  path, free files move to the new path).
- [ ] **Step 3: lab/__init__.py** — update docstring + any re-exports;
  intra-lab imports of moved modules follow the new paths.
- [ ] **Step 4: full suite + batteries; ruff; regen; commit**
  `refactor: split llmopt/figures + llmopt/runs out of lab (phase 5)`.

### Task 3: env-read routing + FRAC provenance

**Files:**
- Modify: `llmopt/lab/config.py` (docstring table), `llmopt/lab/keepsets.py`
  (comment only — canonical battery body, behavior frozen)

- [ ] **Step 1: enumerate** every `os.environ` read in llmopt/ (7 at
  spec time). Table in `lab/config.py` docstring: site, variable,
  default, verdict (LabConfig-routed | exempt-by-purpose + reason).
  Exemptions expected: `reproduce.py` (env passthrough to child),
  `backends/intbirth_native.py` `AXIOM_BUILD_DIR` (build-machine knob),
  `runlog.py` `LLMOPT_LOG` (pre-config bootstrap logging).
- [ ] **Step 2: FRAC provenance** — comment at `keepsets.py:27`: 0.453
  is the next-golden-point keep fraction from the GT keep-frac arms
  (RESULTS.md L18793 "FRACS {0.618, 0.453 (next golden point), 0.40}");
  value UNCHANGED (battery pins behavior).
- [ ] **Step 3: route** any site that is neither exempt nor
  battery-frozen through `LabConfig`/`GateCfg.from_env`. If the table
  shows zero such sites, the routing step is a no-op and the table IS
  the deliverable (say so in the commit).
- [ ] **Step 4: suite; commit** `docs: env-read table + FRAC provenance; route residual reads (phase 5)`.

### Task 4: dead-code sweep + import-every-module smoke

**Files:**
- Create: `scratch/phase5_deadcode.py` (AST: module-level defs never
  referenced anywhere in llmopt/scripts/scratch/tests), report only
- Test: `tests/test_public_imports.py` — walk `llmopt/**/*.py`,
  `importlib.import_module` each, `pytest.skip` on ImportError naming
  an optional extra (mlx, triton, duckdb, transformers), hard-fail on
  anything else

- [ ] **Step 1: write + run the smoke test** (it is the lasting guard).
- [ ] **Step 2: run the sweep**; triage output into: (a) delete-safe
  (zero references anywhere, not `__all__`, not a battery body) —
  delete; (b) everything else — park the list in the commit message,
  no action.
- [ ] **Step 3: suite; regen; commit** `chore: import-all smoke + dead-code sweep (phase 5)`.

### Task 5: >400-line seam check + close

- [ ] **Step 1: seam check on `kernels/metal.py` (1120)**: if the file
  has clean section boundaries (split-K decode family vs flash prefill
  family vs shared helpers), split into `metal_decode.py` /
  `metal_prefill.py` with `metal.py` re-exporting both (docstrings with
  honest numbers travel with their kernels). If the sections share
  kernel-source strings or helpers non-trivially, DO NOT split — record
  the reason in the commit. `search/rules.py` (1013) and
  `mathgen/problems.py` (731): same test, same rule; expectation is
  no-split (rule tables and generator families are natural single
  files).
- [ ] **Step 2: close** — fresh full suite (rc captured), ruff, regen,
  push, 3080 ff-only sync + HEAD assert, spec §Phase 5 exit check.

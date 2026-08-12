# Phase 3 Module 5: gate shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete Phase 3 (spec 2026-08-12 §Phase 3 row 5): canonical
`sample_wave_lp`/`gate_eval` bodies live only in `llmopt/lab/gate.py`;
`scripts/step_grpo_micro.py` becomes a line-count-preserving re-export
shim; the `sys.modules` aliases at `llmopt/lab/gate.py:32-33` are
deleted; the source-identity guard dies in the same commit, replaced by
a booked-number battery.

**Architecture:** Two commits, battery first (the ordering rule).
Battery = replay of AMENDMENT SOFT-PROMPT-1-SAMPLER (RESULTS.md L27709,
Mac-booked: fresh-generator single draws agree 0/200 across a
zero-padded category widening; consecutive draws desync) plus pinned
gate problem-set literals (string-seed law over
`GATE_BAND + 1000*lv + i`). Shim shape is the module-3
line-count-preserving variant because RESULTS cites lines INSIDE both
bodies: `:184` (METALLICITY-1 at L27532, SOFT-PROMPT-1 at L27640) and
`:65` (SOFT-PROMPT-1-SAMPLER at L27709).

**Tech Stack:** pytest, torch (cpu), sympy.

## Global Constraints

- One module per commit; source-identity guard deleted in the SAME
  commit as its shim (spec §Phase 3 step 4).
- `scripts/step_grpo_micro.py` line numbers ≥78 and ≥225 must not
  shift; lines 65 and 184 must carry the cited fragments.
- Never gate a commit on piped pytest; capture rc.
- Public repo: commits end `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>`, never a session URL.
- Regenerate `docs/CODEMAP.md` + `scripts/INDEX.md` in any commit that
  adds/changes scripts or scratch files.
- ruff exit 0 on `llmopt tests scripts`.

---

### Task 1: Booked-number gate battery

**Files:**
- Create: `tests/test_gate_battery.py`
- Create: `scratch/gatepins_freeze.py` (one-shot pin printer)

**Interfaces:**
- Consumes: `llmopt.lab.gen._gen_isolated`, `llmopt.lab.gate.GATE_BAND`
- Produces: battery that Task 2 must keep green.

- [ ] **Step 1: freeze the gate problem pins.** Write
  `scratch/gatepins_freeze.py`:

```python
"""One-shot: print the sstr of the first gate problem per level for
the standard 120 gate's seed grid (GATE_BAND + 1000*lv + 0). Output
becomes the GATE_PINS literal in tests/test_gate_battery.py."""
import sympy as sp

from llmopt.lab.gate import GATE_BAND
from llmopt.lab.gen import _gen_isolated

for lv in (3, 4, 5, 6, 7):
    p = _gen_isolated(lv, GATE_BAND + 1000 * lv + 0)
    print(f"    ({lv}): {sp.sstr(p._expr)!r},")
```

  Run `.venv/bin/python scratch/gatepins_freeze.py`, capture output.

- [ ] **Step 2: write the battery.** `tests/test_gate_battery.py` with
  three tests (fill GATE_PINS from step 1's output):

```python
"""Booked-number battery for the gate module (Phase 3 module 5).

Replays AMENDMENT SOFT-PROMPT-1-SAMPLER (RESULTS.md L27693-27750,
2026-08-12, Mac): the sampler at scripts/step_grpo_micro.py:65 is
category-count-sensitive — a single draw from a FRESH generator
agrees 0/200 when the probability vector is widened with exact
zeros, while consecutive draws from ONE generator desynchronize
(CPU multinomial consumes stream randomness dependent on category
count). Plus the gate problem-set pins: the standard 120 gate's
seed grid is deterministic under the string-seed law.
"""
import pytest

torch = pytest.importorskip("torch")


def _probs(n, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, generator=g)
    return p / p.sum()


def test_fresh_generator_single_draw_agrees_200():
    # booked: "a single draw from a FRESH generator agrees:
    # 0 mismatches in 200" (RESULTS.md L27709 postscript)
    mismatches = 0
    for s in range(200):
        p40 = _probs(40, 1000 + s)
        p48 = torch.cat([p40, torch.zeros(8)])
        a = int(torch.multinomial(
            p40, 1, generator=torch.Generator().manual_seed(s)))
        b = int(torch.multinomial(
            p48, 1, generator=torch.Generator().manual_seed(s)))
        mismatches += a != b
    assert mismatches == 0


def test_shared_generator_stream_desyncs():
    # booked mechanism: category count changes how much of the
    # random stream each draw consumes, so identical distributions
    # diverge on LATER draws from a shared generator
    ga = torch.Generator().manual_seed(7)
    gb = torch.Generator().manual_seed(7)
    seq_a, seq_b = [], []
    for s in range(12):
        p40 = _probs(40, 2000 + s)
        p48 = torch.cat([p40, torch.zeros(8)])
        seq_a.append(int(torch.multinomial(p40, 1, generator=ga)))
        seq_b.append(int(torch.multinomial(p48, 1, generator=gb)))
    assert seq_a != seq_b


GATE_PINS = {
    # (level): sstr of _gen_isolated(lv, GATE_BAND + 1000*lv + 0)
    # frozen 2026-08-12 by scratch/gatepins_freeze.py — pins the
    # standard gate's seed arithmetic + band + generator together
    <FILL FROM STEP 1 OUTPUT — five entries>
}


def test_gate_problem_grid_pinned():
    sp = pytest.importorskip("sympy")
    from llmopt.lab.gate import GATE_BAND
    from llmopt.lab.gen import _gen_isolated
    for lv, want in GATE_PINS.items():
        p = _gen_isolated(lv, GATE_BAND + 1000 * lv + 0)
        assert sp.sstr(p._expr) == want, lv
```

- [ ] **Step 3: run** `.venv/bin/python -m pytest tests/test_gate_battery.py -q > /tmp/gb.log 2>&1; rc=$?` — expect 3 passed, rc 0.
  If `test_shared_generator_stream_desyncs` PASSES trivially or FAILS
  (desync not reproduced with synthetic probs), adjust draw count/pad
  shape until the booked mechanism shows; if it cannot be reproduced,
  STOP — that contradicts the booked amendment and needs Artin.
- [ ] **Step 4: regen** `.venv/bin/python scripts/gen_codemap.py && .venv/bin/python scripts/gen_index.py`.
- [ ] **Step 5: commit** `test: gate battery — sampler-amendment replay + gate problem pins (phase 3 module 5 prep)`.

### Task 2: The shim

**Files:**
- Modify: `llmopt/lab/gate.py` (docstring, delete lines 26-33 alias
  block, switch in-body imports at 111-112)
- Modify: `scripts/step_grpo_micro.py:44-77` and `:165-224`
  (line-count-preserving replacement blocks)
- Modify: `scripts/bench_verify_fast.py:63-66` (remove dead pop
  workaround)
- Modify: `tests/test_lab_adoption.py` (replace source-identity with
  bind + line-anchor tests; drop `inspect`)
- Modify: `llmopt/lab/__init__.py` docstring (adoption-law paragraph
  now historical; all five pairs shimmed)

**Interfaces:**
- Consumes: Task 1's battery.
- Produces: `step_grpo_micro.sample_wave_lp is llmopt.lab.gate.sample_wave_lp`; same for `gate_eval`.

- [ ] **Step 1: lab/gate.py canonicalization.**
  - Docstring: replace the "ADOPTED VERBATIM"/alias paragraphs with
    CANONICAL BODY since 2026-08-12 (Phase 3 module 5); note the
    frozen script is a line-count-preserving shim because RESULTS
    cites `:65`/`:184`; cite `tests/test_gate_battery.py`.
  - Delete `import sys as _sys`, the `_gen_mod`/`_verify_mod` imports,
    and both `_sys.modules.setdefault(...)` lines (26-33).
  - In `gate_eval`, change the two in-body imports to:

```python
    from llmopt.lab.gen import _gen_isolated
    from llmopt.lab.verify import verify_wave
```

- [ ] **Step 2: step_grpo_micro shim blocks.** Replace lines 44-77
  with EXACTLY this 34-line block (line 65 = quote, line 77 = import):

```python
# Phase 3 module 5 (2026-08-12): sample_wave_lp's canonical body
# lives in llmopt.lab.gate (re-export at line 77 below). This block
# is LINE-COUNT-PRESERVING: the original 34-line function occupied
# lines 44-77, and RESULTS cites a line INSIDE that body — AMENDMENT
# SOFT-PROMPT-1-SAMPLER (RESULTS.md L27709) books the sampler defect
# "at scripts/step_grpo_micro.py:65". The replacement keeps every
# later line number unchanged and keeps the cited fragment on its
# booked line: line 65 below quotes it verbatim. Behavior is pinned
# by tests/test_gate_battery.py (replays the booked sampler
# measurements) and tests/test_lab_adoption.py (shim-binds +
# lineage-constant guards + these line anchors).
#
#
#
#
#
#
#
#
#
#
# line 65 was: nxt = int(torch.multinomial(probs[b], 1, generator=gens[b]))
#
#
#
#
#
#
#
#
#
# (canonical body: llmopt/lab/gate.py, sample_wave_lp)
from llmopt.lab.gate import sample_wave_lp  # noqa: E402,F401
```

  Replace lines 165-224 with EXACTLY this 60-line block (line 184 =
  quote, line 224 = import): same header pattern, 11 header comment
  lines citing METALLICITY-1 (RESULTS.md L27532) and SOFT-PROMPT-1
  (L27640) booking "the in-process weights sha printed by gate_eval
  at scripts/step_grpo_micro.py:184", then bare `#` pad lines, with
  line 184 reading:

```python
# line 184 was: print(f"[gate] weights sha {wh.hexdigest()[:16]}", flush=True)
```

  line 223 `# (canonical body: llmopt/lab/gate.py, gate_eval)`, and
  line 224:

```python
from llmopt.lab.gate import gate_eval  # noqa: E402,F401
```

  Constants (lines 28-41) stay — `collect`/`main` use them; the
  lineage-constants test keeps them matched to `lab.gate`.

- [ ] **Step 3: verify anchors.**

```bash
awk 'NR==65||NR==184{print NR": "$0} NR==80||NR==227{print NR": "$0}' scripts/step_grpo_micro.py
```

  Expect: 65 multinomial quote, 184 sha quote, 80 `def collect(`,
  227 `def main(` (unchanged neighbors prove no drift).

- [ ] **Step 4: bench_verify_fast.** Delete the
  `sys.modules.pop("bench_step_tokens", None)` line and its 3-line
  comment in `main()` — dead once the alias is gone.

- [ ] **Step 5: tests.** In `tests/test_lab_adoption.py`:
  - Replace `test_gate_sources_identical` with:

```python
def test_gate_shim_binds_lab_bodies(sgm):
    from llmopt.lab import gate
    assert sgm.sample_wave_lp is gate.sample_wave_lp
    assert sgm.gate_eval is gate.gate_eval


def test_gate_shim_line_anchors():
    # RESULTS cites step_grpo_micro.py:65 and :184 (inside the old
    # bodies); the shim is line-count-preserving and must keep the
    # quoted fragments on their booked lines.
    text = (ROOT / "scripts" / "step_grpo_micro.py").read_text() \
        .splitlines()
    assert "torch.multinomial(probs[b], 1, generator=gens[b])" \
        in text[64]
    assert "[gate] weights sha" in text[183]
    assert text[76].startswith("from llmopt.lab.gate import")
    assert text[223].startswith("from llmopt.lab.gate import")
    assert text[79].startswith("def collect(")
    assert text[226].startswith("def main(")
```

  - Keep `test_gate_constants_match_lineage` unchanged.
  - Remove the `import inspect` (now unused).
  - Update module docstring: all five pairs shimmed; source-identity
    era closed; batteries listed.
- [ ] **Step 6: lab/__init__ docstring** — rewrite the "Verbatim
  adoptions (source-identity guarded)" section as "Canonical bodies
  (Phase 3 shims, 2026-08-12; frozen originals re-export from here)".
- [ ] **Step 7: exit check** — `grep -rn "getsource" tests/` must show
  only the vendor-triple guards (if any) and nothing gate-related.
- [ ] **Step 8: full verify.**

```bash
.venv/bin/python -m pytest -q > /tmp/pytest_m5.log 2>&1; rc=$?
tail -3 /tmp/pytest_m5.log; echo "PYTEST_RC=$rc"
.venv/bin/ruff check llmopt tests scripts; echo "RUFF_RC=$?"
```

- [ ] **Step 9: smoke the by-path entry point** (script still runs as
  `__main__`): `.venv/bin/python -c` is classifier-blocked; instead
  `cd /Users/artin/code/llmopt && .venv/bin/python scripts/step_grpo_micro.py --help > /tmp/sgm_help.log 2>&1; echo rc=$?` — expect rc 0, argparse help.
- [ ] **Step 10: regen + commit** — `gen_codemap.py`, `gen_index.py`,
  then commit `refactor: phase 3 module 5 — gate shim; line-preserving re-export, lab.gate canonical, sys.modules aliases deleted`.
- [ ] **Step 11: push, sync 3080** (`scratch/wsl.sh run "cd ~/code/llmopt && git pull --ff-only"`, assert HEAD hash).

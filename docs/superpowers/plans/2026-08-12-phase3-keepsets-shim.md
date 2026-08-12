# Phase 3 Module 1: keepsets Shim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the dual-copy `scratch/gt2_jaccard.py` / `llmopt/lab/keepsets.py` pair to a single body: the lab module becomes canonical, the scratch file becomes a re-export shim, and the source-identity guard is deleted in the same commit — the booked-number battery passing is the proof.

**Architecture:** `llmopt/lab/keepsets.py` already holds a character-identical adoption of the six symbols (`_frac`, `_flag`, `decode_counts`, `keep`, `jmean`, `coverage`). The shim replaces those six bodies in `scratch/gt2_jaccard.py` with one import line, keeping the file's docstring, `TRAJ_DEFAULTS`, `_traj()`, and `main()` intact so it stays a runnable analysis script and its scratch importers (`from gt2_jaccard import decode_counts, keep`) keep working.

**Tech Stack:** Python 3.11, pytest.

## Global Constraints

- One module per commit; this plan is ONE module (`keepsets`) and lands as ONE code commit (spec Phase 3: "Never more than one in a commit").
- The source-identity guard (`tests/test_lab_keepsets.py::test_source_identity`) is deleted in the SAME commit as the shim (CLAUDE.md dual-copy lifecycle).
- The full acceptance test `test_full_acceptance_booked_stats_and_dump_bytes` must RUN (not skip) and pass on this Mac: booked Jaccards `0.8013 / 0.5331 / 0.5280`, nulls `0.9205 / 0.8670 / 0.6364`, all three `checkpoints/gt2_*_arm0_decode.json` byte-identical. If any TRAJ artifact is missing, the task is BLOCKED — a skipped battery is not proof.
- Never gate a commit on piped pytest: redirect output to a file, capture `rc=$?`, echo it (house rule, fired 2026-08-07).
- Public repo: commit message ends with exactly `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; never a session URL.
- No behavioral change to any of the six symbols: the shim is import-only. `main()` and env semantics (`FRAC`, `GATE_ONLY`, `DROP_TAIL`, `DUMP_DECODE`, call-time env resolution) unchanged.
- `docs/CODEMAP.md` and `scripts/INDEX.md` are regenerated in the same commit (`scripts/gen_codemap.py`, `scripts/gen_index.py`) since gt2_jaccard's import set changes.
- Nothing is booked in RESULTS.md: this is engineering, not an experiment. No FINDINGS bullet needed (the ratchet only fires on RESULTS edits).

---

### Task 1: Shim gt2_jaccard onto llmopt.lab.keepsets

**Files:**
- Modify: `scratch/gt2_jaccard.py` (replace lines 46-112, the six function bodies, with a re-export; keep docstring lines 1-27, imports, `TRAJ_DEFAULTS`, `_traj`, `main`)
- Modify: `llmopt/lab/keepsets.py:1-14` (docstring only: no longer "adopted verbatim"; it is now the canonical body)
- Modify: `tests/test_lab_keepsets.py` (replace `test_source_identity` with `test_shim_binds_lab_bodies`; remove the now-tautological `frozen` parity asserts)
- Regenerate: `docs/CODEMAP.md`, `scripts/INDEX.md`

**Interfaces:**
- Consumes: `llmopt.lab.keepsets._frac, _flag, decode_counts, keep, jmean, coverage` (existing, unchanged signatures).
- Produces: `scratch/gt2_jaccard.py` whose module attributes `_frac, _flag, decode_counts, keep, jmean, coverage` are the SAME objects as `llmopt.lab.keepsets`'s (`is`-identity). Scratch importers (`gt5_union_keep.py:21`, `gt7_draw.py:124`) are untouched and keep working.

- [ ] **Step 1: Write the failing shim-identity test**

In `tests/test_lab_keepsets.py`, replace `test_source_identity` (lines 35-39) with:

```python
def test_shim_binds_lab_bodies(frozen):
    """scratch/gt2_jaccard.py is a shim: its public symbols ARE the
    llmopt.lab.keepsets objects (Phase 3 module 1, 2026-08-12). The
    booked-number acceptance below is the behavioral proof that
    replaced the source-identity guard."""
    for name in ("_frac", "_flag", "decode_counts", "keep", "jmean",
                 "coverage"):
        assert getattr(frozen, name) is getattr(keepsets, name), name
```

Also update the module docstring tier 1 line (file line 3) to:
`1. Shim identity: scratch/gt2_jaccard.py re-exports the lab bodies (always on).`

Then strip the tautological parity asserts (frozen and keepsets are now the same objects):
- `test_drop_tail_and_gate_only_rules`: drop the `frozen` parameter and delete the final parity loop (lines 67-71, `# parity with the frozen implementation...` through the end of the function). The literal-expected asserts at lines 62-66 stay — they are the real behavioral checks.
- `test_keep_tie_break_is_stable`: drop `frozen`; assert becomes `assert k == {0: {5, 1, 2}}`.
- `test_jmean_and_coverage`: drop `frozen`; asserts become `assert keepsets.jmean(ka, kb) == (0.75, 0.5)` and `assert keepsets.coverage(demand, ka) == 0.7`.

The `frozen` fixture itself STAYS (used by `test_shim_binds_lab_bodies` to load the scratch module by path).

- [ ] **Step 2: Run the new test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_lab_keepsets.py::test_shim_binds_lab_bodies -v > /tmp/t1.log 2>&1; rc=$?; tail -5 /tmp/t1.log; echo RC=$rc`
Expected: FAIL (AssertionError on `_frac` — the scratch file still defines its own bodies, so the objects differ). RC=1.

- [ ] **Step 3: Implement the shim in scratch/gt2_jaccard.py**

Replace the block from the `def _frac` comment (line 45, `# Env is resolved at CALL time...`) through the end of `coverage` (line 112) with exactly:

```python
# Phase 3 module 1 (2026-08-12): the six analysis symbols live in
# llmopt.lab.keepsets (canonical since this commit); this file keeps
# the runnable CLI and the TRAJ path table. Booked GT2-REVIEW-2
# numbers are pinned by tests/test_lab_keepsets.py's acceptance
# battery against the frozen TRAJ logs and decode dumps.
from llmopt.lab.keepsets import (  # noqa: F401
    _flag, _frac, coverage, decode_counts, jmean, keep)
```

Keep everything else byte-identical: docstring, `import json` / `import os` / `defaultdict` imports (delete `from collections import defaultdict` and `import json` ONLY if `main()` no longer uses them — it does use `json.dump` and `os.environ`, so keep `import json` and `import os`; `defaultdict` becomes unused, delete that one import line), `TRAJ_DEFAULTS`, `_traj`, `main`, and the `__main__` block.

- [ ] **Step 4: Update llmopt/lab/keepsets.py docstring**

Replace docstring lines 1-7 with:

```python
"""lab.keepsets — keep-set / coalition algebra. CANONICAL BODY since
2026-08-12 (Phase 3 module 1); scratch/gt2_jaccard.py is a re-export
shim over these symbols and keeps only its CLI. Originally adopted
verbatim from that file 2026-08-06. Guarded by
tests/test_lab_keepsets.py (shim identity + synthetic battery +
full acceptance against the booked stats and the byte-frozen
checkpoints/gt2_*_arm0_decode.json dumps).
```

Lines 8-14 (the REGENERATION-SENSITIVE warning and env note) stay unchanged.

- [ ] **Step 5: Run the keepsets battery — acceptance must RUN, not skip**

Run: `.venv/bin/python -m pytest tests/test_lab_keepsets.py -v > /tmp/t2.log 2>&1; rc=$?; cat /tmp/t2.log; echo RC=$rc`
Expected: all tests PASS, RC=0, and `test_full_acceptance_booked_stats_and_dump_bytes` shows PASSED (not SKIPPED). If it shows SKIPPED, report BLOCKED: the TRAJ artifacts (`logs/opus/moe_gt1_traj_v2.jsonl`, `gt2_phys_traj.jsonl`, `gt2_code_traj.jsonl`) or the `checkpoints/gt2_*_arm0_decode.json` dumps are missing on this machine, and the shim must NOT land without the behavioral proof.

- [ ] **Step 6: Smoke-run the CLI and verify booked numbers on stdout**

Run: `.venv/bin/python scratch/gt2_jaccard.py > /tmp/t3.log 2>&1; rc=$?; head -8 /tmp/t3.log; echo RC=$rc`
Expected: RC=0 and the output contains `Jaccard(math,phys): mean 0.8013`, `Jaccard(math,code): mean 0.5331`, `Jaccard(phys,code): mean 0.5280`. This proves `main()` still works through the shim.

- [ ] **Step 7: Verify the scratch importers still resolve**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0, 'scratch'); from gt2_jaccard import decode_counts, keep; import llmopt.lab.keepsets as K; assert decode_counts is K.decode_counts and keep is K.keep; print('shim importers OK')"`
Expected: `shim importers OK`.

- [ ] **Step 8: Regenerate CODEMAP and INDEX**

Run: `.venv/bin/python scripts/gen_codemap.py && .venv/bin/python scripts/gen_index.py`
Expected: both exit 0; `git diff --stat docs/CODEMAP.md scripts/INDEX.md` shows changes consistent with gt2_jaccard now importing llmopt (its imports column may change; class stays `library`).

- [ ] **Step 9: Full suite, real exit code**

Run: `.venv/bin/python -m pytest -q > /tmp/t4.log 2>&1; rc=$?; tail -3 /tmp/t4.log; echo PYTEST_RC=$rc`
Expected: PYTEST_RC=0, no new failures vs the pre-plan baseline (658 passed / 7 skipped; count may shift by the removed source-identity test).

- [ ] **Step 10: Commit (one commit — shim + guard deletion + regens together)**

```bash
git add scratch/gt2_jaccard.py llmopt/lab/keepsets.py tests/test_lab_keepsets.py docs/CODEMAP.md scripts/INDEX.md
git commit -m "phase3 module 1: keepsets shim — gt2_jaccard re-exports lab bodies, source-identity guard deleted, booked battery green

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Expected: single commit; `git show --stat HEAD` lists exactly the five files.

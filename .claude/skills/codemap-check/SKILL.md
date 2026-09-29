---
name: codemap-check
description: Check a file's CODEMAP class before editing anything under scratch/ or scripts/ - frozen evidence files (results-cited, reproduce-pinned) need the dual-copy or adoption path, not a direct edit.
user-invocable: false
---

# CODEMAP class check (before touching scratch/ or scripts/)

`docs/CODEMAP.md` is the move gate: one row per top-level file in
scratch/ and scripts/, class ladder
`library > reproduce-pinned > results-cited > spec-cited > tool-referenced > UNCITED`.
Since 2026-09-29 the law follows the repository's real reach: dotted
and dynamic imports, path-literal and shell invocation, tooling
references (CLAUDE.md, .claude/, jobs/, pyproject), preregs, locked
receipts that name their emitter, brace / glob / bare-stem citations,
and citations inherited THROUGH a cited caller (the `via` column).

Before editing any file there:

```bash
grep -n "<filename>" docs/CODEMAP.md
```

## What each class means for an edit

- **results-cited / reproduce-pinned** — evidence record, frozen in
  place. A booked verdict cites this exact source. Direct edits only
  for: (a) a dual-copy fix landing in BOTH the scratch original and
  its lab adoption in the same commit (source-identity test enforces
  this), or (b) an adoption migration that deletes the guard in the
  same commit. Otherwise extend the adopted `llmopt/lab/` module.
  Never fork a frozen family (no detbwd_r4.py).
- **library** — imported by other code, or invoked by path from
  llmopt/ or tests/; edit freely but run the importers' tests.
- **spec-cited** — a spec references it; edit fine, keep the spec
  claim true.
- **tool-referenced** — only lab tooling names it (a skill, CLAUDE.md,
  a jobs/*.cmd); edit fine, keep the tool pointer true.
- **UNCITED** — no document names it and no code reaches it. Still
  not automatically disposable: a UNCITED file may be the only run
  recipe behind a verdict booked by log path.

## Regen rule

`gen_codemap.py` inventories TRACKED files only — a newly added
scratch file is invisible to a regen run before its own commit and
reddens the NEXT suite. After adding any scratch/scripts file:
commit it, then `.venv/bin/python scripts/gen_codemap.py` and
`scripts/gen_index.py`, same commit or immediately after.

A PreToolUse hook (`.claude/hooks/codemap_guard.py`) asks on frozen
paths as a backstop; this skill is the front door — check first so
the hook never fires.

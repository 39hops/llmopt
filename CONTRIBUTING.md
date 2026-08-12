# Contributing

This repository is a research lab with an append-only evidence ledger.
The rules below exist so that a change cannot silently invalidate a
published number.

## Where code belongs

- `llmopt/` is the package: instruments and reusable machinery, tested
  under `tests/`. API stability tiers are documented in
  `llmopt/__init__.py`.
- `scripts/` holds registered entry points and generators;
  `scripts/INDEX.md` is the generated index. Regenerate with
  `python scripts/gen_index.py` after any change there.
- `scratch/` is the lab notebook: unregistered probes and registered
  experiment drivers, committed as-is. Files cited by booked verdicts
  are evidence and stay frozen in place. Check a file's class in
  `docs/CODEMAP.md` before touching anything under `scratch/` or
  `scripts/`.

## How a finding is booked

Verdicts live in `docs/RESULTS.md`, append-only; corrections are
AMENDMENT entries naming their target, never edits. Every booking adds
a one-bullet summary with a maturity tag to `docs/FINDINGS.md` in the
same commit; CI enforces this coupling. Pre-registration precedes
measurement: the bar is written down before the run fires.

## How a figure is published

Numbers enter `docs/figures.json` once; both renderers
(`llmopt/figures/figsvg.py` for SVG, `llmopt/figures/figures.py` for
matplotlib) read from it, and `scripts/gen_readme.py --check` keeps the
README's generated regions in sync. A number is never typed twice.

## What is never rewritten

- `docs/RESULTS.md` history and any file `docs/CODEMAP.md` classes as
  results-cited, spec-cited, or reproduce-pinned.
- `llmopt/vendor/` — vendored verbatim with provenance headers.
- Frozen scratch drivers. Canonical copies of adopted code live in
  `llmopt/`; the originals remain as import shims or frozen records.

## Checks

`pytest` runs the code tests and the ledger guards (`-m "not docs"`
for code only). `ruff check llmopt tests scripts` must pass. GPU, MLX,
and toolchain tests skip cleanly on machines without them;
`LLMOPT_FULL=1` turns missing-artifact skips into failures on the
machine that holds the artifacts.

Reproduction of pinned results: see `docs/REPRODUCE.md` and
`python -m llmopt.reproduce --list`.

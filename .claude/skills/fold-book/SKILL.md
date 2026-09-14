---
name: fold-book
description: Fold a drafted RESULTS entry (auditor folds), lint it, and only then book it — one chain, every stage gated on a real exit code, no partial fold ever reaches the ledger. User-invoked only.
disable-model-invocation: true
---

# /fold-book — fold, lint, then book (never book an unfolded draft)

Origin: 2026-09-14. A fold script asserted mid-way, left the draft
unfolded, and the booking script appended the unfolded text to
RESULTS.md (reverted before commit). Two scripts run in sequence
with no gate between them is how that happens. This skill is the
gate.

## What it runs

`scripts/fold_book.py --draft <draft.md> --fold <fold.py> --book <book.py>
[--prereg docs/preregs/<x>.json --obs <obs.json>] [--check "<cmd>"]...`

1. **FOLD** — `<fold.py> <copy-of-draft>` on a temporary COPY. The
   copy replaces the draft only if the script exits 0 and changed
   the bytes. A fold that changes nothing stops the chain
   (`--allow-nofold` to accept deliberately; also the way to book a
   draft that had no auditor findings).
2. **LINT** — `scripts/claim_lint.py` on the folded draft (with
   `--prereg/--obs` when the rung has a machine-readable pre-reg).
   ERRORs stop the chain.
3. **CHECKS** — every `--check` command (grammar of a FINDINGS bullet,
   a receipt lock dry run, a wording-auditor script) must exit 0.
4. **BOOK** — only now does `<book.py>` run: the `/book` ritual as a
   script (append RESULTS, regenerate the index, link by id, FINDINGS
   bullet, living-doc edits, receipt lock, index). Its exit code is
   the chain's.

Nothing scientific lives in this skill; `/book` still owns the
ledger semantics (heading grammar, dict sums, `amends` targets,
FINDINGS tags, `code_commit`). Write the booking script the way
`/book` says, then let this skill decide whether it may run.

## Writing the fold script

- One replacement list, each `(old, new)` asserted to occur exactly
  once, applied in memory, `write_text` ONCE at the end. An assert
  anywhere means nothing is written and the chain stops.
- Take the draft path from `sys.argv[1]` (the copy), never a
  hardcoded `/tmp` path.
- Wrapped prose: copy the exact wrapped bytes from the draft into
  `old` (a `grep -n` first); a mismatch is a stop, not a warning.

## After it books

The chain does not commit. Continue with `/book` step 5 (commit gated
on a redirected pytest rc, CODEMAP regen when a new script landed,
`code_commit` = the parent of the booking commit) and the post-commit
regen commit. Do not add hooks that pop `needs_link`; re-pop by id in
the booking script.

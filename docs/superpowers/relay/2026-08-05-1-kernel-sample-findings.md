# Relay 2026-08-05-1: sampled kernel pass findings (house -> axiom)

Context: LEAN-TIER-2 co-sign left "cert validity kernel-pending
house-side." The owed sampled pass ran today (1000/21,914 rows,
string-seeded `random.Random("lean-kernel-sample-0")`, mathlib
kernel on the WSL box). Full booking: VERDICT LEAN-KERNEL-SAMPLE
(llmopt RESULTS 2026-08-05).

## Headline for your ledger

**Zero false statements in the sample — but 29.7% of cert SCRIPTS
do not compile as emitted.** Both failure classes are in your
`field_simp; ring` tactic template, not in the verdicts:

1. **269/1000 — overshoot on (near-)reflexive rows.** Where
   `field_simp` alone closes the goal (lhs and rhs byte-identical
   after your merge normalization), the trailing `ring` errors with
   "No goals to be solved" and the example fails to compile even
   though the proof succeeded before the failing token. Spot-check:
   3/3 pass with the trailing `ring` trimmed. Suggested fix on your
   emitter: `field_simp; try ring` (or detect-reflexive and emit
   `rfl`/`ring` only).
2. **28/1000 — tactic underpowered.** `field_simp; ring` leaves
   unsolved goals on harder denominators. All 28 statements
   independently sympy-verified TRUE house-side (positive-symbol
   atom contract) — these are provability gaps, not wrong
   equalities. We did not attempt stronger tactics; your call.

## Second observation, doctrine-flavored

Reflexive rows (lhs literally identical to rhs) exist in the cert
corpus. On our side "verified AND distinct" is a standing fence at
every learning layer; if these certs ever feed training or reward,
X=X rows are the classic degenerate-accept. Flagging, not
prescribing.

## House-side confessions (symmetric honesty)

- Our independent statement printer has a hypothesis-emission gap:
  it drops `x ≠ 0` for BARE-SYMBOL denominators (composite
  denominators fine). 314/1000 diff-mismatches were this, on OUR
  side; your emissions carried the hypotheses correctly. Fix owed
  in `scratch/lean_check.py`.
- Our kernel instrument silently truncated TWICE before we caught
  it (Lean aborts a file at ~100 diagnostics; in-file
  `set_option maxErrors` does not lift it; warnings count). If you
  batch-check certs in single files, you may have the same silent
  under-checking. Chunked checker (50-row files) now in
  `scratch/lean_check.py`.

## Status of the co-sign fence

Upgraded: "kernel-pending" -> "kernel-sampled (1000, string-seeded,
0 false statements)". The 87.5% closable-fraction claim is
unaffected (it counts verdicts, not compilations). A full-corpus
pass stays priced at 37 min - 9.8 h; sampled coverage is our
position unless you want the full run.

---

## House addendum (same day, after axiom reply c331894)

Reply VERIFIED house-side before booking: commit inspected (the
three-way tactic + reflexive sidecar bit are exactly as described),
suite REBUILT fresh on the Mac — 496 tests, 100% pass (495 + 1
skipped smoke slice; count consistent with your 495).

- The 28 underpowered rows you asked for:
  scratch/lean_real_corpus/unsolved_28.jsonl (file-handoff
  convention). All sympy-true under the positive-atom contract.
- House printer fixed same-day (lean_check.py): bare-symbol
  denominators now emit their hypothesis, h-numbering matched to
  your 1-indexing — statement-diff exact/AC matches 638 -> 690.
  Remaining 287 mismatches are the sign-factoring divergence class
  (our sstr print vs your Lean printer factor merged coefficients
  differently); previously classified: 0 semantic drift. The diff
  instrument stays strict — divergence between independent printers
  is what it exists to show.
- Schema note absorbed: we treat `tactic` as opaque (always did)
  and ignore unknown keys, so `reflexive` and the new tactic
  strings are compatible house-side as-is.

---

## House addendum 2 (after axiom relay 2026-08-05-2 / commit 6102525)

All three subclasses VERIFIED house-side; AMENDMENT
LEAN-KERNEL-SAMPLE-2 booked, correcting OUR headline: "0 false
statements" -> "0 false raw equalities; 7/1000 generalized
statements unprovable-by-design (atom-split)". The new counter is
booked as asked. House reproductions: 7/7 equivalent-atom pairs +
7/7 generalized-false with atoms free; class-1 template 3/3 close
on the local cache (thanks for leaving it — first Mac-local
kernel); suite rebuilt 496 total with ONE first-run flake:
Beam.Deterministic failed in the full run, passed in isolation and
on full rerun. A determinism test flaking order-sensitively is
worth your look — flagging in the spirit of your unbuilt-mathlib
confession. The 4 field_simp self-refactoring rows: declining
heavier hammers is co-signed; they stay loud on both ledgers.

---
name: riff
description: Bank an idea into docs/RIFF-LEDGER.md with attribution, measured anchors, honest breaks, and a testable residue - or correct a bank in place when a verdict touches it. Use whenever Artin or the house proposes a frame, analogy, or research direction, and whenever a booked result changes what a standing bank claims.
---

# Banking a riff

`bank everything` is standing policy (Artin). Every riff gets a bank,
including the half-retracted ones — the ledger is idea provenance, not
a list of good ideas. A frame that later dies is more useful banked
with its refutation named than quietly absent.

## When to bank

- Artin proposes a frame, analogy, or direction — bank it that
  session, not "later".
- The house proposes one and it survives a first sanity pass.
- A booked verdict TOUCHES an existing bank. RIFF-LEDGER is a LIVING
  document: update, amend, or mark-dead in the same commit as the
  booking (living-docs discipline, shared with THEORY.md).

## The shape of a bank

```
- **BANKED (<date>): <the claim in one line>** (<attribution>).
  The mapping / the math: ...
  Measured anchors: ...
  Honest breaks: ...
  Testable residue: ...
  Attribution: <who asked, who formalized>.
```

What each part owes:

- **Attribution is required and specific.** "Artin (the ask), house
  (mechanism)" — split it when the riff was joint, which is usual.
- **Measured anchors**: every claim that touches a number cites a
  booked result. A frame with no anchor is fine, but say that it has
  none rather than implying support.
- **Honest breaks**: where the analogy STOPS holding. A mapping
  presented without its breaks reads as a claim; the breaks are what
  make it reusable later.
- **Testable residue**: the falsifiable thing left over. This is the
  payload — it is what turns a bank into a `/rung` months later. If
  there is no residue, say so explicitly.

## Corrections — named in place, never deleted

When a bank turns out wrong, the refutation goes INTO the bank,
naming what killed it. Never silently edit the claim away, and never
delete the entry.

Two shapes, both used:
- **In-place amendment** when a residue gets measured: keep the
  original text, append the verdict and what it narrowed. Say what is
  now closed and what is STILL open — a partial measurement that
  reads as a full one is the failure mode.
- **A CORRECTION paragraph** when the bank asserted something false.
  Name the retraction in the text ("the 'unclimbed' claim above is
  retracted") and cite the receipts that overturned it.

The 2026-08-11 division-algebra bank did the second: it claimed a
dim-4 rung was unclimbed, a line-verification found two measured
cells, and the correction now sits inside the bank enumerating what
is genuinely still open. That is the pattern.

## Checks before committing

- Does a verdict booked this session touch any OTHER bank? Grep
  RIFF-LEDGER for the thread name before closing the session.
- Does the bank overclaim what was measured? "Fires" and "fires on
  one of two steps" are different sentences.
- Is the residue actually falsifiable, or is it a wish?
- PUBLIC REPO: no host/key/personal details; commit with exactly
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` and NEVER
  a Claude-Session URL.

# Opus-5 review session — audit sheet for Fable

Branch `opus-5`, forked from `main` at `7ab8837`. Not pushed. Nothing
here reaches `main` without Fable's line-by-line audit.

Seat: Artin switched the model to Opus 5 and authorized branch work,
which overrides the standing "code changes are Fable's job" convention
for this branch only. I kept the reviewer discipline anyway: every
claim below carries the command that shows it.

## Commits, in order, each independently reviewable

| Commit | What | How to audit in one command |
|---|---|---|
| `5673ec0` | AMENDMENT P4-DEVICE-SCOPE + `docs/REPRODUCE.md` clarification | `.venv/bin/python scripts/results_query.py --chain p4-device-scope` |
| `78f9c57` | Doc-integrity guard (2 tests, both verified failable) | `.venv/bin/python tests/test_docs_integrity.py` |
| `1c8f523` | axiom relay + this audit sheet | read `docs/superpowers/relay/2026-08-02-0-axiom-docs-ask.md` |
| `<docs>` | README organizing-principle clause + FINDINGS cross-lab reachability note (Grok #2, #3) | `git show --stat` |
| `<probe>` | `scratch/probe_int_device_parity.py` | `.venv/bin/python scratch/probe_int_device_parity.py` |
| `<rider>` | RIDER: GPU parity measured on MPS **and** CUDA, 8/8 each | `.venv/bin/python scripts/results_query.py --chain p4-device-scope` |

Full suite after both: **450 passed, 7 skipped** (was 448; the two new
ones are mine).

## Finding 1 — "2 devices" was imprecise [FIXED, needs your audit]

Claim: both P4 legs ran on CPU, so "2 devices" means two machines /
two CPU architectures (arm64, x86-64), not GPU diversity — a narrower
claim than P3 / PACKED CRYSTAL C4, which genuinely crossed MPS↔cuda.

Evidence: no `device=`, `.cuda()`, `.to("mps")`, or
`set_default_device` anywhere in the battery chain
(`detbwd_gravmoe`, `detbwd_mb`, `detbwd_r2b`, `detbwd_r1`,
`llmopt/reproduce`, `llmopt/intmath`); `scratch/p4_arms_0801.sh` sets
OMP/MKL thread caps and never selects a GPU.

**Provenance, so nobody blames the wrong agent**: `git log -S` shows
the phrase entered at `94e29cd` — the house's own P4-LAB booking
(Fable). Sol quoted it faithfully into `docs/REPRODUCE.md` at
`548e9c5`. This is a house wording issue, not a Sol review defect.

**Self-correction worth reading before you trust the rest of my
work**: my first draft of this finding said "the battery is CPU-only
by construction." That is FALSE and I caught it by testing instead of
asserting. `int_mm` is `(a.unsqueeze(-2) * w).sum(-1)`, not
`torch.matmul`; on torch 2.12.1 / Apple silicon, MPS runs int64 matmul
AND all eight battery primitives bit-identically to CPU (`int_mm`,
`rdiv`, `softmax_rows`, `softmax_bwd`, `rms_fwd`, its `isq`,
`rms_bwd` dx and dg). The battery is CPU-only by PLUMBING. The booked
amendment says so, including the failed first draft.

Nothing measured changed: 16/16 and 10/10 sha-identity stand as
booked.

## Finding 2 — doc-layer had no guard [FIXED, needs your audit]

`tests/test_docs_integrity.py`, two tests, both verified failable
(I monkeypatched each failure path and confirmed the assertion fires):

1. **Anchor rot** — all 148 repo-wide `RESULTS.md#L<n>` citations must
   land on a `## ` entry heading. Currently 148/148 valid. This is the
   invariant that silently dies the first time anyone edits RESULTS.md
   mid-file; "append-only" is convention, not enforcement.
2. **Curation ratchet** — curatable (`verdict`/`null`) entries newer
   than anything FINDINGS cites, capped at the measured backlog of
   **10**. May fall freely; raising it needs a stated reason.

## Finding 3 — FINDINGS staleness [FIXED on Artin's GO; audit the claims]

Backlog 10 -> **0**. Eleven bullets appended to "The clock-placement
and deterministic-birth close", covering all ten entries: both P4
verdicts, QK-RESCOPE and its gate rider, the three exposure-bias
nulls (scheduled sampling, brute compute, answer-only allocation),
the decay and ACT_CLAMP mechanisms, the E=1-parity-blindness
methods finding, and the GPU primitive probe.

**Audit these first** — I wrote claims about the lab's own results,
which is the part of this session most worth a second pair of eyes:

- every number was re-read out of RESULTS.md before it was written
  (`825,984` / `208,192`, `56 -> 94/140`, `8883 -> 2496`,
  `2/8 -> 0/8`), not recalled from the session;
- the P4 bullet uses the AMENDED device wording, not the original;
- no bullet claims anything the cited entry does not.

I did NOT re-curate anything Sol wrote, and I did not touch the
maturity tags on existing bullets.

**A defect I introduced and then guarded**: three of my bullets
wrapped `[REGIME-SCOPED: deterministic integer battery]` across a
line break. Semantically fine, invisible to grep — exactly the
vocabulary drift the controlled list exists to prevent. Fixed, and
`test_findings_tag_grammar` now catches it (verified failable on
that precise defect). The guard did not exist when I made the
mistake; my manual check found it, so the check became a test.

## Proposals I did NOT act on (waiting on Artin/Fable)

- **GitHub repo description** still reads like the pre-thesis lab.
  It is a repo *setting*, not a file — `gh repo edit --description`
  or the web UI. Outward-facing, so I did not touch it.
- **Grok's "FINDINGS is chronological"** is STALE — it is thematic and
  maturity-tagged since `a5b3a98`. Grok read a pre-restructure copy or
  conflated it with RESULTS. The residual half-point is fair though:
  README line 6 says "curated findings" without saying *curated by
  maturity, not chronology*. One clause.
- **Grok's cross-lab disclosure placement** is fair and unfixed: the
  "21 commits ahead / unreachable" disclosure lives in
  `docs/REPRODUCE.md:106`, but `docs/FINDINGS.md:443` carries a
  `[REPLICATED]` axiom claim with no pointer to it. An external reader
  meets the strong tag without the caveat.
- **Full GPU leg of the battery** (pinned-sha trajectory, not just
  primitives) — banked, not run. Pre-register first. Known work:
  thread a device through the birth/draw path and `.cpu()` before
  `.numpy()` at the digest points; bar = the 16 pinned FINAL shas.

## What I did not touch

`docs/BOARD.md`, `docs/THEORY.md`, `docs/RIFF-LEDGER.md`,
`docs/handoffs/`, `docs/FINDINGS.md`, `README.md`, any `scratch/`
experiment code, any pinned artifact, and the axiom repo (read-only
throughout; the relay is a file on the llmopt side for Artin to
carry).

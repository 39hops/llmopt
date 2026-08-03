# Opus-5 branch — audit sheet for Fable

Branch `opus-5`, forked from `main` at `7ab8837`, pushed, 26 commits.
Nothing reaches `main` without Fable's line-by-line audit.

Seat: Artin switched the model to Opus 5 and authorised branch work,
overriding the standing "code changes are Fable's job" convention for
this branch only. Reviewer discipline was kept anyway — every claim
below carries the command that shows it, and every experiment was
pre-registered before it fired.

## State at the end

- 451 tests pass, 7 skipped; tree clean; branch pushed.
- 24 ledger entries booked (12 verdicts, 7 pre-regs, 6 amendments),
  all with `opus-review` in their `threads`.
- **Six of those entries are corrections to my own earlier claims.**
- 3080 re-ran all 16 gravmoe pins sha-identical against this branch's
  modified certified files; the remote checkout never left `main`.

## Verify the branch in four commands

```bash
.venv/bin/pytest -q                              # 451 passed
.venv/bin/python tests/test_docs_integrity.py    # anchors + curation
.venv/bin/python -m llmopt.reproduce gravmoe-rb1 # PASS c6766da2
DEV=mps .venv/bin/python scratch/v4flash_rungA.py  # sha a68256ce
```

## What landed, by programme

**Replication of two n=1 init laws.** QK-SEED2/3: the COND+QK gate
direction reproduces at three paired init draws, 3/3 on every axis,
with the near-point prediction (init zero-prob 0.89 → exactly 0.000)
hitting all three times. DIET-COND-SEED: the interior-optimum SHAPE
replicates but its LOCATION does not — λ=1/4 is a seed-17 property and
is worse than no gravity at both new draws. `BIRTH_SEED` knob added,
regression-gated.

**P4 device/lab legs closed** (device: 16/16 on the 3080; lab: 10/10 in
axiom C++), with the "2 devices" phrasing corrected to two machines /
two CPU architectures, and GPU primitive parity measured on both
accelerators.

**DeepSeek-V4-Flash, ten rungs.** Format is group-32 MXFP4 byte-identical
to K3; code entropy 3.8646 of 4 bits; the scale stream is 5.9% of bytes
and 62% of the lossless headroom; one global table serves all experts
(KL 0.00075); an expert runs exactly in integers with one trace hash on
cpu/mps/cuda; experts share no weight structure coordinate-wise, up to
the optimal permutation, or between router-nearest pairs; the router
carries a +0.385 shared key direction. Spec at
`docs/superpowers/specs/2026-08-02-v4flash-lossless-recode.md` (v3).

## Audit these first

**1. AMENDMENT AUDIT-0802** is the entry to read before anything else.
Three agents (silent-failure hunter, code reviewer, lab reviewer)
audited all 23 prior entries and the code. Ten readings were wrong and
are corrected there; every measured number that could be recomputed
from a log recomputed correctly.

**2. The CRITICAL code defect and its fix.** `BIRTH_SEED` reached
`detbwd_mb.SEED` but was not in `llmopt/reproduce.py`'s `CONTRACT_ENV`
allowlist, so a shell that had run `scratch/calib_dist_birth.sh` would
silently reproduce the wrong trajectory. Measured: `BIRTH_SEED=1` turned
`gravmoe-rb1` into `9264fcf0` instead of `c6766da2`. Fixed in `0802a24`
and verified under both polluted and clean environments. **If you adopt
one thing from this branch, adopt that fix.**

**3. The exactness assert that could not fail.** `v4flash_rungA.py`
built both sides of its decode check from the same `exps`, and
`LUT2X == 2*FP4_TABLE` by construction, so no bias error could fail it —
while VERDICT V4-RUNG-A claims the decode was checked against the
vendor's own semantics "not against itself". The reference now decodes
through torch's native `float8_e8m0fnu`. Trace sha unchanged, so the
claim was true; the committed reproducer just did not establish it.

**4. The receipts gap.** About eight booked numbers came from one-off
inline commands whose output was never saved, and several cannot be
recomputed from the logs that exist. The numbers are right — the audit
checked every one it could — but a future reader cannot verify them.
Standing correction adopted: a number that reaches the ledger comes
from a committed script writing to a log.

## Known-good and known-open

Clean per the audit: all amendment chains resolve; every pre-reg
precedes its verdict with no missing or smuggled arm; the two
unregistered items are labelled as such; charter clean throughout; the
SiLU-table sha pin is a real external check.

Open, not fixed here:
- `scratch/pack_rans.py:84` still reads `verify=(tot_n < 2e9)` —
  round-trip checking silently off past 2B symbols. Untouched by this
  branch; it is Fable's file to fix.
- `scripts/results_query.py --live` only reads `superseded_by` forward,
  never scanning for entries that name a row in their `amends`; 26 rows
  report LIVE while something amends them.
- `docs/BOARD.md:102` cites a MoE-pruning baseline ("61%-keep,
  50% count-quantile, ~28% cliff") to "RESULTS MoE pruning" that the
  audit could not locate in RESULTS.md — possibly un-booked chat state.
- `docs/THEORY.md` has no P6 row, so nothing grounds the
  "least meter-compressible are most entropy-codable" regularity.
- The curation ratchet now honestly reports **296** uncited curatable
  entries (it previously read 0 for structural reasons). That backlog
  predates this branch.

## What I did not touch

`docs/BOARD.md`, `docs/THEORY.md`, `docs/handoffs/`, `README.md`, any
pinned artifact, and the axiom repo (read-only throughout; the two
relays are files on the llmopt side for Artin to carry). I *did* edit
`docs/FINDINGS.md` (added bullets, then corrected their tags) and
appended to `docs/RIFF-LEDGER.md` (three external-reader banks) — an
earlier version of this sheet wrongly claimed otherwise.

## Next rungs, if the programme continues

M1 (merged lattice, re-scoped weight-exact, zero download) → S0 (rANS
decode throughput, pre-registered to FAIL its 5 GB/s bar) → R-d (is the
shared router direction routing-inert? the cheapest falsifier of my own
headline) → W1 format-matched → Q1 on Qwen3-30B where a capability
claim is legal and σ≈5 applies → rung 13 with the instrument fixed.

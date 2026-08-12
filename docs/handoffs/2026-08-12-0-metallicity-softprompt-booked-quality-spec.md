# Handoff 2026-08-12-0 — two rungs booked, a sampler defect found, and the code-quality program spec'd

Seat: Opus 5, Mac. HEAD at close: see the final commit in this session's
run (`3b4909a` when this file was written). 3080 idle, window was open
to 17:00 EST. pytest 653 passed / 7 skipped throughout.

## What landed

**VERDICT METALLICITY-1** (`055a881`). All 16 cells. Ignition widths:
w(z3 verified) = 56; w = none for z2 polluted, z1 Pop-III-duplicated,
and z0 token-shuffled vacuum. P-METALLICITY fires on **one of its two
ordering steps** — z1 and z2 tie at "none", so pollution-versus-
duplication is untested. P-VACUUM holds hard: z0 is exactly 0 at every
width, birth loss 2.2576 against ~0.51-0.61, and it was the best-fed
grade after strict-encode (133,842 sequences). Registered prior correct
3/3. w(z3)=56 is threshold-fragile at n=1 (z3_d48 = 6 against a
threshold of 8). Receipts frozen in-repo at `logs/metallicity1/`.

**VERDICT SOFT-PROMPT-1: INSTRUMENT-INVALID** (`06fd10f`). plain 10 /
random-prefix 1 / trained-prefix 4. P-MECHANISM-CLEAN fires at
|1-10| = 9 > 7, so bar 1 is unreadable by the pre-reg's own clause. No
capability claim either way, and the NO-FIRE prior is NOT scored — an
invalid instrument cannot confirm a prediction.

**AMENDMENT SOFT-PROMPT-1-SAMPLER** (`a4190b6`) — the useful part. The
model is bit-exact (zero logit difference, identical argmax, all 59
tensors equal, emb/head not tied). The defect is the sampler at
`scripts/step_grpo_micro.py:65`: eight appended zero-probability logit
columns leave the distribution identical but change how many random
values `torch.multinomial` consumes. Single draws from a fresh generator
agree 200/200; sequential draws from one generator diverge at the
SECOND draw, and `sample_wave_lp` reuses one generator for up to 120
tokens. **House fence registered**: gate numbers are comparable only
across runs whose sampler saw the same number of categories — padded
vocab, added special tokens, reserved slots all break it, and neither
the weights sha nor a logit comparison detects it. Probe at
`scratch/softprompt_sampler_probe.py`.

**Skill review** (`eca4f61`): 4 skills patched, `/riff` created.
**CLAUDE.md** (`2671030`): the FINDINGS ratchet and the seven house
skills were both absent; now surfaced.

**Code-quality program spec** (`9c3131a`, extended `3b4909a`) —
`docs/superpowers/specs/2026-08-12-code-quality-program-design.md`.
DESIGN, unapproved, nothing implemented.

## The spec in one paragraph

Provenance is stored as a filesystem property when it should be data.
Three measured findings drive it: (1) the 2026-08-05 extraction copied
bodies and never migrated callers, so six of twelve `llmopt/lab`
modules have zero production consumers while `scripts/step_grpo_micro.py`
carries 93 importers; (2) the freeze does not deliver reproducibility —
96 of 178 cited files import package modules that still change, and five
drivers `sed`-patch a frozen probe into `/tmp` and execute that, so some
booked numbers came from programs that exist nowhere in git; (3)
reachable history is 984 MB while `.git` is 47 GB, the excess being
unreachable objects rather than history. Decisions taken by Artin:
shim-plus-provenance-index, reclaim the 46 GB now after safety checks,
tiered lint with `scratch/` report-only.

## Two conditions that bite on the next booking

1. **FINDINGS ratchet headroom is ZERO** (320 of 320). The next verdict
   reddens CI unless its bullet lands in the same commit, or ~20 old
   entries get curated down first.
2. **README ledger drift**: README states 187 (37/42/70/35/3); FINDINGS
   actually holds **191** (37/43/73/35/3). It moved during this session.
   Spec Phase 7b generates it instead.

## Next session

Start: this handoff, then BOARD, then the RESULTS tail. If the task is
the quality program, the spec is the entry point and the next step is
`writing-plans` over it — deliberately NOT done this session, so it gets
a fresh context.

Recommended first working batch (spec Phase 0 + 7b): ratchet headroom,
`git gc`, the CODEMAP citation mask, the five `/tmp` drivers, generated
README regions. None of it changes a line of behavior.

## Open decisions for Artin (spec §9)

1. Device precedence for `pick_device()` — MPS-first or CUDA-first? The
   two current spellings disagree, so it is a product call.
2. Ratchet — curate ~20 entries down, or raise `MAX_UNCURATED` with a
   reason? Recommendation: curate.
3. `llmopt/common/` versus `lab/` as the home for shared helpers.
   Recommendation: a new `common/`.
4. Phase 5 `lab/` split timing — fold into the next BOARD housekeeping
   gate, or schedule it?

## Also standing

- Front-facing voice rule (spec §5b): never narrate a deliberation;
  methodological `we` stays in the paper (66 instances, standard
  register); README/REPRODUCE/FINDINGS already carry zero. "the house"
  appears 9 times in visitor-facing docs and is in-group jargon.
- LinkedIn text fact-checked this session; two claims failed and were
  corrected before posting (the mechanics arm "failed to transfer",
  it did not simply invert; and "selection, not sparsity" replaced an
  uncited coverage/recall claim).
- Relay 2026-08-11-1 remains DRAFT/UNSENT — Artin sends manually.
- v4anat finished clean on the Mac (rc=0), exploration-grade, unbooked.
- Banked and unqueued: BASIN-CENSUS-1, born-quaternion Q9 births plus
  the 1/2/4/8 ladder pre-reg, ignition-mass cell B (true raw farm
  stream), GROW-DECOMP n=3, MPS decay probe arms.

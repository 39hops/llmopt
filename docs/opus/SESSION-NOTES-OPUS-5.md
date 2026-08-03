# Opus-5 branch — audit sheet for Fable

Branch `opus-5`, forked from `main` at `7ab8837`, pushed, **30 commits**.
Nothing reaches `main` without Fable's line-by-line audit.

Seat: Artin switched the model to Opus 5 and authorised branch work,
overriding the standing "code changes are Fable's job" convention for
this branch only. Reviewer discipline was kept anyway — every claim
below carries the command that shows it, and every experiment was
pre-registered before it fired.

## State at the end

- 451 tests pass, 7 skipped; tree clean; branch pushed at `e58b9bb`.
- **30 ledger entries** (13 verdicts, 9 pre-regs, 7 amendments, 1 null),
  all carrying `opus-review` in their `threads`.
- **Ten of those entries correct earlier claims of my own.** That ratio
  is the most useful thing on this sheet: read the amendments first.
- 3080 re-ran all 16 gravmoe pins sha-identical against this branch's
  modified certified files; the remote checkout never left `main`.

## Verify the branch in five commands

```bash
.venv/bin/pytest -q                              # 451 passed
.venv/bin/python tests/test_docs_integrity.py    # anchors + curation
.venv/bin/python -m llmopt.reproduce gravmoe-rb1 # PASS c6766da2
DEV=mps .venv/bin/python scratch/v4flash_rungA.py  # sha a68256ce
.venv/bin/python scratch/v4flash_census.py       # 166.879 GB artifact
```

## Audit these first — the four that change something

**1. `llmopt/reproduce.py` — the CRITICAL fix, adopt this one regardless
of what you do with the rest.** `BIRTH_SEED` reached `detbwd_mb.SEED`
but was not in the `CONTRACT_ENV` allowlist, so a shell that had run
`scratch/calib_dist_birth.sh` would silently reproduce the WRONG
trajectory. Measured: `BIRTH_SEED=1` turned `gravmoe-rb1` into
`9264fcf0` instead of `c6766da2`. Fixed in `0802a24`, verified under
both polluted and clean environments.

**2. AMENDMENT AUDIT-0802** (`RESULTS.md`, 2026-08-03) — ten ledger
corrections from three agents auditing the first 23 entries. Wrong
extrema, a false enumeration presented as "measured, not inferred", an
over-awarded `[REPLICATED]`, and a systematic receipt gap.

**3. AMENDMENT RUNGD-0803** — four reviewers on the V4-RUNG-D booking,
and **the fence was the thing that was wrong**. Read this before any V4
claim. Six substantive corrections, listed below.

**4. RECEIPT V4-CENSUS** — the spec's load-bearing memory argument was
false. See "What the V4 programme actually established".

## What landed, by programme

**Replication of two n=1 init laws.** QK-SEED2/3: the COND+QK gate
direction reproduces at three paired init draws, 3/3 on every axis, with
the near-point prediction (init zero-prob 0.89 → exactly 0.000) hitting
all three times. DIET-COND-SEED: the interior-optimum SHAPE replicates
but its LOCATION does not — λ=1/4 is a seed-17 property and is worse
than no gravity at both new draws. `BIRTH_SEED` knob added,
regression-gated.

**P4 device/lab legs closed** (device: 16/16 on the 3080; lab: 10/10 in
axiom C++), with the "2 devices" phrasing corrected to two machines /
two CPU architectures, and GPU primitive parity measured on both
accelerators.

**DeepSeek-V4-Flash, thirteen rungs.** Format is group-32 MXFP4
byte-identical to K3; code entropy 3.8646 of 4 bits; the scale stream is
5.9% of bytes and 62% of the lossless headroom; one global table serves
all experts (KL 0.00075); an expert runs exactly in integers with one
trace hash on cpu/mps/cuda; experts share no weight structure
coordinate-wise, up to the optimal permutation, or between
router-nearest pairs. Then rungs D, S0, D2 and the census, which
corrected most of the readings the earlier rungs carried. Spec at
`docs/superpowers/specs/2026-08-02-v4flash-lossless-recode.md` (v3,
amended 2026-08-03).

## What the V4 programme actually established (post-correction)

Read this instead of the earlier verdict prose, which overstated in
three places.

- **The router's shared key direction is real and does a MINORITY of the
  work.** All 32,640 key pairs are positively aligned and every key
  shares a large component along one direction `u` — but remove `u` and
  only **35-37%** of residual pairs stay positive, so the positivity
  headline was carried entirely by `u`. Whether `u` steers routing
  depends on the input model: 97% agreement survives deleting it under
  isotropic input, 12-27% under input aligned with it. Inverting the
  shipped load-balancing bias excludes strong alignment and puts `u` at
  roughly **12-17% of top-6 selection** (VERDICT V4-RUNG-D2).
- **Entropy coding is an archive, not a runtime.** Byte-lossless saves
  **8.3%** (not the spec's 15.6%, which was the merged-lattice rate) and
  decodes at 38.1 MB/s single-threaded — ~12x under this machine's
  measured 3.5-4.5 GB/s NVMe even with all 11 cores.
- **V4-Flash cannot run here, but not for the reason the spec gave.**
  The dense path is **9.44 GB**, not 27 GB; 19.3 B of the claimed "27B
  dense params" are three MTP blocks holding full 256-expert layers. The
  binding constraint is simpler: **166.879 GB artifact vs 31 GiB free
  disk**, unfixable by deleting everything deletable.
- **Per-token expert traffic is 4.53 GB** (43 x [6 x 13.37 + 25.17
  shared]), ceiling ~1.10 tok/s at batch 1 — the always-on shared expert
  was missing from every earlier figure.

## The corrections, so you can grade the reasoning not just the code

Six from AMENDMENT RUNGD-0803, in severity order:

1. **The fence was softened in the direction the mechanism forbids.**
   The verdict said a large `<u,x>` "would move all 256 scores
   together". It would not — `sd(c)` is 0.06-0.10, so it scales the
   CONTRAST linearly. Measured: agreement collapses to 12-27%.
2. **The isotropic null assumed its own answer.** `E<u,x> = 0` is
   exactly the condition under which a shared direction cannot matter.
3. **The operation was not a level removal.** It also cuts per-expert
   gain 3.5-20.2%; the clean level arm gives 99.3-99.7%.
4. **"Removes ~99% of the logit mean" was a mean-zero sampling
   artifact** whose sign flips across seeds (24.6-99.4% by seed). The
   level lives in the weights: `||mean key row||` collapses 95.4-97.0%,
   no draw required.
5. **The registered row was not reproducible from its own fields** — one
   RNG threaded through the scale loop, so the registered arm depended
   on the unregistered probe running first (0.9744 vs 0.9705).
6. **Absolute cosines were re-logged**, re-introducing the exact defect
   AUDIT-0802 flagged, in the cell written to pay that debt off.

**The pattern across all of them**, worth more than any single fix: a
bound derived under an assumption travelled without the assumption. "27B
by subtraction", "1.45 tok/s at zero reuse", "+0.385 shared direction"
and "97% inert under isotropy" are all correct as computed and wrong as
quoted. Neither re-checking arithmetic nor a third reading catches this
class — only computing the same quantity a DIFFERENT way does (bottom-up
instead of by subtraction; in the weights instead of by sampling).

## Known-good and known-open

Clean per the audits: all amendment chains resolve; every pre-reg
precedes its verdict with no missing or smuggled arm; unregistered arms
are labelled as such in RESULTS, the scripts and the logs; charter clean
throughout; the SiLU-table sha pin is a real external check.

Open, none of it introduced here except where noted:
- `scratch/pack_rans.py:84` still reads `verify=(tot_n < 2e9)` —
  round-trip checking silently off past 2B symbols. Fable's file.
- `llmopt/moe/offload.py:6-7` asserts "routing is heavy-tailed in
  practice, so a resident set far smaller than E catches most traffic".
  That is an unmeasured docstring claim and it is FALSE for any model
  using aux-loss-free balancing, which is what `noaux_tc` does.
- `scripts/results_query.py --live` only reads `superseded_by` forward,
  never scanning for entries naming a row in their `amends`; rows report
  LIVE while something amends them.
- `docs/BOARD.md` has **zero** occurrences of `V4`, `v4flash` or `RUNG`
  and its header is still dated 08-01 — the whole 13-rung ladder is
  absent from the live board. I did not touch BOARD (Fable's file).
- `docs/handoffs/` stops at `2026-08-01-4`; there is no handoff for
  either day of this branch. This sheet is the substitute.
- `docs/THEORY.md` has no P6 row and no V4 row.
- Curation ratchet honestly reports **296** uncited curatable entries
  (it read 0 for structural reasons before this branch). Predates it.

## What I did not touch

`docs/BOARD.md`, `docs/THEORY.md`, `docs/handoffs/`, `README.md`, any
pinned artifact, and the axiom repo (read-only throughout; axiom's
newest relay `2026-08-02-1` is already answered — AMENDMENT
AXIOM-PUBLICATION at `RESULTS.md:15283`, and `docs/REPRODUCE.md:102-104`
already cites the pinned verifier commit and path, so nothing is owed
there). I DID edit `docs/FINDINGS.md` and appended to
`docs/RIFF-LEDGER.md` (three external-reader banks).

## Next rungs, if the programme continues

**F1 — the real forward (the only way to close the open question).** The
9.44 GB dense path is storable here, so real layer-22 hidden states are
obtainable in principle, which would convert V4-RUNG-D2's bound into a
measurement. Cost is honest and large: the Indexer (Hadamard rotation,
simulated fp4 activation quant), the Compressor and its KV cache, MLA
with sliding window and attention sinks, and hyper-connections with 20
Sinkhorn iterations all have to be right, or the hidden states are wrong
— which is worse than not measuring. mlx-lm 0.31.3 ships no
`deepseek_v4`. Scope it before starting.

Then: **M1** (merged lattice, weight-exact, zero download) → **W1**
format-matched, now with its comparators free (the shared expert uses
128x128 blocks, measured) → **Q1** on Qwen3-30B-A3B where a capability
claim is legal and sigma ≈ 5 applies → **13** with the instrument fixed
(top-k energy subspaces, empirical null).

**H1** is cheap and now fully specified: all three hash layers carry
their own `tid2eid` table `[129280, 6]`, so the exact pairwise
co-activation graph follows from any corpus's unigram distribution with
zero inference. Fence it as designed-not-learned; it is combinatorics,
not a throughput lever (100% hit rate on 3 of 43 layers is a 7.0%
traffic cut).

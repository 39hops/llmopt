# Opus-5 branch — audit sheet for Fable

Branch `opus-5`, forked from `main` at `7ab8837`, pushed, **32 commits**
(this sheet commits last, so it cannot name its own HEAD).
Nothing reaches `main` without Fable's line-by-line audit.

Seat: Artin switched the model to Opus 5 and authorised branch work,
overriding the standing "code changes are Fable's job" convention for
this branch only. Reviewer discipline was kept anyway — every claim
below carries the command that shows it, and every experiment was
pre-registered before it fired.

## State at the end

- 451 tests pass, 7 skipped; tree clean.
- **31 ledger entries** (13 verdicts, 9 pre-regs,
  8 amendments, 1 null), all carrying `opus-review`.
- **Eleven of them correct earlier claims of my own, across THREE audit
  rounds** — and each round found errors the previous one missed,
  including inside its own corrections. That ratio is the most useful
  thing on this sheet: read the amendments first, newest first.
- **Unreceipted, do not rely on it**: an earlier version of this sheet
  said the 3080 re-ran all 16 gravmoe pins against this branch's modified
  files. No receipt exists, and the two booked 3080 runs both state the
  remote checkout was on `main` — which cannot run this branch's modified
  `detbwd_mb.py`. Two pins (`gravmoe-rb1`, `gravmoe-grb1`) ARE verified
  locally against the branch. Re-run the other 14 yourself.

## Verify the branch in five commands

```bash
.venv/bin/pytest -q                              # 451 passed
.venv/bin/python tests/test_docs_integrity.py    # anchors + curation
.venv/bin/python -m llmopt.reproduce gravmoe-rb1 # PASS c6766da2
DEV=mps .venv/bin/python scratch/v4flash_rungA.py  # sha a68256ce
.venv/bin/python scratch/v4flash_census.py       # 166.879 GB artifact
```

## Audit these first — the four that change something

**1. `BIRTH_SEED` — a MATCHED PAIR, adopt or reject together.** This
branch CREATED the knob (`ded6a5f`, `scratch/detbwd_mb.py`) and then
fixed the sanitizer that had to know about it (`0802a24`,
`llmopt/reproduce.py`'s `CONTRACT_ENV`). On `main` the knob does not
exist, so `reproduce.py` alone is a no-op and `detbwd_mb.py` alone
silently breaks reproduction. Measured: with the knob and without the
allowlist entry, `BIRTH_SEED=1` turns `gravmoe-rb1` into `9264fcf0`
instead of `c6766da2`. (An earlier version said `calib_dist_birth.sh`
could pollute a caller's shell — it uses per-command prefixes, not
`export`, so it cannot. The in-code comment carries the same error.)

**2. AMENDMENT AUDIT-0802** (`RESULTS.md`, 2026-08-03) — ten ledger
corrections from three agents auditing the first 23 entries. Wrong
extrema, a false enumeration presented as "measured, not inferred", an
over-awarded `[REPLICATED]`, and a systematic receipt gap.

**3. AMENDMENT FINAL-0803, then AMENDMENT RUNGD-0803** — read them in
that order, newest first. RUNGD-0803 found that RUNG-D's fence was the
thing that was wrong; FINAL-0803 then found errors inside RUNGD-0803
(two arithmetic, one wrong extremum, one statistic that cannot support
its prose) and RETRACTED VERDICT V4-RUNG-D2's headline outright. Read
both before any V4 claim.

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
  only **34.8-36.5%** of residual pairs stay positive, so the positivity
  headline was carried entirely by `u`. (That raw all-pairs-positive
  claim is itself still unrecomputed — `v4_router.jsonl` logs only
  ABSOLUTE cosines — so its receipt debt is only PARTLY paid, contrary
  to what VERDICT V4-RUNG-D + S0 says.) Whether `u` steers routing
  depends on the input model, and the attempt to settle that by
  inverting the shipped load-balancing bias FAILED (VERDICT V4-RUNG-D2,
  retracted by AMENDMENT FINAL-0803): a shuffled-bias null excludes the
  same inputs, so the bound is key geometry, not the balancer. The
  share is bounded to **7-21%** and **UNLOCATED**.
- **Entropy coding is an archive, not a runtime.** Byte-lossless saves
  **8.3%** (not the spec's 15.6%, which was the merged-lattice rate) and
  decodes at 38.1 MB/s single-threaded — **8.4-10.7x** under this
  machine's measured 3.5-4.5 GB/s NVMe even with all 11 cores. (The
  ledger's "~12x" is against 5 GB/s, the rate the same sentence
  disclaims; corrected in AMENDMENT FINAL-0803.)
- **V4-Flash cannot run here, but not for the reason the spec gave.**
  The ALWAYS-ON dense path is **8.85 GB** (the census headline of 9.44 GB
  includes 0.595 GB of non-expert weight inside the three MTP blocks,
  which are not always-on), against 27 GB claimed; 19.3 B of the claimed
  "27B dense params" are MTP blocks holding full 256-expert layers. The
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
   gain **3.0-20.2%** (the ledger's 3.5% low end is layer 40's alone);
   the clean level arm gives 99.28-99.74%.
4. **"Removes ~99% of the logit mean" was a mean-zero sampling
   artifact** whose sign flips across seeds (24.6-99.4% by seed). The
   replacement — `||mean key row||` collapsing 95.4-97.0%, no draw
   required — is correct but weaker than its prose: it is identical for
   the RANK1 and LEVEL arms by algebra, so it cannot support the
   "level, not contrast" reading (AMENDMENT FINAL-0803 item 7).
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

Clean per the audits: **9 pre-regs for exactly 9 plain verdicts, 1:1**,
with all 5 unregistered entries labelled RECEIPT / RIDER / OBSERVATION /
READING FENCE in RESULTS, in the scripts, and via a `registered` boolean
in the logs; charter clean throughout; `scratch/v4flash_ref/silu_tab.pt`
is byte-identical to the gitignored checkpoint, so that transport
survives a fresh clone.
NOT clean, both found in the third audit round and both now fixed: two
index references on this branch did not resolve (`results_query --chain`
returned the amendments alone), and D2's shuffled-bias null arm was run
and logged but omitted from its verdict. An earlier version of this
sheet asserted the opposite of both.

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
  against a ratchet of 300 — 4 slack. The BACKLOG predates this branch;
  the instrument does not (written, read 0, and fixed all on this branch).
- `logs/opus/` is **gitignored**, so no V4 or QK receipt reaches a fresh
  clone. Most are regenerable (the scripts re-fetch by byte range; the
  battery is sha-pinned). Two are uncheckable even here: the "all 32,640
  pairs positive" sign result, and the per-seed logit-mean sweep quoted
  in AMENDMENT RUNGD-0803 item (3).
- Relay `2026-08-02-1` is cited by this sheet, by AMENDMENT
  AXIOM-PUBLICATION and by RIFF-LEDGER, and is **not in the repo** — only
  the outgoing `-0` was committed. It lives in `~/code/axiom/docs/relay/`.
- `GLOSSARY.md` defines none of the V4 vocabulary (MXFP4, MTP, shared
  expert, `noaux_tc`, rANS, MLA, Sinkhorn, Indexer, Compressor), and
  nothing in `BOARD.md` or `docs/handoffs/` points at this sheet.

## What I did not touch

`docs/BOARD.md`, `docs/THEORY.md`, `docs/handoffs/`, `README.md`, any
pinned artifact, and the axiom repo (read-only throughout; axiom's
newest relay `2026-08-02-1` is already answered — AMENDMENT
AXIOM-PUBLICATION at `RESULTS.md:15283`, and `docs/REPRODUCE.md:102-104`
already cites the pinned verifier commit and path, so nothing is owed
there). I DID edit `docs/FINDINGS.md`, `docs/REPRODUCE.md` (24 lines),
`README.md` (5 lines, `89bfc19`), `tests/`, `scripts/INDEX.md` and
`scratch/detbwd_mb.py`, and appended ONE bank to `docs/RIFF-LEDGER.md`.
Earlier versions claimed README was untouched and that there were three
banks; both were wrong.

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
not a throughput lever: 100% hit rate on 3 of 43 layers cuts 3/43 of
ROUTED traffic, which is **5.3%** of the corrected per-token total once
the always-on shared expert is counted. An earlier version said 7.0%,
which credits those layers with shared-expert bytes a cache cannot
save.

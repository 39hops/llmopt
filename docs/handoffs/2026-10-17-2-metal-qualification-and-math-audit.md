# Handoff 2026-10-17-2: Metal path qualified against the CPU arm C census; openai/math audit closed

Session: Fable 5.1 on the Mac. Reviewers and research agents on Opus 5.5.
Previous handoff: 2026-10-17-1 (E0 census, Metal tower). Read BOARD first.

## What happened

1. Independent review of the six local Metal commits (3e7f457f through
   5d72c3cd plus the qualification driver commit d69c48a9). The reviewer
   verified codec offsets, nibble order, scale construction and the
   HF-exact q/k l2norm against source; 110 kernel and contract tests pass.
   It found one provenance defect and several documentation defects, all
   confirmed by the session model against the receipts and git objects
   and booked as AMENDMENT QWEN-METAL-TOWER-0-PROVENANCE (RESULTS L79362):
   receipt commit fields name the parent commit because every run
   happened in a dirty tree before its code was committed; the "remaining
   35 ms" sentence was wrong in sign (bench medians are synced per-call
   latencies, the tower pipelines launches and runs below their sum); the
   "2e-2 to 5e-5" pair mixed two metrics (same metric: 3.69e-2 to 5.14e-5
   per-row max relative); ranges dropped k_proj [1024, 5120] at 10 percent
   of roofline; the 370x is measured Metal against projected CPU, not a
   paired benchmark. Code changes from the review: module and test
   docstrings state the real parity bound and all four deviations, the
   reduction-order comment is honest (bound held empirically), the shape
   check uses the resident's own version, one test tightened to the suite
   bar, and the three drivers now record a provenance block (commit,
   tree_dirty, mlx and mlx-lm versions). No tolerance was loosened.

2. Metal-versus-CPU qualification over the whole frozen E0 census
   (scratch/qwen_metal_qual.py, liverun qwen-metal-qual-0, 32 min):
   OBSERVATION QWEN-METAL-QUAL-0 (RESULTS L79446). Token-identical to the
   CPU arm C rows at 4,722 of 4,722 positions including all 15 under 0.02
   nats of CPU margin; logits at the CPU top-16 within 1.0e-3 absolute;
   margins within 3.0e-4; teacher-referenced sup error within 6.6e-4 of
   the CPU value; zero agreement flips; replay yields identical at every
   k. Cached decode 256 of 256 on two prompts, logits within 2.1e-4 of the
   one-forward pass, 154 ms per step. Long context: decode 155 to 163 ms
   per token from 512 to 4,096; prefill 8.0 to 8.3 tokens per second;
   single-shot prefill memory 12.7, 15.5, 20.9, 31.9 GB at 512, 1,024,
   2,048, 4,096. E0 receipts were read and are unchanged (sha recorded);
   QWEN-METAL-TOWER-0 is unchanged. Receipt audit by an Opus
   receipt-auditor before the commit (see the commit message for its
   outcome).

3. openai/math audit (read-only; four Opus agents; all 372 families
   screened at title level, 14 inspected in depth; nothing downloaded
   into the repo, temp copies removed). No family yields a deployable
   lever for this model. Banked at the RIFF tail: the AUDITED entry with
   per-family reasons, six experiment banks and one DEAD bank. One desk
   finding reproduced by the session model and booked: OBSERVATION
   QWEN-E0-ERROR-FIELD-0 (RESULTS L79537), the flat 5.6-nat sup error is
   the extreme value of a 1.00-nat flat RMS error field; agreement by
   teacher margin falls far short of a Gaussian two-logit model, so the
   error on the deciding tokens is larger than the bulk.

## Numbers to carry

- Draft runtime on the Mac: 155 ms per token (6.44 to 6.47 tokens per
  second), flat to context 4,096; peak unified 10.3 GB at short context.
- Prefill: 8 tokens per second; single-shot memory quadratic in context,
  31.9 GB at 4,096 on a 36 GiB machine. Chunked prefill is required for
  long prompts (banked).
- Break-even at k = 4 with per-stream yield 3.43: 4 x 0.155 + c_v(4) <
  3.43 c_t.

## Shortlist (cheapest first, each with its falsifier; details in RIFF)

1. W4 x-traffic ceiling probe (timing-only kernel variant), then v4 with
   several rows per simdgroup. Falsified if removing x loads leaves 85
   percent of the time.
2. S16 lookup via simd_shuffle or threadgroup table. Falsified under 15
   percent gain.
3. In-tower per-op breakdown. Falsified if matmuls are under 60 percent.
4. Chunked prefill (memory fix, no rate change).
5. Head-error census on the teacher's top-64 from the qualification
   driver. Falsified if SD is at most 1.2 nats.
6. Prompt-lookup self-speculation desk census. Kill under 1.1 accepted
   per verify at k = 4.
7. DeltaNet drift at 4k against a float64 sequential reference.
   Falsified if superlinear.

## Timing rung for the bf16 teacher (still blocked on storage)

spec 2026-10-17-timing-rung.md stands. The draft side is measured now
(0.155 s per token). The teacher needs the 56 GB vendor shards: Mac free
space is about 58 GB (condition for restore was 76 GB and is unmet), the
3080 host has about 25 GB free on C: with the 266 GB vhdx. Options for
Artin: an external volume for the shards, or a storage decision on the
3080 side. Streaming plan if it runs on the Mac: 51.25 GB per teacher
token in bf16 (48.70 GB layers plus 2.54 GB head), pinned-resident subset
of 15 to 18 GB beside arm C (9.4 GB resident) cuts the per-token read to
33 to 36 GB; at the disk's read rate that is the teacher step cost to
measure, with the I/O bytes per step recorded as the spec asks.

## State

- Commits after 90fa3347 pushed in this session once the gate was green
  (see git log); 3080 fast-forwarded and asserted equal to origin/main.
- No live run. logs/qwen_metal_qual/C/positions.jsonl untracked (2.2 MB).
- Vendor weights absent on the Mac; arm C present at ~/qwen_whole0t/C and
  on the 3080; teacher records at logs/qwenteacher_v2.
- No speculative verifier, no MTP, no recompression, no new weights.

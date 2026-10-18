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

## Addendum (2026-10-17, late): W4 activation-traffic probe and decode breakdown

Artin's GO after the push: profile the step without summing synced
kernel times, run the W4 activation-traffic ceiling, decide whether
sharing activation loads across rows has a realistic advantage, project
the full-tower effect, do not implement v4. Booked as OBSERVATION
QWEN-METAL-DECODE-BREAKDOWN-0 with receipts logs/qmetal/
breakdown_01b8c177.json and traffic_probe_01b8c177.json; drivers
scratch/qwen_metal_breakdown.py and scratch/qmetal_traffic_probe.py.

Findings: the 154 ms step is 89 percent fused GEMV (W4 72.6, S16 body
55.8, head 9.5), DeltaNet core 12.1, attention core 1.7, norms 0.8,
framework floor 4.4; the ablation closes to 1 percent. Pipelined
per-launch W4 times (371 and 385 us) times 64 layers reproduce the W4
marginal exactly; the synced bench medians were 45 percent above the
in-tower cost. W4 is bound by re-reading the activation vector per row:
no-x variant 50 percent, no-weights variant 96 percent, x-and-reduce
only 93 percent. share4 probe 60 percent of v3 with bit-equal outputs;
share8 slower than share4. S16 does not move when its activation loads
are removed. Projection: about 28 ms off the step, 154 to 126 ms, 1.22x.

Verdict: W4 opportunity demonstrated; smallest justified design is v4 =
share4 with the TT tile and the v3 per-row reduction order (bit-identical
outputs, parity suite unchanged). Not implemented. Per Artin's order:
chunked prefill next, then v4 on GO. Teacher timing HOLD.

Pitfall recorded: MLX lazy pruning. A pure-constant stand-in for lm_head
made the whole tower dead code (5.8 ms step); stand-ins must keep a data
dependence. The invalid receipt was deleted.

## Addendum (2026-10-18): W4 v4 shipped, CI repaired

GO after the probe: CI fix (matplotlib in the tests job; the figures
regeneration inside check_source.sh imported it and the runner lacked
it, confirmed from the Actions log), W4 v4 as a separate selectable
generation, parity across tiles and row remainders, fresh v4
qualification on the 24 E0 prompts, A/B v3 versus v4 on matrices and
full cached decode at contexts 126 and 1,024, stand-in dependence check.

Outcome (OBSERVATION QWEN-METAL-W4-V4-0): v4 reproduces v3 bit for bit
(margin difference exactly 0.0 at every census position; A/B outputs
equal in all checks; greedy tokens identical); decode 154 to 128 ms per
token, 7.8 tokens per second, 1.21x measured against 1.22x projected;
v4 loses at tiles 4 and 8, so the shipped dispatch (default generation
4) runs v4 at tiles 1 and 2 and v3 above, verified to keep v3's prefill
rate. Receipts: logs/qwen_metal_qual/C_v4/summary.json,
logs/qmetal/ab_v3_v4_338745a7.json, ab_dispatch_check_v4default.json,
dep_check_338745a7.json. The v3 qualification and the E0 receipts are
untouched. QMETAL_VERSION=3 selects the control.

Next workstream per Artin: chunked prefill design (single-shot prefill
memory is quadratic in the block, 31.9 GB at 4,096 tokens). Teacher
timing HOLD.

# Handoff 2026-10-22-2: W4 T = 1 probe (NULL) and lossless BF16 feasibility (NEGATIVE)

Session: Fable 5.1 on the Mac. Auditors on Opus 5.5 (receipt, pre-reg, wording, twice for the W4 draft).
Previous handoff: 2026-10-22-1 (the generation 5 default flip). Read BOARD first.

## What happened

1. Artin's GO after the generation 5 promotion: two bounded tracks with
   separate pre-registrations and receipts. Track A, the W4 T = 1
   bottleneck probe in the S16 probe's shape with a 5 percent
   whole-model hurdle to be registered before any A/B. Track B, a new
   question: can the original BF16 weights be stored reversibly well
   under 16 bits per parameter, enough for the text model to fit the
   36 GB Mac, using only the two retained vendor shards.
2. Track A: scratch/qmetal_w4_probe.py (captured one-token activations
   from the qualified tower, diagnostics and nine candidates on a
   skeleton reproducing the shipped v4 sequence, oracle and bit-identity
   checks). Run twice under liverun (qwen-w4-probe-0 at 7dd4c367, then
   qwen-w4-probe-1 at 39ba1b51 after the receipt audit repaired the
   codebook stand-in and the loads-only variant, added the shipped
   kernel as its own row, an instrumented dispatch witness, a source
   sha pin and per-matrix rotation). OBSERVATION QWEN-METAL-W4-T1-0
   (RESULTS L82977): NULL. The v4 kernel runs at 64 to 68 percent of
   the roofline with no removable term above a fifth; the best candidate
   is a down_proj-only 3 to 5 percent, pooled flat; no A/B registered,
   nothing shipped.
3. Track B: PRE-REG LOSSLESS-BF16-FEASIBILITY-0 (L82860, bars 30 and 28
   GB on the class-weighted best-exact extrapolation, the H16 floor at
   30 GB, refuted above 32 GB) registered at a38c28d4 after reading
   dirty-tree smoke receipts (disclosed); receipt names fixed in place
   at 7dd4c367 before the chain (disclosed in the verdict). Chain under
   liverun bf16-lossless-0 at 7dd4c367 (census 2.7 min, benchmark 58.6
   min). VERDICT (L83153): NEGATIVE, REFUTED, all bars NO-FIRE: 10.47 to
   10.54 bits per weight of entropy in every sampled matrix, best exact
   coders at 10.5 to 10.6, extrapolation 35.4 GB (floor 35.25) against
   the 30, 28 and 32 GB lines; a lossless tower would also be 2.3x
   slower than arm C on traffic alone. No download, no loader.
4. Lean: elan's lean 4.33.0-rc2 is installed (not on PATH);
   scratch/leancheck/BF16Split.lean proves the DFloat11 split round
   trip and injectivity (bv_decide; its native-reflection axiom is
   recorded). The other B4 properties are listed in the verdict, not
   started.
5. Both tracks audited before booking (receipt audit of each new
   driver's first receipts, pre-reg audit of the B verdict, two wording
   audits of the A draft); every finding applied or disclosed.

## Decisions left to Artin

- Whether any W4 work continues: the kernel lever is closed at this
  resolution; the DeltaNet core (12.6 ms of the 100.2 ms step) is the
  largest non-GEMV term; the TT = 1 skeleton without the row mask and
  call wrapper (3.9 to 6.2 percent per launch, about 2 percent of the
  step) is below the hurdle and not pursued.
- Whether a LOSSY transform arm (Hadamard or rotation plus lattice
  codebooks, competing with arm C's 4 to 5 bits) is ever opened; banked
  in the RIFF ledger, not started.
- Retention of the two vendor shards outside the repo (unchanged).
- The release dumps from the previous handoff (2.2 GB untracked) and
  the smoke receipts under logs/qmetal/smoke/ and logs/bf16_lossless/
  smoke/ can be deleted at any time.

## Not done by design

No new vendor shards, whole-model compression, arm C replacement, BF16
teacher restore, fine-tuning, adapters, speculative verifier, lossless
model loader or Metal entropy decoder. No W4 A/B or census (no
candidate could reach the hurdle). xz, brotli and context-modelled
coders out of this round's scope.

## Lessons

- A diagnostic stand-in must cost what it replaces: the first W4 run's
  arithmetic stand-in cost more than the codebook gathers and read as a
  slowdown; a bitcast-and-convert stand-in measured them.
- The process-wide compiled-kernel set is not a dispatch witness once
  anything else has run in the process; instrument the lookup during
  the call instead.
- A pre-reg's receipt paths must be the paths the chain writes; name
  receipts without commit suffixes when the registration commit is not
  the run commit.

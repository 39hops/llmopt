# Sol answer-only gate design

**Date:** 2026-08-01
**Branch:** `sol/review-2`
**Status:** approved design; no experiment may fire before the Sol ledger pre-registration

## Question

Does reallocating the deterministic gravmoe battery's fixed training budget
from prompt/scaffold prediction to the free-run answer region improve the
oracle-scored gate at the same data, model, seed, optimizer, and 2,000 steps?

The closing GRAVMOE-BRUTE result ruled out width and schedule as converters on
the eight-row gate diet. The battery still trains next-token CE over the entire
complete row, although the gate supplies the prompt through `Step: ` and judges
only the continuation. The repository's legacy floating-point LoRA recipe uses
answer-only loss, but that allocation has not been tested in the deterministic
integer battery.

## Approaches considered

1. **Binary answer-only mask (selected).** Supervise only the answer suffix and
   one terminator. This changes one variable and directly answers whether loss
   allocation matters.
2. **Weighted scaffold/answer loss.** This could measure a dose curve, but a
   weight ratio introduces a search before the binary mechanism is established.
   It is eligible only as a follow-up to a binary-cell win.
3. **Scaffold warmup followed by answer-only training.** This mixes allocation
   with curriculum timing and cannot answer the first yes/no question cleanly.

## Contract

Add an opt-in `ANSWER_ONLY=1` mode to
`scratch/detbwd_gravmoe.py`. The default is off.

The treatment arm is otherwise the pinned G-RB1 contract:

```text
RJOB_LOCAL=1 GATE=1 COND=1 QK=1 LN=0 LD=1 STEPS=2000
```

The treatment changes only the loss-gradient tensor passed into `GMB.bwd`.
Forward logits, prompts, targets, optimizer, boost, rows, row order, free-run
decoding, and oracle scoring are unchanged. This is a contract fork and starts
a new trajectory-SHA lineage.

### Exact supervised region

Reuse `find_split(full, tok.encode("Step: "))` verbatim from the scheduled
sampling leg. At startup, print and pin the exact marker token IDs and each
training row's resulting split. `find_split` selects the position immediately
after the final marker.

For `tok_in = full[:T]`, `tgt = full[1:T+1]`, and split `s`, logit row `s-1`
predicts the first answer token. The mask therefore includes logit rows from
`s-1` through the row predicting the first terminating newline or EOS,
inclusive. It excludes all earlier scaffold targets and all repeated EOS
padding after that first terminator. Failure to find the marker or a terminator
is a hard error; the experiment must not silently change the supervised span.

With `ANSWER_ONLY=0`, the loss-gradient tensor must remain byte-for-byte equal
to the existing full-token expression.

## Arms and ordering

1. **REG:** run the unchanged G-RB1 contract with `ANSWER_ONLY=0`. It must
   reproduce FINAL trajectory SHA
   `1fcfd187873d980c7c082a56c0f380ce2c40a859eab1e8a0c9dcf6baa4853eca`.
   A mismatch voids the treatment arm.
2. **AO1:** run the same contract with `ANSWER_ONLY=1` after REG passes.

Both runs are Mac-local only with `RJOB_LOCAL=1`. Logs are written
incrementally under `logs/sol/`. No remote mode, SSH, WSL helper, or other
machine is used.

## Readouts

### Primary capability readout

- TRAIN symbolic solves out of 8, verified with the existing fork-deadline and
  SIGKILL SymPy oracle.
- HELDOUT symbolic solves out of 8 are reported but are not expected to improve
  in this memorization-scale cell.

### Token accuracy

Report two token-accuracy views for TRAIN and HELDOUT:

1. **Standard accuracy:** the existing full continuation readout, preserving
   comparison with prior gate logs (`94/140` for pinned G-RB1 TRAIN).
2. **Suffix-only accuracy:** positions from the first answer token through the
   first terminator, excluding repeated EOS padding. This is the graded
   decision readout because AO1 deliberately removes repeated-padding
   supervision.

The same new gate code must recompute both views for REG, so the suffix-only
comparison uses generations produced by the pinned G-RB1 trajectory rather
than an inferred value. No historical log reconstruction is needed.

### Format diagnostics

For each split, report:

- generations that emit newline or EOS before the row limit;
- generated continuations that parse successfully under a fork-timeboxed
  SymPy parser;
- symbolically equivalent continuations.

These distinguish a format-fidelity failure from a capability null. No string
match is used for correctness.

### Teacher-forced loss

AO1's masked CE proxy is reported only within its new lineage. It is not
numerically compared with any prior full-token cell, including REG. Every loss
statement is explicitly teacher-forced.

## Decision rule

- **Capability win:** AO1 reaches at least 3/8 TRAIN symbolic solves.
- **Graded partial:** AO1 remains at 2/8 solves but exceeds REG's suffix-only
  TRAIN token accuracy.
- **Null:** AO1 reaches at most 2/8 solves and does not exceed REG's suffix-only
  TRAIN token accuracy, unless the format-failure branch applies.
- **Format failure:** a suffix-only token-accuracy drop accompanied by worse
  termination or parseability than REG is booked specifically as format
  failure, not folded silently into a capability null.
- **Mixed regression:** AO1 falls below 2/8 solves while suffix-only accuracy
  rises. This is booked as a negative mixed result, not promoted as a partial
  win and not forced into the null branch.

All outcomes are fenced as one seed, one initialization, and one train/heldout
row set. A 3/8 win is an intentionally weak discovery bar: it queues a paired
seed confirmation before any doctrine movement. It is not treated as a general
capability law.

## Verification

Focused tests must prove:

1. marker IDs and split positions are exact and deterministic;
2. the first answer target and exactly one terminator are supervised;
3. scaffold targets and repeated EOS padding have zero loss gradient;
4. `ANSWER_ONLY=0` produces the legacy loss-gradient tensor exactly;
5. the REG arm reproduces the full G-RB1 trajectory SHA;
6. token-accuracy denominators match their printed spans;
7. parseability and equivalence checks use a fork deadline plus forced kill;
8. train and heldout prompt sets are explicitly disjoint.

Any change touching the pinned deterministic battery is regression-gated
before AO1 is interpreted. The pre-registration in
`docs/sol/RESULTS-SOL.md` must name the final commands, logs, predictions,
decision rule, and relevant Git SHAs before REG or AO1 fires.

## Follow-up boundary

If AO1 wins, the next experiment may test an integer scaffold-loss dose curve.
That curve is not part of this design. If AO1 is null or fails on format, a new
mechanism is required before rerunning it.

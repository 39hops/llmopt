---
name: farm
description: Use when writing or re-running a data farm — any script that GENERATES training rows (shards, atoms, diets, probe sets). Carries the data contract, the split guard, the timebox law, and the selection-function emissions that three contamination incidents and one censoring discovery paid for.
---

# Farming a corpus (the generator ritual)

`/desk` prices a rung by counting. `/rung` runs it. This covers the
step between them — writing the generator — which is where every
contamination incident in this lab happened.

Four incidents are the syllabus. Three contaminations: mathgen L1/L2
at 43% eval-in-train; the ladder's `pick()` with only four possible
bodies; `farm_arith`'s first run exhausting 242/242 mul pairs, so
every probe item would have been a training row. One censoring:
`CENSOR-0` found an 8s wall had silently deleted 15.2% of solvable
L4 outputs from a shard that had already trained a booked rung.

## 1. State the data contract BEFORE writing the generator

Write these three into the pre-reg (or the script docstring if the
farm is unregistered). Not after — the point is that stating them
catches the defect.

- **Row GRAIN** — what exactly one row is, in one sentence. Every
  family in the shard shares it, or the dose number is meaningless.
- **LABEL TIMING** — is the target computable from the input alone
  at emit time, or does it join information the model will not
  have? A row whose label needs future or sibling context trains
  confident guessing (determinability law, RESULTS L3401). The
  stock diet already violates this in places:
  `Integral(0, x) -> "+ 4"` materializes a constant from chain
  context (RULE-POLICY-0-CENSUS).
- **SPLIT POLICY** — what property separates train from eval. A
  seed band is NOT a split. Small generator spaces collide, and
  "random split leaks future data" is the generic form of all three
  booked contaminations.

## 2. Guard the split by exclusion, then COUNT it

`exclude=` the actual eval/probe prompts by normalized key, never
seed offsets alone. Widen the operand ranges past the probe's own
space so the farm is not drawing from an exhausted pool.

Then compute and PRINT `probe_cur INTERSECT shard_cur` under the
same normalization, assert it is zero, and book the number. An
assert with no surviving log leaves the commit message as the only
record — acceptable, but say so in the verdict.

## 3. Seeds are STRINGS

`random.Random(f"kind-{level}-{seed}")`. Tuple `__hash__` is
per-process randomized and killed reproducibility once. Use a fresh
seed band and record which bands are already spent.

## 4. Fork is the only timebox

NO sympy call is safely boxed by SIGALRM — fork, join with a
deadline, SIGKILL (the `gen_magic_labels.solve_isolated` pattern).
Generalized from pathology #7 after the alarm-boxed oracle
live-locked anyway (pathology #10). Applies to generation, rules,
routing probes, verifiers, and any oracle-on-model-text.

## 5. Stream rows, or the killed class is invisible

Write and `flush` each row as it lands. A worker killed by an outer
wall that accumulated in memory takes its whole class with it, and
whatever trains on the data never sees the hole — the checkpoint
selection-effect, bitten three times.

## 6. Emit the selection function (the newest clause)

An observed corpus is never the generator. It is:

```
generator o representation-transform o survival/censor o scorer
```

Every farm reports, at the end, four distributions — because a layer
you do not measure is a layer that can silently reshape the corpus:

- **attempted -> emitted survival**, by rule and by level
- **wall-time distribution** of successes, with the farm's own
  timeout marked inside it (this is the one CENSOR-0 needed)
- **raw-vs-canonical answer form** — the model trains on RAW text,
  so a serialization dialect is a real corpus difference even at
  math-disagreement zero (ANSWER-FORM-0: 75.5% formdiff)
- **output length**, since length interacts with any `SEQ_CAP` drop

If a wall is chosen for speed, state what you believe it censors and
whether that belief has been measured. "Bimodal, so the wall is
free" was the belief CENSOR-0 refuted.

## 7. Refuse to overwrite; keep paths off frozen ground

`assert not OUT.exists()`, unconditional. Check `docs/CODEMAP.md`
for the class of every path written. Never write into a shard a
booked verdict cites.

## 8. Spawn `farm-auditor` before the shard trains anything

Pre-training is the only moment this audit is cheap. It recomputes
the intersection, prices the selection layers, and reports the
classes that died. Its findings are proposals — verify each against
the source before adopting.

## Then, and only then, hand off to `/rung`

The shard is an artifact with provenance now. `/rung` picks it up at
the pre-reg step.

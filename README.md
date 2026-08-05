# llmopt

Training that uses the weight budget as a capacity-achieving code can make
calibration machinery unnecessary; this lab measures when that statement holds.

Start with the [glossary](GLOSSARY.md), [curated findings](docs/FINDINGS.md)
— organized by evidence maturity and scope rather than chronology, one tag per
claim — the [reproduction walkthrough](docs/REPRODUCE.md), and the living
[verdict ledger](docs/RESULTS.md). The [measured-history appendix](docs/MEASURED-HISTORY.md)
keeps supported earlier mathematics and physics results without making this
page chronological.

Citation policy: name the exact repository commit SHA and the exact verdict
entry in [`docs/RESULTS.md`](docs/RESULTS.md) that supports the claim. The
ledger is living, so an unpinned citation is not reproducible. Repository
metadata is in [`CITATION.cff`](CITATION.cff).

## What was built

`llmopt` is a mathematics and physics research lab organized around
executable instruments, declared comparisons, and oracle-checked readouts.
The main paths are:

- [`llmopt/search/`](llmopt/search/) — symbolic derivation search with
  explicit rewrite rules, structural and learned evaluators, proposal
  policies, transposition memory, and verification at the boundary.
- [`llmopt/mathgen/`](llmopt/mathgen/) — seeded generators for calculus,
  linear algebra, ordinary differential equations, mechanics, and proofs,
  with symbolic checks built into generation or evaluation.
- [`llmopt/quantum/`](llmopt/quantum/) — model-Hamiltonian ground-state
  instruments and a verified ZX-graph search path for circuit reduction.
- [`llmopt/train/`](llmopt/train/) and [`llmopt/intmath.py`](llmopt/intmath.py)
  — closed-system model births, exact integer primitives, controlled diets,
  and training interventions that can be compared trajectory by trajectory.
- [`llmopt/quantize/`](llmopt/quantize/) — weight diagnostics, sensitivity
  probes, closed-form allocation, packed crystal artifacts, and the capacity
  meter that selects which allocation regime deserves testing.
- [`llmopt/eval/`](llmopt/eval/) — equivalence, calibration, latency, and
  statistical instruments used as supporting readouts rather than substitutes
  for capability gates.

Experiments are recorded as named arms and cells. A gate accepts or rejects a
declared result; a pin fixes an artifact or trajectory contract. Those words,
the maturity labels, and the controlled scope vocabulary are defined in
[`GLOSSARY.md`](GLOSSARY.md).

Install the editable package and development dependencies with:

```bash
python -m pip install -e ".[dev]"
```

The status board, theory map, riff ledger, handoffs, and machine-readable
index remain living authority surfaces: [`docs/BOARD.md`](docs/BOARD.md),
[`docs/THEORY.md`](docs/THEORY.md),
[`docs/RIFF-LEDGER.md`](docs/RIFF-LEDGER.md),
[`docs/handoffs/`](docs/handoffs/), and
[`docs/results-index.jsonl`](docs/results-index.jsonl).

## What was discovered

- `[SINGLE-SEED] [REGIME-SCOPED: at-capacity house crystals]` The house
  crystals tested at capacity could be packed from weight sigma alone, with
  no calibration data or allocator search, while remaining inside their
  declared capability gate. This is a Mac MPS, `n=1`-per-crystal result, not a
  universal quantization rule: **PACKED CRYSTAL C0+C1 VERDICT** in
  [`docs/RESULTS.md`](docs/RESULTS.md).

- `[SINGLE-SEED] [DEVICE-SCOPED] [FORMAT-BOUND]
  [REGIME-SCOPED: Qwen2.5-0.5B]` **THE SIGMA-PACK CLAIM DOES NOT TRANSPORT TO
  Qwen2.5-0.5B.** On the registered RTX 3080 fake-quant arm, max-anchored and
  calibrated grids exploited weight-tail structure that the house crystals
  did not have. The boundary is one model, one device, and `n=1`; it is not a
  law of web-trained dense models: **PACKED CRYSTAL C6 VERDICT** and **PACKED
  CRYSTAL C6c VERDICT** in [`docs/RESULTS.md`](docs/RESULTS.md).

- `[REPLICATED] [DEVICE-SCOPED] [FORMAT-BOUND]
  [REGIME-SCOPED: measured deployment artifacts]` The packed integer GEMM
  path reproduced its digest across the registered Mac MPS and RTX 3080 CUDA
  devices. This independent-device route covers one house crystal's integer
  GEMM path, not a full integer end-to-end decoder: **PACKED CRYSTAL C4
  VERDICT** and **R-PASS VERDICT** in [`docs/RESULTS.md`](docs/RESULTS.md).

- `[NULL] [DEVICE-SCOPED] [FREE-RUN-GATED]
  [REGIME-SCOPED: specified diet and recipe]` Removing scaffold-token loss
  did not repair the registered gravmoe capability gate; the arm degraded
  while its format diagnostics remained intact. This is a Mac-local `n=1`
  null: **VERDICT SOL-ADOPTION-1** in
  [`docs/RESULTS.md`](docs/RESULTS.md).

- `[REPLICATED] [FORMAT-BOUND] [FREE-RUN-GATED]
  [REGIME-SCOPED: measured deployment artifacts]` **A resident 30B-class
  MoE scored HIGHER on the registered mathematics gate with 45.3% of its
  experts masked out than at full width**, +14.7 pooled across six paired
  seeds. The effect is selection, not sparsity: random and anti-demand
  masks at the identical fraction scored `0/120`. It is also
  domain-specific — the same recipe on mechanics returned `+3` and then
  `-59` pooled, with open and closed recall matched to the mathematics
  arm to within `0.0001` and `0.003`, so coverage and recall do not
  predict the sign of the effect. The scope is one vehicle, one keep
  rule, one gate, and Mac MLX: **VERDICT MOE-GT-1-R5**, **VERDICT
  MOE-GT-1-R6**, and **VERDICT MOE-GT-2-D4-PHYS-B** in
  [`docs/RESULTS.md`](docs/RESULTS.md).

The broader closed-system record, including positive results, nulls,
retractions, and amendments, is curated by evidence maturity in
[`docs/FINDINGS.md`](docs/FINDINGS.md). Historical mathematics and physics
measurements that are still supported but are not part of that main arc are in
[`docs/MEASURED-HISTORY.md`](docs/MEASURED-HISTORY.md).

## What can be reproduced today

```bash
RJOB_LOCAL=1 python -m llmopt.reproduce gravmoe-rb1
```

On the current Mac-local contract, `PASS` means the final training-trajectory
digest exactly matches the committed `gravmoe-rb1` pin. The adopted command is
booked by **VERDICT SOL-ADOPTION-1** in [`docs/RESULTS.md`](docs/RESULTS.md);
repository commit `4ef9cd511369023d69db7332aebf36517de62951` made the path self-contained from committed windows.

Trajectory agreement is not oracle correctness. It certifies the pinned
weight path and teacher-forced training readouts. Free-run symbolic solve
scoring additionally needs the uncommitted diet row text and its oracle; the
artifact-backed gate arms therefore run in an explicit trajectory-only mode.

The walkthrough in [`docs/REPRODUCE.md`](docs/REPRODUCE.md) explains setup,
expected output, the full pin registry, and where oracle-backed correctness
claims live. Use `python -m llmopt.reproduce --list` to inspect the registry;
do not interpret digest equality as a capability score.

## What remains uncertain

The calibration-free packing law is established only for the measured
at-capacity house crystals. The negative **PACKED CRYSTAL C6 VERDICT** is
equally narrow: Qwen2.5-0.5B, one registered RTX 3080 device, fake quantization,
and `n=1`. More models, independent implementations, and capability-gated
deployment artifacts are needed before either boundary can move.

Many training comparisons remain single-seed and device-scoped. The README
keeps those fences visible; [`docs/FINDINGS.md`](docs/FINDINGS.md) carries the
current maturity labels, and [`docs/BOARD.md`](docs/BOARD.md) separates live
work from closed results.

Why masking a deployed MoE to its demand coalition beats full width on
mathematics is unexplained, and the two quantities a keep rule optimizes
— coverage and recall of demanded experts — were measured not to predict
even the sign of that effect. One piece is now
measured and it removes a class of explanation rather than supplying
one: above the shared symbolic core, restoring capability tracks how
much demand mass the kept population covers, not which class the added
experts belong to — a matched-size random fill resurrected a dead core
about as well as the verbal-branch fill (51/36/48 versus 55 of 120,
one seed, three draws). Why the crest beats full width remains
unexplained, and the crest stays a booked observation on one vehicle
and one domain, not a deployment recommendation.

Trajectory reproduction still stops short of self-contained free-run oracle
scoring because the row text is not committed. Until that input can be shared
lawfully, the public artifact proves the trajectory and teacher-forced
readouts only.

The mathematics and physics charter is intentionally narrow. Candidate work
belongs in [`docs/RIFF-LEDGER.md`](docs/RIFF-LEDGER.md); uncertainty becomes a
claim only after a declared arm, a named verdict, and the applicable regime,
device, format, and replication fences are booked.

Licensed under [Apache-2.0](LICENSE).

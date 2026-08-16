# Reproduce the deterministic gravmoe trajectory

This is the external start-here path for the repository's mathematics and
physics work. It replays one pinned deterministic-birth training trajectory
from committed inputs and checks its digest. It is not a general model-quality
benchmark.

## Quick start

From the repository root, use a supported Python environment and install the
project with its development dependencies:

```bash
pip install -e ".[dev]"
```

Then run the adopted RB1 reproduction:

```bash
RJOB_LOCAL=1 python -m llmopt.reproduce gravmoe-rb1
```

Plan about 80 seconds for an arm. **VERDICT SOL-ADOPTION-1**, at repository
commit `0dea97283d4a270c4f8b2b1ad48adcf01b42e5f8`, records the exact house RB1
run as 81.7 seconds.

The runner streams training diagnostics. Its success line has this exact
shape:

```text
PASS gravmoe-rb1 c6766da235cf0b76be20035b893cb41fd0a2f8dbbc6339c96e8527ce2cb3f65f
```

`PASS` means that the full milestone-weight trajectory digest matched the
committed pin exactly. It is trajectory-exact determinism evidence. It is not
symbolic correctness evidence, and it is not a capability score.

The determinism this reproduction certifies belongs to the pinned integer and
CPU float paths it replays. It does not extend to ordinary float training on
Apple-silicon MPS, which is run-level nondeterministic at fixed seed: a paired
20-step probe on 2026-08-15 produced different weight digests across two
consecutive runs of the same script, same seed, same batches
([AMENDMENT SOFT-SPEED-1-PRECONDITION](RESULTS.md#L29985 "id:2026-08-15-amendment-soft-speed-1-precondition-target")). Experiments on
that substrate compare arms within one run and never assert cross-run weight
identity.

## What a fresh clone contains

Repository commit `4ef9cd511369023d69db7332aebf36517de62951` made the
reproduction artifact-backed. The trajectory and teacher-forced loss reproduce
from committed window bytes; the runner verifies their contract and SHA before
using them. The relevant implementation is
[`llmopt/reproduce.py`](../llmopt/reproduce.py), and the artifacts and registry
live in [`scratch/detbwd_gmoe_ref/`](../scratch/detbwd_gmoe_ref/).

Gate arms have one additional boundary. SymPy solve scoring needs the original
row text, which is not committed. Consequently, artifact-backed gate arms run
in an explicit trajectory-only mode: their trajectory and teacher-forced
diagnostics remain reproducible, but their free-run symbolic correctness does
not. A gate-arm trajectory `PASS` must never be described as a free-run
correctness result.

## The complete pin registry

The command above selects RB1. **VERDICT SOL-ADOPTION-1**, at repository commit
`0dea97283d4a270c4f8b2b1ad48adcf01b42e5f8`, books `--list` coverage of all
16 pins, including the other 15:

```bash
python -m llmopt.reproduce --list
```

The source of truth is
[`scratch/detbwd_gmoe_ref/pins.json`](../scratch/detbwd_gmoe_ref/pins.json).
The 16-arm family map below is the battery covered by **VERDICT
GRAVMOE-P4-DEVICE** at repository commit
`b9372e967ab7269afd06fa52027ce19450ca4d95`:

| Public reproduction names | Family |
|---|---|
| `gravmoe-a0`, `gravmoe-a1`, `gravmoe-a2`, `gravmoe-a3` | Saturated-initialization, truncated-window gravity-dose sweep. |
| `gravmoe-ca0`, `gravmoe-ca1`, `gravmoe-ca2`, `gravmoe-ca3` | Conditioned residual-writer initialization, with the same gravity-dose sweep. |
| `gravmoe-ga0`, `gravmoe-ga2`, `gravmoe-ga3` | Complete-row gate-diet counterparts at the registered gravity doses; public replay is trajectory-only. |
| `gravmoe-rb1`, `gravmoe-rb3`, `gravmoe-rb1s16` | Corrected wq/wk-only initialization family: base, learned-temperature, and wider-shift arms. |
| `gravmoe-grb1` | Complete-row gate arm on the conditioned wq/wk initialization; public replay is trajectory-only. |
| `gravmoe-s1` | Complete-row scheduled-sampling gate arm; public replay is trajectory-only. |

Names and digests printed by `--list` come directly from the committed JSON;
do not copy a digest from prose when the registry can answer it.

## What the external close establishes

**VERDICT GRAVMOE-P4-DEVICE**, at repository commit
`b9372e967ab7269afd06fa52027ce19450ca4d95`, books the device leg at 16/16
pinned arms SHA-identical. **VERDICT GRAVMOE-P4-LAB**, at repository commit
`94e29cd61ef3b0cfbe44f5848185053bcb9bdb87`, closes the ladder at 3
implementations / 2 labs / 2 devices.

In **VERDICT GRAVMOE-P4-LAB**, 2 labs means independent Python and C++ code
paths under one human operator, not independent investigators. 2 devices means
two machines with different CPU architectures — Apple silicon arm64 and
x86-64 — and both legs execute on CPU: the battery carries no device
placement, so this is narrower than the MPS-to-CUDA GPU transport established
by [P3 VERDICT](RESULTS.md#L11357 "id:2026-07-30-p3-verdict-the-deterministic-decode-lands") and [PACKED CRYSTAL C4
VERDICT](RESULTS.md#L10657 "id:2026-07-29-packed-crystal-c4-verdict-claim-3"). Integer execution does not require CPU; see
[AMENDMENT P4-DEVICE-SCOPE](RESULTS.md#L15160 "id:2026-08-02-amendment-p4-device-scope-amends-verdict"). The close
establishes exact transport of the registered trajectory; it does not enlarge
the correctness or capability claim.

The cross-lab implementation is the public
[`39hops/axiom`](https://github.com/39hops/axiom) repository. The pinned
verifier identity is axiom commit
`8f8376d86ce6a25fdd6fee2455c220e7055cb018`, path
`tools/int_adamw/verify_gravmoe.py`. As of 2026-08-02 both halves of that
reproduction are public: axiom pushed, so the pinned commit, the verifier, and
its `r2b_tables.bin` are fetchable from origin, and the reference artifacts it
consumes are in `scratch/detbwd_gmoe_ref/` here. An outside reader can run the
cross-lab leg rather than only read its receipt. What stays true is the
authorship caveat above — two sessions, one operator — not a reachability
limit ([AMENDMENT AXIOM-PUBLICATION](RESULTS.md#L15283 "id:2026-08-02-amendment-axiom-publication-amends-verdict-gravmoe")).

Its interface from `tools/int_adamw` is:

```bash
python verify_gravmoe.py <build_dir> <ref_dir> [arm ...]
```

**VERDICT GRAVMOE-P4-LAB**, at repository commit
`94e29cd61ef3b0cfbe44f5848185053bcb9bdb87`, books that verifier's 10/10
engine-arm result. No build command is reproduced here because the ledger books
the verifier and interface, not a portable external build recipe.

## Reproducing the MoE ground-truth program (MOE-GT-1 / MOE-GT-2)

This is a second, independent reproduction path with different
properties from the pinned trajectories above. It is not
artifact-backed and it is not digest-checked. It regenerates its own
inputs by running a 30B mixture-of-experts model, so it reproduces
*measurements*, not bytes, and every number below carries a sampling
fence.

### What it needs

An Apple-silicon Mac with at least 36 GB of unified memory, `mlx-lm`
in the project environment, and one Hugging Face download of
`mlx-community/Qwen3-30B-A3B-4bit` (the model id is pinned in
[`scratch/moe_gt1.py`](../scratch/moe_gt1.py)). A 120-item gate takes
roughly 3-7 minutes on an M3-class machine. The code arm additionally
needs a Homebrew LLVM toolchain: check
`llmopt.codegen.llvm.llvm_available()` before spending model time,
because Xcode's clang has no `llvm-mc` and the corpus fails silently
empty.

Two operating rules travel with these runs. **One resident 30B at a
time** — a smoke test loaded beside a live job on 2026-08-04 and the
swap storm panicked the machine. And the gate solve counts below are
Apple-silicon numbers: they do not transport to another device, and
cross-device comparison of them is forbidden by house doctrine.

### The artifacts are regenerated, not downloaded

`checkpoints/` and `logs/` are untracked by design (file-handoff
convention). Nothing in this section ships its inputs: the demand log
`checkpoints/moe_gt1_arm0.json`, the trajectory logs under
`logs/opus/`, and the per-problem logs are all produced by the first
command you run. A reproduction therefore re-derives the keep-sets
rather than replaying the original run's.

### Crest confirmation (VERDICT MOE-GT-1-R5)

First produce the arm-0 demand log — the full-residency oracle run
that every later mask reads:

```bash
.venv/bin/python scratch/moe_gt1.py
```

Then replay at the crest fraction against a paired full baseline, at
the three registered seeds:

```bash
for s in 111 222 333; do
  SEED=$s FRACS=1.0,0.453 PERPROB=1 .venv/bin/python scratch/moe_gt1_arm2.py
done
```

**VERDICT MOE-GT-1-R5** books, seed by seed, full 63 -> 80 (+17),
73 -> 82 (+9), 63 -> 81 (+18); pooled +14.7 against a registered +7
bar, 3/3 positive. Keep the fence attached: the 120-item gate has
sigma about 5 solves, so the +9 seed is by itself inside the noise
band — the claim rests on the sign holding in 3/3 seeds and on the
pooled delta, not on any single arm. The keep-set is derived from a
seed-1234 demand log and transfers to seeds it was never fitted to;
that transfer is the finding.

This is a gate claim only. Free-run probe text degenerates at some
fractions where the gate improves (ARM2 P4), so a solve delta must
never be read as a general-quality improvement.

### The sparsity control (VERDICT MOE-GT-1-R6)

The same driver carries the falsifier, selected by `RULE`:

```bash
RULE=random RULESEED=0 FRACS=0.453 .venv/bin/python scratch/moe_gt1_arm2.py
RULE=random RULESEED=1 FRACS=0.453 .venv/bin/python scratch/moe_gt1_arm2.py
RULE=anti              FRACS=0.453 .venv/bin/python scratch/moe_gt1_arm2.py
```

All three score 0/120 at seed 1234. At the identical residency
fraction, *which* experts remain is the difference between zero and
the low eighties. Generic sparsity contributes nothing measurable
here.

### The domain program (VERDICT MOE-GT-2-D2 / D3)

Each domain arm is a demand run with the trajectory instrument on,
followed by one offline analysis pass. Physics reuses the math driver
through its corpus knobs (this is the exact house invocation, from the
job receipt); levels are fenced to 2-3 because the L1 cell is
exhaustible and `make_dataset`'s dedup loop does not terminate on an
exhausted cell:

```bash
KINDS=eom,small_osc,kinematics LEVELS=2,3 SEED=606 \
  TRAJ=1 TRAJ_OUT=logs/opus/gt2_phys_traj.jsonl \
  OUT=checkpoints/gt2_phys_arm0.json \
  .venv/bin/python scratch/moe_gt1.py
```

Code has its own driver, because its corpus is the toolchain-scored
ladder and its system prompt is deliberately code-specific:

```bash
TRAJ=1 TRAJ_OUT=logs/opus/gt2_code_traj.jsonl \
  .venv/bin/python scratch/gt2_code_arm0.py
```

Then the coalition readout, which consumes all three trajectory files:

```bash
.venv/bin/python scratch/gt2_jaccard.py
```

With the defaults (`DROP_TAIL=1`, `GATE_ONLY=1`) this prints the
corrected coalition distances booked by **AMENDMENT GT2-REVIEW-2**:
Jaccard(math, phys) 0.8013, (math, code) 0.5331, (phys, code) 0.5280,
with split-half nulls 0.9205 / 0.8670 / 0.6364, plus the three-domain
core of **OBSERVATION GT2-CORE-0** (37.1 of 58 experts per layer
against an 11.9 independence null, containment 0.92 (the leave-one-out form — math&code inside physics — not a fourth-corpus measurement)). Setting
`DROP_TAIL=0` reproduces the numbers as originally booked in
**VERDICT MOE-GT-2-D2** and **D3** (0.804 / 0.543 / 0.539, code null
0.653); `GATE_ONLY=0` additionally restores the probe rows behind
D2's headline 0.767. The filters are exposed precisely so the booked
variants are distinguishable rather than looking like drift; all
claims survive the correction. `DUMP_DECODE=1 DROP_TAIL=0` writes the
decode-only demand JSONs byte-identical to the ones the D4 second-crest
and cross arms consumed (verified 2026-08-04), which makes those arms
reproducible end-to-end through `ARM0=` on the mask drivers.

The core observation is desk-only and carries its own fences: one seed
per domain and tie-fill at the keep boundary. The "generic decoding
substrate" alternative it could not exclude has since been excluded by
arms: the core is symbolic (the proofs coalition contains 0.901 of it,
plain prose 0.250), and the shared-system-prompt confound is priced at
0.05-0.10 Jaccard — too small to generate the branch separation
([VERDICT MOE-GT-3](RESULTS.md#L19852 "id:2026-08-05-verdict-moe-gt-3-the-core"); [VERDICT
MOE-GT-4b](RESULTS.md#L20111 "id:2026-08-05-verdict-moe-gt-4b-the-branch")).

### The register split and the recall shoulder (MOE-GT-3 through GT-5c)

The branch program adds three arm-0 demand runs (a proofs corpus and two
verbal corpora) driven by `scratch/gt3_probe_arm0.py` over prompt lists;
the dialog grid regenerates from `scratch/gt4_dialog_prompts.py`, and
`FORCE_SYS=math` on the unchanged dialog corpus is the system-prompt
control. The readout script `scratch/gt4_verbal_core.py` consumes the
trajectories and re-derives every GT-3 number that was originally a desk
calculation, to a uniform ±0.006 of the booked values (containment
proofs 0.8987 vs booked 0.901, prose 0.2453 vs 0.250) — treat the
±0.006 as an unexplained desk-calc discrepancy, not drift ([AMENDMENT
MOE-GT-4-REVIEW](RESULTS.md#L20006 "id:2026-08-05-amendment-moe-gt-4-review-corrected"), item 6).

The mask arms reuse D3's instrument, `scratch/moe_gt1_arm2.py`, through
`KEEPSET=`. Keep-set builders are committed: `scratch/gt5_union_keep.py`
(union-of-bases, 78.2/128 per layer) and `scratch/gt5c_randfill_keep.py`
(matched-size random fills over the frozen core). The seed replication
is three within-seed pairs (1234, 777, 2026), full arm vs masked arm on
the same device. Enable `PERPROB=1` for the per-answer degeneracy
readout — the original run did not, and booked that as a readout miss; GT-6
collected the degeneracy readout via per-arm probe text instead, with
PERPROB off, named in its pre-reg.
Reproduce the direction and the magnitude class, never the digits: the
full baselines themselves move 60-73/120 across seed problem sets.

The GT-6 ladder arms build with `scratch/gt6_recall_ladder.py` (frozen
D3 core + uniform per-layer random fill, count tuned so arm0-axis open
recall lands within ~0.01 of target; the verbal-excluded arms draw from
128 minus core minus verbal-core). Run mask arms ONLY with the boxed
oracle: `check_isolated()` in `scratch/moe_gt1_arm2.py` talks to
`scratch/oracle_worker.py`, a subprocess line-server with a parent-side
RSS watchdog (kill at 3GB, loud `ORACLE-MEMBOMB`), because shoulder
arms mass-produce pathological-but-parseable completions whose sympy
simplify balloons gigabytes and gets the 17GB resident driver
jetsam-killed. Never fork a Metal-resident driver; Darwin
`RLIMIT_AS`/`RLIMIT_DATA` are verified no-ops. Timeouts and crashes are
conservative rejects, always printed and counted ([AMENDMENT
MOE-GT-6-ORACLE-BOX-3](RESULTS.md#L20703 "id:2026-08-05-amendment-moe-gt-6-oracle-box-b-b")).

### The Lean certificate kernel check

`scratch/lean_check.py` re-derives each cert's statement with an
independent printer and kernel-checks the corpus in 50-row chunks
against a mathlib cache (`scratch/leancheck`; a Linux/WSL or Mac `lake`
both work — a Mac-local cache exists since 2026-08-05). The chunking is
load-bearing, not cosmetic: Lean aborts a file at ~100 diagnostics and
in-file `set_option maxErrors` does not lift it, so an unchunked
single-file check silently truncates while looking complete — this bit
the check twice before the fix ([VERDICT
LEAN-KERNEL-SAMPLE](RESULTS.md#L20365 "id:2026-08-05-verdict-lean-kernel-sample-registered-1000")).

The 120-gate sigma of about 5 solves is a *mathgen* number;
the physics and code gates' dispersion is unmeasured, so no delta
claims on those gates.

### What you cannot reproduce without re-running

- **CHURN-JUDGE-1.** [`scratch/churn_judge_eval.py`](../scratch/churn_judge_eval.py)
  reads `logs/opus/moe_gt1_perprob.jsonl`, which is untracked. To
  reproduce it you must first re-run the crest arm with `PERPROB=1` at
  all seven seeds (111, 222, 333, 555, 4242, 777, 90210) plus their
  paired full arms, and even then your log will not be byte-identical
  to the original, which carries two duplicate seed-4242 rows from an
  aborted write (booked, harmless — the analysis keys by problem). The
  booked readouts are a held-out AUC of 0.679 against a 0.60 bar, and
  an escalation spend of 31 against 23.5 expected. Read the second
  number with its amendment: the 1.5x bar is **unresolved**, not
  missed — the shortfall is 4.25 solves, inside the measured noise
  fence, from a single n=50 draw.
- **Exact solve counts, anywhere.** These are greedy decodes of a
  quantized model on one device family. Reproduce the *direction and
  magnitude class* of a delta, not its digits, and treat any
  single-seed difference under about five solves as unresolved.

## If it does not pass

- **SHA mismatch:** a `FAIL ... expected ... got ...` line means the observed
  milestone-weight trajectory differs from the committed pin. Treat it as a
  failed reproduction, not a near-pass; record the repository commit,
  interpreter, platform, and full output before investigating.
- **Malformed or altered artifacts:** raw-byte SHA, record shape, overlap, and
  reconstructed-row checks are refusal boundaries. Do not regenerate or edit
  an artifact to get past them; restore the committed files and retry from a
  clean tree.
- **Missing optional oracle row text:** this is expected for artifact-backed
  gate arms. It removes SymPy free-run solve scoring, not trajectory replay or
  teacher-forced loss. The resulting trajectory-only `PASS` is determinism
  evidence only.

For the claims and their fences, read the exact named entries in the living
[`docs/RESULTS.md`](RESULTS.md): **VERDICT SOL-ADOPTION-1**, **VERDICT
GRAVMOE-P4-DEVICE**, and **VERDICT GRAVMOE-P4-LAB**. Use the full repository
commit SHAs above when citing those entries because the ledger is living.

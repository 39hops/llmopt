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
by [P3 VERDICT](RESULTS.md#L11357) and [PACKED CRYSTAL C4
VERDICT](RESULTS.md#L10657). Integer execution does not require CPU; see
[AMENDMENT P4-DEVICE-SCOPE](RESULTS.md#L15160). The close
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
limit ([AMENDMENT AXIOM-PUBLICATION](RESULTS.md#L15283)).

Its interface from `tools/int_adamw` is:

```bash
python verify_gravmoe.py <build_dir> <ref_dir> [arm ...]
```

**VERDICT GRAVMOE-P4-LAB**, at repository commit
`94e29cd61ef3b0cfbe44f5848185053bcb9bdb87`, books that verifier's 10/10
engine-arm result. No build command is reproduced here because the ledger books
the verifier and interface, not a portable external build recipe.

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

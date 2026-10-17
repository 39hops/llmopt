# Checkpoint decision sheet

Tree: /Users/artin/code/llmopt/checkpoints. Files 1159, logical 70.51 GB, allocated 70.51 GB, tracked 45 files (0.16 GB), symlinks 0, hard-linked 0.

| class | files | logical GB | allocated GB |
|---|---:|---:|---:|
| FROZEN | 547 | 46.55 | 46.55 |
| ESSENTIAL | 334 | 7.82 | 7.82 |
| DUPLICATE | 6 | 0.12 | 0.12 |
| CANDIDATE | 43 | 0.85 | 0.85 |
| UNKNOWN | 229 | 15.16 | 15.16 |

Duplicate groups: 35; redundant logical 2.60 GB; of which distinct blocks 2.60 GB, clone-shared 0.00 GB, physical unknown 0.00 GB.

Reclaim, verified (DUPLICATE class on distinct blocks): 0.12 GB. Reclaim, estimate (CANDIDATE class allocated bytes, pending decision): 0.85 GB. UNKNOWN class allocated: 15.16 GB (no estimate).

SHA mismatches: none. Unresolved references: {'absent-path': 474, 'generic-loader': 26, 'dynamic-prefix': 149}. Generic loaders (code that opens any name under checkpoints/): 26; cited paths absent from this tree: 214 (absent_cited.json).

## Families (by allocated size)

| family | files | alloc GB | newest | FROZEN | ESSENTIAL | DUPLICATE | CANDIDATE | UNKNOWN | dup distinct GB | copies | reached by patterns in | benefit GB (conf) | action |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| root:mathnative | 141 | 20.82 | 2026-08-23 | 34 | 30 | 0 | 0 | 77 | 0.00 | repair:1,3080:6 | UNCITED x1, results-cited x9 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| fmel2 | 48 | 4.24 | 2026-09-15 | 48 | 0 | 0 | 0 | 0 | 0.00 | - | UNCITED x1 | 0.00 (high) | keep (evidence or required) |
| rdc1 | 48 | 4.24 | 2026-09-17 | 48 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| backsched19m | 19 | 4.16 | 2026-08-13 | 19 | 0 | 0 | 0 | 0 | 0.00 | - | library x4 | 0.00 (high) | keep (evidence or required) |
| phase19m | 19 | 4.16 | 2026-08-13 | 19 | 0 | 0 | 0 | 0 | 0.00 | - | library x4 | 0.00 (high) | keep (evidence or required) |
| root:tourn | 12 | 3.97 | 2026-08-10 | 12 | 0 | 0 | 0 | 0 | 0.00 | - | UNCITED x2, library x1, results-cited x3 | 0.00 (high) | keep (evidence or required) |
| root:svp | 48 | 3.68 | 2026-09-03 | 48 | 0 | 0 | 0 | 0 | 0.00 | - | library x1, results-cited x2 | 0.00 (high) | keep (evidence or required) |
| root:metabolic | 12 | 2.65 | 2026-08-07 | 2 | 1 | 0 | 0 | 9 | 0.00 | - | UNCITED x1, results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:gallery19m | 28 | 2.04 | 2026-08-16 | 16 | 5 | 0 | 0 | 7 | 0.00 | 3080:3 | UNCITED x1, library x2, results-cited x10 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:metab | 5 | 1.79 | 2026-08-07 | 4 | 0 | 0 | 0 | 1 | 0.00 | - | results-cited x2 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| sgwriter1 | 37 | 1.64 | 2026-09-10 | 37 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:crown | 6 | 1.27 | 2026-08-10 | 6 | 0 | 0 | 0 | 0 | 0.00 | - | UNCITED x1, results-cited x1 | 0.00 (high) | keep (evidence or required) |
| frozenbb1 | 15 | 1.13 | 2026-09-10 | 15 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| writercaf1 | 16 | 0.99 | 2026-09-09 | 16 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| sgbb7 | 18 | 0.92 | 2026-09-11 | 18 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| writerdfa1 | 16 | 0.91 | 2026-09-09 | 16 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| fme1 | 10 | 0.91 | 2026-09-15 | 10 | 0 | 0 | 0 | 0 | 0.00 | - | library x1 | 0.00 (high) | keep (evidence or required) |
| root:rev2 | 3 | 0.91 | 2026-08-09 | 3 | 0 | 0 | 0 | 0 | 0.00 | - | UNCITED x1, results-cited x1 | 0.00 (high) | keep (evidence or required) |
| root:merged | 2 | 0.72 | 2026-07-23 | 2 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| _attempt3_QUALIFIED_prefold_atomtraj1_smoke | 9 | 0.68 | 2026-09-07 | 3 | 3 | 1 | 2 | 0 | 0.08 | - | - | 0.23 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| atomtraj1_smoke | 9 | 0.68 | 2026-09-07 | 9 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:fmt | 15 | 0.61 | 2026-07-28 | 4 | 3 | 0 | 0 | 8 | 0.00 | - | UNCITED x1, results-cited x3 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:ternary | 3 | 0.60 | 2026-07-22 | 1 | 2 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| oma1 | 5 | 0.53 | 2026-09-14 | 5 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:snap19m | 6 | 0.45 | 2026-07-27 | 0 | 2 | 0 | 0 | 4 | 0.00 | - | results-cited x3 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:sat | 2 | 0.44 | 2026-08-11 | 2 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:cplx | 5 | 0.38 | 2026-07-28 | 3 | 2 | 0 | 0 | 0 | 0.00 | - | results-cited x1 | 0.00 (high) | keep (evidence or required) |
| root:exchange | 1 | 0.36 | 2026-08-07 | 1 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:sym | 52 | 0.35 | 2026-08-08 | 8 | 1 | 0 | 0 | 43 | 0.00 | - | library x1, results-cited x4 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:grid | 2 | 0.34 | 2026-07-22 | 0 | 2 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:step | 9 | 0.32 | 2026-08-07 | 5 | 3 | 1 | 0 | 0 | 0.04 | - | - | 0.04 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:p3 | 10 | 0.31 | 2026-08-08 | 10 | 0 | 0 | 0 | 0 | 0.00 | - | UNCITED x1, results-cited x2 | 0.00 (high) | keep (evidence or required) |
| root:seedvar | 3 | 0.23 | 2026-07-21 | 0 | 0 | 0 | 0 | 3 | 0.00 | - | UNCITED x1, results-cited x2 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:fp64 | 2 | 0.23 | 2026-07-23 | 0 | 1 | 0 | 1 | 0 | 0.00 | - | - | 0.08 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:growdecomp | 1 | 0.22 | 2026-08-11 | 1 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:grpo | 1 | 0.22 | 2026-07-21 | 0 | 1 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:union | 1 | 0.20 | 2026-08-07 | 1 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:floorhk | 4 | 0.18 | 2026-08-11 | 4 | 0 | 0 | 0 | 0 | 0.00 | - | results-cited x2 | 0.00 (high) | keep (evidence or required) |
| root:smoke | 3 | 0.15 | 2026-08-23 | 0 | 1 | 0 | 2 | 0 | 0.00 | - | - | 0.08 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:phys | 2 | 0.15 | 2026-07-23 | 0 | 1 | 0 | 1 | 0 | 0.00 | - | - | 0.08 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:cc | 2 | 0.15 | 2026-07-22 | 0 | 0 | 0 | 0 | 2 | 0.00 | - | UNCITED x1, results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:vrm | 2 | 0.15 | 2026-07-21 | 0 | 0 | 0 | 0 | 2 | 0.00 | - | results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:warmbirth | 2 | 0.15 | 2026-07-18 | 2 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:calib | 3 | 0.10 | 2026-07-27 | 1 | 2 | 0 | 0 | 0 | 0.00 | - | results-cited x2 | 0.00 (high) | keep (evidence or required) |
| root:smoke4 | 1 | 0.08 | 2026-08-23 | 0 | 0 | 0 | 1 | 0 | 0.00 | - | - | 0.08 (estimate) | decide: archive off-volume or delete after manual review; nothing reaches these files |
| root:vmasm2 | 1 | 0.08 | 2026-07-23 | 0 | 0 | 0 | 1 | 0 | 0.00 | - | - | 0.08 (estimate) | decide: archive off-volume or delete after manual review; nothing reaches these files |
| root:vmasm | 1 | 0.08 | 2026-07-22 | 0 | 0 | 0 | 1 | 0 | 0.00 | - | - | 0.08 (estimate) | decide: archive off-volume or delete after manual review; nothing reaches these files |
| root:dual | 1 | 0.08 | 2026-07-22 | 0 | 0 | 0 | 1 | 0 | 0.00 | - | - | 0.08 (estimate) | decide: archive off-volume or delete after manual review; nothing reaches these files |
| root:port | 1 | 0.08 | 2026-07-22 | 0 | 0 | 0 | 1 | 0 | 0.00 | - | - | 0.08 (estimate) | decide: archive off-volume or delete after manual review; nothing reaches these files |
| root:template | 1 | 0.08 | 2026-07-18 | 0 | 0 | 0 | 1 | 0 | 0.00 | - | - | 0.08 (estimate) | decide: archive off-volume or delete after manual review; nothing reaches these files |
| root:umoe | 18 | 0.07 | 2026-07-31 | 6 | 0 | 1 | 0 | 11 | 0.00 | - | library x2, results-cited x3 | 0.00 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:msearch | 48 | 0.07 | 2026-08-11 | 48 | 0 | 0 | 0 | 0 | 0.00 | 3080:48 | results-cited x4 | 0.00 (high) | keep (evidence or required) |
| root:scorer | 2 | 0.07 | 2026-07-27 | 1 | 0 | 0 | 0 | 1 | 0.00 | - | results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:quat | 2 | 0.07 | 2026-07-28 | 1 | 0 | 0 | 0 | 1 | 0.00 | 3080:1 | library x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:rot | 2 | 0.07 | 2026-07-28 | 1 | 0 | 0 | 0 | 1 | 0.00 | - | results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:proposer | 3 | 0.05 | 2026-07-08 | 3 | 0 | 0 | 0 | 0 | 0.00 | repair:3,3080:3 | - | 0.00 (high) | keep (evidence or required) |
| v4flash_sample | 235 | 0.05 | 2026-08-03 | 0 | 235 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:matryoshka | 3 | 0.04 | 2026-07-29 | 3 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:qwen05 | 2 | 0.03 | 2026-08-08 | 0 | 1 | 0 | 0 | 1 | 0.00 | - | results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:muon3 | 1 | 0.03 | 2026-07-28 | 0 | 1 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| confirmed | 11 | 0.03 | 2026-07-31 | 11 | 0 | 0 | 0 | 0 | 0.00 | repair:11,3080:11 | UNCITED x1, library x2 | 0.00 (high) | keep (evidence or required) |
| root:value | 2 | 0.02 | 2026-07-26 | 2 | 0 | 0 | 0 | 0 | 0.00 | repair:2,3080:2 | - | 0.00 (high) | keep (evidence or required) |
| root:diff | 1 | 0.02 | 2026-07-05 | 1 | 0 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | - | 0.00 (high) | keep (evidence or required) |
| root:int | 1 | 0.02 | 2026-07-05 | 1 | 0 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | - | 0.00 (high) | keep (evidence or required) |
| root:calculus | 1 | 0.02 | 2026-07-05 | 1 | 0 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | - | 0.00 (high) | keep (evidence or required) |
| k3_expert_l45_e7 | 7 | 0.02 | 2026-07-30 | 0 | 7 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| k3_expert_l70_e512 | 7 | 0.02 | 2026-07-30 | 0 | 0 | 0 | 7 | 0 | 0.00 | - | - | 0.02 (estimate) | decide: archive off-volume or delete after manual review; nothing reaches these files |
| root:grav2 | 8 | 0.02 | 2026-08-07 | 0 | 0 | 2 | 0 | 6 | 0.00 | - | library x1 | 0.00 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:fourier | 2 | 0.00 | 2026-07-31 | 0 | 0 | 0 | 0 | 2 | 0.00 | - | results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:fourier3 | 2 | 0.00 | 2026-07-31 | 1 | 0 | 0 | 1 | 0 | 0.00 | - | - | 0.00 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:cal | 2 | 0.00 | 2026-07-30 | 0 | 1 | 1 | 0 | 0 | 0.00 | - | results-cited x1 | 0.00 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:fourier2 | 1 | 0.00 | 2026-07-31 | 1 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:fourier4 | 1 | 0.00 | 2026-07-31 | 1 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:microstar | 1 | 0.00 | 2026-08-11 | 1 | 0 | 0 | 0 | 0 | 0.00 | 3080:1 | - | 0.00 (high) | keep (evidence or required) |
| root:exact | 1 | 0.00 | 2026-07-29 | 1 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:packed | 6 | 0.00 | 2026-07-30 | 0 | 0 | 0 | 0 | 6 | 0.00 | - | UNCITED x1, results-cited x2 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:pred | 4 | 0.00 | 2026-07-13 | 2 | 2 | 0 | 0 | 0 | 0.00 | repair:4,3080:4 | - | 0.00 (high) | keep (evidence or required) |
| root:k3 | 1 | 0.00 | 2026-07-30 | 0 | 1 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:gt2 | 6 | 0.00 | 2026-08-04 | 5 | 0 | 0 | 0 | 1 | 0.00 | - | UNCITED x2, library x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:ex2 | 24 | 0.00 | 2026-08-06 | 0 | 0 | 0 | 0 | 24 | 0.00 | - | results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:ex1 | 19 | 0.00 | 2026-08-07 | 3 | 0 | 0 | 12 | 4 | 0.00 | - | results-cited x1 | 0.00 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:gt6 | 12 | 0.00 | 2026-08-05 | 0 | 6 | 0 | 0 | 6 | 0.00 | - | results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:moe | 1 | 0.00 | 2026-08-04 | 1 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:router | 1 | 0.00 | 2026-07-05 | 1 | 0 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | - | 0.00 (high) | keep (evidence or required) |
| root:ex5 | 6 | 0.00 | 2026-08-20 | 0 | 0 | 0 | 0 | 6 | 0.00 | - | results-cited x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:gt7 | 10 | 0.00 | 2026-08-06 | 0 | 0 | 0 | 10 | 0 | 0.00 | - | - | 0.00 (estimate) | decide: archive off-volume or delete after manual review; nothing reaches these files |
| root:magic | 5 | 0.00 | 2026-07-12 | 1 | 4 | 0 | 0 | 0 | 0.00 | repair:5,3080:5 | - | 0.00 (high) | keep (evidence or required) |
| root:dispatcher | 4 | 0.00 | 2026-07-12 | 1 | 3 | 0 | 0 | 0 | 0.00 | repair:4,3080:4 | - | 0.00 (high) | keep (evidence or required) |
| root:ex3 | 6 | 0.00 | 2026-08-07 | 5 | 1 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | library x1 | 0.00 (high) | keep (evidence or required) |
| root:syndrome | 2 | 0.00 | 2026-07-10 | 0 | 2 | 0 | 0 | 0 | 0.00 | repair:2,3080:2 | - | 0.00 (high) | keep (evidence or required) |
| v4flash_vendor | 5 | 0.00 | 2026-08-03 | 2 | 3 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:ex4 | 3 | 0.00 | 2026-08-13 | 3 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:gt5 | 4 | 0.00 | 2026-08-05 | 1 | 0 | 0 | 0 | 3 | 0.00 | - | reproduce-pinned x1 | 0.00 (none) | hold: references of an unresolved form (dynamic path, bare basename, recent) |
| root:rule | 1 | 0.00 | 2026-07-10 | 0 | 1 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | - | 0.00 (high) | keep (evidence or required) |
| root:gt3 | 4 | 0.00 | 2026-08-05 | 1 | 2 | 0 | 1 | 0 | 0.00 | - | - | 0.00 (mixed) | mixed: keep FROZEN/ESSENTIAL rows; DUPLICATE rows may be removed after per-row review; CANDIDATE rows need a decision; UNKNOWN rows hold |
| root:nnue | 1 | 0.00 | 2026-07-06 | 1 | 0 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | - | 0.00 (high) | keep (evidence or required) |
| root:gt4 | 1 | 0.00 | 2026-08-05 | 1 | 0 | 0 | 0 | 0 | 0.00 | - | - | 0.00 (high) | keep (evidence or required) |
| root:engine | 1 | 0.00 | 2026-07-12 | 0 | 1 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | - | 0.00 (high) | keep (evidence or required) |
| root:zx | 2 | 0.00 | 2026-07-09 | 0 | 2 | 0 | 0 | 0 | 0.00 | repair:2,3080:2 | - | 0.00 (high) | keep (evidence or required) |
| root:MANIFEST | 1 | 0.00 | 2026-07-31 | 1 | 0 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | library x1 | 0.00 (high) | keep (evidence or required) |
| root:markov | 1 | 0.00 | 2026-07-10 | 1 | 0 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | - | 0.00 (high) | keep (evidence or required) |
| root:weight | 1 | 0.00 | 2026-07-05 | 0 | 1 | 0 | 0 | 0 | 0.00 | repair:1,3080:1 | - | 0.00 (high) | keep (evidence or required) |

## Largest duplicate groups

| redundant GB | physical | canonical | copies | classes |
|---:|---|---|---|---|
| 0.23 | distinct | checkpoints/fmel2/A/C/h8220.pt | checkpoints/rdc1/A/C/h8220.pt | FROZEN |
| 0.23 | distinct | checkpoints/atomtraj1_smoke/stock_s11/step_00000.pt | checkpoints/_attempt3_QUALIFIED_prefold_atomtraj1_smoke/stock_s11/step_00000.pt; checkpoints/_attempt3_QUALIFIED_prefold_atomtraj1_smoke/stock_s11_probe_emit1/step_00000.pt; checkpoints/atomtraj1_smoke/stock_s11_probe_emit1/step_00000.pt | FROZEN |
| 0.23 | distinct | checkpoints/writerdfa1/qual_dfa_s21_S0.25_lr0.0003/step_00000.pt | checkpoints/writerdfa1/qual_dfa_s21_S1_lr0.0001/step_00000.pt; checkpoints/writerdfa1/qual_dfa_s21_S1_lr0.0003/step_00000.pt; checkpoints/writerdfa1/qual_dfa_s21_S4_lr0.0001/step_00000.pt | FROZEN |
| 0.23 | distinct | checkpoints/atomtraj1_smoke/stock_s11_probe_emit0/final.pt | checkpoints/_attempt3_QUALIFIED_prefold_atomtraj1_smoke/stock_s11_probe_emit0/final.pt; checkpoints/_attempt3_QUALIFIED_prefold_atomtraj1_smoke/stock_s11_probe_emit1/final.pt; checkpoints/atomtraj1_smoke/stock_s11_probe_emit1/final.pt | ESSENTIAL,FROZEN |
| 0.15 | distinct | checkpoints/fmel2/A/C/h0001.pt | checkpoints/oma1/A/C/h0001.pt; checkpoints/rdc1/A/C/h0001.pt | FROZEN |
| 0.15 | distinct | checkpoints/fmel2/A/C/h0005.pt | checkpoints/oma1/A/C/h0005.pt; checkpoints/rdc1/A/C/h0005.pt | FROZEN |
| 0.15 | distinct | checkpoints/fmel2/A/C/h0020.pt | checkpoints/oma1/A/C/h0020.pt; checkpoints/rdc1/A/C/h0020.pt | FROZEN |
| 0.15 | distinct | checkpoints/fmel2/A/C/h0100.pt | checkpoints/oma1/A/C/h0100.pt; checkpoints/rdc1/A/C/h0100.pt | FROZEN |
| 0.08 | distinct | checkpoints/svp_forder_init_s21001.pt | checkpoints/svp_forder_init_s21001_smoke.pt | FROZEN |
| 0.08 | distinct | checkpoints/svp_forder_init_s22001.pt | checkpoints/svp_forder_init_s22001_smoke.pt | FROZEN |
| 0.08 | distinct | checkpoints/svp_forder_init_s23001.pt | checkpoints/svp_forder_init_s23001_smoke.pt | FROZEN |
| 0.08 | distinct | checkpoints/svp_forder_init_s24001.pt | checkpoints/svp_forder_init_s24001_smoke.pt | FROZEN |
| 0.08 | distinct | checkpoints/atomtraj1_smoke/stock_s11_probe_emit1/step_00300.pt | checkpoints/_attempt3_QUALIFIED_prefold_atomtraj1_smoke/stock_s11_probe_emit1/step_00300.pt | DUPLICATE,FROZEN |
| 0.08 | distinct | checkpoints/sgbb7/sgb7_control_s28_lr0.0003/step_00000.pt | checkpoints/sgbb7/sgb7_sg7_local_plr0.001_s28_lr0.0003/step_00000.pt | FROZEN |
| 0.08 | distinct | checkpoints/fmel2/A/C/h0300.pt | checkpoints/rdc1/A/C/h0300.pt | FROZEN |
| 0.08 | distinct | checkpoints/fmel2/A/C/h0900.pt | checkpoints/rdc1/A/C/h0900.pt | FROZEN |
| 0.08 | distinct | checkpoints/fmel2/A/C/h1800.pt | checkpoints/rdc1/A/C/h1800.pt | FROZEN |
| 0.08 | distinct | checkpoints/fmel2/A/C/h3080.pt | checkpoints/rdc1/A/C/h3080.pt | FROZEN |
| 0.08 | distinct | checkpoints/fmel2/A/C/h4500.pt | checkpoints/rdc1/A/C/h4500.pt | FROZEN |
| 0.08 | distinct | checkpoints/fmel2/A/C/h6000.pt | checkpoints/rdc1/A/C/h6000.pt | FROZEN |
| 0.08 | distinct | checkpoints/fmel2/A/C/h7200.pt | checkpoints/rdc1/A/C/h7200.pt | FROZEN |
| 0.04 | distinct | checkpoints/step_lora.pt | checkpoints/step_lora_grpo.pt | FROZEN |
| 0.04 | distinct | checkpoints/step_lora_grpo.pt.wsl | checkpoints/step_lora_grpo_c010.pt | DUPLICATE,FROZEN |
| 0.01 | distinct | checkpoints/confirmed/determinism/p3_tables.pt | checkpoints/p3_tables.pt | FROZEN |
| 0.00 | distinct | checkpoints/confirmed/scaffold-moe/umoe_lb_s1.pt | checkpoints/umoe_lb_s1.pt | FROZEN |
| 0.00 | distinct | checkpoints/confirmed/scaffold-moe/umoe_lb_s2.pt | checkpoints/umoe_lb_s2.pt | FROZEN |
| 0.00 | distinct | checkpoints/confirmed/scaffold-moe/umoe_gravmoe_s1.pt | checkpoints/umoe_gravmoe_s1.pt | FROZEN |
| 0.00 | distinct | checkpoints/confirmed/scaffold-moe/umoe_gravmoe_s2.pt | checkpoints/umoe_gravmoe_s2.pt | DUPLICATE,FROZEN |
| 0.00 | distinct | checkpoints/confirmed/calibration/cal_dilute_10_s1.pt | checkpoints/cal_dilute_10_s1.pt | ESSENTIAL,FROZEN |
| 0.00 | distinct | checkpoints/confirmed/calibration/cal_dilute_30_s1.pt | checkpoints/cal_dilute_30_s1.pt | DUPLICATE,FROZEN |
| 0.00 | distinct | checkpoints/confirmed/grav-spacetime/grav2_contract_s1.pt | checkpoints/grav2_contract_s1.pt | DUPLICATE,FROZEN |
| 0.00 | distinct | checkpoints/confirmed/grav-spacetime/grav2_ctl_s1.pt | checkpoints/grav2_ctl_s1.pt | DUPLICATE,FROZEN |
| 0.00 | distinct | checkpoints/confirmed/crystal-math/sym_birth_dense_mps_h8_ema.pt | checkpoints/sym_birth_dense_mps_h8_ema.pt | FROZEN |
| 0.00 | distinct | checkpoints/writercaf1/qual_hybrid_k1_s23_S1_lr0.0003/feedback.pt | checkpoints/writercaf1/qual_hybrid_k2_s23_S1_lr0.0003/feedback.pt; checkpoints/writercaf1/qual_hybrid_k4_s23_S1_lr0.0003/feedback.pt; checkpoints/writerdfa1/qual_dfa_s21_S1_lr0.0001/feedback.pt; checkpoints/writerdfa1/qual_dfa_s21_S1_lr0.0003/feedback.pt | FROZEN |
| 0.00 | distinct | checkpoints/confirmed/determinism/k3_silu_tab.pt | checkpoints/k3_silu_tab.pt | ESSENTIAL,FROZEN |

## Baseline check

changed 0, missing 0, added 0 (must all be 0 for a read-only run).

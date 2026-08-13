# Archive rebirth: pair reconstruction for 11 frozen figures (2026-08-13)

READ-ONLY survey. Target set: `docs/assets/archive/2026-08-12/`.
Purpose: give Phase A of the archive-rebirth program
(`docs/superpowers/specs/2026-08-13-archive-rebirth-animation-program-design.md:31`)
a pair-verified list before the plan freezes.

Method: figure -> adding commit (`git log --diff-filter=A`) -> commit
subject -> the titles PRINTED IN THE PIXELS (read directly from the PNGs;
the panel titles/footers are the only surviving invocation record) ->
RESULTS/scratch cross-check -> on-disk existence check (`stat`, no weights
loaded).

Standing fact that frames everything below: no recorded invocation
survives for any of these — `scratch/assets_classify.py:32-38` resolved
the whole set to [H] on exactly that ground. Everything here is
RECONSTRUCTION, not recovery.

## Instrument notes (apply to every row)

- `scripts/plot_neurons.py` renders at most TWO panels
  (`--ckpt` + `--compare`, `scripts/plot_neurons.py:191-193`). All four
  three-minds figures are THREE panels, so they came from a driver that
  was never committed. Rebirth needs a `GALLERY` entry that loops over N
  checkpoints, not a plot_neurons flag.
- `--displace` renders ONE panel and exits
  (`scripts/plot_neurons.py:155-188`). Both two-panel displacement
  figures were composites of two runs or of an uncommitted driver.
- Default cmap is `cool` (`scripts/plot_neurons.py:136`). The archived
  three-minds / ternary / qwen-pca panels are inferno/magma-family, so
  they were rendered with an explicit cmap (or by the pre-plot_neurons
  ad-hoc scripts). New renders should take the ramp from
  `figstyle.continuous("magnitude", mode)` anyway, per the figure
  overhaul.
- `neuron_matrix()` picks the MIDDLE matrix whose key contains `--key`
  (`scripts/plot_neurons.py:69-75`), so `--key gate.weight` on a
  single-matrix Qwen extract (`blocks.14.gate.weight`) is exact.

---

## 1. three-minds-crystal.png — LOW (partly UNRECOVERABLE)

Panels read off the pixels: `fp32 champion (69/120)` |
`DeepSeek-V3 expert 42 (671B)` | `ternary-born ep1 (1.58 bits)`;
suptitle `three minds, one lens — the crystal view (PCA)`.

| panel | checkpoint | exists | size |
|---|---|---|---|
| fp32 champion | `checkpoints/mathnative_45m_gen4_std.pt` | YES | 201,572,974 |
| DeepSeek-V3 expert 42 | none on disk | NO | — |
| ternary-born ep1 | none on disk | NO | — |

Best reconstruction (a new 3-entry driver, PCA, mid gate):

```
panels = [("checkpoints/mathnative_45m_gen4_std.pt", "fp32 champion (69/120)"),
          (<V3 expert 42 extract>,                   "DeepSeek-V3 expert 42 (671B)"),
          ("checkpoints/mathnative_45m_ternary_3ep.pt", "ternary-born 3ep (1.58 bits)")]
key="gate.weight", method="pca"
```

Blockers:
- The V3 expert was gauged in-memory from one downloaded 4GB fp8 shard,
  "never inferenced" (`docs/RESULTS.md:2163-2178`); no extract script and
  no `.pt` was ever committed (`git log --grep=V3` over 07-16..07-19
  shows only RESULTS + the two asset commits). The shard is not on disk.
- `ternary-born ep1` is a per-epoch state of the 07-17 ternary birth
  (`docs/RESULTS.md:2296` "Ternary-from-birth: 63/120"). `train_ternary.py`
  saves only the FINAL deployed + latent snapshots
  (`scripts/train_ternary.py:57,70`), so ep0/ep1 were overwritten. Only
  `mathnative_45m_ternary_3ep.pt` (201,569,297) and
  `mathnative_45m_ternary.pt` (6ep, 201,569,297) survive.

Honest rebirth options (each changes the claim and must be RELABELED,
not passed off as the old figure):
- swap the missing MoE panel for a real on-disk expert:
  `checkpoints/v4flash_f1/` (DeepSeek-V4-Flash, MXFP4 group-32; needs
  `llmopt.lab.shards.dequant`) or `checkpoints/k3_expert_l45_e7/`;
- swap `ternary ep1` for `mathnative_45m_ternary_3ep.pt`.
Evidence: `docs/assets/archive/2026-08-12/three-minds-crystal.png`
(panel titles), commit `da2ec63`, `docs/RESULTS.md:2163`,
`docs/RESULTS.md:2296`, `scripts/train_ternary.py:57`.

## 2. three-minds-sphere.png — LOW (same pair as #1)

Same three checkpoints, `method="sphere"`. Same two missing panels, same
substitution options. Evidence: commit `da2ec63` (single commit, four
lenses), `scripts/plot_neurons.py:8-12` (sphere definition).

## 3. three-minds-polar.png — LOW (same pair as #1)

Same three checkpoints, `method="polar"`, `normalize=False`.
Evidence: commit `da2ec63`.

## 4. three-minds-polar-normalized.png — LOW (same pair as #1)

Verified from the pixels: suptitle `three minds — polar, normalized`,
y-axis `magnitude / median`, shared y-limits — i.e. `method="polar"`,
`normalize=True` (`scripts/plot_neurons.py:204-217`, the median-normalize
+ shared-ylim path). Same missing panels.

## 5. neurons-qwen-vs-19m.png — MEDIUM

Pixels: `TWO CRYSTALS — each dot a neuron in its own weight-space PCA
plane, color = neuron magnitude`; panels `Qwen-0.5B — grown on the
internet` (foot: `L14 gate: 4,864 neurons, 896-dim · trained on
everything`) and `math-native 19M — grown in the closed system` (foot:
`mid gate: 1,536 neurons, 384-dim · has only ever seen verified
calculus`).

| panel | checkpoint | exists | size |
|---|---|---|---|
| Qwen-0.5B L14 gate | `checkpoints/qwen05b_base_l14gate.pt` | YES | 17,434,308 |
| math-native 19M mid gate | `checkpoints/mathnative_19m.pt` | YES | 75,667,321 |

Reconstruction:

```
.venv/bin/python scripts/plot_neurons.py \
  --ckpt checkpoints/qwen05b_base_l14gate.pt --title "Qwen-0.5B — grown on the internet" \
  --foot "L14 gate: 4,864 neurons, 896-dim · trained on everything" \
  --compare checkpoints/mathnative_19m.pt --title2 "math-native 19M — grown in the closed system" \
  --foot2 "mid gate: 1,536 neurons, 384-dim · has only ever seen verified calculus" \
  --key gate.weight --method pca --out docs/assets/gallery/<new-name>.png
```

Caveats (why MEDIUM, not HIGH): the original PREDATES plot_neurons — it
was `two_crystals.png`, added by `c29b2a2` ("two-crystals render") and
only renamed into the neurons-* namespace by `e77e4a3`; the bold
suptitle/footers are that ad-hoc script's styling, which plot_neurons
does not reproduce. The Qwen matrix is the L14 gate_proj [4864, 896] —
the shape matches the archived footer exactly — but the surviving
extract `qwen05b_base_l14gate.pt` was written 2026-08-08
(`scratch/qwen_displace_extract.py:1-14`), three weeks after the figure;
base-vs-Instruct for the original panel is UNRECORDED
(`checkpoints/qwen05b_instruct_l14gate.pt`, 17,434,400, also on disk).
19M identity is strong: `mathnative_19m.pt` mtime Jul 15 14:43, figure
mtime Jul 15 14:57, and the footer's 1,536 x 384 matches the 19M mid
gate used by `scratch/crystal_recreate_test.py:49-52`.
Evidence: commits `c29b2a2`, `e77e4a3`; `scratch/qwen_displace_extract.py:8-9`.

## 6. neurons-polar-qwen-vs-19m.png — HIGH

Pixels: suptitle `each dot a neuron (gate.weight, polar), color =
magnitude` — the exact plot_neurons suptitle format
(`scripts/plot_neurons.py:237-241`), so `--key gate.weight --method
polar`, no `--normalize` (y-axis reads `neuron magnitude`, unshared
limits). Panel titles `Qwen2.5-0.5B (internet)` / `math-native 19M
(closed system)`.

| panel | checkpoint | exists | size |
|---|---|---|---|
| Qwen | `checkpoints/qwen05b_base_l14gate.pt` | YES | 17,434,308 |
| 19M | `checkpoints/mathnative_19m.pt` | YES | 75,667,321 |

```
.venv/bin/python scripts/plot_neurons.py \
  --ckpt checkpoints/qwen05b_base_l14gate.pt --title "Qwen2.5-0.5B (internet)" \
  --compare checkpoints/mathnative_19m.pt --title2 "math-native 19M (closed system)" \
  --key gate.weight --method polar --out docs/assets/gallery/<new-name>.png
```

Same base-vs-Instruct caveat as #5 (the archived left panel's
dead-neuron clump at phase 0, magnitude ~0.22, is a discriminator a
rebirth run can check against). Evidence: commit `1b5bcd0` ("generated
by plot_neurons.py"), `scripts/plot_neurons.py:237`.

## 7. neurons-pca-ternary-vs-fp32.png — MEDIUM (one panel substituted)

Sibling of #9 (same commit `5910768`, three lenses in one push).
Panels (read from #9, same pair): `ternary-born, epoch 0 (1.58 bits)` |
`fp32-born gen4_std (champion)`, feet `nonzero frac 0.73` / `1.00`.

| panel | checkpoint | exists | size |
|---|---|---|---|
| ternary-born ep0 | none (overwritten) | NO | — |
| fp32 champion | `checkpoints/mathnative_45m_gen4_std.pt` | YES | 201,572,974 |

```
.venv/bin/python scripts/plot_neurons.py \
  --ckpt checkpoints/mathnative_45m_ternary_3ep.pt --title "ternary-born, 3 epochs (1.58 bits)" \
  --compare checkpoints/mathnative_45m_gen4_std.pt --title2 "fp32-born gen4_std (champion)" \
  --key gate.weight --method pca --out docs/assets/gallery/<new-name>.png
```

Blocker: ep0 is gone (`scripts/train_ternary.py:57,70` saves final only).
The 3ep substitute is a DIFFERENT claim (the archived foot `nonzero frac
0.73` is an ep0 statistic; a 3ep render will print its own). The fp32
panel is exact: `mathnative_45m_gen4_std.pt` is named as production in
`docs/RESULTS.md:2132-2146` ("production = mathnative_45m_gen4_std.pt"),
written Jul 17 13:59, figure Jul 17 19:28.

## 8. neurons-sphere-ternary-vs-fp32.png — MEDIUM (same pair as #7)

Same two checkpoints, `--method sphere`. Same ep0 blocker.
Evidence: commit `5910768`.

## 9. neurons-polar-ternary-vs-fp32.png — MEDIUM (same pair as #7)

Verified from the pixels: suptitle `the ternary toddler vs the fp32
champion — mid-layer gate neurons, polar` (a CUSTOM suptitle — the
committed plot_neurons hardcodes its own, `scripts/plot_neurons.py:237`,
so either the July version differed or a wrapper set it), x-label
`phase (rad) | nonzero frac 0.73`. Unnormalized (panel y-ranges differ:
0.40-0.65 vs 0.58-0.87). Same ep0 blocker; the normalized sibling
`neurons-polar-ternary-vs-fp32-normalized.png` is in the same archive
directory and rides the same pair.

## 10. neurons-19m-zoom.png — MEDIUM (name is a misnomer)

Pixels: `THE WHISPER, DEVELOPED — zoom into the RL crystal` /
`central lattice of L14 gate · displacements drawn x60 · every neuron
nudged, none uprooted` / foot `the entire 2.4x verified climb`. This is
NOT the 19M: `L14 gate` + `2.4x RL climb` is the Qwen2.5-0.5B LoRA-RL
pair from the weight-anatomy era. The 19M name arrived only when
`crystal_zoom.png` was bulk-renamed by `e77e4a3`.

| role | artifact | exists | size |
|---|---|---|---|
| base matrix | `checkpoints/qwen05b_base_l14gate.pt` | YES | 17,434,308 |
| pre-RL adapter | `checkpoints/step_lora_pre_grpo_backup.pt` | YES | 35,298,315 |
| post-RL adapter | `checkpoints/step_lora_grpo.pt` | YES | 35,301,393 |

Reconstruction: materialize W0 = base + 2.0 * (B@A) from
`step_lora_pre_grpo_backup.pt` and W1 = base + 2.0 * (B@A) from
`step_lora_grpo.pt` at key `model.layers.14.mlp.gate_proj` — the exact
recipe already committed in `scratch/crystal_recreate_test.py:33-44`
(SCALE = 2.0 = lora alpha/r) — then either draw the LineCollection
directly (that script's panel 2) or write the two matrices as
`blocks.14.gate.weight` .pt files and call:

```
.venv/bin/python scripts/plot_neurons.py --ckpt W0.pt --displace W1.pt \
  --key gate.weight --mult 60 --zoom 0.2 --out docs/assets/gallery/<new-name>.png
```

Caveats: the archived `x60` matches the plot_neurons `--mult` default of
60 (`scripts/plot_neurons.py:142`) but the ZOOM quantile is unrecorded;
the original may have been the pre-plot_neurons ad-hoc renderer from
`10b972e` ("THE CRYSTAL as repo hero + whisper heatmap + x60 zoom").
`step_lora_grpo.pt` mtime Jul 15 09:09 vs figure Jul 15 11:58 — the
pairing is consistent. Evidence: commit `10b972e`,
`scratch/crystal_recreate_test.py:5-8,33-44`.

## 11. neurons-zoom-fp32-vs-ternary-displacement.png — LOW (right panel UNRECOVERABLE)

Pixels: suptitle `displacement zoom — how each mind moves: analog nudges
vs discrete flips`; left `fp32: RL climb (v2.1 base -> grpo c10), x60`,
right `ternary: growth (ep0 -> ep1, deployed), x8`. Two panels, so a
composite (plot_neurons `--displace` exits after one,
`scripts/plot_neurons.py:187`).

| panel | checkpoints | exists | size |
|---|---|---|---|
| left, from | `checkpoints/mathnative_45m_v21.pt` | YES | 201,572,445 |
| left, to | `checkpoints/mathnative_45m_grpo_c010.pt` | YES | 201,572,538 |
| right, from/to | ternary ep0 / ep1 deployed | NO | — |

Left panel reconstruction (MEDIUM):

```
.venv/bin/python scripts/plot_neurons.py \
  --ckpt checkpoints/mathnative_45m_v21.pt \
  --displace checkpoints/mathnative_45m_grpo_c010.pt \
  --key gate.weight --mult 60 \
  --title "fp32: RL climb (v2.1 base -> grpo c10)" --out .../left.png
```

Left-panel ambiguity: three `*_c010.pt` candidates exist from that week
(`mathnative_45m_grpo_c010.pt`, `mathnative_45m_grpo_run1_c010.pt`,
`mathnative_45m_grpo3_c010.pt`, all 201,572,538/201,572,631). The plain
`grpo_c010` is the best read of the title but is not certain.

Right panel: UNRECOVERABLE — the ternary per-epoch deployed snapshots
were never saved separately (`scripts/train_ternary.py:57,70`). Honest
substitute for a REBORN figure, relabeled: the 19M growth pair
`checkpoints/gallery19m_s1_ep0.pt` (75,668,024) ->
`checkpoints/gallery19m_s1_ep1.pt` (75,668,024), which exists precisely
because `scratch/birth19m_snaps.py` was run to give the displacement
view an [R] pair (`scripts/INDEX.md:1297`). Evidence: commit `6160727`,
`scripts/plot_neurons.py:155-188`, `scripts/INDEX.md:1297`.

---

## Summary

| figure | confidence | blocker |
|---|---|---|
| three-minds-crystal | LOW | V3 expert 42 shard gone; ternary ep1 gone |
| three-minds-sphere | LOW | same |
| three-minds-polar | LOW | same |
| three-minds-polar-normalized | LOW | same |
| neurons-qwen-vs-19m | MEDIUM | pre-plot_neurons styling; Qwen base-vs-Instruct unrecorded |
| neurons-polar-qwen-vs-19m | HIGH | Qwen base-vs-Instruct unrecorded |
| neurons-pca-ternary-vs-fp32 | MEDIUM | ternary ep0 gone (3ep substitute) |
| neurons-sphere-ternary-vs-fp32 | MEDIUM | same |
| neurons-polar-ternary-vs-fp32 | MEDIUM | same + custom suptitle not in the committed script |
| neurons-19m-zoom | MEDIUM | mislabeled (Qwen L14, not 19M); zoom quantile unrecorded |
| neurons-zoom-fp32-vs-ternary-displacement | LOW | ternary ep0/ep1 gone; c010 run ambiguous |

Curated-set recommendation for Phase A: ship the two qwen-vs-19m views
and the ternary-vs-fp32 trio (with the 3ep relabel), rebuild the
19m-zoom under an honest name, and either drop the three-minds set or
rebuild it as a NEW "three minds" with an on-disk MoE expert
(`v4flash_f1` / `k3_expert_*`) and the 3ep ternary — never as a
reproduction of the 07-17 pixels.

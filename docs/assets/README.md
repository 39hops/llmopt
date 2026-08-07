# Asset index — the crystal gallery

Figures are claims: each row names the claim, the fence, and where
the verdict lives in `docs/RESULTS.md`. House background #0d1117.
Grouped by question, not by date.

Provenance policy (2026-08-08 pass): `scripts/plot_neurons.py` now
stamps every render with checkpoint sha256[:8] + repo HEAD, and has
`--normalize` (per-panel median) for cross-substrate SHAPE claims —
raw magnitude scales differ per alphabet and would fake or hide
texture agreement. Historical portraits are FROZEN as first
rendered (git history is their provenance); regenerated figures
carry the stamp in their bottom-left corner. Generator status is
marked per row: `[R]` = reproducible from a committed script,
`[H]` = historical one-off, generator not committed (frozen
as-rendered; do not overwrite).

## The identity era (the repo's strongest claim)

| Image | Claim / fence |
|---|---|
| `identity-crest-fresh-seeds.png` `[R]` (`plot_identity_crest.py`) | capability follows expert IDENTITY: swap-derived keep-set +53, named 80-carrier deletion +55, rank-matched control +28 (router over-inclusive) — all 3/3 at fresh seeds 1001/2002/3003. VERDICT EX-FRESH. FORMAT-BOUND, one vehicle, mathgen L1-3, Mac MLX |

## Internet diet vs closed system

| Image | Claim / fence |
|---|---|
| `neurons-qwen-vs-19m.png` `[H]`, `neurons-polar-qwen-vs-19m.png` `[H]` | the easy tell: web-pretrained Qwen-0.5B (dense isotropic cloud) vs math-native 19M (sparse ring structure), same PCA/polar lens. Qualitative, n=1 per side, gate.weight lens only |
| `neurons-polar-four-diets.png` + `-normalized` + `neurons-phase-density-four-diets.png` `[H]` | four diets, four textures — diet moves the imprint (CV as dial). Phase-density instrument had a booked bug-and-fix cycle; see RESULTS before quoting |
| `neurons-polar-v3expert-vs-19m.png` `[H]` | one 671B-expert shard vs the 19M under the same polar lens — texture continuity, not a scale law. n=1 shard |

## Alphabet / precision (the tie)

| Image | Claim / fence |
|---|---|
| `neurons-gen6-ternary-vs-grown.png` `[R]` (regenerated 2026-08-08, normalized + stamped) | crown pair: ternary-born latent vs fp32 grown champion — same recipe, different alphabet; normalized view shows ternary's WIDER magnitude spread (0.78-1.37 v tight fp32), a real difference auto-scaling hid. Crown tie itself is n=1-births (BOARD fence) |
| `neurons-pca-ternary6ep-vs-fp32.png`, `neurons-polar-ternary6ep-vs-fp32.png` `[H]` | the 69/120-both tie in weight space at 1.58 bits v fp32 |
| `neurons-pca-ternary-vs-fp32.png`, `-sphere-`, `-polar-` (+`-normalized`) `[H]` | earlier 3-epoch ternary (63) vs champion |
| `neurons-binary768-vs-ternary.png` `[H]` | binary-768 vs ternary geometry |
| `three-minds-crystal/polar/sphere.png` + `three-minds-polar-normalized.png` `[H]` | fp32 champion / DeepSeek-V3 expert 42 shard / ternary-born under one lens — shared banded texture across substrates. Prefer the `-normalized` variant for the shape claim (unnormalized panels have incommensurable y-ranges) |

## Training dynamics

| Image | Claim / fence |
|---|---|
| `neurons-19m.png`, `neurons-19m-zoom.png` `[H]` | the first crystal (README hero image) + the SFT-v-RL displacement view: SFT mean displacement ~0.123, RL climb ~0.007 (~19x stiller) — policy edit as small coordinated geometry. FROZEN as first rendered |
| `rl-vs-sft-weight-delta.png` `[H]` | layer x projection heatmaps: SFT pours the crystal (dW ~61), RL whispers (~4); CKA(pre, climbed) ~0.9998 |
| `neurons-ternary-growth-ep3-ep6.png`, `neurons-113m-growth-*.png` `[H]` | growth movies: texture forms in epoch 1, then freezes |
| `neurons-gen6-grown-vs-champion.png`, `neurons-gen6-ternary-polar.png` `[H]` | gen6 lineage portraits |
| `neurons-zoom-fp32-vs-ternary-displacement.png` `[H]` | cross-substrate displacement |

## Width / capacity

| Image | Claim / fence |
|---|---|
| `neurons-wfloor-d128-vs-19m.png`, `neurons-wfloor-d256-vs-d64.png` `[H]` | width-floor ladder geometry (the d-cliff sits in (48,56]) |
| `neurons-sphere-45m-vs-113m.png` `[H]` | 45M v 113M direction structure |
| `width-curve-gen4.png` `[H]` | the width-capability curve |

## Spectral / crystallography lenses

| Image | Claim / fence |
|---|---|
| `symmetry-rmt-and-gtheta.png` `[H]` | eigenvalue density v Marchenko-Pastur edge (mass beyond edge = learned structure) + angular pair-distribution g(theta): DeepSeek expert shows a sharp ~90-degree peak. Physics-as-method lens; claims live in the symmetry RESULTS entries |
| `neuron-weighting-pr.png`, `neuron-density-vs-phase.png` `[H]` | participation-ratio / density instruments (1/3 law + MP-sliver bookings) |

## Wanted (figures the record has earned)

- GT-1 crest small-multiples (masked v full, 6 paired seeds +
  domain-specificity: math +14.7 v mechanics -59) — data booked,
  needs a `plot_*` script in the `[R]` class.
- Re-render of `three-minds` from the committed script + the
  on-disk `k3_expert_*` shards, so the gallery's best cross-scale
  figure leaves the `[H]` class.

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
| `gt1-crest-small-multiples.png` `[R]` (`plot_gt1_crest.py`, Grok-authored via Artin relay) | the crest itself: 45.3% math-demand mask beats paired full at 6/6 seeds (+14.7 pooled, R4+R5) AND the same recipe craters on mechanics (-59 pooled, D4-PHYS-B) — matched recall does not predict sign. Per-seed pairing explicit (different baselines per seed pool) |

## Internet diet vs closed system

| Image | Claim / fence |
|---|---|
| `neurons-qwen-vs-19m.png` `[H]`, `neurons-polar-qwen-vs-19m.png` `[H]` | the easy tell: web-pretrained Qwen-0.5B (dense isotropic cloud) vs math-native 19M (sparse ring structure), same PCA/polar lens. Qualitative, n=1 per side, gate.weight lens only |
| `neurons-polar-four-diets.png` + `-normalized` + `neurons-phase-density-four-diets.png` `[H]` | four diets, four textures — diet moves the imprint (CV as dial). Phase-density instrument had a booked bug-and-fix cycle; see RESULTS before quoting |
| `neurons-polar-v3expert-vs-19m.png` `[H]` | one 671B-expert shard vs the 19M under the same polar lens — texture continuity, not a scale law. n=1 shard |

## Alphabet / precision (the tie)

| Image | Claim / fence |
|---|---|
| `neurons-gen6-ternary-vs-grown.png` `[R]` (regenerated 2026-08-08, normalized + stamped) | crown pair: ternary-born latent vs fp32 grown champion — same recipe, different alphabet; normalized view shows ternary's WIDER magnitude spread (0.78-1.37 v tight fp32 — IN-IMAGE render-time measurement, quote only as rendered), a real difference auto-scaling hid. Crown tie itself is n=1-births (BOARD fence) |
| `neurons-pca-ternary6ep-vs-fp32.png`, `neurons-polar-ternary6ep-vs-fp32.png` `[H]` | the 69/120-both tie in weight space at 1.58 bits v fp32 |
| `neurons-pca-ternary-vs-fp32.png`, `-sphere-`, `-polar-` (+`-normalized`) `[H]` | earlier 3-epoch ternary (63) vs champion |
| `neurons-binary768-vs-ternary.png` `[H]` | binary-768 vs ternary geometry |
| `three-minds-crystal/polar/sphere.png` + `three-minds-polar-normalized.png` `[H]` | fp32 champion / a 671B-MoE expert shard / ternary-born under one lens (VEHICLE UNVERIFIED 2026-08-08 claims-audit: 'DeepSeek-V3 expert 42' appears nowhere in RESULTS — the booked DeepSeek shard entry names expert 71, and the Wanted re-render names k3_expert_* = Kimi-K3; pin the source ckpt at re-render before captioning) — shared banded texture across substrates. Prefer the `-normalized` variant for the shape claim (unnormalized panels have incommensurable y-ranges) |

## Training dynamics

| Image | Claim / fence |
|---|---|
| `neurons-19m.png`, `neurons-19m-zoom.png` `[H]` | the whisper suite, BOTH panels/files Qwen-0.5B run-2b (L14 gate, 4,864 neurons, 896-dim — the 19M's mid gate is 1,536@384; the "19m" filename prefix is a rename-era misnomer, kept frozen per [H]). `neurons-19m.png` = SFT arm (displacement 0.123, "lattice on fire") v RL arm (0.007, "nineteen times stiller") side-by-side; the ZOOM = the faint RL panel developed at x60 (Artin's ask, 07-15). The 0.123/0.007/19x figures are IN-IMAGE render-time measurements frozen with the artifact — quote them only as "as rendered"; the RESULTS-booked whisper numbers are the weight-anatomy entry's (dW 4.0 v 61-87, ~6% of one SFT run's movement, CKA 0.9998 all layers but last). FROZEN as first rendered |
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
| `symmetry-rmt-and-gtheta.png` `[H]` | eigenvalue density v Marchenko-Pastur edge (mass beyond edge = learned structure) + angular pair-distribution g(theta): DeepSeek expert shows a sharp ~90-degree peak. Physics-as-method lens; AUDIT 2026-08-08: NO booked RESULTS entry carries these claims (repo-wide grep clean) — all numbers incl. the ~90-degree peak are in-image render-time only, quote only as rendered |
| `neuron-weighting-pr.png`, `neuron-density-vs-phase.png` `[H]` | participation-ratio / density instruments (1/3 law + MP-sliver: RIFF-LEDGER banks, NOT RESULTS bookings — unregistered, quote as riff-tier) |

## Wanted (figures the record has earned)

- ~~GT-1 crest small-multiples~~ SHIPPED 2026-08-08
  (`gt1-crest-small-multiples.png`, Grok-authored).
- Re-render of `three-minds` from the committed script + the
  on-disk `k3_expert_*` shards, so the gallery's best cross-scale
  figure leaves the `[H]` class.

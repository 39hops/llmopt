# Flagship animation storyboard — crystal_rotation prototype (2026-08-13)

Direction source: Artin/Sol art brief (2026-08-13): cinematic data
visualization, data occupies the frame, minimal persistent text,
12-18s, motion explains relationships, complexity from truthful
layers only. Approval checkpoint before propagating the language to
training_morph, crest_race, and the static figures.

Story: the same 12,288 weight vectors reveal different structure
under three mathematical views.

## Beats (total ~15.5s)

| # | t | frame | motion |
|---|-----|-------|--------|
| 1 | 0.0-2.5 | INSIDE the PCA cloud, dots nearly edge-to-edge; opening claim "12,288 weight vectors - one model" fades in and OUT | slow drift; camera already moving |
| 2 | 2.5-5.0 | camera pulls back; the full geometry lands; direct label "PCA - global variation" appears beside the cloud, brief | continuous zoom-out |
| 3 | 5.0-6.5 | 60 tracked neurons ignite (magnitude ramp + stroke); the other 2,900 dots recede to a quiet density field | opacity crossfade, no text |
| 4 | 6.5-9.5 | morph PCA -> SPHERE; tracked neurons draw motion trails; label "SPHERE - magnitude removed" briefly near the cloud | correspondence-preserving morph |
| 5 | 9.5-12.5 | morph SPHERE -> POLAR; trails continue; label "POLAR - phase vs magnitude" | same |
| 6 | 12.5-15.5 | cloud settles left; two small data-true echoes (PCA, SPHERE) fade in right, tracked neurons highlighted in all three - the composed conclusion; provenance fence, small, 1.2s | settle + compose |

## Visual layers (all data-derived)

- projection ENDPOINTS: data/anim/crystal.npz (pca/sphere/polar xy
  from gallery19m_s1.pt, per-axis normalized) - data-derived
- color: house magnitude ramp on row-norm rank (npz `order`)
- density field: the real remaining ~2,900 subsampled neurons at low
  opacity - not a synthetic texture
- tracked subset: 16 neurons, deterministic stratified sample (per
  layer, the 30th and 70th percentile of within-layer magnitude
  rank; 8 layers x 2) - identities real and preserved across views
- trails: decaying comet tails between projection endpoints. These
  are ANIMATION-GENERATED correspondence guides (linear interpolation
  is a rendering choice), not data - only the endpoints are true
- outro echoes: the same tracked+background dots re-plotted at the
  other two views' true coordinates

Endpoints and identities are data; in-between motion is presentation.

## Honesty constraints carried

- per-axis normalization per view is a PROJECTION choice, stated in
  the assets README (unchanged); no normalization implies temporal
  change (single checkpoint - no time axis here)
- tracked subset is a fixed seeded draw, not curated
- provenance outro stays (compressed to one line, 1.2s)

## Checkpoint

Draft MP4 + poster delivered with this storyboard. On approval:
propagate language to training_morph (temporal honesty: angle churn
v small norm growth, tracked trajectories), crest_race (paired seeds
as the central object), then static figures 5-12. Naming cleanup to
snake_case rides with the propagation commit.

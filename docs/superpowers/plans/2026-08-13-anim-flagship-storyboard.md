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
| 3 | 5.0-6.5 | 16 tracked neurons ignite (magnitude ramp + stroke); the other 2,900 dots recede to a quiet density field | opacity crossfade, no text |
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

## House animation grammar (adopted 2026-08-13, Artin ruling)

North-star reference: 2swap, "Double Pendulums are Chaoticn't"
(youtube dtjb2OhEQcU, ~6:27) - principles only, never its imagery:
full-frame data fields, luminous density with crisp detail, few
foreground actors over a rich population, decaying temporal trails,
continuous evolution (no slide transitions), minimal text, quiet
guides, restrained color, generous holds.

Every llmopt scene uses this layer structure:

- **Field**: the complete population, filling most of the canvas
- **Actors**: 8-20 highlighted identity-preserving elements
- **Memory**: fading trajectory/history
- **Geometry**: understated axes/contours/density/projection structure
- **Text**: one opening claim at most; short direct labels only when
  necessary
- **Receipt**: provenance on an isolated end card - NEVER over live
  data; posters use the clean composition, never the receipt card

Density glow rule: any glow/bloom must come from the actual point
density (e.g. the same points drawn wide and faint), never synthetic
noise or invented fields.

## Checkpoint

Draft MP4 + poster delivered with this storyboard. On approval:
propagate language to training_morph (temporal honesty: angle churn
v small norm growth, tracked trajectories), crest_race (paired seeds
as the central object), then static figures 5-12. Naming cleanup to
snake_case rides with the propagation commit.

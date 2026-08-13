# Sol 5.6 figure redesigns — work list (2026-08-13)

Source: Sol 5.6 external review (relayed by Artin, morning ruling
adopted). Executed immediately: honesty-ledger drift fix + drift
guard test + fourth README result block (`ce24884`). Everything
below is the standing redesign queue, in ruling order.

**Ruling: no README GIF until the three scenes are redesigned;
crystal-rotation is then the only likely candidate.**

## Animations (scripts/anim/)

1. `crystal_rotation` — rename the concept "Three views of the same
   weights" (it is a projection morph, not a rotation); shorten to
   ~6s; visually track a small fixed set of neurons (highlight
   30-100 dots) so the transformation is intelligible.
2. `training_morph` — REDESIGN, currently misleading: per-epoch
   normalization suppresses absolute-scale change and rank coloring
   is distribution-invariant, so it looks static while claiming
   growth. Fix: ONE fixed coordinate system across checkpoints, a
   measured statistic on screen (e.g. mean row norm per epoch),
   track 100-300 neurons over a faint density background.
3. `crest_race` — motion adds nothing over a mean bar chart and
   hides the paired-seed evidence. Replace with three paired
   full→mask comparisons appearing sequentially; unpaired zero
   controls in a visually separate gray group.
4. All scenes: light variant too (currently dark-only), caption
   sizes readable at README width, ≤6-8s loops, reduced-motion
   path = poster + MP4 link.

## Static figures

5. Hero — lead with the POLAR view + two direct annotations;
   PCA/sphere become a smaller strip below (triptych reads as three
   similar blobs on mobile; no "so what" for a newcomer).
6. routing_crest — paired slope/dot plot for seeds 111/222/333
   (the design IS the evidence); zero controls in a separate gray
   callout. Needs per-seed values in figures.json (they are in the
   fence already: 63→80, 73→82, 63→81).
7. merge_space — births as a range band, six independent merges
   stacked at zero, same-init replications as points; stop
   presenting unlike evidence as seven equivalent rails.
8. effective_context — plot change from k=16 (or subdued lines +
   heavy mean + a "≈1 nat" bracket); current crossings distract
   from the registered finding.
9. honesty_ledger — KEEP (strongest composition); reduce the
   explanatory duplication around it in README prose.
10. Gallery crests — grouped bars become paired deltas; GT-1's two
    panels need one shared takeaway grammar.
11. qwen-vs-19m-polar — title states a FINDING not object names;
    direct annotations on the structural difference; caption makes
    explicit that color is within-model rank, not cross-model
    magnitude.
12. General: stop repeating each headline three times (README
    heading, figure title, prose) — one claim, said once.

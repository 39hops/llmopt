# Figure and asset overhaul — design (2026-08-12)

Status: DESIGN, approved direction (Artin, 2026-08-12 night, after a
Grok + GPT cross-review of the first proposal). No implementation in
this commit.

## Goal

The README's first screenful is the lab's front door: thesis, hero,
first result. The hero and the analysis figures currently under-sell
the work — heavy text stacks, an all-blue sequential ramp, and a
`docs/assets/` directory where 6 MB run renders sit beside 80 KB
publication graphics. Reorganize the assets so the existing taxonomy
is visible on disk, and evolve the figure style so charts talk and
text shrinks.

## Decisions locked (with provenance)

1. **The taxonomy stays; directories mirror it.** The
   `[PUBLISHED] / [HERO] / [R] / [H]` classes in `docs/assets/README.md`
   are the classification system; the move makes them physical:

   ```
   docs/assets/
   ├── README.md
   ├── web/                  # [PUBLISHED] — unchanged
   ├── hero/                 # [HERO] — neurons-19m-{light,dark}.png
   ├── gallery/              # [R] — regenerable renders
   └── archive/2026-08-12/   # [H] — frozen pixels, moved as-is
   figs/                     # run exhaust, SEPARATE, untouched
   ```

   `figs/` never merges into assets: run exhaust vs curated assets is
   a hard boundary (GPT point, adopted).

2. **No Seaborn.** The problem is text hierarchy and ramp choice, not
   the plotting library. Style v2 is an evolution of
   `llmopt/figures/figstyle.py` + `anatomy.py`, no new dependency.

3. **The magnitude ramp is inferno — by evidence, not aesthetics.**
   The crystal-era look Artin wants is `cmap="inferno"`:
   `scratch/crystal_recreate_test.py:80` and
   `scratch/nineteen_m_displace.py:38` both use it. Style v2 exposes
   `figstyle.continuous("magnitude", mode)` returning an inferno-based
   colormap, sliced per surface (dark keeps the luminous top, light
   drops the near-white top steps; exact slice tuned at implementation
   against both surfaces). `anatomy.render_dot_views()` consumes it —
   every future dot view inherits the treatment. No `crystal_axes()`
   wrapper (rejected as unnecessary abstraction). Categorical series
   palette untouched.

4. **Hero: same geometry, crystal color language, radically less
   text.** `checkpoints/gallery19m_s1.pt` exists, so this is a
   re-render, not a reconstruction. Keep the three-panel
   PCA / SPHERE / POLAR composition (one matrix, three coordinate
   systems — the README explains it directly underneath, so the image
   stops repeating the explanation). Target text budget:

   - title: a few words (≤ ~6), no em dash subtitle
   - panel labels: PCA / SPHERE / POLAR, optional one tiny descriptor
     each; no equations
   - one small ramp cue (low → high ‖w‖ rank)
   - one tiny mono provenance line (ckpt sha + HEAD) — the fence
     stays; it is part of the lab's identity

   `three-minds-crystal.png` is a DESIGN REFERENCE (color, density,
   drama), not a promotion candidate; it stays [H] in the archive.

5. **Two passes, strictly ordered.**
   - **Pass 1 — mechanical, no pixel changes**: classify per the
     assets README rows, `git mv` into the mirror layout, update live
     references (README + current docs only), add a link-integrity
     test scoped to README + docs/paper + assets README. Historical
     handoffs/RESULTS keep their old paths untouched — a handoff is
     itself historical evidence; the test does NOT police history.
   - **Pass 2 — visual, [HERO] and [R] only**: style v2 in figstyle,
     hero re-render, then regenerate the [R] gallery from surviving
     script+checkpoint pairs. Rule: **if it can be regenerated,
     improve it; if it cannot, preserve it.** [H] pixels are never
     touched.

6. **Style v2 — the frozen ruleset** (small, not a design system):
   - Inter for presentation text; JetBrains Mono for fences only.
   - Inferno-family continuous ramp for magnitude/density; validated
     categorical palette for series; status colors reserved.
   - Near-white / near-black surfaces; dark is selected, not flipped.
   - One claim-sized title (≤ ~6 words); one OPTIONAL scope line;
     no paragraph annotations on the chart.
   - Direct labels over legends where practical.
   - No decorative grids on geometry/dot views; minimal axes when the
     coordinates are not themselves the result.
   - One tiny provenance fence line.
   - Light + dark from the same script, same run.
   - Raster sized to display target: README-width PNGs, no casual
     5-6 MB emissions (quantize/downscale on install, as the hero
     already does).
   - **The master rule: if the README prose beside the image already
     says it, it does not go inside the chart.**

7. **web/ [PUBLISHED] polish is scoped small**: layouts stay; trim
   scope-line verbosity, fix any text overflow, add `<title>` hover
   tooltips (free in hand-emitted SVG). Numbers and structure
   unchanged; `gen_readme.py --check` must stay green.

## What could go wrong

| risk | mitigation |
|---|---|
| A move breaks a live image link | link-integrity test lands in the SAME commit as the moves; scoped to live docs only |
| Re-rendered [R] figure silently changes a story a doc tells | [R] figures carry no booked numbers (those are [PUBLISHED]); still, each re-render is eyeballed against its predecessor before replacing |
| Inferno fails contrast on the light surface | the slice is validated per surface before adoption; if the light variant cannot pass, light keeps a darker inferno slice (not a different hue family) |
| Hero re-render drifts from the frozen provenance contract | renderer already stamps ckpt sha + HEAD; the new render re-stamps with current HEAD, old hero pixels freeze into archive as [H] |
| Scope creep into animations | explicitly out of scope here; animated-SVG capability is its own future spec |

## Out of scope

- Animated figures / GIFs (separate spec when taken up).
- Re-rendering [H] (forbidden by class definition).
- figs/ reorganization (run exhaust, stays as-is).
- Categorical palette changes.

# Figure/Asset Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute spec `2026-08-12-figure-overhaul-design.md`: mirror
the `[PUBLISHED]/[HERO]/[R]/[H]` taxonomy on disk, then restyle
[HERO] + [R] with the inferno magnitude ramp and the minimal-text
rules. Two passes, strictly ordered; Pass 1 changes zero pixels.

**Architecture:** Pass 1 = classify, `git mv`, fix live links, guard
test. Pass 2 = `figstyle.continuous("magnitude", mode)` (inferno
slice), hero re-render from `checkpoints/gallery19m_s1.pt`, [R]
gallery regeneration, web/ text polish. All work Mac-side (headless
Chrome for web PNGs is Mac-only).

**Tech Stack:** matplotlib (in-tree), Pillow (hero quantize, already
used), pytest. No new dependencies. No Seaborn.

## Global Constraints

- [H] pixels are never regenerated, edited, or deleted; they move
  once into `docs/assets/archive/2026-08-12/` and freeze.
- Historical docs (handoffs, RESULTS) keep their old asset paths —
  the link test polices LIVE surfaces only: `README.md`,
  `docs/paper/main.tex`, `docs/assets/README.md`.
- Categorical series palette untouched; `scripts/gen_readme.py
  --check` stays green through every commit.
- Hero text budget (spec §4): title ≤ ~6 words, panel labels + one
  tiny descriptor max, one ramp cue, one mono fence line. Nothing
  else. Master rule: if README prose beside the image says it, it
  does not go in the chart.
- Every commit: pytest rc captured (never piped into commit), ruff
  clean. Public repo: `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>`, never a session URL.
- Each re-rendered [R] figure is eyeballed against its predecessor
  (Read both PNGs) before the old one is replaced.

---

### Task 1: classification manifest (read-only)

**Files:**
- Create: `scratch/assets_classify.py`

**Interfaces:**
- Produces: printed table `class \t filename` for every file in
  `docs/assets/*.png`, classes {PUBLISHED, HERO, R, H, UNKNOWN},
  derived from the rows in `docs/assets/README.md` (HERO =
  `neurons-19m-{light,dark}.png`; R = the enumerated stems
  `gt1-crest-*`, `identity-crest-*`, `neurons-gen6-*`,
  `neurons-pca-*`, four-diets set, `neurons-113m-growth-*`; H = the
  enumerated frozen set + anything cited by a RESULTS/scratch
  reference; web/ = PUBLISHED). UNKNOWN rows are printed first.

- [ ] **Step 1:** Write the script: glob `docs/assets/*.png`, match
  against the class stem lists copied from the assets README, print
  the table with UNKNOWN first.
- [ ] **Step 2:** Run it. If UNKNOWN is nonempty, classify each by
  checking for a surviving script+checkpoint pair (`grep` the stem in
  scripts/ + scratch/, `ls` the checkpoint it names): pair exists → R,
  else → H. Record the final table in the Task 2 commit message.
- [ ] **Step 3:** Commit `chore: assets classification manifest`
  (regen CODEMAP/INDEX after `git add` — scratch file added).

### Task 2: Pass 1 — mirror the taxonomy on disk

**Files:**
- Move: per Task 1 table — `git mv` into `docs/assets/hero/`,
  `docs/assets/gallery/`, `docs/assets/archive/2026-08-12/`
- Modify: `README.md:12,16` (hero srcset/src paths),
  `docs/assets/README.md` (add the layout block from the spec §1,
  update file paths in class rows)
- Modify: `scripts/render_hero_neurons.py` (output path gains
  `hero/`)
- Create: `tests/test_asset_links.py`

**Interfaces:**
- Produces: the directory layout of spec §1; a guard test other
  tasks must keep green.

- [ ] **Step 1:** Write the failing test first:

```python
"""Live-surface image links resolve (spec 2026-08-12 figure overhaul).

Scoped to LIVE docs only: README.md, docs/paper/main.tex,
docs/assets/README.md. Historical handoffs/RESULTS keep their old
paths as evidence and are exempt by design.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = ["README.md", "docs/paper/main.tex", "docs/assets/README.md"]
RX = re.compile(r"docs/assets/[\w\-./]+\.(?:png|svg)")


def test_live_asset_links_resolve():
    missing = []
    for doc in LIVE:
        p = ROOT / doc
        if not p.exists():
            continue
        for ref in set(RX.findall(p.read_text())):
            if not (ROOT / ref).exists():
                missing.append(f"{doc} -> {ref}")
    assert not missing, "\n".join(missing)
```

- [ ] **Step 2:** Run it — must PASS pre-move (baseline sanity), then
  perform the `git mv` batch per the Task 1 table, run again — now it
  FAILS on the two README hero refs. Fix `README.md:12,16` to
  `docs/assets/hero/…`, update `docs/assets/README.md` rows and add
  the layout tree, update the output path in
  `scripts/render_hero_neurons.py`. Test PASSES.
- [ ] **Step 3:** `grep -rn "docs/assets" scripts/ llmopt/ | grep -v
  web/` — any generator writing to a moved path gets its output path
  updated in the same commit (plot_neurons.py, plot_gt1_crest.py,
  plot_identity_crest.py write into gallery/).
- [ ] **Step 4:** Full suite rc captured; commit
  `refactor: docs/assets mirrors the figure taxonomy (Pass 1, zero
  pixel changes)` with the classification table in the body.

### Task 3: Pass 2a — the magnitude ramp

**Files:**
- Modify: `llmopt/figures/figstyle.py`
- Modify: `llmopt/figures/anatomy.py:134-137`
- Test: `tests/test_figstyle.py` (extend)

**Interfaces:**
- Produces: `figstyle.continuous(kind: str, mode: str)` returning a
  matplotlib `Colormap`; only `kind="magnitude"` defined for now
  (ValueError otherwise). Dark mode: inferno sliced to keep the
  luminous top (start ≈ 0.12, end = 1.0). Light mode: inferno sliced
  to drop the near-white/yellow top (start = 0.0, end ≈ 0.82).
  Exact endpoints tuned by eye against both surfaces in Task 4 and
  then FROZEN in the test.

- [ ] **Step 1:** Failing test:

```python
def test_continuous_magnitude():
    import matplotlib
    dark = figstyle.continuous("magnitude", "dark")
    light = figstyle.continuous("magnitude", "light")
    assert isinstance(dark, matplotlib.colors.Colormap)
    assert dark(0.9) != light(0.9)          # per-surface slices differ
    with pytest.raises(ValueError):
        figstyle.continuous("volume", "dark")
```

- [ ] **Step 2:** Implement via
  `matplotlib.colors.LinearSegmentedColormap.from_list` over
  `plt.get_cmap("inferno")(np.linspace(start, end, 256))`.
- [ ] **Step 3:** `anatomy.render_dot_views` swaps its
  SEQUENTIAL-reversal block for `figstyle.continuous("magnitude",
  mode)`. No other behavior change.
- [ ] **Step 4:** Targeted tests + ruff; commit.

### Task 4: Pass 2b — the hero re-render

**Files:**
- Modify: `scripts/render_hero_neurons.py` (title/caption text per
  the spec budget; ramp cue line; fence stays)
- Replace: `docs/assets/hero/neurons-19m-{light,dark}.png`

- [ ] **Step 1:** Cut the text: title to `19M gate-neuron geometry`
  (5 words), drop the scope sentence and equations; panel labels
  PCA / SPHERE / POLAR each with at most one tiny descriptor; add the
  ramp cue `low ← ‖w‖ rank → high`; keep the one-line mono fence
  (ckpt sha + HEAD).
- [ ] **Step 2:** Render both modes from
  `checkpoints/gallery19m_s1.pt`; Read the PNGs and eyeball against
  the current hero: dot geometry identical, inferno ramp, text
  budget respected, no clipped text. Tune the Task 3 slice endpoints
  here if the light surface fails contrast, then freeze them in the
  Task 3 test.
- [ ] **Step 3:** Quantize/downscale per the existing hero install
  step; check file sizes stay in the current band (≤ ~1.5 MB each).
- [ ] **Step 4:** Old hero pixels are ALREADY safe (Pass 1 moved
  nothing for hero; the replace overwrites regenerable [HERO] files —
  the pre-overhaul pixels remain in git history, which the assets
  README notes). Suite + ruff; commit; verify README renders both
  variants (Read the PNGs once more post-quantize).

### Task 5: Pass 2c — [R] gallery regeneration

**Files:**
- Modify: `scripts/plot_neurons.py`, `scripts/plot_gt1_crest.py`,
  `scripts/plot_identity_crest.py` (consume
  `figstyle.continuous("magnitude", mode)` where they color by
  magnitude; apply the text-budget rules: one short title, no
  paragraph annotations)
- Replace: regenerable PNGs under `docs/assets/gallery/`

- [ ] **Step 1:** For each [R] file from the Task 1 table, confirm
  its script+checkpoint pair still exists (`ls` the checkpoint). Pair
  gone → reclassify H, `git mv` to archive, note in commit.
- [ ] **Step 2:** Regenerate survivors one script at a time; Read old
  vs new side by side before replacing (the eyeball gate from Global
  Constraints).
- [ ] **Step 3:** Suite + ruff; one commit per script family
  (`plot_neurons`, `gt1_crest`, `identity_crest`) so a bad batch
  reverts alone.

### Task 6: Pass 2d — web/ [PUBLISHED] polish

**Files:**
- Modify: `llmopt/figures/figsvg.py` (scope-line trim: cap scope text
  at one line, drop any second sentence; `<title>` child on every
  data mark for hover tooltips; fix text overflow if the trim reveals
  any)
- Regenerate: `docs/assets/web/*.svg` + PNGs via
  `scripts/gen_figures_web.py` (headless Chrome, Mac)

- [ ] **Step 1:** Add `<title>{label}: {value}</title>` inside the
  rect/circle emit paths in `gate_track` and `curves` (SVG-native
  tooltip, zero JS).
- [ ] **Step 2:** Scope lines: render each figure, Read the SVGs,
  shorten any scope text that wraps past one line by editing the
  scope strings in `docs/figures.json`? NO — figures.json is the
  numbers source; scope text lives there too, so trims are EDITS to
  figures.json scope fields only (titles/numbers untouched), same
  commit, `gen_readme.py --check` green.
- [ ] **Step 3:** Regenerate SVG + PNG pairs, Read 2-3 spot checks,
  suite + ruff, commit.

### Task 7: close

- [ ] Full suite rc captured; ruff; push; 3080 ff-only sync with HEAD
  hash asserted; BOARD line for the thread updated; CI watched to
  green with the bounded headSha watcher pattern.

---

## Self-review notes

- Spec §1-§7 map: §1→Task 2, §2→(no task, constraint), §3→Task 3,
  §4→Task 4, §5→Tasks 1-2 vs 3-6 ordering, §6→Tasks 3-6 constraints,
  §7→Task 6. Risks table: link test same-commit (Task 2), eyeball
  gate (Tasks 4-5), light-surface fallback (Task 4 Step 2), no
  animation scope anywhere.
- Type consistency: `continuous(kind, mode)` used identically in
  Tasks 3, 4, 5.
- The one open value (inferno slice endpoints) is explicitly a
  tune-then-freeze, not a placeholder: Task 4 Step 2 tunes, Task 3
  test freezes.

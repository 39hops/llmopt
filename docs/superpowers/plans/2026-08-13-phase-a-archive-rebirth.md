# Phase A: Archive Rebirth + Multi-Format Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebirth the curated archive figures as recorded-invocation
[R] renders and ship a one-render/many-profiles export layer
(README + LinkedIn + source sizes).

**Architecture:** `llmopt/figures/export.py` (profile emitter) +
`scripts/render_gallery.py` (the GALLERY list = the recorded
invocations). Renders flow through the existing anatomy/figstyle
stack; no new style decisions.

**Tech Stack:** matplotlib, Pillow (both in-tree). No new deps.

## Global Constraints

- Frozen archive pixels are never edited, replaced, or deleted;
  rebirths land under NEW names in `docs/assets/gallery/`.
- Every gallery render's invocation lives in the GALLERY list —
  no unrecorded one-off render commands.
- Inferno ramp + text budget via `figstyle.continuous` and the
  style v2 rules; provenance fence on every render.
- One-resident-30B rule: no big-model loads beside a live job;
  checkpoint EXISTENCE comes from the rebirth survey, weights load
  one at a time.
- Every commit: pytest rc captured separately (never piped into
  commit), ruff clean, `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>`, never a session URL.
- GALLERY entries whose checkpoint pair the survey marks
  UNRECOVERABLE are dropped with a note in the commit body — never
  approximated silently.

---

### Task 1: export profiles

**Files:**
- Create: `llmopt/figures/export.py`
- Test: `tests/test_figure_export.py`

**Interfaces:**
- Produces: `export_profiles(render_png: Path, outdir: Path, stem:
  str) -> dict[str, Path]` — takes ONE full-res source PNG, emits
  `{stem}-readme.png` (1600px wide, 256-color median-cut),
  `{stem}-linkedin.png` (exactly 1200×627, letterboxed on the
  source's corner-sampled surface color), `{stem}-source.png`
  (copy). Returns profile->path map.

- [ ] **Step 1:** Failing test:

```python
def test_export_profiles(tmp_path):
    from PIL import Image
    from llmopt.figures.export import export_profiles
    src = tmp_path / "fig.png"
    Image.new("RGB", (3200, 1400), "#101010").save(src)
    out = export_profiles(src, tmp_path, "fig")
    readme = Image.open(out["readme"])
    linkedin = Image.open(out["linkedin"])
    assert readme.width == 1600
    assert (linkedin.width, linkedin.height) == (1200, 627)
    assert out["source"].exists()
```

- [ ] **Step 2:** Run it — FAIL (module missing). Implement with
  Pillow: LANCZOS resize for readme + `quantize(colors=256,
  method=Image.MEDIANCUT)`; linkedin = fit-within-1200×627 +
  letterbox with the (0,0)-pixel color; source = copy.
- [ ] **Step 3:** Test green; ruff; commit
  `feat: figure export profiles (readme/linkedin/source)`.

### Task 2: the gallery driver

**Files:**
- Create: `scripts/render_gallery.py`

**Interfaces:**
- Consumes: `anatomy.neuron_rows`, `anatomy.render_dot_views`,
  `figstyle.continuous`, `export.export_profiles`.
- Produces: `GALLERY: list[Entry]` where
  `Entry = (name, ckpts, key, method, normalize, title)`;
  `python scripts/render_gallery.py` regenerates every entry into
  `docs/assets/gallery/`, `--only <name>` filters.

- [ ] **Step 1:** Skeleton with an EMPTY GALLERY list + main loop
  (iterate entries, render via anatomy for single-matrix views and
  a compare path for pairs, then export_profiles). Smoke: runs with
  empty list, exits 0.
- [ ] **Step 2:** Populate GALLERY from
  `scratch/rebirth_pairs_2026-08-13.md` (the survey report):
  high-confidence entries only; verify each checkpoint path exists
  (`ls`) at entry-add time. Medium/low confidence entries go into a
  `# DEFERRED — needs Artin confirmation` comment block, not the
  list.
- [ ] **Step 3:** Render one entry (`--only`), Read the PNG, eyeball
  against its archived ancestor (Read both), then render all.
  Commit per family (three-minds / qwen / ternary / zoom) so a bad
  batch reverts alone.
- [ ] **Step 4:** Update `docs/assets/README.md` [R] row with the
  rebirth rule + new files; regen CODEMAP/INDEX (new script);
  full suite rc captured; ruff; commit.

### Task 3: close

- [ ] Full suite + ruff; push; 3080 ff-only sync with HEAD hash
  asserted; bounded CI watcher (headSha-startswith pattern);
  BOARD figure-thread line updated.

---

## Self-review notes

- Spec §A.1→Task 2, §A.2→Task 1, §A.3→Task 2 Step 2 (survey-gated),
  §A.4→Task 2 Step 4. No placeholders; the one external input (the
  rebirth survey) is an explicit gate, not a TBD.
- Type consistency: `export_profiles(render_png, outdir, stem)`
  used identically in Tasks 1 and 2.

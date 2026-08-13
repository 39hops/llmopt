# Archive rebirth + animation + housekeeping program — design (2026-08-13)

Status: DESIGN, approved direction (Artin, 2026-08-13 night session:
targets GitHub README + LinkedIn; engine = manim; curated ~8 rebirth
set; three scenes; full housekeeping pass). Overnight-scale program;
each phase gets its own implementation plan.

## Phase A — archive rebirth + multi-format export

The 26 files in `docs/assets/archive/2026-08-12/` are [H] because no
recorded invocation survives. Rebirth = NEW renders under NEW names in
`gallery/`, invocation recorded in code, so the [R] bar (regenerable
without reconstruction) holds by construction. Frozen pixels never
touched; old files stay in the archive as the evidence record.

1. `scripts/render_gallery.py` — the driver. A `GALLERY` list of
   entries `(name, ckpt(s), key, method, normalize, title, foot)`;
   running the script with no args regenerates every entry. The list
   IS the recorded invocation. New gallery renders are added by
   adding an entry — never by an unrecorded one-off command.
2. `llmopt/figures/export.py` — render once at high dpi, emit
   profiles per figure:
   - `readme`: 1600px-wide PNG, 256-color median-cut, light + dark
   - `linkedin`: 1200×627 PNG, letterboxed on the house surface
   - `source`: full-res PNG (not committed if large; export dir
     configurable)
   Profile sizes pinned by test. Consumes anatomy/figstyle (inferno
   ramp, text budget) unchanged.
3. Curated rebirth set (~8, pair-verified BEFORE the plan freezes;
   any entry whose checkpoint is gone is dropped with a note, never
   faked): three-minds set (crystal/sphere/polar), qwen-vs-19m (two
   views), ternary-vs-fp32 (two views), 19m-zoom displacement.
4. Assets README gains the rebirth rule: archive files may be
   REBORN as new [R] entries via render_gallery.py; the frozen
   original is never replaced or deleted.

## Phase B — manim animations (three scenes, data-true)

Engine: manim CE (Artin's pick, deps accepted). Isolation: manim
lives in `.venv-anim`, never in the lab venv; the main suite must
stay green with `.venv-anim` absent. Scenes in `scripts/anim/`.

1. Style bridge `scripts/anim/house_theme.py`: maps figstyle
   constants (surfaces, CHROME inks, inferno slices via
   `figstyle.continuous`, series slots, Inter/JetBrains Mono) into
   manim config. No colors named anywhere else.
2. Scenes, each loading real artifacts and ending on a
   provenance-fence outro frame (ckpt sha / figures.json + HEAD):
   - `CrystalRotation` — the 19M gate-neuron cloud
     (gallery19m_s1.pt), camera orbit, PCA→sphere→polar morphs.
   - `TrainingMorph` — neuron positions interpolated across training
     checkpoints (growth family), the crystal forming.
   - `CrestRace` — routing-crest rails animating to the booked
     numbers read from docs/figures.json; the zero arms land silent
     and stay flat. Numbers never hardcoded in the scene.
3. Export profiles per scene: MP4 1080p H.264 (LinkedIn), GIF ≤10MB
   at 880w (README-embeddable; drop to 15fps/640w if needed), poster
   PNG. One render config, three artifacts.
4. Publication: animations land in `docs/assets/anim/` (new class
   row [ANIM] in the assets README: regenerable, script+checkpoint
   recorded, GIF/MP4/poster triplet). README embeds at most ONE GIF
   (front door stays fast); the rest link.

Fallback (pre-registered): if manim install fails on this Mac, the
same three scenes ship via matplotlib frames + ffmpeg with identical
export profiles; the scene data loaders are shared either way.

## Phase C — housekeeping: execute the APPROVED taxonomy spec

Not a new design. `docs/superpowers/specs/2026-07-24-repo-taxonomy.md`
is already approved (BOARD housekeeping gate row, 2026-07-24):
program-based `checkpoints/{engine,lora05b,micromodel,ternary,
metabolic,continents,instruments}` + `data/{chains,continents,diets,
exchange,qual,archive}` + typed `scripts/` subdirectories.

1. Survey first (Opus 5 sub-agents, read-only): classify the 201
   UNCITED CODEMAP files, map cross-imports (`sys.path` couplings),
   list hardcoded data globs, produce the move manifest with
   per-file class + risk.
2. Execution gates (unchanged from the standing policy): natural
   freeze point, pytest green + smoke-launch every entry point +
   both checkouts hash-lockstep; CODEMAP class checked per file;
   results-cited files never move; gate_rarity census glob goes
   recursive + bins re-frozen in the same pass (named in the BOARD
   row).
3. The 51GB checkpoint triage (BOARD row, Artin GO) stays SEPARATE —
   bulk deletion is not part of this program.

## Cross-cutting

- Order A → B → C. A's loaders feed B's scenes. C last so moves
  never race figure work.
- Every commit: suite green (rc captured), ruff clean, public-repo
  commit rules.
- One-resident-30B rule: no big-model loads beside a live job.
- Sub-agent policy: Opus 5 reviewers/surveyors, cap 5; Opus may
  write when directed; session model verifies everything that lands.

## Risks

| risk | mitigation |
|---|---|
| manim deps fail on Mac | pre-registered matplotlib+ffmpeg fallback, same scenes/exports |
| Qwen/three-minds checkpoints missing | pair-verify before plan freeze; drop with note |
| GIF too big for README | 15fps/640w profile; README embeds one GIF max |
| taxonomy moves break sys.path cross-imports | survey maps imports first; smoke-launch gate per entry point |
| 3080 checkout drift during moves | hash-lockstep assert after every push |

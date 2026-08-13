# manim CE feasibility scout — Mac, 2026-08-13

READ-ONLY scout. Nothing was installed; nothing outside this file was modified.
Platform: Darwin 25.3.0, Apple silicon, `/opt/homebrew` present.

**VERDICT: pip-only feasible = YES** (no brew, no sudo), because the native
libraries manim's source-built dependency needs (cairo, pango, pkg-config) are
already installed and the Xcode toolchain is present. One caveat, LaTeX-only,
in the CAVEATS section.

## 1. Prerequisites already on this machine

| Prereq | Status |
|---|---|
| `ffmpeg` | `/opt/homebrew/bin/ffmpeg`, version 8.1.2 |
| `palettegen` / `paletteuse` filters | both present in that ffmpeg |
| brew `cairo` | installed, `/opt/homebrew/opt/cairo`, pkg-config 1.18.4 |
| brew `pango` | installed, pkg-config 1.58.0 |
| brew `pkg-config` | installed |
| brew `glib`, `harfbuzz` | installed (pango deps) |
| brew `py3cairo` | NOT installed (not needed; pip builds pycairo itself) |
| `pkg-config --exists cairo pango` | ok |
| C toolchain | `/usr/bin/clang`, full Xcode at `/Applications/Xcode.app` |
| TeX | TeX Live 2026 at `/Library/TeX/texbin` (latex, pdflatex, xelatex) — **`dvisvgm` MISSING** |

Pythons:

- lab venv `/Users/artin/code/llmopt/.venv/bin/python` = **3.12.4**
  (has matplotlib 3.11.0, numpy 2.5.1, torch 2.12.1)
- `python3` on PATH = pyenv shim -> 3.14.4
- homebrew pythons available: 3.9, 3.11, 3.12, 3.13, 3.14
  (`/opt/homebrew/bin/python3.12`, `python3.13`, ...)

## 2. manim CE requirements (from https://pypi.org/pypi/manim/json)

Latest version **0.21.0**, `requires_python >= 3.11`.

requires_dist: audioop-lts (py>=3.13), av>=15.0, beautifulsoup4, click, cloup,
decorator, isosurfaces, **manimpango>=0.6.1,<1.0.0**, mapbox-earcut, moderngl
>=5.7,<6, moderngl-window>=2, networkx, numpy>=2.1, pillow>=11, **pycairo
>=1.14,<2**, pydub, pygments, rich, scipy>=1.13 (>=1.15 on py3.13),
screeninfo, skia-pathops>=0.9, srt, svgelements, tqdm, typing-extensions,
watchdog. Extras: gui (dearpygui), jupyterlab, typst.

Native-dependency wheel audit on PyPI (macOS arm64):

| Package | latest | arm64 wheel? |
|---|---|---|
| **pycairo** | 1.29.1 | **NO wheels at all** on macOS (Windows-only wheels). sdist -> builds from source against cairo via pkg-config. Same for 1.26–1.29. |
| **manimpango** | 0.6.1 | YES — `cp39/310/311/312/313/314 macosx_11_0_arm64` |
| skia-pathops | 0.9.2 | YES — `cp310-abi3-macosx_10_9_universal2` (abi3 covers 3.12/3.13; universal2 includes arm64). Only the pypy wheel is arch-tagged arm64, which is why a naive filename scan looks empty. |
| av | 18.1.0 | YES — `cp311-abi3-macosx_14_0_arm64` (abi3, covers 3.12+; macOS 14+ min, satisfied) |
| moderngl | 5.12.0 | YES — cp38–cp313 arm64. **No cp314 wheel** |
| mapbox-earcut | 2.0.0 | YES — cp39–cp314 arm64 |
| numpy / scipy / pillow | current | YES — cp312/313/314 arm64 |
| moderngl-window, screeninfo, isosurfaces, svgelements, pydub | — | pure-python wheels |

## 3. Will `python3 -m venv .venv-anim && pip install manim` succeed without brew?

**Yes on this machine.** The only source build is **pycairo**, and its build
needs exactly `cairo` headers discoverable by `pkg-config` plus a C compiler.
Both are already satisfied (brew cairo 1.18.4, pkg-config ok, clang present).
manimpango ships an arm64 wheel that statically carries what it needs, so brew
pango is not strictly required at install time either (it is installed anyway).

The claim "arm64 wheels exist for pycairo" is **false** — verified against the
release JSON for 1.26.0 through 1.29.1. pip-only still works here only because
the brew cairo dev files happen to be installed. On a clean Mac without brew
cairo, `pip install manim` WOULD fail at the pycairo build, and
`brew install cairo pkg-config` would be mandatory.

**Python version choice: use 3.12 or 3.13, not 3.14.** moderngl 5.12.0 has no
cp314 arm64 wheel, and the pyenv `python3` shim resolves to 3.14.4. Use
`/opt/homebrew/bin/python3.13` (or 3.12) explicitly.

Do NOT install manim into the lab `.venv`: manim pins `numpy>=2.1` and drags
in av/moderngl/pycairo next to torch 2.12.1 and the transformers<5 mlx pin
(see CLAUDE.md "Mac MLX env pin"). Separate venv, no exceptions.

## 4. GIF pipeline

- manim CE supports `--format [png|gif|mp4|webm|mov]` natively (confirmed
  against the current docs configuration page). The old `-i/--save_as_gif`
  shorthand is deprecated; use `--format=gif`. `-t/--transparent` gives an
  alpha channel.
- manim's built-in GIF writer is convenient but does not do a two-pass
  palette. For anything with gradients or antialiased text, render mp4 and
  convert with the ffmpeg palette pass; the local ffmpeg 8.1.2 has both
  `palettegen` and `paletteuse`.

Palette recipe (already runnable today, no new installs):

```
ffmpeg -y -i in.mp4 -vf "fps=30,scale=1280:-1:flags=lanczos,palettegen=stats_mode=diff" /tmp/pal.png
ffmpeg -y -i in.mp4 -i /tmp/pal.png -lavfi "fps=30,scale=1280:-1:flags=lanczos,paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" out.gif
```

## 5. torch checkpoints inside a manim scene — recommended interface

Do **not** import torch in the anim venv. Two-stage, precompute-then-render:

1. **Lab venv (torch)** runs a precompute script that loads the checkpoint,
   does whatever projection/PCA/trajectory extraction is wanted, and writes a
   single `.npz` of plain float32 arrays plus a small JSON-able metadata dict
   (`np.savez_compressed`, with a `meta` key holding `json.dumps(...)`).
2. **Anim venv (numpy only)** loads that `.npz` and renders. The scene never
   sees a `state_dict`, never needs the model class on the import path, and
   never pulls torch's 2+ GB of wheels into a second environment.

This is cleaner for four concrete reasons: (a) it keeps torch and manim's
numpy pin from fighting; (b) the `.npz` is a reviewable, diffable artifact so
a figure can be re-rendered without re-running the checkpoint load; (c) it
matches the existing house pattern where `llmopt/figures/` consumes computed
arrays rather than models; (d) the precompute step can run under the lab's
existing oracle/verification discipline, and only verified numbers cross the
boundary.

Suggested contract (mirrors the figstyle/save shape already in-tree):

```
# lab venv:  scripts/anim_precompute_<topic>.py  ->  figs/<YYYY-MM-DD>/<topic>.npz
#   keys: one array per animated quantity, shape (T, N, D) time-major
#         "meta" : 0-d array of a JSON string (labels, units, source ckpt path, seed)
# anim venv: scratch/anim/<topic>_scene.py  ->  media/videos/... -> figs/<date>/<topic>.gif
```

Pin the palette by importing the hex values from `llmopt/figures/figstyle.py`
by hand-copy into a small `anim_palette.py` in the anim venv (figstyle imports
matplotlib, which the anim venv should not need). Do not copy `scripts/figlib.py`
colors — that file is marked SUPERSEDED and its palette fails the colorblind
check.

## INSTALL PLAN (exact commands, in order)

Nothing here needs sudo. Steps marked BREW are only needed on a machine that
does not already have cairo/pkg-config; **on this Mac they are already
satisfied and can be skipped**.

```
# 0. (BREW, SKIP ON THIS MAC — already installed) prerequisites for the
#    pycairo source build:
#    brew install cairo pkg-config
#    (pango, glib, harfbuzz, ffmpeg also already present)

# 1. (plain) create the isolated animation venv with an explicit 3.13
#    (NOT `python3`, which is the pyenv 3.14 shim — moderngl has no cp314 wheel)
/opt/homebrew/bin/python3.13 -m venv /Users/artin/code/llmopt/.venv-anim

# 2. (plain) modern pip so abi3/universal2 wheels resolve
/Users/artin/code/llmopt/.venv-anim/bin/python -m pip install -U pip wheel

# 3. (plain) install manim; pycairo is the one source build, everything else
#    lands as a wheel
/Users/artin/code/llmopt/.venv-anim/bin/pip install "manim==0.21.0"

# 4. (plain) verify
/Users/artin/code/llmopt/.venv-anim/bin/manim --version
/Users/artin/code/llmopt/.venv-anim/bin/python -c "import manim, cairo, manimpango, numpy; print(manim.__version__, cairo.version, numpy.__version__)"

# 5. (plain) smoke render to GIF, into a scratch dir (not figs/)
/Users/artin/code/llmopt/.venv-anim/bin/manim -ql --format=gif -o smoke \
    --media_dir /tmp/manim_media scratch/anim/smoke_scene.py SmokeScene

# 6. gitignore the venv and media dir before any commit
#    add to .gitignore:  .venv-anim/   media/
```

Optional, only if `MathTex`/`Tex` is wanted (see CAVEATS):

```
# (needs sudo, TeX Live's package manager)
sudo /Library/TeX/texbin/tlmgr install dvisvgm
# or (BREW) brew install dvisvgm
```

## CAVEATS

1. **`dvisvgm` is missing.** TeX Live 2026 is installed with latex/pdflatex/
   xelatex, but manim converts LaTeX output to SVG with `dvisvgm`, which is not
   in `/Library/TeX/texbin`. Any scene using `Tex` or `MathTex` will fail until
   it is installed (tlmgr, sudo — or brew). Scenes using `Text` /
   `MarkupText` go through Pango and work without any TeX at all. For a first
   animation, prefer `Text` and dodge the whole issue.
2. `/Library/TeX/texbin` is not on the default PATH in a non-login shell;
   export it if TeX scenes are used.
3. pycairo compiles from source, so the first install takes ~30-60 s and will
   break if brew ever removes cairo. Record the resolved versions
   (`pip freeze > scratch/anim-requirements.lock.txt`) so the render env is
   reproducible.
4. manim writes into `media/` under the cwd by default — pass `--media_dir`
   somewhere untracked, per logs doctrine (`logs/` is run exhaust; big
   mp4/gif intermediates must not land in git).

## FALLBACK: matplotlib + ffmpeg (no new installs at all)

Already available in the lab venv today: matplotlib 3.11.0, numpy 2.5.1,
torch 2.12.1, and ffmpeg 8.1.2 on PATH. Existing in-repo material to build on:

- `/Users/artin/code/llmopt/llmopt/figures/figstyle.py` — the validated house
  palette, vendored Inter/JetBrains Mono fonts in `assets/fonts/`, light+dark
  surfaces, and a `figure()/save()` pair. Colors were computed against a CVD
  gate, so an animation inherits an accessible palette for free.
- `/Users/artin/code/llmopt/llmopt/figures/figures.py`, `anatomy.py`,
  `figsvg.py`, `export.py` — existing figure forms.
- `/Users/artin/code/llmopt/scripts/figlib.py` — SUPERSEDED 2026-08-11; kept
  only for the three scripts that import it. Do not build new work on it, and
  do not copy its palette.
- Output convention already in place: `figs/<YYYY-MM-DD>/<name>.svg` (dirs for
  2026-07-22, 07-23, 08-08, 08-09 exist).

No `FuncAnimation`, `imageio`, or ffmpeg subprocess call exists anywhere in the
repo today, so the animation layer is greenfield either way.

Fallback path: `matplotlib.animation.FuncAnimation` -> save PNG frames with the
figstyle figure/save helpers -> `ffmpeg -framerate N -i frame_%04d.png` -> mp4
-> the palettegen/paletteuse pair above for GIF. Zero installs, reuses the
validated palette and the vendored fonts, and keeps everything in one venv.

Honest tradeoff: matplotlib gives frame-by-frame plotting and no easement,
transform, or camera system. For "a chart that moves" it is strictly simpler
and should be the default. manim earns its install only for constructed
explanatory motion — objects transforming into one another, camera moves,
staged reveals of an argument. Pick the fallback unless the target is the
latter.

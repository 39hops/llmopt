"""One render, many surfaces: profile emitter for published figures.

A gallery render is produced ONCE at full resolution; this module
fans it out to the surfaces that consume it:

  readme    1600px wide, LANCZOS, 256-color median-cut quantized —
            the size band the README hero already ships at
  linkedin  exactly 1200x627 (LinkedIn's link/post raster), figure
            fit-within and letterboxed on the render's own surface
            color (sampled at the top-left pixel)
  source    untouched full-res copy

Sizes are pinned by tests/test_figure_export.py. The caller decides
which profiles to commit; big source files can stay untracked.
"""
from __future__ import annotations

import shutil
from pathlib import Path

LINKEDIN_SIZE = (1200, 627)
README_WIDTH = 1600


def export_profiles(render_png: Path, outdir: Path, stem: str,
                    include_source: bool = True) -> dict[str, Path]:
    """Emit the profile set for one full-res render. Returns
    profile name -> written path. Set include_source=False when the
    base render already lives at its permanent path (a copy would be
    a byte-identical duplicate)."""
    from PIL import Image

    render_png = Path(render_png)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    im = Image.open(render_png).convert("RGB")
    out: dict[str, Path] = {}

    # readme: width-pinned, quantized like the hero install step
    h = round(im.height * README_WIDTH / im.width)
    readme = im.resize((README_WIDTH, h), Image.LANCZOS)
    readme = readme.quantize(colors=256, method=Image.MEDIANCUT)
    out["readme"] = outdir / f"{stem}-readme.png"
    readme.save(out["readme"], optimize=True)

    # linkedin: fit-within + letterbox on the render's surface color
    lw, lh = LINKEDIN_SIZE
    scale = min(lw / im.width, lh / im.height)
    fitted = im.resize((round(im.width * scale),
                        round(im.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", LINKEDIN_SIZE, im.getpixel((0, 0)))
    canvas.paste(fitted, ((lw - fitted.width) // 2,
                          (lh - fitted.height) // 2))
    out["linkedin"] = outdir / f"{stem}-linkedin.png"
    canvas.save(out["linkedin"], optimize=True)

    if include_source:
        out["source"] = outdir / f"{stem}-source.png"
        if render_png.resolve() != out["source"].resolve():
            shutil.copyfile(render_png, out["source"])
    return out

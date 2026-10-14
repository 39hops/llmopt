"""Render the web figures: SVG from docs/figures.json, PNG via Chrome.

    .venv/bin/python scripts/gen_figures_web.py            # write
    .venv/bin/python scripts/gen_figures_web.py --check    # exit 1 on drift, never writes

Writes docs/assets/web/<name>[-dark].{svg,png}. The SVG is the artifact;
the PNG exists because GitHub markdown and LinkedIn previews need a
raster. Capture uses the installed Chrome, so PNG regeneration is
Mac-only by design: figures are regenerated deliberately, never in CI,
and the SVGs are committed so nothing downstream depends on a browser.

Before rendering, the honesty_ledger figure is recounted from
docs/FINDINGS.md through llmopt.figures.spec, the same recount and the
same serializer scripts/gen_readme.py uses, so the two generators are
order-independent on docs/figures.json.

--check verifies, in this order, and reports every finding before
exiting 1:
  * docs/figures.json is a fixed point of the recount (no stale counts);
  * every SVG on disk equals a fresh in-memory render of its spec;
  * every PNG exists and its header dimensions equal SCALE x the SVG's;
  * raster content: when Chrome is present and neither --no-raster nor
    LLMOPT_FIG_NO_RASTER=1 is set, each figure is re-captured and
    compared PIXEL BY PIXEL to the committed PNG (Chrome's PNG bytes
    change with its encoder version; the pixels do not). When Chrome is
    absent the check says so explicitly: a timestamp or a dimension
    proves nothing about content, so without a re-render the raster is
    reported as NOT verified, never as current.
"""
from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from llmopt.figures import spec as figspec  # noqa: E402
from llmopt.figures.figsvg import DATA, render_spec  # noqa: E402

OUT = ROOT / "docs" / "assets" / "web"
FONTS = ROOT / "assets" / "fonts"
FINDINGS = ROOT / "docs" / "FINDINGS.md"
CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
SCALE = 2  # retina PNGs; GitHub serves them at half size and stays crisp
NO_RASTER_ENV = "LLMOPT_FIG_NO_RASTER"


def page(svg: str, w: int, h: int) -> str:
    """Wrap the SVG with @font-face pointing at the vendored files, so
    the capture uses Inter/JetBrains Mono rather than a system guess."""
    return f"""<!doctype html><meta charset="utf-8">
<style>
  @font-face {{ font-family:'Inter'; font-weight:400;
    src:url('file://{FONTS}/Inter-Regular.ttf') format('truetype'); }}
  @font-face {{ font-family:'Inter'; font-weight:600;
    src:url('file://{FONTS}/Inter-SemiBold.ttf') format('truetype'); }}
  @font-face {{ font-family:'JetBrains Mono'; font-weight:400;
    src:url('file://{FONTS}/JetBrainsMono-Regular.ttf') format('truetype'); }}
  html,body {{ margin:0; padding:0; background:transparent; }}
  svg {{ display:block; }}
</style>
{svg}"""


def chrome_available() -> bool:
    return Path(CHROME).exists()


def capture(svg: str, w: int, h: int, png: Path) -> bool:
    if not chrome_available():
        return False
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "f.html"
        src.write_text(page(svg, w, h))
        shot = Path(td) / "f.png"
        # No --user-data-dir: asking Chrome to build a fresh profile
        # hangs indefinitely here. It reuses the default profile,
        # which is fine for rendering a local file.
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--hide-scrollbars",
             f"--force-device-scale-factor={SCALE}",
             f"--screenshot={shot}", f"--window-size={w},{h}", str(src)],
            capture_output=True, timeout=60)
        if not shot.exists():
            return False
        shutil.move(str(shot), png)
        return True


def svg_size(svg: str) -> tuple[int, int]:
    w = int(svg.split('width="', 1)[1].split('"', 1)[0])
    h = int(svg.split('height="', 1)[1].split('"', 1)[0])
    return w, h


def png_size(path: Path) -> tuple[int, int] | None:
    """Width and height from the PNG IHDR chunk, or None if the file is
    not a PNG."""
    head = path.read_bytes()[:24]
    if len(head) < 24 or head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", head[16:24])


def pixel_diff(a: Path, b: Path) -> tuple[int, int]:
    """(number of differing pixels, max channel delta) between two PNGs;
    (-1, -1) when the sizes differ. Exact equality is (0, 0): a changed
    digit moves hundreds of pixels, a Chrome anti-aliasing revision
    moves tens with deltas of a few units, and the caller reports both
    numbers so a reviewer can tell which happened."""
    import numpy as np
    from PIL import Image
    ia, ib = Image.open(a).convert("RGBA"), Image.open(b).convert("RGBA")
    if ia.size != ib.size:
        return -1, -1
    # all four channels, alpha included (Pillow's own getbbox() on an
    # RGBA difference image inspects alpha alone and would call any two
    # opaque captures equal)
    d = np.abs(np.asarray(ia, dtype=np.int16) - np.asarray(ib, dtype=np.int16))
    return int((d.max(axis=2) > 0).sum()), int(d.max())


def pixels_equal(a: Path, b: Path) -> bool:
    return pixel_diff(a, b) == (0, 0)


def chrome_version() -> str | None:
    if not chrome_available():
        return None
    r = subprocess.run([CHROME, "--version"], capture_output=True, text=True,
                       timeout=30)
    return r.stdout.strip() or None


def capture_record(out: Path) -> dict | None:
    p = out / "CAPTURE.json"
    if not p.is_file():
        return None
    try:
        import json
        return json.loads(p.read_text())
    except Exception:
        return None


def figures(spec: dict) -> list[str]:
    return [k for k in spec if not k.startswith("_")]


def write(spec_path: Path, out: Path, findings: Path,
          raster: bool = True) -> int:
    out.mkdir(parents=True, exist_ok=True)
    text = spec_path.read_text()
    new = figspec.recount(text, findings.read_text())
    if new != text:
        spec_path.write_text(new)
        print(f"[recount] {spec_path.relative_to(ROOT) if spec_path.is_relative_to(ROOT) else spec_path} updated")
    spec = figspec.load(new)
    made = pngs = 0
    for name in figures(spec):
        for mode, suffix in (("light", ""), ("dark", "-dark")):
            svg = render_spec(spec[name], mode=mode)
            w, h = svg_size(svg)
            (out / f"{name}{suffix}.svg").write_text(svg)
            ok = raster and capture(svg, w, h, out / f"{name}{suffix}.png")
            made += 1
            pngs += bool(ok)
            print(f"[web] {name}{suffix}  {w}x{h}  "
                  f"svg{' + png' if ok else ' (png not captured)'}")
    if pngs:
        # the raster is reproducible only against the renderer that made
        # it; record it so --check can name version drift when pixels move
        import json
        (out / "CAPTURE.json").write_text(json.dumps({
            "renderer": chrome_version(), "scale": SCALE,
            "pngs": pngs, "note": "PNG pixels are compared exactly by "
            "gen_figures_web.py --check against a fresh capture with the "
            "local Chrome; a different Chrome version can move "
            "anti-aliasing pixels, which the check reports with counts."},
            indent=2) + "\n")
    print(f"\n{made} figures -> {out}")
    return 0


def check(spec_path: Path, out: Path, findings: Path,
          raster: bool = True) -> int:
    problems: list[str] = []
    text = spec_path.read_text()
    try:
        recounted = figspec.recount(text, findings.read_text())
    except Exception as e:  # malformed spec: say so, do not guess
        print(f"[figures] ERROR: {spec_path} cannot be parsed: {e}")
        return 2
    if recounted != text:
        problems.append(f"{spec_path.name}: honesty_ledger counts or "
                        "serialization are stale (run gen_figures_web.py "
                        "or gen_readme.py)")
    spec = figspec.load(text)
    do_raster = raster and chrome_available()
    captured = compared = 0
    with tempfile.TemporaryDirectory() as td:
        for name in figures(spec):
            for mode, suffix in (("light", ""), ("dark", "-dark")):
                stem = f"{name}{suffix}"
                try:
                    svg = render_spec(spec[name], mode=mode)
                except Exception as e:
                    problems.append(f"{stem}.svg: render failed: {e}")
                    continue
                w, h = svg_size(svg)
                svg_path, png_path = out / f"{stem}.svg", out / f"{stem}.png"
                if not svg_path.is_file():
                    problems.append(f"{stem}.svg: missing")
                elif svg_path.read_text() != svg:
                    problems.append(f"{stem}.svg: differs from a fresh render "
                                    "of its spec (stale or hand-edited)")
                if not png_path.is_file():
                    problems.append(f"{stem}.png: missing")
                    continue
                size = png_size(png_path)
                if size != (SCALE * w, SCALE * h):
                    problems.append(f"{stem}.png: header size {size}, expected "
                                    f"{(SCALE * w, SCALE * h)} (stale raster)")
                    continue
                if do_raster:
                    shot = Path(td) / f"{stem}.png"
                    if capture(svg, w, h, shot):
                        captured += 1
                        n_px, delta = pixel_diff(shot, png_path)
                        if (n_px, delta) == (0, 0):
                            compared += 1
                        else:
                            problems.append(
                                f"{stem}.png: {n_px} pixel(s) differ from a "
                                f"fresh Chrome capture of the SVG, max channel "
                                f"delta {delta} (a changed value moves hundreds "
                                "of pixels; a renderer revision moves tens at "
                                "small deltas; either way the raster is not "
                                "reproducible as committed)")
                    else:
                        problems.append(f"{stem}.png: Chrome capture failed; "
                                        "raster not verified")
    n = 2 * len(figures(spec))
    if do_raster:
        rec = capture_record(out) or {}
        local = chrome_version()
        print(f"[figures] raster: verified by re-render, {compared}/{n} "
              "pixel-identical to the committed PNG")
        if rec.get("renderer") and rec["renderer"] != local:
            print(f"[figures] raster: committed PNGs were captured with "
                  f"{rec['renderer']}; this machine has {local}. Pixel "
                  "differences below may be renderer drift; re-capture "
                  "with gen_figures_web.py once reviewed.")
    elif raster:
        print(f"[figures] raster: NOT verified on this machine (no Chrome at "
              f"{CHROME}); PNG existence and header dimensions checked only. "
              "Run --check on a machine with Chrome to verify raster content.")
    else:
        print("[figures] raster: NOT verified (--no-raster or "
              f"{NO_RASTER_ENV}=1); PNG existence and header dimensions "
              "checked only.")
    if problems:
        print(f"[figures] STALE: {len(problems)} finding(s):\n  "
              + "\n  ".join(problems))
        return 1
    print(f"[figures] current ({n} SVG renders match; {n} PNG present)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render or check the web figures.")
    ap.add_argument("--check", action="store_true",
                    help="verify instead of write; exit 1 on drift")
    ap.add_argument("--no-raster", action="store_true",
                    help="skip Chrome: do not capture or compare PNG pixels")
    ap.add_argument("--spec", default=str(DATA), help=argparse.SUPPRESS)
    ap.add_argument("--out", default=str(OUT), help=argparse.SUPPRESS)
    ap.add_argument("--findings", default=str(FINDINGS), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    raster = not (a.no_raster or os.environ.get(NO_RASTER_ENV) == "1")
    fn = check if a.check else write
    return fn(Path(a.spec), Path(a.out), Path(a.findings), raster=raster)


if __name__ == "__main__":
    sys.exit(main())

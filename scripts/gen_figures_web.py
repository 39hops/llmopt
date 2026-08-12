"""Render the web figures: SVG from docs/figures.json, PNG via Chrome.

    .venv/bin/python scripts/gen_figures_web.py

Writes docs/assets/web/<name>[-dark].{svg,png}. The SVG is the artifact;
the PNG exists because GitHub markdown and LinkedIn previews need a
raster. Capture uses the installed Chrome, so PNG regeneration is
Mac-only by design — figures are regenerated deliberately, never in CI,
and the SVGs are committed so nothing downstream depends on a browser.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


from llmopt.lab.figsvg import DATA, render  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "assets" / "web"
FONTS = ROOT / "assets" / "fonts"
CHROME = ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
SCALE = 2  # retina PNGs; GitHub serves them at half size and stays crisp


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


def capture(svg: str, w: int, h: int, png: Path) -> bool:
    if not Path(CHROME).exists():
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


def recount_findings() -> None:
    """Recount FINDINGS by maturity tag and write it back into the spec.

    The ledger figure's whole point is that the negative share is real,
    so it must not be a number someone typed once. Booking a null moves
    this figure the next time it is built.
    """
    import json
    import re
    spec = json.loads(DATA.read_text())
    fig = spec.get("honesty_ledger")
    if not fig:
        return
    text = (ROOT / "docs" / "FINDINGS.md").read_text()
    bullets = re.findall(r"^- \[.*?(?=^- \[|^#|\Z)", text, re.S | re.M)
    tags = {"Replicated": "REPLICATED",
            "Mechanism confirmed": "MECHANISM-CONFIRMED",
            "Single seed": "SINGLE-SEED", "Null": "NULL",
            "Retracted": "RETRACTED"}
    counts = dict.fromkeys(tags.values(), 0)
    for b in bullets:
        head = b.split("(")[0]
        found = [t for t in re.findall(r"\[([A-Z][A-Z-]*)\]", head)
                 if t in counts]
        if len(found) == 1:
            counts[found[0]] += 1
    for part in fig["parts"]:
        part["value"] = counts[tags[part["label"]]]
    total = sum(counts.values())
    fig["fence"] = (f"docs/FINDINGS.md · {total} curated claims · recounted "
                    f"from the source at build time")
    DATA.write_text(json.dumps(spec, indent=2) + "\n")
    neg = counts["NULL"] + counts["RETRACTED"]
    print(f"[recount] {total} claims · {neg} negative "
          f"({100 * neg / total:.0f}%)")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    recount_findings()
    names = [k for k in json.loads(DATA.read_text()) if not k.startswith("_")]
    made = 0
    for name in names:
        for mode, suffix in (("light", ""), ("dark", "-dark")):
            svg = render(name, mode=mode)
            w = int(svg.split('width="', 1)[1].split('"', 1)[0])
            h = int(svg.split('height="', 1)[1].split('"', 1)[0])
            (OUT / f"{name}{suffix}.svg").write_text(svg)
            ok = capture(svg, w, h, OUT / f"{name}{suffix}.png")
            made += 1
            print(f"[web] {name}{suffix}  {w}x{h}  "
                  f"svg{' + png' if ok else ' (png skipped: no Chrome)'}")
    print(f"\n{made} figures -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

"""The gallery driver: every [R] render's invocation, recorded.

An image is [R] (reproducible gallery) only if running THIS script
regenerates it — the GALLERY list is the recorded invocation the
2026-08-13 rebirth program requires. Adding a render = adding an
entry; one-off unrecorded render commands do not produce [R] files.

Entries render through llmopt.lab.anatomy (single matrix -> the
dot-view triptych) and fan out through llmopt.figures.export
profiles into docs/assets/gallery/.

Usage:
  .venv/bin/python scripts/render_gallery.py            # all entries
  .venv/bin/python scripts/render_gallery.py --only NAME
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from llmopt.lab import anatomy
from llmopt.figures.export import export_profiles

OUTDIR = Path("docs/assets/gallery")


@dataclass(frozen=True)
class Entry:
    name: str          # output stem under docs/assets/gallery/
    ckpt: str          # checkpoint path (existence checked at render)
    key: str           # matrix family substring, e.g. "gate.weight"
    title: str         # display title (style v2 budget: <= ~6 words)


# The recorded invocations. Populated from the verified rebirth
# survey (scratch/rebirth_pairs_2026-08-13.md) — high-confidence
# pairs only; deferred candidates stay commented until confirmed.
GALLERY: list[Entry] = []


def render_entry(e: Entry, outdir: Path = OUTDIR) -> list[str]:
    if not Path(e.ckpt).exists():
        raise FileNotFoundError(f"{e.name}: checkpoint {e.ckpt} missing")
    label, W = anatomy.neuron_rows(e.ckpt, e.key)
    outs = anatomy.render_dot_views(
        W, str(outdir / e.name), e.title, source_label=f"rows of {label}",
        provenance=(anatomy.checkpoint_provenance(e.ckpt)
                    + " · render_gallery.py"))
    for full in outs:
        p = Path(full)
        export_profiles(p, outdir, p.stem)
    return outs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    entries = [e for e in GALLERY if a.only in (None, e.name)]
    if a.only and not entries:
        raise SystemExit(f"no GALLERY entry named {a.only}")
    for e in entries:
        for o in render_entry(e):
            print(f"rendered {o}")
    print(f"{len(entries)} entries rendered")


if __name__ == "__main__":
    main()

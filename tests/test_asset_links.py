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

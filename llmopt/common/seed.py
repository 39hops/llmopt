"""String-seed law (CLAUDE.md data-hygiene): stable STRING seeds
only — tuple __hash__ is per-process randomized and killed
reproducibility once."""
from __future__ import annotations

import random


def srng(*parts) -> random.Random:
    """random.Random seeded by the dash-joined string of parts."""
    return random.Random("-".join(str(p) for p in parts))

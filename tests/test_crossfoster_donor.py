"""VERIFIED-ENDOGENOUS-DATA-CROSSFOSTER-1 donor-stage guards: the root
order is a fixed interleave with the atoms-shard level mix (level 4 twice
the others), root seeds follow the registered band law, the sampling seeds
are identical for both donors, the stage law (DONOR-INACCESSIBLE /
DONOR-HISTORIES-DEGENERATE / DONOR-ACCESSIBLE) is pure and inclusive at
the registered thresholds, and the registered constants are literal."""
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    prev = os.environ.get("SMOKE")
    os.environ["SMOKE"] = "1"
    for p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "scratch")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        m = importlib.import_module("crossfoster_donor")
    finally:
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


def test_root_order_interleaves_levels_with_level4_doubled(mod):
    mod.N_PER_LEVEL = {3: 2, 4: 4, 5: 2, 6: 2, 7: 2}
    order = mod.root_order()
    assert order[:6] == [(3, 0), (4, 0), (5, 0), (4, 1), (6, 0), (7, 0)]
    assert len(order) == 12 and sorted(order) == sorted((lv, i) for lv in mod.LEVELS for i in range(mod.N_PER_LEVEL[lv]))
    assert order == mod.root_order()


def test_registered_constants_literal():
    src = (ROOT / "scratch" / "crossfoster_donor.py").read_text()
    for s in ('{3: 3600, 4: 9600, 5: 3600, 6: 3600, 7: 3600}', 'if SMOKE else 6000', 'K = 1', 'OVERLAP_DEGENERATE = 0.9', 'if SMOKE else 4\n', 'WAVE = 8',
              'ROOT_BAND = 8_900_000 if SMOKE else 8_800_000', 'SAMPLE_BAND = 7_900_000 if SMOKE else 7_700_000', 'I0 = 0 if SMOKE else 10',
              'seeds = [SAMPLE_BAND + 1000 * root_index + WAVE * w + b for b in range(WAVE)]', 'if norm(t) == cur_norm:', 'sample_wave_lp(model, tok, prompt, seeds, dev)',
              'verify_wave(cur, cands)', 'DONORS = {"A": "checkpoints/gallery19m_phase_s2.pt", "B": "checkpoints/gallery19m_backsched_s2.pt"}'):
        assert s in src, s


def test_stage_law(mod):
    mod.N_RETAIN = 10
    assert mod.adjudicate(9, 0, 9) == "DONOR-INACCESSIBLE"
    assert mod.adjudicate(10, 9, 10) == "DONOR-HISTORIES-DEGENERATE"      # 0.9 inclusive
    assert mod.adjudicate(10, 8, 10) == "DONOR-ACCESSIBLE"
    assert mod.adjudicate(12, 0, 12) == "DONOR-ACCESSIBLE"
    assert mod.adjudicate_size(10) == "DONOR-ACCESSIBLE" and mod.adjudicate_size(9) == "DONOR-INACCESSIBLE"


def test_library_rows_carry_the_atoms_schema(mod):
    keys = {"cur", "nxt", "level", "rule", "source"}
    row = {"cur": "Integral(x, x)", "nxt": "x**2/2", "level": 3, "rule": "donor", "source": "donor-A", "root_index": 0, "root_seed": 1, "sample_seed": 2, "wave": 0, "solved": True}
    assert keys <= set(row)
    import birth19m_curric as C
    import train_mathnative as TM
    enc, levels = C.encode_with_levels([row], TM.MathTokenizer())
    assert len(enc) == 1 and levels == [3]

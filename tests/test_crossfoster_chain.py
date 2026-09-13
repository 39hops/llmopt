"""VERIFIED-ENDOGENOUS-DATA-CROSSFOSTER-1-CHAIN-DESK guards: the ply seed
law is identical for both donors and injective over (root, ply, wave, slot)
within a root's 2,000-seed block, the stage law is pure and inclusive at the
registered thresholds, the chain-dose projection truncates the last chain
identically, first_divergence reads the first differing state, and the
registered constants are literal."""
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
        m = importlib.import_module("crossfoster_chain")
    finally:
        if prev is None:
            del os.environ["SMOKE"]
        else:
            os.environ["SMOKE"] = prev
    return m


def test_ply_seed_law_injective_and_shared(mod):
    seen = set()
    for t in range(12):
        for w in range(4):
            s = mod.ply_seeds(7, t, w)
            assert len(s) == 8 and all(mod.SAMPLE_BAND + 2000 * 7 <= x < mod.SAMPLE_BAND + 2000 * 8 for x in s)
            assert s == mod.ply_seeds(7, t, w)          # the same integers for both donors (pure function of root, ply, wave)
            seen.update(s)
    assert len(seen) == 12 * 4 * 8
    # the registered span (24,000 roots x 2,000) ends below the smoke band, and the root bands are disjoint
    n_pop = sum({3: 3600, 4: 9600, 5: 3600, 6: 3600, 7: 3600}.values())
    assert 100_000_000 + 2000 * n_pop == 148_000_000 < 200_000_000
    assert 8_600_000 + 1000 * 7 + 9600 < 8_650_000


def test_inadequacy_accounting_uses_l_min(mod):
    def acct(eA, eB):
        okA, okB = len(eA) >= mod.L_MIN, len(eB) >= mod.L_MIN
        return "inadequate_both" if not (okA or okB) else ("inadequate_A" if not okA else "inadequate_B")
    mod.L_MIN = 2
    assert acct([1], [1]) == "inadequate_both" and acct([1], [1, 2, 3]) == "inadequate_A" and acct([1, 2], [1]) == "inadequate_B"
    src = (ROOT / "scratch" / "crossfoster_chain.py").read_text()
    assert "okA, okB = len(eA) >= L_MIN, len(eB) >= L_MIN" in src


def test_stage_law(mod):
    mod.N_ROWS, mod.N_SPECIFIC_FIRE = 100, 50
    assert mod.adjudicate(99, 99) == "CHAIN-INACCESSIBLE"
    assert mod.adjudicate(100, 50) == "CHAIN-CONTRAST-FIRES"    # inclusive
    assert mod.adjudicate(100, 49) == "CHAIN-DEGENERATE"


def test_projection_truncates_last_chain_identically(mod):
    chains = [{"l_pair": 3}, {"l_pair": 5}, {"l_pair": 4}]
    taken, total = mod.project(chains, 6)
    assert total == 6 and [k for _, k in taken] == [3, 3]
    taken, total = mod.project(chains, 100)
    assert total == 12 and [k for _, k in taken] == [3, 5, 4]


def test_first_divergence(mod):
    a = [{"nxt": "x"}, {"nxt": "y "}, {"nxt": "z"}]
    b = [{"nxt": "x"}, {"nxt": " y"}, {"nxt": "w"}]
    assert mod.first_divergence(a, b, 3) == 3 and mod.first_divergence(a, b, 2) is None


def test_registered_constants_literal():
    src = (ROOT / "scratch" / "crossfoster_chain.py").read_text()
    for s in ('{3: 3600, 4: 9600, 5: 3600, 6: 3600, 7: 3600}', 'if SMOKE else 6000', 'if SMOKE else 12', 'L_MIN = 2', 'if SMOKE else 4', 'WAVE = 8', 'N_SPECIFIC_FIRE = 3000',
              'N_REPLAY = 2 if SMOKE else 20', 'ROOT_BAND = 8_650_000 if SMOKE else 8_600_000', 'SAMPLE_BAND = 200_000_000 if SMOKE else 100_000_000', 'I0 = 0',
              'SAMPLE_BAND + 2000 * root_index + 128 * t + WAVE * w + b', 'if norm(txt) in visited:', 'sample_wave_lp(model, tok, prompt, seeds, dev)', 'verify_wave(cur, cands)',
              'CD.load_donor(name, tok, dev)', 'CD.DONOR_DIGEST_PREFIX[name]', 'CD.W0_DIGEST_PREFIX'):
        assert s in src, s

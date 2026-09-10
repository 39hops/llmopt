"""Source-invariant guard for FROZEN-BACKBONE-1: the driver is the
results-cited CREDIT-ANCHOR-FRONTIER-1 driver plus the FB phase law; the
stock loop lines and the caf branches survive verbatim; the FB law is
literal (seeds 24, 25, 26; MODE bp | zero; K_BP 4; LR 3e-4); the gate
script's replication law is literal (delta >= -7 on all three pairs)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAF = ROOT / "scratch" / "birth19m_caf.py"
FB = ROOT / "scratch" / "birth19m_fb.py"
TRAJ = ROOT / "scratch" / "birth19m_atoms_traj.py"
GATE = ROOT / "scratch" / "fb_gate.py"


def _code_lines(path):
    src = path.read_text()
    src = re.sub(r'^""".*?"""', "", src, count=1, flags=re.S)
    return [l.strip() for l in src.splitlines() if l.strip() and not l.strip().startswith("#")]


def _loop_body(path):
    src = path.read_text()
    m = re.search(r"for a, b in stream:\n(.*?)opt\.zero_grad\(\)\n", src, flags=re.S)
    assert m, path
    return [l.strip() for l in m.group(1).splitlines() if l.strip()]


def test_stock_loop_lines_survive():
    fb = set(_code_lines(FB))
    assert not [l for l in _loop_body(TRAJ) if l not in fb]


def test_caf_loop_survives_verbatim():
    assert _loop_body(CAF) == _loop_body(FB)


def test_fb_law_literal():
    src = FB.read_text()
    for s in ("FB_SEEDS = (24, 25, 26)", "FB_K = 4", 'assert MODE in ("bp", "zero")', 'assert PEAK_LR == 3e-4, "FROZEN-BACKBONE-1 runs the stock LR 3e-4"',
              "assert K_BP == FB_K", 'assert QUAL + DISCOVERY + SMOKE + FB == 1', 'ROOT = Path("checkpoints/frozenbb1_smoke" if SMOKE else "checkpoints/frozenbb1")',
              "frozen_names, trainable_names = freeze_lower(model, K_BP)"):
        assert s in src, s
    assert "gate_eval" not in src


def test_replication_law_literal():
    src = GATE.read_text()
    assert "LAW = -7" in src and "SEEDS = (24, 25, 26)" in src
    assert "FLOOR = 24" in src
    assert 'rep = all(d >= LAW for d in deltas)' in src
    assert 'verdict, rep = "NOT-RESOLVABLE-CONTROL", None' in src
    assert "mean_delta_descriptive" in src

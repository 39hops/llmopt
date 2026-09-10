"""SG-PREDICTOR-AUDIT-0 reductions (scratch/sg_target_audit.py
reduce_vectors / constants): exact-zero and nonfinite counting, RMS and
Frobenius norm, per-token-norm quantiles and dynamic range, and the
registered normalization-constant rule (geometric mean over FROZEN states
for blocks 4..7, over FULL states for 0..7). Pure tensor checks; no
checkpoint is read."""
import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    for p in (ROOT, ROOT / "scripts", ROOT / "scratch"):
        sys.path.insert(0, str(p))
    spec = importlib.util.spec_from_file_location("sg_target_audit", ROOT / "scratch" / "sg_target_audit.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.CHUNK == 32 and m.SEEDS == (24, 25, 26) and m.STEPS == ("step_00463", "final")
    return m


def test_reductions_on_known_matrix(mod):
    import torch
    v = torch.zeros(4, 3, dtype=torch.float64)
    v[0] = torch.tensor([3.0, 4.0, 0.0])      # norm 5, one exact zero
    v[1] = torch.tensor([0.0, 0.0, 0.0])      # zero token
    v[2] = torch.tensor([1.0, float("nan"), 2.0])
    v[3] = torch.tensor([0.0, 0.0, float("inf")])
    r = mod.reduce_vectors(v)
    assert r["n_tokens"] == 4 and r["dim"] == 3
    assert r["nonfinite_elements"] == 2
    assert r["zero_elements"] == 6            # row0: 1, row1: 3, row3: 2
    assert r["zero_token_norms"] == 2         # row1 and row3 (its inf is masked to 0)
    assert math.isclose(r["fro"], math.sqrt(25 + 5))
    assert math.isclose(r["rms"], math.sqrt(30 / 12))
    assert r["token_norm_max"] == 5.0 and r["token_norm_min"] == 0.0


def test_dynamic_range_and_quantiles(mod):
    import torch
    n = 1000
    scale = torch.logspace(-3, 1, n, dtype=torch.float64)   # token norms from 1e-3 to 10
    v = torch.zeros(n, 2, dtype=torch.float64)
    v[:, 0] = scale
    r = mod.reduce_vectors(v)
    assert set(r["token_norm_q"]) == {"q01", "q10", "q50", "q90", "q99"}
    assert r["token_norm_q"]["q01"] < r["token_norm_q"]["q50"] < r["token_norm_q"]["q99"]
    assert 3.5 < r["dynamic_range_log10"] < 4.0
    assert r["zero_token_norms"] == 0 and r["nonfinite_elements"] == 0


def test_dynamic_range_none_when_q01_zero(mod):
    import torch
    v = torch.zeros(10, 2, dtype=torch.float64)
    v[-1, 0] = 1.0
    assert mod.reduce_vectors(v)["dynamic_range_log10"] is None


def test_constants_rule(mod):
    def st(arm, rms_by_block):
        return {"arm": arm, "blocks": {str(l): {"delta": {"rms": r}} for l, r in rms_by_block.items()}}
    states = [st("FROZEN", {4: 1.0, 5: 2.0, 6: 4.0, 7: 8.0}), st("FROZEN", {4: 4.0, 5: 8.0, 6: 16.0, 7: 32.0}),
              st("FULL", {l: 10.0 for l in range(8)}), st("FULL", {l: 1000.0 for l in range(8)})]
    c = mod.constants(states)
    assert c["arena_frozen_4_7"] == {"4": pytest.approx(2.0), "5": pytest.approx(4.0), "6": pytest.approx(8.0), "7": pytest.approx(16.0)}
    assert all(v == pytest.approx(100.0) for v in c["full_stack_0_7"].values()) and len(c["full_stack_0_7"]) == 8
    assert mod.arena_blocks("FULL") == list(range(8)) and mod.arena_blocks("FROZEN") == [4, 5, 6, 7]

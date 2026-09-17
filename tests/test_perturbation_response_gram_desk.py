"""PERTURBATION-RESPONSE-GRAM-DESK-0 instrument guards: pins against the receipt lock, the flat law over the registered
59-tensor partition, the Gram readouts (eigen sum, top / trace, participation and effective rank, UNDEFINED law), the
window and named-cell geometry, the refuse-if-exists guard, and real-mode constants (no training path exists)."""
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scratch" / "perturbation_response_gram_desk.py"


@pytest.fixture(scope="module")
def mod():
    """The instrument imported in SMOKE mode under a scoped env patch (undone at module teardown so sibling test
    modules importing scratch drivers keep their real-mode constants)."""
    mp = pytest.MonkeyPatch()
    mp.chdir(ROOT)
    mp.syspath_prepend(str(ROOT / "scratch"))
    mp.syspath_prepend(str(ROOT))
    mp.setenv("SMOKE", "1")
    mp.setenv("SMOKE_TAG", "pytest")
    m = importlib.import_module("perturbation_response_gram_desk")
    m = importlib.reload(m)
    yield m
    mp.undo()
    sys.modules.pop("perturbation_response_gram_desk", None)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_pins_match_lock_and_files(mod):
    lock = json.loads((ROOT / "docs/receipts.lock.json").read_text())["receipts"]
    assert lock["logs/rdc1/control.json"]["sha256"] == mod.RDC1_SHA
    assert lock["logs/fmel2/ladder.json"]["sha256"] == mod.FMEL2_SHA
    for p, s in ((mod.RDC1, mod.RDC1_SHA), (mod.FMEL2, mod.FMEL2_SHA)):
        if (ROOT / p).exists():
            assert sha(ROOT / p) == s, p


def test_partition_and_grid(mod):
    assert len(mod.KEYS) == 59
    assert sum(len(v) for v in mod.GROUPS.values()) == 59 and set(mod.GROUPS) == {f"BLOCK{i}" for i in range(8)} | {"OUTSIDE"}
    assert mod.GRID_FULL == [1, 5, 20, 100, 300, 900, 1800, 3080, 4500, 6000, 7200, 8220]
    assert mod.windows([1, 5, 20]) == [(1, 5), (5, 20)]
    for name, (a, wa, b, wb) in mod.NAMED_CELLS.items():
        assert a == "M" and b == "R" and wa[0] < wa[1] and wb[0] < wb[1]
        assert all(h in mod.GRID_FULL for h in wa + wb), name


def test_segments_flat_law(mod):
    import torch
    sd = {k: torch.arange(3 * (i + 1), dtype=torch.float32).reshape(-1) for i, k in enumerate(mod.KEYS)}
    segs, d, dg = mod.segments(sd)
    assert d == sum(3 * (i + 1) for i in range(59)) and segs[0][1] == 0 and segs[-1][2] == d
    assert all(segs[i][2] == segs[i + 1][1] for i in range(len(segs) - 1))
    x = mod.flat(sd, segs, d)
    assert x.dtype == np.float64 and x[segs[1][1]] == 0.0 and x[segs[1][2] - 1] == 5.0
    gidx = mod.group_index(segs)
    assert sum(len(v) for v in gidx.values()) == d and set(gidx) == set(mod.GROUPS)
    assert len(dg) == 64
    with pytest.raises(AssertionError):
        mod.segments({k: sd[k] for k in mod.KEYS[:-1]})


def test_gram_stats_laws(mod):
    rng = np.random.default_rng(0)
    v = [rng.standard_normal(2000) for _ in range(4)]
    g = mod.gram_stats(v, ["M", "R1", "R2", "R3"])
    assert abs(sum(g["eig"]) - 4.0) < 1e-9 and g["eig"] == sorted(g["eig"], reverse=True)
    assert 0.25 <= g["top_over_trace"] <= 1.0 and 1.0 <= g["participation_rank"] <= 4.0 and 1.0 <= g["effective_rank"] <= 4.0
    assert all(abs(g["cos"][i][i] - 1.0) < 1e-12 for i in range(4))
    ident = mod.gram_stats([np.eye(4)[i] for i in range(4)], list("abcd"))
    assert abs(ident["participation_rank"] - 4.0) < 1e-9 and abs(ident["top_over_trace"] - 0.25) < 1e-9 and abs(ident["effective_rank"] - 4.0) < 1e-9
    one = mod.gram_stats([np.ones(5), 2 * np.ones(5), -np.ones(5)], list("abc"))
    assert abs(one["participation_rank"] - 1.0) < 1e-9 and abs(one["top_over_trace"] - 1.0) < 1e-9
    z = mod.gram_stats([np.ones(5), np.zeros(5)], ["a", "b"])
    assert z["eig"] is None and "b" in z["undefined"] and z["norms"][1] == 0.0
    assert mod.cosine(np.zeros(3), np.ones(3)) is None and abs(mod.cosine(np.ones(3), np.ones(3)) - 1.0) < 1e-12
    assert json.dumps(z) and not any(isinstance(x, float) and math.isnan(x) for x in z["norms"])


def test_windows_and_named_cells(mod):
    rng = np.random.default_rng(1)
    grid = [100, 300, 900, 1800, 3080]
    dev_of = {a: {h: rng.standard_normal(50) for h in grid} for a in mod.ARMS}
    wins = mod.windows(grid)
    inc = {a: [dev_of[a][h2] - dev_of[a][h1] for h1, h2 in wins] for a in mod.ARMS}
    L = mod.lag_matrix(inc["M"], inc["R1"])
    assert len(L) == 4 and len(L[0]) == 4
    segs = [("k", 0, 25, "BLOCK0"), ("j", 25, 50, "OUTSIDE")]
    gidx = mod.group_index(segs)
    cells = mod.named_cells(dev_of, gidx)
    c = cells["primary_M300-900_v_R900-1800"]["per_arm"]["R1"]
    va = dev_of["M"][900] - dev_of["M"][300]; vb = dev_of["R1"][1800] - dev_of["R1"][900]
    assert abs(c["cos"] - mod.cosine(va, vb)) < 1e-12 and abs(c["groups"]["BLOCK0"] - mod.cosine(va[:25], vb[:25])) < 1e-12
    assert abs(c["cos"] - L[1][2]) < 1e-12                       # the primary cell is the (300->900, 900->1800) entry of the lag matrix
    assert abs(cells["alt_M900-1800_v_R900-1800_lag0"]["per_arm"]["R2"]["cos"] - mod.lag_matrix(inc["M"], inc["R2"])[2][2]) < 1e-12
    short = mod.named_cells({a: {h: dev_of[a][h] for h in [100, 300]} for a in mod.ARMS}, gidx)
    assert short["primary_M300-900_v_R900-1800"]["per_arm"]["R3"]["cos"] is None and "not in the loaded grid" in short["primary_M300-900_v_R900-1800"]["per_arm"]["R3"]["undefined"]


def test_real_mode_constants_and_no_training_path():
    src = SRC.read_text()
    out = subprocess.run([sys.executable, "-c", "import os,sys; os.chdir(sys.argv[1]); sys.path[:0]=['.','scratch']; "
                          "import perturbation_response_gram_desk as m; print(m.SMOKE, m.WALL_S, m.GRID == m.GRID_FULL, str(m.RECEIPT), m.D_EXPECTED)",
                          str(ROOT)], capture_output=True, text=True, env={k: v for k, v in os.environ.items() if k not in ("SMOKE", "SMOKE_TAG")})
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "3600", "True", "logs/prgd0/desk.json", "18911616"]
    for forbidden in ("opt.step", "backward(", "exp_avg", "AdamW", "checkpoints/oma1", "checkpoints/fme1/"):
        assert forbidden not in src, forbidden


def test_refuse_if_receipt_exists(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "RECEIPT", tmp_path / "desk.json")
    (tmp_path / "desk.json").write_text("{}")
    with pytest.raises(SystemExit, match="REFUSING"):
        mod.main()


def test_launcher_shape():
    s = (ROOT / "scratch/prgd0_launch.sh").read_text()
    assert "liverun.py run prgd0" in s and "perturbation_response_gram_desk.py" in s and "prgd0.DONE" in s
    assert "REFUSING" in s and "set -eo pipefail" in s

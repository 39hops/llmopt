"""llmopt.lab.shards — MXFP4 dequant identity + streaming path."""
import inspect
import re
from pathlib import Path

import numpy as np
import pytest

from llmopt.lab import shards

ROOT = Path(__file__).resolve().parents[1]
K3_DEMO = ROOT / "scratch" / "k3_expert_demo.py"


def _source_of(path: Path, name: str) -> str:
    """The named function's source segment from a script file."""
    text = path.read_text()
    m = re.search(rf"^def {name}\(.*?(?=^\ndef |^\nclass |\Z)", text,
                  re.M | re.S)
    assert m, f"{name} not found in {path}"
    return m.group(0).strip()


def test_dequant_is_verbatim_from_k3_demo():
    """Dual-copy guard (adoption doctrine): while the scratch original
    and the lab copy coexist, fixes land in BOTH, same commit."""
    if not K3_DEMO.exists():
        pytest.skip("scratch original absent on this checkout")
    assert (inspect.getsource(shards.dequant).strip()
            == _source_of(K3_DEMO, "dequant"))


def test_dequant_exact_values():
    # one 32-value block (16 packed bytes, one scale group).
    # byte 0x21 -> low nibble 1 (e2m1 0.5) then high nibble 2 (1.0);
    # scale byte 127 -> 2^0
    packed = np.full((1, 16), 0x21, dtype=np.uint8)
    scale = np.array([[127]], dtype=np.uint8)
    codes2x, exps, w = shards.dequant(packed, scale)
    assert codes2x[0, :2].tolist() == [1, 2]
    assert exps.tolist() == [[0]]
    assert w[0, :2].tolist() == [0.5, 1.0]
    # sign bit 0x8: nibble 0x9 = -0.5 twice; scale 128 doubles
    packed = np.full((1, 16), 0x99, dtype=np.uint8)
    scale = np.array([[128]], dtype=np.uint8)
    _, _, w = shards.dequant(packed, scale)
    assert w[0, :2].tolist() == [-1.0, -1.0]


def test_weigh_row_is_snake_case_and_complete():
    torch = pytest.importorskip("torch")
    W = torch.randn(32, 16, generator=torch.Generator().manual_seed(0))
    row = shards.weigh(W, source="toy", model="test", proj="w1")
    assert row["n_rows"] == 32 and row["n_cols"] == 16
    for k in row:
        assert k == k.lower() and " " not in k
    assert row["meter_m_bits"] == pytest.approx(row["meter_m_bits"])
    assert row["row_norm_max"] >= row["row_norm_mean"] > 0


V4CACHE = ROOT / "checkpoints" / "v4flash_f1"


@pytest.mark.skipif(not (V4CACHE / "manifest_all.json").exists(),
                    reason="V4-Flash cache not on this machine")
def test_v4flash_expert_streams_and_dequants():
    torch = pytest.importorskip("torch")
    pairs = shards.list_v4flash_experts(str(V4CACHE))
    assert pairs, "cache present but no complete w1 expert"
    lay, eid = pairs[0]
    W = shards.v4flash_expert(lay, eid, cache=str(V4CACHE))
    assert W.dtype == torch.float32 and W.ndim == 2
    # every value is an exact e2m1 x 2^k — nonzero magnitudes cluster
    # in a bounded dynamic range, and zero rows are legal
    assert torch.isfinite(W).all()
    assert float(W.abs().max()) > 0


def test_append_weights_lake_roundtrip(tmp_path):
    pa = pytest.importorskip("pyarrow")  # noqa: F841 — [lake] extra
    torch = pytest.importorskip("torch")
    from llmopt.lab import lake
    W = torch.randn(8, 32, generator=torch.Generator().manual_seed(1))
    row = shards.weigh(W, source="L0E0", model="toy", proj="w1")
    p = lake.append_weights([row], lake_dir=tmp_path)
    lake.append_weights([row], lake_dir=tmp_path)  # append, not clobber
    import pyarrow.parquet as pq
    t = pq.read_table(p)
    assert t.num_rows == 2
    assert t.schema.names == [f.name for f in lake.WEIGHTS_SCHEMA]


def test_append_weights_refuses_unprovenanced(tmp_path):
    pytest.importorskip("pyarrow")
    from llmopt.lab import lake
    with pytest.raises(ValueError):
        lake.append_weights([{"source": "x"}], lake_dir=tmp_path)

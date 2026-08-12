"""Guards for llmopt/vendor/axiom/ (vendored 2026-08-11, axiom sha
b785601).

(a) Source identity: vendored body after the 3-line provenance header
    is byte-equal to the upstream file (the dual-copy guard from the
    2026-08-06 scratch doctrine — it is doing its job when annoying).
    Skips when the axiom checkout is absent.
(b) AXNN roundtrip: parse the real fixture data/scorer_s2_dist.axnn
    against nn_exact_ref.write_axnn's container spec (magic AXNN, u32
    version, cfg json, [name, ndim, dims-u64, f32 payload] tensors).
    Skips when the 33MB fixture is absent (file-handoff convention:
    big artifacts stay untracked)."""
import json
import os
import struct

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(REPO, "llmopt", "vendor", "axiom")
AXIOM = os.environ.get("AXIOM_CHECKOUT", "")
HEADER_LINES = 3

PAIRS = [
    ("nn_exact_ref.py", "scripts/nn_exact_ref.py"),
    ("divergence.py", "tools/exact_anchor/divergence.py"),
    ("classify_sample.py", "tools/exact_anchor/classify_sample.py"),
]


@pytest.mark.parametrize("vend,up", PAIRS)
def test_source_identity(vend, up):
    from conftest import artifact_or_skip
    artifact_or_skip(bool(AXIOM) and os.path.isdir(AXIOM),
                     "axiom checkout absent (set AXIOM_CHECKOUT)")
    upstream = os.path.join(AXIOM, up)
    artifact_or_skip(os.path.exists(upstream), f"upstream absent: {up}")
    with open(os.path.join(VENDOR, vend), "rb") as f:
        body = b"".join(f.readlines()[HEADER_LINES:])
    with open(upstream, "rb") as f:
        assert body == f.read(), f"{vend} drifted from {up}"


def _parse_axnn(path):
    """Reader for nn_exact_ref.write_axnn's container format."""
    tensors = []
    with open(path, "rb") as f:
        magic = f.read(4)
        (version,) = struct.unpack("<I", f.read(4))
        (clen,) = struct.unpack("<I", f.read(4))
        cfg = json.loads(f.read(clen))
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            (nlen,) = struct.unpack("<I", hdr)
            name = f.read(nlen).decode()
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = struct.unpack(f"<{ndim}Q", f.read(8 * ndim))
            numel = 1
            for d in shape:
                numel *= d
            # write_axnn's tensor() casts every payload <f4 —
            # including the f64-built tables.
            width = 4
            payload = f.read(width * numel)
            assert len(payload) == width * numel, f"truncated {name}"
            tensors.append((name, shape))
    return magic, version, cfg, tensors


def test_axnn_roundtrip_real_fixture():
    path = os.path.join(REPO, "data", "scorer_s2_dist.axnn")
    from conftest import artifact_or_skip
    artifact_or_skip(os.path.exists(path),
                     "data/scorer_s2_dist.axnn absent (untracked 33MB)")
    magic, version, cfg, tensors = _parse_axnn(path)
    assert magic == b"AXNN"
    assert version == 1
    assert isinstance(cfg, dict) and "d_model" in cfg
    assert len(tensors) >= 1
    assert all(name for name, _ in tensors)


def test_write_axnn_roundtrip_synthetic(tmp_path):
    """Writer -> our reader on a tiny synthetic container."""
    np = pytest.importorskip("numpy")
    from llmopt.vendor.axiom import nn_exact_ref
    cfg = {"d_model": 4, "n_layers": 1}
    w = {"tok_emb.weight": np.arange(8, dtype=np.float32).reshape(2, 4)}
    tables = {"fx.exp.table": [1, 2, 3]}
    p = tmp_path / "tiny.axnn"
    nn_exact_ref.write_axnn(str(p), cfg, w, tables)
    magic, version, got_cfg, tensors = _parse_axnn(str(p))
    assert magic == b"AXNN" and version == 1 and got_cfg == cfg
    assert tensors == [("tok_emb.weight", (2, 4)),
                       ("fx.exp.table", (3,))]

"""Tests for llmopt.lab.merge — tiny synthetic state dicts, CPU only.

Skips cleanly without torch (optional-dep convention). No gating runs
here: gate_cmd is string-only by device doctrine.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from llmopt.lab import merge  # noqa: E402


def _sd(seed: int, ffn: int = 8, d: int = 4) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "blocks.0.gate.weight": torch.randn(ffn, d, generator=g),
        "blocks.0.up.weight": torch.randn(ffn, d, generator=g),
        "blocks.0.down.weight": torch.randn(d, ffn, generator=g),
        "emb.weight": torch.randn(6, d, generator=g),
    }


def _save(tmp_path, name, sd):
    p = tmp_path / name
    torch.save(sd, p)
    return str(p)


def test_average_refuses_without_shared_lineage(tmp_path):
    a = _save(tmp_path, "a.pt", _sd(1))
    b = _save(tmp_path, "b.pt", _sd(2))
    with pytest.raises(ValueError, match="shared_lineage"):
        merge.average(a, b, str(tmp_path / "o.pt"))


def test_average_shape_mismatch(tmp_path):
    a = _save(tmp_path, "a.pt", _sd(1, ffn=8))
    b = _save(tmp_path, "b.pt", _sd(2, ffn=12))
    with pytest.raises(ValueError, match="shape mismatch"):
        merge.average(a, b, str(tmp_path / "o.pt"), shared_lineage=True)


def test_average_math_and_sidecar(tmp_path):
    sa, sb = _sd(1), _sd(2)
    a = _save(tmp_path, "a.pt", sa)
    b = _save(tmp_path, "b.pt", sb)
    out = str(tmp_path / "avg.pt")
    row = merge.average(a, b, out, alpha=0.25, shared_lineage=True)
    got = torch.load(out, weights_only=True)
    for k in sa:
        assert torch.allclose(got[k], 0.75 * sa[k] + 0.25 * sb[k])
    side = json.loads((tmp_path / "avg.pt.merge.json").read_text())
    assert side["op"] == "average" and side["alpha"] == 0.25
    assert len(side["inputs"]) == 2
    assert all(len(i["sha256"]) == 64 for i in side["inputs"])
    assert row["op"] == "average" and "git_sha" in row and "ts" in row


def test_average_never_overwrites_input(tmp_path):
    a = _save(tmp_path, "a.pt", _sd(1))
    b = _save(tmp_path, "b.pt", _sd(2))
    with pytest.raises(ValueError, match="overwrite"):
        merge.average(a, b, a, shared_lineage=True)


def test_task_vector_exact(tmp_path):
    s0, sa, sb = _sd(3), _sd(4), _sd(5)
    base = _save(tmp_path, "base.pt", s0)
    a = _save(tmp_path, "a.pt", sa)
    b = _save(tmp_path, "b.pt", sb)
    out = str(tmp_path / "tv.pt")
    alpha = 0.5
    merge.task_vector(base, a, b, out, alpha=alpha)
    got = torch.load(out, weights_only=True)
    for k in s0:
        want = s0[k] + alpha * ((sa[k] - s0[k]) + (sb[k] - s0[k]))
        assert torch.equal(got[k], want), k


def test_shell_graft_refuses_ternary(tmp_path):
    sd = _sd(6)
    w = torch.randint(-1, 2, (8, 4), generator=torch.Generator()
                      .manual_seed(0)).float() * 0.037
    sd["blocks.0.gate.weight"] = w
    small = _save(tmp_path, "tern.pt", sd)
    with pytest.raises(ValueError, match="ternary"):
        merge.shell_graft(small, {"ffn": 16}, str(tmp_path / "o.pt"))


def test_shell_graft_function_preserving(tmp_path):
    sd = _sd(7, ffn=8, d=4)
    small = _save(tmp_path, "small.pt", sd)
    out = str(tmp_path / "big.pt")
    row = merge.shell_graft(small, {"ffn": 16}, out,
                            arch={"d": 4, "layers": 1, "ffn": 16,
                                  "heads": 2})
    big = torch.load(out, weights_only=True)
    assert big["blocks.0.gate.weight"].shape == (16, 4)
    assert row["grow"] == 8

    def hidden(s, x):
        g = x @ s["blocks.0.gate.weight"].T
        u = x @ s["blocks.0.up.weight"].T
        return torch.nn.functional.silu(g) * u

    x = torch.randn(3, 4, generator=torch.Generator().manual_seed(9))
    h0 = hidden(sd, x)
    h1 = hidden(big, x)
    # old shells untouched: the copied weight rows are byte-equal...
    for name in ("gate", "up"):
        key = f"blocks.0.{name}.weight"
        assert torch.equal(big[key][:8], sd[key])
    # ...and the hidden prefix matches to reduction tolerance. Not
    # bit-equal on every CPU: x @ W.T at N=16 vs N=8 can pick a
    # different kernel/pairing for the SAME summands (an x86 runner
    # broke the exact compare that held on Apple silicon) — the same
    # split-reduction class as the down-proj note below.
    assert torch.allclose(h0, h1[:, :8], rtol=1e-6, atol=1e-7)
    # grafted down-columns are exact zeros => their contribution is
    # exactly 0.0 (split-reduction check: a single fused matmul over
    # K=16 vs K=8 may re-pair the SAME summands, which is reduction
    # noise, not graft error — the fp16 near-tie non-bug's cousin)
    y0 = h0 @ sd["blocks.0.down.weight"].T
    dn = big["blocks.0.down.weight"]
    y1 = h1[:, :8] @ dn[:, :8].T + h1[:, 8:] @ dn[:, 8:].T
    assert torch.equal(dn[:, 8:], torch.zeros(4, 8))
    assert torch.equal(y0, y1)


def test_gate_cmd_device_lineages(tmp_path):
    row = {"out": "checkpoints/x.pt",
           "arch": {"d": 128, "layers": 4, "ffn": 512, "heads": 4}}
    mac = merge.gate_cmd(row, "mps")
    cuda = merge.gate_cmd(row, "cuda")
    assert "scratch/gate_ckpt.py" in mac
    assert "scratch/gate_ckpt_cuda.py" in cuda
    assert mac.endswith("checkpoints/x.pt 128 4 512 4 x")
    with pytest.raises(ValueError, match="arch"):
        merge.gate_cmd({"out": "checkpoints/x.pt"}, "mps")

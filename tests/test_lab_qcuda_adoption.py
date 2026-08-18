"""Adoption guard for llmopt.lab.qcuda.

The CUDA decode primitives were adopted verbatim from the frozen
receipt-cited driver scratch/qwen_cuda_rung4.py (2026-08-17 CUDA
ladder). While the two files coexist, every adopted symbol must stay
CHARACTER-IDENTICAL between them — a fix lands in both in the same
commit or in neither.

The identity comparison is PURE TEXT (parsed with ast, never
imported), so it runs everywhere, including hosts with no triton. Only
the runtime checks skip when triton is missing.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch" / "qwen_cuda_rung4.py"
LAB = ROOT / "llmopt" / "lab" / "qcuda.py"

ADOPTED = ["SUB127", "w4_decode_kernel", "s16_gemv_kernel",
           "w4_gemv_kernel", "W4Gpu", "S16Gpu", "FusedW4Linear"]


def _top_level_sources(path: Path) -> dict[str, str]:
    """name -> source text, for top-level defs/classes/assignments.

    Mirrors inspect.getsource's span (decorators included) without
    importing the module.
    """
    text = path.read_text()
    lines = text.splitlines()
    out: dict[str, str] = {}

    def span(node, first):
        return "\n".join(lines[first - 1:node.end_lineno])

    for node in ast.parse(text).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            first = min([node.lineno]
                        + [d.lineno for d in node.decorator_list])
            out[node.name] = span(node, first)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = span(node, node.lineno)
    return out


@pytest.fixture(scope="module")
def pair():
    return _top_level_sources(SCRATCH), _top_level_sources(LAB)


@pytest.mark.parametrize("name", ADOPTED)
def test_adopted_source_identical(pair, name):
    scr, lab = pair
    assert name in scr, f"{name} missing from {SCRATCH}"
    assert name in lab, f"{name} missing from {LAB}"
    assert lab[name] == scr[name], (
        f"{name} drifted between {SCRATCH} and {LAB}; fix BOTH copies "
        "in the same commit")


def test_qcuda_imports_without_triton():
    """Import must never require triton/CUDA."""
    import llmopt.lab.qcuda as q
    assert isinstance(q.HAVE_TRITON, bool)
    for name in ADOPTED:
        assert hasattr(q, name)


def test_kernel_launch_errors_at_use_time_when_triton_absent():
    import llmopt.lab.qcuda as q
    if q.HAVE_TRITON:
        pytest.skip("triton present; stand-in path not exercised")
    with pytest.raises(RuntimeError, match="needs triton"):
        q.require_triton()
    with pytest.raises(RuntimeError, match="needs triton"):
        q.w4_decode_kernel[(1,)](0, 0, 0, 0, 1, BLK=1024)
    with pytest.raises(RuntimeError, match="needs triton"):
        q.triton.cdiv(4, 2)


def test_runtime_parity_smoke():
    """Decode bit-exactness v qcodec — CUDA machines only."""
    pytest.importorskip("triton")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    import numpy as np
    from llmopt.lab.qcodec import BLOCK, dec_w4
    from llmopt.lab.qcuda import W4Gpu
    rng = np.random.default_rng(3)
    R, C = 8, 256
    nb = R * C // BLOCK
    buf = (rng.integers(120, 132, nb, dtype=np.uint8).tobytes()
           + (rng.standard_normal((256, 4)) * 0.3)
           .astype(np.float16).tobytes()
           + rng.integers(0, 256, R * C // 4, dtype=np.uint8).tobytes())
    ref = dec_w4(buf, [R, C])
    got = W4Gpu(buf, [R, C]).decode_rows(0, R).cpu().numpy()
    assert np.array_equal(got, ref, equal_nan=True)

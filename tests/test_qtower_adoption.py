"""llmopt.lab.qtower is the verbatim adoption of the qualified tower
construction in scratch/qwen_metal_qual.py (the frozen record of the
census and the serving receipts). The two bodies must stay identical until
a registered re-run migrates the drivers (scratch doctrine 2026-08-06)."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _scratch():
    spec = importlib.util.spec_from_file_location("qwen_metal_qual_frozen", ROOT / "scratch" / "qwen_metal_qual.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen_metal_qual_frozen"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_tower_builder_is_the_frozen_body():
    from llmopt.lab import qtower
    frozen = _scratch()
    assert inspect.getsource(qtower.build_metal_model) == inspect.getsource(frozen.build_metal_model)
    assert inspect.getsource(qtower.fsha) == inspect.getsource(frozen.fsha)
    assert qtower.NORM_SHIFT_SUFFIXES == frozen.NORM_SHIFT_SUFFIXES


def test_tower_builder_keeps_the_mlx_lm_source_guard():
    from llmopt.lab import qtower
    src = inspect.getsource(qtower.build_metal_model)
    assert "REFUSING: mlx-lm GatedDeltaNet source changed" in src
    assert "inv_scale * (q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6))" in src

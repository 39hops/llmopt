"""llmopt.kernels — hand-written Metal and Triton kernels, with honest benchmarks.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.kernels.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "flash_prefill": "metal",
    "flash_prefill_v2": "metal",
    "int4_gemv": "metal",
    "patch_swiglu": "mlx_integration",
    "quantize_pack_int4": "metal",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}") from None
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)


def __dir__() -> list[str]:
    return __all__


if TYPE_CHECKING:  # let type checkers see through the lazy layer
    from llmopt.kernels.metal import (
        flash_prefill,
        flash_prefill_v2,
        int4_gemv,
        quantize_pack_int4,
    )
    from llmopt.kernels.mlx_integration import (patch_swiglu)

"""llmopt.backends — decode backends behind one interface.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.backends.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "DecodeBackend": "base",
    "MLXBackend": "mlx_backend",
    "TorchStaticBackend": "torch_static",
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
    from llmopt.backends.base import (DecodeBackend)
    from llmopt.backends.mlx_backend import (MLXBackend)
    from llmopt.backends.torch_static import (TorchStaticBackend)

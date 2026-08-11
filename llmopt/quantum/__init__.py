"""llmopt.quantum — model-Hamiltonian ground-state instruments.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.quantum.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "ansatz_state": "ground",
    "build_tfim": "ground",
    "energy": "ground",
    "exact_ground": "ground",
    "hva_state": "ground",
    "param_shift_grad": "ground",
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
    from llmopt.quantum.ground import (
        ansatz_state,
        build_tfim,
        energy,
        exact_ground,
        hva_state,
        param_shift_grad,
    )

"""llmopt.internals — activation and attention diagnostics.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.internals.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "activation_stats": "activations",
    "attention_stats": "attention_stats",
    "layer_cka_matrix": "cka",
    "linear_cka": "cka",
    "logit_lens": "logit_lens",
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
    from llmopt.internals.activations import (activation_stats)
    from llmopt.internals.attention_stats import (attention_stats)
    from llmopt.internals.cka import (layer_cka_matrix, linear_cka)
    from llmopt.internals.logit_lens import (logit_lens)

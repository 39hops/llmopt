"""llmopt.distill — logit and sequence-level knowledge distillation.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.distill.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "distill_logits": "logit_kd",
    "generalized_jsd": "sequence_kd",
    "gkd": "sequence_kd",
    "kd_loss": "logit_kd",
    "sequence_kd": "sequence_kd",
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
    from llmopt.distill.logit_kd import (distill_logits, kd_loss)
    from llmopt.distill.sequence_kd import (generalized_jsd, gkd, sequence_kd)

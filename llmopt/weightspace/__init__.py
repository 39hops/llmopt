"""llmopt.weightspace — predicting what a network computes from its parameters.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.weightspace.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "Subject": "subjects",
    "WeightReader": "reader",
    "canonicalize": "subjects",
    "evaluate_reader": "reader",
    "make_dataset": "subjects",
    "make_subject": "subjects",
    "permute_hidden": "subjects",
    "train_reader": "reader",
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
    from llmopt.weightspace.reader import (WeightReader, evaluate_reader, train_reader)
    from llmopt.weightspace.subjects import (
        Subject,
        canonicalize,
        make_dataset,
        make_subject,
        permute_hidden,
    )

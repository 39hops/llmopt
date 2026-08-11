"""llmopt.context — long-context machinery and its evaluation.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.context.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "add_gist_tokens": "gist",
    "apply_rope_scaling": "rope_scaling",
    "compress_prompt": "compression",
    "compress_to_gist_kv": "gist",
    "effective_context_length": "ruler",
    "evaluate": "ruler",
    "make_suite": "ruler",
    "ntk_inv_freq": "rope_scaling",
    "pi_inv_freq": "rope_scaling",
    "yarn_inv_freq": "rope_scaling",
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
    from llmopt.context.compression import (compress_prompt)
    from llmopt.context.gist import (add_gist_tokens, compress_to_gist_kv)
    from llmopt.context.rope_scaling import (
        apply_rope_scaling,
        ntk_inv_freq,
        pi_inv_freq,
        yarn_inv_freq,
    )
    from llmopt.context.ruler import (effective_context_length, evaluate, make_suite)

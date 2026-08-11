"""llmopt.moe — mixture-of-experts routing anatomy.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.moe.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "ExpertCache": "offload",
    "MoELayer": "layer",
    "RouterStats": "router_stats",
    "SwiGLUExpert": "layer",
    "keep_fraction": "prune",
    "keep_sets": "prune",
    "load_balance_loss": "layer",
    "mask_router": "prune",
    "overlap": "router_stats",
    "router_z_loss": "layer",
    "stats_from_json": "prune",
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
    from llmopt.moe.layer import (
        MoELayer,
        SwiGLUExpert,
        load_balance_loss,
        router_z_loss,
    )
    from llmopt.moe.offload import (ExpertCache)
    from llmopt.moe.prune import (keep_fraction, keep_sets, mask_router, stats_from_json)
    from llmopt.moe.router_stats import (RouterStats, overlap)

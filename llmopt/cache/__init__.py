"""llmopt.cache — KV cache structures: radix reuse, paging, quantization, eviction.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.cache.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "BlockAllocator": "paged",
    "BlockTable": "paged",
    "PagedTensorStore": "paged",
    "QuantizedPagedStore": "kv_quant",
    "RadixCache": "radix",
    "apply_eviction": "eviction",
    "attention_sinks": "eviction",
    "h2o": "eviction",
    "payloads_to_cache": "prefix_reuse",
    "sliding_window": "eviction",
    "snapkv": "eviction",
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
    from llmopt.cache.eviction import (
        apply_eviction,
        attention_sinks,
        h2o,
        sliding_window,
        snapkv,
    )
    from llmopt.cache.kv_quant import (QuantizedPagedStore)
    from llmopt.cache.paged import (BlockAllocator, BlockTable, PagedTensorStore)
    from llmopt.cache.prefix_reuse import (payloads_to_cache)
    from llmopt.cache.radix import (RadixCache)

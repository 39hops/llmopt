"""llmopt.quantize — weight diagnostics, bit allocation, and packed artifacts.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.quantize.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "Assignment": "allocator",
    "LayerSensitivity": "sensitivity",
    "allocate_bits": "allocator",
    "awq": "methods",
    "gptq": "methods",
    "hadamard": "rotate",
    "hqq": "methods",
    "load_state_dict": "pack",
    "magnitude_prune": "sparsity",
    "measure_sensitivity": "sensitivity",
    "meter": "meter",
    "meter_group": "meter",
    "pack_state_dict": "pack",
    "pack_tensor": "pack",
    "pareto_front": "allocator",
    "prune_24": "sparsity",
    "rtn": "methods",
    "svd_factorize": "lowrank",
    "unpack_tensor": "pack",
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
    from llmopt.quantize.allocator import (Assignment, allocate_bits, pareto_front)
    from llmopt.quantize.lowrank import (svd_factorize)
    from llmopt.quantize.meter import (meter, meter_group)
    from llmopt.quantize.methods import (awq, gptq, hqq, rtn)
    from llmopt.quantize.pack import (
        load_state_dict,
        pack_state_dict,
        pack_tensor,
        unpack_tensor,
    )
    from llmopt.quantize.rotate import (hadamard)
    from llmopt.quantize.sensitivity import (LayerSensitivity, measure_sensitivity)
    from llmopt.quantize.sparsity import (magnitude_prune, prune_24)

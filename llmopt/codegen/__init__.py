"""llmopt.codegen — toolchain-scored code generation — assemble it, run it.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.codegen.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "CProgram": "generator",
    "assemble": "llvm",
    "build_ladder": "ladder",
    "compile_c": "llvm",
    "evaluate_ladder": "ladder",
    "llvm_available": "llvm",
    "make_arith": "generator",
    "make_branch": "generator",
    "make_loop": "generator",
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
    from llmopt.codegen.generator import (CProgram, make_arith, make_branch, make_loop)
    from llmopt.codegen.ladder import (build_ladder, evaluate_ladder)
    from llmopt.codegen.llvm import (assemble, compile_c, llvm_available)

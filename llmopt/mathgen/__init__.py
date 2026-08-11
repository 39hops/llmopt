"""llmopt.mathgen — seeded generators with symbolic checks built in.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.mathgen.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "Problem": "problems",
    "check_induction": "proofs",
    "evaluate_model": "evaluate",
    "extract_expression": "evaluate",
    "from_prefix": "prefix",
    "make_cdiff": "problems",
    "make_cint": "problems",
    "make_determinant": "linalg",
    "make_eigenvalues": "linalg",
    "make_eom": "mechanics",
    "make_inverse": "linalg",
    "make_kinematics": "mechanics",
    "make_lincong": "ntheory",
    "make_linear_first_order": "odes",
    "make_modpow": "ntheory",
    "make_partial": "multivar",
    "make_prove_ind": "proofs",
    "make_rank": "linalg",
    "make_recurrence": "problems",
    "make_second_order_cc": "odes",
    "make_separable_growth": "odes",
    "make_small_osc": "mechanics",
    "make_sum": "problems",
    "make_taylor": "problems",
    "parse_answer": "problems",
    "to_prefix": "prefix",
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
    from llmopt.mathgen.evaluate import (evaluate_model, extract_expression)
    from llmopt.mathgen.linalg import (
        make_determinant,
        make_eigenvalues,
        make_inverse,
        make_rank,
    )
    from llmopt.mathgen.mechanics import (make_eom, make_kinematics, make_small_osc)
    from llmopt.mathgen.multivar import (make_partial)
    from llmopt.mathgen.ntheory import (make_lincong, make_modpow)
    from llmopt.mathgen.odes import (
        make_linear_first_order,
        make_second_order_cc,
        make_separable_growth,
    )
    from llmopt.mathgen.prefix import (from_prefix, to_prefix)
    from llmopt.mathgen.problems import (
        Problem,
        make_cdiff,
        make_cint,
        make_recurrence,
        make_sum,
        make_taylor,
        parse_answer,
    )
    from llmopt.mathgen.proofs import (check_induction, make_prove_ind)

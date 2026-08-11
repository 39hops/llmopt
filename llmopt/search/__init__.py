"""llmopt.search — symbolic derivation search and verified circuit reduction.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.search.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "AxiomOracle": "axiom_oracle",
    "MarkovPrior": "engine",
    "SearchResult": "derivation",
    "State": "derivation",
    "SyndromePolicy": "engine",
    "ZXState": "zx_engine",
    "best_first_zx": "zx_engine",
    "build_prompt": "proposer",
    "default_jobs": "parallel",
    "featurize": "features",
    "is_solved": "derivation",
    "make_proposer": "proposer",
    "make_scoring_proposer": "proposer",
    "pmap": "parallel",
    "replay_verify": "derivation",
    "solve": "engine",
    "successors": "derivation",
    "tcount": "zx_engine",
    "verify_edge": "derivation",
    "verify_equal": "zx_engine",
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
    from llmopt.search.axiom_oracle import (AxiomOracle)
    from llmopt.search.derivation import (
        SearchResult,
        State,
        is_solved,
        replay_verify,
        successors,
        verify_edge,
    )
    from llmopt.search.engine import (MarkovPrior, SyndromePolicy, solve)
    from llmopt.search.features import (featurize)
    from llmopt.search.parallel import (default_jobs, pmap)
    from llmopt.search.proposer import (build_prompt, make_proposer, make_scoring_proposer)
    from llmopt.search.zx_engine import (ZXState, best_first_zx, tcount, verify_equal)

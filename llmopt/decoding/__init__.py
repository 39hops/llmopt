"""llmopt.decoding — generation-time optimization, all oracle-checked for token identity.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.decoding.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "BatchEngine": "batching",
    "DFA": "fsm",
    "DeterministicLM": "deterministic",
    "FSMConstraint": "fsm",
    "Mirostat": "samplers",
    "Request": "batching",
    "SamplerPipeline": "samplers",
    "Scheduler": "scheduler",
    "StackedEngine": "stacked",
    "TokenFSM": "fsm",
    "TokenTree": "tree_verify",
    "compile_regex": "fsm",
    "find_ngram_continuation": "prompt_lookup",
    "generate_eagle": "eagle",
    "generate_lookahead": "lookahead",
    "generate_lookup": "lookup_generic",
    "generate_lookup_tree": "tree_verify",
    "generate_medusa": "medusa",
    "generate_self_speculative": "self_speculative",
    "generate_speculative": "speculative",
    "generate_speculative_adaptive": "speculative_adaptive",
    "generate_with_prompt_lookup": "prompt_lookup",
    "min_p": "samplers",
    "temperature": "samplers",
    "top_k": "samplers",
    "top_p": "samplers",
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
    from llmopt.decoding.batching import (BatchEngine, Request)
    from llmopt.decoding.deterministic import (DeterministicLM)
    from llmopt.decoding.eagle import (generate_eagle)
    from llmopt.decoding.fsm import (DFA, FSMConstraint, TokenFSM, compile_regex)
    from llmopt.decoding.lookahead import (generate_lookahead)
    from llmopt.decoding.lookup_generic import (generate_lookup)
    from llmopt.decoding.medusa import (generate_medusa)
    from llmopt.decoding.prompt_lookup import (find_ngram_continuation, generate_with_prompt_lookup)
    from llmopt.decoding.samplers import (
        Mirostat,
        SamplerPipeline,
        min_p,
        temperature,
        top_k,
        top_p,
    )
    from llmopt.decoding.scheduler import (Scheduler)
    from llmopt.decoding.self_speculative import (generate_self_speculative)
    from llmopt.decoding.speculative import (generate_speculative)
    from llmopt.decoding.speculative_adaptive import (generate_speculative_adaptive)
    from llmopt.decoding.stacked import (StackedEngine)
    from llmopt.decoding.tree_verify import (TokenTree, generate_lookup_tree)

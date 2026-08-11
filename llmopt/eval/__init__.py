"""llmopt.eval — supporting readouts — never substitutes for a capability gate.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.eval.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "EquivalenceReport": "equivalence",
    "assert_logits_close": "equivalence",
    "assert_tokens_equal": "equivalence",
    "bench_generate": "bench",
    "bootstrap_ci": "stats",
    "ece": "calibration",
    "eval_noise": "stats",
    "measure_latency": "latency",
    "model_ece": "calibration",
    "paired_bootstrap_pvalue": "stats",
    "pass_at_k": "stats",
    "perplexity": "bench",
    "report": "roofline",
    "wilson_interval": "stats",
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
    from llmopt.eval.bench import (bench_generate, perplexity)
    from llmopt.eval.calibration import (ece, model_ece)
    from llmopt.eval.equivalence import (
        EquivalenceReport,
        assert_logits_close,
        assert_tokens_equal,
    )
    from llmopt.eval.latency import (measure_latency)
    from llmopt.eval.roofline import (report)
    from llmopt.eval.stats import (
        bootstrap_ci,
        eval_noise,
        paired_bootstrap_pvalue,
        pass_at_k,
        wilson_interval,
    )

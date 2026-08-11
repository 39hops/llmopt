"""llmopt.train — closed-system births, controlled diets, and comparable interventions.

Symbols resolve lazily: importing this package costs nothing, and torch
or sympy load only when a name is actually used. `llmopt.train.<module>`
still works for anything not re-exported here.
"""
from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "DoRALinear": "lora",
    "LoRALinear": "lora",
    "MathTokenizer": "mathnative",
    "apply_lora": "lora",
    "apply_task_vector": "task_vector",
    "build_model": "mathnative",
    "dpo_loss": "preference",
    "fused_ce": "fused_ce",
    "grpo_advantages": "preference",
    "grpo_loss": "preference",
    "ipo_loss": "preference",
    "kl_vs_ref": "ref_logprobs",
    "kto_loss": "preference",
    "merge_experts": "hebbian_moe",
    "orpo_loss": "preference",
    "pack_batch": "packing",
    "pack_greedy": "packing",
    "precompute_ref_logprobs": "ref_logprobs",
    "simpo_loss": "preference",
    "trainable_fraction": "lora",
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
    from llmopt.train.fused_ce import (fused_ce)
    from llmopt.train.hebbian_moe import (merge_experts)
    from llmopt.train.lora import (DoRALinear, LoRALinear, apply_lora, trainable_fraction)
    from llmopt.train.mathnative import (MathTokenizer, build_model)
    from llmopt.train.packing import (pack_batch, pack_greedy)
    from llmopt.train.preference import (
        dpo_loss,
        grpo_advantages,
        grpo_loss,
        ipo_loss,
        kto_loss,
        orpo_loss,
        simpo_loss,
    )
    from llmopt.train.ref_logprobs import (kl_vs_ref, precompute_ref_logprobs)
    from llmopt.train.task_vector import (apply_task_vector)

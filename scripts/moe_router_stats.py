"""Measure MoE router domain bias: math prompts vs general prose.

Loads an MLX MoE model (default Qwen3-30B-A3B-4bit: 48 layers x 128
experts, 8 active per token), wraps every sparse-MoE block to record
which experts the router picks, runs a mathgen corpus and a general
corpus through the model (prefill only — routing statistics need no
generation), and reports per-layer keep-sets under all three criteria
in moe/router_stats.py plus the domain/general Jaccard overlap.

If the overlap is ~1.0 the router is not domain-biased and expert
pruning is dead on arrival — that result gets reported too.

Output: stats printed per layer + raw counts to
checkpoints/router_stats.json for the pruning step.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from llmopt.moe.router_stats import RouterStats, prune_summary

MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
N_PROMPTS = 64
OUT = Path("checkpoints/router_stats.json")

GENERAL = [
    "The harbor was quiet that morning, fishing boats rocking gently at anchor.",
    "Her grandmother's recipe called for slow-roasted tomatoes and fresh basil.",
    "The referee blew the whistle and the stadium erupted in noise.",
    "By the eighteenth century the trade routes had shifted decisively south.",
    "He tuned the old guitar carefully before the evening's first song.",
    "The documentary follows three families through a season of drought.",
    "Paint peeled from the lighthouse after decades of salt and wind.",
    "She packed two sweaters, a map, and her father's old compass.",
]


def general_corpus(n):
    return [GENERAL[i % len(GENERAL)] + f" (take {i // len(GENERAL)})" for i in range(n)]


def math_corpus(n):
    from llmopt.mathgen.problems import make_dataset

    return [p.prompt for p in make_dataset(n, seed=7)]


def instrument(model, n_experts):
    """Patch the sparse-MoE block CLASS so every forward also records
    the router's top-k picks. Class-level patching because obj(x)
    dispatches through type(obj).__call__ — instance attributes named
    __call__ are never consulted. Returns the mutable state dict whose
    "stats" slot the wrapper writes into (set to None to pause)."""
    state = {"stats": None}
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    layer_of = {id(block): li for li, block in moe_layers}
    cls = type(moe_layers[0][1])

    def wrapped(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        if state["stats"] is not None:
            flat_i = inds.reshape(-1, k).tolist()
            flat_s = scores.reshape(-1, k).tolist()
            state["stats"].update(layer_of[id(self)], flat_i, flat_s)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    cls.__call__ = wrapped
    print(f"instrumented {len(moe_layers)} MoE layers ({cls.__name__})")
    return state


def run_corpus(model, tok, prompts, stats, state):
    state["stats"] = stats
    for i, p in enumerate(prompts):
        ids = tok.encode(p)
        mx.eval(model(mx.array([ids])))
        if (i + 1) % 16 == 0:
            print(f"  {i + 1}/{len(prompts)} prompts")
    state["stats"] = None


def main() -> None:
    from mlx_lm import load

    model, tok = load(MODEL)
    cfg = model.args if hasattr(model, "args") else None
    n_experts = getattr(cfg, "num_experts", 128)

    state = instrument(model, n_experts)
    math_stats = RouterStats(n_experts=n_experts)
    gen_stats = RouterStats(n_experts=n_experts)

    print("math corpus:")
    run_corpus(model, tok, math_corpus(N_PROMPTS), math_stats, state)
    print("general corpus:")
    run_corpus(model, tok, general_corpus(N_PROMPTS), gen_stats, state)

    for crit, thr in (("ever", 0.0), ("mass", 0.99), ("topq", 0.25)):
        summary = prune_summary(math_stats, gen_stats, crit, thr or 0.99)
        sizes = [s["domain_kept"] for s in summary.values()]
        jacc = [s["jaccard"] for s in summary.values()]
        print(f"\ncriterion={crit}: math keeps "
              f"{min(sizes)}-{max(sizes)} of {n_experts} experts/layer, "
              f"mean jaccard vs general {sum(jacc) / len(jacc):.3f}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "model": MODEL,
        "n_experts": n_experts,
        "math": {"counts": math_stats.counts, "mass": math_stats.mass},
        "general": {"counts": gen_stats.counts, "mass": gen_stats.mass},
    }))
    print(f"\nraw stats saved: {OUT}")


if __name__ == "__main__":
    main()

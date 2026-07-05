"""Accuracy-vs-pruning chart for a routing-masked MoE (MLX).

Requires checkpoints/router_stats.json (scripts/moe_router_stats.py).
Evaluates mathgen symbolic accuracy of Qwen3-30B-A3B-4bit: full model
first, then routing-masked under each keep criterion. The eval problems
are a fresh seed, disjoint from the prompts used to collect routing
stats — otherwise the keep-sets would be fitted to the eval set.

Honest-loss framing: if masked accuracy collapses even at high keep
fractions, that kills the "load only the math weights" idea for this
model and we report it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.mathgen.evaluate import SYSTEM, extract_expression
from llmopt.mathgen.problems import make_dataset
from llmopt.moe.prune import keep_fraction, keep_sets, mask_router, stats_from_json
from llmopt.moe.router_stats import RouterStats  # noqa: F401 (unpickling types)

MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
STATS = Path("checkpoints/router_stats.json")
N_EVAL = 120
MAX_TOKENS = 96

CRITERIA = (
    ("ever", 0.0),
    ("mass", 0.99),
    ("mass", 0.95),
    ("topq", 0.50),
    ("topq", 0.25),
)


def evaluate(model, tok, problems) -> float:
    from mlx_lm import generate

    n_ok = 0
    for i, p in enumerate(problems):
        msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": p.prompt},
        ]
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        completion = generate(model, tok, prompt=text, max_tokens=MAX_TOKENS)
        n_ok += p.check(extract_expression(completion))
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{len(problems)}  running acc {n_ok / (i + 1):.1%}")
    return n_ok / len(problems)


def main() -> None:
    from mlx_lm import load

    math_stats, _general, n_experts = stats_from_json(STATS)
    model, tok = load(MODEL)
    # eval seed differs from the router-stats corpus seed (7): keep-sets
    # must not be fitted to the problems they are scored on
    problems = make_dataset(N_EVAL, seed=1234)

    print("full model:")
    base = evaluate(model, tok, problems)
    print(f"  accuracy {base:.1%}\n")

    for crit, thr in CRITERIA:
        keep = keep_sets(math_stats, crit, thr or 0.99)
        frac = keep_fraction(keep, n_experts)
        label = crit if crit == "ever" else f"{crit}@{thr}"
        print(f"masked criterion={label} (keeping {frac:.1%} of experts):")
        unmask = mask_router(model, keep, n_experts)
        try:
            acc = evaluate(model, tok, problems)
            print(f"  accuracy {acc:.1%}  (delta {acc - base:+.1%})\n")
        finally:
            unmask()


if __name__ == "__main__":
    main()

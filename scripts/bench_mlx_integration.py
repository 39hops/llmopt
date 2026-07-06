"""End-to-end tokens/sec: stock mlx-lm vs llmopt fused-swiglu patch.

The only stock chain our kernels beat (rather than tie) is the MLP's
unfused silu*up elementwise pair, so the honest expectation is a
single-digit-percent end-to-end win at decode (MLP elementwise is a
small slice of a decode step next to the projections and attention).
Report whatever falls out, including a loss.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
PROMPT = "Explain, step by step, why the derivative of sin(x**2) is 2*x*cos(x**2)."
MAX_TOKENS = 256
ROUNDS = 5


def decode_tps(model, tok) -> float:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    text = tok.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True, tokenize=False,
    )
    sampler = make_sampler(temp=0.0)
    generate(model, tok, prompt=text, max_tokens=32, sampler=sampler)  # warmup
    best = 0.0
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        out = generate(model, tok, prompt=text, max_tokens=MAX_TOKENS, sampler=sampler)
        n = len(tok.encode(out))
        best = max(best, n / (time.perf_counter() - t0))
    return best


def main() -> None:
    from mlx_lm import load

    from llmopt.kernels.mlx_integration import patch_swiglu

    model, tok = load(MODEL)
    stock = decode_tps(model, tok)
    print(f"stock mlx-lm      {stock:7.1f} tok/s")

    n, unpatch = patch_swiglu(model)
    print(f"patched {n} MLPs with fused swiglu")
    try:
        fused = decode_tps(model, tok)
        print(f"fused swiglu      {fused:7.1f} tok/s  ({fused / stock - 1:+.1%})")
    finally:
        unpatch()


if __name__ == "__main__":
    main()

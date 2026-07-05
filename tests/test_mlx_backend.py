"""MLX backend integration test (Apple silicon only, skipped elsewhere).

Downloads a small 4-bit model on first run; verifies the generic lookup
loop over MLXBackend is exactly greedy-equivalent to a manual decode.
"""

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from llmopt.backends.mlx_backend import MLXBackend
from llmopt.decoding.lookup_generic import generate_lookup

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


def _greedy_reference(model, ids: list[int], n: int) -> list[int]:
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(mx.array([ids]), cache=cache)
    out = [int(mx.argmax(logits[0, -1]))]
    for _ in range(n - 1):
        logits = model(mx.array([[out[-1]]]), cache=cache)
        out.append(int(mx.argmax(logits[0, -1])))
    return out


def test_mlx_lookup_matches_greedy():
    model, tok = mlx_lm.load(MODEL)
    prompt = (
        "Repeat this exactly, then continue the pattern: the quick brown "
        "fox jumps over the lazy dog. the quick brown fox jumps over the "
        "lazy dog. the quick brown"
    )
    ids = tok.encode(prompt)
    n = 80

    ref = _greedy_reference(model, ids, n)
    out, stats = generate_lookup(
        MLXBackend(model), ids, max_new_tokens=n, num_draft=16, max_ngram=4
    )

    assert out[len(ids):] == ref
    assert stats["forward_passes"] <= n + 1

"""Fused-kernel patching must be logit-equivalent to the stock model."""

import pytest

mx = pytest.importorskip("mlx.core")


def _load_small():
    try:
        from mlx_lm import load

        return load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    except Exception as e:  # not cached / no network
        pytest.skip(f"model unavailable: {e}")


def test_patched_model_matches_stock_logits():
    from llmopt.kernels.mlx_integration import patch_swiglu

    model, tok = _load_small()
    ids = mx.array([tok.encode("The derivative of x**2 is")])

    stock = model(ids)
    mx.eval(stock)

    n, unpatch = patch_swiglu(model)
    assert n > 0, "no MLP modules found to patch"
    try:
        fused = model(ids)
        mx.eval(fused)
        # fp16 accumulation-order differences compound across 24
        # layers; logits land within ~0.02 on magnitude ~10. The
        # semantic contract is the argmax; near-equality is a sanity
        # band, not bit-identity.
        assert mx.allclose(stock, fused, atol=0.1, rtol=0.05)
        assert mx.argmax(stock[0, -1]) == mx.argmax(fused[0, -1])
    finally:
        unpatch()

    restored = model(ids)
    assert mx.array_equal(stock, restored), "unpatch did not restore stock path"

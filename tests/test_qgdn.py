"""llmopt.lab.qgdn: the recurrence selector proves which GatedDeltaNet
implementation executed, restores the vendor names, refuses nesting, and
the two vendor implementations agree on small realistic operands."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from llmopt.lab import qgdn  # noqa: E402

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="the Metal kernel needs a Metal device")


def operands(T=8, B=1, Hk=2, Hv=4, Dk=32, Dv=32, seed=0):
    """Realistic shapes and ranges: unit-norm-ish q and k, v of order one, a
    and b of order one (so g = exp(-exp(A_log) softplus(a + dt_bias)) is in
    (0, 1) and beta = sigmoid(b) in (0, 1)), A_log as the vendor initialises it."""
    r = np.random.default_rng(seed)
    q = r.standard_normal((B, T, Hk, Dk)).astype(np.float32) / np.sqrt(Dk)
    k = r.standard_normal((B, T, Hk, Dk)).astype(np.float32) / np.sqrt(Dk)
    v = r.standard_normal((B, T, Hv, Dv)).astype(np.float32)
    a = r.standard_normal((B, T, Hv)).astype(np.float32)
    b = r.standard_normal((B, T, Hv)).astype(np.float32)
    A_log = np.log(r.uniform(0.5, 16, size=(Hv,))).astype(np.float32)
    dt_bias = np.ones((Hv,), dtype=np.float32)
    return tuple(mx.array(x) for x in (q, k, v, a, b, A_log, dt_bias))


def small_gdn():
    """A GatedDeltaNet module with random weights (training mode, as a tower
    built by qtower is)."""
    from mlx_lm.models import qwen3_5
    args = qwen3_5.TextModelArgs(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                 vocab_size=256, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32)
    m = qwen3_5.GatedDeltaNet(args)
    mx.eval(m.parameters())
    return m


def test_invalid_path_refused():
    with pytest.raises(ValueError):
        qgdn.recurrence_path("fast")


def test_check_vendor_reports_provenance():
    v = qgdn.check_vendor()
    assert v["mlx_lm_version"] and len(v["qwen3_5_sha256"]) == 64 and len(v["gated_delta_sha256"]) == 64


@metal
@pytest.mark.parametrize("path,want", [("loop", "ops"), ("kernel", "kernel"), ("vendor", "ops")])
def test_selection_dispatches_exactly_one_implementation(path, want):
    from mlx_lm.models import gated_delta, qwen3_5
    m = small_gdn()
    assert m.training        # the vendor flag alone would select the loop
    x = mx.random.normal((1, 6, 64))
    before = (gated_delta.gated_delta_ops, gated_delta.gated_delta_kernel, qwen3_5.gated_delta_update)
    with qgdn.recurrence_path(path) as sel:
        y = m(x)
        mx.eval(y)
    other = "kernel" if want == "ops" else "ops"
    assert sel.count.update_calls == 1 and sel.count.executed[want] == 1 and sel.count.executed[other] == 0
    assert sel.count.requested == {"kernel": 0, "loop": 1}      # what the vendor flag asked for, in training mode
    assert sel.count.max_T == 6
    sel.count.assert_exclusive()
    assert (gated_delta.gated_delta_ops, gated_delta.gated_delta_kernel, qwen3_5.gated_delta_update) == before
    r = sel.receipt()
    assert r["path"] == path and r["executed"][want] == 1 and r["vendor"]["mlx_lm_version"]


@metal
def test_nesting_refused_and_restored_after_error():
    from mlx_lm.models import qwen3_5
    real = qwen3_5.gated_delta_update
    with qgdn.recurrence_path("loop"):
        with pytest.raises(RuntimeError):
            with qgdn.recurrence_path("kernel"):
                pass
    assert qwen3_5.gated_delta_update is real
    with pytest.raises(ZeroDivisionError):
        with qgdn.recurrence_path("kernel"):
            1 / 0
    assert qwen3_5.gated_delta_update is real
    assert qgdn.recurrence_path._active is None


def test_assert_exclusive_refuses_mixed_or_empty():
    c = qgdn.DispatchCount("kernel")
    with pytest.raises(RuntimeError):
        c.assert_exclusive()
    c.update_calls, c.executed = 2, {"ops": 1, "kernel": 1}
    with pytest.raises(RuntimeError):
        c.assert_exclusive()
    c.executed = {"ops": 0, "kernel": 2}
    c.assert_exclusive()


@metal
def test_kernel_and_loop_agree_on_realistic_operands():
    from mlx_lm.models import gated_delta
    q, k, v, a, b, A_log, dt_bias = operands()
    y0, s0 = gated_delta.gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, None, use_kernel=False)
    y1, s1 = gated_delta.gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, None, use_kernel=True)
    mx.eval(y0, s0, y1, s1)
    ey, es = qgdn.normalized_errors(y1, y0), qgdn.normalized_errors(s1, s0)
    assert ey["finite"] and es["finite"]
    assert ey["max_abs_over_rms"] < 1e-4 and es["max_abs_over_rms"] < 1e-4
    assert s0.dtype == mx.float32 and s1.dtype == mx.float32


@metal
def test_kernel_state_continues_across_a_chunk_boundary():
    from mlx_lm.models import gated_delta
    q, k, v, a, b, A_log, dt_bias = operands(T=8)
    y_full, s_full = gated_delta.gated_delta_update(q, k, v, a, b, A_log, dt_bias, None, None, use_kernel=True)
    y_a, s_a = gated_delta.gated_delta_update(q[:, :4], k[:, :4], v[:, :4], a[:, :4], b[:, :4], A_log, dt_bias, None, None, use_kernel=True)
    y_b, s_b = gated_delta.gated_delta_update(q[:, 4:], k[:, 4:], v[:, 4:], a[:, 4:], b[:, 4:], A_log, dt_bias, s_a, None, use_kernel=True)
    mx.eval(y_full, s_full, y_a, s_a, y_b, s_b)
    y_cat = mx.concatenate([y_a, y_b], axis=1)
    assert qgdn.normalized_errors(y_cat, y_full)["max_abs_over_rms"] < 1e-5
    assert qgdn.normalized_errors(s_b, s_full)["max_abs_over_rms"] < 1e-5


def test_normalized_errors_handles_near_zero_elements():
    ref = np.array([1.0, 1e-9, -2.0, 0.0])
    same = qgdn.normalized_errors(ref, ref)
    assert same["max_abs"] == 0.0 and same["max_abs_over_rms"] == 0.0 and same["max_rel_on_large"] == 0.0 and same["finite"]
    assert same["n_large"] == 2 and same["n_near_zero"] == 2       # 1e-9 and 0 are below 1e-3 of the max
    e = qgdn.normalized_errors(ref + np.array([0.0, 1e-4, 0.0, 1e-4]), ref)
    assert e["max_abs"] == pytest.approx(1e-4)
    assert e["max_rel_on_large"] == 0.0                              # the perturbed elements are near-zero, so excluded from the relative read
    assert e["n_elements_over_1e-5_of_max"] == 2
    with pytest.raises(ValueError):
        qgdn.normalized_errors(np.zeros(3), np.zeros(4))
    nan = qgdn.normalized_errors(np.array([np.nan, 1.0]), np.array([1.0, 1.0]))
    assert not nan["finite"]


@metal
def test_selection_reaches_an_exec_rewritten_call_like_qtower():
    """llmopt.lab.qtower replaces GatedDeltaNet.__call__ with a body compiled by
    exec under vars(qwen3_5) as globals; the selector must still be what that
    body calls. Replicate the rewrite on the vendor source (no l2norm edit) and
    check the dispatch counts."""
    import inspect
    import textwrap
    from mlx_lm.models import qwen3_5
    original = qwen3_5.GatedDeltaNet.__call__
    src = textwrap.dedent(inspect.getsource(original))
    ns: dict = {}
    exec(compile(src, "qwen3_5_gdn_test_rewrite", "exec"), vars(qwen3_5), ns)
    qwen3_5.GatedDeltaNet.__call__ = ns["__call__"]
    try:
        m = small_gdn()
        x = mx.random.normal((1, 5, 64))
        with qgdn.recurrence_path("kernel") as sel:
            mx.eval(m(x))
        assert sel.count.executed == {"ops": 0, "kernel": 1} and sel.count.requested == {"kernel": 0, "loop": 1}
        with qgdn.recurrence_path("loop") as sel:
            mx.eval(m(x))
        assert sel.count.executed == {"ops": 1, "kernel": 0}
    finally:
        qwen3_5.GatedDeltaNet.__call__ = original


@metal
def test_exit_detects_a_foreign_change_inside_the_block():
    from mlx_lm.models import qwen3_5
    real = qwen3_5.gated_delta_update
    with pytest.raises(RuntimeError, match="changed by someone else"):
        with qgdn.recurrence_path("loop"):
            qwen3_5.gated_delta_update = lambda *a, **k: None
    assert qwen3_5.gated_delta_update is real

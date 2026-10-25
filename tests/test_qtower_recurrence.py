"""llmopt.lab.qtower.select_recurrence: the per-instance selection of the
GatedDeltaNet recurrence implementation (QWEN-GDN-PROMOTE-0). Two instances
selected differently dispatch differently in one process; a plain vendor
instance beside them keeps the vendor's own behaviour and class; the
vendor class attribute and module dictionary are untouched by a selection;
the training flag is never touched; an unselected re-classed instance
refuses to run; the exact-l2norm edit of build_metal_model is carried."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from llmopt.lab import qgdn, qtower  # noqa: E402

metal = pytest.mark.skipif(not mx.metal.is_available(), reason="the Metal kernel needs a Metal device")


def small_gdn():
    from mlx_lm.models import qwen3_5
    args = qwen3_5.TextModelArgs(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                                 vocab_size=256, linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=32)
    m = qwen3_5.GatedDeltaNet(args)
    mx.eval(m.parameters())
    return m


def executed(m, x):
    with qgdn.recurrence_path("vendor") as sel:
        y = m(x)
        mx.eval(y)
    return sel.count.executed, sel.count.requested, y


def test_invalid_path_and_empty_tree_refused():
    import mlx.nn as nn
    with pytest.raises(ValueError):
        qtower.select_recurrence(small_gdn(), "fast")
    with pytest.raises(RuntimeError, match="no GatedDeltaNet"):
        qtower.select_recurrence(nn.Linear(4, 4), "kernel")


def test_selected_class_source_carries_the_l2norm_edit_and_the_instance_dispatch():
    cls = qtower.selected_gdn_class()
    src = cls._source
    assert qtower.DISPATCH_LINE not in src and qtower.SELECTED_DISPATCH_LINE in src
    for old, new in qtower.L2NORM_EDITS:
        assert old not in src and new in src
    # the same edit build_metal_model installs (the frozen body's literal lines)
    import inspect
    body = inspect.getsource(qtower.build_metal_model)
    for old, new in qtower.L2NORM_EDITS:
        assert repr(old)[1:-1] in body and repr(new)[1:-1] in body


@metal
def test_two_instances_select_independently_and_a_vendor_instance_is_untouched():
    from mlx_lm.models import gated_delta, qwen3_5
    vendor_call = qwen3_5.GatedDeltaNet.__call__
    vendor_names = (gated_delta.gated_delta_ops, gated_delta.gated_delta_kernel, qwen3_5.gated_delta_update)
    a, b, c = small_gdn(), small_gdn(), small_gdn()
    sa = qtower.select_recurrence(a, "kernel")
    sb = qtower.select_recurrence(b, "loop")
    assert sa["n_selected"] == sb["n_selected"] == 1 and sa["vendor"]["mlx_lm_version"]
    assert type(a).__name__ == "SelectedGatedDeltaNet" and type(b) is type(a) and type(c) is qwen3_5.GatedDeltaNet
    assert qwen3_5.GatedDeltaNet.__call__ is vendor_call                       # the vendor class is not modified by a selection
    assert (gated_delta.gated_delta_ops, gated_delta.gated_delta_kernel, qwen3_5.gated_delta_update) == vendor_names
    assert a.training and b.training and c.training                        # the training flag is never touched
    x = mx.random.normal((1, 6, 64))
    ea, ra, ya = executed(a, x)
    eb, rb, yb = executed(b, x)
    ec, rc, _ = executed(c, x)
    assert ea == {"ops": 0, "kernel": 1} and ra == {"kernel": 1, "loop": 0}
    assert eb == {"ops": 1, "kernel": 0} and rb == {"kernel": 0, "loop": 1}
    assert ec == {"ops": 1, "kernel": 0} and rc == {"kernel": 0, "loop": 1}   # the vendor instance: use_kernel = not training = False
    assert qtower.recurrence_of(a) == {"loop": 0, "kernel": 1, "unselected": 0}
    assert qtower.recurrence_of(c) == {"loop": 0, "kernel": 0, "unselected": 1}
    # re-selection flips the same instance; the other is unaffected
    qtower.select_recurrence(a, "loop")
    assert executed(a, x)[0] == {"ops": 1, "kernel": 0} and executed(b, x)[0] == {"ops": 1, "kernel": 0}
    qtower.select_recurrence(b, "kernel")
    assert executed(b, x)[0] == {"ops": 0, "kernel": 1} and executed(a, x)[0] == {"ops": 1, "kernel": 0}
    # a selection is not a parameter
    assert "recurrence" not in a and "recurrence" not in dict(a.parameters())


@metal
def test_selected_instances_agree_with_the_forced_instrument_on_the_same_weights():
    """The promoted per-instance selection and the qualification's forced
    selection are the same computation at the update call: on one selected
    instance (the body with the exact-l2norm edit, as in the tower), forcing a
    path through the instrument gives bit-identical outputs to selecting it.
    (The plain vendor body differs from the selected body by the l2norm edit
    build_metal_model installs, so the comparison is made on the selected body.)"""
    m = small_gdn()
    x = mx.random.normal((1, 7, 64))
    qtower.select_recurrence(m, "loop")
    with qgdn.recurrence_path("kernel") as sel:
        y_forced_k = m(x)
        mx.eval(y_forced_k)
    assert sel.count.executed == {"ops": 0, "kernel": 1} and sel.count.requested == {"kernel": 0, "loop": 1}   # the instance asked for the loop; the instrument forced the kernel
    qtower.select_recurrence(m, "kernel")
    with qgdn.recurrence_path("loop") as sel:
        y_forced_l = m(x)
        mx.eval(y_forced_l)
    assert sel.count.executed == {"ops": 1, "kernel": 0} and sel.count.requested == {"kernel": 1, "loop": 0}
    e, _, y_k = executed(m, x)
    assert e == {"ops": 0, "kernel": 1}
    qtower.select_recurrence(m, "loop")
    e, _, y_l = executed(m, x)
    assert e == {"ops": 1, "kernel": 0}
    assert qgdn.normalized_errors(y_k, y_forced_k)["max_abs"] == 0.0
    assert qgdn.normalized_errors(y_l, y_forced_l)["max_abs"] == 0.0
    assert qgdn.normalized_errors(y_k, y_l)["max_abs_over_rms"] < 1e-4        # and the two paths agree to the qualified level


def test_tree_selection_reaches_every_gdn_and_an_unselected_instance_refuses():
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5

    class Tree(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [small_gdn(), small_gdn()]
            self.other = nn.Linear(4, 4)

    t = Tree()
    sel = qtower.select_recurrence(t, "loop")
    assert sel["n_selected"] == 2 and qtower.recurrence_of(t) == {"loop": 2, "kernel": 0, "unselected": 0}
    # an instance re-classed by hand without a selection refuses to run
    m = small_gdn()
    m.__class__ = qtower.selected_gdn_class()
    assert isinstance(m, qwen3_5.GatedDeltaNet)
    with pytest.raises(RuntimeError, match="no recurrence selection"):
        mx.eval(m(mx.random.normal((1, 3, 64))))


def test_kernel_refused_under_an_unqualified_vendor_source_and_loop_still_selectable(monkeypatch):
    m = small_gdn()
    monkeypatch.setitem(qtower.QUALIFIED_VENDOR, "gated_delta_sha256", "0" * 64)
    with pytest.raises(RuntimeError, match="REFUSING the kernel recurrence"):
        qtower.select_recurrence(m, "kernel")
    assert qtower.select_recurrence(m, "loop")["n_selected"] == 1


def test_qualified_vendor_pins_are_the_installed_sources():
    """The installed mlx-lm is the qualified one (VERDICT QWEN-GDN-KERNEL-0); if this
    fails after an upgrade, the kernel default refuses by design and needs a re-qualification."""
    v = qgdn.check_vendor()
    assert {k: v[k] for k in qtower.QUALIFIED_VENDOR} == qtower.QUALIFIED_VENDOR

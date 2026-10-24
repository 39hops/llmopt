"""Explicit selection of the GatedDeltaNet recurrence implementation for
the arm C Metal tower, with proof of dispatch in the receipt.

mlx-lm's GatedDeltaNet.__call__ passes ``use_kernel=not self.training`` to
``gated_delta_update``; a module built by llmopt.lab.qtower is never put in
evaluation mode, so every prefill and decode of arm C has run
``gated_delta_ops`` (the per-token Python loop) while ``gated_delta_kernel``
(the Metal kernel the same module ships) sat unused (VERDICT
QWEN-PREFILL-WALL-0). Flipping the training flag would select the kernel
by a side effect of a mode meant for dropout; this module selects the
implementation by name instead and leaves the training flag alone.

How. The tower's rewritten ``__call__`` executes with ``vars(qwen3_5)`` as
its globals, so it resolves ``gated_delta_update`` from the qwen3_5 module
dictionary at call time. ``recurrence_path`` replaces that name, for the
duration of a ``with`` block, with a wrapper that forwards to the vendor
function with ``use_kernel`` forced to the chosen path. The wrapper also
wraps ``gated_delta_ops`` and ``gated_delta_kernel`` inside the
``gated_delta`` module, so the receipt counts which implementation
EXECUTED, not which one was requested. Nested selection is refused, exit
restores every name and asserts the restoration, and nothing here is
imported by the serving code (llmopt.lab.qserve), so a serving process is
unaffected.

Vendor guard. ``check_vendor`` refuses when the installed mlx-lm no longer
carries the dispatch line, the condition inside ``gated_delta_update`` or
the two implementation functions this module relies on; the receipt records
the sha256 of both vendor source files and the installed versions.
"""
from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field

PATHS = ("loop", "kernel", "vendor")
DISPATCH_LINE = "use_kernel=not self.training,"
UPDATE_CONDITION = "if not use_kernel or mx.default_device() != mx.gpu or not mx.metal.is_available():"


def _modules():
    from mlx_lm.models import gated_delta, qwen3_5
    return gated_delta, qwen3_5


def _file_sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_vendor() -> dict:
    """Refuse unless the installed mlx-lm has the dispatch this module
    relies on; return the vendor provenance for a receipt."""
    import mlx.core as mx
    import mlx_lm
    gated_delta, qwen3_5 = _modules()
    # the ORIGINAL vendor class body (qtower rewrites the bound __call__ in
    # place; the module source on disk is what we can read)
    src = open(qwen3_5.__file__).read()
    if DISPATCH_LINE not in src:
        raise RuntimeError("REFUSING: mlx-lm GatedDeltaNet no longer dispatches on use_kernel=not self.training; re-derive the selector")
    upd = inspect.getsource(gated_delta.gated_delta_update)
    if UPDATE_CONDITION not in upd:
        raise RuntimeError("REFUSING: mlx-lm gated_delta_update dispatch condition changed; re-derive the selector")
    for name in ("gated_delta_ops", "gated_delta_kernel"):
        if not callable(getattr(gated_delta, name, None)):
            raise RuntimeError(f"REFUSING: mlx-lm gated_delta has no {name}")
    if qwen3_5.__dict__.get("gated_delta_update") is not gated_delta.gated_delta_update:
        raise RuntimeError("REFUSING: qwen3_5.gated_delta_update is already replaced by something else")
    return {"mlx_version": mx.__version__, "mlx_lm_version": mlx_lm.__version__,
            "qwen3_5_sha256": _file_sha(qwen3_5.__file__), "gated_delta_sha256": _file_sha(gated_delta.__file__),
            "metal_available": bool(mx.metal.is_available()), "default_device": str(mx.default_device())}


@dataclass
class DispatchCount:
    """What executed inside one selection: calls to the vendor update
    entry point, with the use_kernel the caller asked for, and calls that
    reached each implementation."""
    path: str
    update_calls: int = 0
    requested: dict = field(default_factory=lambda: {"kernel": 0, "loop": 0})
    executed: dict = field(default_factory=lambda: {"ops": 0, "kernel": 0})
    max_T: int = 0

    def as_dict(self) -> dict:
        return {"path": self.path, "update_calls": self.update_calls, "requested_by_vendor_flag": dict(self.requested),
                "executed": dict(self.executed), "max_T": self.max_T}

    def assert_exclusive(self) -> None:
        """Refuse a receipt in which the unselected implementation ran (or
        nothing ran at all)."""
        if self.update_calls == 0:
            raise RuntimeError("no recurrence call happened under the selection")
        want = {"loop": "ops", "kernel": "kernel"}.get(self.path)
        if want is None:
            return
        other = "kernel" if want == "ops" else "ops"
        if self.executed[other] or not self.executed[want]:
            raise RuntimeError(f"dispatch proof failed for path {self.path}: executed {self.executed}")


class recurrence_path:
    """``with recurrence_path("kernel") as sel: ...`` runs every GatedDeltaNet
    recurrence inside the block through the Metal kernel; "loop" through the
    Python reference loop (today's released behaviour, bit for bit the same
    code path); "vendor" leaves the vendor's own flag in force and only
    counts. ``sel.count`` is the DispatchCount; ``sel.receipt()`` the dict
    for a receipt. Exit restores the vendor names and raises if anyone else
    had changed them inside the block (checked before restoring)."""

    _active = None

    def __init__(self, path: str):
        if path not in PATHS:
            raise ValueError(f"path must be one of {PATHS}, got {path!r}")
        self.path = path
        self.count = DispatchCount(path)
        self.vendor = None
        self._saved = None
        self._installed = None

    def __enter__(self):
        if recurrence_path._active is not None:
            raise RuntimeError("a recurrence_path selection is already active; nesting is refused")
        self.vendor = check_vendor()
        gated_delta, qwen3_5 = _modules()
        real_update, real_ops, real_kernel = gated_delta.gated_delta_update, gated_delta.gated_delta_ops, gated_delta.gated_delta_kernel
        self._saved = (gated_delta, qwen3_5, real_update, real_ops, real_kernel)
        count, path = self.count, self.path

        def counted_ops(*a, **k):
            count.executed["ops"] += 1
            return real_ops(*a, **k)

        def counted_kernel(*a, **k):
            count.executed["kernel"] += 1
            return real_kernel(*a, **k)

        def selected_update(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
            count.update_calls += 1
            count.requested["kernel" if use_kernel else "loop"] += 1
            count.max_T = max(count.max_T, int(q.shape[1]))
            forced = {"loop": False, "kernel": True}.get(path, use_kernel)
            return real_update(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=forced)

        # gated_delta_update resolves its implementations from its own module globals
        gated_delta.gated_delta_ops = counted_ops
        gated_delta.gated_delta_kernel = counted_kernel
        qwen3_5.gated_delta_update = selected_update
        self._installed = (counted_ops, counted_kernel, selected_update)
        recurrence_path._active = self
        return self

    def __exit__(self, *exc):
        gated_delta, qwen3_5, real_update, real_ops, real_kernel = self._saved
        mine = self._installed
        changed = (gated_delta.gated_delta_ops is not mine[0] or gated_delta.gated_delta_kernel is not mine[1] or qwen3_5.gated_delta_update is not mine[2])
        try:
            gated_delta.gated_delta_ops = real_ops
            gated_delta.gated_delta_kernel = real_kernel
            qwen3_5.gated_delta_update = real_update
        finally:
            recurrence_path._active = None
        if changed:
            raise RuntimeError("the vendor names were changed by someone else inside the selection; restored, but the counts cannot be trusted")
        return False

    def receipt(self) -> dict:
        return {"selector": "llmopt.lab.qgdn.recurrence_path", **self.count.as_dict(), "vendor": dict(self.vendor or {})}


def normalized_errors(a, b, rel_floor: float = 1e-3) -> dict:
    """Error of array ``a`` against reference ``b`` (numpy or mlx arrays),
    robust to near-zero elements: max absolute difference; that maximum
    over the root mean square of the reference (the normalized error the
    bars read); the largest element-wise relative error over the elements
    whose reference magnitude is at least ``rel_floor`` times the
    reference's largest magnitude (near-zero elements are excluded from
    the relative read and counted); the count of elements differing by
    more than 1e-5 of the reference's largest magnitude; finiteness of
    both arrays."""
    import numpy as np
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch {x.shape} != {y.shape}")
    finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
    d = np.abs(x - y)
    ymax = float(np.abs(y).max()) if y.size else 0.0
    rms = float(np.sqrt(np.mean(y * y))) if y.size else 0.0
    big = np.abs(y) >= rel_floor * ymax if ymax > 0 else np.zeros_like(y, dtype=bool)
    rel = float((d[big] / np.abs(y[big])).max()) if big.any() else 0.0
    return {"finite": finite, "max_abs": float(d.max()) if d.size else 0.0,
            "max_abs_over_rms": (float(d.max()) / rms) if rms > 0 else (0.0 if d.size and d.max() == 0 else float("inf")),
            "max_rel_on_large": rel, "n_large": int(big.sum()), "n_near_zero": int(y.size - big.sum()),
            "n_elements_over_1e-5_of_max": int((d > 1e-5 * ymax).sum()) if ymax > 0 else int((d > 0).sum()),
            "ref_rms": rms, "ref_max_abs": ymax, "n": int(y.size)}

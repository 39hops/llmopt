"""The qualified arm C Metal tower, ADOPTED verbatim from
scratch/qwen_metal_qual.py (build_metal_model, fsha, NORM_SHIFT_SUFFIXES;
the construction of scratch/qwen_metal_tower.py frozen as the
QWEN-METAL-TOWER-0 record and qualified in OBSERVATION QWEN-METAL-QUAL-0):
the mlx-lm qwen3_5 architecture with the fused W4/S16 linears
(llmopt.lab.qmetal) in place of every quantised nn.Linear, raw tensors
with the sanitizer conventions (the +1 norm shift, the conv1d axis), the
host-decoded embedding, and the exact HF l2norm on the DeltaNet q and k
installed by rewriting mlx-lm's GatedDeltaNet.__call__ source under a
guard that REFUSES if the vendor source changed. Nothing here changes
numerical semantics; the scratch original stays in place as the frozen
record and tests/test_qtower_adoption.py pins the two bodies equal.

Serving code (llmopt.lab.qserve) imports from here; research drivers keep
importing scratch/qwen_metal_qual.py until a registered re-run migrates
them.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

NORM_SHIFT_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                       "model.norm.weight", ".q_norm.weight", ".k_norm.weight")


def fsha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build_metal_model(art: str, vdir: str):
    """Arm C through the mlx-lm Qwen3.5 architecture with fused linears.
    Same construction as scratch/qwen_metal_tower.py (frozen as the
    QWEN-METAL-TOWER-0 record); returns (model, info)."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5
    from llmopt.lab import qartifact, qmetal
    from llmopt.lab.qcodec import dec_raw
    from llmopt.lab.qcodec_fast import S16Rows, W4Rows

    arm = os.path.basename(art.rstrip("/"))
    chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    handles: dict = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(art, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    import inspect
    import textwrap
    src = textwrap.dedent(inspect.getsource(qwen3_5.GatedDeltaNet.__call__))
    a_line = "q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)"
    b_line = "k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)"
    if a_line not in src or b_line not in src:
        raise SystemExit("REFUSING: mlx-lm GatedDeltaNet source changed; re-derive the l2norm override")
    src = src.replace(a_line, "q = inv_scale * (q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6))")
    src = src.replace(b_line, "k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)")
    ns: dict = {}
    exec(compile(src, "qwen3_5_gdn_exact_l2norm", "exec"), vars(qwen3_5), ns)
    qwen3_5.GatedDeltaNet.__call__ = ns["__call__"]

    cfg = json.load(open(os.path.join(vdir, "config.json")))
    args = qwen3_5.ModelArgs.from_dict(cfg)
    model = qwen3_5.Model(args)

    class QLinear(nn.Module):
        def __init__(self, res):
            super().__init__()
            self.res = res
            self.out_features, self.in_features = res.shape

        def __call__(self, x):
            lead = x.shape[:-1]
            y = self.res.gemm(x.reshape(-1, x.shape[-1]).astype(mx.float32))
            return y.reshape(*lead, self.out_features)

    class QEmbedding(nn.Module):
        def __init__(self, rows):
            super().__init__()
            self.rows = rows

        def __call__(self, ids):
            flat = np.asarray(ids).reshape(-1)
            vecs = np.stack([self.rows.rows(int(t), int(t) + 1)[0] for t in flat]).astype(np.float32)
            return mx.array(vecs).reshape(*ids.shape, -1)

    def set_path(root, path, value):
        parts = path.split(".")
        obj = root
        for p in parts[:-1]:
            obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
        leaf = parts[-1]
        if leaf.isdigit():
            obj[int(leaf)] = value
        else:
            setattr(obj, leaf, value)

    n_fused = n_raw = 0
    bytes_resident = 0
    for key, e in MAN.items():
        if not isinstance(e, dict) or e["codec"] == "excluded":
            continue
        if key.startswith("model.visual") or ".mtp." in key or key.startswith("mtp."):
            continue
        k = key.replace("model.language_model.", "language_model.model.") if key.startswith("model.language_model.") \
            else "language_model." + key
        if e["codec"] in ("w4", "s16"):
            if k.endswith("embed_tokens.weight"):
                buf = payload(e)
                rows = (S16Rows if e["codec"] == "s16" else W4Rows)(buf, e["shape"])
                set_path(model, k[:-len(".weight")], QEmbedding(rows))
                bytes_resident += len(buf)
            else:
                res = qmetal.resident(payload(e), e)
                bytes_resident += res.payload_bytes()
                set_path(model, k[:-len(".weight")], QLinear(res))
            n_fused += 1
        else:
            arr = dec_raw(payload(e), e["shape"], e["dtype"]).astype(np.float32)
            if "conv1d.weight" in k and arr.shape[-1] != 1:
                arr = np.moveaxis(arr, 2, 1)
            if any(k.endswith(s) for s in NORM_SHIFT_SUFFIXES) and arr.ndim == 1:
                arr = arr + 1.0
            set_path(model, k, mx.array(arr))
            n_raw += 1
    mx.eval(model.parameters())
    tm = model.language_model.model
    fams = ["linear_attn" if hasattr(ly, "linear_attn") else "full_attn" for ly in tm.layers]
    info = {"chain": chain, "chain_sha256": fsha(chain), "chain_report": q["report"],
            "n_layers": len(tm.layers), "families": {"linear_attn": fams.count("linear_attn"),
                                                     "full_attn": fams.count("full_attn")},
            "fused": n_fused, "raw": n_raw, "payload_resident_bytes": bytes_resident,
            "kernel_version": qmetal.VERSION,
            "prefill_gemm": bool(qmetal._Resident.prefill_gemm), "gemm_min_t": (qmetal._Resident.gemm_min_t if qmetal._Resident.gemm_min_t is not None else dict(qmetal.GEMM_MIN_T_BY_CODEC)),
            "gemm_slab_bytes": int(qmetal.GEMM_SLAB_BYTES),
            "gemm_ineligible_matrices": sum(1 for _, m in model.named_modules()
                                            if type(m).__name__ == "QLinear" and not m.res.gemm_ok)}
    return model, info


# ---------------------------------------------------------------------------
# Explicit, per-instance selection of the GatedDeltaNet recurrence
# implementation (QWEN-GDN-PROMOTE-0). mlx-lm's GatedDeltaNet.__call__
# passes use_kernel=not self.training to gated_delta_update, so a tower
# built in training mode runs the Python reference loop (gated_delta_ops)
# and never the Metal kernel the same module ships (VERDICT
# QWEN-PREFILL-WALL-0). VERDICT QWEN-GDN-KERNEL-0 qualified the kernel
# against the loop on arm C. The promoted mechanism selects the
# implementation BY NAME, PER MODULE INSTANCE: every GatedDeltaNet of one
# tower is re-classed to SelectedGatedDeltaNet, whose __call__ is the
# vendor body (with the same exact-l2norm edit build_metal_model installs)
# except that the dispatch argument reads the instance's own `recurrence`
# attribute. The training flag is never touched; the vendor class and its
# module dictionary are not modified by the selection, so a GatedDeltaNet
# built by plain mlx-lm in the same process, or a second tower selected
# differently, keeps its own behaviour. Dispatch PROOF (counting which
# implementation executed) stays with the qualification instrument
# llmopt.lab.qgdn.recurrence_path("vendor"), which drivers wrap around a
# measured call; nothing in serving depends on it.

RECURRENCE_PATHS = ("loop", "kernel")
RECURRENCE_FLAGS = {"loop": False, "kernel": True}
# the vendor sources the kernel path was QUALIFIED against (VERDICT QWEN-GDN-KERNEL-0: mlx-lm 0.31.3);
# selecting "kernel" under any other source refuses (the loop stays selectable: it is the vendor's own reference path)
QUALIFIED_VENDOR = {"qwen3_5_sha256": "f0daa30bba5cb521c8bdfa7093101a544c6a37bbba09bca582288219cb04ae3a",
                    "gated_delta_sha256": "79c8376a51c694b03e54d2f996ced6ea6c8c42868b8571529f97334db165a3e1"}
DISPATCH_LINE = "use_kernel=not self.training,"
SELECTED_DISPATCH_LINE = "use_kernel=self._llmopt_use_kernel(),"
L2NORM_EDITS = (("q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)", "q = inv_scale * (q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6))"),
                ("k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)", "k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)"))
_selected_class = None


def selected_gdn_class():
    """The SelectedGatedDeltaNet class (built once per process from the
    installed mlx-lm source under a guard that REFUSES if the vendor body no
    longer carries the dispatch line or the two l2norm lines)."""
    global _selected_class
    if _selected_class is not None:
        return _selected_class
    from mlx_lm.models import qwen3_5
    src = _vendor_call_source(qwen3_5)      # the body as written in the module file, independent of the in-process rewrite build_metal_model installs
    if DISPATCH_LINE not in src:
        raise RuntimeError("REFUSING: mlx-lm GatedDeltaNet no longer dispatches on use_kernel=not self.training; re-derive the selector")
    for old, new in L2NORM_EDITS:
        if old not in src:
            raise RuntimeError("REFUSING: mlx-lm GatedDeltaNet source changed; re-derive the l2norm override")
        src = src.replace(old, new)
    src = src.replace(DISPATCH_LINE, SELECTED_DISPATCH_LINE)
    ns: dict = {}
    exec(compile(src, "qwen3_5_gdn_selected_recurrence", "exec"), vars(qwen3_5), ns)

    class SelectedGatedDeltaNet(qwen3_5.GatedDeltaNet):
        """A GatedDeltaNet whose recurrence implementation is the instance's
        `recurrence` attribute ("loop" or "kernel"); refuses to run unselected."""
        _source = src

        def _llmopt_use_kernel(self):
            path = getattr(self, "recurrence", None)
            if path not in RECURRENCE_FLAGS:
                raise RuntimeError(f"GatedDeltaNet instance has no recurrence selection (got {path!r}); call llmopt.lab.qtower.select_recurrence")
            return RECURRENCE_FLAGS[path]

    SelectedGatedDeltaNet.__call__ = ns["__call__"]
    _selected_class = SelectedGatedDeltaNet
    return _selected_class


def _vendor_call_source(qwen3_5) -> str:
    """The GatedDeltaNet.__call__ body as written in the installed module
    file (independent of any in-process rewrite of the class attribute)."""
    import ast
    import textwrap
    text = open(qwen3_5.__file__).read()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "GatedDeltaNet":
            for fn in node.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "__call__":
                    lines = text.splitlines()[fn.lineno - 1: fn.end_lineno]
                    return textwrap.dedent("\n".join(lines)) + "\n"
    raise RuntimeError("REFUSING: GatedDeltaNet.__call__ not found in the installed mlx-lm qwen3_5 module")


def select_recurrence(module, path: str) -> dict:
    """Select the recurrence implementation for every GatedDeltaNet inside
    `module` (an mlx nn.Module tree, e.g. the tower returned by
    build_metal_model, or one GatedDeltaNet), by re-classing each one to
    SelectedGatedDeltaNet and setting its `recurrence` attribute. Returns
    {"path", "n_selected", "vendor"} where vendor is the installed mlx-lm
    provenance (llmopt.lab.qgdn.check_vendor: versions, source sha256s,
    Metal availability). Refuses an unknown path, a tree with no
    GatedDeltaNet, and the kernel under a vendor source other than the
    qualified one (QUALIFIED_VENDOR) or without the Metal GPU as the default
    device (the vendor would silently fall back to the loop). Re-selection of
    an already selected instance just changes its attribute; never re-select
    under a live Session whose cache carries recurrent state (the two paths
    agree to about 1e-4 logits, not bit for bit)."""
    if path not in RECURRENCE_PATHS:
        raise ValueError(f"recurrence must be one of {RECURRENCE_PATHS}, got {path!r}")
    from mlx_lm.models import qwen3_5
    from llmopt.lab import qgdn
    vendor = qgdn.check_vendor()
    if path == "kernel":
        for k, want in QUALIFIED_VENDOR.items():
            if vendor.get(k) != want:
                raise RuntimeError(f"REFUSING the kernel recurrence: installed {k} {vendor.get(k)} is not the qualified {want} (mlx-lm {vendor.get('mlx_lm_version')}); select 'loop' or re-qualify")
        if not vendor.get("metal_available") or "gpu" not in str(vendor.get("default_device")):
            raise RuntimeError(f"REFUSING the kernel recurrence: default device {vendor.get('default_device')} is not the Metal GPU (the vendor would run the loop silently)")
    cls = selected_gdn_class()
    targets = [module] if isinstance(module, qwen3_5.GatedDeltaNet) else [m for _, m in module.named_modules() if isinstance(m, qwen3_5.GatedDeltaNet)]
    if not targets:
        raise RuntimeError("no GatedDeltaNet module found to select a recurrence for")
    for m in targets:
        if type(m) is not cls:
            m.__class__ = cls
        m.recurrence = path             # a str: an object attribute, never a parameter (mlx Module stores arrays/containers in its dict)
    return {"path": path, "n_selected": len(targets), "vendor": vendor}


def recurrence_of(module) -> dict:
    """{"loop": n, "kernel": n, "unselected": n} over the GatedDeltaNet
    instances in the tree: what each instance will dispatch to."""
    from mlx_lm.models import qwen3_5
    counts = {"loop": 0, "kernel": 0, "unselected": 0}
    mods = [module] if isinstance(module, qwen3_5.GatedDeltaNet) else [m for _, m in module.named_modules() if isinstance(m, qwen3_5.GatedDeltaNet)]
    for m in mods:
        r = getattr(m, "recurrence", None) if type(m) is not qwen3_5.GatedDeltaNet else None
        counts[r if r in RECURRENCE_FLAGS else "unselected"] += 1
    return counts

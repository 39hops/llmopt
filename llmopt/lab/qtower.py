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

"""End-to-end decode breakdown of the arm C Metal tower by ABLATION, not by
summing synchronized kernel times. Each configuration replaces one family
of modules with a stand-in of negligible cost (a constant output of the
right shape, or the identity), then the full teacher-forced decode step
is timed end to end under the same pipelined execution the real tower
uses. The marginal cost of a family is the end-to-end time of the full
tower minus the time with that family removed; because GPU work overlaps
and the framework overhead is shared, the marginal costs are not expected
to add to the total, and the residual is reported as such. Stand-in
outputs make the model's numbers meaningless; nothing here is a
correctness claim. Stand-ins keep a cheap data dependence on their input:
MLX evaluates lazily and prunes anything the requested logits do not
depend on, so a pure constant would silently remove the upstream tower.

Configurations:
  full              the tower as qualified (OBSERVATION QWEN-METAL-QUAL-0)
  no_w4             every W4 fused linear (the MLP matrices) -> constant
  no_s16_body       every S16 fused linear except lm_head -> constant
  no_head           lm_head -> constant
  no_linears        all fused linears -> constant (framework, attention,
                    recurrence, conv, norms, embedding remain)
  no_deltanet_core  GatedDeltaNet keeps its linears, the conv + recurrence
                    + gating core is replaced by a pass-through of the
                    projected v (isolates the recurrent/conv operators)
  no_attention_core full attention keeps its linears, the SDPA + RoPE core
                    is replaced by a pass-through of v
  no_norms          every RMSNorm -> identity
  skeleton          all of the above removed: embedding, residual adds and
                    framework dispatch only (the floor)
Per configuration: prompt prefill into a fresh cache (not timed), then
STEPS single-token steps, median and p90 of steps 5..STEPS, with each
step forced by mx.eval on its logits (the same discipline as the tower
and qualification drivers). Receipt logs/qmetal/breakdown_<commit>.json
(refuse-if-exists). Env: ART_DIR, VENDOR_DIR, STEPS (32), CONTEXT (126),
OUT.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)


def main() -> None:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models import qwen3_5
    from qwen_metal_qual import build_metal_model, fsha  # noqa: F401  (same tower construction)

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    steps = int(os.environ.get("STEPS", "32"))
    ctx = int(os.environ.get("CONTEXT", "126"))
    out = os.environ.get("OUT", f"logs/qmetal/breakdown_{head}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")

    t0 = time.time()
    model, info = build_metal_model(art, vdir)
    print(f"[breakdown] built {arm}: {info['fused']} fused, {info['raw']} raw, {time.time()-t0:.0f}s", flush=True)
    tm = model.language_model.model
    QLinearT = type(tm.layers[0].mlp.gate_proj)
    import mlx_lm.models.qwen3_5 as q35
    assert QLinearT.__name__ == "QLinear"

    # ---- stand-ins ----
    class ConstLinear(nn.Module):
        """Near-constant output of the linear's shape: no weight stream, no
        GEMV, but the output keeps a (cheap) data dependence on x so that
        MLX's lazy graph cannot prune the upstream computation. A pure
        constant here made the whole tower dead code: a first run of this
        driver measured 5.8 ms for "no_head" because nothing upstream of a
        constant logits array is evaluated."""
        def __init__(self, out_features):
            super().__init__()
            self.out_features = out_features
            self.c = mx.full((out_features,), 1e-3, dtype=mx.float32)

        def __call__(self, x):
            return x[..., :1] * 0.0 + self.c

    class Identity(nn.Module):
        def __call__(self, x, *a, **k):
            return x

    # family membership from the loaded model
    def linears():
        for i, ly in enumerate(tm.layers):
            blk = ly.linear_attn if hasattr(ly, "linear_attn") else ly.self_attn
            for nm in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
                       "q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(blk, nm) and isinstance(getattr(blk, nm), QLinearT):
                    yield (blk, nm, getattr(blk, nm))
            for nm in ("gate_proj", "up_proj", "down_proj"):
                if isinstance(getattr(ly.mlp, nm), QLinearT):
                    yield (ly.mlp, nm, getattr(ly.mlp, nm))
    fam = {"w4": [], "s16_body": []}
    for parent, nm, mod in linears():
        fam[mod.res.codec if mod.res.codec == "w4" else "s16_body"].append((parent, nm, mod))
    head_parent, head_name = model.language_model, "lm_head"
    head_mod = getattr(head_parent, head_name)
    assert isinstance(head_mod, QLinearT)
    counts = {k: len(v) for k, v in fam.items()}
    counts["head"] = 1
    print(f"[breakdown] families: {counts}", flush=True)

    # the DeltaNet and attention cores: swap the class-level __call__ for the
    # duration of a configuration (instances dispatch on the type)
    gdn_call = q35.GatedDeltaNet.__call__
    attn_call = q35.Qwen3NextAttention.__call__ if hasattr(q35, "Qwen3NextAttention") else None
    attn_cls = q35.Qwen3NextAttention if hasattr(q35, "Qwen3NextAttention") else None
    if attn_cls is None:
        for cand in ("Attention", "Qwen3_5Attention"):
            if hasattr(q35, cand):
                attn_cls = getattr(q35, cand)
                attn_call = attn_cls.__call__
                break
    if attn_cls is None:
        raise SystemExit("REFUSING: attention class not found in mlx_lm qwen3_5")

    def gdn_passthrough(self, x, mask=None, cache=None):
        # linears kept (in_proj_qkv, in_proj_z, out_proj), conv + recurrence + gate removed
        B, L, _ = x.shape
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        v = qkv[..., -self.value_dim:] if hasattr(self, "value_dim") else qkv[..., -z.shape[-1]:]
        return self.out_proj(v[..., :z.shape[-1]] * z)

    def attn_passthrough(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        del k
        half = q.shape[-1] // 2
        o = q[..., :half] * 0.0 + mx.broadcast_to(v[..., :1], q[..., :half].shape)
        return self.o_proj(o)

    norm_cls = q35.nn.RMSNorm if hasattr(q35, "nn") else nn.RMSNorm

    def norms():
        for i, ly in enumerate(tm.layers):
            for nm in ("input_layernorm", "post_attention_layernorm"):
                yield (ly, nm, getattr(ly, nm))
            blk = ly.self_attn if hasattr(ly, "self_attn") else None
            if blk is not None:
                for nm in ("q_norm", "k_norm"):
                    if hasattr(blk, nm):
                        yield (blk, nm, getattr(blk, nm))
        yield (tm, "norm", tm.norm)
    norm_list = list(norms())

    # ---- prompt and timing ----
    rng = np.random.default_rng(0)
    prompt = [int(t) for t in rng.integers(1000, 100000, size=ctx)]

    def time_config():
        cache = model.make_cache()
        lp = model(mx.array([prompt]), cache=cache)
        mx.eval(lp)
        cur = 1000
        walls = []
        for s in range(steps):
            t0 = time.perf_counter()
            lz = model(mx.array([[cur]]), cache=cache)[0, -1]
            mx.eval(lz)
            walls.append(time.perf_counter() - t0)
            cur = 1000 + s
        st = walls[4:]
        return {"median_s": float(np.median(st)), "p90_s": float(np.percentile(st, 90)),
                "mean_s": float(np.mean(st)), "steps_timed": len(st)}

    results = {}

    def with_swaps(name, swaps=(), gdn=False, attn=False, nrm=False):
        saved = []
        for parent, nm, mod in swaps:
            saved.append((parent, nm, mod))
            setattr(parent, nm, ConstLinear(mod.out_features))
        saved_norm = []
        if nrm:
            for parent, nm, mod in norm_list:
                saved_norm.append((parent, nm, mod))
                setattr(parent, nm, Identity())
        if gdn:
            q35.GatedDeltaNet.__call__ = gdn_passthrough
        if attn:
            attn_cls.__call__ = attn_passthrough
        try:
            r = time_config()
        finally:
            for parent, nm, mod in saved:
                setattr(parent, nm, mod)
            for parent, nm, mod in saved_norm:
                setattr(parent, nm, mod)
            q35.GatedDeltaNet.__call__ = gdn_call
            attn_cls.__call__ = attn_call
        results[name] = r
        print(f"[breakdown] {name:18s} median {r['median_s']*1e3:6.1f} ms  p90 {r['p90_s']*1e3:6.1f} ms", flush=True)
        return r

    all_lin = fam["w4"] + fam["s16_body"] + [(head_parent, head_name, head_mod)]
    with_swaps("full")
    with_swaps("full_repeat")
    with_swaps("no_w4", fam["w4"])
    with_swaps("no_s16_body", fam["s16_body"])
    with_swaps("no_head", [(head_parent, head_name, head_mod)])
    with_swaps("no_linears", all_lin)
    with_swaps("no_deltanet_core", gdn=True)
    with_swaps("no_attention_core", attn=True)
    with_swaps("no_norms", nrm=True)
    with_swaps("skeleton", all_lin, gdn=True, attn=True, nrm=True)
    with_swaps("full_final")

    full = float(np.median([results["full"]["median_s"], results["full_repeat"]["median_s"], results["full_final"]["median_s"]]))
    marg = {k: full - results[k]["median_s"] for k in
            ("no_w4", "no_s16_body", "no_head", "no_linears", "no_deltanet_core", "no_attention_core", "no_norms")}
    receipt = {"commit": head, "tree_dirty_at_launch": dirty, "mlx_version": mx.__version__,
               "arm": arm, "tower": info, "context": ctx, "steps": steps, "families": counts,
               "configs": results, "full_median_s": full,
               "marginal_s": marg,
               "marginal_fraction_of_full": {k: v / full for k, v in marg.items()},
               "skeleton_s": results["skeleton"]["median_s"],
               "note": "marginal costs are end-to-end deltas under pipelined execution; they need not sum to the "
                       "full time (overlap, shared dispatch). Stand-ins make the outputs meaningless; timing only."}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[breakdown] full {full*1e3:.1f} ms; marginal: " +
          ", ".join(f"{k} {v*1e3:.1f}" for k, v in marg.items()) +
          f"; skeleton {results['skeleton']['median_s']*1e3:.1f} ms; wrote {out}", flush=True)


if __name__ == "__main__":
    main()

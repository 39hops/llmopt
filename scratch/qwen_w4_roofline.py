"""QWEN-W4-PREFILL-ROOFLINE-0: where the first-token wall of the PROMOTED
arm C serving path (generation 5, chunk-256 decode-once GEMM prefill, the
Metal GatedDeltaNet recurrence kernel as the default since VERDICT
QWEN-GDN-PROMOTE-0, final-position head, 2 GiB allocator-cache limit)
goes now that the recurrence is gone, and what any W4 matrix design could
still remove. Assessment and bounded measurement only: the qualified
kernels, the codec, the serving defaults and every frozen receipt are
untouched; every stand-in here computes wrong values by construction and
is a timing instrument in this process only.

Stages (STAGE env; each writes its own receipt and refuses to overwrite):
  decomp  the 1,024-token development prompt of QWEN-PREFILL-WALL-0 at
          chunk 256 on the promoted default (dispatch counted through the
          qgdn instrument in counting mode: the kernel must be the only
          recurrence that executes). (a) MARGINAL replacement as in the
          frozen scratch/qwen_prefill_wall.py decomposition (full minus the
          path with one family replaced by a near-free stand-in that keeps
          a data dependence), order-balanced, with the W4 MLP family split
          two ways that the earlier study could not: no_w4_decode keeps the
          production float32 GEMM but multiplies against pre-decoded slabs
          (two per shape, alternated by matrix so consecutive matmuls never
          share one) instead of decoding each matrix (DECODE REMOVAL AT A
          FIXED float32 GEMM: the decode kernel, its slab write and its
          allocation gone, the matrix work and the slab read intact; this is
          NOT the ceiling of a fused dequant-GEMM, whose tiling can also
          beat the float32 GEMM, which the library reference in stage micro
          measures), and no_w4_matmul
          keeps every production decode and replaces the matmul by a cheap
          dependence on the decoded slab (the matmul's cost gone, the decode
          intact). The DeltaNet stand-ins are installed on the PROMOTED
          class (the selected subclass's own __call__ and the module global
          its compiled body resolves at call time), never on the vendor
          class alone. (b) ADDITIVE synchronous attribution with the W4
          decode timed exclusively inside the W4 linear's frame. (c) one
          instrumented run separating CPU issue from GPU completion per
          chunk, plus memory.
  micro   on the two real W4 shapes of the tower (layer 0 gate_proj
          17408 x 5120 and down_proj 5120 x 17408, the residents the serving
          path runs) at T = 256, 512 and 1,024 activations: the production
          decode kernel alone (single and pipelined), a plain float32 fill of
          the same slab (the write floor; decode minus fill is the index and
          codebook cost), the float32 matmul alone on the pre-decoded slab,
          the production decode-then-matmul (res._gemm_decoded, the serving
          call), the same with the allocator pool cleared before every call
          (allocation cost), float16 and bfloat16 matmuls on pre-cast
          weights (timing and error against the float32 product; numerics
          NOT qualified), and mx.quantized_matmul on an affine 4-bit
          group-128 quantization of the decoded slab as the LIBRARY fused
          dequant-GEMM reference (a different codec, timing only: what a
          fused in-tile dequantization achieves in this framework at these
          shapes). Peak rates measured in the same process: square float32
          and float16 matmuls (achievable GEMM throughput) and a streaming
          add over 1 GiB (achievable bandwidth). The receipt carries the
          shape census of every fused resident (count, bytes, FLOPs per
          256-token chunk) and the roofline arithmetic derived from it and
          the measured rates.
Receipts logs/qwen_w4_roofline/<stage>.json (OUT overrides). SMOKE=1
shortens every stage into logs/qwen_w4_roofline/smoke/.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE = os.environ.get("STAGE", "decomp")
SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT", f"logs/qwen_w4_roofline/{'smoke/' if SMOKE else ''}{STAGE}.json")
CHUNK = 256
DEV_SEED = "prefill-wall-dev-1"        # the development prompt family of QWEN-PREFILL-WALL-0 (never a held-out prompt)
BANDWIDTH_BYTES = 1 << 30


def tm_(fn, n=5, pipelined=1):
    """Median wall of fn over n repeats after one warm call; pipelined > 1
    issues that many calls and evaluates once, reporting the per-call
    time (what a queue of launches costs) next to the single-call time."""
    import mlx.core as mx
    mx.eval(fn())
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        if pipelined == 1:
            y = fn()
            mx.eval(y)
        else:
            ys = [fn() for _ in range(pipelined)]
            mx.eval(*ys)
            y = ys[-1]
        ts.append((time.perf_counter() - t0) / pipelined)
    return statistics.median(ts), y


def provenance(eng):
    from llmopt.lab import qmetal
    import mlx.core as mx
    from scratch.qwen_gdn_promote import model_pin
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    return {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "stage": STAGE, "driver": os.path.relpath(__file__), "cold_load_s": eng.load_s,
            "engine": {"kernel_version": eng.info.get("kernel_version"), "chain_sha256": eng.info.get("chain_sha256"), "recurrence": eng.info.get("recurrence"),
                       "recurrence_layers": eng.info.get("recurrence_layers"), "recurrence_vendor": eng.info.get("recurrence_vendor"),
                       "cache_limit_bytes": eng.cfg.cache_limit_bytes, "chunk": eng.cfg.chunk, "matmul": eng.cfg.matmul,
                       "gemm_min_t_by_codec": dict(qmetal.GEMM_MIN_T_BY_CODEC), "gemm_slab_bytes": qmetal.GEMM_SLAB_BYTES, "default_recurrence_in_code": __import__("llmopt.lab.qserve", fromlist=["x"]).DEFAULT_RECURRENCE},
            "device": {k: v for k, v in mx.device_info().items()}, "mlx_version": mx.__version__,
            "model_pin": model_pin(eng.vendor_dir, eng.art_dir, eng.info)}


def shape_census(model, tokens_per_chunk: int) -> dict:
    """Every fused resident of the tower by codec and shape: count, payload
    bytes, decoded float32 bytes, FLOPs for one chunk of tokens_per_chunk."""
    from llmopt.lab import qmetal
    rows = {}
    for name, mod in model.named_modules():
        res = getattr(mod, "res", None)
        if not isinstance(res, qmetal._Resident):
            continue
        R, C = res.shape
        key = f"{res.codec} [{R}, {C}]"
        r = rows.setdefault(key, {"codec": res.codec, "shape": [R, C], "count": 0, "example": name, "payload_bytes_each": res.payload_bytes(), "gemm_ok_all": True,
                                  "decoded_f32_bytes_each": R * C * 4, "flops_per_chunk_each": 2 * tokens_per_chunk * R * C})
        r["count"] += 1
        r["gemm_ok_all"] = r["gemm_ok_all"] and bool(res.gemm_ok)
    for r in rows.values():
        r["payload_bytes_total"] = r["payload_bytes_each"] * r["count"]
        r["decoded_f32_bytes_total"] = r["decoded_f32_bytes_each"] * r["count"]
        r["flops_per_chunk_total"] = r["flops_per_chunk_each"] * r["count"]
    head = [r for r in rows.values() if r["example"].endswith("lm_head")]
    w4 = [r for r in rows.values() if r["codec"] == "w4"]
    s16 = [r for r in rows.values() if r["codec"] == "s16" and r not in head]     # the body only: the head runs once per prompt on one row (head="final"), fused
    return {"by_shape": rows, "tokens_per_chunk": tokens_per_chunk,
            "head": {"matrices": sum(r["count"] for r in head), "weights": sum(r["count"] * r["shape"][0] * r["shape"][1] for r in head), "payload_bytes": sum(r["payload_bytes_total"] for r in head),
                     "note": "once per prompt on the final row through the fused kernel; excluded from the per-chunk families"},
            "w4": {"matrices": sum(r["count"] for r in w4), "weights": sum(r["count"] * r["shape"][0] * r["shape"][1] for r in w4), "payload_bytes": sum(r["payload_bytes_total"] for r in w4),
                   "decoded_f32_bytes_per_chunk": sum(r["decoded_f32_bytes_total"] for r in w4), "flops_per_chunk": sum(r["flops_per_chunk_total"] for r in w4)},
            "s16": {"matrices": sum(r["count"] for r in s16), "weights": sum(r["count"] * r["shape"][0] * r["shape"][1] for r in s16), "payload_bytes": sum(r["payload_bytes_total"] for r in s16),
                    "decoded_f32_bytes_per_chunk": sum(r["decoded_f32_bytes_total"] for r in s16), "flops_per_chunk": sum(r["flops_per_chunk_total"] for r in s16)}}


# ---------------------------------------------------------------- decomp

def stage_decomp(eng, receipt):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm.models.qwen3_5 as q35
    from llmopt.lab import qgdn, qmetal, qprefill, qtower
    from scratch.qwen_prefill_wall import fixed_prompt, prefill_instrumented, sha, mem
    model = eng.model
    tm = model.language_model.model
    QLinearT = type(tm.layers[0].mlp.gate_proj)
    assert QLinearT.__name__ == "QLinear"
    assert eng.info["recurrence"] == "kernel", eng.info.get("recurrence")
    target = 256 if SMOKE else 1024
    prompt = fixed_prompt(eng, f"{DEV_SEED}-{target}", target)
    receipt["prompt"] = {"tokens": len(prompt), "sha256": sha(prompt), "chunk": CHUNK, "seed": f"{DEV_SEED}-{target}"}
    receipt["census"] = shape_census(model, CHUNK)

    class ConstLinear(nn.Module):       # keeps a cheap data dependence on x so the upstream graph is not pruned
        def __init__(self, out_features):
            super().__init__()
            self.out_features = out_features
            self.c = mx.full((out_features,), 1e-3, dtype=mx.float32)

        def __call__(self, x):
            return x[..., :1] * 0.0 + self.c

    class Identity(nn.Module):
        def __call__(self, x, *a, **k):
            return x

    def linears():
        for ly in tm.layers:
            blk = ly.linear_attn if hasattr(ly, "linear_attn") else ly.self_attn
            for nm in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(blk, nm) and isinstance(getattr(blk, nm), QLinearT):
                    yield (blk, nm, getattr(blk, nm))
            for nm in ("gate_proj", "up_proj", "down_proj"):
                if isinstance(getattr(ly.mlp, nm), QLinearT):
                    yield (ly.mlp, nm, getattr(ly.mlp, nm))
    fam = {"w4_mlp": [], "s16_body": []}
    for parent, nm, mod in linears():
        fam["w4_mlp" if mod.res.codec == "w4" else "s16_body"].append((parent, nm, mod))
    head_mod = model.language_model.lm_head
    assert isinstance(head_mod, QLinearT)
    fam["head"] = [(model.language_model, "lm_head", head_mod)]
    w4_res = [m.res for _, _, m in fam["w4_mlp"]]
    assert all(r.gemm_ok for r in w4_res), "every W4 matrix must take the GEMM path for the split to mean what it says"
    receipt["families"] = {k: len(v) for k, v in fam.items()}
    receipt["families"]["head_codec"] = head_mod.res.codec
    receipt["family_members"] = {k: sorted({n for _, n, _ in v}) for k, v in fam.items()}

    # the promoted DeltaNet class: every linear_attn module must be the selected subclass, and its compiled
    # __call__ must resolve gated_delta_update from the vendor module's globals (so the recurrence stand-in applies)
    gdn_mods = [ly.linear_attn for ly in tm.layers if hasattr(ly, "linear_attn")]
    gdn_cls = type(gdn_mods[0])
    assert all(type(m) is gdn_cls for m in gdn_mods) and gdn_cls is not q35.GatedDeltaNet and issubclass(gdn_cls, q35.GatedDeltaNet), gdn_cls
    assert gdn_cls.__call__.__globals__ is vars(q35), "the selected class's __call__ must resolve names in the vendor module"
    rec = qtower.recurrence_of(model)
    assert rec == {"loop": 0, "kernel": len(gdn_mods), "unselected": 0}, rec
    attn_cls = next(getattr(q35, c) for c in ("Qwen3NextAttention", "Attention", "Qwen3_5Attention") if hasattr(q35, c))
    gdn_call, attn_call = gdn_cls.__call__, attn_cls.__call__
    receipt["promoted_class"] = {"name": gdn_cls.__name__, "n_layers": len(gdn_mods), "vendor_base": q35.GatedDeltaNet.__name__}

    def gdn_passthrough(self, x, mask=None, cache=None):
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        v = qkv[..., -z.shape[-1]:]
        return self.out_proj(v * z)

    def attn_passthrough(self, x, mask=None, cache=None):
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x)
        vh = v.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            cache.update_and_fetch(k, vh)
        half = q.shape[-1] // 2
        o = q[..., :half] * 0.0 + mx.broadcast_to(v[..., :1], q[..., :half].shape)
        return self.o_proj(o)

    gdu_real = q35.gated_delta_update
    standin_calls = [0]       # positive proof that the recurrence stand-in ran (48 layers x chunks per run)

    def gdu_standin(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True):
        standin_calls[0] += 1
        dep = q.sum(axis=(-1, -2), keepdims=True) + k.sum(axis=(-1, -2), keepdims=True) + a.sum(-1)[..., None, None] + b.sum(-1)[..., None, None]
        out = v + 0.0 * dep
        if state is None:
            state = mx.zeros((v.shape[0], v.shape[2], k.shape[-1], v.shape[-1]), dtype=v.dtype)
        return out, state

    def norms():
        for ly in tm.layers:
            for nm in ("input_layernorm", "post_attention_layernorm"):
                yield (ly, nm, getattr(ly, nm))
            blk = getattr(ly, "self_attn", None)
            if blk is not None:
                for nm in ("q_norm", "k_norm"):
                    if hasattr(blk, nm):
                        yield (blk, nm, getattr(blk, nm))
        yield (tm, "norm", tm.norm)
    norm_list = list(norms())

    # W4 split stand-ins on the resident's GEMM driver (the fused path is never taken by these matrices at chunk 256)
    gemm_decoded_real = qmetal._Resident._gemm_decoded
    # two pre-decoded float32 slabs per W4 shape, built BEFORE the warm run (so every timed config sees the same
    # resident memory) and alternated by matrix index, so two consecutive matmuls never read the same slab
    # (gate_proj and up_proj share a shape and run back to back; one shared slab would sit warmer in the cache)
    shared = {}
    slab_index = {}
    for i, (_, _, m) in enumerate(fam["w4_mlp"]):
        key = tuple(m.res.shape)
        if key not in shared:
            shared[key] = [m.res.decode_f32(0, m.res.shape[0]), m.res.decode_f32(0, m.res.shape[0])]
            mx.eval(*shared[key])
        slab_index[id(m.res)] = i % 2

    def gemm_no_decode(self, x2):
        if self.codec != "w4":
            return gemm_decoded_real(self, x2)
        return mx.matmul(x2, shared[tuple(self.shape)][slab_index[id(self)]].T)   # same FLOPs, same slab read, no decode kernel, no slab write

    def gemm_no_matmul(self, x2):
        if self.codec != "w4":
            return gemm_decoded_real(self, x2)
        R, C = self.shape
        slab = self.decode_f32(0, R)                           # the production decode (one slab: every W4 matrix fits GEMM_SLAB_BYTES)
        return x2[:, :1] * 0.0 + slab.reshape(-1)[:R][None, :]  # a view of the decoded slab forces the whole decode; the matmul is gone

    def timed_full():
        mx.clear_cache()
        cache = model.make_cache()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(model, prompt, cache, chunk=CHUNK, head="final", matmul="gemm")
        mx.eval(lg)
        w = time.perf_counter() - t0
        del cache
        return w

    def with_swaps(swaps=(), gdn=False, attn=False, nrm=False, rec=False, w4=None, label="warm"):
        saved = [(p, n, m) for p, n, m in swaps]
        for p, n, m in swaps:
            setattr(p, n, ConstLinear(m.res.shape[0]))
        saved_norm = []
        if nrm:
            for p, n, m in norm_list:
                saved_norm.append((p, n, m))
                setattr(p, n, Identity())
        if gdn:
            gdn_cls.__call__ = gdn_passthrough
        if attn:
            attn_cls.__call__ = attn_passthrough
        if rec:
            q35.gated_delta_update = gdu_standin
        if w4 == "no_decode":
            qmetal._Resident._gemm_decoded = gemm_no_decode
        elif w4 == "no_matmul":
            qmetal._Resident._gemm_decoded = gemm_no_matmul
        try:
            if rec:
                standin_calls[0] = 0
                w = timed_full()             # the recurrence stand-in replaces the name the instrument wraps: this run is not counted by it
                standin_runs.append({"config": label, "standin_calls": standin_calls[0]})
                return w
            with qgdn.recurrence_path("vendor") as sel:
                w = timed_full()
            ex = sel.count.executed
            # a config whose stand-in removes the DeltaNet block (no_gdn_core) runs no recurrence at all: counted with zero calls, never "ops"
            counted.append({"config": label, "executed": dict(ex), "update_calls": sel.count.update_calls, "kernel_only": bool(ex["ops"] == 0 and (ex["kernel"] > 0 or sel.count.update_calls == 0))})
            return w
        finally:
            for p, n, m in saved:
                setattr(p, n, m)
            for p, n, m in saved_norm:
                setattr(p, n, m)
            gdn_cls.__call__ = gdn_call
            attn_cls.__call__ = attn_call
            q35.gated_delta_update = gdu_real
            qmetal._Resident._gemm_decoded = gemm_decoded_real

    configs = {"full": {}, "no_w4_mlp": {"swaps": fam["w4_mlp"]}, "no_w4_decode": {"w4": "no_decode"}, "no_w4_matmul": {"w4": "no_matmul"},
               "no_s16_body": {"swaps": fam["s16_body"]}, "no_head": {"swaps": fam["head"]},
               "no_gdn_core": {"gdn": True}, "no_gdn_recurrence": {"rec": True}, "no_attn_core": {"attn": True}, "no_norms": {"nrm": True},
               "no_all_linears": {"swaps": fam["w4_mlp"] + fam["s16_body"] + fam["head"]}}
    receipt["config_definitions"] = {
        "full": "the serving prefill_chunked call, GEMM scope, chunk 256, final head, on the promoted default",
        "no_w4_mlp": "every W4 linear (gate_proj, up_proj, down_proj) replaced by a constant with a cheap dependence on x",
        "no_w4_decode": "every W4 linear keeps the production float32 GEMM against a pre-decoded slab (two per shape, alternated by matrix index, decoded before the warm run): the decode kernel, its slab write and its allocation are gone, the matmul and its slab read intact (decode removal at a fixed float32 GEMM; not a fused dequant-GEMM ceiling, see the library reference in stage micro)",
        "no_w4_matmul": "every W4 linear runs the production decode kernel and replaces the matmul by a cheap dependence on the decoded slab: the matmul is gone, the decode intact",
        "no_s16_body": "every S16 body linear replaced by a constant", "no_head": "the vocabulary head replaced by a constant",
        "no_gdn_core": "the promoted DeltaNet class's __call__ replaced by in_proj_qkv, in_proj_z and out_proj only (the convolution, the recurrence, the gated norm and the small projections gone)",
        "no_gdn_recurrence": "gated_delta_update replaced by a stand-in carrying the state unchanged (the Metal kernel gone, everything else intact)",
        "no_attn_core": "the attention block minus q, k, v, o: norms, rope, scores and softmax gone", "no_norms": "every RMSNorm replaced by identity",
        "no_all_linears": "every W4, S16 and head linear replaced by a constant"}
    reps = 1 if SMOKE else 2
    order = list(configs) + list(configs)[::-1]
    marg = {k: [] for k in configs}
    counted = []        # one entry per counted run (every config except no_gdn_recurrence, whose stand-in replaces the wrapped name)
    standin_runs = []   # the no_gdn_recurrence runs: how many times the stand-in was called (48 layers x chunks)
    with_swaps()        # warm
    for _ in range(reps):
        for k in order:
            w = with_swaps(**configs[k], label=k)
            marg[k].append(w)
            print(f"[w4roof] decomp marginal {k:18s} {w:.3f}s", flush=True)
    receipt["dispatch_marginal"] = {"selector": "llmopt.lab.qgdn.recurrence_path (counting mode, one block per run)", "runs": counted,
                                    "executed": {"ops": sum(c["executed"]["ops"] for c in counted), "kernel": sum(c["executed"]["kernel"] for c in counted)},
                                    "update_calls": sum(c["update_calls"] for c in counted), "vendor": qgdn.check_vendor(),
                                    "kernel_only": bool(counted) and all(c["kernel_only"] for c in counted) and any(c["executed"]["kernel"] > 0 for c in counted),
                                    "runs_without_recurrence": sorted({c["config"] for c in counted if c["update_calls"] == 0}),
                                    "standin_runs": standin_runs, "standin_calls_expected_per_run": len(gdn_mods) * -(-len(prompt) // CHUNK),
                                    "note": "kernel_only: no run executed the loop (ops) and every run that called the recurrence executed the kernel; the warm run is counted under config 'warm'; the no_gdn_recurrence runs replace gated_delta_update by a counted stand-in and are not seen by the instrument; the no_gdn_core runs remove the block and call no recurrence"}
    full = statistics.median(marg["full"])
    receipt["marginal"] = {k: {"median_s": statistics.median(v), "runs": v, "marginal_s": full - statistics.median(v), "marginal_pct_of_full": 100 * (full - statistics.median(v)) / full}
                           for k, v in marg.items()}
    receipt["marginal_note"] = ("marginal cost = full minus the path with the family replaced; GPU work overlaps and dispatch is shared, so these do NOT add to the full time. "
                                "The no_gdn_recurrence runs executed the stand-in, so the instrument counts fewer kernel executions than full runs; kernel_only reads the whole block.")
    w4m = receipt["marginal"]["no_w4_mlp"]["marginal_s"]
    receipt["w4_split"] = {"w4_marginal_s": w4m, "decode_marginal_s": receipt["marginal"]["no_w4_decode"]["marginal_s"], "matmul_marginal_s": receipt["marginal"]["no_w4_matmul"]["marginal_s"],
                           "decode_share_of_w4": receipt["marginal"]["no_w4_decode"]["marginal_s"] / w4m if w4m else None,
                           "matmul_share_of_w4": receipt["marginal"]["no_w4_matmul"]["marginal_s"] / w4m if w4m else None,
                           "split_sum_over_w4": (receipt["marginal"]["no_w4_decode"]["marginal_s"] + receipt["marginal"]["no_w4_matmul"]["marginal_s"]) / w4m if w4m else None,
                           "shared_slabs": {str(list(k)): [int(a.nbytes) for a in v] for k, v in shared.items()},
                           "note": "decode_marginal is what removing the decode at a fixed float32 GEMM saves in situ (slab write, decode kernel, allocation; the GEMM's slab read stays); matmul_marginal is what the float32 GEMM costs in situ (FLOPs and slab read); their sum over w4 is the overlap check (1.0 = no overlap between the two)"}
    del shared

    # additive synchronous attribution; the W4 decode timed exclusively inside the W4 linear's frame
    acc = {"w4_mlp": 0.0, "w4_decode": 0.0, "s16_body": 0.0, "s16_decode": 0.0, "head": 0.0, "gdn_core": 0.0, "attn_core": 0.0, "norms": 0.0, "embed_other": 0.0}
    stack = []
    fam_of = {id(m): ("w4_mlp" if m.res.codec == "w4" else "s16_body") for lst in (fam["w4_mlp"], fam["s16_body"]) for _, _, m in lst}
    fam_of[id(head_mod)] = "head"
    orig_qlin = QLinearT.__call__
    orig_decode = qmetal._Resident.decode_f32

    def exclusive(name_fn, call):
        def wrapper(self, *a, **k):
            name = name_fn(self)
            ins = [v for v in a if isinstance(v, mx.array)] + [v for v in k.values() if isinstance(v, mx.array)]
            if ins:
                mx.eval(*ins)
            frame = [name, 0.0]
            stack.append(frame)
            t0 = time.perf_counter()
            out = call(self, *a, **k)
            mx.eval(out)
            dt = time.perf_counter() - t0
            stack.pop()
            acc[name] += dt - frame[1]
            if stack:
                stack[-1][1] += dt
            return out
        return wrapper

    norm_cls = type(tm.norm)
    orig_norm = norm_cls.__call__
    QLinearT.__call__ = exclusive(lambda self: fam_of.get(id(self), "embed_other"), orig_qlin)
    qmetal._Resident.decode_f32 = exclusive(lambda self: "w4_decode" if self.codec == "w4" else "s16_decode", orig_decode)
    gdn_cls.__call__ = exclusive(lambda self: "gdn_core", gdn_call)
    attn_cls.__call__ = exclusive(lambda self: "attn_core", attn_call)
    norm_cls.__call__ = exclusive(lambda self: "norms", orig_norm)
    try:
        with qgdn.recurrence_path("vendor") as sel2:      # the additive run keeps the vendor name (only the class calls are wrapped)
            mx.clear_cache()
            cache = model.make_cache()
            t0 = time.perf_counter()
            lg = qprefill.prefill_chunked(model, prompt, cache, chunk=CHUNK, head="final", matmul="gemm")
            mx.eval(lg)
            sync_wall = time.perf_counter() - t0
            del cache
    finally:
        QLinearT.__call__ = orig_qlin
        qmetal._Resident.decode_f32 = orig_decode
        gdn_cls.__call__ = gdn_call
        attn_cls.__call__ = attn_call
        norm_cls.__call__ = orig_norm
    receipt["dispatch_additive"] = sel2.receipt()
    receipt["dispatch_additive"]["kernel_only"] = bool(sel2.count.executed["kernel"] > 0 and sel2.count.executed["ops"] == 0)
    unsync = statistics.median([timed_full() for _ in range(reps)])
    receipt["additive_synchronous"] = {"exclusive_s": acc, "sum_s": sum(acc.values()), "synchronised_wall_s": sync_wall, "unsynchronised_wall_s": unsync,
                                       "untimed_residual_s": sync_wall - sum(acc.values()), "serialisation_penalty_s": sync_wall - unsync,
                                       "w4_matmul_and_rest_s": acc["w4_mlp"], "w4_total_s": acc["w4_mlp"] + acc["w4_decode"], "s16_total_s": acc["s16_body"] + acc["s16_decode"],
                                       "note": "inputs evaluated before each timed call, outputs after; w4_decode and s16_decode are the exclusive times of the decode kernels (slab evaluated inside the linear's frame and subtracted from it, which moves the slab allocation into the decode term and removes any decode-matmul overlap), so w4_mlp and s16_body here are the matmuls plus the linears' reshapes; the untimed residual is the host-side embedding decode, the residual adds, the chunk-end evaluation and anything in no family; the serialisation penalty is what forcing every call costs over the pipelined path"}
    print(f"[w4roof] decomp additive {json.dumps({k: round(v, 3) for k, v in acc.items()})} sum {sum(acc.values()):.2f}s sync {sync_wall:.2f}s unsync {unsync:.2f}s", flush=True)

    # one instrumented run: CPU issue against GPU completion per chunk, plus memory
    mx.clear_cache()
    cache = model.make_cache()
    mx.reset_peak_memory()
    t2 = time.perf_counter()
    lg2, chunks = prefill_instrumented(model, prompt, cache, CHUNK, "gemm")
    mx.eval(lg2)
    wall2 = time.perf_counter() - t2
    receipt["instrumented"] = {"wall_s": wall2, "chunks": chunks, "issue_s_total": sum(c["issue_s"] for c in chunks), "complete_s_total": sum(c["complete_s"] for c in chunks),
                               "gpu_side_s_total": sum(c["complete_s"] - c["issue_s"] for c in chunks), "gpu_side_share": sum(c["complete_s"] - c["issue_s"] for c in chunks) / wall2,
                               "memory": mem(), "note": "issue_s is graph construction plus the host-side embedding decode per chunk; complete_s is measured from the same start"}
    del cache
    mx.clear_cache()


# ---------------------------------------------------------------- micro

def stage_micro(eng, receipt):
    import mlx.core as mx
    from llmopt.lab import qmetal
    model = eng.model
    tm = model.language_model.model
    receipt["census"] = shape_census(model, CHUNK)
    Ts = (64,) if SMOKE else (256, 512, 1024)
    n = 3 if SMOKE else 7

    # achievable rates in this process
    rates = {}
    N = 2048 if SMOKE else 4096
    a32 = mx.random.normal((N, N), dtype=mx.float32, key=mx.random.key(1)); b32 = mx.random.normal((N, N), dtype=mx.float32, key=mx.random.key(2))
    mx.eval(a32, b32)
    t, _ = tm_(lambda: mx.matmul(a32, b32), n)
    rates["matmul_f32_square"] = {"n": N, "s": t, "tflops": 2 * N ** 3 / t / 1e12}
    a16, b16 = a32.astype(mx.float16), b32.astype(mx.float16)
    mx.eval(a16, b16)
    t, _ = tm_(lambda: mx.matmul(a16, b16), n)
    rates["matmul_f16_square"] = {"n": N, "s": t, "tflops": 2 * N ** 3 / t / 1e12}
    ab, bb = a32.astype(mx.bfloat16), b32.astype(mx.bfloat16)
    mx.eval(ab, bb)
    t, _ = tm_(lambda: mx.matmul(ab, bb), n)
    rates["matmul_bf16_square"] = {"n": N, "s": t, "tflops": 2 * N ** 3 / t / 1e12}
    del a32, b32, a16, b16, ab, bb
    nb = BANDWIDTH_BYTES // (8 if SMOKE else 1)
    big = mx.zeros((nb // 4,), dtype=mx.float32)
    mx.eval(big)
    t, _ = tm_(lambda: big + 1.0, n)
    rates["stream_add_1gib"] = {"bytes_read": nb, "bytes_written": nb, "s": t, "gb_per_s": 2 * nb / t / 1e9, "note": "one read plus one write of the same byte count"}
    t, _ = tm_(lambda: mx.full((nb // 4,), 1.0, dtype=mx.float32), n)
    rates["fill_1gib"] = {"bytes_written": nb, "s": t, "gb_per_s": nb / t / 1e9, "note": "write only"}
    del big
    receipt["rates"] = rates
    print(f"[w4roof] rates f32 {rates['matmul_f32_square']['tflops']:.2f} TFLOPS f16 {rates['matmul_f16_square']['tflops']:.2f} bf16 {rates['matmul_bf16_square']['tflops']:.2f} "
          f"stream {rates['stream_add_1gib']['gb_per_s']:.0f} GB/s fill {rates['fill_1gib']['gb_per_s']:.0f} GB/s", flush=True)

    mats = {"layers.0.mlp.gate_proj": tm.layers[0].mlp.gate_proj.res, "layers.0.mlp.down_proj": tm.layers[0].mlp.down_proj.res}
    receipt["matrices"] = {}
    for name, res in mats.items():
        assert res.codec == "w4" and res.gemm_ok
        R, C = res.shape
        row = {"shape": [R, C], "payload_bytes": res.payload_bytes(), "decoded_f32_bytes": R * C * 4, "rows": {}}
        w32 = res.decode_f32(0, R)
        mx.eval(w32)
        # decode alone: single and pipelined; the plain fill of the same slab is the write floor
        t1, _ = tm_(lambda: res.decode_f32(0, R), n)
        t4, _ = tm_(lambda: res.decode_f32(0, R), n, pipelined=4)
        t2, _ = tm_(lambda: res.decode_f32(0, R), n, pipelined=2)
        tf, _ = tm_(lambda: mx.full((R, C), 1.0, dtype=mx.float32), n)
        # allocation: the same fill and decode with the allocator pool cleared before every call (every slab a fresh device allocation)
        tfc, tdc = [], []
        for _ in range(n):
            mx.clear_cache(); t0 = time.perf_counter(); y = mx.full((R, C), 1.0, dtype=mx.float32); mx.eval(y); tfc.append(time.perf_counter() - t0); del y
            mx.clear_cache(); t0 = time.perf_counter(); y = res.decode_f32(0, R); mx.eval(y); tdc.append(time.perf_counter() - t0); del y
        bytes_moved = R * C * 4 + res.payload_bytes()
        row["decode"] = {"single_s": t1, "pipelined2_per_call_s": t2, "pipelined4_per_call_s": t4, "fill_same_slab_s": tf, "fill_cleared_pool_s": statistics.median(tfc), "decode_cleared_pool_s": statistics.median(tdc),
                         "bytes_written": R * C * 4, "bytes_read_payload": res.payload_bytes(),
                         "effective_gb_per_s_single": bytes_moved / t1 / 1e9, "effective_gb_per_s_pipelined4": bytes_moved / t4 / 1e9,
                         "index_and_codebook_over_fill": t1 / tf, "write_floor_share_of_decode": tf / t1, "cleared_over_warm_decode": statistics.median(tdc) / t1,
                         "pipelined4_over_single": t4 / t1,
                         "note": "single: one decode evaluated per call with a warm pool (the slab's buffer reused); pipelined k: k decodes issued then one evaluation, per-call time (k slabs live at once, so buffers beyond the pool are fresh allocations); cleared: the pool emptied before each call (every slab a fresh allocation)"}
        w16 = w32.astype(mx.float16); wb = w32.astype(mx.bfloat16)
        mx.eval(w16, wb)
        row["fp16_weights_exact"] = bool(mx.all(w16.astype(mx.float32) == w32).item())
        row["bf16_weights_exact"] = bool(mx.all(wb.astype(mx.float32) == w32).item())
        wq, sq, bq = mx.quantize(w32, group_size=128, bits=4)
        wq16, sq16, bq16 = mx.quantize(w16, group_size=128, bits=4)
        mx.eval(wq, sq, bq, wq16, sq16, bq16)
        for T in Ts:
            x = mx.random.normal((T, C), dtype=mx.float32, key=mx.random.key(T)) * 0.1
            x16 = x.astype(mx.float16); xb = x.astype(mx.bfloat16)
            mx.eval(x, x16, xb)
            r = {}
            t32, y32 = tm_(lambda: mx.matmul(x, w32.T), n)
            r["matmul_f32_s"] = t32
            r["matmul_f32_tflops"] = 2 * T * R * C / t32 / 1e12
            r["matmul_f32_share_of_peak"] = r["matmul_f32_tflops"] / rates["matmul_f32_square"]["tflops"]
            tprod, yprod = tm_(lambda: res._gemm_decoded(x), n)
            r["production_decode_then_matmul_s"] = tprod
            r["production_equals_matmul_on_slab"] = bool(mx.array_equal(yprod, y32).item())
            r["production_over_sum_of_parts"] = tprod / (t1 + t32)
            # allocation: the pool cleared before every call (every slab a fresh device allocation) against the warm pool above
            ts = []
            for _ in range(n):
                mx.clear_cache()
                t0 = time.perf_counter(); y = res._gemm_decoded(x); mx.eval(y); ts.append(time.perf_counter() - t0)
            r["production_cleared_pool_s"] = statistics.median(ts)
            r["cleared_over_warm_pool"] = r["production_cleared_pool_s"] / tprod
            t16, y16 = tm_(lambda: mx.matmul(x16, w16.T), n)
            tb, yb = tm_(lambda: mx.matmul(xb, wb.T), n)
            r["matmul_f16_s"], r["matmul_bf16_s"] = t16, tb
            r["f16_over_f32"], r["bf16_over_f32"] = t16 / t32, tb / t32
            scale = float(mx.max(mx.abs(y32)).item())
            r["f16_max_abs_err"] = float(mx.max(mx.abs(y32 - y16.astype(mx.float32))).item())
            r["bf16_max_abs_err"] = float(mx.max(mx.abs(y32 - yb.astype(mx.float32))).item())
            r["f16_rel_err"], r["bf16_rel_err"] = r["f16_max_abs_err"] / scale, r["bf16_max_abs_err"] / scale
            tq, _ = tm_(lambda: mx.quantized_matmul(x, wq, sq, bq, transpose=True, group_size=128, bits=4), n)
            tq16, _ = tm_(lambda: mx.quantized_matmul(x16, wq16, sq16, bq16, transpose=True, group_size=128, bits=4), n)
            r["library_qmm_affine4_f32_s"], r["library_qmm_affine4_f16_s"] = tq, tq16
            r["library_qmm_f32_over_production"] = tq / tprod
            r["library_qmm_f16_over_production"] = tq16 / tprod
            r["library_qmm_f32_over_matmul_f32"] = tq / t32
            r["ideal_fused_over_production"] = t32 / tprod
            r["note"] = ("matmuls on a seeded random activation (not a real activation); the library quantized_matmul is an AFFINE 4-bit group-128 codec of the decoded slab, a different "
                         "codec with different numbers, timed as the framework's fused in-tile dequantization reference only; f16/bf16 errors against the float32 product, numerics not qualified")
            row["rows"][str(T)] = r
            print(f"[w4roof] micro {name} T={T} f32 {t32 * 1e3:.1f} ms ({r['matmul_f32_tflops']:.2f} TFLOPS, {r['matmul_f32_share_of_peak']:.2f} of peak) decode {t1 * 1e3:.1f} prod {tprod * 1e3:.1f} "
                  f"cleared {r['production_cleared_pool_s'] * 1e3:.1f} f16 {t16 * 1e3:.1f} bf16 {tb * 1e3:.1f} qmm32 {tq * 1e3:.1f} qmm16 {tq16 * 1e3:.1f} ms", flush=True)
        receipt["matrices"][name] = row
        del w32, w16, wb, wq, sq, bq, wq16, sq16, bq16
        mx.clear_cache()

    # roofline arithmetic from the census and the measured rates (one 256-token chunk, then per token)
    c = receipt["census"]
    f32 = rates["matmul_f32_square"]["tflops"] * 1e12
    bw = rates["stream_add_1gib"]["gb_per_s"] * 1e9
    wbw = rates["fill_1gib"]["gb_per_s"] * 1e9
    w4 = c["w4"]
    receipt["roofline"] = {
        "per_chunk_256": {
            "w4_flops": w4["flops_per_chunk"], "w4_gemm_floor_s_at_peak_f32": w4["flops_per_chunk"] / f32,
            "w4_decoded_slab_bytes_written": w4["decoded_f32_bytes_per_chunk"], "w4_decoded_slab_write_floor_s": w4["decoded_f32_bytes_per_chunk"] / wbw,
            "w4_decoded_slab_bytes_read_by_gemm": w4["decoded_f32_bytes_per_chunk"], "w4_payload_bytes_read": w4["payload_bytes"], "w4_payload_read_floor_s": w4["payload_bytes"] / bw,
            "w4_activation_bytes_read_once": CHUNK * 4 * sum(r["count"] * r["shape"][1] for r in c["by_shape"].values() if r["codec"] == "w4"),
            "s16_flops": c["s16"]["flops_per_chunk"], "s16_gemm_floor_s_at_peak_f32": c["s16"]["flops_per_chunk"] / f32},
        "note": ("floors at the measured achievable rates of this process (square float32 GEMM; streaming add; fill), not vendor peaks; the decoded slab is written by the decode "
                 "kernel and read by the GEMM on every chunk, so a prompt of k chunks moves k times these bytes; a fused dequant-GEMM removes the slab write and read and keeps the "
                 "FLOPs; caching decoded slabs across chunks would hold w4_decoded_slab_bytes_written resident")}
    rl = receipt["roofline"]["per_chunk_256"]
    rl["w4_output_bytes_written"] = CHUNK * 4 * sum(r["count"] * r["shape"][0] for r in c["by_shape"].values() if r["codec"] == "w4")
    rl["s16_body_decoded_slab_bytes_written_and_read"] = c["s16"]["decoded_f32_bytes_per_chunk"]
    rl["s16_body_decoded_slab_write_floor_s"] = c["s16"]["decoded_f32_bytes_per_chunk"] / wbw
    rl["read_terms_note"] = "slab and activation read bytes are ONE-PASS lower bounds; a tiled GEMM re-reads weight tiles once per activation tile and activation tiles once per weight tile"
    rl["decoded_slab_cache_resident_gb_f32"] = w4["decoded_f32_bytes_per_chunk"] / 1e9
    rl["decoded_slab_cache_resident_gb_f16"] = w4["decoded_f32_bytes_per_chunk"] / 2e9
    rl["device_max_working_set_gb"] = mx.device_info()["max_recommended_working_set_size"] / 1e9


def main() -> None:
    from llmopt.lab import qserve
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    if SMOKE and "smoke" not in OUT:
        raise SystemExit(f"REFUSING: SMOKE=1 must write under a smoke path, not {OUT}")
    t_proc = time.perf_counter()
    eng = qserve.Engine()
    receipt = provenance(eng)
    {"decomp": stage_decomp, "micro": stage_micro}[STAGE](eng, receipt)
    receipt["wall_s"] = time.perf_counter() - t_proc
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[w4roof] {STAGE} wrote {OUT} in {receipt['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()

"""Prefill operator study for the arm C Metal tower: why one fused launch
over a tile of 8 activations costs almost eight single-activation
launches, what each stream of the fused kernel costs at tile 8, and what
the realistic alternatives (decode a matrix once, multiply with the
framework GEMM) achieve on the same matrices. TIMING ONLY: the ablation
variants compute wrong values by construction, the dense and framework
alternatives are references for a design decision, nothing here is a
parity claim and no frozen kernel or codec is touched.

Sections (one process, MODE=kernels):
  fused      every distinct fused (codec, shape) of the tower, one
             representative tensor each, through the production dispatch
             (qmetal.VERSION with its per-tile law) at tiles 1, 2, 4, 8:
             synced median latency and pipelined per-launch time (QUEUE
             launches on independent inputs, one eval), plus the
             per-activation cost tile_time / T.
  ablation   the frozen v3 source at TT=8 with one stream removed at a
             time (W4: activation loads, codebook lookups, index stream,
             both weight streams; S16: activation loads, code stream), the
             reduction and write replaced, and an empty kernel: the
             separable costs of activation traffic, decoding, weight
             traffic, synchronisation and dispatch at tile 8.
  alternatives
             for the same matrices and T in {1, 8, 64, 256}: dense float32
             matmul on resident decoded weights (the pure GEMM reference),
             dense float16 matmul (timing reference only, float16 weights
             are not an exact decode), mx.quantized_matmul with affine
             4-bit group-128 weights (the framework's fused quantised GEMM,
             a different codec, timing reference only), a one-pass Metal
             decode of the frozen payload to float32 (one thread per code
             word) with its decode error against the numpy oracle, and
             decode + float32 matmul as one pipelined pair.
             The vocabulary head is studied on a 31,040-row slab (one
             eighth of the 248,320 rows); whole-head figures are the slab
             times eight and are labelled as projections.
  totals     per-chunk (256 tokens) matrix cost of the whole tower under
             the fused tile-8 path and under decode-once + float32 matmul,
             from the per-shape measurements and the manifest counts.
MODE=floor (separate process, loads the tower): a 256-token chunk prefill
with all fused linears replaced by dependence-preserving stand-ins
(the ConstLinear shape of scratch/qwen_metal_breakdown.py), with only the head
replaced, with only the W4 MLP matrices replaced, and the full model, in
rotating order; the non-matrix floor of chunked prefill and the share of
the matrices and the head in a chunk. Env: ART_DIR, VENDOR_DIR, REPEATS
(30), QUEUE (32), CHUNK (256), ROUNDS (3), OUT.
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

HEAD_SLAB_ROWS = 31040          # 248,320 / 8
HEAD_KEY = "lm_head.weight"

# one-pass decode of a payload row block to float32: one thread per code
# word (16 W4 weights or 8 S16 weights), exactly the kernel's value law
_W4_DECODE_SRC = """
    uint gid = thread_position_in_grid.x;            // word index over the whole matrix
    const uint C4 = (uint)C / 4;
    const uint nwords_row = C4 / 4;
    if (gid >= (uint)R * nwords_row) return;
    uint row = gid / nwords_row;
    uint j = gid - row * nwords_row;
    uint word = ((const device uint*)(idx + row * C4))[j];
    uint c0 = j * 16;
    uint e = (uint)exps[row * ((uint)C / 128) + c0 / 128];
    uint cls = q_class(e);
    float s = q_scale(e, cls);
    float rest = cls == 1u ? LO_REST : (cls == 2u ? HI_REST : 1.0f);
    const device half4* cb4 = (const device half4*)cb;
    device float4* o = (device float4*)(out + row * (uint)C + c0);
    o[0] = ((float4)cb4[(word      ) & 0xFFu] * s) * rest;
    o[1] = ((float4)cb4[(word >>  8) & 0xFFu] * s) * rest;
    o[2] = ((float4)cb4[(word >> 16) & 0xFFu] * s) * rest;
    o[3] = ((float4)cb4[(word >> 24) & 0xFFu] * s) * rest;
"""

_S16_DECODE_SRC = """
    uint gid = thread_position_in_grid.x;
    const uint C2 = (uint)C / 2;
    const uint nwords_row = C2 / 4;
    if (gid >= (uint)R * nwords_row) return;
    uint row = gid / nwords_row;
    uint j = gid - row * nwords_row;
    uint word = ((const device uint*)(codes + row * C2))[j];
    uint c0 = j * 8;
    uint e = (uint)exps[row * ((uint)C / 128) + c0 / 128];
    uint cls = q_class(e);
    float s = q_scale(e, cls);
    float rest = cls == 1u ? LO_REST : (cls == 2u ? HI_REST : 1.0f);
    float lvf[16];
    for (uint i = 0; i < 16u; i++) lvf[i] = (float)lv[i];
    uint b0 = word & 0xFFu, b1 = (word >> 8) & 0xFFu, b2 = (word >> 16) & 0xFFu, b3 = (word >> 24) & 0xFFu;
    device float4* o = (device float4*)(out + row * (uint)C + c0);
    o[0] = (float4(lvf[b0 >> 4], lvf[b0 & 0xFu], lvf[b1 >> 4], lvf[b1 & 0xFu]) * s) * rest;
    o[1] = (float4(lvf[b2 >> 4], lvf[b2 & 0xFu], lvf[b3 >> 4], lvf[b3 & 0xFu]) * s) * rest;
"""


def provenance() -> dict:
    import mlx.core as mx
    import mlx_lm
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return {"commit": head, "tree_dirty_at_launch": bool(dirty), "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__, "device": mx.device_info().get("device_name")}


def ablation_variant(codec: str, name: str) -> str:
    """Frozen v3 source with one stream removed (scratch/qmetal_traffic_probe
    variants) plus 'nosum' (the simd reductions replaced by the lane value:
    the reduction and the write stay, the cross-lane synchronisation goes)."""
    from qmetal_traffic_probe import s16_variant, w4_variant
    base = w4_variant if codec == "w4" else s16_variant
    if name == "nosum":
        src = base("v3")
        old = ("        float r = metal::simd_sum(acc[t]);\n"
               "        float rl = metal::simd_sum(lo[t]);\n"
               "        float rh = metal::simd_sum(hi[t]);\n")
        new = ("        float r = acc[t];\n        float rl = lo[t];\n        float rh = hi[t];\n")
        if old not in src:
            raise ValueError("v3 reduction block not found")
        return src.replace(old, new)
    return base(name)


def main() -> None:
    import mlx.core as mx
    from llmopt.lab import qartifact, qmetal
    from llmopt.lab.qcodec import decode_entry
    from llmopt.lab.qcodec_fast import S16Rows, W4Rows

    mode = os.environ.get("MODE", "kernels")
    prov = provenance()
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    repeats = int(os.environ.get("REPEATS", "30"))
    queue = int(os.environ.get("QUEUE", "32"))
    chunk = int(os.environ.get("CHUNK", "256"))
    out = os.environ.get("OUT", f"logs/qmetal/prefill_{mode}_{prov['commit']}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    if mode == "floor":
        floor(prov, art, vdir, chunk, int(os.environ.get("ROUNDS", "3")), out)
        return

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

    # one representative tensor per distinct fused (codec, shape); counts for the totals
    shapes: dict = {}
    for key, e in MAN.items():
        if not isinstance(e, dict) or e["codec"] not in ("w4", "s16"):
            continue
        if key.startswith("model.visual") or ".mtp." in key or key.startswith("mtp.") or key.endswith("embed_tokens.weight"):
            continue
        k = (e["codec"], tuple(e["shape"]))
        shapes.setdefault(k, {"codec": e["codec"], "shape": list(e["shape"]), "count": 0, "example": key})
        shapes[k]["count"] += 1
    print(f"[study] {len(shapes)} distinct fused shapes, {sum(s['count'] for s in shapes.values())} matrices", flush=True)

    def timed_synced(fn):
        for _ in range(4):
            mx.eval(fn())
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts)), float(np.percentile(ts, 90))

    def timed_pipelined(fn_i, q_=None):
        q_ = q_ or queue
        mx.eval(*[fn_i(i) for i in range(min(4, q_))])
        ts = []
        for _ in range(max(5, repeats // 5)):
            t0 = time.perf_counter()
            mx.eval(*[fn_i(i) for i in range(q_)])
            ts.append((time.perf_counter() - t0) / q_)
        return float(np.median(ts))

    kernels: dict = {}

    def kern(codec, name):
        if (codec, name) not in kernels:
            inputs = ["x", "idx", "cb", "exps"] if codec == "w4" else ["x", "codes", "lv", "exps"]
            kernels[(codec, name)] = mx.fast.metal_kernel(
                name=f"study_{codec}_{name}", input_names=inputs, output_names=["out"],
                header=qmetal._HEADER, source=ablation_variant(codec, name))
        return kernels[(codec, name)]

    def decode_kernel(codec):
        if ("dec", codec) not in kernels:
            inputs = ["idx", "cb", "exps"] if codec == "w4" else ["codes", "lv", "exps"]
            kernels[("dec", codec)] = mx.fast.metal_kernel(
                name=f"study_decode_{codec}", input_names=inputs, output_names=["out"],
                header=qmetal._HEADER, source=_W4_DECODE_SRC if codec == "w4" else _S16_DECODE_SRC)
        return kernels[("dec", codec)]

    def decode_once(res):
        R, C = res.shape
        nwords = R * (C // (16 if res.codec == "w4" else 8))
        (o,) = decode_kernel(res.codec)(inputs=list(res._parts()), template=[("C", C), ("R", R)],
                                        grid=(nwords, 1, 1), threadgroup=(256, 1, 1),
                                        output_shapes=[(R, C)], output_dtypes=[mx.float32])
        return o

    rng = np.random.default_rng(0)
    gbs = lambda nbytes, s: nbytes / s / 1e9

    rows_out = []
    for key, info in sorted(shapes.items(), key=lambda kv: -kv[1]["count"]):
        codec, shape = key
        e = MAN[info["example"]]
        buf = payload(e)
        res = qmetal.resident(buf, e)
        R, C = res.shape
        nbytes = res.payload_bytes()
        row = dict(info, payload_bytes=nbytes, fused={}, per_activation_s={})
        xs = {T: [mx.array(rng.normal(size=(T, C)).astype(np.float32)) for _ in range(queue)] for T in (1, 2, 4, 8)}
        for T in (1, 2, 4, 8):
            mx.eval(*xs[T])
            med, p90 = timed_synced(lambda: res.gemm(xs[T][0]))
            pipe = timed_pipelined(lambda i: res.gemm(xs[T][i]))
            row["fused"][T] = {"synced_median_s": med, "synced_p90_s": p90, "pipelined_per_launch_s": pipe,
                               "payload_gbs_pipelined": gbs(nbytes, pipe)}
            row["per_activation_s"][T] = pipe / T
        row["tile8_over_tile1_pipelined"] = row["fused"][8]["pipelined_per_launch_s"] / row["fused"][1]["pipelined_per_launch_s"]
        print(f"[study] fused {codec} {R}x{C} (x{info['count']}): per launch T1 {row['fused'][1]['pipelined_per_launch_s']*1e6:.0f} us, "
              f"T8 {row['fused'][8]['pipelined_per_launch_s']*1e6:.0f} us ({row['tile8_over_tile1_pipelined']:.2f}x), "
              f"{gbs(nbytes, row['fused'][1]['pipelined_per_launch_s']):.0f} GB/s at T1", flush=True)
        rows_out.append(row)
        del xs
    shapes_measured = rows_out

    # ---- ablation at TT=8 on the representative W4 and S16 matrices ----
    ablation = []
    abl_targets = [("model.language_model.layers.0.mlp.gate_proj.weight", "w4", ["v3", "nox", "nocb", "noidx", "xonly", "nosum", "empty"]),
                   ("model.language_model.layers.0.mlp.down_proj.weight", "w4", ["v3", "nox", "nocb", "noidx", "xonly", "nosum", "empty"]),
                   ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", "s16", ["v3", "nox", "nocodes", "nosum", "empty"])]
    for name, codec, variants in abl_targets:
        e = MAN[name]
        res = qmetal.resident(payload(e), e)
        R, C = res.shape
        xs8 = [mx.array(rng.normal(size=(8, C)).astype(np.float32)) for _ in range(queue)]
        xs1 = [mx.array(rng.normal(size=(1, C)).astype(np.float32)) for _ in range(queue)]
        mx.eval(*xs8, *xs1)

        def launch(vname, x, TT):
            (o,) = kern(codec, vname)(inputs=[x, *res._parts()], template=[("C", C), ("R", R), ("TT", TT)],
                                      grid=(R * 32, 1, 1), threadgroup=(32, 1, 1),
                                      output_shapes=[(TT, R)], output_dtypes=[mx.float32])
            return o
        rec = {"tensor": name, "codec": codec, "shape": [R, C], "variants": []}
        for v in variants:
            r8 = timed_pipelined(lambda i, v=v: launch(v, xs8[i], 8))
            r1 = timed_pipelined(lambda i, v=v: launch(v, xs1[i], 1))
            rec["variants"].append({"variant": v, "TT8_pipelined_per_launch_s": r8, "TT1_pipelined_per_launch_s": r1})
            print(f"[study] ablation {codec} {R}x{C} {v}: TT8 {r8*1e6:.0f} us, TT1 {r1*1e6:.0f} us", flush=True)
        ablation.append(rec)
        del xs8, xs1

    # ---- alternatives: dense, framework quantised, decode once ----
    alternatives = []
    alt_targets = [("model.language_model.layers.0.mlp.gate_proj.weight", None),
                   ("model.language_model.layers.0.mlp.down_proj.weight", None),
                   ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", None),
                   ("model.language_model.layers.0.linear_attn.out_proj.weight", None),
                   (HEAD_KEY, HEAD_SLAB_ROWS)]
    Ts = (1, 8, 64, 256)
    for name, slab in alt_targets:
        e = MAN[name]
        buf = payload(e)
        R_full, C = int(e["shape"][0]), int(e["shape"][1])
        if slab is None:
            W = decode_entry(buf, e).astype(np.float32)
            res = qmetal.resident(buf, e)
            R = R_full
            scale_note = None
        else:
            # a bounded slab of the head: decode rows [0, slab) with the row
            # decoder and build a resident over the same rows (the payload
            # layout is block exps | table | codes, each row-major)
            rows = (W4Rows if e["codec"] == "w4" else S16Rows)(buf, e["shape"])
            W = rows.rows(0, slab).astype(np.float32)
            n_full = R_full * C
            nb_full = n_full // 128
            nb = slab * C // 128
            if e["codec"] == "s16":
                hdr = 32
                codes_per = C // 2
                sub = buf[:nb] + buf[nb_full:nb_full + hdr] + buf[nb_full + hdr: nb_full + hdr + slab * codes_per]
            else:
                hdr = 2048
                codes_per = C // 4
                sub = buf[:nb] + buf[nb_full:nb_full + hdr] + buf[nb_full + hdr: nb_full + hdr + slab * codes_per]
            res = qmetal.resident(sub, {"codec": e["codec"], "shape": [slab, C]})
            R = slab
            scale_note = f"slab of {slab} rows out of {R_full}; whole-head figures are slab times {R_full / slab:g}"
        Wm = mx.array(W)
        Wh = Wm.astype(mx.float16)
        wq, sc, bi = mx.quantize(Wm, group_size=128, bits=4)
        mx.eval(Wm, Wh, wq, sc, bi)
        # decode once: exactness against the numpy decode
        Wd = decode_once(res)
        mx.eval(Wd)
        dec_err = float(np.abs(np.asarray(Wd) - W).max())
        dec_ref = float(np.abs(W).max())
        # fused reference value for the same x (sanity that the slab resident is the same rows)
        x_chk = mx.array(rng.normal(size=(8, C)).astype(np.float32))
        y_f = np.asarray(res.gemm(x_chk), dtype=np.float64)
        y_d = np.asarray(mx.matmul(x_chk, Wd.T), dtype=np.float64)
        y_ref = np.asarray(x_chk, dtype=np.float64) @ W.astype(np.float64).T
        rec = {"tensor": name, "codec": e["codec"], "shape": [R, C], "slab_note": scale_note,
               "payload_bytes": res.payload_bytes(), "dense_f32_bytes": int(Wm.nbytes),
               "decode_max_abs_err_vs_numpy": dec_err, "decode_max_abs_ref": dec_ref,
               "fused_vs_oracle_max_rel": float(np.abs(y_f - y_ref).max() / np.abs(y_ref).max()),
               "dense_decoded_vs_oracle_max_rel": float(np.abs(y_d - y_ref).max() / np.abs(y_ref).max()),
               "decode_once_pipelined_s": timed_pipelined(lambda i: decode_once(res), q_=min(8, queue)),
               "by_T": {}}
        del Wd
        for T in Ts:
            xs = [mx.array(rng.normal(size=(T, C)).astype(np.float32)) for _ in range(queue)]
            xh = [x.astype(mx.float16) for x in xs]
            mx.eval(*xs, *xh)
            r = {}
            r["fused_pipelined_s"] = timed_pipelined(lambda i: res.gemm(xs[i]))
            r["dense_f32_matmul_pipelined_s"] = timed_pipelined(lambda i: mx.matmul(xs[i], Wm.T))
            r["dense_f16_matmul_pipelined_s"] = timed_pipelined(lambda i: mx.matmul(xh[i], Wh.T))
            r["framework_affine4_qmm_pipelined_s"] = timed_pipelined(
                lambda i: mx.quantized_matmul(xs[i], wq, sc, bi, transpose=True, group_size=128, bits=4))
            r["decode_once_plus_f32_matmul_pipelined_s"] = timed_pipelined(
                lambda i: mx.matmul(xs[i], decode_once(res).T), q_=min(8, queue))
            r["flops"] = 2.0 * R * C * T
            r["dense_f32_tflops"] = r["flops"] / r["dense_f32_matmul_pipelined_s"] / 1e12
            rec["by_T"][T] = r
            print(f"[study] alt {name.split('.')[-2]} {R}x{C} T{T}: fused {r['fused_pipelined_s']*1e6:.0f} us, "
                  f"f32 mm {r['dense_f32_matmul_pipelined_s']*1e6:.0f}, f16 mm {r['dense_f16_matmul_pipelined_s']*1e6:.0f}, "
                  f"qmm4 {r['framework_affine4_qmm_pipelined_s']*1e6:.0f}, decode+mm {r['decode_once_plus_f32_matmul_pipelined_s']*1e6:.0f} "
                  f"(decode alone {rec['decode_once_pipelined_s']*1e6:.0f})", flush=True)
            del xs, xh
        alternatives.append(rec)
        del Wm, Wh, wq, sc, bi, W, res
        mx.clear_cache()

    # ---- totals per 256-token chunk from the per-shape measurements ----
    # fused: launches per chunk = chunk / 8 at tile 8 (chunk a multiple of 8)
    launches = chunk // 8
    fused_chunk = 0.0
    for row in shapes_measured:
        fused_chunk += row["count"] * launches * row["fused"][8]["pipelined_per_launch_s"]
    # decode-once alternative needs a 256-token matmul per shape; measured on
    # the representative matrices only, so the total is projected by payload
    # bytes from the closest measured matrix of the same codec
    def nearest_alt(codec, nbytes):
        cands = [a for a in alternatives if a["codec"] == codec and a["slab_note"] is None]
        return min(cands, key=lambda a: abs(a["payload_bytes"] - nbytes))
    alt_chunk = 0.0
    alt_detail = []
    for row in shapes_measured:
        a = nearest_alt(row["codec"], row["payload_bytes"])
        f = row["payload_bytes"] / a["payload_bytes"]
        per = f * (a["decode_once_pipelined_s"] + a["by_T"][256]["dense_f32_matmul_pipelined_s"])
        alt_chunk += row["count"] * per
        alt_detail.append({"shape": row["shape"], "codec": row["codec"], "count": row["count"],
                           "scaled_from": a["tensor"], "scale": f, "decode_plus_mm256_s": per,
                           "fused_chunk_s": launches * row["fused"][8]["pipelined_per_launch_s"]})
    head = next(a for a in alternatives if a["tensor"] == HEAD_KEY)
    head_shape = next(r for r in shapes_measured if r["example"] == HEAD_KEY or r["shape"] == [248320, 5120])
    totals = {"chunk": chunk, "launches_per_matrix": launches,
              "fused_tile8_matrix_time_per_chunk_s": fused_chunk,
              "fused_tile8_matrix_time_per_token_s": fused_chunk / chunk,
              "decode_once_f32_matmul_time_per_chunk_s_projected": alt_chunk,
              "decode_once_f32_matmul_time_per_token_s_projected": alt_chunk / chunk,
              "head_fused_tile8_per_chunk_s": launches * head_shape["fused"][8]["pipelined_per_launch_s"],
              "head_slab_decode_plus_mm256_s_times8_projected": 8 * (head["decode_once_pipelined_s"] + head["by_T"][256]["dense_f32_matmul_pipelined_s"]),
              "detail": alt_detail,
              "note": "fused figures are per-launch pipelined times of each distinct shape times its count and the launches "
                      "per chunk; the decode-once figures scale the measured representative of each codec by payload bytes "
                      "and are projections, not measurements of every matrix"}
    print(f"[study] totals per {chunk}-token chunk: fused tile-8 matrices {fused_chunk:.2f} s ({fused_chunk/chunk*1e3:.1f} ms/token); "
          f"decode-once + f32 matmul projected {alt_chunk:.2f} s ({alt_chunk/chunk*1e3:.1f} ms/token)", flush=True)
    receipt = {"provenance": prov, "arm": arm, "chain_report": q["report"].get("rung0", q["report"]),
               "repeats": repeats, "queue": queue, "kernel_version": qmetal.VERSION,
               "note": "TIMING ONLY; ablation variants compute wrong values by construction; dense, float16 and framework "
                       "affine-4 figures are references for a design decision, not parity claims",
               "shapes": shapes_measured, "ablation_TT8": ablation, "alternatives": alternatives, "totals": totals}
    json.dump(receipt, open(out, "w"), indent=1)
    print(f"[study] wrote {out}", flush=True)


def floor(prov, art, vdir, chunk, rounds, out):
    import mlx.core as mx
    import mlx.nn as nn
    from qwen_metal_qual import build_metal_model
    from qwen_metal_chunked import cache_arrays
    model, info = build_metal_model(art, vdir)
    lm = model.language_model
    # fused linears by path
    fused = [(n, m) for n, m in model.named_modules() if type(m).__name__ == "QLinear"]
    w4 = [(n, m) for n, m in fused if m.res.codec == "w4"]
    headp = [(n, m) for n, m in fused if n.endswith("lm_head")]
    print(f"[floor] {len(fused)} fused linears, {len(w4)} w4, head {len(headp)}", flush=True)

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

    class Const(nn.Module):
        def __init__(self, out_features):
            super().__init__()
            self.c = mx.full((out_features,), 1e-3, dtype=mx.float32)

        def __call__(self, x):
            return x[..., :1] * 0.0 + self.c

    configs = {"full": [], "no_head": headp, "no_w4_mlp": w4, "no_fused": fused}
    ids = [int(t) for t in np.random.default_rng(1).integers(1000, 100000, size=chunk)]
    results = {k: [] for k in configs}
    for r in range(rounds):
        for name, swap in configs.items():
            for n, m in swap:
                set_path(model, n, Const(m.out_features))
            cache = model.make_cache()
            t0 = time.perf_counter()
            lg = model(mx.array([ids]), cache=cache)
            mx.eval(lg, *cache_arrays(cache))
            t = time.perf_counter() - t0
            for n, m in swap:
                set_path(model, n, m)
            results[name].append(t)
            print(f"[floor] round {r} {name}: {t:.2f} s for {chunk} tokens", flush=True)
            del cache, lg
            mx.clear_cache()
    med = {k: float(np.median(v)) for k, v in results.items()}
    rec = {"provenance": prov, "tower": {k: v for k, v in info.items() if k != "chain_report"}, "chunk": chunk,
           "rounds": rounds, "configs": {k: {"n_replaced": len(v)} for k, v in configs.items()},
           "walls_s": results, "median_s": med,
           "matrix_share_s": med["full"] - med["no_fused"], "head_share_s": med["full"] - med["no_head"],
           "w4_mlp_share_s": med["full"] - med["no_w4_mlp"],
           "note": "stand-ins keep a data dependence on their input so the upstream graph is evaluated; "
                   "the replaced path still runs the embedding, norms, DeltaNet conv and recurrence and attention cores"}
    json.dump(rec, open(out, "w"), indent=1)
    print(f"[floor] medians {med}; matrices {rec['matrix_share_s']:.2f} s, head {rec['head_share_s']:.2f} s, "
          f"w4 mlp {rec['w4_mlp_share_s']:.2f} s of {med['full']:.2f} s; wrote {out}", flush=True)


if __name__ == "__main__":
    main()

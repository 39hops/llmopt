"""Benchmark the fused Metal decode-and-multiply kernels on real arm C
payloads (representative Qwen3.8-27B layer shapes) against the existing
CPU path (canonical decode + numpy GEMV) and, as a reference only,
mx.quantized_matmul int4 (a DIFFERENT 4-bit format with a different
byte count; shown because it is the tuned Metal GEMV on this machine).

Every timed iteration is forced with mx.eval. Warmup and first-call
compilation are reported separately and excluded from steady state.
Effective bandwidth = payload bytes / time, against the 150 GB/s
roofline of the M3 Pro (a bound, not a target). Peak memory from
mx.get_peak_memory after a reset. Writes a JSON receipt (refuse-if-exists).

Usage: .venv/bin/python scripts/bench_qmetal.py [--art ~/qwen_whole0t/C]
       [--out logs/qmetal/bench_<commit>.json] [--repeats 50]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHAPES = [  # (tensor suffix, layer index) representative of the 64-layer tower
    ("linear_attn.in_proj_qkv.weight", 0),
    ("linear_attn.in_proj_z.weight", 0),
    ("linear_attn.out_proj.weight", 0),
    ("mlp.gate_proj.weight", 0),
    ("mlp.up_proj.weight", 0),
    ("mlp.down_proj.weight", 0),
    ("self_attn.q_proj.weight", 3),
    ("self_attn.k_proj.weight", 3),
    ("self_attn.o_proj.weight", 3),
]
ROOFLINE_GBS = 150.0



def provenance() -> dict:
    """Code and library identity of the run: short HEAD, whether the tree was
    dirty (a dirty tree means HEAD may lack the code that wrote the
    receipt), mlx and mlx-lm versions. Call it at launch, before any
    model code loads, so later edits cannot change what it records."""
    import mlx.core as mx
    import mlx_lm
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True,
                           text=True).stdout.strip()
    return {"commit": head, "tree_dirty": bool(dirty), "mlx_version": mx.__version__,
            "mlx_lm_version": mlx_lm.__version__}

def main() -> None:
    import mlx.core as mx
    from llmopt.lab import qmetal
    from llmopt.lab.qcodec import decode_entry

    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default=os.path.expanduser("~/qwen_whole0t/C"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--head", action="store_true", help="also time lm_head [248320, 5120]")
    a = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    prov = provenance()
    out = a.out or f"logs/qmetal/bench_{commit}.json"
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    MAN = json.load(open(os.path.join(a.art, "manifest.json")))
    handles: dict = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(a.art, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    def timed(fn, repeats):
        mx.eval(fn())                      # warmup (includes compile on first call)
        t0 = time.perf_counter()
        mx.eval(fn())
        first = time.perf_counter() - t0   # a second warm call, reported separately
        mx.synchronize()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mx.eval(fn())
            times.append(time.perf_counter() - t0)
        return first, float(np.median(times)), float(np.percentile(times, 90))

    names = [(f"model.language_model.layers.{li}.{suf}", suf, li) for suf, li in SHAPES]
    if a.head:
        names.append(("lm_head.weight", "lm_head", -1))
    rows = []
    for full, suf, li in names:
        e = MAN[full]
        buf = payload(e)
        R, C = e["shape"]
        x = np.random.default_rng(0).normal(size=C).astype(np.float32)
        xm = mx.array(x)
        # CPU path: canonical decode + GEMV (what the census runtime pays per token)
        t0 = time.perf_counter(); W = decode_entry(buf, e); t_dec = time.perf_counter() - t0
        t0 = time.perf_counter(); y_cpu = W @ x; t_mm = time.perf_counter() - t0
        mx.reset_peak_memory()
        t0 = time.perf_counter(); res = qmetal.resident(buf, e); t_up = time.perf_counter() - t0
        comp0 = time.perf_counter()
        y = res.gemv(xm); mx.eval(y)
        t_first_call = time.perf_counter() - comp0
        def gemv_call(res=res, xm=xm):
            return res.gemv(xm)
        first, med, p90 = timed(gemv_call, a.repeats)
        peak = mx.get_peak_memory()
        rel = float(np.abs(np.asarray(y) - y_cpu).max() / (np.abs(y_cpu).max() + 1e-30))
        nbytes = res.payload_bytes()
        # reference: tuned int4 GEMV of a same-shape random matrix (different format)
        Wr = mx.random.normal((R, C)).astype(mx.float16)
        wq, sc, bi = mx.quantize(Wr, group_size=128, bits=4)
        xh = xm.astype(mx.float16)
        def q4_call(wq=wq, sc=sc, bi=bi, xh=xh):
            return mx.quantized_matmul(xh, wq, sc, bi, transpose=True, group_size=128, bits=4)
        _, med_q4, _ = timed(q4_call, a.repeats)
        q4_bytes = int(wq.nbytes + sc.nbytes + bi.nbytes)
        # multi-token (prefill) at T=8 and T=64
        X8 = mx.random.normal((8, C)).astype(mx.float32)
        X64 = mx.random.normal((64, C)).astype(mx.float32)
        def g8(res=res, X8=X8):
            return res.gemm(X8)

        def g64(res=res, X64=X64):
            return res.gemm(X64)
        _, med8, _ = timed(g8, max(5, a.repeats // 5))
        _, med64, _ = timed(g64, max(3, a.repeats // 10))
        row = {"tensor": full, "layer": li, "codec": e["codec"], "shape": [R, C], "payload_bytes": nbytes,
               "cpu_decode_s": t_dec, "cpu_gemv_s": t_mm, "upload_s": t_up,
               "first_call_s": t_first_call, "second_call_s": first,
               "fused_gemv_median_s": med, "fused_gemv_p90_s": p90,
               "fused_eff_gbs": nbytes / med / 1e9, "roofline_frac": nbytes / med / 1e9 / ROOFLINE_GBS,
               "gemm_T8_median_s": med8, "gemm_T64_median_s": med64,
               "per_token_at_T8_s": med8 / 8, "per_token_at_T64_s": med64 / 64,
               "mx_q4_same_shape_median_s": med_q4, "mx_q4_bytes": q4_bytes,
               "max_abs_rel_vs_cpu": rel, "peak_mem_bytes": int(peak)}
        rows.append(row)
        print(f"{suf:30s} {e['codec']} {str([R, C]):16s} cpu dec+gemv {t_dec*1e3:7.1f}+{t_mm*1e3:5.1f} ms | fused {med*1e6:8.0f} us "
              f"({nbytes/med/1e9:5.1f} GB/s, {100*nbytes/med/1e9/ROOFLINE_GBS:4.1f}% roof) | mx q4 {med_q4*1e6:7.0f} us | "
              f"T8 {med8/8*1e6:6.0f} us/tok T64 {med64/64*1e6:6.0f} us/tok | rel {rel:.1e} | peak {peak/1e6:.0f} MB", flush=True)
    lin = [r for r in rows if r["layer"] == 0]
    full_ = [r for r in rows if r["layer"] == 3]
    mlp = [r for r in lin if "mlp" in r["tensor"]]
    per_linear = sum(r["fused_gemv_median_s"] for r in lin)
    per_full = sum(r["fused_gemv_median_s"] for r in full_) + sum(r["fused_gemv_median_s"] for r in mlp)
    proj = 48 * per_linear + 16 * per_full
    summary = {"commit": commit, "provenance": prov, "kernel_version": qmetal.VERSION,
               "timing_note": "fused_gemv_median_s is synced per-call latency (host dispatch, kernel, mx.eval sync), "
                              "an upper bound on kernel execution; the tower pipelines launches and runs below the sum of these",
               "art": a.art, "repeats": a.repeats, "roofline_gbs": ROOFLINE_GBS,
               "rows": rows,
               "projection_labelled": {"note": "PROJECTION from per-layer fused GEMV medians only; excludes attention, norms, "
                                               "DeltaNet recurrence, embedding/head, launch gaps; not a measured token rate",
                                       "linear_layer_matmul_s": per_linear, "full_layer_matmul_s": per_full,
                                       "tower_matmul_s_per_token": proj,
                                       "cpu_tower_decode_s_per_token": 48 * sum(r["cpu_decode_s"] for r in lin) + 16 * (sum(r["cpu_decode_s"] for r in full_) + sum(r["cpu_decode_s"] for r in mlp))}}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(summary, open(out, "w"), indent=1)
    p = summary["projection_labelled"]
    print(f"\nPROJECTION (matmuls only): linear layer {p['linear_layer_matmul_s']*1e3:.1f} ms, full layer {p['full_layer_matmul_s']*1e3:.1f} ms, "
          f"tower {p['tower_matmul_s_per_token']*1e3:.0f} ms/token vs CPU decode {p['cpu_tower_decode_s_per_token']:.0f} s/token; wrote {out}")


if __name__ == "__main__":
    main()

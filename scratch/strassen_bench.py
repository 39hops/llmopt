"""Bounded test of a small-block bilinear algorithm against the framework
GEMM on the prefill shapes (Phase D of the 2026-10-19 round).

The 9/4 exponent result is asymptotic and its proof names no competitive
finite size, so the only finite bilinear algorithm with a known practical
crossover is Strassen's 7-product 2 x 2 block scheme (Winograd variant: 7
block products, 15 block additions). This driver measures it, one and two
recursion levels, against the plain product on (a) square matrices where
CPU libraries are known to show a crossover and (b) the tower's prefill
shapes: a 256-row activation tile against the gate (17408 x 5120), the
down projection (5120 x 17408) and the head (248320 x 5120).

Per shape and arm: wall (median of ROUNDS, every result evaluated), the
multiplication count ratio (7/8 per level), the extra temporaries
(bytes of the block sums and partial products alive at once), and the
numerical error against a float64 reference on the same inputs
(max abs over max abs of the reference, and relative Frobenius).
CPU oracle: numpy float64 Strassen equals the float64 product to 1e-12
relative on every shape before any timing (refused otherwise).
Backends: numpy float32 (Accelerate) and MLX float32 on the GPU.
Env: ROUNDS (5), OUT (logs/qmetal/strassen/bench_<commit>.json), SMOKE=1
(small shapes, OUT under logs/qmetal/strassen_smoke/). Refuses to overwrite.
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

SMOKE = os.environ.get("SMOKE") == "1"
ROUNDS = int(os.environ.get("ROUNDS", "5"))


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty():
    return bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())


def effective_levels(m, k, n, levels):
    """Recursion depth the scheme reaches: it stops when a dimension is odd or below 64."""
    d = 0
    while d < levels and min(m, k, n) % 2 == 0 and min(m, k, n) >= 64:
        m, k, n, d = m // 2, k // 2, n // 2, d + 1
    return d


def split2(A, lib):
    m, n = A.shape
    return A[: m // 2, : n // 2], A[: m // 2, n // 2:], A[m // 2:, : n // 2], A[m // 2:, n // 2:]


def strassen(A, B, lib, levels):
    """A[m, k] @ B[k, n] with Winograd's 7-product scheme, recursing `levels`
    times; lib is numpy or mlx.core (both expose matmul and concatenate)."""
    if levels == 0 or min(A.shape[0], A.shape[1], B.shape[1]) % 2 or min(A.shape[0], A.shape[1], B.shape[1]) < 64:
        return lib.matmul(A, B)
    A11, A12, A21, A22 = split2(A, lib)
    B11, B12, B21, B22 = split2(B, lib)
    S1 = A21 + A22
    S2 = S1 - A11
    S3 = A11 - A21
    S4 = A12 - S2
    T1 = B12 - B11
    T2 = B22 - T1
    T3 = B22 - B12
    T4 = T2 - B21
    P1 = strassen(A11, B11, lib, levels - 1)
    P2 = strassen(A12, B21, lib, levels - 1)
    P3 = strassen(S4, B22, lib, levels - 1)
    P4 = strassen(A22, T4, lib, levels - 1)
    P5 = strassen(S1, T1, lib, levels - 1)
    P6 = strassen(S2, T2, lib, levels - 1)
    P7 = strassen(S3, T3, lib, levels - 1)
    U1 = P1 + P2
    U2 = P1 + P6
    U3 = U2 + P7
    U4 = U2 + P5
    U5 = U4 + P3
    U6 = U3 - P4
    U7 = U3 + P5
    top = lib.concatenate([U1, U5], axis=1)
    bot = lib.concatenate([U6, U7], axis=1)
    return lib.concatenate([top, bot], axis=0)


def temporaries_bytes_model(m, k, n, levels, itemsize=4):
    """A counting model, not a measurement: block sums (4 of A's quarter, 4
    of B's quarter), 7 quarter products, the 7 combination sums and the
    three concatenation outputs of one level (no reuse), recursively."""
    if levels == 0:
        return 0
    a_q, b_q, c_q = (m // 2) * (k // 2), (k // 2) * (n // 2), (m // 2) * (n // 2)
    return itemsize * (4 * a_q + 4 * b_q + 14 * c_q + 2 * (2 * c_q) + m * n) + temporaries_bytes_model(m // 2, k // 2, n // 2, levels - 1, itemsize)


def mult_ratio(levels):
    return (7 / 8) ** levels


def time_fn(fn, evaluate, rounds):
    evaluate(fn())          # one untimed warm-up (kernel selection and allocation)
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        y = fn()
        evaluate(y)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), y


def main():
    import mlx.core as mx
    out_dir = "logs/qmetal/strassen_smoke" if SMOKE else "logs/qmetal/strassen"
    out = os.path.join(out_dir, f"bench_{git_head()}.json") if SMOKE else os.environ.get("OUT", os.path.join(out_dir, f"bench_{git_head()}.json"))
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rng = np.random.default_rng(1)
    shapes = ([(256, 512, 1024), (512, 512, 512)] if SMOKE else
              [(2048, 2048, 2048), (4096, 4096, 4096), (256, 5120, 17408), (256, 17408, 5120), (256, 5120, 248320),
               (1024, 5120, 17408), (4096, 5120, 17408)])
    # ---- CPU oracle: float64 Strassen equals the float64 product ----
    oracle = []
    for (m, k, n) in ([(64, 128, 256), (128, 128, 128)] if SMOKE else [(256, 512, 1024), (512, 512, 512), (256, 5120, 2048)]):
        A, B = rng.standard_normal((m, k)), rng.standard_normal((k, n))
        ref = A @ B
        for lv in (1, 2):
            y = strassen(A, B, np, lv)
            rel = float(np.linalg.norm(y - ref) / np.linalg.norm(ref))
            oracle.append({"shape": [m, k, n], "levels": lv, "effective_levels": effective_levels(m, k, n, lv), "rel_fro_vs_f64_product": rel, "pass": rel < 1e-12})
    if not all(o["pass"] for o in oracle):
        raise SystemExit(f"REFUSING: float64 Strassen oracle failed: {oracle}")
    rec = {"code_commit": git_head(), "tree_dirty": git_dirty(), "smoke": SMOKE, "rounds": ROUNDS, "mlx_version": mx.__version__,
           "device": str(mx.default_device()), "oracle_f64": oracle, "shapes": []}
    for (m, k, n) in shapes:
        A32 = rng.standard_normal((m, k)).astype(np.float32)
        B32 = rng.standard_normal((k, n)).astype(np.float32)
        ref64 = A32.astype(np.float64) @ B32.astype(np.float64)
        ref_scale = float(np.abs(ref64).max())
        row = {"shape": [m, k, n], "flops_direct": 2.0 * m * k * n, "arms": {}}
        Am, Bm = mx.array(A32), mx.array(B32)
        mx.eval(Am, Bm)
        for lib_name, lib, A, B, ev in (("numpy_f32", np, A32, B32, lambda y: None), ("mlx_f32", mx, Am, Bm, lambda y: mx.eval(y))):
            if lib_name == "numpy_f32" and not SMOKE and m * k * n > 2048 ** 3 * 4:
                continue          # the big rectangular products take minutes on CPU; MLX is the comparison of interest
            for lv in (0, 1, 2):
                if lv and (min(m, k, n) % (2 ** lv)):
                    continue
                if lib_name == "mlx_f32":
                    mx.reset_peak_memory()
                t, y = time_fn(lambda: strassen(A, B, lib, lv) if lv else lib.matmul(A, B), ev, ROUNDS)
                peak = int(mx.get_peak_memory()) if lib_name == "mlx_f32" else None
                y64 = np.asarray(y, dtype=np.float64)
                err = {"max_abs_over_ref_max": float(np.abs(y64 - ref64).max() / ref_scale),
                       "rel_fro": float(np.linalg.norm(y64 - ref64) / np.linalg.norm(ref64))}
                el = effective_levels(m, k, n, lv)
                row["arms"][f"{lib_name}_levels{lv}"] = {
                    "wall_s": t, "tflops_direct_equivalent": 2.0 * m * k * n / t / 1e12, "levels_requested": lv,
                    "effective_levels": el, "mult_ratio": mult_ratio(el),
                    "temporaries_bytes_model": temporaries_bytes_model(m, k, n, el), "mlx_peak_active_bytes": peak, "error": err}
                print(f"[strassen] {m}x{k}x{n} {lib_name} levels {lv} (effective {el}): {t * 1e3:.1f} ms, err max {err['max_abs_over_ref_max']:.2e} fro {err['rel_fro']:.2e}, "
                      f"temps model {temporaries_bytes_model(m, k, n, el) / 1e6:.0f} MB" + (f", mlx peak {peak / 1e6:.0f} MB" if peak else ""), flush=True)
            del y
        for lib_name in ("numpy_f32", "mlx_f32"):
            base = row["arms"].get(f"{lib_name}_levels0")
            for lv in (1, 2):
                a = row["arms"].get(f"{lib_name}_levels{lv}")
                if base and a:
                    a["speedup_vs_direct"] = base["wall_s"] / a["wall_s"]
                    a["error_ratio_vs_direct"] = a["error"]["rel_fro"] / base["error"]["rel_fro"]
        rec["shapes"].append(row)
        del Am, Bm, ref64
    json.dump(rec, open(out, "w"), indent=1)
    print("[strassen] wrote", out)


if __name__ == "__main__":
    main()

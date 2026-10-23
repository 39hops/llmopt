"""QWEN-LOSSY-GEOMETRY-0, stage 3: what the input rotation costs on Apple
Metal as it would be placed in the serving decode step, measured, not
assumed. The rotation x' = H_blk (s * x) is a sign flip and a block
Walsh-Hadamard transform of the activation vector right before a fused
linear launch; it is exact in the reals (scratch/leancheck/InputRotation.lean)
and this driver measures its float32 wall on this Mac as MLX operations
(no fused Metal kernel is built in this phase): block sizes 128 and
1024, two implementations (a batched matmul against the dense block
Hadamard, and the butterfly), one token (T = 1) and the prefill tile (T
= 256), synced median of REPEATS and pipelined per-call time over QUEUE
queued calls. Then the step-level projection: one transform per DISTINCT
input vector per layer when every linear map is rotated (gate and up
share theirs; in_proj_qkv, in_proj_z, in_proj_a, in_proj_b share theirs;
q, k, v share theirs; out_proj, o_proj and down_proj have their own), 4
per layer, 256 per decode step, against the measured 100.2 ms step and a
240 us W4 launch; and the metadata: one sign bit per input channel per
rotated matrix (shareable by matrices with the same input) and nothing
else (the Hadamard is implicit). Receipt
logs/qwen_geometry/transform.json (refuse-if-exists). Env: REPEATS (50),
QUEUE (32), OUT.
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
OUT = os.environ.get("OUT", "logs/qwen_geometry/transform.json")
REPEATS = int(os.environ.get("REPEATS", "50"))
QUEUE = int(os.environ.get("QUEUE", "32"))
STEP_MS = 100.2          # OBSERVATION QWEN-METAL-DECODE-BREAKDOWN-2
W4_LAUNCH_US = 240.0     # OBSERVATION QWEN-METAL-W4-T1-0, shipped kernel pipelined
LAYERS = 64
TRANSFORMS_PER_LAYER = 4


def main() -> None:
    import mlx.core as mx
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")

    def dense(b):
        H = np.array([[1.0]])
        while H.shape[0] < b:
            H = np.block([[H, H], [H, -H]])
        return mx.array((H / np.sqrt(b)).astype(np.float32))

    def via_matmul(x, s, Hm, b):
        T, C = x.shape
        return ((x * s).reshape(T, C // b, b) @ Hm).reshape(T, C)

    def via_butterfly(x, s, b):
        T, C = x.shape
        y = (x * s).reshape(T, C // b, b)
        h = 1
        while h < b:
            y = y.reshape(T, C // b, b // (2 * h), 2, h)
            y = mx.concatenate([y[..., 0, :] + y[..., 1, :], y[..., 0, :] - y[..., 1, :]], axis=-1)
            h *= 2
        return y.reshape(T, C) * (1.0 / np.sqrt(b))

    rows = []
    rng = np.random.default_rng(0)
    for C in (5120, 17408):
        s = mx.array(rng.choice([-1.0, 1.0], size=C).astype(np.float32))
        for T in (1, 256):
            xs = [mx.array(rng.normal(size=(T, C)).astype(np.float32)) for _ in range(QUEUE)]
            mx.eval(s, *xs)
            for b in (128, 1024):
                Hm = dense(b)
                mx.eval(Hm)
                for impl, fn in (("matmul", lambda x, b=b, Hm=Hm: via_matmul(x, s, Hm, b)),
                                 ("butterfly", lambda x, b=b: via_butterfly(x, s, b))):
                    y = fn(xs[0])
                    mx.eval(y)
                    # exactness against float64 numpy on the same input
                    X = np.asarray(xs[0], dtype=np.float64)
                    Hn = np.asarray(Hm, dtype=np.float64)
                    ref = ((X * np.asarray(s, dtype=np.float64)).reshape(T, C // b, b) @ Hn).reshape(T, C)
                    rel = float(np.linalg.norm(np.asarray(y, dtype=np.float64) - ref) / np.linalg.norm(ref))
                    for _ in range(5):
                        mx.eval(fn(xs[0]))
                    ts = []
                    for _ in range(REPEATS):
                        t0 = time.perf_counter()
                        mx.eval(fn(xs[0]))
                        ts.append(time.perf_counter() - t0)
                    pipe = []
                    for _ in range(max(5, REPEATS // 5)):
                        t0 = time.perf_counter()
                        mx.eval(*[fn(xs[i]) for i in range(QUEUE)])
                        pipe.append((time.perf_counter() - t0) / QUEUE)
                    row = {"C": C, "T": T, "block": b, "impl": impl, "synced_median_us": 1e6 * float(np.median(ts)),
                           "synced_p90_us": 1e6 * float(np.percentile(ts, 90)), "pipelined_us": 1e6 * float(np.median(pipe)),
                           "f32_vs_f64_rel": rel}
                    rows.append(row)
                    print(f"[xform] C={C} T={T} b={b} {impl:9s} synced {row['synced_median_us']:7.0f} us pipelined {row['pipelined_us']:7.0f} us rel {rel:.1e}", flush=True)
    # projection at T = 1: the cheapest implementation per C
    best = {}
    for C in (5120, 17408):
        cands = [r for r in rows if r["C"] == C and r["T"] == 1]
        best[C] = min(cands, key=lambda r: r["pipelined_us"])
    per_layer_us = 3 * best[5120]["pipelined_us"] + best[17408]["pipelined_us"]   # three 5,120-wide inputs, one 17,408-wide
    step_ms = LAYERS * per_layer_us / 1e3
    proj = {"transforms_per_layer": TRANSFORMS_PER_LAYER, "per_layer_us_best_impl": per_layer_us,
            "per_step_ms_unfused_mlx_ops": step_ms, "per_step_fraction_of_step": step_ms / STEP_MS,
            "per_transform_over_w4_launch": {str(C): best[C]["pipelined_us"] / W4_LAUNCH_US for C in best},
            "best_impl": {str(C): {k: best[C][k] for k in ("block", "impl", "pipelined_us")} for C in best},
            "metadata_bits_per_rotated_matrix": "one sign bit per input channel (5,120 or 17,408 bits), shareable by matrices with the same input; the Hadamard is implicit",
            "note": "unfused MLX ops; a fused Metal rotation inside the GEMV kernel would cost arithmetic, not launches, and is not measured here"}
    receipt = {"commit": head, "tree_dirty": dirty, "repeats": REPEATS, "queue": QUEUE, "step_ms_reference": STEP_MS,
               "w4_launch_us_reference": W4_LAUNCH_US, "rows": rows, "projection": proj}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[xform] per step {step_ms:.1f} ms ({100*step_ms/STEP_MS:.0f} percent) as unfused ops; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()

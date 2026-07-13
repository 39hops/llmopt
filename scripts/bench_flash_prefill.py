"""Flash prefill (Metal) vs mx.fast.scaled_dot_product_attention,
causal, prefill shapes. tq_tile sweep = the config-estimator rung's
revival data (a config axis with real variance, unlike the 6-point
GEMV space). mx.eval every timed iteration (the lazy-graph scar)."""
import json
import time
from pathlib import Path

import mlx.core as mx

from llmopt.kernels.metal import flash_prefill

OUT = Path("data/flash_tile_labels.jsonl")


def bench(f, it=50, warmup=10):
    for _ in range(warmup):
        mx.eval(f())
    t0 = time.perf_counter()
    for _ in range(it):
        mx.eval(f())
    return (time.perf_counter() - t0) / it * 1e6


def main() -> None:
    rows = []
    print(f"{'shape':>18s} {'mx sdpa':>9s}" +
          "".join(f"{'tq=' + str(t):>9s}" for t in (4, 8, 16, 32)))
    for h, t, d in [(8, 512, 64), (8, 1024, 64), (8, 2048, 64),
                    (8, 1024, 128), (16, 1024, 64), (8, 4096, 64)]:
        mx.random.seed(0)
        q = mx.random.normal((h, t, d)).astype(mx.float16)
        k = mx.random.normal((h, t, d)).astype(mx.float16)
        v = mx.random.normal((h, t, d)).astype(mx.float16)
        scale = 1 / d**0.5
        t_ref = bench(lambda: mx.fast.scaled_dot_product_attention(
            q[None], k[None], v[None], scale=scale, mask="causal"))
        line = f"H{h} T{t} D{d:<4d} {t_ref:8.0f}u"
        for tq in (4, 8, 16, 32):
            us = bench(lambda: flash_prefill(q, k, v, causal=True,
                                             tq_tile=tq))
            rows.append({"h": h, "t": t, "d": d, "tq": tq,
                         "us": round(us, 1),
                         "ref_us": round(t_ref, 1)})
            line += f" {us:8.0f}u"
        print(line, flush=True)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    best = {}
    for r in rows:
        key = (r["h"], r["t"], r["d"])
        if key not in best or r["us"] < best[key]["us"]:
            best[key] = r
    wins = sum(1 for r in best.values() if r["us"] < r["ref_us"])
    print(f"best-tile wins vs mx sdpa: {wins}/{len(best)} shapes "
          f"(labels -> {OUT})")
    print("bar: honest — record the split; the tile sweep is the "
          "config-estimator's training data either way")


if __name__ == "__main__":
    main()

"""Fused chunked CE vs naive full-logits CE at Qwen-0.5B head shapes.

Bar (pre-registered): fused must cut peak memory materially at 8k+
tokens; report tok/s honestly — the vjp recompute doubles matmul FLOPs,
so a throughput loss is expected and recorded, not hidden. The win
condition is training configurations that OOM (or swap) under naive
becoming feasible, at tolerable tok/s cost.

    .venv/bin/python scripts/bench_fused_ce.py
"""
import sys
import time
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.train.fused_ce import fused_ce, naive_ce

D, V = 896, 151936  # Qwen2.5-0.5B hidden / vocab


def run(fn, h, w, t, it=5):
    lg = mx.value_and_grad(lambda h_, w_: fn(h_, w_, t), argnums=(0, 1))
    mx.eval(*lg(h, w))  # warmup
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for _ in range(it):
        loss, grads = lg(h, w)
        mx.eval(loss, *grads)
    dt = (time.perf_counter() - t0) / it
    return loss.item(), mx.get_peak_memory() / 2**20, dt


def main() -> None:
    print(f"{'tokens':>7s} {'arm':>12s} {'loss':>7s} {'peak MB':>9s} "
          f"{'ms':>8s} {'tok/s':>9s}")
    for n in (2048, 8192, 16384, 32768):
        mx.random.seed(0)
        h = mx.random.normal((n, D)).astype(mx.float16)
        w = (mx.random.normal((V, D)) * 0.02).astype(mx.float16)
        t = mx.random.randint(0, V, (n,))
        arms = [("naive", lambda h_, w_, t_: naive_ce(h_, w_, t_))]
        for c in (1024, 4096):
            arms.append((f"fused c={c}",
                         lambda h_, w_, t_, c=c: fused_ce(h_, w_, t_, c)))
        for name, fn in arms:
            if name == "naive" and n > 16384:
                print(f"{n:7d} {'naive':>12s}    (skipped: ~19GB logits"
                      " fp32 fwd+bwd)")
                continue
            loss, mb, dt = run(fn, h, w, t)
            print(f"{n:7d} {name:>12s} {loss:7.3f} {mb:9.0f} "
                  f"{dt * 1e3:8.1f} {n / dt:9.0f}", flush=True)
        print()


if __name__ == "__main__":
    main()

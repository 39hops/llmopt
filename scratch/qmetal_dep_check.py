"""Does the ablation stand-in of scratch/qwen_metal_breakdown.py really keep
its upstream work alive? The stand-in returns x[..., :1] * 0.0 + c. MLX
evaluates lazily and prunes anything the requested output does not depend
on, so the question is whether the multiply by zero is folded away. Three
timings of mx.eval on identical shapes, medians of REPEATS:
  upstream   y = A @ x          (a large float32 matmul, the upstream work)
  standin    z = (A @ x)[..., :1] * 0.0 + c
  constant   w = c              (no upstream at all)
If standin is within a few percent of upstream and far above constant,
the dependence is preserved and the breakdown's attribution stands as
approximate (overlap and dispatch sharing remain). A second check runs
the real stand-in class on a fused W4 linear from arm C layer 0: the
time of eval(standin(res.gemm(x))) against eval(res.gemm(x)).
Receipt logs/qmetal/dep_check_<commit>.json (refuse-if-exists).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> None:
    import mlx.core as mx
    from llmopt.lab import qartifact, qmetal

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = os.environ.get("OUT", f"logs/qmetal/dep_check_{head}.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    repeats = int(os.environ.get("REPEATS", "30"))

    def med(fn):
        for _ in range(3):
            mx.eval(fn())
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            mx.eval(fn())
            ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    rng = np.random.default_rng(0)
    A = mx.array(rng.normal(size=(8192, 8192)).astype(np.float32))
    x = mx.array(rng.normal(size=(8192, 1)).astype(np.float32))
    c = mx.full((5120,), 1e-3, dtype=mx.float32)
    mx.eval(A, x, c)
    r = {"commit": head, "repeats": repeats,
         "dense_upstream_s": med(lambda: A @ x),
         "dense_standin_s": med(lambda: (A @ x)[:1, 0] * 0.0 + c),
         "dense_constant_s": med(lambda: c + 0.0)}
    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    arm = os.path.basename(art.rstrip("/"))
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"),
                                   f"logs/qwenwhole/artifact_digest_{arm}.txt")
    e = q["manifest"]["model.language_model.layers.0.mlp.gate_proj.weight"]
    with open(os.path.join(art, e["shard"] + ".bin"), "rb") as f:
        f.seek(e["off"])
        buf = f.read(e["len"])
    res = qmetal.W4Metal(buf, e["shape"])
    xr = mx.array(rng.normal(size=(1, 5120)).astype(np.float32))
    cr = mx.full((17408,), 1e-3, dtype=mx.float32)
    mx.eval(xr, cr)
    r.update({"w4_upstream_s": med(lambda: res.gemm(xr)),
              "w4_standin_s": med(lambda: res.gemm(xr)[..., :1] * 0.0 + cr),
              "w4_constant_s": med(lambda: cr + 0.0)})
    r["dependence_preserved"] = bool(r["dense_standin_s"] > 0.8 * r["dense_upstream_s"]
                                     and r["w4_standin_s"] > 0.8 * r["w4_upstream_s"])
    json.dump(r, open(out, "w"), indent=1)
    print(json.dumps(r, indent=1), flush=True)


if __name__ == "__main__":
    main()

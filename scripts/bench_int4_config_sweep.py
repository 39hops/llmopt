"""Config estimator for the int4 dequant-GEMV kernel (Artin's rung:
"can't the kernel packing itself be estimated?"). The learned-autotuner
recipe: sweep configs honestly, the sweep IS the training data, a tiny
net predicts latency from (shape, config) features, and the config it
picks per shape is scored by REGRET vs the exhaustive-sweep oracle on
held-out shapes (the FA Law with zero indirection: the oracle is the
wall clock). Precedent: TVM/Ansor cost models.

Grid: decode shapes (D, N) x group_size {32, 64, 128} x kernel variant
{v2 scalar-uint32, v3 vector-uint2}. Held-out = every third shape.
Bar: net-picked config regret < 5% of oracle-best latency on held-out
shapes AND beats the static default (v3, gs=128).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx

from llmopt.kernels.metal import (_int4_gemv, _int4_gemv_v3,
                                  quantize_pack_int4)

OUT = Path("data/int4_config_labels.jsonl")

SHAPES = [(512, 2048), (768, 3072), (896, 4864), (1024, 4096),
          (1536, 6144), (2048, 8192), (2560, 10240), (3072, 12288),
          (4096, 14336), (896, 896), (2048, 2048), (4096, 4096)]
GROUPS = (32, 64, 128)
VARIANTS = ("v2", "v3")


def run_kernel(variant, x, packed, sc, mn, gs):
    n, d2 = packed.shape
    kern = _int4_gemv_v3 if variant == "v3" else _int4_gemv
    (out,) = kern(
        inputs=[x, packed, sc, mn],
        template=[("T", x.dtype), ("D2", d2), ("NG", sc.shape[1]),
                  ("GS", gs)],
        grid=(n * 32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(n,)], output_dtypes=[x.dtype])
    return out


def bench(f, it=100, warmup=15):
    for _ in range(warmup):
        mx.eval(f())
    t0 = time.perf_counter()
    for _ in range(it):
        mx.eval(f())
    return (time.perf_counter() - t0) / it * 1e6


def sweep() -> list[dict]:
    rows = []
    for d, n in SHAPES:
        mx.random.seed(0)
        w = mx.random.normal((n, d)).astype(mx.float16)
        x = mx.random.normal((d,)).astype(mx.float16)
        for gs in GROUPS:
            packed, sc, mn = quantize_pack_int4(w, group_size=gs)
            mx.eval(packed, sc, mn, x)
            for v in VARIANTS:
                us = bench(lambda: run_kernel(v, x, packed, sc, mn, gs))
                rows.append({"d": d, "n": n, "gs": gs, "variant": v,
                             "us": round(us, 2)})
                print(f"D={d} N={n} gs={gs} {v}: {us:.0f}us", flush=True)
    OUT.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


def fit_and_score(rows: list[dict]) -> None:
    import torch

    def feats(r):
        import math
        return [math.log(r["d"]), math.log(r["n"]),
                math.log(r["d"] * r["n"]), float(r["gs"]) / 128.0,
                1.0 if r["variant"] == "v3" else 0.0]

    shapes = sorted({(r["d"], r["n"]) for r in rows})
    test_shapes = set(shapes[::3])          # held-out: every third
    tr = [r for r in rows if (r["d"], r["n"]) not in test_shapes]
    te = [r for r in rows if (r["d"], r["n"]) in test_shapes]
    import math
    Xt = torch.tensor([feats(r) for r in tr], dtype=torch.float32)
    yt = torch.tensor([[math.log(r["us"])] for r in tr],
                      dtype=torch.float32)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(2000):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(net((Xt - mu) / sd), yt)
        loss.backward()
        opt.step()

    def pred(r):
        with torch.no_grad():
            x = (torch.tensor([feats(r)], dtype=torch.float32) - mu) / sd
            return float(net(x))

    tot_regret = tot_default = tot_best = 0.0
    print(f"\n{'shape':16s} {'oracle':>8s} {'net-pick':>9s} "
          f"{'default':>8s}")
    for d, n in sorted(test_shapes):
        cfgs = [r for r in te if r["d"] == d and r["n"] == n]
        best = min(cfgs, key=lambda r: r["us"])
        pick = min(cfgs, key=pred)
        default = next(r for r in cfgs
                       if r["variant"] == "v3" and r["gs"] == 128)
        tot_best += best["us"]
        tot_regret += pick["us"]
        tot_default += default["us"]
        print(f"D={d:<5d} N={n:<6d} {best['us']:7.0f}u "
              f"{pick['us']:8.0f}u {default['us']:7.0f}u"
              f"  (oracle: gs={best['gs']} {best['variant']})")
    reg_pct = (tot_regret - tot_best) / tot_best * 100
    def_pct = (tot_default - tot_best) / tot_best * 100
    print(f"\nnet-pick regret {reg_pct:.1f}% vs oracle; static default "
          f"(v3/gs128) regret {def_pct:.1f}%")
    print("bar: net regret < 5% AND < default regret")


if __name__ == "__main__":
    rows = sweep() if not OUT.exists() else \
        [json.loads(l) for l in OUT.read_text().splitlines()]
    fit_and_score(rows)

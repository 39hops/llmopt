"""RUNG R (pre-reg V4-RUNG-R + 2B-ROUTER): read DeepSeek-V4-Flash's MoE
router for free — 2 MB of gate weights and 1 KB of bias, no inference.

Why the bias is worth reading (vendor inference/model.py, Gate.forward):
    scores = sqrtsoftplus(x @ gate.weight)
    original_scores = scores
    scores = scores + gate.bias        # SELECTION only
    indices = scores.topk(6)[1]
    weights = original_scores.gather(indices)   # UNBIASED
The bias never touches the output weight, so it is purely DeepSeek's
aux-loss-free load balancing: a trained record of how much correction
each expert needed to win its share of tokens.

Readouts: R-a bias distribution, R-b key-vector geometry against a
matched random null, R-c presence of the hash-routing table.

Env: LAYERS (default "4,22,40"), SHARDS (default "6,24,42").
Usage: .venv/bin/python scratch/v4flash_router.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import v4flash_rungA as RA  # noqa: E402

LAYERS = [int(x) for x in os.environ.get("LAYERS", "4,22,40").split(",")]
SHARDS = [int(x) for x in os.environ.get("SHARDS", "6,24,42").split(",")]
OUT = "logs/opus/v4_router.jsonl"


def bf16_to_f32(raw, shape):
    """safetensors BF16 -> float32 (upper 16 bits of the f32 word)."""
    u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
    return u.view(np.float32).reshape(shape)


def read_router(shard, layer):
    RA.SHARD = shard
    hdr, url, base = RA.header()
    wn, bn = f"layers.{layer}.ffn.gate.weight", f"layers.{layer}.ffn.gate.bias"
    if wn not in hdr:
        return None
    W = bf16_to_f32(RA.cached(wn, hdr, url, base), hdr[wn]["shape"])
    b = None
    if bn in hdr:
        b = np.frombuffer(RA.cached(bn, hdr, url, base),
                          dtype=np.float32).copy()
    return W.astype(np.float64), b, hdr


def null_cosines(n, d, rng, reps=1):
    """Matched null: cosines among n random Gaussian vectors in R^d."""
    out = []
    for _ in range(reps):
        G = rng.standard_normal((n, d))
        G /= np.linalg.norm(G, axis=1, keepdims=True)
        C = G @ G.T
        out.append(C[np.triu_indices(n, 1)])
    return np.concatenate(out)


def main():
    os.makedirs("logs/opus", exist_ok=True)
    rng = np.random.default_rng(2026_08_02)
    sink = open(OUT, "a")
    for shard, layer in zip(SHARDS, LAYERS):
        got = read_router(shard, layer)
        if got is None:
            print(f"[R] layer {layer}: no ffn.gate in shard {shard}")
            continue
        W, b, _ = got
        n, d = W.shape
        U = W / np.linalg.norm(W, axis=1, keepdims=True)
        C = U @ U.T
        off = C[np.triu_indices(n, 1)]
        nul = null_cosines(n, d, rng)
        p999 = float(np.quantile(np.abs(nul), 0.999))
        row = {"layer": layer, "n_experts": n, "dim": d,
               "cos_absmax": float(np.abs(off).max()),
               "cos_absmean": float(np.abs(off).mean()),
               "null_absmax": float(np.abs(nul).max()),
               "null_absmean": float(np.abs(nul).mean()),
               "null_p999": p999,
               "frac_above_null_p999": float((np.abs(off) > p999).mean())}
        if b is not None:
            row.update(bias_min=float(b.min()), bias_max=float(b.max()),
                       bias_mean=float(b.mean()), bias_std=float(b.std()),
                       bias_argmax=int(b.argmax()), bias_argmin=int(b.argmin()))
        # the five most-similar pairs, selection rule fixed in the pre-reg
        iu = np.triu_indices(n, 1)
        order = np.argsort(-off)[:5]
        row["top_pairs"] = [[int(iu[0][k]), int(iu[1][k]), float(off[k])]
                            for k in order]
        sink.write(json.dumps(row) + "\n")
        print(f"[R] layer {layer}: |cos| max {row['cos_absmax']:.4f} "
              f"mean {row['cos_absmean']:.4f} | null max "
              f"{row['null_absmax']:.4f} p99.9 {p999:.4f} | above "
              f"{row['frac_above_null_p999']:.4%}")
        if b is not None:
            print(f"[R]   bias [{row['bias_min']:+.4f}, "
                  f"{row['bias_max']:+.4f}] mean {row['bias_mean']:+.4f} "
                  f"sd {row['bias_std']:.4f} | most-boosted e"
                  f"{row['bias_argmax']} most-suppressed e{row['bias_argmin']}")
        print(f"[R]   top pairs: " + ", ".join(
            f"e{a}~e{c} {v:.3f}" for a, c, v in row["top_pairs"]))
    sink.close()


if __name__ == "__main__":
    main()

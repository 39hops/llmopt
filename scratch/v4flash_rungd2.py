"""RUNG D2 (pre-reg V4-RUNG-D2): measure <u,x> on DeepSeek-V4-Flash's REAL
traffic by inverting its trained load-balancing bias. No forward pass.

AMENDMENT RUNGD-0803 left the shared-router-direction question
undetermined: deleting u leaves 97% of routing under isotropic input and
12-27% under input aligned with u, and nothing said which regime V4 is
in. That looked like it needed real hidden states.

It does not, because the bias IS a record of real hidden states.
DeepSeek's aux-loss-free balancer (topk_method noaux_tc) drives the bias
until expert loads equalise under the traffic the model actually saw,
and the bias touches SELECTION only (inference/model.py:579-585 -- the
output weights gather original_scores). So:

    if the shipped bias balances load at the true input distribution,
    the mu that balances load under a one-parameter input family is
    that family's best fit to it.

The family holds the router-input norm at its RMS-normalised value
sqrt(d) and sweeps the angle to u:

    x(mu) = sqrt(d) * (cos t * u + sin t * v),   <u,x> = mu = sqrt(d) cos t

with v drawn uniformly on the unit sphere orthogonal to u. Imbalance is
the coefficient of variation of expert load, which is 0 at perfect
balance and grows either way from the optimum -- so the minimum is a
genuine estimate rather than a boundary artifact.

Three arms: WITH the shipped bias (the estimator), WITHOUT it (the
control -- if the minimum barely moves, the bias is not what carries the
signal), and a SHUFFLED bias (the null -- a bias with the same marginal
distribution but no relationship to c).

Env: LAYERS (default "4,22,40"), SHARDS (default "6,24,42"),
     NDRAW (default 4000), MUS (default "-32,...,64" grid).
Usage: .venv/bin/python scratch/v4flash_rungd2.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
from v4flash_router import read_router  # noqa: E402
from v4flash_rungd import score, shared_directions, topk_sets  # noqa: E402

LAYERS = [int(x) for x in os.environ.get("LAYERS", "4,22,40").split(",")]
SHARDS = [int(x) for x in os.environ.get("SHARDS", "6,24,42").split(",")]
NDRAW = int(os.environ.get("NDRAW", "4000"))
SEED = 2026_08_03
OUT = "logs/opus/v4_rungd2.jsonl"
assert len(SHARDS) == len(LAYERS), "SHARDS and LAYERS must pair up"


def perp_basis(u, n, rng):
    """n unit vectors orthogonal to u, uniform on that sphere."""
    V = rng.standard_normal((n, u.size))
    V -= np.outer(V @ u, u)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def imbalance(W, bias, X, topk):
    """Coefficient of variation of expert load under top-k selection.

    CV, not max/mean: the balancer equalises the whole load vector, and
    an extremum would make the estimate hostage to one expert -- which is
    the wrong-extreme failure this branch has now booked three times.
    """
    idx, _ = topk_sets(score(X, W, bias), topk)
    load = np.bincount(idx.ravel(), minlength=len(W)).astype(np.float64)
    return float(load.std() / load.mean())


def main():
    os.makedirs("logs/opus", exist_ok=True)
    mus = np.array([float(x) for x in os.environ["MUS"].split(",")]) \
        if "MUS" in os.environ else np.arange(-32, 64.1, 2.0)
    topk = 6
    sink = open(OUT, "a")
    for shard, layer in zip(SHARDS, LAYERS):
        got = read_router(shard, layer)
        assert got is not None, f"no gate for layer {layer}"
        W, bias, _ = got
        assert bias is not None, f"layer {layer} has no bias to invert"
        n, d = W.shape
        u = shared_directions(W)[1]["unit_mean"][0]
        c = W @ u
        rng = np.random.default_rng(SEED + layer)
        V = perp_basis(u, NDRAW, rng)
        shuf = rng.permutation(bias)          # same marginal, no link to c
        nrm = np.linalg.norm(W, axis=1)
        A = np.c_[np.ones(n), nrm]
        rb = bias - A @ np.linalg.lstsq(A, bias, rcond=None)[0]
        rc = c - A @ np.linalg.lstsq(A, c, rcond=None)[0]

        curves = {"with_bias": [], "no_bias": [], "shuffled_bias": []}
        for mu in mus:
            t = np.arccos(np.clip(mu / np.sqrt(d), -1, 1))
            X = np.sqrt(d) * (np.cos(t) * u + np.sin(t) * V)
            curves["with_bias"].append(imbalance(W, bias, X, topk))
            curves["no_bias"].append(imbalance(W, None, X, topk))
            curves["shuffled_bias"].append(imbalance(W, shuf, X, topk))
        best = {k: float(mus[int(np.argmin(v))]) for k, v in curves.items()}
        row = {"layer": layer, "n_experts": n, "dim": d, "topk": topk,
               "ndraw": NDRAW, "seed": SEED + layer,
               "mus": mus.tolist(), "curves": curves, "argmin_mu": best,
               "corr_bias_c": float(np.corrcoef(bias, c)[0, 1]),
               "partial_corr_bias_c": float(np.corrcoef(rb, rc)[0, 1]),
               "min_cv": {k: float(np.min(v)) for k, v in curves.items()},
               "cv_at_mu0": {k: float(v[int(np.argmin(np.abs(mus)))])
                             for k, v in curves.items()}}
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        print(f"[D2] L{layer}: corr(bias,c) {row['corr_bias_c']:+.4f} "
              f"(partial {row['partial_corr_bias_c']:+.4f})", flush=True)
        for k in curves:
            print(f"[D2]   {k:14s} argmin <u,x> = {best[k]:+6.1f} | "
                  f"CV there {row['min_cv'][k]:.4f} | CV at 0 "
                  f"{row['cv_at_mu0'][k]:.4f}", flush=True)
    sink.close()


if __name__ == "__main__":
    main()

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

The family holds the router-input norm at the value the model actually
produces and sweeps the component along u:

    x(mu) = mu*u + sqrt(R^2 - mu^2) * vhat,   R = ||ffn_norm.weight||

R IS NOT sqrt(d). V4's RMSNorm returns weight * x/rms(x)
(inference/model.py:197-202) and the gate is fed ffn_norm(x) (:704), so a
unit-RMS vector leaves the norm with length ||ffn_norm.weight|| --
measured 15.51 / 29.24 / 41.08 at layers 4 / 22 / 40, against sqrt(d) =
64. The first version of this cell used 64 and was therefore 4.1x / 2.2x
/ 1.6x too wide, LAYER-DEPENDENTLY, so the three layers were not even on
a common mu axis (reviewer catch, 2026-08-03 -- the third instance on
this branch of a bound travelling without its assumption, in the cell
written to correct the second).
vhat is the unit perpendicular of weight (*) ghat with ghat unit-RMS
isotropic, so the perpendicular directions carry the layer's own channel
gains rather than being flat. Imbalance is
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
import v4flash_rungA as RA  # noqa: E402
from v4flash_router import bf16_to_f32, read_router  # noqa: E402
from v4flash_rungd import (  # noqa: E402
    agreement, score, shared_directions, topk_sets, vendor_scoring)

LAYERS = [int(x) for x in os.environ.get("LAYERS", "4,22,40").split(",")]
SHARDS = [int(x) for x in os.environ.get("SHARDS", "6,24,42").split(",")]
NDRAW = int(os.environ.get("NDRAW", "4000"))
NPERM = int(os.environ.get("NPERM", "40"))
SEED = 2026_08_03
OUT = "logs/opus/v4_rungd2.jsonl"
assert len(SHARDS) == len(LAYERS), "SHARDS and LAYERS must pair up"


def input_norm(shard, layer):
    """||ffn_norm.weight|| -- the length the gate's input actually has.

    Free: 8 KB of BF16 in the shard the cell already opens.
    """
    RA.SHARD = shard
    hdr, url, base = RA.header()
    k = f"layers.{layer}.ffn_norm.weight"
    assert hdr[k]["dtype"] == "BF16", hdr[k]["dtype"]
    w = bf16_to_f32(RA.cached(k, hdr, url, base), hdr[k]["shape"])
    return float(np.linalg.norm(w)), w.astype(np.float64)


def perp_basis(u, n, rng, gain):
    """n unit vectors orthogonal to u, shaped by the layer's channel gains.

    A flat sphere would make every channel equally likely to carry the
    perpendicular signal; the norm's learned per-channel weight says
    otherwise, and that weight is what sets R in the first place.
    """
    G = rng.standard_normal((n, u.size))
    G /= np.sqrt((G ** 2).mean(axis=1, keepdims=True))   # unit RMS
    V = G * gain
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
    # Vendor guards, which the first version of this cell skipped
    # entirely while hardcoding topk=6 (reviewer catch): without them no
    # D2 row is pinned to a source version, and a group-limited routing
    # stage would make flat top-k the wrong operator.
    sha, topk, n_exp_cfg, n_hash = vendor_scoring()
    print(f"[D2] vendor model.py sha {sha[:16]} | topk {topk} | "
          f"n_routed {n_exp_cfg}", flush=True)
    sink = open(OUT, "a")
    for shard, layer in zip(SHARDS, LAYERS):
        got = read_router(shard, layer)
        assert got is not None, f"no gate for layer {layer}"
        W, bias, _ = got
        assert bias is not None, f"layer {layer} has no bias to invert"
        n, d = W.shape
        assert n == n_exp_cfg, f"{n} keys, config says {n_exp_cfg}"
        assert layer >= n_hash, f"layer {layer} is hash-routed"
        R, gain = input_norm(shard, layer)
        u = shared_directions(W)[1]["unit_mean"][0]
        c = W @ u
        rng = np.random.default_rng(SEED + layer)
        V = perp_basis(u, NDRAW, rng, gain)
        # the grid is expressed as a FRACTION of R, so the three layers
        # are on a common axis despite having different input norms.
        fracs = (np.array([float(x) for x in os.environ["FRACS"].split(",")])
                 if "FRACS" in os.environ else np.arange(-0.5, 1.001, 0.05))
        mus = fracs * R
        # 40 permutations, not one: an n=1 "null" has no spread to
        # compare against (reviewer catch).
        shufs = [rng.permutation(bias) for _ in range(NPERM)]
        nrm = np.linalg.norm(W, axis=1)
        A = np.c_[np.ones(n), nrm]
        rb = bias - A @ np.linalg.lstsq(A, bias, rcond=None)[0]
        rc = c - A @ np.linalg.lstsq(A, c, rcond=None)[0]

        # RANK1 deflation agreement, measured in THIS cell's input family
        # rather than inferred from RUNG-D's. RUNG-D's shift arm used a
        # base of ||x|| ~ sqrt(d) = 64, so mapping its m onto this mu was
        # never apples to apples; measuring both curves under one family
        # makes the bound a measurement instead of a cross-cell inference.
        W_rank1 = W - np.outer(c, u)
        curves = {"with_bias": [], "no_bias": [], "shuffled_bias": [],
                  "rank1_set_agreement": []}
        shuf_sd = []
        for mu in mus:
            # exact, and asserted rather than clipped: the first version
            # np.clip'd out-of-range mu and still logged the REQUESTED
            # value, so unreachable grid points looked measured.
            assert abs(mu) <= R + 1e-9, f"|mu|={abs(mu)} exceeds R={R}"
            X = mu * u + np.sqrt(max(R * R - mu * mu, 0.0)) * V
            assert abs(float((X @ u).mean()) - mu) < 1e-6 * max(R, 1), "mu off"
            curves["with_bias"].append(imbalance(W, bias, X, topk))
            curves["no_bias"].append(imbalance(W, None, X, topk))
            curves["rank1_set_agreement"].append(
                agreement(X, W, W_rank1, bias, topk)["set_agreement"])
            sh = [imbalance(W, s_, X, topk) for s_ in shufs]
            curves["shuffled_bias"].append(float(np.mean(sh)))
            shuf_sd.append(float(np.std(sh)))
        best = {k: float(mus[int(np.argmin(v))]) for k, v in curves.items()
                if k != "rank1_set_agreement"}
        row = {"layer": layer, "n_experts": n, "dim": d, "topk": topk,
               "ndraw": NDRAW, "seed": SEED + layer, "nperm": NPERM,
               "model_py_sha": sha[:16], "input_norm_R": R,
               "sqrt_d": float(np.sqrt(d)),
               "mus": mus.tolist(), "fracs": fracs.tolist(),
               "argmin_frac": {k: float(fracs[int(np.argmin(v))])
                               for k, v in curves.items()},
               "shuffled_sd": shuf_sd,
               "curves": curves, "argmin_mu": best,
               "corr_bias_c": float(np.corrcoef(bias, c)[0, 1]),
               "partial_corr_bias_c": float(np.corrcoef(rb, rc)[0, 1]),
               "min_cv": {k: float(np.min(v)) for k, v in curves.items()},
               "cv_at_mu0": {k: float(v[int(np.argmin(np.abs(mus)))])
                             for k, v in curves.items()}}
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        print(f"[D2] L{layer}: R=||ffn_norm.w||={R:.2f} (sqrt(d)=64.00) | "
              f"corr(bias,c) {row['corr_bias_c']:+.4f} "
              f"(partial {row['partial_corr_bias_c']:+.4f})", flush=True)
        for k in best:
            print(f"[D2]   {k:14s} argmin <u,x> = {best[k]:+6.2f} "
                  f"({row['argmin_frac'][k]:+.2f} R) | "
                  f"CV there {row['min_cv'][k]:.4f} | CV at 0 "
                  f"{row['cv_at_mu0'][k]:.4f}", flush=True)
    sink.close()


if __name__ == "__main__":
    main()

"""RUNG D (pre-reg V4-RUNG-D + S0): is DeepSeek-V4-Flash's shared router
key direction routing-INERT?

VERDICT V4-RUNG-R booked "the confluence lives in the ROUTER" because
all 32,640 key pairs have positive cosine and every key shares a +0.385
mean direction. Hazard 8 says a CONSTANT additive term is a top-k no-op
-- that is why gate.bias can shift selection without touching the output
weight. The same argument was never turned on the KEYS: if every key
carries the same component along a unit u, every logit picks up the same
<u,x> term, which a top-k comparing scores to each other cannot see.

Scoring is the vendor's, verified against the shipped source
(inference/model.py Gate.forward, sha pinned below):
    scores = F.softplus(linear(x.float(), weight.float())).sqrt()
    indices = (scores + bias).topk(topk)      # bias = SELECTION only
So the deflation is applied to the KEYS, pre-nonlinearity, which is
where the shared direction lives.

Also recomputes the +0.385 / +0.254 / 0.0276 decomposition from the
sha-pinned blobs and LOGS it -- AMENDMENT AUDIT-0802 item (8) recorded
that those numbers came from an inline command and are not recoverable
from v4_router.jsonl (which stores only absolute cosines).

Registered prediction is evaluated at SCALE 1.0 (x ~ N(0, I)). The other
scales are an UNREGISTERED robustness probe of the fence: softplus is
not scale-invariant, so a conclusion drawn only in its saturated regime
would be an artifact of the null model, not a fact about the router.

Env: LAYERS (default "4,22,40"), SHARDS (default "6,24,42"),
     NDRAW (default 2000), SCALES (default "0.1,1.0,10.0").
Usage: .venv/bin/python scratch/v4flash_rungd.py
"""
import hashlib
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import v4flash_rungA as RA  # noqa: E402
from v4flash_router import read_router  # noqa: E402

LAYERS = [int(x) for x in os.environ.get("LAYERS", "4,22,40").split(",")]
SHARDS = [int(x) for x in os.environ.get("SHARDS", "6,24,42").split(",")]
NDRAW = int(os.environ.get("NDRAW", "2000"))
SCALES = [float(s) for s in os.environ.get("SCALES", "0.1,1.0,10.0").split(",")]
SEED = 2026_08_03
OUT = "logs/opus/v4_rungd.jsonl"
assert len(SHARDS) == len(LAYERS), "SHARDS and LAYERS must pair up"
# The scoring function is READ, not remembered: this cell's whole claim
# rests on top-k being applied to sqrt(softplus(.)) + bias, so the file
# that defines it is pinned like any other artifact.
MODEL_PY = f"{RA.REPO}/inference/model.py"
CONFIG = f"{RA.REPO}/config.json"


def vendor_scoring():
    """Confirm the score function and top-k from the shipped source."""
    with urllib.request.urlopen(MODEL_PY) as r:
        src = r.read()
    sha = hashlib.sha256(src).hexdigest()
    txt = src.decode()
    assert "F.softplus(scores).sqrt()" in txt, "score function moved"
    assert "scores = scores + self.bias" in txt, "bias path moved"
    assert "indices = scores.topk(self.topk, dim=-1)[1]" in txt, "topk moved"
    with urllib.request.urlopen(CONFIG) as r:
        cfg = json.loads(r.read())
    # config.json is the HF namespace (num_experts_per_tok), NOT
    # inference/model.py's ModelArgs namespace (n_activated_experts) --
    # the two ship side by side and only one of them is in this file.
    assert cfg["scoring_func"] == "sqrtsoftplus", cfg["scoring_func"]
    return sha, int(cfg["num_experts_per_tok"]), int(cfg["n_routed_experts"])


def shared_directions(W):
    """Four defensible definitions of "the" shared key direction.

    They matter because VERDICT V4-RUNG-R's inline decomposition did not
    record which one it used. Measured here (layer 22): the MEAN
    projection is stable to 0.0008 across all four and the residual
    |cos| to 0.0001, but the MIN swings 0.2397-0.2643 -- so the extremum
    is the definition-sensitive statistic, which is the same class of
    number AUDIT-0802 caught twice. The booked "min +0.254" is
    reproduced by mean-of-RAW-rows; this cell's registered arm uses
    mean-of-UNIT-rows, and the deflation result is insensitive to the
    choice (the four unit vectors agree to |cos| > 0.999).
    """
    U = W / np.linalg.norm(W, axis=1, keepdims=True)
    out = {}
    for name, v in (("unit_mean", U.mean(axis=0)),
                    ("raw_mean", W.mean(axis=0)),
                    ("svd_raw", np.linalg.svd(W, full_matrices=False)[2][0]),
                    ("svd_unit", np.linalg.svd(U, full_matrices=False)[2][0])):
        u = v / np.linalg.norm(v)
        p = U @ u
        if p.mean() < 0:            # singular vectors have a free sign
            u, p = -u, -p
        R = U - np.outer(p, u)
        Rn = R / np.linalg.norm(R, axis=1, keepdims=True)
        rc = (Rn @ Rn.T)[np.triu_indices(len(W), 1)]
        out[name] = (u, {"proj_mean": float(p.mean()),
                         "proj_min": float(p.min()),
                         "proj_max": float(p.max()),
                         "proj_std": float(p.std()),
                         "resid_abscos_mean": float(np.abs(rc).mean()),
                         "resid_abscos_max": float(np.abs(rc).max())})
    return U, out


def score(X, W, bias):
    """Vendor scores for selection: sqrt(softplus(X W^T)) + bias."""
    z = X @ W.T
    # softplus via the numerically stable identity; float64 throughout
    # so the comparison is not decided by rounding in a near-tie.
    sp = np.logaddexp(0.0, z)
    s = np.sqrt(sp)
    return s if bias is None else s + bias


def topk_sets(S, k):
    """Top-k indices per row, unordered, plus the rank-1 index."""
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return np.sort(idx, axis=1), S.argmax(axis=1)


def main():
    os.makedirs("logs/opus", exist_ok=True)
    sha, topk, n_exp_cfg = vendor_scoring()
    print(f"[Rd] vendor model.py sha {sha[:16]} | topk {topk} | "
          f"n_routed {n_exp_cfg}", flush=True)
    sink = open(OUT, "a")
    for shard, layer in zip(SHARDS, LAYERS):
        got = read_router(shard, layer)
        if got is None:
            print(f"[Rd] layer {layer}: no ffn.gate in shard {shard}")
            continue
        W, bias, _ = got
        n, d = W.shape
        assert n == n_exp_cfg, f"{n} keys, config says {n_exp_cfg}"

        # --- the decomposition VERDICT V4-RUNG-R reported inline ---
        _, dirs = shared_directions(W)
        u = dirs["unit_mean"][0]           # the registered definition

        # --- deflate the KEYS along u, pre-nonlinearity ---
        Wd = W - np.outer(W @ u, u)
        # the deflation must actually annihilate u, or every number below
        # measures a partial projection and the null is not the null.
        assert np.abs(Wd @ u).max() < 1e-9 * np.abs(W).max(), \
            "deflation left mass along u"

        rng = np.random.default_rng(SEED + layer)
        for sc in SCALES:
            X = rng.standard_normal((NDRAW, d)) * sc
            S_raw, S_def = score(X, W, bias), score(X, Wd, bias)
            k_raw, a_raw = topk_sets(S_raw, topk)
            k_def, a_def = topk_sets(S_def, topk)
            # set agreement: |intersection| / k, row by row
            inter = np.array([len(np.intersect1d(r, q, assume_unique=True))
                              for r, q in zip(k_raw, k_def)])
            z_raw, z_def = X @ W.T, X @ Wd.T
            # The shared component adds c_i * <u,x> to expert i, with
            # c = W u. sd(c)/|mean(c)| is the whole question in one
            # number: a LEVEL if it is small, a CONTRAST if not.
            c = W @ u
            row = {
                "layer": layer, "n_experts": n, "dim": d, "topk": topk,
                "scale": sc, "ndraw": NDRAW, "seed": SEED + layer,
                "model_py_sha": sha[:16],
                "registered": sc == 1.0,
                "decomposition": {k: v for k, (_, v) in dirs.items()},
                "set_agreement": float(inter.mean() / topk),
                "frac_identical_set": float((inter == topk).mean()),
                "rank1_agreement": float((a_raw == a_def).mean()),
                # is the shared component a LEVEL or a CONTRAST?
                "logit_mean_raw": float(z_raw.mean()),
                "logit_mean_def": float(z_def.mean()),
                "logit_absmean_raw": float(np.abs(z_raw).mean()),
                "logit_absmean_def": float(np.abs(z_def).mean()),
                "logit_across_expert_sd_raw": float(z_raw.std(axis=1).mean()),
                "logit_across_expert_sd_def": float(z_def.std(axis=1).mean()),
                "shared_c_mean": float(c.mean()), "shared_c_sd": float(c.std()),
                "shared_c_cv": float(c.std() / abs(c.mean())),
            }
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            tag = "REGISTERED" if row["registered"] else "robustness"
            print(f"[Rd] L{layer} scale {sc:<5g} {tag:11s} "
                  f"set {row['set_agreement']:.4f} | identical "
                  f"{row['frac_identical_set']:.4f} | rank1 "
                  f"{row['rank1_agreement']:.4f} | across-expert sd "
                  f"{row['logit_across_expert_sd_raw']:.3f} -> "
                  f"{row['logit_across_expert_sd_def']:.3f}", flush=True)
        for name, (_, d_) in dirs.items():
            star = " <- registered" if name == "unit_mean" else ""
            print(f"[Rd] L{layer} u={name:9s} proj mean {d_['proj_mean']:+.4f} "
                  f"min {d_['proj_min']:+.4f} max {d_['proj_max']:+.4f} | "
                  f"residual |cos| {d_['resid_abscos_mean']:.4f}{star}",
                  flush=True)
    sink.close()


if __name__ == "__main__":
    main()

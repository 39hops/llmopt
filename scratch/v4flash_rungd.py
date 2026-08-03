"""RUNG D (pre-reg V4-RUNG-D + S0; rewritten after AMENDMENT RUNGD-0803):
how much of DeepSeek-V4-Flash's routing survives deleting the shared
router key direction, and under WHICH input model?

VERDICT V4-RUNG-R booked "the confluence lives in the ROUTER" because
all 32,640 key pairs have positive cosine and every key shares a large
mean direction u. Hazard 8 notes that a term identical across experts is
a top-k no-op -- top-k compares experts to each other, so a level cancels
(exactly, since sqrt(softplus(.)) is strictly monotone, absent bias).
The question is whether the shared key component IS such a level.

Scoring is the vendor's, asserted against the shipped source
(inference/model.py Gate.forward, sha pinned at run time):
    scores = F.softplus(linear(x.float(), weight.float())).sqrt()
    indices = (scores + bias).topk(topk)      # bias = SELECTION only
so the deflations act on the KEYS, pre-nonlinearity.

THREE ARMS, because the first version conflated them (reviewer catch,
2026-08-03):
  RANK1  W - (W u) u^T   -- delete the shared direction outright. This
         removes a level AND cuts each key's gain by sqrt(1-p_i^2),
         which is 3.5-20.2% and VARIES per expert. It is not a pure
         level removal and the original verdict wrongly read it as one.
  LEVEL  W - 1 cbar u^T  -- subtract the MEAN coefficient from every
         key, leaving each expert's deviation intact. THIS is the level.
  SHIFT  RANK1 scored on x = N(0,I) + m*u. The isotropic null has
         E<u,x> = 0, which is exactly the condition that makes a shared
         direction irrelevant -- so the null quietly assumed the answer.
         Varying m is the only arm that tests the fence.

Statistics are chosen to be draw-free where possible. The first version
logged z.mean(), whose expectation is ZERO on both arms under an
isotropic x; it is sampling noise, its sign flips across seeds, and the
"% removed" it implies ranges 24.6-99.4%. The level lives in the WEIGHTS
(the norm of the mean key row), so it is computed there, with no draw.

Env: LAYERS (default "4,22,40"), SHARDS (default "6,24,42"),
     NDRAW (default 2000), SCALES (default "0.1,1.0,10.0"),
     SHIFTS (default "0,5,10,30,64").
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
SHIFTS = [float(s) for s in os.environ.get("SHIFTS", "0,5,10,30,64").split(",")]
SEED = 2026_08_03
OUT = "logs/opus/v4_rungd.jsonl"
assert len(SHARDS) == len(LAYERS), "SHARDS and LAYERS must pair up"
MODEL_PY = f"{RA.REPO}/inference/model.py"
CONFIG = f"{RA.REPO}/config.json"


def vendor_scoring():
    """Confirm the score function, top-k, and the ABSENCE of group routing.

    The three substring asserts alone are not sufficient: DeepSeek's V3
    lineage puts a group-limited stage (n_group / topk_group, top-2-sum
    per group, masked_fill_) BETWEEN the bias add and the final topk, and
    all three substrings survive verbatim in that implementation. A flat
    top-k would then be the wrong operator. So the config keys are
    asserted too (reviewer catch, 2026-08-03).
    """
    with urllib.request.urlopen(MODEL_PY) as r:
        src = r.read()
    sha = hashlib.sha256(src).hexdigest()
    txt = src.decode()
    assert "F.softplus(scores).sqrt()" in txt, "score function moved"
    assert "scores = scores + self.bias" in txt, "bias path moved"
    assert "indices = scores.topk(self.topk, dim=-1)[1]" in txt, "topk moved"
    assert "group_scores" not in txt and "topk_group" not in txt, \
        "group-limited routing appeared: flat top-k is the wrong operator"
    with urllib.request.urlopen(CONFIG) as r:
        cfg = json.loads(r.read())
    # config.json is the HF namespace (num_experts_per_tok), NOT
    # inference/model.py's ModelArgs namespace (n_activated_experts) --
    # the two ship side by side and only one of them is in this file.
    assert cfg["scoring_func"] == "sqrtsoftplus", cfg["scoring_func"]
    assert int(cfg.get("n_group", 1)) == 1, "grouped routing in config"
    return (sha, int(cfg["num_experts_per_tok"]), int(cfg["n_routed_experts"]),
            int(cfg["num_hash_layers"]))


def shared_directions(W):
    """Four defensible definitions of "the" shared key direction.

    They matter because VERDICT V4-RUNG-R's inline decomposition did not
    record which one it used. Measured (layer 22): the MEAN projection is
    stable to 0.0008 across all four and the residual |cos| to 0.0002,
    but the MIN swings 0.2397-0.2643 and the residual |cos| MAX swings
    0.5277-0.5297 -- the EXTREMA are the definition-sensitive statistics.
    The booked "min +0.254" is reproduced by mean-of-RAW-rows; the
    registered arm here uses mean-of-UNIT-rows.
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
        rn = np.linalg.norm(R, axis=1, keepdims=True)
        assert rn.min() > 1e-12, "a key is parallel to u; residual undefined"
        Rn = R / rn
        C = (Rn @ Rn.T)[np.triu_indices(len(W), 1)]
        out[name] = (u, {
            "proj_mean": float(p.mean()), "proj_min": float(p.min()),
            "proj_max": float(p.max()), "proj_std": float(p.std()),
            # SIGNED extrema too: AUDIT-0802 item (8) complained that
            # v4_router.jsonl stored only absolute cosines, so the
            # all-pairs-positive result was unrecoverable. Logging |cos|
            # alone here would re-introduce exactly that defect.
            "resid_cos_min": float(C.min()), "resid_cos_max": float(C.max()),
            "resid_cos_mean": float(C.mean()),
            "resid_abscos_mean": float(np.abs(C).mean()),
            "resid_abscos_max": float(np.abs(C).max()),
            "resid_frac_positive": float((C > 0).mean())})
    return U, out


def score(X, W, bias):
    """Vendor scores for selection: sqrt(softplus(X W^T)) + bias."""
    z = X @ W.T
    # stable softplus; float64 throughout so a near-tie is not decided by
    # rounding. The vendor computes in float32, so set_agreement is a
    # property of a float64 idealization -- a disclosed difference.
    s = np.sqrt(np.logaddexp(0.0, z))
    return s if bias is None else s + bias


def topk_sets(S, k):
    """Top-k indices per row (sorted, so the sets are canonical)."""
    idx = np.argpartition(-S, k - 1, axis=1)[:, :k]
    return np.sort(idx, axis=1), S.argmax(axis=1)


def agreement(X, W, Wd, bias, k):
    kr, ar = topk_sets(score(X, W, bias), k)
    kd, ad = topk_sets(score(X, Wd, bias), k)
    inter = np.array([len(np.intersect1d(r, q, assume_unique=True))
                      for r, q in zip(kr, kd)])
    return {"set_agreement": float(inter.mean() / k),
            "frac_identical_set": float((inter == k).mean()),
            "rank1_agreement": float((ar == ad).mean())}


def main():
    os.makedirs("logs/opus", exist_ok=True)
    sha, topk, n_exp_cfg, n_hash = vendor_scoring()
    print(f"[Rd] vendor model.py sha {sha[:16]} | topk {topk} | "
          f"n_routed {n_exp_cfg} | hash layers {n_hash}", flush=True)
    sink, measured = open(OUT, "a"), 0
    for shard, layer in zip(SHARDS, LAYERS):
        # layers < num_hash_layers route by tid2eid lookup and have NO
        # bias and NO top-k; a set-agreement table for them would be a
        # fabrication (reviewer catch).
        assert layer >= n_hash, f"layer {layer} is hash-routed, not top-k"
        got = read_router(shard, layer)
        assert got is not None, f"no ffn.gate for layer {layer} in shard {shard}"
        W, bias, _ = got
        n, d = W.shape
        assert n == n_exp_cfg, f"{n} keys, config says {n_exp_cfg}"
        assert bias is not None, f"layer {layer} has no gate.bias"
        measured += 1

        U, dirs = shared_directions(W)
        u = dirs["unit_mean"][0]           # the registered definition
        c = W @ u                          # per-key shared coefficient
        p = U @ u                          # ... normalized
        W_rank1 = W - np.outer(c, u)
        W_level = W - np.outer(np.full(n, c.mean()), u)
        # NON-VACUOUS guards. `Wd @ u ~= 0` is true for ANY normalized u,
        # so it certifies nothing (it passed for a RANDOM u -- the exact
        # failure that would void the null). These check that the removed
        # energy is what the decomposition predicts, and that u is the
        # direction the keys actually share.
        assert abs(np.linalg.norm(u) - 1) < 1e-12, "u is not a unit vector"
        assert p.min() > 0, "u is not shared by every key"
        removed = np.linalg.norm(W)**2 - np.linalg.norm(W_rank1)**2
        assert abs(removed - float(c @ c)) < 1e-6 * abs(removed), \
            "rank-1 deflation removed the wrong energy"
        # LEVEL removes the common part and KEEPS each expert's deviation,
        # so the surviving coefficient is c - cbar: mean zero, same spread.
        cl = W_level @ u
        assert abs(cl.mean()) < 1e-9 * abs(c.mean()), "level not removed"
        assert abs(cl.std() - c.std()) < 1e-9 * c.std(), "deviation altered"

        # The LEVEL, measured in the weights -- no draw, no seed. The
        # first version sampled z.mean(), whose expectation is 0.
        wbar, wbar1, wbarL = (np.linalg.norm(M.mean(axis=0))
                              for M in (W, W_rank1, W_level))
        gain = np.sqrt(1 - p**2)           # per-expert logit-scale cut
        base = {"layer": layer, "n_experts": n, "dim": d, "topk": topk,
                "ndraw": NDRAW, "model_py_sha": sha[:16],
                "decomposition": {k: v for k, (_, v) in dirs.items()},
                "mean_row_norm_raw": float(wbar),
                "mean_row_norm_rank1": float(wbar1),
                "mean_row_norm_level": float(wbarL),
                "shared_c_mean": float(c.mean()), "shared_c_sd": float(c.std()),
                "shared_c_cv": float(c.std() / abs(c.mean())),
                "gain_min": float(gain.min()), "gain_max": float(gain.max()),
                "gain_sd": float(gain.std()), "bias_present": bias is not None}
        print(f"[Rd] L{layer} level in the WEIGHTS: ||mean row|| "
              f"{wbar:.4f} -> {wbar1:.4f} rank1 ({100*(1-wbar1/wbar):.1f}% "
              f"removed) / {wbarL:.4f} level | per-expert gain "
              f"{gain.min():.3f}-{gain.max():.3f}", flush=True)

        # PAIRED arms: one draw block per layer, reused for every scale
        # and every deflation. The first version threaded one rng through
        # the scale loop, so the REGISTERED row's numbers depended on how
        # many unregistered arms ran first (0.9744 with the sweep, 0.9705
        # alone) and the row could not be regenerated from its own fields.
        X0 = np.random.default_rng(SEED + layer).standard_normal((NDRAW, d))
        for sc in SCALES:
            X = X0 * sc
            for arm, Wd in (("rank1", W_rank1), ("level", W_level)):
                row = dict(base, arm=arm, scale=sc, shift=0.0,
                           seed=SEED + layer,
                           registered=(sc == 1.0 and arm == "rank1"),
                           **agreement(X, W, Wd, bias, topk))
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                tag = "REGISTERED" if row["registered"] else "unregistered"
                print(f"[Rd] L{layer} {arm:5s} scale {sc:<5g} {tag:12s} "
                      f"set {row['set_agreement']:.4f} | identical "
                      f"{row['frac_identical_set']:.4f} | rank1 "
                      f"{row['rank1_agreement']:.4f}", flush=True)
        # THE FENCE ARM. Scaling x scales the shared term and the contrast
        # together, so the scale sweep tests nothing about isotropy. The
        # hazard is a MEAN along u, which only this varies.
        for m in SHIFTS:
            X = X0 + m * u
            row = dict(base, arm="rank1", scale=1.0, shift=float(m),
                       seed=SEED + layer, registered=False,
                       proj_x_mean=float((X @ u).mean()),
                       **agreement(X, W, W_rank1, bias, topk))
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            print(f"[Rd] L{layer} SHIFT m={m:<5g} <u,x>={row['proj_x_mean']:+7.2f}"
                  f" set {row['set_agreement']:.4f} | identical "
                  f"{row['frac_identical_set']:.4f} | rank1 "
                  f"{row['rank1_agreement']:.4f}", flush=True)
        for name, (_, d_) in dirs.items():
            star = " <- registered" if name == "unit_mean" else ""
            print(f"[Rd] L{layer} u={name:9s} proj mean {d_['proj_mean']:+.4f} "
                  f"min {d_['proj_min']:+.4f} max {d_['proj_max']:+.4f} | "
                  f"residual cos [{d_['resid_cos_min']:+.4f}, "
                  f"{d_['resid_cos_max']:+.4f}] {d_['resid_frac_positive']:.1%}"
                  f" positive{star}", flush=True)
    sink.close()
    # A run that measured nothing must not look like a run that passed.
    assert measured == len(LAYERS), f"measured {measured}/{len(LAYERS)} layers"


if __name__ == "__main__":
    main()

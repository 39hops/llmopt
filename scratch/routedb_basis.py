"""ROUTE-BASIS census (EXPERTDB rung 3, desk, observation-only;
representation, information boundary, priors, and metrics FROZEN
here before any value is read). Frozen rung-1 module imported for
traces/expert bytes, never edited.

INFORMATION BOUNDARY (the deployability law, frozen): a predictor
may use (i) anything fit on TRAIN prompts (even ids, all six
slices pooled) and (ii) the test prompt's OWN PREFILL routing
only. It may never see the decode demand it predicts.

REPRESENTATION: per prompt, the PREFILL routing signature = the
6144-dim (48 layers x 128 experts) selection-count vector,
L1-normalized. Signatures are what the census decomposes and what
the latent predictor projects.

DESCRIPTIVE LEG (train signatures): PCA spectrum + effective rank
(exp(entropy of normalized eigenvalues)); NMF at r=16
(multiplicative updates, seed "route-basis-0", 200 iters);
per-component domain composition (interpretation AFTER fitting);
cross-domain nearest-neighbor rate of TEST signatures against
TRAIN (cosine).

PREDICTIVE LEG — prefill -> held-out decode demand. Each predictor
emits a per-layer top-K residency table (K in {32, 48}) for each
TEST prompt; score = coverage of the prompt's actual decode
selections (1 - static miss rate) and implied MB/decode-token at
the rung-1 per-expert bytes. Predictors:
  GLOBAL         train pooled DECODE demand top-K (one table)
  NAMED-DOMAIN   train DECODE demand of the prompt's own named
                 slice
  RAW-PREFILL    the test prompt's own prefill counts top-K
  LATENT-MIX     project the test prefill signature onto the NMF
                 basis (nonneg least squares by multiplicative
                 updates); mix the components' train DECODE
                 profiles by the weights; top-K of the mixture
  ORACLE         the test prompt's own decode top-K (clairvoyant
                 upper bound, not implementable)
Reference rows: rung-2 held-out warm LRU@K32/K48 dynamic numbers
are quoted for context in the booking, not recomputed here.

Receipt: logs/routedb/basis_receipt.json (refuse-if-exists).

    .venv/bin/python scratch/routedb_basis.py              (Mac desk)
"""
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

_spec = importlib.util.spec_from_file_location(
    "routedb_replay", Path(__file__).parent / "routedb_replay.py")
r1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r1)

N_LAYERS, N_EXP, D = 48, 128, 48 * 128
KS = [32, 48]
R_NMF = 16
OUT = Path("logs/routedb/basis_receipt.json")


def phase2(ph):
    return "prefill" if ph in ("prefill", "prompt_tail") else "decode"


def signatures():
    """per (domain, prompt): prefill vec, decode vec (counts)."""
    pre = {}
    dec = {}
    for dom, path in r1.TRACES.items():
        for p, _, ph, layers in r1.load_events(path):
            key = (dom, p)
            tgt = pre if phase2(ph) == "prefill" else dec
            v = tgt.setdefault(key, np.zeros(D, np.float64))
            for li, tk in layers.items():
                for e in tk:
                    v[li * N_EXP + e] += 1
    return pre, dec


def nmf(X, r, iters=200, seed="route-basis-0"):
    import hashlib
    rng = np.random.default_rng(
        int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16))
    W = rng.random((X.shape[0], r)) + 0.1
    Hm = rng.random((r, X.shape[1])) + 0.1
    for _ in range(iters):
        Hm *= (W.T @ X) / (W.T @ W @ Hm + 1e-9)
        W *= (X @ Hm.T) / (W @ Hm @ Hm.T + 1e-9)
    return W, Hm


def project(x, Hm, iters=200):
    w = np.full(Hm.shape[0], 0.1)
    for _ in range(iters):
        w *= (Hm @ x) / (Hm @ Hm.T @ w + 1e-9)
    return w


def topk_table(vec, k):
    t = []
    for li in range(N_LAYERS):
        seg = vec[li * N_EXP:(li + 1) * N_EXP]
        t.append(set(int(i) for i in np.argsort(-seg)[:k]))
    return t


def coverage(table, dec_vec):
    hit = tot = 0
    for li in range(N_LAYERS):
        seg = dec_vec[li * N_EXP:(li + 1) * N_EXP]
        for e in np.nonzero(seg)[0]:
            n = seg[e]
            tot += n
            if int(e) in table[li]:
                hit += n
    return hit, tot


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/routedb_basis.py",
                              "scratch/routedb_replay.py"])
    eb = r1.expert_bytes()
    pre, dec = signatures()
    keys = sorted(pre)
    train = [k for k in keys if k[1] % 2 == 0]
    test = [k for k in keys if k[1] % 2 == 1 and k in dec]

    def l1(v):
        return v / max(v.sum(), 1)
    Xtr = np.stack([l1(pre[k]) for k in train])
    # descriptive: PCA spectrum + effective rank
    Xc = Xtr - Xtr.mean(0)
    sv = np.linalg.svd(Xc, compute_uv=False)
    lam = sv**2 / max((sv**2).sum(), 1e-12)
    lam = lam[lam > 1e-12]
    eff_rank = float(np.exp(-(lam * np.log(lam)).sum()))
    W, Hm = nmf(Xtr, R_NMF)
    # per-component domain composition (post-fit interpretation)
    comp_dom = []
    for c in range(R_NMF):
        w = W[:, c]
        by = defaultdict(float)
        for i, (dom, _) in enumerate(train):
            by[dom] += w[i]
        s = sum(by.values()) or 1
        comp_dom.append({d: round(v / s, 3) for d, v in
                         sorted(by.items())})
    # cross-domain NN rate of test v train (cosine)
    Xtr_n = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)
    cross = 0
    for k in test:
        x = l1(pre[k])
        x = x / (np.linalg.norm(x) + 1e-12)
        j = int((Xtr_n @ x).argmax())
        cross += train[j][0] != k[0]
    cross_rate = round(cross / max(len(test), 1), 4)

    # train decode profiles
    g_dec = np.zeros(D)
    dom_dec = defaultdict(lambda: np.zeros(D))
    for k in train:
        if k in dec:
            g_dec += dec[k]
            dom_dec[k[0]] += dec[k]
    comp_dec = np.zeros((R_NMF, D))
    wsum = np.zeros(R_NMF)
    for i, k in enumerate(train):
        if k in dec:
            comp_dec += np.outer(W[i], dec[k])
            wsum += W[i]
    comp_dec /= (wsum[:, None] + 1e-9)

    preds = ("GLOBAL", "NAMED-DOMAIN", "RAW-PREFILL", "LATENT-MIX",
             "ORACLE")
    out = {K: {p: [0, 0] for p in preds} for K in KS}
    for k in test:
        xin = l1(pre[k])
        w = project(xin, Hm)
        mix = w @ comp_dec
        for K in KS:
            tables = {
                "GLOBAL": topk_table(g_dec, K),
                "NAMED-DOMAIN": topk_table(dom_dec[k[0]], K),
                "RAW-PREFILL": topk_table(xin, K),
                "LATENT-MIX": topk_table(mix, K),
                "ORACLE": topk_table(dec[k], K),
            }
            for p, t in tables.items():
                h, tot = coverage(t, dec[k])
                out[K][p][0] += h
                out[K][p][1] += tot
    summary = {}
    for K in KS:
        summary[f"K{K}"] = {}
        for p in preds:
            h, tot = out[K][p]
            miss = 1 - h / max(tot, 1)
            mb = miss * 8 * N_LAYERS * eb / 1e6
            summary[f"K{K}"][p] = {
                "decode_coverage": round(h / max(tot, 1), 4),
                "mb_per_token_upper": round(mb, 1)}
    rcpt = {"note": "ROUTE-BASIS census: prefill->decode "
                    "information boundary; PCA/NMF descriptives + "
                    "four implementable priors v oracle",
            "start": START, "completion_commit": completion_commit(),
            "n_train": len(train), "n_test": len(test),
            "pca_effective_rank": round(eff_rank, 2),
            "pca_top10_eigfrac": [round(float(x), 4)
                                  for x in lam[:10]],
            "nmf_r": R_NMF,
            "nmf_component_domain_mix": comp_dom,
            "cross_domain_nn_rate": cross_rate,
            "predictors": summary}
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    print(f"eff_rank {eff_rank:.2f} crossNN {cross_rate}")
    for K in KS:
        for p in preds:
            s = summary[f"K{K}"][p]
            print(f"K{K} {p:12s} cov {s['decode_coverage']:.4f} "
                  f"mb_ub {s['mb_per_token_upper']}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

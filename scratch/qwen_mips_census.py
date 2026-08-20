"""QWEN-MIPS-CENSUS-0 driver: exact Cauchy-Schwarz branch-and-bound
top-256 over k-means-clustered decoded W4 lm_head rows (PRE-REG in
docs/RESULTS.md; machine projection docs/preregs/
qwen-mips-census-0.json). Mac, CPU, plain .venv — no model load.

Recipe frozen in the prereg: K grid {256, 1024, 4096}, k-means++
minibatch (8192) k-means, string seed "mips-census-0", 25
iterations, fp64 centers, radii and bounds fp64 with conservative
inflation. Exactness is a precondition: the mechanism fixture
(max member score <= U_j for every query x cluster x K) and the
small-vocab (first 8192 rows) 100% brute-equality qualification
both run BEFORE the census; any failure exits INSTRUMENT-INVALID.

    .venv/bin/python scratch/qwen_mips_census.py            (mac)
    SMOKE=1 ...  (smoke path logs/qwenmips_smoke; first 8192 rows
        as the whole vocab, K grid {16,64}, 24 queries)

Receipts -> logs/qwenmips/mips_receipt.json +
mips_observations.json (refuse-if-exists); the index npz stays
untracked, sha-pinned in the receipt.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.lab.qcodec_fast import W4Rows  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
ART = os.path.expanduser(os.environ.get("ART_DIR",
                                        "~/qwen_whole0t/A"))
QUERIES = "logs/qwencheapread/census_arrays_A.npz"
QUERIES_SHA = ("8912d27533dbec860891b3cded8fc445c34e3040ece2003"
               "d10e59ea6af180549")
OUT = "logs/qwenmips_smoke" if SMOKE else "logs/qwenmips"
K_GRID = (16, 64) if SMOKE else (256, 1024, 4096)
TOPK = 256
SEED_TAG = "mips-census-0"
KM_ITERS = 25
KM_BATCH = 8192
RAD_INFLATE = 1 + 1e-12
U_EPS = 1e-3  # ADDITIVE fp64 slack: covers fp32 score rounding
# above a tight fp64 bound (measured excursion 1.3e-5 in smoke);
# multiplicative inflation is sign-broken for negative bounds —
# AMENDMENT QWEN-MIPS-CENSUS-0-BOUND
SMALLV = 8192


def string_seed(tag):
    return int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)


def load_head():
    man = json.loads(open(os.path.join(ART, "manifest.json"),
                          "rb").read().decode())
    e = man["lm_head.weight"]
    fh = open(os.path.join(ART, e["shard"] + ".bin"), "rb")
    fh.seek(e["off"])
    buf = fh.read(e["len"])
    rows = W4Rows(buf, e["shape"])
    R, C = e["shape"]
    W = np.empty((R, C), np.float32)
    step = 16384
    for lo in range(0, R, step):
        W[lo:lo + step] = rows.rows(lo, min(lo + step, R))
    bytes_per_row = len(buf) / R
    return W, bytes_per_row


def kmeans(W, K, seed):
    """Minibatch k-means, k-means++ init on a sample, fp32 assign /
    fp64 centers."""
    rng = np.random.default_rng(seed)
    n = W.shape[0]
    # k-means++ on a bounded sample for tractability
    samp = W[rng.choice(n, size=min(n, 65536), replace=False)]
    centers = [samp[rng.integers(len(samp))]]
    d2 = ((samp - centers[0]) ** 2).sum(1)
    for _ in range(K - 1):
        p = d2 / d2.sum()
        c = samp[rng.choice(len(samp), p=p)]
        centers.append(c)
        d2 = np.minimum(d2, ((samp - c) ** 2).sum(1))
    C = np.stack(centers).astype(np.float64)
    for _ in range(KM_ITERS):
        idx = rng.choice(n, size=KM_BATCH, replace=False)
        X = W[idx].astype(np.float64)
        a = np.argmax(X @ C.T - 0.5 * (C * C).sum(1), axis=1)
        for j in np.unique(a):
            pts = X[a == j]
            C[j] = C[j] * 0.5 + pts.mean(0) * 0.5
    # final full assignment (chunked)
    assign = np.empty(n, np.int32)
    cc = -0.5 * (C * C).sum(1)
    for lo in range(0, n, 16384):
        z = W[lo:lo + 16384].astype(np.float64) @ C.T + cc
        assign[lo:lo + 16384] = z.argmax(1)
    # recentre on final assignment, then radii
    for j in range(K):
        m = assign == j
        if m.any():
            C[j] = W[m].astype(np.float64).mean(0)
    radii = np.zeros(K, np.float64)
    for j in range(K):
        m = assign == j
        if m.any():
            d = np.linalg.norm(W[m].astype(np.float64) - C[j],
                               axis=1)
            radii[j] = d.max() * RAD_INFLATE
    return C, radii, assign


def certify_query(zrow, h, C, radii, groups):
    """Branch-and-bound for one query (fp32 scores, fp64 bounds).
    zrow is the precomputed full fp32 score vector for this query —
    IDENTICAL numbers to scoring rows on visit (same W, same h, one
    matmul); visitation is counted for exactly the rows the
    algorithm would touch. Returns (top ids, rows visited,
    clusters visited)."""
    hn = np.linalg.norm(h.astype(np.float64))
    U = h.astype(np.float64) @ C.T + hn * radii + U_EPS
    order = np.argsort(-U)
    best_scores = np.full(TOPK, -np.inf, np.float32)
    best_ids = np.full(TOPK, -1, np.int64)
    visited_rows = 0
    visited_clusters = 0
    for j in order:
        tau = best_scores[-1]
        # STRICT inequality: on tau == U[j] an unseen equal-score
        # lower-id row could still enter the top-256 under the
        # frozen tie rule, so equality must scan (tie-safety,
        # external review pre-rerun)
        if tau > U[j]:
            break
        rows = groups[j]
        if rows.size == 0:
            continue
        z = zrow[rows]
        visited_rows += rows.size
        visited_clusters += 1
        alls = np.concatenate([best_scores, z])
        alli = np.concatenate([best_ids, rows])
        # stable exact top-K with the frozen tie rule (score desc,
        # then lower id)
        sel = np.lexsort((alli, -alls))[:TOPK]
        best_scores = alls[sel].copy()
        best_ids = alli[sel].copy()
    return best_ids, visited_rows, visited_clusters


def topk_ids_tie(z_row):
    return np.lexsort((np.arange(z_row.size), -z_row))[:TOPK]


def main():
    os.makedirs(OUT, exist_ok=True)
    rcpt_path = os.path.join(OUT, "mips_receipt.json")
    obs_path = os.path.join(OUT, "mips_observations.json")
    guards = [rcpt_path, obs_path] + [
        os.path.join(OUT, f"mips_index_K{K}.npz") for K in K_GRID]
    for p in guards:
        if os.path.exists(p):
            raise SystemExit(f"REFUSING: {p} exists")
    qsha = hashlib.sha256(open(QUERIES, "rb").read()).hexdigest()
    if qsha != QUERIES_SHA:
        raise SystemExit(f"REFUSING: query npz sha {qsha[:12]} != "
                         f"pinned {QUERIES_SHA[:12]}")
    START = start_provenance(
        ["scratch/qwen_mips_census.py", "llmopt/lab/qcodec_fast.py",
         "llmopt/lab/provenance.py",
         "docs/preregs/qwen-mips-census-0.json"],
        artifacts={"A": ART})
    q = np.load(QUERIES)
    h_c, h_p = q["h_corpus"], q["h_prefix"]
    if SMOKE:
        h_c, h_p = h_c[:16], h_p[:8]
    H = np.concatenate([h_c, h_p])
    pop = ["corpus_X"] * len(h_c) + ["prefix_K"] * len(h_p)
    t0 = time.time()
    W, bytes_per_row = load_head()
    if SMOKE:
        W = W[:SMALLV]
    n = W.shape[0]
    print(f"[mc] head decoded {W.shape} {time.time()-t0:.0f}s "
          f"bytes/row {bytes_per_row:.1f}", flush=True)
    # full score matrix once (the scorer, batched), then brute
    # ground truth from it
    t0 = time.time()
    Z = np.empty((len(H), n), np.float32)
    for lo in range(0, len(H), 64):
        Z[lo:lo + 64] = H[lo:lo + 64] @ W.T
    gt = {qi: topk_ids_tie(Z[qi]) for qi in range(len(H))}
    print(f"[mc] scores + brute ground truth {time.time()-t0:.0f}s",
          flush=True)
    # REGISTERED SMALL-VOCAB QUALIFICATION (prereg mechanism
    # fixture, second clause): first SMALLV rows as the whole
    # vocab, ALL queries, EVERY registered K, 100% certified-v-
    # brute top-256 equality through the SAME production path.
    # Any miss exits INSTRUMENT-INVALID before the census.
    qual = {}
    if not SMOKE:
        Wq = W[:SMALLV]
        Zq = Z[:, :SMALLV]
        gtq = {qi: topk_ids_tie(Zq[qi]) for qi in range(len(H))}
        for K in K_GRID:
            Cq, rq, aq = kmeans(Wq, K, string_seed(
                f"{SEED_TAG}-qual-K{K}"))
            gq = [np.where(aq == j)[0] for j in range(K)]
            okn = 0
            for qi in range(len(H)):
                ids, _, _ = certify_query(Zq[qi], H[qi], Cq, rq,
                                          gq)
                okn += int(np.array_equal(np.sort(ids),
                                          np.sort(gtq[qi])))
            qual[str(K)] = {"exact": okn, "n": len(H)}
            print(f"[mc] QUAL smallv K={K} exact {okn}/{len(H)}",
                  flush=True)
            if okn != len(H):
                rc = {"start": START, "INSTRUMENT-INVALID":
                      f"small-vocab qualification failed at K={K}: "
                      f"{okn}/{len(H)}", "qual": qual}
                with open(rcpt_path, "w") as f:
                    f.write(json.dumps(rc, indent=1) + "\n")
                raise SystemExit("INSTRUMENT-INVALID")
    results = {}
    index_shas = {}
    for K in K_GRID:
        t0 = time.time()
        C, radii, assign = kmeans(W, K, string_seed(
            f"{SEED_TAG}-K{K}"))
        groups = [np.where(assign == j)[0] for j in range(K)]
        ip = os.path.join(OUT, f"mips_index_K{K}.npz")
        np.savez_compressed(ip, centers=C, radii=radii,
                            assign=assign)
        index_shas[str(K)] = hashlib.sha256(
            open(ip, "rb").read()).hexdigest()
        print(f"[mc] K={K} clustered {time.time()-t0:.0f}s",
              flush=True)
        # mechanism fixture: max member score <= U_j for EVERY
        # query x cluster (segment max over the precomputed scores)
        t0 = time.time()
        fix_ok = True
        M = np.full((K, len(H)), -np.inf, np.float64)
        np.maximum.at(M, assign, Z.T.astype(np.float64))
        hn = np.linalg.norm(H.astype(np.float64), axis=1)
        Uall = (H.astype(np.float64) @ C.T
                + hn[:, None] * radii[None, :] + U_EPS)
        bad = M.T > Uall
        if bad.any():
            fix_ok = False
            qi, j = map(int, np.argwhere(bad)[0])
            print(f"[mc] FIXTURE FAIL K={K} q={qi} cluster={j}",
                  flush=True)
        print(f"[mc] K={K} bound fixture "
              f"{'PASS' if fix_ok else 'FAIL'} "
              f"{time.time()-t0:.0f}s", flush=True)
        if not fix_ok:
            rc = {"start": START, "INSTRUMENT-INVALID":
                  f"bound fixture failed at K={K}"}
            with open(rcpt_path, "w") as f:
                f.write(json.dumps(rc, indent=1) + "\n")
            raise SystemExit("INSTRUMENT-INVALID")
        # census
        t0 = time.time()
        exact_n = 0
        rows_v = []
        for qi in range(len(H)):
            ids, nr, nc = certify_query(Z[qi], H[qi], C, radii,
                                        groups)
            rows_v.append(nr)
            if np.array_equal(np.sort(ids), np.sort(gt[qi])):
                exact_n += 1
        rows_v = np.array(rows_v)
        res = {}
        for tag in ("corpus_X", "prefix_K"):
            sel = np.array([p == tag for p in pop])
            rv = rows_v[sel]
            frac = rv / n
            # index bytes: fp64 centroids + radius + posting
            # offset per cluster, plus 4-byte row ids for the
            # posting lists actually visited (modeled, descriptive)
            idx_bytes = K * (W.shape[1] * 8 + 8 + 8)
            res[tag] = {
                "n_queries": int(sel.sum()),
                "rows_visited": {
                    "q50": float(np.quantile(rv, .5)),
                    "q90": float(np.quantile(rv, .9)),
                    "q95": float(np.quantile(rv, .95)),
                    "max": int(rv.max())},
                "fraction_q50": float(np.quantile(frac, .5)),
                "modeled_bytes_q50": float(
                    np.quantile(rv, .5) * (bytes_per_row + 4)
                    + idx_bytes),
                "index_bytes": idx_bytes,
                "modeled_ratio_q50": float(
                    (np.quantile(rv, .5) * (bytes_per_row + 4)
                     + idx_bytes) / (n * bytes_per_row))}
        results[str(K)] = {"exact": exact_n, "n": len(H),
                           "per_pop": res,
                           "wall_s": round(time.time() - t0, 1)}
        print(f"[mc] K={K} exact {exact_n}/{len(H)} corpus q50 "
              f"frac {res['corpus_X']['fraction_q50']:.3f} "
              f"prefix q50 frac "
              f"{res['prefix_K']['fraction_q50']:.3f} "
              f"{time.time()-t0:.0f}s", flush=True)
    rc = {"start": START, "completion_commit": completion_commit(),
          "smoke": SMOKE, "n_vocab": n,
          "bytes_per_row_w4": bytes_per_row,
          "queries_npz_sha256": qsha,
          "smallv_qualification": qual,
          "index_npz_sha256": index_shas,
          "results": results,
          "note": "byte figures are MODELED visitation (index + "
                  "visited-row w4 payload), never physical memory "
                  "traffic; exact = exact under the offline "
                  "decoded-W4 fp32 scorer"}
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rc, indent=1) + "\n")
    # smoke exercises the observations/adjudication block too
    # (receipt-audit S2): it writes into the smoke dir and skips
    # only the prereg adjudication (whose bars name the real
    # populations)
    total_cells = sum(r["n"] for r in results.values())
    exact_cells = sum(r["exact"] for r in results.values())
    fracs = {K: r["per_pop"]["corpus_X"]["fraction_q50"]
             for K, r in results.items()}
    bestK = min(fracs, key=fracs.get)
    obs = {
        "note": "counts from the receipt's results block; best K = "
                "argmin corpus_X fraction_q50 per the registered "
                "selection metric",
        "measurement_valid": True,
        "arms": {"A": {"admissible": True,
                       "reason": f"query npz sha pinned; artifact "
                                 f"identity recorded; "
                                 f"{total_cells} query-K cells"}},
        "measurements": {
            "1": {"value": exact_cells,
                  "metric": "n_query_K_cells_exact",
                  "population": "queries:454 x K:3 (full vocab)",
                  "aggregation": "count",
                  "provenance": f"per-K exact counts "
                                f"{ {K: r['exact'] for K, r in results.items()} }"},
            "2": {"value": fracs[bestK],
                  "metric":
                      "corpusX_q50_rows_visited_fraction_bestK",
                  "population": "corpus_X:356 at the best K of "
                                "the frozen grid",
                  "aggregation": "median fraction",
                  "provenance": f"best K {bestK}; all K "
                                f"{fracs}"},
            "refute:corpusX_q50_fraction_minK": {
                "value": min(fracs.values()),
                "metric":
                    "corpusX_q50_rows_visited_fraction_bestK",
                "population": "corpus_X:356 at the best K of "
                              "the frozen grid",
                "aggregation": "median fraction",
                "provenance": "min over the K grid, read by the "
                              "refutation predicate (>= 0.90 at "
                              "EVERY K refutes)"}}}
    with open(obs_path, "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    if SMOKE:
        print("[mc] smoke done (obs written to smoke dir)",
              flush=True)
        return 0
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation,
                                   load as load_prereg)
    doc = load_prereg("docs/preregs/qwen-mips-census-0.json")
    outcomes = adjudicate_prereg(doc, obs)
    ref = adjudicate_refutation(doc, obs, bar_outcomes=outcomes)
    print(json.dumps({"bars": {o.bar_id: o.outcome
                               for o in outcomes},
                      "refutation": ref, "fracs": fracs,
                      "bestK": bestK}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""K2-HORIZON-TRANSPORT-0 wall-clock audit (Artin GO 2026-09-06, before the
7B prereg is frozen). Three probes, none opening any 7B weight:

  svdbench   synthetic tensors at the exact manifest shapes of a size
             (SIZE env), the REGISTERED structural routine
             (k2h_transport.tensor_row with structural=True) timed per
             tensor class, and a candidate smaller-side decomposition
             (Gram eigendecomposition on the short side, left vectors
             recovered as d V / S) timed and checked for parity on the same
             synthetic deltas: effrank_d, top1_share_d, ipr_left_d,
             ipr_right_d within 1e-6 relative, stable ranks and Hill alpha
             unchanged (they do not go through the candidate). Parity on
             synthetic tensors is a NECESSARY condition; adoption additionally
             requires reproducing booked 3.7B rows (parity3b).
  parity3b   re-download one booked 3.7B pair by manifest commit, recompute
             the structural rows with the registered routine AND the
             candidate, compare against the booked census rows (exact match
             for the registered routine, 1e-6 relative for the candidate).
  dlprobe    aggregate download throughput on pinned immutable shards of a
             tag with W in {3, 6, 9} workers (N shards each, deleted after).
  batchprobe on a booked gated tag re-downloaded by commit: the ladder at
             BATCH in {8, 16, 24, 32}, generated-token digest and oracle
             parity against the booked gate rows for that tag at B = 8.

Every probe writes a receipt under logs/k2h/wallaudit/<probe>.json.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "logs/k2h/wallaudit")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


K = _load("k2h", "scratch/k2h_stagecensus.py")


def write(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"{name}.json")
    if os.path.exists(p):
        raise SystemExit(f"REFUSING: {p} exists")
    json.dump(obj, open(p, "w"), indent=1, default=str)
    print("wrote", p)


# ---------------------------------------------------------------- candidate decomposition
def gram_stats(d):
    """Smaller-side decomposition of delta d (m x n): eigendecompose the
    Gram matrix on the short side, singular values = sqrt(eigs), the
    short-side singular vectors are the eigenvectors, the long-side
    vectors are d v / s. Returns the same four delta statistics as the
    registered routine."""
    d = d.double()  # float64 Gram path: the float32 Gram squares the condition number and misses the 1e-6 parity criterion
    m, n = d.shape
    if n <= m:
        G = d.T @ d
        evals, V = torch.linalg.eigh(G)
        order = torch.argsort(evals, descending=True)
        evals, V = evals[order].clamp_min(0), V[:, order]
        S = evals.sqrt()
        k = min(K.IPR_K, S.shape[0])
        U = (d @ V[:, :k]) / S[:k].clamp_min(1e-30)
        left, right = U, V[:, :k]
    else:
        G = d @ d.T
        evals, U = torch.linalg.eigh(G)
        order = torch.argsort(evals, descending=True)
        evals, U = evals[order].clamp_min(0), U[:, order]
        S = evals.sqrt()
        k = min(K.IPR_K, S.shape[0])
        V = (d.T @ U[:, :k]) / S[:k].clamp_min(1e-30)
        left, right = U[:, :k], V
    Sn = S.numpy()
    return {"effrank_d": K.entropy_effrank(Sn), "top1_share_d": float(Sn[0] ** 2 / (Sn ** 2).sum()),
            "ipr_left_d": K.ipr(left.numpy()), "ipr_right_d": K.ipr(right.numpy())}


def registered_stats(d):
    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    S = S.numpy()
    k = min(K.IPR_K, S.shape[0])
    return {"effrank_d": K.entropy_effrank(S), "top1_share_d": float(S[0] ** 2 / (S ** 2).sum()),
            "ipr_left_d": K.ipr(U[:, :k].numpy()), "ipr_right_d": K.ipr(Vh[:k, :].T.numpy())}


def shapes_for(size):
    man = json.load(open(os.path.join(ROOT, "docs/preregs/k2h-transport-0.manifest.json")))["sizes"][size]
    c = man["selected"]["pretrain_100000"]["config"]
    h, f, v = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    hd = 128
    return {"q_proj": (c["num_attention_heads"] * hd, h), "k_proj": (c["num_key_value_heads"] * hd, h), "v_proj": (c["num_key_value_heads"] * hd, h),
            "o_proj": (h, c["num_attention_heads"] * hd), "gate_proj": (f, h), "up_proj": (f, h), "down_proj": (h, f), "embed": (v, h), "lm_head": (v, h)}


def svdbench():
    size = os.environ["SIZE"]
    g = torch.Generator().manual_seed(0)
    rec = {"size": size, "shapes": {}, "per_class": {}, "parity": {}, "torch_threads": torch.get_num_threads()}
    T = _load("k2h_transport_bench", "scratch/k2h_transport.py") if os.environ.get("SIZE") else None
    for cls, (m, n) in shapes_for(size).items():
        rec["shapes"][cls] = [m, n]
        A = torch.randn(m, n, generator=g) * 0.02
        # structured delta: low-rank + noise so the statistics are non-trivial
        d = (torch.randn(m, 4, generator=g) @ torch.randn(4, n, generator=g)) * 1e-3 + torch.randn(m, n, generator=g) * 1e-4
        B = A + d
        t = time.time()
        row, _ = T.tensor_row(f"bench.{cls}", A.bfloat16(), B.bfloat16(), True, None)
        t_reg = time.time() - t
        d32 = B.bfloat16().float() - A.bfloat16().float()
        t = time.time()
        reg = registered_stats(d32)
        t_svd_only = time.time() - t
        t = time.time()
        cand = gram_stats(d32)
        t_cand = time.time() - t
        t = time.time()
        _ = torch.linalg.svdvals(B.bfloat16().float())
        t_svdvals = time.time() - t
        t = time.time()
        _ = T.power_sigma_max(A.bfloat16().float())
        t_power = time.time() - t
        par = {k: abs(cand[k] - reg[k]) / max(abs(reg[k]), 1e-12) for k in reg}
        rec["per_class"][cls] = {"registered_row_s": round(t_reg, 2), "svd_delta_s": round(t_svd_only, 2), "candidate_s": round(t_cand, 2),
                                 "svdvals_after_s": round(t_svdvals, 2), "power_iter_s": round(t_power, 3), "row_stats": {k: row.get(k) for k in ("effrank_d", "top1_share_d", "ipr_left_d", "ipr_right_d", "stable_rank_a", "hill_alpha_b")}}
        rec["parity"][cls] = {"max_rel_diff": max(par.values()), "per_stat": par, "pass_1e-6": max(par.values()) <= 1e-6}
        print(cls, (m, n), f"registered {t_reg:.1f}s (svd {t_svd_only:.1f}, svdvals {t_svdvals:.1f}, power {t_power:.2f}) candidate {t_cand:.1f}s parity {max(par.values()):.2e}", flush=True)
    per_layer = sum(rec["per_class"][c]["registered_row_s"] for c in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
    L = 36
    rec["projection"] = {"registered_per_structural_pair_s": round(per_layer * L + rec["per_class"]["embed"]["registered_row_s"] + rec["per_class"]["lm_head"]["registered_row_s"], 1),
                         "candidate_delta_only_per_pair_s": round(sum(rec["per_class"][c]["candidate_s"] for c in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")) * L + rec["per_class"]["embed"]["candidate_s"] + rec["per_class"]["lm_head"]["candidate_s"], 1),
                         "svdvals_after_per_pair_s": round(sum(rec["per_class"][c]["svdvals_after_s"] for c in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")) * L + rec["per_class"]["embed"]["svdvals_after_s"] + rec["per_class"]["lm_head"]["svdvals_after_s"], 1)}
    print(json.dumps(rec["projection"], indent=1))
    write(f"svdbench_{size}", rec)


def dlprobe():
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    size, tag, n = os.environ["SIZE"], os.environ.get("TAG", "pretrain_100000"), int(os.environ.get("N_SHARDS", "6"))
    man = json.load(open(os.path.join(ROOT, "docs/preregs/k2h-transport-0.manifest.json")))["sizes"][size]
    e = man["selected"][tag]
    shards = sorted(e["shards"])[:n * 3]
    rec = {"size": size, "tag": tag, "commit": e["commit"], "n_shards_per_probe": n, "probes": {}}
    for i, W in enumerate((3, 6, 9)):
        sub = shards[i * n:(i + 1) * n]
        d = os.path.join(ROOT, f"logs/k2h/_dlprobe_{W}")
        shutil.rmtree(d, ignore_errors=True)
        t = time.time()
        with ThreadPoolExecutor(max_workers=W) as ex:
            paths = list(ex.map(lambda s: hf_hub_download(man["repo"], s, revision=e["commit"], local_dir=d), sub))
        wall = time.time() - t
        nbytes = sum(os.path.getsize(p) for p in paths)
        ok = all(K.sha_file(p) == e["shards"][s]["sha256"] for p, s in zip(paths, sub))
        rec["probes"][W] = {"shards": sub, "bytes": nbytes, "wall_s": round(wall, 1), "MB_per_s": round(nbytes / wall / 1e6, 1), "sha_ok": ok}
        print(f"workers {W}: {nbytes / 1e9:.2f} GB in {wall:.0f} s = {nbytes / wall / 1e6:.1f} MB/s sha_ok={ok}", flush=True)
        shutil.rmtree(d, ignore_errors=True)
    write(f"dlprobe_{size}", rec)


def batchprobe():
    """Ladder at several batch sizes on one booked gated tag re-downloaded
    by commit; parity against the booked B=8 gate rows for that tag."""
    size, tag = os.environ["SIZE"], os.environ["TAG"]
    T = _load("k2h_transport_bp", "scratch/k2h_transport.py")
    LAD = _load("k2h_ladder_bp", "scratch/k2h_gateladder.py")
    booked_dir = os.path.join(ROOT, f"logs/k2h/transport_{size}")
    booked = [json.loads(l) for l in open(os.path.join(booked_dir, "gate_rows.jsonl"))]
    ref = {(r["seed"], r["tier"], r["item"]): r for r in booked if r["tag"] == tag}
    items = LAD.make_items()
    T.fetch_small(tag)
    shards, _ = T.shard_list(tag)
    T.prefetch(tag, shards)
    for s in shards:
        T.fetch_shard(tag, s)
    tok_dir = T.fetch_small(T.TOKENIZER_TAG)
    model, tok, _, _, _ = T.load_model_from_dir(T.tag_dir(tag), tok_dir)
    rec = {"size": size, "tag": tag, "commit": T.MAN["selected"][tag]["commit"], "batches": {}}
    tiers = [int(x) for x in os.environ.get("TIERS", "1,2,4").split(",")]
    for B in (8, 16, 24, 32):
        K.BATCH = B
        rows_path = os.path.join(OUT, f"batchprobe_{size}_{tag}_B{B}.jsonl")
        os.makedirs(OUT, exist_ok=True)
        rows_f = open(rows_path, "w")
        counters = {"timeout": 0, "empty": 0}
        t = time.time()
        res = {}
        for tier in tiers:
            res[tier] = LAD.run_tier(model, tok, items, tier, 0, tag, rows_f, counters)
        wall = time.time() - t
        rows_f.close()
        rows = [json.loads(l) for l in open(rows_path)]
        ident = sum(ref[(r["seed"], r["tier"], r["item"])]["gen_ids"] == r["gen_ids"] for r in rows)
        same_correct = sum(ref[(r["seed"], r["tier"], r["item"])]["correct"] == r["correct"] for r in rows)
        rec["batches"][B] = {"wall_s": round(wall, 1), "n_rows": len(rows), "gen_ids_identical": ident, "correct_identical": same_correct,
                             "digest_parity": ident == len(rows), "oracle_parity": same_correct == len(rows), "per_tier": res}
        print(f"B={B}: {wall:.0f} s, gen_ids identical {ident}/{len(rows)}, oracle identical {same_correct}/{len(rows)}", flush=True)
    del model
    shutil.rmtree(T.tag_dir(tag), ignore_errors=True)
    write(f"batchprobe_{size}_{tag}", rec)


def parity3b():
    """Recompute the structural rows of one booked 3.7B pair from the
    manifest commits and compare to the booked census rows."""
    size, a, b = os.environ["SIZE"], os.environ["TAG_A"], os.environ["TAG_B"]
    T = _load("k2h_transport_p3", "scratch/k2h_transport.py")
    booked = [json.loads(l) for l in open(os.path.join(ROOT, f"logs/k2h/transport_{size}/census.jsonl"))]
    ref = {r["tensor"]: r for r in booked if r["pair"] == f"{a}->{b}"}
    for tag in (a, b):
        T.fetch_small(tag)
        shards, _ = T.shard_list(tag)
        T.prefetch(tag, shards)
        for s in shards:
            T.fetch_shard(tag, s)
    ra, rb = T.TagReader(a), T.TagReader(b)
    keys = ("effrank_d", "top1_share_d", "ipr_left_d", "ipr_right_d", "stable_rank_a", "stable_rank_b", "hill_alpha_b", "fro_d", "rel_d")
    n_exact = n_cand_ok = n = 0
    worst_reg = worst_cand = 0.0
    t_reg = t_cand = 0.0
    for name in sorted(ref):
        A, B = ra.get(name), rb.get(name)
        t = time.time()
        row, _ = T.tensor_row(name, A, B, True, None)
        t_reg += time.time() - t
        r = ref[name]
        for k in keys:
            if k in r and r[k] is not None:
                rel = abs(row[k] - r[k]) / max(abs(r[k]), 1e-12)
                worst_reg = max(worst_reg, rel)
        if "effrank_d" in r:
            n += 1
            d32 = B.float() - A.float()
            t = time.time()
            c = gram_stats(d32)
            t_cand += time.time() - t
            rel = max(abs(c[k] - r[k]) / max(abs(r[k]), 1e-12) for k in c)
            worst_cand = max(worst_cand, rel)
            n_cand_ok += rel <= 1e-6
            n_exact += all(row[k] == r[k] for k in c)
    rec = {"size": size, "pair": f"{a}->{b}", "n_2d": n, "registered_bit_identical_2d": n_exact, "registered_worst_rel": worst_reg,
           "candidate_within_1e-6": n_cand_ok, "candidate_worst_rel": worst_cand, "registered_s": round(t_reg, 1), "candidate_s": round(t_cand, 1)}
    print(json.dumps(rec, indent=1))
    for tag in (a, b):
        shutil.rmtree(T.tag_dir(tag), ignore_errors=True)
    write(f"parity3b_{size}_{a}_{b}", rec)


if __name__ == "__main__":
    {"svdbench": svdbench, "dlprobe": dlprobe, "batchprobe": batchprobe, "parity3b": parity3b}[sys.argv[1]]()

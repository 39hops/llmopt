"""Independent verifier for K2-HORIZON-TRANSPORT-0 (one size per run,
SIZE env). Re-reads census.jsonl and gate_rows.jsonl and recomputes every
pair aggregate (total relative delta, 36-layer profile, 10-bin
normalized-depth profile, centroid, class shares and class order,
identical counts, lag-1 median / fraction negative / whole-model cosine
from the per-tensor rows, medians of the 2-D statistics), every gate
count and digest per (tag, label, seed, tier), re-scores every ninth
gate row with its own boxed sympy, checks the tag commits and shard
shas against the immutable manifest, the instrument / ladder / census
module sources against the receipt's start provenance, and refuses to
overwrite its receipt. Bars are adjudicated by the booking against the
prereg text, not here: this verifier certifies the numbers.

Usage:
    SIZE=3.7B .venv-k2/bin/python scratch/k2h_transportverify.py
"""
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE = os.environ["SIZE"]
OUT = os.path.join(ROOT, f"logs/k2h/transport_{SIZE}")
D = []
NBINS = 10


def chk(c, m):
    if not c:
        D.append(m)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _w(conn, ans, truth):
    try:
        import sympy as sp
        X = sp.Symbol("x")
        conn.send(bool(sp.simplify(sp.sympify(ans, locals={"x": X}) - sp.sympify(truth, locals={"x": X})) == 0))
    except Exception:
        conn.send(False)


def boxed(ans, truth):
    if not ans:
        return False
    ctx = mp.get_context("fork")
    a, b = ctx.Pipe(False)
    p = ctx.Process(target=_w, args=(b, ans, truth))
    p.start()
    b.close()
    p.join(10)
    if p.is_alive():
        p.kill()
        p.join()
        return False
    return a.recv() if a.poll() else False


def main():
    chk(not os.path.exists(os.path.join(OUT, "verify_receipt.json")), "REFUSE OVERWRITE")
    if D:
        raise SystemExit(D[-1])
    rec = json.load(open(os.path.join(OUT, "receipt.json")))
    chk(rec["smoke"] is False and rec["prereg"] == "K2-HORIZON-TRANSPORT-0" and rec["size"] == SIZE, "identity")
    for f in ("scratch/k2h_transport.py", "scratch/k2h_gateladder.py", "scratch/k2h_stagecensus.py", "docs/preregs/k2h-transport-0.manifest.json"):
        chk(sha(os.path.join(ROOT, f)) == rec["start"]["file_sha256"][f], f"pin {f}")
    chk(rec["start"]["start_commit"] == rec["completion_commit"], "start v completion commit")
    man = json.load(open(os.path.join(ROOT, "docs/preregs/k2h-transport-0.manifest.json")))["sizes"][SIZE]
    for tag, v in rec["tags"].items():
        chk(v["commit"] == man["selected"][tag]["commit"], f"commit {tag}")
        chk(all(man["selected"][tag]["shards"][s]["sha256"] == h for s, h in v["shard_sha256"].items()) and len(v["shard_sha256"]) == man["selected"][tag]["n_shards"], f"shards {tag}")
    chk(rec["tokenizer"]["tokenizer_sha256"] == man["selected"]["pretrain_final"]["small_sha256"]["tokenizer.json"], "tokenizer pin")
    if D:
        raise SystemExit(D[-1])
    rows = [json.loads(l) for l in open(os.path.join(OUT, "census.jsonl"))]
    chk(all(r["smoke"] is False for r in rows), "census smoke flags")
    by = defaultdict(list)
    for r in rows:
        by[r["pair"]].append(r)
    L = 1 + max(r["layer"] for r in rows if r["layer"] is not None)
    chk(L == 36, f"L {L}")
    n_tensors = None
    for pair, rs in by.items():
        got = rec["pairs"][pair]
        n_tensors = n_tensors or len(rs)
        chk(len(rs) == n_tensors == got["n_tensors"], f"{pair} n_tensors {len(rs)}")
        tot = sum(r["fro_d"] ** 2 for r in rs)
        tot_a = sum(r["fro_a"] ** 2 for r in rs)
        chk(abs((tot / tot_a) ** 0.5 - got["total_rel_d"]) < 1e-9, f"{pair} total_rel_d")
        lay = [0.0] * L
        cls = defaultdict(float)
        for r in rs:
            if r["layer"] is not None:
                lay[r["layer"]] += r["fro_d"] ** 2
            cls[r["cls"]] += r["fro_d"] ** 2
        inl = sum(lay)
        prof = [v / inl for v in lay] if inl > 0 else [0.0] * L
        chk(max(abs(p - q) for p, q in zip(prof, got["layer_profile"])) < 1e-9, f"{pair} profile")
        bins = [0.0] * NBINS
        for l, p in enumerate(prof):
            bins[min(NBINS - 1, (NBINS * l) // L)] += p
        chk(max(abs(p - q) for p, q in zip(bins, got["depth_bins"])) < 1e-9, f"{pair} bins")
        if inl > 0:
            chk(abs(sum(p * l / (L - 1) for l, p in enumerate(prof)) - got["depth_centroid"]) < 1e-9, f"{pair} centroid")
        for k, v in cls.items():
            chk(abs(v / tot - got["share_by_class"][k]) < 1e-9, f"{pair} class {k}")
        chk(got["class_order"] == sorted(cls, key=lambda k: -cls[k]), f"{pair} class order")
        chk(got["n_identical"] == sum(r["identical"] for r in rs), f"{pair} identical")
        cp = [r["cos_prev_delta"] for r in rs if r.get("cos_prev_delta") is not None]
        if cp:
            chk(abs(float(np.median(cp)) - got["lag1_median_cos"]) < 1e-9 and abs(float(np.mean([c < 0 for c in cp])) - got["lag1_frac_negative"]) < 1e-9, f"{pair} lag1")
            num = sum(r["fro_d"] * r["fro_prev"] * r["cos_prev_delta"] for r in rs if r.get("cos_prev_delta") is not None)
            den = (sum(r["fro_d"] ** 2 for r in rs if "fro_prev" in r) ** 0.5) * (sum(r["fro_prev"] ** 2 for r in rs if "fro_prev" in r) ** 0.5)
            chk(abs(num / den - got["lag1_whole_model_cos"]) < 1e-6, f"{pair} whole-model lag1")
        else:
            chk(got["lag1_median_cos"] is None, f"{pair} lag1 None")
        for k, v in got["medians_2d"].items():
            vals = [r[k] for r in rs if k in r and r[k] is not None]
            chk(vals and abs(float(np.median(vals)) - v) < 1e-9, f"{pair} median {k}")
        chk(got["structural"] == any("effrank_d" in r for r in rs) or got["total_rel_d"] == 0, f"{pair} structural flag")
    # gates
    g = [json.loads(l) for l in open(os.path.join(OUT, "gate_rows.jsonl"))]
    for tag, gv in rec["gate"].items():
        for sd, tiers in gv["seeds"].items():
            for tier, got in tiers.items():
                rs = sorted((r for r in g if r["tag"] == tag and r["seed"] == int(sd) and r["tier"] == int(tier)), key=lambda r: r["item"])
                chk(len(rs) == got["n"] == 40, f"{tag} s{sd} T{tier} rows {len(rs)}")
                chk(sum(r["correct"] for r in rs) == got["correct"], f"{tag} s{sd} T{tier} count")
                fam = defaultdict(int)
                for r in rs:
                    fam[r["family"]] += int(r["correct"])
                chk(all(got["by_family"].get(k, 0) == v for k, v in fam.items()), f"{tag} s{sd} T{tier} families")
                chk(hashlib.sha256(json.dumps([r["gen_ids"] for r in rs]).encode()).hexdigest() == got["gen_sha256"], f"{tag} s{sd} T{tier} digest")
    for tag, gv in rec["gate_control"].items():
        for sd, tiers in gv["seeds"].items():
            for tier, got in tiers.items():
                rs = sorted((r for r in g if r["tag"] == tag + "|common" and r["seed"] == int(sd) and r["tier"] == int(tier)), key=lambda r: r["item"])
                chk(len(rs) == got["n"] == 40 and sum(r["correct"] for r in rs) == got["correct"], f"{tag} control s{sd} T{tier}")
                chk(hashlib.sha256(json.dumps([r["gen_ids"] for r in rs]).encode()).hexdigest() == got["gen_sha256"], f"{tag} control s{sd} T{tier} digest")
    sub = g[::9]
    mism = sum(boxed(r["answer"], r["truth"]) != r["correct"] for r in sub)
    chk(mism == 0, f"rescore mismatches {mism}/{len(sub)}")
    out = {"prereg": "K2-HORIZON-TRANSPORT-0", "size": SIZE, "verdict": "VERIFIED" if not D else "DISCREPANCIES", "discrepancies": D[:40], "n_discrepancies": len(D),
           "n_census_rows": len(rows), "n_gate_rows": len(g), "rescored_subset": len(sub), "receipt_sha256": sha(os.path.join(OUT, "receipt.json")),
           "census_sha256": sha(os.path.join(OUT, "census.jsonl")), "gate_rows_sha256": sha(os.path.join(OUT, "gate_rows.jsonl")),
           "instrument_run_commit": rec["start"]["start_commit"],
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "status_porcelain": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout, "verifier_sha256": sha(__file__)}
    json.dump(out, open(os.path.join(OUT, "verify_receipt.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:1200])


if __name__ == "__main__":
    main()

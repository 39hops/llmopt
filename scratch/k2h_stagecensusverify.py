"""Independent verifier for K2-HORIZON-STAGE-DELTA-CENSUS-0. Shares no
aggregation code with scratch/k2h_stagecensus.py: re-reads census.jsonl
and gate_rows.jsonl, recomputes every per-pair aggregate (total relative
delta, layer profile, depth centroid, class shares, identical counts),
every profile Spearman, every gate count and generation digest, the
NO-OP precondition and the three bars; re-scores a fixed subset of gate
answers with its own fork-boxed sympy check; pins the instrument source
against its start provenance and the ancestry pin; refuses to overwrite
its receipt.

Usage:
    .venv-k2/bin/python scratch/k2h_stagecensusverify.py
"""
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs/k2h/stagecensus")
ANC = os.path.join(ROOT, "docs/preregs/k2h-stagecensus-0.ancestry.json")
D = []


def chk(c, m):
    if not c:
        D.append(m)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rank(x):
    x = np.asarray(x, float)
    r = np.empty(len(x))
    s = sorted(range(len(x)), key=lambda i: x[i])
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[s[j + 1]] == x[s[i]]:
            j += 1
        for t in range(i, j + 1):
            r[s[t]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spear(a, b):
    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


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
    chk(not os.path.exists(os.path.join(OUT, "verify_receipt.json")), "REFUSE OVERWRITE verify_receipt.json")
    if D:
        raise SystemExit(D[-1])
    rec = json.load(open(os.path.join(OUT, "receipt.json")))
    chk(rec.get("smoke") is False and rec["prereg"] == "K2-HORIZON-STAGE-DELTA-CENSUS-0", "receipt identity")
    src = sha(os.path.join(ROOT, "scratch/k2h_stagecensus.py"))
    chk(src == rec["start"]["file_sha256"]["scratch/k2h_stagecensus.py"], "instrument source sha v receipt")
    chk(sha(ANC) == rec["start"]["file_sha256"]["docs/preregs/k2h-stagecensus-0.ancestry.json"], "ancestry pin sha")
    chk(rec["start"]["start_commit"] == rec["completion_commit"], "start v completion commit")
    anc = json.load(open(ANC))
    chk(rec["ladder"] == anc["ladder"], "ladder v ancestry")
    for t, c in anc["tags"].items():
        chk(rec["tags"][t]["commit"].startswith(c), f"tag commit {t}")
    if D:
        raise SystemExit(D[-1])
    rows = [json.loads(l) for l in open(os.path.join(OUT, "census.jsonl"))]
    chk(all(r["smoke"] is False for r in rows), "census smoke flag")
    by_pair = defaultdict(list)
    for r in rows:
        by_pair[r["pair"]].append(r)
    L = 1 + max(r["layer"] for r in rows if r["layer"] is not None)
    chk(L == 28, f"n_layers {L}")
    for pair, rs in by_pair.items():
        got = rec["pair_stats"][pair]
        chk(len(rs) == 255 and got["n_tensors"] == 255, f"{pair} tensor count {len(rs)}")
        tot = sum(r["fro_d"] ** 2 for r in rs)
        tot_a = sum(r["fro_a"] ** 2 for r in rs)
        lay = [0.0] * L
        cls = defaultdict(float)
        for r in rs:
            if r["layer"] is not None:
                lay[r["layer"]] += r["fro_d"] ** 2
            cls[r["cls"]] += r["fro_d"] ** 2
        chk(abs((tot / tot_a) ** 0.5 - got["total_rel_d"]) < 1e-9, f"{pair} total_rel_d")
        inl = sum(lay)
        if inl > 0:
            prof = [v / inl for v in lay]
            chk(max(abs(p - q) for p, q in zip(prof, got["layer_profile"])) < 1e-9, f"{pair} profile")
            chk(abs(sum(p * l / (L - 1) for l, p in enumerate(prof)) - got["depth_centroid"]) < 1e-9, f"{pair} centroid")
        else:
            chk(got["depth_centroid"] is None, f"{pair} centroid None")
        for k, v in cls.items():
            chk(abs(v / tot - got["share_by_class"][k]) < 1e-9 if tot > 0 else got["share_by_class"][k] == 0.0, f"{pair} class {k}")
        chk(got["n_identical"] == sum(r["identical"] for r in rs), f"{pair} identical")
        for r in rs:
            if "rel_d" in r and r["rel_d"] is not None:
                chk(abs(r["rel_d"] - r["fro_d"] / r["fro_a"]) < 1e-12, f"{pair} {r['tensor']} rel_d")
            if r["identical"]:
                chk(r["fro_d"] == 0.0, f"{pair} {r['tensor']} identical but fro_d>0")
    prof = rec["profiles"]
    for k, v in rec["profile_spearman"].items():
        x, y = k.split("|")
        s = spear(prof[x], prof[y])
        chk((s is None and v is None) or (s is not None and v is not None and abs(s - v) < 1e-9), f"spearman {k}")
    for k in ("PRETRAIN-INTERNAL", "MID_1-INTERNAL", "MID_2-INTERNAL", "DISTILL", "MERGE"):
        chk(k in prof, f"profile {k} present")
    # gates
    g = [json.loads(l) for l in open(os.path.join(OUT, "gate_rows.jsonl"))]
    chk(all(r["smoke"] is False for r in g), "gate smoke flag")
    main_rows = [r for r in g if r["rider"] is None]
    for tag in rec["ladder"]:
        for s in rec["gate_seeds"]:
            rs = sorted((r for r in main_rows if r["tag"] == tag and r["seed"] == s), key=lambda r: r["item"])
            got = rec["gate"][tag][str(s)]
            chk(len(rs) == got["n"] == rec["n_items"] and [r["item"] for r in rs] == list(range(len(rs))), f"{tag} s{s} rows")
            chk(sum(r["correct"] for r in rs) == got["correct"], f"{tag} s{s} correct count")
            cells = defaultdict(int)
            for r in rs:
                cells[f"{r['family']}{r['tier']}"] += int(r["correct"])
            chk(dict(cells) == {k: v for k, v in got["by_cell"].items()} or all(got["by_cell"].get(k, 0) == v for k, v in cells.items()), f"{tag} s{s} cells")
            chk(hashlib.sha256(json.dumps([r["gen_ids"] for r in rs]).encode()).hexdigest() == got["gen_sha256"], f"{tag} s{s} gen sha")
    # re-score a fixed subset: every 7th row of the main gate rows
    mism = 0
    sub = main_rows[::7]
    for r in sub:
        if boxed(r["answer"], r["truth"]) != r["correct"]:
            mism += 1
    chk(mism == 0, f"rescore mismatches {mism}/{len(sub)}")
    # precondition + bars
    noop = rec["pair_stats"]["rl-mopd_final->main"]
    ti = noop["n_identical"] == noop["n_tensors"] == 255
    gi = all(rec["gate"]["rl-mopd_final"][str(s)]["gen_sha256"] == rec["gate"]["main"][str(s)]["gen_sha256"] for s in rec["gate_seeds"])
    chk(rec["precondition_noop"] == {"tensors_identical": ti, "gate_ids_identical": gi, "fires": ti and gi}, "precondition")
    chk({v["device"] for v in rec["tags"].values()} == {"mps"}, "one device (mps)")
    chk(all(k in rec["versions"] for k in ("python", "torch", "transformers", "sympy", "numpy")), "versions recorded")
    if not rec["precondition_noop"]["fires"]:
        chk(rec["bars"] == {"status": "PRECONDITION-FAILED"}, "bars withheld on failed precondition")
        finish(rec, rows, g, [], src)
        return
    G = lambda t, s: rec["gate"][t][str(s)]["correct"]
    deltas = [G("rl-mopd_final", s) - G("mid_2_final", s) for s in rec["gate_seeds"]]
    b = rec["bars"]
    chk(b["a_post_training_gate_move"]["deltas"] == deltas and b["a_post_training_gate_move"]["fires"] == (all(d > 7 for d in deltas) or all(d < -7 for d in deltas)), "bar a")
    st = [v for k, v in rec["profile_spearman"].items() if all(x in ("PRETRAIN-INTERNAL", "MID_1-INTERNAL", "MID_2-INTERNAL") for x in k.split("|"))]
    chk(len(st) == 3 and b["b_stage_profiles_distinct"]["fires"] == all(v is not None and v <= 0.5 for v in st), "bar b")
    c = rec["pair_stats"]["rl_rl-merged->rl-mopd_final"]["depth_centroid"]
    chk(b["c_distill_output_proximal"]["fires"] == ((c or 0) > 0.5), "bar c")
    sp_ = rec["profile_spearman"]
    ri = (sp_["PRETRAIN-INTERNAL|MID_1-INTERNAL"] is not None and sp_["PRETRAIN-INTERNAL|MID_1-INTERNAL"] > 0.5
          and sp_["PRETRAIN-INTERNAL|MID_2-INTERNAL"] is not None and sp_["PRETRAIN-INTERNAL|MID_2-INTERNAL"] > 0.5)
    chk(rec["refuted_if"]["context_extension_distinct_refuted"] == ri, "refuted-if (b)")
    finish(rec, rows, g, sub, src)


def finish(rec, rows, g, sub, src):
    out = {"prereg": "K2-HORIZON-STAGE-DELTA-CENSUS-0", "verdict": "VERIFIED" if not D else "DISCREPANCIES",
           "discrepancies": D[:40], "n_discrepancies": len(D), "n_census_rows": len(rows), "n_gate_rows": len(g),
           "rescored_subset": len(sub), "instrument_source_sha256": src, "receipt_sha256": sha(os.path.join(OUT, "receipt.json")),
           "census_sha256": sha(os.path.join(OUT, "census.jsonl")), "gate_rows_sha256": sha(os.path.join(OUT, "gate_rows.jsonl")),
           "instrument_run_commit": rec["start"]["start_commit"],
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "status_porcelain": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout,
           "verifier_sha256": sha(__file__)}
    json.dump(out, open(os.path.join(OUT, "verify_receipt.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:1500])


if __name__ == "__main__":
    main()

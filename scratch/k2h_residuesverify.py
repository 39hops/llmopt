"""Independent verifier for K2-HORIZON-RESIDUES-0. Re-reads
factorial_rows.jsonl and autocorr_rows.jsonl and recomputes every gate
count and digest, every main effect and interaction, the reproduction
gate against the locked L66337 receipt, every per-delta lag aggregate
(median cosine, fraction negative) from the per-tensor rows, the stage
aggregates and the three B bars; pins the instrument and the imported
frozen census module against the receipt's start provenance; refuses to
overwrite its receipt. Coarse-spacing cosines and whole-model cosines
are weight-level primitives with no row stream and are checked for
presence and sign-consistency only.

Usage:
    .venv-k2/bin/python scratch/k2h_residuesverify.py
"""
import hashlib
import json
import os
import subprocess
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "logs/k2h/residues")
D = []


def chk(c, m):
    if not c:
        D.append(m)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    chk(not os.path.exists(os.path.join(OUT, "verify_receipt.json")), "REFUSE OVERWRITE")
    if D:
        raise SystemExit(D[-1])
    rec = json.load(open(os.path.join(OUT, "receipt.json")))
    chk(rec["smoke"] is False and rec["prereg"] == "K2-HORIZON-RESIDUES-0", "identity")
    for f in ("scratch/k2h_residues.py", "scratch/k2h_stagecensus.py", "docs/preregs/k2h-stagecensus-0.ancestry.json", "logs/k2h/stagecensus/receipt.json"):
        chk(sha(os.path.join(ROOT, f)) == rec["start"]["file_sha256"][f], f"pin {f}")
    chk(rec["start"]["start_commit"] == rec["completion_commit"], "start v completion commit")
    lock = json.load(open(os.path.join(ROOT, "docs/receipts.lock.json")))["receipts"]
    chk(lock["logs/k2h/stagecensus/receipt.json"]["sha256"] == rec["booked_receipt_sha256"] == sha(os.path.join(ROOT, "logs/k2h/stagecensus/receipt.json")), "booked receipt v lock")
    booked = json.load(open(os.path.join(ROOT, "logs/k2h/stagecensus/receipt.json")))
    if D:
        raise SystemExit(D[-1])
    # A
    rows = [json.loads(l) for l in open(os.path.join(OUT, "factorial_rows.jsonl"))]
    chk(all(r["smoke"] is False for r in rows), "A smoke flags")
    seeds = rec["seeds"]
    C = {}
    for cell, cv in rec["A"]["cells"].items():
        tag, mode = cell.split("|")
        for s in seeds:
            rs = sorted((r for r in rows if r["cell"] == cell and r["seed"] == s), key=lambda r: r["item"])
            got = cv["seeds"][str(s)]
            chk(len(rs) == got["n"] == rec["n_items"] and [r["item"] for r in rs] == list(range(len(rs))), f"{cell} s{s} rows")
            chk(sum(r["correct"] for r in rs) == got["correct"], f"{cell} s{s} count")
            chk(hashlib.sha256(json.dumps([r["gen_ids"] for r in rs]).encode()).hexdigest() == got["gen_sha256"], f"{cell} s{s} digest")
            chk(all(r["rope"] == mode and r["tag"] == tag for r in rs), f"{cell} s{s} labels")
            C[(tag, mode, s)] = got["correct"]
    chk(rec["A"]["cells"]["mid_2_final|none"]["rope_parameters"]["rope_type"] == "default" and rec["A"]["cells"]["rl_rl-merged|yarn"]["rope_parameters"]["rope_type"] == "yarn", "rope modes")
    for s in seeds:
        m2n, m2y, mgn, mgy = C[("mid_2_final", "none", s)], C[("mid_2_final", "yarn", s)], C[("rl_rl-merged", "none", s)], C[("rl_rl-merged", "yarn", s)]
        e = rec["A"]["effects"][str(s)]
        chk(e["weight_main"] == ((mgn - m2n) + (mgy - m2y)) / 2 and e["rope_main"] == ((m2y - m2n) + (mgy - mgn)) / 2 and e["interaction"] == mgy - mgn - m2y + m2n, f"effects s{s}")
    repro = all(rec["A"]["cells"][f"{t}|{m}"]["seeds"][str(s)]["gen_sha256"] == booked["gate"][t][str(s)]["gen_sha256"] for t, m in (("mid_2_final", "none"), ("rl_rl-merged", "yarn")) for s in seeds)
    chk(rec["A"]["reproduction"]["fires"] == repro, "reproduction gate")
    E = rec["A"]["effects"]
    if repro:
        b = rec["A"]["bars"]
        chk(b["A1_weight_main_positive"]["fires"] == all(E[str(s)]["weight_main"] > 0 for s in seeds), "A1")
        chk(b["A2_rope_main_positive"]["fires"] == all(E[str(s)]["rope_main"] > 0 for s in seeds), "A2")
        chk(b["A3_additive"]["fires"] == all(abs(E[str(s)]["interaction"]) <= 3 for s in seeds), "A3")
    else:
        chk(rec["A"]["bars"] == {"status": "REPRODUCTION-FAILED"}, "A bars withheld")
    # B
    ar = [json.loads(l) for l in open(os.path.join(OUT, "autocorr_rows.jsonl"))]
    chk(all(r["smoke"] is False for r in ar), "B smoke flags")
    by = defaultdict(list)
    for r in ar:
        by[(r["pair"], r["lag"])].append(r)
    chain = rec["B"]["chain"]
    chk([e["pair"] for e in rec["B"]["deltas"]] == [f"{a}->{b}" for a, b in zip(chain, chain[1:])], "delta order v chain")
    for i, e in enumerate(rec["B"]["deltas"]):
        for lag, v in e["lags"].items():
            rs = by[(e["pair"], int(lag))]
            chk(len(rs) == 255 == v["n"], f"{e['pair']} lag{lag} rows {len(rs)}")
            cs = [r["cos"] for r in rs if r["cos"] is not None]
            chk(abs(float(np.median(cs)) - v["median_cos"]) < 1e-9 and abs(float(np.mean([c < 0 for c in cs])) - v["frac_negative"]) < 1e-9, f"{e['pair']} lag{lag} agg")
            chk(rec["B"]["deltas"][i - int(lag)]["pair"] == v["prev_pair"], f"{e['pair']} lag{lag} prev")
            chk(abs(sum(r["fro_d"] ** 2 for r in rs) ** 0.5 - e["fro_total"]) < 1e-6 * max(1.0, e["fro_total"]), f"{e['pair']} fro_total")
        exp_cls = "WITHIN" if e["pair"].split("->")[0].split("_")[0:2] == e["pair"].split("->")[1].split("_")[0:2] else "BOUNDARY"
        chk(e["class"] == exp_cls, f"{e['pair']} class")
    Dl = rec["B"]["deltas"]
    agg = {}
    for st in ("pretrain", "mid_1", "mid_2"):
        w1 = [e["lags"]["1"]["median_cos"] for i, e in enumerate(Dl) if i > 0 and e["stage"] == st and e["class"] == "WITHIN" and "1" in e["lags"] and Dl[i - 1]["class"] == "WITHIN" and Dl[i - 1]["stage"] == st]
        agg[st] = w1
        got = rec["B"]["aggregates"][st]
        chk(got["within_lag1_medians"] == w1 and got["n_pairs"] == len(w1) and abs(got["frac_pairs_negative"] - float(np.mean([x < 0 for x in w1]))) < 1e-9, f"agg {st}")
    chk({s: len(v) for s, v in agg.items()} == {"pretrain": 9, "mid_1": 4, "mid_2": 8}, f"pair counts {[len(v) for v in agg.values()]}")
    bb = rec["B"]["bars"]
    chk(bb["B1_repeatable_anti_alignment"]["fires"] == all(float(np.mean([x < 0 for x in agg[s]])) >= 0.8 for s in agg), "B1")
    chk(bb["B2_spacing_sign_agrees"]["fires"] == all(np.sign(rec["B"]["coarse"][s]["median_cos"]) == np.sign(np.median(agg[s])) for s in agg), "B2")
    bnd = rec["B"]["boundary"]
    chk(set(bnd) == {"mid_1", "mid_2"}, "boundary entries")
    chk(bb["B3_boundary_specific"]["fires"] == (len(bnd) == 2 and all(abs(bnd[s]["cos_first_within_v_boundary"]) > float(np.median([abs(x) for x in agg[s]])) for s in bnd)), "B3")
    for s in ("mid_1", "mid_2"):
        i = [k for k, e in enumerate(Dl) if e["pair"] == bnd[s]["pair"]][0]
        chk(Dl[i - 1]["class"] == "BOUNDARY" and Dl[i]["lags"]["1"]["median_cos"] == bnd[s]["cos_first_within_v_boundary"], f"boundary {s}")
    out = {"prereg": "K2-HORIZON-RESIDUES-0", "verdict": "VERIFIED" if not D else "DISCREPANCIES", "discrepancies": D[:40], "n_discrepancies": len(D),
           "n_factorial_rows": len(rows), "n_autocorr_rows": len(ar), "receipt_sha256": sha(os.path.join(OUT, "receipt.json")),
           "factorial_rows_sha256": sha(os.path.join(OUT, "factorial_rows.jsonl")), "autocorr_rows_sha256": sha(os.path.join(OUT, "autocorr_rows.jsonl")),
           "instrument_run_commit": rec["start"]["start_commit"],
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "status_porcelain": subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout, "verifier_sha256": sha(__file__)}
    json.dump(out, open(os.path.join(OUT, "verify_receipt.json"), "w"), indent=1)
    print(json.dumps(out, indent=1)[:1200])


if __name__ == "__main__":
    main()

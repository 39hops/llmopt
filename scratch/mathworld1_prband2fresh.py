"""MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-FRESH-SEED-0
Stage-A validation scorer (adopt-not-fork of the atlas scorer
scratch/mathworld1_prband2atlasscore.py, RESULTS L65695).

Prompts come ONLY from the tracked 192-row artifact
logs/mathworld1/prband2fresh/validation_prompts.jsonl (views RAW
and R488; no renderer, no atlas lookup, renders.jsonl never
opened). Checkpoints: the eight fresh terminal checkpoints listed
in the freeze receipt (all must exist and match their shas before
any fresh logit) plus the four old discovery checkpoints for the
SCORER-REPLAY gate (exact parsed-float equality of every FULL RAW
row against the tracked raw_scores.jsonl; failure stops the run
before any fresh logit). Scoring law unchanged: masked_token_lps,
8 action symbols + EOS, T = 9, TOTAL SUM, strict top-1, exact tie
= loss, semantic order i_sum / A0 / B0 / I0-t5. Matched pairs by
EXPLICIT pair_id join (48 pair_ids, one SIN_LOW + one COS_LOW),
never row adjacency; the old-checkpoint replay also asserts that
the pair_id-join B equals the adjacency B on this population.
MASK0 (mask 0) scored under RAW and R488 for every fresh
checkpoint as the render-invariance sanity. Full 9-token LP
vectors persisted for every row.

Usage:
    PRBAND2FR_SMOKE=1 .venv/bin/python scratch/mathworld1_prband2fresh.py
    .venv/bin/python scratch/mathworld1_prband2fresh.py
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_prband2score import (A0, B0, CKPTS,  # noqa: E402
                                             CODE_BASE, NAMES, NOISE,
                                             PRIMARY, SEM, TOK, VOCAB,
                                             ctup, fsha, top1_of)
from scratch.mathworld1_respath import masked_token_lps  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import CTX  # noqa: E402
from scratch.mathworld1_svpcode import factor_decode  # noqa: E402
from scratch.mathworld1_svpforder import (PERM, pf_decode,  # noqa: E402
                                          pf_encode)

SMOKE = os.environ.get("PRBAND2FR_SMOKE") == "1"
PREREG = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-FRESH-SEED-0"
PREREG_COMMIT = "632e57dd5593cc33c3f5588066c8e6176b7b14dd"
OUTDIR = Path("logs/mathworld1/prband2fresh_score_smoke" if SMOKE
              else "logs/mathworld1/prband2fresh_score")
SMOKE_RECEIPT = Path("logs/mathworld1/prband2fresh_score_smoke/prband2fresh_receipt.json")
ARTIFACT = "logs/mathworld1/prband2fresh/validation_prompts.jsonl"
FREEZE = "logs/mathworld1/prband2fresh_train/fresh_checkpoint_freeze.json"
OLD_RAW = "logs/mathworld1/prband2score/raw_scores.jsonl"
PINS = {
    ARTIFACT: "ce06c5bc4f687d8fada5797cab9e7fa052da8cab4accf730816f7be4d4313386",
    "logs/mathworld1/prband2fresh/validation_prompts_receipt.json":
        "33dea1bed26b362ece66ad2733aab1193ecaba4f2b5073fbf29cd76d5f4f64f3",
    PRIMARY: "209391ef3b2e5c87308571d6ef309bb5724a214160caa1b7857f4a31f9112c34",
    OLD_RAW: "68d014ac2a0bf5b085941daeeae7def5a30506c7bb45c91630a96113b01cb31e",
}
VIEWS = {"RAW": (12, "8f1479a9402429ac18d1d1e55f803d02bd47313f8c69dfbfae6abd0a4f5f26f2",
                 "c4edb35529ea6075b904ef84981b10f480e4be06ccd4e6bbac3111525fbad272"),
         "R488": (488, "91e4098b48717d6b611445824a6c987594c38df1b9c5f40e815ff95b36a5f9d9",
                  "0d0c796f7f4185ba86cf1b528d270ae2965cb4dd98a65a0f488ae463b3d67b72")}
R488_ROLES = ["K", "HI_D", "LO_D", "LO_L", "HI_L", "W"]
FRESH_SEEDS = ["21001", "22001", "23001", "24001"]
EPS_SCORE, EPS_D = NOISE, 2e-05
TORCH_PIN = "2.12.1"


def sgn(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def load_model(path, sha_exp, dev):
    gate(fsha(path) == sha_exp, f"CKPT SHA {path}")
    m = build_model(VOCAB, ctx=4096)
    m.load_state_dict(torch.load(path, weights_only=True))
    gate(sum(q.numel() for q in m.parameters()) == 19142016, "PARAMS")
    m.eval()
    m = m.to(dev)
    gate(len(m.blocks) == 8, "8 BLOCKS")
    return m


def metrics(sc, states, pairs):
    """sc: {state_i: {sem: sum}}. Explicit pair_id join."""
    corr, ties, tops, mm = {}, 0, {}, {}
    for st in states:
        w, tie, _mg = top1_of(sc[st["i"]])
        corr[st["i"]] = (w == st["gold"])
        ties += tie
        tops[st["i"]] = NAMES.get(w) if w else None
        mm[st["i"]] = st["g"] * (sc[st["i"]][A0] - sc[st["i"]][B0])
    B = sum(corr[p["SIN_LOW"]] and corr[p["COS_LOW"]] for p in pairs.values())
    return {"T": sum(corr.values()), "B": B,
            "A0_correct": sum(corr[st["i"]] for st in states if st["gold"] == A0),
            "B0_correct": sum(corr[st["i"]] for st in states if st["gold"] == B0),
            "ties": ties, "top_census": dict(Counter(tops.values())),
            "pair_top_switches": sum(tops[p["SIN_LOW"]] != tops[p["COS_LOW"]]
                                     for p in pairs.values()),
            "margin_census": {"gold_directed": sum(v >= EPS_D for v in mm.values()),
                              "opposite_gold": sum(v <= -EPS_D for v in mm.values()),
                              "subnoise": sum(abs(v) < EPS_D for v in mm.values())},
            "_corr": corr, "_tops": tops, "_mm": mm}


def main():
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    if not SMOKE:
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(sr.get("smoke") is True and sr.get("verdict") == "SMOKE OK", "SMOKE NOT GREEN")
        for pth, h in sr["start"]["file_sha256"].items():
            gate(fsha(pth) == h, f"SMOKE STALE {pth}")
    START = start_provenance(
        ["scratch/mathworld1_prband2fresh.py", "scratch/mathworld1_prband2atlasscore.py",
         "scratch/mathworld1_prband2score.py", "scratch/mathworld1_respath.py",
         "scratch/mathworld1_svpchal.py", "scratch/mathworld1_svpforder.py",
         "scratch/mathworld1_svpcode.py", "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_svpbirth.py", "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    gate(torch.__version__ == TORCH_PIN, f"TORCH {torch.__version__}")
    gate(not OUTDIR.exists(), f"REFUSING: {OUTDIR} exists")
    t0 = time.monotonic()
    P = [json.loads(l) for l in open(PRIMARY)]
    gate(len(P) == 96, "N=96")
    # explicit pair_id join
    pairs = defaultdict(dict)
    for i, p in enumerate(P):
        gate(p["theta"] not in pairs[p["pair_id"]], "DUP THETA IN PAIR")
        pairs[p["pair_id"]][p["theta"]] = i
    gate(len(pairs) == 48 and all(sorted(v) == ["COS_LOW", "SIN_LOW"] for v in pairs.values()),
         "48 PAIRS x (SIN_LOW, COS_LOW)")
    pairs = dict(pairs)
    # validation artifact
    art = defaultdict(dict)
    for l in open(ARTIFACT):
        r = json.loads(l)
        gate(r["view"] in VIEWS and r["atlas_index"] == VIEWS[r["view"]][0]
             and r["render_id"] == VIEWS[r["view"]][1], "ARTIFACT IDENTITY")
        if r["view"] == "R488":
            gate(r["roles"] == R488_ROLES, "R488 ROLES")
        k = (r["pair_id"], r["theta"])
        gate(k not in art[r["view"]], "DUP ARTIFACT ROW")
        gate(hashlib.sha256(r["cur"].encode()).hexdigest() == r["cur_sha"], "CUR SHA")
        art[r["view"]][k] = r["cur"]
    order = [(p["pair_id"], p["theta"]) for p in P]
    for vn, (_i, _r, msha) in VIEWS.items():
        gate(len(art[vn]) == 96 and set(art[vn]) == set(order), f"{vn} 96")
        gate(hashlib.sha256(json.dumps([art[vn][k] for k in order]).encode()).hexdigest()
             == msha, f"{vn} MATRIX SHA")
    for i, p in enumerate(P):
        gate(art["RAW"][(p["pair_id"], p["theta"])] == p["cur"], "RAW IS PRIMARY")
    # old RAW rows (FULL) for the replay gate
    old = {}
    for l in open(OLD_RAW):
        r = json.loads(l)
        if r["mask"] == 255:
            old[(r["seed"], r["representation"], r["state"], tuple(r["candidate"]))] = (r["lps"], r["sum"])
    gate(len(old) == 1536, "OLD FULL ROWS")
    # fresh checkpoints from the freeze receipt
    if SMOKE:
        P = P[:2]
        pairs = {pid: v for pid, v in pairs.items() if all(i < 2 for i in v.values())}
        fresh = []
    else:
        fz = json.loads(open(FREEZE).read())
        gate(fz["n_checkpoints"] == 8 and set(fz["seeds"]) == set(FRESH_SEEDS), "FREEZE 8")
        fresh = []
        for c in fz["checkpoints"]:
            gate(Path(c["path"]).exists() and fsha(c["path"]) == c["sha256"], f"FRESH {c['path']}")
            fresh.append((c["seed"], c["representation"], c["path"], c["sha256"]))
        gate(len(fresh) == 8 and len({f[3] for f in fresh}) == 8, "8 DISTINCT FRESH")
    states = []
    for i, p in enumerate(P):
        by = {ctup(c): c for c in p["candidates"]}
        conts = {"CANONICAL": [], "PARAM_FIRST": []}
        for s in SEM:
            cz = by[s]["factor_code"]
            gate(factor_decode(cz) == s, "C RT")
            pz = pf_encode(s)
            gate(pf_decode(pz) == s and pz == [cz[PERM[i2]] for i2 in range(8)], "PF RT / PERM")
            conts["CANONICAL"].append([CODE_BASE + x for x in cz] + [TOK.eos_id])
            conts["PARAM_FIRST"].append([CODE_BASE + x for x in pz] + [TOK.eos_id])
        states.append({"i": i, "pair_id": p["pair_id"], "theta": p["theta"],
                       "gold": tuple(p["gold_tuple"]),
                       "g": 1 if p["theta"] == "SIN_LOW" else -1, "conts": conts,
                       "cur": {vn: art[vn][(p["pair_id"], p["theta"])] for vn in VIEWS}})
    n = len(states)
    OUTDIR.mkdir(parents=True)
    out = open(OUTDIR / "scores.jsonl", "w")
    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    receipt = {"smoke": SMOKE, "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
               "n_states": n, "pins": {p: fsha(p) for p in PINS},
               "views": {vn: {"atlas_index": v[0], "render_id": v[1], "matrix_sha": v[2]}
                         for vn, v in VIEWS.items()},
               "semantic_order": [list(s) for s in SEM],
               "eps": {"EPS_SCORE": EPS_SCORE, "EPS_D": EPS_D},
               "semantic_beyond_all_surface_identifiable": False,
               "device": "mps", "torch": torch.__version__, "pair_join": "explicit pair_id",
               "replay": {}, "fresh": {}}

    def score(m, seed, rep, sha, cohort, vn, mask, arm):
        sc = {}
        for st in states:
            lps = masked_token_lps(m, dev, st["cur"][vn], st["conts"][rep], mask)
            sc[st["i"]] = {}
            for s, lp, cont in zip(SEM, lps, st["conts"][rep]):
                gate(len(lp) == 9, "T!=9")
                tot = float(sum(lp))
                sc[st["i"]][s] = (lp, tot)
                out.write(json.dumps({
                    "cohort": cohort, "seed": seed, "representation": rep,
                    "ckpt_sha": sha, "arm": arm, "mask": mask, "view": vn,
                    "state": st["i"], "pair_id": st["pair_id"], "theta": st["theta"],
                    "cur_sha": hashlib.sha256(st["cur"][vn].encode()).hexdigest(),
                    "candidate": list(s), "name": NAMES[s], "continuation": cont,
                    "lps": lp, "sum": tot, "gold": list(st["gold"])}) + "\n")
        out.flush()
        return sc

    def finish(verdict):
        out.close()
        receipt["verdict"] = verdict
        receipt["wall_s"] = round(time.monotonic() - t0, 1)
        receipt["scores_sha256"] = fsha(str(OUTDIR / "scores.jsonl"))
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "prband2fresh_receipt.json").write_text(json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items() if k not in ("start", "pins")},
                         indent=1)[:5000], flush=True)

    # ---- SCORER-REPLAY gate on the old discovery checkpoints ----
    dtype = None
    for seed, rep, path, sha in (CKPTS[:1] if SMOKE else CKPTS):
        m = load_model(path, sha, dev)
        dtype = str(next(m.parameters()).dtype)
        sc = score(m, seed, rep, sha, "OLD", "RAW", 255, "FULL")
        exact = 0
        drift = 0.0
        for st in states:
            for s in SEM:
                lp, tot = sc[st["i"]][s]
                olp, osum = old[(seed, rep, st["i"], s)]
                exact += (olp == lp) and (osum == tot)
                drift = max(drift, abs(osum - tot), max(abs(a - b) for a, b in zip(olp, lp)))
        sums = {i: {s: v[1] for s, v in c.items()} for i, c in sc.items()}
        met = metrics(sums, states, pairs)
        adj = sum(met["_corr"][k] and met["_corr"][k + 1] for k in range(0, n, 2))
        receipt["replay"][f"{seed}|{rep}"] = {
            "rows": n * 4, "exact": exact, "max_abs_drift": drift, "pass": exact == n * 4,
            "T": met["T"], "B_pairid": met["B"], "B_adjacency": adj,
            "join_equals_adjacency": met["B"] == adj}
        print(f"[REPLAY {seed} {rep}] exact {exact}/{n * 4} drift {drift} "
              f"T {met['T']} B {met['B']} (adj {adj})", flush=True)
        del m
        torch.mps.empty_cache()
    receipt["dtype"] = dtype
    if not all(r["pass"] and r["join_equals_adjacency"] for r in receipt["replay"].values()):
        finish("SCORER-REPLAY FAILURE — NO FRESH CHECKPOINT SCORED")
        return
    if SMOKE:
        finish("SMOKE OK")
        return

    # ---- Stage A: eight fresh checkpoints x {RAW, R488} x {FULL, MASK0} ----
    for seed, rep, path, sha in fresh:
        ck = f"{seed}|{rep}"
        m = load_model(path, sha, dev)
        cell = {"path": path, "sha256": sha, "dtype": str(next(m.parameters()).dtype)}
        full, m0 = {}, {}
        for vn in VIEWS:
            sc = score(m, seed, rep, sha, "FRESH", vn, 255, "FULL")
            full[vn] = {i: {s: v[1] for s, v in c.items()} for i, c in sc.items()}
            sc0 = score(m, seed, rep, sha, "FRESH", vn, 0, "MASK0")
            m0[vn] = {i: {s: v[1] for s, v in c.items()} for i, c in sc0.items()}
        met = {vn: metrics(full[vn], states, pairs) for vn in VIEWS}
        r, x = met["RAW"], met["R488"]
        cell["FULL"] = {vn: {k: v for k, v in met[vn].items() if not k.startswith("_")}
                        for vn in VIEWS}
        cell["delta"] = {
            "Delta_T": x["T"] - r["T"], "Delta_B": x["B"] - r["B"],
            "top_changed": sum(r["_tops"][i] != x["_tops"][i] for i in range(n)),
            "gained": sum((not r["_corr"][i]) and x["_corr"][i] for i in range(n)),
            "lost": sum(r["_corr"][i] and (not x["_corr"][i]) for i in range(n)),
            "margin_direction_changes": sum(
                sgn(r["_mm"][i]) != sgn(x["_mm"][i]) and abs(r["_mm"][i]) >= EPS_D
                and abs(x["_mm"][i]) >= EPS_D for i in range(n))}
        # MASK0 sanity across the two renders
        spread = max(abs(m0["RAW"][i][s] - m0["R488"][i][s]) for i in range(n) for s in SEM)
        same_top = all(top1_of(m0["RAW"][i])[0] == top1_of(m0["R488"][i])[0] for i in range(n))
        switches = flips = 0
        for i in range(n):
            a, b = m0["RAW"][i], m0["R488"][i]
            if top1_of(a)[0] != top1_of(b)[0] and top1_of(a)[2] >= EPS_D and top1_of(b)[2] >= EPS_D:
                switches += 1
            da, db = a[A0] - a[B0], b[A0] - b[B0]
            if sgn(da) != sgn(db) and abs(da) >= EPS_D and abs(db) >= EPS_D:
                flips += 1
        cell["mask0"] = {"max_spread": spread, "same_top": same_top,
                         "robust_top_switches": switches, "robust_margin_flips": flips,
                         "pass": spread <= EPS_SCORE and same_top and switches == 0 and flips == 0}
        receipt["fresh"][ck] = cell
        print(f"[FRESH {ck}] RAW T {r['T']} B {r['B']} | R488 T {x['T']} B {x['B']} "
              f"| mask0 {cell['mask0']['pass']}", flush=True)
        del m
        torch.mps.empty_cache()
    finish("STAGE A SCORED")


if __name__ == "__main__":
    main()

"""MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-FRESH-SEED-PREVALENCE-0
Stage-B scorer: the complete 720-policy atlas on the EIGHT frozen
fresh checkpoints (adopt-not-fork of scratch/mathworld1_prband2atlasscore.py:
its atlas authority / anchor / subset-digest gates, constants and
MASK0 law are IMPORTED and reused; only the checkpoint source (the
Stage-A freeze receipt), the replay reference (the Stage-A stream's
FULL RAW rows, since the tracked raw_scores.jsonl has no fresh rows)
and the anchor expectations (Stage-A T/B of index 12 and 488) differ).
Checkpoint-major in freeze-receipt order, ascending atlas_index,
per-policy flush, chunk receipts with stream sha, partial chunks set
aside without authority, LP vectors on the frozen 16-policy subset only.

Usage:
    PRBAND2AF_SMOKE=1 .venv/bin/python scratch/mathworld1_prband2atlasfresh.py
    .venv/bin/python scratch/mathworld1_prband2atlasfresh.py
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
from scratch.mathworld1_prband2atlasscore import (ANCHORS, EPS_D,  # noqa: E402
                                                  EPS_SCORE, LP_SET,
                                                  MASK0_SET, PINS as ATLAS_PINS,
                                                  RAW_IDX, SUBSET13,
                                                  SUBSET13_SHA, TORCH_PIN,
                                                  load_atlas, sgn)
from scratch.mathworld1_prband2score import (A0, B0, CODE_BASE,  # noqa: E402
                                             NAMES, PRIMARY, SEM, TOK,
                                             VOCAB, ctup, fsha, top1_of)
from scratch.mathworld1_respath import masked_token_lps  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import factor_decode  # noqa: E402
from scratch.mathworld1_svpforder import (PERM, pf_decode,  # noqa: E402
                                          pf_encode)

SMOKE = os.environ.get("PRBAND2AF_SMOKE") == "1"
PREREG = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-FRESH-SEED-PREVALENCE-0"
OUTDIR = Path("logs/mathworld1/prband2atlasfresh_smoke" if SMOKE
              else "logs/mathworld1/prband2atlasfresh")
SMOKE_RECEIPT = Path("logs/mathworld1/prband2atlasfresh_smoke/prband2atlasfresh_receipt.json")
FREEZE = "logs/mathworld1/prband2fresh_train/fresh_checkpoint_freeze.json"
STAGEA = "logs/mathworld1/prband2fresh_score/scores.jsonl"
STAGEA_AGG = "logs/mathworld1/prband2fresh_score/aggregate.json"
PINS = {**ATLAS_PINS,
        FREEZE: "67003be7d421d4a7d470d8ee80cb254625d67a0c134e2b9514e1ef64e706b22c",
        STAGEA: "d9063cfbb3bcbb1b1a03d96eafc5edc5b998caf542fe7f86435cbb4b18983959",
        STAGEA_AGG: "9815b527db863bc453612cfa806946f35ace07c7de4faa82a362e1e32a99773a"}
R488 = 488
LP_SET_B = set(LP_SET) | {R488}   # additive: R488 rows carry LPs for row-level replay
SMOKE_POLICIES = [0, 12, 300, 480, 488, 268, 719]


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
        ["scratch/mathworld1_prband2atlasfresh.py",
         "scratch/mathworld1_prband2atlasscore.py", "scratch/mathworld1_prband2cf.py",
         "scratch/mathworld1_prband2score.py", "scratch/mathworld1_respath.py",
         "scratch/mathworld1_svpchal.py", "scratch/mathworld1_svpforder.py",
         "scratch/mathworld1_svpcode.py", "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_svpbirth.py", "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    gate(torch.__version__ == TORCH_PIN, f"TORCH {torch.__version__}")
    t0 = time.monotonic()
    P = [json.loads(l) for l in open(PRIMARY)]
    gate(len(P) == 96 and len({p["pair_key"] for p in P}) == 48, "N/keys")
    for k in range(0, 96, 2):
        gate(P[k]["pair_key"] == P[k + 1]["pair_key"]
             and P[k]["theta"] == "SIN_LOW" and P[k + 1]["theta"] == "COS_LOW",
             "pair adjacency")
    for p in P:
        gate(sorted(tuple(t) for t in p["cand_tuples"]) == sorted(SEM), "SEMANTIC SET")
        gate(tuple(p["gold_tuple"]) == (A0 if p["theta"] == "SIN_LOW" else B0), "gold mapping")
    man, pol, renders = load_atlas(P)
    fz = json.load(open(FREEZE))
    gate(fz["n_checkpoints"] == 8 and len(fz["checkpoints"]) == 8, "FREEZE 8")
    CK = [(c["seed"], c["representation"], c["path"], c["sha256"]) for c in fz["checkpoints"]]
    gate(len({c[3] for c in CK}) == 8, "8 DISTINCT")
    agg = json.load(open(STAGEA_AGG))
    ANCHOR_EXPECT = {}
    for ck, v in agg["per_checkpoint"].items():
        ANCHOR_EXPECT[ck] = {RAW_IDX: (v["RAW"]["T"], v["RAW"]["B"]),
                             R488: (v["R488"]["T"], v["R488"]["B"])}
    gate(len(ANCHOR_EXPECT) == 8, "ANCHOR EXPECT 8")
    policies = SMOKE_POLICIES if SMOKE else list(range(720))
    mask0_set = [i for i in MASK0_SET if i in policies] if SMOKE else MASK0_SET
    if SMOKE:
        P = P[:2]
        CK = CK[:1]
    n = len(P)
    # Stage-A FULL RAW rows as the replay reference
    old = {}
    for l in open(STAGEA):
        r = json.loads(l)
        if r["cohort"] == "FRESH" and r["arm"] == "FULL":
            k = (r["seed"], r["representation"], r["view"], r["state"], r["pair_id"],
                 r["theta"], r["cur_sha"], tuple(r["candidate"]))
            gate(k not in old, "DUP STAGE-A ROW")
            old[k] = (r["lps"], r["sum"], r["gold"])
    gate(len(old) == 2 * 8 * 384, "STAGE-A FULL ROWS")
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
                       "raw_cur_sha": p["cur_sha"], "gold": tuple(p["gold_tuple"]),
                       "g": 1 if p["theta"] == "SIN_LOW" else -1, "conts": conts})
    for st in states:
        gate(renders[(RAW_IDX, st["pair_id"], st["theta"])] == P[st["i"]]["cur"], "RAW RENDER IS PRIMARY CUR")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    receipt = {"smoke": SMOKE, "prereg": PREREG, "n_states": n, "n_policies": len(policies),
               "pins": {p: fsha(p) for p in PINS},
               "anchors": {k: {"atlas_index": v[0], "render_id": v[1], "roles": list(v[2])}
                           for k, v in ANCHORS.items()},
               "r488_index": R488, "mask0_subset": mask0_set, "subset13": SUBSET13,
               "subset13_sha": SUBSET13_SHA, "lp_subset": sorted(LP_SET_B),
               "semantic_order": [list(s) for s in SEM],
               "eps": {"EPS_SCORE": EPS_SCORE, "EPS_D": EPS_D},
               "semantic_beyond_all_surface_identifiable": False,
               "torch": torch.__version__, "device": "mps", "chunks": {}}

    def finish(verdict):
        receipt["verdict"] = verdict
        receipt["wall_s"] = round(time.monotonic() - t0, 1)
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "prband2atlasfresh_receipt.json").write_text(json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items() if k not in ("start", "pins")},
                         indent=1)[:4000], flush=True)

    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    for seed, rep, path, sha_exp in CK:
        ck = f"{seed}|{rep}"
        cdir = OUTDIR / f"chunk_{seed}_{rep}"
        crec = cdir / "chunk_receipt.json"
        if crec.exists():
            cr = json.loads(crec.read_text())
            if cr.get("verdict") == "CHUNK COMPLETE" and \
                    fsha(str(cdir / "scores.jsonl")) == cr["scores_sha256"]:
                receipt["chunks"][ck] = {"retained": True, **cr}
                print(f"[RETAINED {ck}]", flush=True)
                continue
        if cdir.exists():
            aside = cdir.with_name(cdir.name + f"_partial_{int(time.time())}")
            cdir.rename(aside)
            receipt.setdefault("partials_set_aside", []).append(str(aside))
        cdir.mkdir(parents=True)
        tc = time.monotonic()
        gate(fsha(path) == sha_exp, f"CKPT SHA {path}")
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(path, weights_only=True))
        gate(sum(q.numel() for q in m.parameters()) == 19142016, "PARAMS")
        m.eval()
        m = m.to(dev)
        gate(len(m.blocks) == 8, "8 BLOCKS")
        chunk = {"seed": seed, "representation": rep, "path": path, "sha256": sha_exp,
                 "device": "mps", "torch": torch.__version__,
                 "dtype": str(next(m.parameters()).dtype), "n_states": n,
                 "policies": len(policies)}
        out = open(cdir / "scores.jsonl", "w")

        def score_policy(idx, mask, arm, keep_lps):
            rows_out = []
            rid = man[idx]["render_id"]
            for st in states:
                cur = renders[(idx, st["pair_id"], st["theta"])]
                lps = masked_token_lps(m, dev, cur, st["conts"][rep], mask)
                for s, lp, cont in zip(SEM, lps, st["conts"][rep]):
                    gate(len(lp) == 9, "T!=9")
                    tot = float(sum(lp))
                    row = {"cohort": "FRESH", "seed": seed, "representation": rep,
                           "ckpt_sha": sha_exp, "arm": arm, "mask": mask,
                           "atlas_index": idx, "render_id": rid, "state": st["i"],
                           "pair_id": st["pair_id"], "theta": st["theta"],
                           "cur_sha": hashlib.sha256(cur.encode()).hexdigest(),
                           "candidate": list(s), "name": NAMES[s], "sum": tot,
                           "gold": list(st["gold"])}
                    if keep_lps:
                        row["lps"] = lp
                        row["continuation"] = cont
                    rows_out.append((st, s, lp, tot))
                    out.write(json.dumps(row) + "\n")
            out.flush()
            return rows_out

        gate_rows = score_policy(RAW_IDX, 255, "FULL", True)
        exact, drift = 0, 0.0
        for st, s, lp, tot in gate_rows:
            k = (seed, rep, "RAW", st["i"], st["pair_id"], st["theta"], st["raw_cur_sha"], s)
            gate(k in old, f"STAGE-A ROW MISSING {k}")
            olp, osum, ogold = old[k]
            gate(ogold == list(st["gold"]), "STAGE-A GOLD")
            exact += (olp == lp) and (osum == tot)
            drift = max(drift, abs(osum - tot), max(abs(a - b) for a, b in zip(olp, lp)))
        chunk["stage_a_replay"] = {"rows": len(gate_rows), "exact": exact,
                                   "max_abs_drift": drift, "pass": exact == len(gate_rows) == n * 4}
        print(f"[REPLAY {ck}] exact {exact}/{len(gate_rows)} drift {drift}", flush=True)
        if not chunk["stage_a_replay"]["pass"]:
            out.close()
            chunk["verdict"] = "STAGE-A-REPLAY FAILURE"
            crec.write_text(json.dumps(chunk, indent=1))
            receipt["chunks"][ck] = chunk
            finish("STAGE-A-REPLAY FAILURE — ATLAS NOT SCORED")
            return
        gate_key = {(st["i"], s): (lp, tot) for st, s, lp, tot in gate_rows}
        T, Bc, seq_ok, r488_ok = {}, {}, True, True
        pairs = defaultdict(dict)
        for st in states:
            pairs[st["pair_id"]][st["theta"]] = st["i"]
        for pi, idx in enumerate(policies):
            rows = score_policy(idx, 255, "FULL", idx in LP_SET_B)
            if idx == RAW_IDX:
                seq_ok = all(gate_key[(st["i"], s)] == (lp, tot) for st, s, lp, tot in rows)
            if idx == R488 and not SMOKE:
                for st, s, lp, tot in rows:
                    k = (seed, rep, "R488", st["i"], st["pair_id"], st["theta"],
                         hashlib.sha256(renders[(R488, st["pair_id"], st["theta"])].encode()).hexdigest(), s)
                    gate(k in old, f"STAGE-A R488 ROW MISSING {k}")
                    olp, osum, _g = old[k]
                    r488_ok = r488_ok and (olp == lp) and (osum == tot)
            sc = defaultdict(dict)
            for st, s, lp, tot in rows:
                sc[st["i"]][s] = tot
            corr = {st["i"]: top1_of(sc[st["i"]])[0] == st["gold"] for st in states}
            T[idx] = sum(corr.values())
            Bc[idx] = sum(corr[p["SIN_LOW"]] and corr[p["COS_LOW"]]
                          for p in pairs.values() if len(p) == 2)
            if pi % 20 == 0 or pi == len(policies) - 1:
                print(f"[{ck}] policy {pi + 1}/{len(policies)} idx {idx} T {T[idx]} B {Bc[idx]} "
                      f"{time.monotonic() - tc:.0f}s", flush=True)
        chunk["sequence_replay_pass"] = seq_ok
        chunk["r488_replay_pass"] = r488_ok
        if not (seq_ok and r488_ok):
            out.close()
            chunk["verdict"] = "SEQUENCE-REPLAY FAILURE"
            crec.write_text(json.dumps(chunk, indent=1))
            receipt["chunks"][ck] = chunk
            finish("SEQUENCE-REPLAY FAILURE — CHUNK WITHOUT AUTHORITY")
            return
        m0 = {}
        for idx in mask0_set:
            for st, s, lp, tot in score_policy(idx, 0, "MASK0", True):
                m0.setdefault((st["i"], s), {})[idx] = tot
        spread = max(max(v.values()) - min(v.values()) for v in m0.values())
        switches = flips = 0
        same_top = True
        for st in states:
            per = {idx: {s: m0[(st["i"], s)][idx] for s in SEM} for idx in mask0_set}
            ws = {idx: top1_of(v)[0] for idx, v in per.items()}
            same_top = same_top and len(set(ws.values())) == 1
            base = mask0_set[0]
            for idx in mask0_set[1:]:
                if ws[idx] != ws[base] and top1_of(per[base])[2] >= EPS_D and top1_of(per[idx])[2] >= EPS_D:
                    switches += 1
                d0 = per[base][A0] - per[base][B0]
                d1 = per[idx][A0] - per[idx][B0]
                if sgn(d0) != sgn(d1) and abs(d0) >= EPS_D and abs(d1) >= EPS_D:
                    flips += 1
        chunk["mask0"] = {"policies": mask0_set, "max_spread": spread,
                          "same_top_all_renders": same_top, "robust_top_switches": switches,
                          "robust_margin_flips": flips,
                          "pass": spread <= EPS_SCORE and same_top and switches == 0 and flips == 0}
        out.close()
        anc = {}
        if not SMOKE:
            for idx in (RAW_IDX, R488):
                exp = ANCHOR_EXPECT[ck][idx]
                anc[str(idx)] = {"T": T[idx], "B": Bc[idx], "expected_T": exp[0],
                                 "expected_B": exp[1], "match": (T[idx], Bc[idx]) == exp}
        chunk["anchor_reproduction"] = anc
        chunk["anchor_reproduction_pass"] = all(a["match"] for a in anc.values())
        chunk["T"] = {str(i): T[i] for i in policies}
        chunk["B"] = {str(i): Bc[i] for i in policies}
        chunk["scores_sha256"] = fsha(str(cdir / "scores.jsonl"))
        seen, keys = Counter(), set()
        for l in open(cdir / "scores.jsonl"):
            r = json.loads(l)
            keys.add((r["arm"], r["atlas_index"], r["state"], tuple(r["candidate"])))
            seen[r["arm"]] += 1
        gate(seen["FULL"] == (len(policies) + 1) * n * 4, "FULL ROWS")
        gate(seen["MASK0"] == len(mask0_set) * n * 4, "MASK0 ROWS")
        gate(len(keys) == (len(policies) + len(mask0_set)) * n * 4, "KEYS")
        chunk["rows"] = dict(seen)
        chunk["distinct_keys"] = len(keys)
        chunk["wall_s"] = round(time.monotonic() - tc, 1)
        chunk["verdict"] = "CHUNK COMPLETE"
        crec.write_text(json.dumps(chunk, indent=1))
        receipt["chunks"][ck] = chunk
        print(f"[CHUNK COMPLETE {ck}] wall {chunk['wall_s']}s mask0 {chunk['mask0']['pass']} "
              f"anchors {chunk['anchor_reproduction_pass']}", flush=True)
        del m
        torch.mps.empty_cache()
    complete = [c for c in receipt["chunks"].values() if c.get("verdict") == "CHUNK COMPLETE"]
    ok = len(complete) == (1 if SMOKE else 8) and all(
        c["stage_a_replay"]["pass"] and c["mask0"]["pass"] and c["r488_replay_pass"]
        and (SMOKE or c["anchor_reproduction_pass"]) for c in complete)
    finish("SMOKE OK" if (SMOKE and ok) else "ATLAS SCORED" if ok else "ATLAS INCOMPLETE OR GATE FAILURE")


if __name__ == "__main__":
    main()

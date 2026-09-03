"""MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-NUISANCE-COUNTERFACTUAL-0
execution scorer (adopt-not-fork of scratch/mathworld1_prband2score.py).

Scores the SAME 96 frozen states under three PERSISTED prompt
renderings (RAW, K_FIRST, LOW_PAIR_FIRST; read by (pair_id, theta,
view) from the pinned logs/mathworld1/prband2nuis/views.jsonl, never
regenerated) x the same four sha-pinned checkpoints x {FULL mask
255, MASK0 mask 0} = 24 cells. Scoring law, candidate semantic
order, continuations, T = 9 SUM, strict top-1 and the 1e-05 noise
bound are imported from the booked scorer unchanged; the only new
responsibility is the prompt lookup.

Execution order (frozen): pins -> pre-score artifact gates on the
three views (SymPy + candidate law + tokenizer + CTX, 96/96 each)
-> checkpoint hashes -> ALL 8 RAW cells re-scored and compared to
the tracked booked stream with exact parsed-float equality (RAW
REPLAY, abort-level: any mismatch writes the failure receipt with
the measured drift and stops before any novel-view logit) -> K_FIRST
cells -> LOW_PAIR_FIRST cells -> frozen per-state / pair classes,
transition matrices, MASK0 render-invariance sanity, reading law.

Frozen numerics: EPS_SCORE 1e-05, EPS_D 2e-05, EPS_DELTA 4e-05.
semantic_beyond_all_surface_identifiable = False, permanently.

Usage:
    PRBAND2CF_SMOKE=1 .venv/bin/python scratch/mathworld1_prband2cf.py
    .venv/bin/python scratch/mathworld1_prband2cf.py
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sympy as sp  # noqa: E402
import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_prband import cand_sig  # noqa: E402
from scratch.mathworld1_prband2score import (A0, ARMS, B0,  # noqa: E402
                                             CKPTS, CODE_BASE, NAMES,
                                             NOISE, PRIMARY, SEM, TOK,
                                             VOCAB, ctup, fsha, top1_of)
from scratch.mathworld1_respath import masked_token_lps  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import (AFTER_D, CTX,  # noqa: E402
                                        qualify_parent)
from scratch.mathworld1_svpcode import factor_decode  # noqa: E402
from scratch.mathworld1_svpforder import (PERM, pf_decode,  # noqa: E402
                                          pf_encode)

SMOKE = os.environ.get("PRBAND2CF_SMOKE") == "1"
PREREG = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-NUISANCE-COUNTERFACTUAL-0"
PREREG_COMMIT = "8800ec8bec67748bad8d7e7a7fc5089324dd25ac"
OUTDIR = Path("logs/mathworld1/prband2cf_smoke" if SMOKE
              else "logs/mathworld1/prband2cf")
SMOKE_RECEIPT = Path("logs/mathworld1/prband2cf_smoke/prband2cf_receipt.json")
VIEWS = "logs/mathworld1/prband2nuis/views.jsonl"
OLD_RAW = "logs/mathworld1/prband2score/raw_scores.jsonl"
PINS = {
    VIEWS: "677201ccc0cf34fbdf2b2e060146b68c157a2450926ac062f1c0f16cac8a72bb",
    "logs/mathworld1/prband2nuis/prband2nuis_receipt.json":
        "27729b4234fc98016cceaf2411ebca85a41d00cd617a039cadd853c3c23d4fc6",
    PRIMARY: "209391ef3b2e5c87308571d6ef309bb5724a214160caa1b7857f4a31f9112c34",
    OLD_RAW: "68d014ac2a0bf5b085941daeeae7def5a30506c7bb45c91630a96113b01cb31e",
}
VIEW_NAMES = ["RAW", "K_FIRST", "LOW_PAIR_FIRST"]
EPS_SCORE = NOISE          # 1e-05, carried
EPS_D = 2e-05
EPS_DELTA = 4e-05
SLACK_PAIRS, SLACK_TOP1, SLACK_SHIFT = 2, 4, 2
PROMPT = "Current: {cur}\nHints: none\nStep: "
RAW_ANCHORS = {"19001|CANONICAL": {"top1": 29, "both": 0},
               "19001|PARAM_FIRST": {"top1": 48, "both": 0},
               "20001|CANONICAL": {"top1": 96, "both": 48},
               "20001|PARAM_FIRST": {"top1": 62, "both": 14}}


def sgn(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def view_gates(P, views):
    """Frozen pre-score artifact gates, per view, 96/96 (or the smoke
    slice). Runs SymPy and the candidate law on the artifact only."""
    report = {}
    for vn in VIEW_NAMES:
        ok = Counter()
        for p in P:
            key = (p["pair_id"], p["theta"], vn)
            gate(key in views, f"VIEW ROW MISSING {key}")
            cur_v = views[key]["cur"]
            parent = sp.sympify(p["cur"])
            gate(hashlib.sha256(sp.srepr(parent).encode()).hexdigest()[:16]
                 == p["parent_srepr_sha"], "PARENT PIN")
            if vn == "RAW":
                gate(cur_v == p["cur"] and hashlib.sha256(
                    cur_v.encode()).hexdigest() == p["cur_sha"], "RAW BYTES")
            p2 = sp.sympify(cur_v)
            gate(sp.srepr(p2) == sp.srepr(parent), f"SREPR {vn}")
            gate(sp.simplify(p2 - parent) == 0, f"SIMPLIFY {vn}")
            f = [a for a in parent.args if isinstance(a, sp.Integral)
                 and a.function != AFTER_D][0].function
            f2 = [a for a in p2.args if isinstance(a, sp.Integral)
                  and a.function != AFTER_D][0].function
            gate(Counter(map(sp.srepr, f2.args))
                 == Counter(map(sp.srepr, f.args)) and len(f2.args) == 6,
                 f"SIX TERMS {vn}")
            q, why = qualify_parent(f2, AFTER_D)
            gate(why is None, f"LAW {vn} {why}")
            _js, sid = cand_sig(q["candidates"])
            gate(sid == p["cand_sig_id"], f"CAND_SIG {vn}")
            gate([q["chosen_rule"], q["chosen_site_kind"], q["chosen_ordinal"],
                  q["chosen_param_kind"], q["chosen_term"]] == p["teacher"]
                 and p["teacher"] == p["gold_tuple"], f"TEACHER/GOLD {vn}")
            gate(sorted(c["child_srepr"] for c in q["candidates"])
                 == sorted(c["child_srepr"] for c in p["candidates"]),
                 f"CHILD SET {vn}")
            gate(q["min_hce_ties"] == p["min_hce_ties"] == 1, f"TIES {vn}")
            gate(sorted(ctup(c) for c in q["candidates"]) == sorted(SEM),
                 f"SEMANTIC SET {vn}")
            prompt = PROMPT.format(cur=cur_v)
            ids = TOK.encode(prompt)
            gate(TOK.decode(ids) == prompt, f"TOK RT {vn}")
            gate(len(ids) == views[key]["prompt_tokens"], f"TOKENS {vn}")
            gate(len(ids) + 9 <= CTX, f"CTX {vn}")
            ok[vn] += 1
        report[vn] = ok[vn]
    return report


def classify(dR, dK, dL, g):
    """Frozen per-state flags and class."""
    mR, mK, mL = g * dR, g * dK, g * dL
    DK, DL = dK - dR, dL - dR
    rob = lambda a, b: sgn(a) != sgn(b) and abs(a) >= EPS_D and abs(b) >= EPS_D  # noqa: E731
    fl = {"ROBUST_RAW_LOW_FLIP": rob(dR, dL),
          "ROBUST_RAW_K_FLIP": rob(dR, dK),
          "ROBUST_DELTA_LOW": abs(DL) >= EPS_DELTA,
          "ROBUST_DELTA_K": abs(DK) >= EPS_DELTA}
    if min(abs(dR), abs(dK), abs(dL)) < EPS_D:
        cls, pol = "SUBNOISE", None
    elif fl["ROBUST_RAW_LOW_FLIP"] and fl["ROBUST_DELTA_LOW"]:
        cls = "CUE-FOLLOWING"
        pol = ("CORRECT-POLARITY" if mR >= EPS_D and mL <= -EPS_D
               else "REVERSED-POLARITY" if mR <= -EPS_D and mL >= EPS_D
               else None)
    elif fl["ROBUST_RAW_K_FLIP"]:
        cls, pol = "NEUTRAL-SHIFT", None
    else:
        cls, pol = "RENDER-SIGN-INVARIANT", None
    return {"d_RAW": dR, "d_K": dK, "d_LOW": dL, "m_RAW": mR, "m_K": mK,
            "m_LOW": mL, "Delta_K": DK, "Delta_LOW": DL, "flags": fl,
            "class": cls, "polarity": pol}


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN DRIFT {p}")
    if not SMOKE:
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(sr.get("smoke") is True and sr.get("verdict") == "SMOKE OK",
             "SMOKE NOT GREEN")
        for pth, h in sr["start"]["file_sha256"].items():
            gate(fsha(pth) == h, f"SMOKE STALE {pth}")
    START = start_provenance(
        ["scratch/mathworld1_prband2cf.py", "scratch/mathworld1_prband2score.py",
         "scratch/mathworld1_respath.py", "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpforder.py", "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py", "scratch/mathworld1_svpbirth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    t0 = time.monotonic()
    P = [json.loads(l) for l in open(PRIMARY)]
    if SMOKE:
        P = P[:2]
    n = len(P)
    if not SMOKE:
        gate(n == 96 and len({p["pair_key"] for p in P}) == 48, "N/keys")
        gate(Counter(p["theta"] for p in P) == Counter({"SIN_LOW": 48,
                                                        "COS_LOW": 48}), "theta")
        gate(Counter(tuple(p["gold_tuple"]) for p in P)
             == Counter({A0: 48, B0: 48}), "gold")
        gate(len({p["cand_sig_id"] for p in P}) == 1, "one signature")
    for k in range(0, n, 2):
        gate(P[k]["pair_key"] == P[k + 1]["pair_key"]
             and P[k]["theta"] == "SIN_LOW" and P[k + 1]["theta"] == "COS_LOW",
             "pair adjacency")
    for p in P:
        gate(sorted(tuple(t) for t in p["cand_tuples"]) == sorted(SEM),
             "SEMANTIC SET")
        gate(tuple(p["gold_tuple"]) == (A0 if p["theta"] == "SIN_LOW" else B0),
             "gold mapping")
    views = {}
    for l in open(VIEWS):
        r = json.loads(l)
        if r["view"] in VIEW_NAMES:
            key = (r["pair_id"], r["theta"], r["view"])
            gate(key not in views, "DUP VIEW ROW")
            views[key] = r
    gate(len(views) == 96 * 3, "VIEW ROWS")
    OUTDIR.mkdir(parents=True)
    gates = view_gates(P, views)
    gate(all(gates[v] == n for v in VIEW_NAMES), "VIEW GATES")
    # old raw stream, canonical payload
    old = {}
    for l in open(OLD_RAW):
        r = json.loads(l)
        k = (r["seed"], r["representation"], r["arm"], r["mask"], r["state"],
             r["pair_id"], r["theta"], r["cur_sha"], tuple(r["candidate"]))
        gate(k not in old, "DUP OLD ROW")
        old[k] = (r["lps"], r["sum"], r["gold"])
    gate(len(old) == 3072, "OLD ROWS")
    states = []
    for i, p in enumerate(P):
        by = {ctup(c): c for c in p["candidates"]}
        conts = {"CANONICAL": [], "PARAM_FIRST": []}
        for s in SEM:
            cz = by[s]["factor_code"]
            gate(factor_decode(cz) == s, "C RT")
            pz = pf_encode(s)
            gate(pf_decode(pz) == s and pz == [cz[PERM[i2]] for i2 in range(8)],
                 "PF RT / PERM")
            conts["CANONICAL"].append([CODE_BASE + x for x in cz] + [TOK.eos_id])
            conts["PARAM_FIRST"].append([CODE_BASE + x for x in pz]
                                        + [TOK.eos_id])
        states.append({"i": i, "pair_id": p["pair_id"], "theta": p["theta"],
                       "raw_cur_sha": p["cur_sha"], "gold": tuple(p["gold_tuple"]),
                       "g": 1 if p["theta"] == "SIN_LOW" else -1,
                       "conts": conts,
                       "cur": {vn: views[(p["pair_id"], p["theta"], vn)]["cur"]
                               for vn in VIEW_NAMES}})
    receipt = {"smoke": SMOKE, "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
               "n_states": n, "pins": {p: fsha(p) for p in PINS},
               "views": VIEW_NAMES, "view_gates": gates,
               "semantic_order": [list(s) for s in SEM],
               "eps": {"EPS_SCORE": EPS_SCORE, "EPS_D": EPS_D,
                       "EPS_DELTA": EPS_DELTA},
               "slack": {"pairs": SLACK_PAIRS, "top1": SLACK_TOP1,
                         "shift_pairs": SLACK_SHIFT},
               "semantic_beyond_all_surface_identifiable": False,
               "checkpoints": []}

    def finish(verdict, extra):
        receipt["verdict"] = verdict
        receipt.update(extra)
        receipt["wall_s"] = round(time.monotonic() - t0, 1)
        receipt["artifact_sha256"] = {
            f.name: fsha(f) for f in sorted(OUTDIR.glob("*.json*"))
            if f.name != "prband2cf_receipt.json"}
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "prband2cf_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ("start", "pins", "cells", "per_state",
                                       "pairs")}, indent=1), flush=True)

    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    receipt["device"], receipt["torch"] = "mps", torch.__version__
    models = []
    for seed, rep, path, sha_exp in (CKPTS[:1] if SMOKE else CKPTS):
        gate(fsha(path) == sha_exp, f"CKPT SHA {path}")
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(path, weights_only=True))
        gate(sum(q.numel() for q in m.parameters()) == 19142016, "PARAMS")
        m.eval()
        m = m.to(dev)
        gate(len(m.blocks) == 8, "8 BLOCKS")
        receipt["checkpoints"].append({"seed": seed, "representation": rep,
                                       "path": path, "sha256": sha_exp})
        models.append((seed, rep, m))
    raw = open(OUTDIR / "cf_scores.jsonl", "w")
    S = defaultdict(dict)   # (seed,rep,arm,view) -> {i: {sem: sum}}

    def score_cell(seed, rep, m, arm, mask, vn):
        cell = {}
        for st in states:
            lps = masked_token_lps(m, dev, st["cur"][vn], st["conts"][rep], mask)
            cell[st["i"]] = {}
            for s, lp, cont in zip(SEM, lps, st["conts"][rep]):
                gate(len(lp) == 9, "T!=9")
                tot = float(sum(lp))
                cell[st["i"]][s] = (lp, tot)
                raw.write(json.dumps({
                    "seed": seed, "representation": rep, "arm": arm,
                    "mask": mask, "view": vn, "state": st["i"],
                    "pair_id": st["pair_id"], "theta": st["theta"],
                    "cur_sha": hashlib.sha256(st["cur"][vn].encode()).hexdigest(),
                    "raw_cur_sha": st["raw_cur_sha"], "candidate": list(s),
                    "name": NAMES[s], "continuation": cont, "lps": lp,
                    "sum": tot, "gold": list(st["gold"])}) + "\n")
        raw.flush()
        S[(seed, rep, arm, vn)] = {i: {s: v[1] for s, v in c.items()}
                                   for i, c in cell.items()}
        return cell

    # ---- RAW REPLAY (abort-level) ----
    replay = {"cells": {}, "rows_compared": 0, "rows_exact": 0,
              "max_abs_lp_drift": 0.0, "max_abs_sum_drift": 0.0}
    for seed, rep, m in models:
        for arm, mask in ARMS:
            cell = score_cell(seed, rep, m, arm, mask, "RAW")
            exact = 0
            for st in states:
                for s in SEM:
                    lp, tot = cell[st["i"]][s]
                    k = (seed, rep, arm, mask, st["i"], st["pair_id"],
                         st["theta"], st["raw_cur_sha"], s)
                    gate(k in old, f"OLD ROW MISSING {k}")
                    olp, osum, ogold = old[k]
                    gate(ogold == list(st["gold"]), "OLD GOLD")
                    same = (olp == lp) and (osum == tot)
                    exact += same
                    replay["rows_compared"] += 1
                    replay["max_abs_lp_drift"] = max(
                        replay["max_abs_lp_drift"],
                        max(abs(a - b) for a, b in zip(olp, lp)))
                    replay["max_abs_sum_drift"] = max(
                        replay["max_abs_sum_drift"], abs(osum - tot))
            replay["rows_exact"] += exact
            replay["cells"][f"{seed}|{rep}|{arm}"] = {
                "rows": n * 4, "exact": exact}
            print(f"[REPLAY {seed} {rep} {arm}] exact {exact}/{n * 4}", flush=True)
    replay["exact"] = replay["rows_exact"] == replay["rows_compared"]
    (OUTDIR / "replay.json").write_text(json.dumps(replay, indent=1))
    if not replay["exact"]:
        raw.close()
        finish("RAW-REPLAY FAILURE — COUNTERFACTUAL NOVEL VIEWS NOT RUN",
               {"replay": replay})
        return
    receipt["replay"] = replay

    # ---- novel views, RAW-first order kept: K_FIRST then LOW_PAIR_FIRST ----
    for vn in ("K_FIRST", "LOW_PAIR_FIRST"):
        for seed, rep, m in models:
            for arm, mask in ARMS:
                score_cell(seed, rep, m, arm, mask, vn)
                print(f"[SCORED {seed} {rep} {arm} {vn}]", flush=True)
    raw.close()

    # ---- aggregates ----
    def top(scores):
        w, tie, mg = top1_of(scores)
        return w, tie, mg

    cells, per_state_all, pairs_all, mask0 = {}, {}, {}, {}
    for seed, rep, m in models:
        ck = f"{seed}|{rep}"
        for arm, mask in ARMS:
            tops = {}
            for vn in VIEW_NAMES:
                sc = S[(seed, rep, arm, vn)]
                rows = []
                for st in states:
                    w, tie, mg = top(sc[st["i"]])
                    rows.append({"i": st["i"], "top": NAMES.get(w) if w else None,
                                 "tie": tie, "margin": mg,
                                 "correct": w == st["gold"]})
                tops[vn] = rows
                corr = [r["correct"] for r in rows]
                d = [sc[st["i"]][A0] - sc[st["i"]][B0] for st in states]
                mm = [st["g"] * dv for st, dv in zip(states, d)]
                cells[f"{ck}|{arm}|{vn}"] = {
                    "top1": sum(corr), "ties": sum(r["tie"] for r in rows),
                    "A0_correct": sum(c for c, st in zip(corr, states)
                                      if st["gold"] == A0),
                    "B0_correct": sum(c for c, st in zip(corr, states)
                                      if st["gold"] == B0),
                    "both_correct_pairs": sum(corr[k] and corr[k + 1]
                                              for k in range(0, n, 2)),
                    "top_census": dict(Counter(r["top"] for r in rows)),
                    "target_sign_census": {
                        "gold_directed": sum(v >= EPS_D for v in mm),
                        "opposite_gold": sum(v <= -EPS_D for v in mm),
                        "subnoise": sum(abs(v) < EPS_D for v in mm)},
                    "correct_direction_pairs": sum(
                        mm[k] >= EPS_D and mm[k + 1] >= EPS_D
                        for k in range(0, n, 2)),
                    "reversed_direction_pairs": sum(
                        mm[k] <= -EPS_D and mm[k + 1] <= -EPS_D
                        for k in range(0, n, 2))}
            # render contrasts
            def trans(a, b):
                t = Counter((r1["top"], r2["top"]) for r1, r2 in
                            zip(tops[a], tops[b]))
                return {f"{x}->{y}": c for (x, y), c in sorted(
                    t.items(), key=lambda kv: str(kv[0]))}
            changes = {}
            for a, b in (("RAW", "K_FIRST"), ("RAW", "LOW_PAIR_FIRST"),
                         ("K_FIRST", "LOW_PAIR_FIRST")):
                ch = [r1["top"] != r2["top"] for r1, r2 in zip(tops[a], tops[b])]
                robust = [c and r1["margin"] >= EPS_SCORE
                          and r2["margin"] >= EPS_SCORE
                          for c, r1, r2 in zip(ch, tops[a], tops[b])]
                changes[f"{a}->{b}"] = {
                    "states_changed": sum(ch), "robust_changes": sum(robust),
                    "pairs_with_change": sum(ch[k] or ch[k + 1]
                                             for k in range(0, n, 2)),
                    "transition": trans(a, b)}
            ps = []
            for st in states:
                i = st["i"]
                dR = S[(seed, rep, arm, "RAW")][i][A0] - S[(seed, rep, arm, "RAW")][i][B0]
                dK = S[(seed, rep, arm, "K_FIRST")][i][A0] - S[(seed, rep, arm, "K_FIRST")][i][B0]
                dL = S[(seed, rep, arm, "LOW_PAIR_FIRST")][i][A0] - S[(seed, rep, arm, "LOW_PAIR_FIRST")][i][B0]
                c = classify(dR, dK, dL, st["g"])
                c.update({"i": i, "pair_id": st["pair_id"], "theta": st["theta"],
                          "top": {vn: tops[vn][i]["top"] for vn in VIEW_NAMES}})
                ps.append(c)
            census = Counter(c["class"] for c in ps)
            pol = Counter(f'{c["class"]} {c["polarity"]}' for c in ps
                          if c["polarity"])
            prs = []
            for k in range(0, n, 2):
                a, b = ps[k], ps[k + 1]
                both_raw_ok = tops["RAW"][k]["correct"] and tops["RAW"][k + 1]["correct"]
                lab = {
                    "pair_id": a["pair_id"], "classes": [a["class"], b["class"]],
                    "polarities": [a["polarity"], b["polarity"]],
                    "both_correct": {vn: tops[vn][k]["correct"]
                                     and tops[vn][k + 1]["correct"]
                                     for vn in VIEW_NAMES},
                    "both_reverse_RAW_LOW": a["flags"]["ROBUST_RAW_LOW_FLIP"]
                    and b["flags"]["ROBUST_RAW_LOW_FLIP"],
                    "both_preserve_RAW_K": all(
                        sgn(x["d_RAW"]) == sgn(x["d_K"]) and abs(x["d_RAW"]) >= EPS_D
                        and abs(x["d_K"]) >= EPS_D for x in (a, b)),
                    "both_preserve_RAW_LOW": all(
                        sgn(x["d_RAW"]) == sgn(x["d_LOW"]) and abs(x["d_RAW"]) >= EPS_D
                        and abs(x["d_LOW"]) >= EPS_D for x in (a, b)),
                    "PAIR-BUNDLE-FOLLOWING CORRECT-POLARITY": all(
                        x["m_RAW"] >= EPS_D and x["m_LOW"] <= -EPS_D for x in (a, b)),
                    "PAIR-BUNDLE-FOLLOWING REVERSED-POLARITY": all(
                        x["m_RAW"] <= -EPS_D and x["m_LOW"] >= EPS_D for x in (a, b)),
                    "raw_both_correct_with_cue_following": both_raw_ok and any(
                        x["class"] == "CUE-FOLLOWING" for x in (a, b)),
                    "raw_both_correct_with_neutral_shift": both_raw_ok and any(
                        x["class"] == "NEUTRAL-SHIFT" for x in (a, b))}
                lab["PAIR-REGISTERED-CUE-REMOVAL-STABLE"] = lab["both_preserve_RAW_K"]
                lab["PAIR-BUNDLE-REVERSAL-STABLE"] = lab["both_preserve_RAW_LOW"]
                prs.append(lab)
            pair_census = {k: sum(p[k] for p in prs) for k in (
                "PAIR-BUNDLE-FOLLOWING CORRECT-POLARITY",
                "PAIR-BUNDLE-FOLLOWING REVERSED-POLARITY",
                "PAIR-REGISTERED-CUE-REMOVAL-STABLE",
                "PAIR-BUNDLE-REVERSAL-STABLE", "both_reverse_RAW_LOW",
                "raw_both_correct_with_cue_following",
                "raw_both_correct_with_neutral_shift")}
            dist = {}
            for key in ("Delta_K", "Delta_LOW"):
                vals = sorted(c[key] for c in ps)
                dist[key] = {"min": vals[0], "max": vals[-1],
                             "median": vals[len(vals) // 2],
                             "n_abs_ge_eps_delta": sum(abs(v) >= EPS_DELTA
                                                       for v in vals),
                             "n_pos": sum(v > 0 for v in vals),
                             "n_neg": sum(v < 0 for v in vals)}
            # reading law (frozen slack); every boolean reported independently
            cR = cells[f"{ck}|{arm}|RAW"]
            reading = {}
            for vn, shift_key in (("K_FIRST", "raw_both_correct_with_neutral_shift"),
                                  ("LOW_PAIR_FIRST", "raw_both_correct_with_cue_following")):
                cv = cells[f"{ck}|{arm}|{vn}"]
                surv = (cv["both_correct_pairs"] >= cR["both_correct_pairs"] - SLACK_PAIRS
                        and cv["top1"] >= cR["top1"] - SLACK_TOP1
                        and pair_census[shift_key] <= SLACK_SHIFT)
                rd = {"SURVIVES": surv}
                if vn == "LOW_PAIR_FIRST":
                    rd["COLLAPSES"] = cv["both_correct_pairs"] <= cR["both_correct_pairs"] // 2
                    rd["REVERSES"] = (pair_census["PAIR-BUNDLE-FOLLOWING CORRECT-POLARITY"]
                                      + pair_census["PAIR-BUNDLE-FOLLOWING REVERSED-POLARITY"]) >= 24
                rd["PARTIAL"] = not any(rd.values())
                reading[vn] = rd
            entry = {"changes": changes, "class_census": dict(census),
                     "polarity_census": dict(pol), "pair_census": pair_census,
                     "delta_dist": dist, "reading": reading}
            if arm == "MASK0":
                spread_max = 0.0
                for st in states:
                    for s in SEM:
                        vals = [S[(seed, rep, arm, vn)][st["i"]][s] for vn in VIEW_NAMES]
                        spread_max = max(spread_max, max(vals) - min(vals))
                same_top = all(len({tops[vn][st["i"]]["top"] for vn in VIEW_NAMES}) == 1
                               for st in states)
                rob_sw = sum(changes[k]["robust_changes"] for k in changes)
                rob_fl = sum(c["flags"]["ROBUST_RAW_K_FLIP"]
                             or c["flags"]["ROBUST_RAW_LOW_FLIP"] for c in ps)
                mask0[ck] = {"max_cross_view_spread": spread_max,
                             "spread_le_bound": spread_max <= EPS_SCORE,
                             "same_top_all_views": same_top,
                             "robust_top_switches": rob_sw,
                             "robust_margin_flips": rob_fl,
                             "sane": spread_max <= EPS_SCORE and same_top
                             and rob_sw == 0 and rob_fl == 0}
            cells[f"{ck}|{arm}|CONTRASTS"] = entry
            per_state_all[f"{ck}|{arm}"] = ps
            pairs_all[f"{ck}|{arm}"] = prs
            print(f"[{ck} {arm}] top1 " + " ".join(
                f"{vn}:{cells[f'{ck}|{arm}|{vn}']['top1']}/{n} both "
                f"{cells[f'{ck}|{arm}|{vn}']['both_correct_pairs']}"
                for vn in VIEW_NAMES) + f" census {dict(census)} pol {dict(pol)} "
                f"reading {reading}", flush=True)
    (OUTDIR / "cells.json").write_text(json.dumps(cells, indent=1))
    (OUTDIR / "per_state.json").write_text(json.dumps(per_state_all, indent=1))
    (OUTDIR / "pairs.json").write_text(json.dumps(pairs_all, indent=1))
    (OUTDIR / "mask0.json").write_text(json.dumps(mask0, indent=1))
    all_sane = all(v["sane"] for v in mask0.values())
    verdict = ("SMOKE OK" if SMOKE else
               "COUNTERFACTUAL SCORED" if all_sane else
               "INSTRUMENT FAILURE (MASK0 render invariance)")
    summary = {ck: {arm: {vn: {k: cells[f"{ck}|{arm}|{vn}"][k] for k in
                               ("top1", "A0_correct", "B0_correct",
                                "both_correct_pairs", "correct_direction_pairs",
                                "reversed_direction_pairs")}
                          for vn in VIEW_NAMES}
                    for arm, _ in ARMS}
               for ck in {f"{s}|{r}" for s, r, _ in models}}
    finish(verdict, {"summary": summary, "mask0": mask0,
                     "raw_anchors": RAW_ANCHORS,
                     "raw_anchor_reproduced": {
                         ck: (cells[f"{ck}|FULL|RAW"]["top1"] == RAW_ANCHORS[ck]["top1"]
                              and cells[f"{ck}|FULL|RAW"]["both_correct_pairs"]
                              == RAW_ANCHORS[ck]["both"])
                         for ck in summary} if not SMOKE else "smoke"})


if __name__ == "__main__":
    main()

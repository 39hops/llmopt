"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-V2-SCORING-EXECUTION-0 — execute
the frozen scoring prereg (PRE-REG ...V2-SCORING-0, ee9dfa5c, RESULTS
L64865) on the frozen RAW N=96 matched-pair primary artifact.
Adopt-not-fork: the token-lp law is scratch/mathworld1_respath.py
masked_token_lps (the svpfoheld token_lps law plus the loop-skip
block mask; mask 255 = FULL, mask 0 = MASK0), the model load law is
svpfoheld's (build_model(340, ctx=4096), weights_only load, param
count gate, eval, mps). No new scoring semantics.

ORDER OF OPERATIONS (frozen): pins -> population gates -> model-blind
baselines F0 / F_LEN / F_SIGN / F_ORDER / F_SURFACE persisted with
semantic_beyond_surface_identifiable = false -> candidate
canonicalization to semantic order -> checkpoint hash + load -> raw
per-token lps streamed per (checkpoint, arm, state, candidate) ->
metrics. Score = TOTAL SUM of the 9 teacher-forced continuation lps
incl. EOS; strict top-1; exact max ties = SCORE-TIE (loss); frozen
float-noise bound 1e-05 for MARGIN-SUBNOISE / FLOAT-NOISE-TIE
classes; MASK0 hard sanities; per-cell metrics; target-margin
anatomy; FULL-v-MASK0 tables; 2x2 matrix; no pooling, no p-value
primary.

Outputs (refuse-if-exists) under logs/mathworld1/prband2score/
(smoke: prband2score_smoke/): baselines.json (written BEFORE any
checkpoint load), raw_scores.jsonl (every checkpoint x arm x state x
candidate row with 9 lps), cells.json, receipt
prband2score_receipt.json.

    PRBAND2S_SMOKE=1 .venv/bin/python scratch/mathworld1_prband2score.py
    .venv/bin/python scratch/mathworld1_prband2score.py          (Mac)
"""
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_respath import masked_token_lps  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import factor_decode  # noqa: E402
from scratch.mathworld1_svpforder import (PERM, pf_decode,  # noqa: E402
                                          pf_encode)

SMOKE = os.environ.get("PRBAND2S_SMOKE") == "1"
PREREG = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-SCORING-0"
PREREG_COMMIT = "ee9dfa5c70190dead546b0a080ed9c3fb735a435"
OUTDIR = Path("logs/mathworld1/prband2score_smoke" if SMOKE
              else "logs/mathworld1/prband2score")
SMOKE_RECEIPT = Path("logs/mathworld1/prband2score_smoke/"
                     "prband2score_receipt.json")
VOCAB, CODE_BASE = 340, 332
TOK = ActionGCTok()
NOISE = 1e-05
PRIMARY = "logs/mathworld1/prband2prod/primary.jsonl"
PINS = {
    PRIMARY: "209391ef3b2e5c87308571d6ef309bb5724a214160caa1b7857f4a31f9112c34",
    "logs/mathworld1/prband2prod/nuisance.json":
        "e98eaed34df4a57deb0238ef20d35f2a9bef0e5371794f7b297317a01075d678",
    "logs/mathworld1/prband2prod/pairs.jsonl":
        "ee6cf884b5140a1f3bd2dff812b8bba475a1d2509b8e586377fd0395330a5d33",
    "logs/mathworld1/prband2prod/permutations.json":
        "b57626ae20d7c8645a11bafec69c917708fa26ae679fc65f8d654f54c8023919",
}
CKPTS = [
    ("19001", "CANONICAL", "checkpoints/svp_forder_canonical_s19001.pt",
     "ae0a86e027d8b0ca1cd7a97a83a6927d326da5bd34258910b1b81d3492322e1d"),
    ("19001", "PARAM_FIRST", "checkpoints/svp_forder_paramfirst_s19001.pt",
     "0fe38f785f68165e868c54fff482844ea4b2476c737f2e4af50990ece6df390f"),
    ("20001", "CANONICAL", "checkpoints/svp_forder_canonical_s20001.pt",
     "0a841a5f2a43b6f64b0dac8259c26fd79961e6ab91359a54be9c2582815b3e34"),
    ("20001", "PARAM_FIRST", "checkpoints/svp_forder_paramfirst_s20001.pt",
     "b7198ff2e7b903ab5ed075fe947cb29142c5790ec84831434c53a598e466c322"),
]
SEM = [("i_sum", "I", 0, "none", -1),
       ("i_unprod", "I", 0, "term_index", 1),
       ("i_unprod", "I", 0, "term_index", 3),
       ("i_unprod", "I", 0, "term_index", 5)]
NAMES = {SEM[0]: "i_sum", SEM[1]: "A0", SEM[2]: "B0", SEM[3]: "I0/t5"}
A0, B0 = SEM[1], SEM[2]
ARMS = [("FULL", 255), ("MASK0", 0)]


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def top1_of(scores):
    """scores: dict sem->float. Returns (winner or None, tie flag,
    runner-up margin)."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best = ranked[0][1]
    ties = [k for k, v in scores.items() if v == best]
    margin = best - ranked[1][1]
    return (ranked[0][0] if len(ties) == 1 else None), len(ties) > 1, margin


def baselines(P):
    perm = json.loads(Path("logs/mathworld1/prband2prod/permutations.json")
                      .read_text())
    # F0 recomputed from the 24 orders
    best_top, best_both = 0, 0
    for order in itertools.permutations(SEM):
        corr = [tuple(p["gold_tuple"]) == order[0] for p in P]
        best_top = max(best_top, sum(corr))
        best_both = max(best_both, sum(corr[k] and corr[k + 1]
                                       for k in range(0, len(P), 2)))
    if not SMOKE:  # smoke scores a 2-state slice; the pin is population-wide
        gate(perm["max_top1"] == best_top
             and perm["max_both_correct_pairs"] == best_both,
             "F0 v permutations.json")
    def acc(pred):
        corr = [pred(p) == tuple(p["gold_tuple"]) for p in P]
        return sum(corr), sum(corr[k] and corr[k + 1]
                              for k in range(0, len(P), 2))
    f_len = acc(lambda p: A0 if p["prompt_tokens"] == 89 else B0)
    f_sign = acc(lambda p: A0 if p["cur"].startswith("Integral(-") else B0)
    sigs = defaultdict(Counter)
    for p in P:
        sigs[json.dumps(p["cand_tuples"])][json.dumps(p["gold_tuple"])] += 1
    lookup = {s: json.loads(max(c, key=c.get)) for s, c in sigs.items()}
    f_order = acc(lambda p: tuple(lookup[json.dumps(p["cand_tuples"])]))
    out = {"F0": {"max_top1": best_top, "max_both_correct": best_both,
                  "n_orders": 24},
           "F_LEN": {"top1": f_len[0], "both_correct": f_len[1]},
           "F_SIGN": {"top1": f_sign[0], "both_correct": f_sign[1]},
           "F_ORDER": {"n_raw_signatures": len(sigs),
                       "signature_gold_census": {s[:80]: dict(c)
                                                 for s, c in sigs.items()},
                       "top1": f_order[0], "both_correct": f_order[1]},
           "F_SURFACE": {"top1": max(f_len[0], f_sign[0], f_order[0]),
                         "both_correct": max(f_len[1], f_sign[1],
                                             f_order[1])},
           "semantic_beyond_surface_identifiable": False,
           "reason": ("perfect prompt-length, leading-sign and raw-order "
                      "nuisance classes exist on the exact scored "
                      "artifact")}
    n, npair = len(P), len(P) // 2
    if not SMOKE:
        gate(out["F0"]["max_top1"] == 48 and out["F0"]["max_both_correct"] == 0
             and f_len == (n, npair) and f_sign == (n, npair)
             and f_order == (n, npair) and len(sigs) == 2,
             "BASELINE MISMATCH")
    return out


def cell_metrics(rows_cell, P, arm):
    """rows_cell: {state_idx: {sem: {'sum':..}}}. P: population rows."""
    n = len(P)
    per_state = []
    for i, p in enumerate(P):
        sc = {s: rows_cell[i][s]["sum"] for s in SEM}
        win, tie, margin = top1_of(sc)
        gold = tuple(p["gold_tuple"])
        per_state.append({"i": i, "theta": p["theta"], "gold": gold,
                          "pred": win, "tie": tie, "margin": margin,
                          "correct": (win == gold), "d": sc[A0] - sc[B0],
                          "scores": sc})
    top1 = sum(s["correct"] for s in per_state)
    ties = sum(s["tie"] for s in per_state)
    a_ok = sum(s["correct"] for s in per_state if s["gold"] == A0)
    b_ok = sum(s["correct"] for s in per_state if s["gold"] == B0)
    pairs = []
    for k in range(0, n, 2):
        s1, s2 = per_state[k], per_state[k + 1]
        both = s1["correct"] and s2["correct"]
        one = s1["correct"] != s2["correct"]
        switch = (s1["pred"] != s2["pred"]) and s1["pred"] is not None \
            and s2["pred"] is not None
        switch_margin = min(s1["margin"], s2["margin"])
        switch_noise = switch and switch_margin < NOISE
        dS, dC = s1["d"], s2["d"]
        if dS == 0 or dC == 0:
            fl = "TARGET-MARGIN-TIE"
        elif min(abs(dS), abs(dC)) < NOISE:
            fl = "MARGIN-SUBNOISE"
        elif dS > 0 and dC < 0:
            fl = "CORRECT-DIRECTION"
        elif dS < 0 and dC > 0:
            fl = "REVERSED"
        else:
            fl = "NO-FLIP"
        both_margin = min(s1["margin"], s2["margin"]) if both else None
        pairs.append({"pair": k // 2, "both": both, "one": one,
                      "neither": not (s1["correct"] or s2["correct"]),
                      "both_subnoise": both and both_margin < NOISE,
                      "both_margin": both_margin,
                      "switch": switch, "switch_noise": switch_noise,
                      "switch_margin": switch_margin, "d_SIN": dS,
                      "d_COS": dC, "flip": fl})
    conf = Counter()
    for s in per_state:
        conf[f"{NAMES[s['gold']]}->{NAMES[s['pred']] if s['pred'] else 'TIE'}"] += 1
    margins = [s["margin"] for s in per_state]
    dd = [pr["d_SIN"] - pr["d_COS"] for pr in pairs]
    spread = {}
    if arm == "MASK0":
        for s in SEM:
            vals = [rows_cell[i][s]["sum"] for i in range(n)]
            spread[NAMES[s]] = max(vals) - min(vals)
    robust_flips = sum(pr["flip"] in ("CORRECT-DIRECTION", "REVERSED")
                       for pr in pairs)
    m = {"top1": top1, "n": n, "score_ties": ties,
         "A0_correct": a_ok, "B0_correct": b_ok,
         "both_correct_pairs": sum(pr["both"] for pr in pairs),
         "both_correct_robust": sum(pr["both"] and not pr["both_subnoise"]
                                    for pr in pairs),
         "both_correct_subnoise": sum(pr["both_subnoise"] for pr in pairs),
         "exactly_one_pairs": sum(pr["one"] for pr in pairs),
         "neither_pairs": sum(pr["neither"] for pr in pairs),
         "switch_pairs_robust": sum(pr["switch"] and not pr["switch_noise"]
                                    for pr in pairs),
         "switch_pairs_float_noise": sum(pr["switch_noise"] for pr in pairs),
         "pred_census": dict(Counter(NAMES[s["pred"]] if s["pred"] else "TIE"
                                     for s in per_state)),
         "pred_census_by_theta": {t: dict(Counter(
             NAMES[s["pred"]] if s["pred"] else "TIE"
             for s in per_state if s["theta"] == t))
             for t in ("SIN_LOW", "COS_LOW")},
         "confusion": dict(conf),
         "margin_mean": statistics.mean(margins),
         "margin_median": statistics.median(margins),
         "flips": {"correct_direction": sum(pr["flip"] == "CORRECT-DIRECTION"
                                            for pr in pairs),
                   "reversed": sum(pr["flip"] == "REVERSED" for pr in pairs),
                   "robust_sign_flips": robust_flips,
                   "no_flip": sum(pr["flip"] == "NO-FLIP" for pr in pairs),
                   "tie": sum(pr["flip"] == "TARGET-MARGIN-TIE" for pr in pairs),
                   "subnoise": sum(pr["flip"] == "MARGIN-SUBNOISE"
                                   for pr in pairs),
                   "median_dSIN_minus_dCOS": statistics.median(dd) if dd else None,
                   "min_dSIN_minus_dCOS": min(dd) if dd else None,
                   "max_dSIN_minus_dCOS": max(dd) if dd else None},
         "d_SIN_range": [min(pr["d_SIN"] for pr in pairs),
                         max(pr["d_SIN"] for pr in pairs)],
         "d_COS_range": [min(pr["d_COS"] for pr in pairs),
                         max(pr["d_COS"] for pr in pairs)],
         "mask0_spread": spread}
    if arm == "MASK0":
        san = {"top1_le_half": top1 <= n // 2,
               "both_correct_zero": m["both_correct_pairs"] == 0,
               "switch_robust_zero": m["switch_pairs_robust"] == 0,
               "flips_robust_zero": robust_flips == 0,
               "spread_le_bound": all(v <= NOISE for v in spread.values())}
        m["mask0_sanities"] = san
        m["mask0_sanity_pass"] = all(san.values())
    if arm == "FULL":
        m["fixed_order_violated"] = top1 > n // 2
        m["state_conditioned_observed"] = m["both_correct_robust"] > 0
        # registered clause: both-correct > 0 OR any frozen fixed-
        # ranking theorem violated (top-1 > n/2; robust switch pairs;
        # robust margin sign flips). The production run at 3b72e21f
        # applied only the first two legs (booked, receipts frozen).
        m["switch_theorem_violated"] = m["switch_pairs_robust"] > 0
        m["flip_theorem_violated"] = robust_flips > 0
        m["classification"] = ("STATE-CONDITIONED-RANKING OBSERVED"
                               if m["state_conditioned_observed"]
                               or m["fixed_order_violated"]
                               or m["switch_theorem_violated"]
                               or m["flip_theorem_violated"]
                               else "STATE-BLIND FIXED-RANKING CLASS "
                               "NOT REFUTED")
        m["nuisance_agreement"] = {
            "F_LEN": sum((s["pred"] == (A0 if P[s["i"]]["prompt_tokens"] == 89
                                        else B0)) for s in per_state),
            "F_SIGN": sum((s["pred"] == (A0 if P[s["i"]]["cur"].startswith(
                "Integral(-") else B0)) for s in per_state),
            "note": ("all nuisance baselines equal gold in 96/96, so "
                     "agreement is numerically identical to correctness "
                     "and cannot identify feature use")}
    return m, per_state, pairs


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
        ["scratch/mathworld1_prband2score.py",
         "scratch/mathworld1_respath.py", "scratch/mathworld1_svpfoheld.py",
         "scratch/mathworld1_svpforder.py", "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py", "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    t0 = time.monotonic()
    P = [json.loads(l) for l in open(PRIMARY)]
    if SMOKE:
        P = P[:2]
    n = len(P)
    if not SMOKE:
        gate(n == 96, "N")
        gate(len({p["pair_key"] for p in P}) == 48, "48 keys")
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
    OUTDIR.mkdir(parents=True)
    base = baselines(P)
    (OUTDIR / "baselines.json").write_text(json.dumps(base, indent=1))
    # canonicalize candidates to semantic order; build continuations
    states = []
    for p in P:
        by = {ctup(c): c for c in p["candidates"]}
        conts = {"CANONICAL": [], "PARAM_FIRST": []}
        for s in SEM:
            c = by[s]
            cz = c["factor_code"]
            gate(factor_decode(cz) == s, "C RT")
            pz = pf_encode(s)
            gate(pf_decode(pz) == s and pz == [cz[PERM[i]] for i in range(8)],
                 "PF RT / PERM")
            conts["CANONICAL"].append([CODE_BASE + x for x in cz] + [TOK.eos_id])
            conts["PARAM_FIRST"].append([CODE_BASE + x for x in pz]
                                        + [TOK.eos_id])
        states.append({"cur": p["cur"], "cur_sha": p["cur_sha"],
                       "pair_id": p["pair_id"], "theta": p["theta"],
                       "gold": tuple(p["gold_tuple"]), "conts": conts})
    receipt = {"smoke": SMOKE, "prereg": PREREG, "prereg_commit": PREREG_COMMIT,
               "n_states": n, "pins": {p: fsha(p) for p in PINS},
               "checkpoints": [], "semantic_order": [list(s) for s in SEM],
               "noise_bound": NOISE, "baselines": base,
               "semantic_beyond_surface_identifiable": False,
               "reason": base["reason"]}

    def finish(verdict, extra):
        receipt["verdict"] = verdict
        receipt.update(extra)
        receipt["wall_s"] = round(time.monotonic() - t0, 1)
        receipt["artifact_sha256"] = {
            f.name: fsha(f) for f in sorted(OUTDIR.glob("*.json*"))
            if f.name != "prband2score_receipt.json"}
        receipt["start"] = START
        receipt["completion_commit"] = completion_commit()
        (OUTDIR / "prband2score_receipt.json").write_text(
            json.dumps(receipt, indent=1))
        print(json.dumps({k: v for k, v in receipt.items()
                          if k not in ("start", "pins", "baselines")},
                         indent=1), flush=True)

    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    receipt["device"] = "mps"
    receipt["torch"] = torch.__version__
    raw = open(OUTDIR / "raw_scores.jsonl", "w")
    cells = {}
    ck_list = CKPTS[:1] if SMOKE else CKPTS
    for seed, rep, path, sha_exp in ck_list:
        gate(fsha(path) == sha_exp, f"CKPT SHA {path}")
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(path, weights_only=True))
        gate(sum(q.numel() for q in m.parameters()) == 19142016, "PARAMS")
        m.eval()
        m = m.to(dev)
        receipt["checkpoints"].append({"seed": seed, "representation": rep,
                                       "path": path, "sha256": sha_exp,
                                       "params": 19142016,
                                       "blocks": len(m.blocks)})
        gate(len(m.blocks) == 8, "8 BLOCKS")
        for arm, mask in ARMS:
            rows_cell = {}
            for i, st in enumerate(states):
                lps = masked_token_lps(m, dev, st["cur"], st["conts"][rep], mask)
                rows_cell[i] = {}
                for s, lp, cont in zip(SEM, lps, st["conts"][rep]):
                    gate(len(lp) == 9, "T!=9")
                    tot = float(sum(lp))
                    rows_cell[i][s] = {"sum": tot}
                    raw.write(json.dumps({
                        "seed": seed, "representation": rep, "arm": arm,
                        "mask": mask, "state": i, "pair_id": st["pair_id"],
                        "theta": st["theta"], "cur_sha": st["cur_sha"],
                        "candidate": list(s), "name": NAMES[s],
                        "continuation": cont, "lps": lp, "sum": tot,
                        "gold": list(st["gold"])}) + "\n")
                raw.flush()
            met, per_state, pairs = cell_metrics(rows_cell, P, arm)
            cells[f"{seed}|{rep}|{arm}"] = {"metrics": met,
                                             "pairs": pairs,
                                             "per_state": [
                                                 {k: (list(v) if isinstance(v, tuple)
                                                      else v)
                                                  for k, v in s.items()
                                                  if k != "scores"}
                                                 for s in per_state]}
            print(f"[{seed} {rep} {arm}] top1 {met['top1']}/{n} ties "
                  f"{met['score_ties']} both {met['both_correct_pairs']} "
                  f"switch {met['switch_pairs_robust']} flips "
                  f"{met['flips']['robust_sign_flips']}", flush=True)
        del m
    raw.close()
    # ---- FULL v MASK0 tables + matrix ---------------------------------
    contrasts = {}
    matrix = {}
    for seed, rep, _p, _s in ck_list:
        F = cells[f"{seed}|{rep}|FULL"]
        M = cells[f"{seed}|{rep}|MASK0"]
        fc = [s["correct"] for s in F["per_state"]]
        mc = [s["correct"] for s in M["per_state"]]
        contrasts[f"{seed}|{rep}"] = {
            "both_correct": sum(a and b for a, b in zip(fc, mc)),
            "full_only": sum(a and not b for a, b in zip(fc, mc)),
            "mask0_only": sum(b and not a for a, b in zip(fc, mc)),
            "both_wrong": sum(not a and not b for a, b in zip(fc, mc)),
            "full_both_correct_pairs": F["metrics"]["both_correct_pairs"],
            "mask0_both_correct_pairs": M["metrics"]["both_correct_pairs"],
            "full_robust_flips": F["metrics"]["flips"]["robust_sign_flips"],
            "mask0_robust_flips": M["metrics"]["flips"]["robust_sign_flips"],
            "full_switches": F["metrics"]["switch_pairs_robust"],
            "mask0_switches": M["metrics"]["switch_pairs_robust"],
            "mask0_sanity_pass": M["metrics"]["mask0_sanity_pass"]}
        fm = F["metrics"]
        matrix[f"{seed}|{rep}"] = {
            "top1": fm["top1"], "A0_correct": fm["A0_correct"],
            "B0_correct": fm["B0_correct"],
            "both_correct_robust": fm["both_correct_robust"],
            "robust_flips": fm["flips"]["robust_sign_flips"],
            "correct_direction_flips": fm["flips"]["correct_direction"],
            "classification": fm["classification"],
            "mask0_sanity_pass": M["metrics"]["mask0_sanity_pass"]}
    obs = [k for k, v in matrix.items()
           if v["classification"].startswith("STATE-CONDITIONED")]
    reps = {"CANONICAL": all(matrix.get(f"{s}|CANONICAL", {}).get(
        "classification", "").startswith("STATE-CONDITIONED")
        for s in ("19001", "20001")),
        "PARAM_FIRST": all(matrix.get(f"{s}|PARAM_FIRST", {}).get(
            "classification", "").startswith("STATE-CONDITIONED")
            for s in ("19001", "20001"))}
    cross = {s: all(matrix.get(f"{s}|{r}", {}).get(
        "classification", "").startswith("STATE-CONDITIONED")
        for r in ("CANONICAL", "PARAM_FIRST")) for s in ("19001", "20001")}
    (OUTDIR / "cells.json").write_text(json.dumps(
        {"cells": cells, "contrasts": contrasts, "matrix": matrix}, indent=1))
    sanity_fail = [k for k, v in contrasts.items() if not v["mask0_sanity_pass"]]
    lead = (f"STATE-CONDITIONED RANKING OBSERVED IN {len(obs)}/{len(matrix)} "
            "FULL CHECKPOINTS" if obs else
            "NO FULL CHECKPOINT REFUTES THE STATE-BLIND FIXED-RANKING CLASS")
    if sanity_fail:
        lead = "INSTRUMENT FAILURE (MASK0 sanity) " + str(sanity_fail)
    finish("SMOKE OK" if SMOKE else lead,
           {"matrix": matrix, "contrasts": contrasts,
            "within_representation_replication": reps,
            "cross_representation_support": cross,
            "mask0_sanity_failures": sanity_fail,
            "cell_summary": {k: {kk: v["metrics"][kk] for kk in (
                "top1", "score_ties", "A0_correct", "B0_correct",
                "both_correct_pairs", "both_correct_robust",
                "both_correct_subnoise", "exactly_one_pairs",
                "neither_pairs", "switch_pairs_robust",
                "switch_pairs_float_noise", "pred_census",
                "pred_census_by_theta", "flips", "mask0_spread",
                "margin_median")} for k, v in cells.items()}})
    return 0


if __name__ == "__main__":
    sys.exit(main())

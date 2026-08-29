"""MATH-CYBER-1 SVP-GRID-CH-F3-NUISANCE-DESK-0 — post-hoc
mechanism diagnostic of the booked SUPPORT-NOT-LEARNED verdict:
is the CH-F3 covered-cell collapse associated with parameter
EXTRAPOLATION (P degree / trig frequency beyond the trained
augmentation envelope), or is covered competence low even
INSIDE the envelope? Descriptive/mechanistic only; can never
rescue CALIBRATION-SCORE-15001 or grant heldout authority. The
sealed test artifact is neither named nor read anywhere here.

2x2 CROSSING (CH-F3, covered-I0 ONLY — after-variant states,
labels restricted to i_unprod I0/t2 and I0/t3; no I1/t2-t3
label is ever constructed or scored):
  A = P IN  x c IN     B = P IN  x c OUT
  C = P OUT x c IN     D = P OUT x c OUT
  P IN  = 6 exactly-TRAINED augmentation P forms
          (x^7, x^8, x^9, x^6+x^3, x^7+x^4, x^8+x^5).
  P OUT = 6 degree-10 forms of the failed calibration class
          (x^10, x^10+x^2, x^10+x^4, x^10+x^6, x^10+x^8,
          x^10+x^9).
  c IN  = {8, 10, 12, 15, 17, 19} (trained support values).
  c OUT = {20, 21, 22, 23, 24, 25} (failed regime values).
DISCLOSED CONSTRAINT: every two-term monomial-sum P of degree
<= 9 is already consumed across the burned banks, so P IN
REUSES trained forms verbatim (the sharpest possible IN
condition) and P OUT reuses the eval bank's degree-10 forms
(the exact failed class); parent NOVELTY comes from a fresh
inert-addend axis shared by all four cells: k in {1,2} over
fresh polys (4*x**2, 6*x**5) — every parent is exact-cur-novel
against every burned horizon (gated refute-on-hit). Matched
nuisance pairs: each (P, T, w, k) template appears under all 12
c values, giving within-template IN/OUT frequency contrasts.

HORIZON (frozen): per cell 6P x 6c x 2T x 2w x 2k = 288 bases;
1,152 total; after-variant (sin(sin(x))) qualification only via
the frozen qualify_parent; a base enters scoring iff its unique
teacher label is (i_unprod, I, ordinal 0, term_index, 2|3);
all other outcomes are counted, never scored. POPULATION =
every qualified I0/t2 and I0/t3 row (full census, no
subselection — selection law frozen as take-all).

SCORING: the two sealed seed-15001 checkpoints (pinned), frozen
svpadj law (standing prompt, factor_code/hash_code + EOS
continuations, pessimistic top-1, T=9 gates), blind over all
scored states in one pass, is_label consumed only after scores
exist. Candidate 0 is PRESERVED (never removed); legal K
(n_candidates) and competitive K (candidate-0 top-rank census)
reported per cell.

FROZEN RESOLUTION (thresholds fixed before counting; primary
statistic = POOLED-ARMS accuracy per cell over t2+t3 =
(F_correct + H_correct) / (2 * n); per-arm persisted but small
F-H differences are never converted into a representation
claim):
  RANGE-SENSITIVE   iff pooled A >= 75% AND
                    min(pooled B, C, D) <= pooled A - 20 pts.
  SUPPORT-STILL-WEAK iff pooled A < 75%.
  MIXED             otherwise.

Outputs under logs/mathworld1/svpnuisdesk/ (refuse-if-exists):
attempts.jsonl (all 1,152 qualification outcomes),
scores.jsonl, svpnuisdesk_receipt.json.
SMOKE (SVPNUIS_SMOKE=1): first 2 bases per cell, path-isolated
under logs/mathworld1/svpnuisdesk_smoke/, novelty report-only.

    SVPNUIS_SMOKE=1 .venv/bin/python scratch/mathworld1_svpnuisdesk.py
    .venv/bin/python scratch/mathworld1_svpnuisdesk.py        (Mac)
"""
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
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpadj import (rank_metrics,  # noqa: E402
                                       score_decision)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpchal import (AFTER_D,  # noqa: E402
                                        SMALL_D, X, fsha,
                                        qualify_parent)
from scratch.mathworld1_svpchal import \
    build_horizon as d0_horizon  # noqa: E402
from scratch.mathworld1_svpchal2 import \
    build_horizon1 as d1_horizon  # noqa: E402
from scratch.mathworld1_svpdiet import (BURNED_F3_POLYS,  # noqa: E402
                                        F3_EVAL_C,
                                        F3_EVAL_P,
                                        F3_EVAL_POLYS,
                                        F3_TRAIN_C, F3_TRAIN_P,
                                        F3_TRAIN_POLYS,
                                        F4_EVAL_FREQS,
                                        F4_EVAL_P1,
                                        F4_TRAIN_FREQS,
                                        F4_TRAIN_P1,
                                        PILOT_RECEIPTS,
                                        f3_bases, f4_bases)
from scratch.mathworld1_svpdiet2 import (E1_F3_C,  # noqa: E402
                                         E1_F3_P, E1_F3_POLYS,
                                         E1_F4_FREQS,
                                         E1_F4_P1)

SMOKE = os.environ.get("SVPNUIS_SMOKE") == "1"
OUTDIR = Path("logs/mathworld1/svpnuisdesk_smoke" if SMOKE
              else "logs/mathworld1/svpnuisdesk")
SMOKE_RECEIPT = Path(
    "logs/mathworld1/svpnuisdesk_smoke/svpnuisdesk_receipt.json")
PAIRED = Path("data/matsub_paired.jsonl")
CKPTS = {
    "FACTOR": ("checkpoints/svp_grid_factor_s15001.pt",
               "6ef7b85ce9215c1ba299d64fdf61e2b32573b2aaac1e0e"
               "a4bb1dedf4cf790cd3"),
    "HASH": ("checkpoints/svp_grid_hash_s15001.pt",
             "3b9da5a7ee3bd72bc40a21afcbc8b114ebd19040079ce1f7"
             "194941946052501a"),
}
INIT_SHA = ("4b085795f9e8b0be874cabdc6d58899a2a4554f8b42cb711"
            "f5154614e41797bc")
PAIRED_SHA = ("a943ba7fc581db743b07192e5d951fadd"
              "dd2ba19bca3225b75d8402351d468e8")
VOCAB = 340
CODE_BASE = 332
ARMS = ["FACTOR", "HASH"]
TOK = ActionGCTok()

P_IN = (X**7, X**8, X**9, X**6 + X**3, X**7 + X**4,
        X**8 + X**5)
P_OUT = (X**10, X**10 + X**2, X**10 + X**4, X**10 + X**6,
         X**10 + X**8, X**10 + X**9)
C_IN = (8, 10, 12, 15, 17, 19)
C_OUT = (20, 21, 22, 23, 24, 25)
K_POLYS = (4 * X**2, 6 * X**5)
WS = (sp.exp(X) / X, sp.sin(X) / X)
BAR_A = 0.75
BAR_DROP = 0.20


def build_cell(cell, Ps, Cs):
    out = []
    for P in Ps:
        for T in (sp.sin, sp.cos):
            for c in Cs:
                for w in WS:
                    for k in (1, 2):
                        f = (sp.expand(sp.diff(P * T(c * X), X))
                             + sp.Integral(w, X)
                             + sp.Add(*K_POLYS[:k]))
                        sig = (f"NUIS|{cell}|P={P}|"
                               f"T={T.__name__}|c={c}|w={w}|"
                               f"k={k}")
                        out.append((cell, sig, f, sp.sstr(P),
                                    T.__name__, c,
                                    sp.sstr(w), k))
    return out


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    gate(fsha(PAIRED) == PAIRED_SHA, "PAIRED PIN")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {a}")
    gate(fsha("checkpoints/svp_grid_init_s15001.pt")
         == INIT_SHA, "INIT PIN")
    if not SMOKE:
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(sr.get("smoke") is True and "verdict" in sr,
             "SMOKE NOT GREEN")
        for pth, h in sr["start"]["file_sha256"].items():
            gate(fsha(pth) == h, f"SMOKE STALE {pth}")
    START = start_provenance(
        ["scratch/mathworld1_svpnuisdesk.py",
         "scratch/mathworld1_svpdiet.py",
         "scratch/mathworld1_svpdiet2.py",
         "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpchal2.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "llmopt/search/derivation.py",
         "llmopt/search/rules.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    # P/c class sanity v the frozen bank literals
    for P in P_IN:
        gate(P in set(F3_TRAIN_P), f"P_IN NOT TRAINED {P}")
    for P in P_OUT:
        gate(P in set(E1_F3_P), f"P_OUT NOT EVAL-CLASS {P}")
    for c in C_IN:
        gate(c in set(F3_TRAIN_C), f"C_IN NOT TRAINED {c}")
    for c in C_OUT:
        gate(c in set(E1_F3_C), f"C_OUT NOT EVAL-CLASS {c}")
    for pv in K_POLYS:
        gate(sp.sstr(pv) not in set(BURNED_F3_POLYS) | {
            sp.sstr(e) for e in
            F3_TRAIN_POLYS + F3_EVAL_POLYS + E1_F3_POLYS},
            f"K POLY USED {pv}")

    # burned cur set (reconstructed from frozen banks; no
    # svpdiet2 output file is read)
    train_cur = set()
    for l in open(PAIRED):
        train_cur.add(json.loads(l)["cur"])
    band_cur = set()
    for bf in ("logs/mathworld1/svpeval/decisions.jsonl",
               "logs/mathworld1/svpeval2/decisions.jsonl",
               "logs/mathworld1/svpeval3/decisions.jsonl"):
        for l in open(bf):
            r = json.loads(l)
            if r.get("cur"):
                band_cur.add(r["cur"])
    pilot_cur = set()
    for pr in PILOT_RECEIPTS:
        for a2 in json.loads(Path(pr).read_text())["attempts"]:
            pilot_cur.add(a2["parent_sstr"])
    gate(len(pilot_cur) == 566, "PILOT")
    e1_hz = (f3_bases("E1", E1_F3_P, E1_F3_C, E1_F3_POLYS)
             + f4_bases("E1", E1_F4_P1, E1_F4_FREQS))
    diet_hz = (f3_bases("TRAIN", F3_TRAIN_P, F3_TRAIN_C,
                        F3_TRAIN_POLYS)
               + f4_bases("TRAIN", F4_TRAIN_P1, F4_TRAIN_FREQS)
               + f3_bases("EVAL", F3_EVAL_P, F3_EVAL_C,
                          F3_EVAL_POLYS)
               + f4_bases("EVAL", F4_EVAL_P1, F4_EVAL_FREQS))
    dd_cur = set()
    for hz, n in ((d0_horizon(), 720), (d1_horizon(), 2160),
                  (diet_hz, 15912), (e1_hz, 10224)):
        before = len(dd_cur)
        for tup in hz:
            fam, sig, f = tup[0], tup[1], tup[2]
            for D in (SMALL_D[0], SMALL_D[1], AFTER_D):
                dd_cur.add(sp.sstr(sp.Add(sp.Integral(f, X),
                                          sp.Integral(D, X))))
        gate(len(dd_cur) - before == n,
             f"BURNED HORIZON {len(dd_cur) - before}")
    burned = train_cur | band_cur | pilot_cur | dd_cur

    horizon = (build_cell("A", P_IN, C_IN)
               + build_cell("B", P_IN, C_OUT)
               + build_cell("C", P_OUT, C_IN)
               + build_cell("D", P_OUT, C_OUT))
    if SMOKE:
        # per-cell picks covering both T, both w, both k
        picks = (0, 2, 121, 215)
        horizon = [h for cell in ("A", "B", "C", "D")
                   for i2, h in enumerate(
                       [x for x in horizon if x[0] == cell])
                   if i2 in picks]
    else:
        gate(len(horizon) == 1152, f"HORIZON {len(horizon)}")
    sigs = [h[1] for h in horizon]
    gate(len(sigs) == len(set(sigs)), "SIG DUP")

    OUTDIR.mkdir(parents=True)
    receipt = {"smoke": SMOKE, "prereg":
               "MATH-CYBER-1-SVP-GRID-CH-F3-NUISANCE-DESK-0",
               "n_horizon": len(horizon),
               "bars": {"A_competence": BAR_A,
                        "drop": BAR_DROP},
               "smoke_receipt_sha": (fsha(SMOKE_RECEIPT)
                                     if not SMOKE else None)}

    # phase 1: qualify (after-variant only)
    t_all = time.monotonic()
    scored_pop = []
    census = Counter()
    nov_hits = 0
    af = open(OUTDIR / "attempts.jsonl", "w")
    for i, (cell, sig, f_t, Ps, Tn, c, wn, k) in enumerate(
            horizon):
        row, why = qualify_parent(f_t, AFTER_D)
        if row.get("cur") in burned:
            nov_hits += 1
        rec = {"cell": cell, "base_signature": sig,
               "P": Ps, "T": Tn, "c": c, "w": wn, "k": k,
               "fail": why}
        if why is None:
            tup = (row["chosen_rule"], row["chosen_site_kind"],
                   row["chosen_ordinal"],
                   row["chosen_param_kind"], row["chosen_term"])
            rec["teacher"] = list(tup)
            gate(not (tup[0] == "i_unprod" and tup[1] == "I"
                      and tup[2] == 1 and tup[4] in (2, 3)),
                 "I1 HELD-OUT LABEL CONSTRUCTED")
            if (tup[0] == "i_unprod" and tup[1] == "I"
                    and tup[2] == 0 and tup[4] in (2, 3)):
                census[f"{cell}|scored_t{tup[4]}"] += 1
                scored_pop.append((cell, sig, Ps, Tn, c, wn,
                                   k, tup[4], row))
            else:
                census[f"{cell}|out_of_scope"] += 1
        else:
            census[f"{cell}|fail"] += 1
        af.write(json.dumps(rec) + "\n")
        if (i + 1) % 100 == 0 or SMOKE:
            print(f"[qual {i + 1}/{len(horizon)}]", flush=True)
    af.close()
    if SMOKE:
        receipt["smoke_novelty_hits"] = nov_hits
        gate(len(scored_pop) >= 1, "SMOKE ZERO SCORED")
    else:
        gate(nov_hits == 0, f"BURNED CUR {nov_hits}")
    curs = [r[8]["cur"] for r in scored_pop]
    gate(len(curs) == len(set(curs)), "DUP CUR")
    receipt["qualification_census"] = dict(census)
    receipt["n_scored_states"] = len(scored_pop)

    # phase 2: load sealed checkpoints, blind score
    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    arms = {}
    for a in ARMS:
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(CKPTS[a][0],
                                     weights_only=True))
        gate(sum(p.numel() for p in m.parameters())
             == 19142016, f"PARAMS {a}")
        m.eval()
        arms[a] = m.to(dev)
    t_sc = time.monotonic()
    recs = []
    sf = open(OUTDIR / "scores.jsonl", "w")
    for (cell, sig, Ps, Tn, c, wn, k, term, row) in scored_pop:
        cands = row["candidates"]
        conts = {
            "FACTOR": [[CODE_BASE + s for s in
                        cd["factor_code"]] + [TOK.eos_id]
                       for cd in cands],
            "HASH": [[CODE_BASE + s for s in cd["hash_code"]]
                     + [TOK.eos_id] for cd in cands]}
        rec = {"cell": cell, "base_signature": sig,
               "P": Ps, "T": Tn, "c": c, "w": wn, "k": k,
               "term": term, "cur": row["cur"],
               "n_candidates": row["n_candidates"]}
        raw = {}
        for a in ARMS:
            raw[a] = score_decision(arms[a], dev, row["cur"],
                                    conts[a])
        li = [i2 for i2, cd in enumerate(cands)
              if cd["is_label"]][0]
        rec["label_index"] = li
        for a in ARMS:
            means = [t[0] for t in raw[a]]
            top1, rank = rank_metrics(means, li)
            rec[a] = {"mean_lp": means,
                      "sum_lp": [t[1] for t in raw[a]],
                      "T9": [t[2] for t in raw[a]],
                      "top1": top1, "rank": rank}
            gate(all(t == 9 for t in rec[a]["T9"]), "T!=9")
            top = max(means)
            rec[a]["cand0_strict_top"] = (
                means[0] > max(means[1:]) if len(means) > 1
                else True)
            rec[a]["argmax_first_of_ties"] = means.index(top)
        recs.append(rec)
        sf.write(json.dumps(rec) + "\n")
    sf.close()
    raw_sha = fsha(OUTDIR / "scores.jsonl")

    # phase 3: frozen resolution (pooled-arms per cell)
    def acc(sel, arm=None):
        if not sel:
            return None
        if arm:
            return round(sum(1 for d in sel
                             if d[arm]["top1"]) / len(sel), 4)
        return round(sum(1 for d in sel for a in ARMS
                         if d[a]["top1"]) / (2 * len(sel)), 4)
    cells = {}
    for cell in ("A", "B", "C", "D"):
        sel = [d for d in recs if d["cell"] == cell]
        cells[cell] = {
            "n": len(sel),
            "pooled": acc(sel),
            "FACTOR": acc(sel, "FACTOR"),
            "HASH": acc(sel, "HASH"),
            "by_term": {f"t{t}": {
                "n": sum(1 for d in sel if d["term"] == t),
                "pooled": acc([d for d in sel
                               if d["term"] == t]),
                "FACTOR": acc([d for d in sel
                               if d["term"] == t], "FACTOR"),
                "HASH": acc([d for d in sel
                             if d["term"] == t], "HASH")}
                for t in (2, 3)},
            "legal_K": dict(Counter(
                d["n_candidates"] for d in sel)),
            "cand0_strict_top_ranked": {a: sum(
                1 for d in sel if d[a]["cand0_strict_top"])
                for a in ARMS},
            "cand0_is_label": sum(
                1 for d in sel if d["label_index"] == 0)}
    pooled = {c: cells[c]["pooled"] for c in cells}
    MIN_N = 40
    if SMOKE:
        verdict = "SMOKE"
    elif any(v is None for v in pooled.values()) or any(
            cells[c]["n"] < MIN_N for c in cells):
        verdict = "INSUFFICIENT-CELLS"
    elif pooled["A"] >= BAR_A and min(
            pooled["B"], pooled["C"], pooled["D"]) \
            <= pooled["A"] - BAR_DROP:
        verdict = "RANGE-SENSITIVE"
    elif pooled["A"] < BAR_A:
        verdict = "SUPPORT-STILL-WEAK"
    else:
        verdict = "MIXED"
    # matched within-template frequency contrast (descriptive)
    tmpl = defaultdict(lambda: {"IN": [0, 0], "OUT": [0, 0]})
    pclass = {sp.sstr(P): "P_IN" for P in P_IN}
    pclass.update({sp.sstr(P): "P_OUT" for P in P_OUT})
    for d in recs:
        key = (pclass[d["P"]], d["P"], d["T"], d["w"], d["k"])
        side = "IN" if d["c"] in C_IN else "OUT"
        for a in ARMS:
            tmpl[key][side][0] += int(d[a]["top1"])
            tmpl[key][side][1] += 1
    matched = {}
    for pc in ("P_IN", "P_OUT"):
        both = [v for k2, v in tmpl.items()
                if k2[0] == pc and v["IN"][1] and v["OUT"][1]]
        matched[pc] = {
            "n_templates_both_sides": len(both),
            "mean_in_acc": round(sum(
                v["IN"][0] / v["IN"][1] for v in both)
                / len(both), 4) if both else None,
            "mean_out_acc": round(sum(
                v["OUT"][0] / v["OUT"][1] for v in both)
                / len(both), 4) if both else None}
    receipt.update({
        "verdict": verdict,
        "cells": cells,
        "pooled": pooled,
        "matched_template_freq_contrast": matched,
        "raw_scores_sha": raw_sha,
        "attempts_sha": fsha(OUTDIR / "attempts.jsonl"),
        "wall_qual_s": round(t_sc - t_all, 1),
        "wall_score_s": round(time.monotonic() - t_sc, 1),
        "pins": {PAIRED.as_posix(): fsha(PAIRED),
                 **{p: fsha(p) for a, (p, h)
                    in CKPTS.items()}},
        "start": START,
        "completion_commit": completion_commit()})
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT PIN {a}")
    gate(fsha("checkpoints/svp_grid_init_s15001.pt")
         == INIT_SHA, "POST INIT PIN")
    gate(fsha(PAIRED) == PAIRED_SHA, "POST PAIRED PIN")
    (OUTDIR / "svpnuisdesk_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "pooled", "cells")
                      if k in receipt}, indent=1)[:4000],
          flush=True)
    print("[svpnuisdesk] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

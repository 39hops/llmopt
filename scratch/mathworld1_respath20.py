"""MATH-CYBER-1 RESIDUAL-PATH-ROUTING-REPLICATION-0 — the
seed-20001 replication of the exhaustive 256-mask census
(frozen prereg 5d69d303d8a355f9719b0bb4242ed686b2d9fc09,
RESULTS L61903). Adopt-not-fork copy of the results-cited
scratch/mathworld1_respath.py: the ONLY deltas are the five
frozen seed-20001 reference pins, the four booked baseline
totals, the fresh namespace, and the ADDED calibration
score-drift rider registered in the prereg.

Two frozen seed-20001 checkpoints (CANONICAL / PARAM_FIRST,
same init) x two sealed 96-state populations (strict heldout
site_role=="heldout-I1"; covered calibration) x all 256
whole-block masks. Bypass = skip the ModuleList loop
iteration (x_out = x_in exact); final RMSNorm + head always
active; eval mode, no grad, attn_mask=None (is_causal path),
teacher-forced T=9, no cache. Scoring verbatim per
population: heldout total-sum k=9 / calibration mean-lp with
per-mask MEAN!=SUM-RANK + ORDER-MISMATCH disclosure flags;
pessimistic strict top-1. PARAM_FIRST payload re-derived via
pf_encode with the three frozen roundtrip/PERM gates.

Baseline coherence per cell BEFORE the cube: mask 255 top-1
v the frozen raw reference, disagreement count <= 1/96 else
STOP (BASELINE-COHERENCE-FAILURE); max-score-deviation drift
rider. Mask 0 must execute with finite scores (non-finite
ABORTS). Cell order: (heldout, C), (heldout, PF),
(calibration, PF), (calibration, C); mask ids 0..255 (bit i
= block i, LSB = b0). RAW FIRST: per mask x state rows with
candidate sums streamed and hashed BEFORE any envelope.

Outputs under logs/mathworld1/respath/ (refuse-if-exists):
raw_census.jsonl, respath_receipt.json, riders.json.

    .venv/bin/python scratch/mathworld1_respath.py           (Mac)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import factor_decode  # noqa: E402
from scratch.mathworld1_svpforder import (PERM,  # noqa: E402
                                          pf_decode, pf_encode)

PREREG_COMMIT = "5d69d303d8a355f9719b0bb4242ed686b2d9fc09"
CKPTS = {
    "CANONICAL": ("checkpoints/svp_forder_canonical_s20001.pt",
                  "0a841a5f2a43b6f64b0dac8259c26fd79961e6ab91"
                  "359a54be9c2582815b3e34"),
    "PARAM_FIRST": ("checkpoints/svp_forder_paramfirst_s20001"
                    ".pt",
                    "b7198ff2e7b903ab5ed075fe947cb29142c5790e"
                    "c84831434c53a598e466c322"),
}
INIT_CK = "checkpoints/svp_forder_init_s20001.pt"
INIT_SHA = ("7c95e77f8d7ccea5f4dd71c989e4d3225e347a178032d435"
            "39a1ae6ef62c9452")
BIRTH_RECEIPT = "logs/mathworld1/svpforepl_s20001_receipt.json"
BIRTH_SHA = ("70ff3248a9b2f6e584c2d7bb8e9fc7b59853ebb7055f5819"
             "3783fbd61eab78b3")
POPS = {
    "heldout": ("logs/mathworld1/svpdiet3/heldout_test16.jsonl",
                "a3f6103b3733d909281849dcb3fd6ba9fba3891f2014"
                "bec13881b4509df46ddb"),
    "calibration": ("logs/mathworld1/svpdiet3/"
                    "covered_calibration.jsonl",
                    "af1a4aa1df7bf3224745e91a90e1a77c36e5c54f"
                    "7ff9b08509794d0fb7978db3"),
}
RAW_REFS = {
    # full shas DERIVED from the authoritative receipts
    # (svpfoheld_receipt raw_scores_sha / svpfocal_receipt
    # raw_scores_sha) — never reconstructed from a prefix
    "heldout": ("logs/mathworld1/svpfoheld20/raw_token_scores"
                ".jsonl",
                "18f27b666ce54c5402abff746e696f9b8c35c94235d0"
                "96a060f97cf015aae17f"),
    "calibration": ("logs/mathworld1/svpfocal20/scores.jsonl",
                    "5df6ee83ff684920395d634a3f6f8ba9c6b7c96f"
                    "5805540c8b8cefc25f968ddf"),
}
BOOKED_FULL = {("heldout", "CANONICAL"): 80,
               ("heldout", "PARAM_FIRST"): 87,
               ("calibration", "CANONICAL"): 85,
               ("calibration", "PARAM_FIRST"): 91}
CELL_ORDER = [("heldout", "CANONICAL"),
              ("heldout", "PARAM_FIRST"),
              ("calibration", "PARAM_FIRST"),
              ("calibration", "CANONICAL")]
VOCAB = 340
CODE_BASE = 332
OUTDIR = Path("logs/mathworld1/respath20")
TOK = ActionGCTok()
NEAR_FULL_TOL = 2


def fsha(p) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def load_pop(pop):
    rows = [json.loads(l) for l in open(POPS[pop][0])]
    if pop == "heldout":
        rows = [r for r in rows
                if r["site_role"] == "heldout-I1"]
    gate(len(rows) == 96, f"POP SIZE {pop}")
    states = []
    for r in rows:
        cands = r["candidates"]
        gate(len(cands) == r["n_candidates"], "CAND COUNT")
        li = [i for i, c in enumerate(cands)
              if c["is_label"]]
        gate(len(li) == 1, "LABEL COUNT")
        conts = {"CANONICAL": [], "PARAM_FIRST": []}
        for c in cands:
            t = ctup(c)
            cz = c["factor_code"]
            gate(factor_decode(cz) == t, "C RT")
            pz = pf_encode(t)
            gate(pf_decode(pz) == t, "PF RT")
            gate(pz == [cz[PERM[i]] for i in range(8)],
                 "PERM IDENTITY")
            conts["CANONICAL"].append(
                [CODE_BASE + s for s in cz] + [TOK.eos_id])
            conts["PARAM_FIRST"].append(
                [CODE_BASE + s for s in pz] + [TOK.eos_id])
        states.append({"block_id": r["block_id"],
                       "cur": r["cur"], "label": li[0],
                       "conts": conts})
    return states


def masked_token_lps(model, dev, cur, conts, mask_id):
    """token_lps law with a whole-block mask: skipped blocks
    are exact identity. eval mode, no grad, attn_mask=None
    (is_causal path), one teacher-forced forward."""
    assert not model.training
    pre = TOK.encode(f"Current: {cur}\nHints: none\nStep: ")
    enc = [pre + c for c in conts]
    L = max(len(e) for e in enc)
    ids = torch.full((len(enc), L), TOK.pad_id)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e)
    ids = ids.to(dev)
    with torch.no_grad():
        x = model.emb(ids)
        for i, b in enumerate(model.blocks):
            if (mask_id >> i) & 1:
                x, _ = b(x, None, None)
        logits = model.head(model.norm(x))
        logp = torch.log_softmax(logits[:, :-1].float(), -1)
        tok_lp = logp.gather(
            -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu()
    out = []
    for i, c in enumerate(conts):
        s = tok_lp[i, len(pre) - 1:len(pre) - 1 + len(c)]
        gate(bool(torch.isfinite(s).all()),
             f"NON-FINITE LP mask {mask_id}")
        gate(len(c) == 9, "T!=9")
        out.append([float(v) for v in s])
    return out


def heldout_ref_top1(arm):
    """Re-derive per-state top-1 from the frozen heldout raw
    (per-token lps) via the assay's cums + gold_top law."""
    ref = {}
    lps_by = {}
    for l in open(RAW_REFS["heldout"][0]):
        r = json.loads(l)
        if r["arm"] != arm:
            continue
        cum = []
        for v in r["token_lps"]:
            s, c = 0.0, []
            for x in v:
                s += x
                c.append(s)
            cum.append(c)
        li = r["label_index"]
        g = cum[li][8]
        ref[r["block_id"]] = all(
            cum[j][8] < g for j in range(len(cum)) if j != li)
        lps_by[r["block_id"]] = r["token_lps"]
    gate(len(ref) == 96, "HELDOUT REF ROWS")
    return ref, lps_by


def calibration_ref_top1(arm):
    """Also returns per-state reference sum_lp lists for the
    prereg-added calibration score-drift rider."""
    ref, sums = {}, {}
    for l in open(RAW_REFS["calibration"][0]):
        r = json.loads(l)
        ref[r["block_id"]] = bool(r[arm]["top1"])
        sums[r["block_id"]] = r[arm]["sum_lp"]
    gate(len(ref) == 96, "CAL REF ROWS")
    return ref, sums


def score_state(lps, pop, label):
    """Per-state scores + top1 + rr + margin under the
    population's own frozen law."""
    sums = [sum(v) for v in lps]
    if pop == "heldout":
        scores = sums
    else:
        scores = [s / 9.0 for s in sums]
    g = scores[label]
    rivals = [scores[j] for j in range(len(scores))
              if j != label]
    gate(len(rivals) >= 1, "SINGLETON CANDIDATE LIST")
    top1 = all(v < g for v in rivals)
    rank = 1 + sum(1 for v in rivals if v >= g)
    margin = g - max(rivals)
    # choice for agreement: argmax, tie -> lowest index
    best = max(scores)
    choice = min(i for i, v in enumerate(scores) if v == best)
    flags = {}
    if pop == "calibration":
        mo = sorted(range(len(scores)),
                    key=lambda i: (-scores[i], i))
        so = sorted(range(len(sums)),
                    key=lambda i: (-sums[i], i))
        flags["mean_sum_order_mismatch"] = mo != so
        rank_s = 1 + sum(1 for j in range(len(sums))
                         if j != label and sums[j] >= sums[label])
        flags["mean_sum_rank_mismatch"] = rank_s != rank
    return {"sums": sums,
            "top1": top1, "rr": round(1.0 / rank, 6),
            "margin": round(margin, 6), "choice": choice,
            "flags": flags}


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "INIT PIN")
    for k, (p, h) in POPS.items():
        gate(fsha(p) == h, f"POP PIN {k}")
    for k, (p, h) in RAW_REFS.items():
        gate(fsha(p) == h, f"RAW REF PIN {k}")
    gate(fsha(BIRTH_RECEIPT) == BIRTH_SHA, "BIRTH RECEIPT PIN")
    birth = json.loads(open(BIRTH_RECEIPT).read())
    gate(birth.get("seed") == 20001
         or "20001" in json.dumps(birth)[:2000],
         "BIRTH SEED")
    START = start_provenance(
        ["scratch/mathworld1_respath20.py",
         "scratch/mathworld1_respath.py",
         "scratch/mathworld1_svpfoheld.py",
         "scratch/mathworld1_svpfocal.py",
         "scratch/mathworld1_svpforder.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_actiontok.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    pops = {p: load_pop(p) for p in POPS}
    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    models = {}
    for a, (p, _) in CKPTS.items():
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(p, weights_only=True))
        gate(sum(q.numel() for q in m.parameters())
             == 19142016, f"PARAM COUNT {a}")
        m.eval()
        models[a] = m.to(dev)

    OUTDIR.mkdir(parents=True)
    t_run = time.time()
    coherence = {}
    cell_results = {}   # (pop, arm) -> mask_id -> summary
    with open(OUTDIR / "raw_census.jsonl", "w") as sink:
        for pop, arm in CELL_ORDER:
            model = models[arm]
            states = pops[pop]
            # --- baseline coherence: mask 255 first ---
            full_rows = []
            t_full = time.monotonic()
            for st in states:
                lps = masked_token_lps(model, dev, st["cur"],
                                       st["conts"][arm], 255)
                full_rows.append(
                    (st, lps,
                     score_state(lps, pop, st["label"])))
            full_wall = round(time.monotonic() - t_full, 3)
            ref_sums = None
            if pop == "heldout":
                ref, ref_lps = heldout_ref_top1(arm)
            else:
                ref, ref_sums = calibration_ref_top1(arm)
                ref_lps = None
            mism = sum(
                1 for st, _, sc in full_rows
                if sc["top1"] != ref[st["block_id"]])
            drift = None
            if ref_lps is not None:
                dmax = 0.0
                for st, lps, _ in full_rows:
                    for v_new, v_ref in zip(
                            lps, ref_lps[st["block_id"]]):
                        for a2, b2 in zip(v_new, v_ref):
                            dmax = max(dmax, abs(a2 - b2))
                drift = round(dmax, 8)
            elif ref_sums is not None:
                # prereg-added calibration score-drift rider:
                # max |mask-255 candidate sum - reference sum_lp|
                dmax = 0.0
                for st, _, sc in full_rows:
                    for a2, b2 in zip(sc["sums"],
                                      ref_sums[st["block_id"]]):
                        dmax = max(dmax, abs(a2 - b2))
                drift = round(dmax, 8)
            full_total = sum(
                1 for _, _, sc in full_rows if sc["top1"])
            coherence[f"{pop}:{arm}"] = {
                "mismatch_count": mism,
                "full_mask_top1": full_total,
                "booked_reference": BOOKED_FULL[(pop, arm)],
                "max_score_drift": drift}
            print(f"[respath] {pop}/{arm} coherence: "
                  f"mismatch {mism}/96, full {full_total}, "
                  f"drift {drift}", flush=True)
            gate(mism <= 1,
                 f"BASELINE-COHERENCE-FAILURE {pop}/{arm}")
            full_choice = {st["block_id"]: sc["choice"]
                           for st, _, sc in full_rows}
            # --- the cube, mask ids 0..255 ---
            per_mask = {}
            for mid in range(256):
                t0 = time.monotonic()
                if mid == 255:
                    rows = full_rows  # computed once in the
                    # coherence pass; wall carried from there
                else:
                    rows = []
                    for st in states:
                        lps = masked_token_lps(
                            model, dev, st["cur"],
                            st["conts"][arm], mid)
                        rows.append(
                            (st, lps,
                             score_state(lps, pop,
                                         st["label"])))
                k = bin(mid).count("1")
                top1 = sum(1 for _, _, sc in rows
                           if sc["top1"])
                mrr = sum(sc["rr"] for _, _, sc in rows) / 96
                agree = sum(
                    1 for st, _, sc in rows
                    if sc["choice"]
                    == full_choice[st["block_id"]])
                marg = sum(sc["margin"]
                           for _, _, sc in rows) / 96
                flags = {}
                if pop == "calibration":
                    flags = {
                        "order_mismatch_states": sum(
                            1 for _, _, sc in rows
                            if sc["flags"]
                            ["mean_sum_order_mismatch"]),
                        "rank_mismatch_states": sum(
                            1 for _, _, sc in rows
                            if sc["flags"]
                            ["mean_sum_rank_mismatch"])}
                wall = (full_wall if mid == 255 else
                        round(time.monotonic() - t0, 3))
                summary = {"mask": mid, "k": k, "top1": top1,
                           "mrr": round(mrr, 4),
                           "agree_full": agree,
                           "mean_margin": round(marg, 4),
                           "wall_s": wall, **flags}
                per_mask[mid] = {
                    **summary,
                    "per_state_top1": [sc["top1"]
                                       for _, _, sc in rows],
                    "per_state_choice": [sc["choice"]
                                         for _, _, sc in rows]}
                sink.write(json.dumps({
                    "pop": pop, "arm": arm, **summary,
                    "states": [
                        {"block_id": st["block_id"],
                         "label": st["label"],
                         "sums": sc["sums"],
                         "top1": sc["top1"],
                         "rr": sc["rr"],
                         "margin": sc["margin"],
                         "choice": sc["choice"],
                         "agree_full": sc["choice"]
                         == full_choice[st["block_id"]]}
                        for st, _, sc in rows]}) + "\n")
                if mid % 64 == 0:
                    sink.flush()
                    print(f"[respath] {pop}/{arm} mask {mid} "
                          f"k={k} top1={top1}", flush=True)
            sink.flush()
            cell_results[(pop, arm)] = per_mask
    raw_sha = fsha(OUTDIR / "raw_census.jsonl")
    print(f"[respath] RAW SEALED sha256={raw_sha}", flush=True)

    # ---- envelopes + metrics (post-hash) ----
    def cell_summary(pop, arm):
        pm = cell_results[(pop, arm)]
        full = pm[255]["top1"]
        by_k = {}
        for mid, s in pm.items():
            by_k.setdefault(s["k"], []).append(s["top1"])
        def pct(xs, q):
            xs = sorted(xs)
            return xs[min(len(xs) - 1,
                          int(q * (len(xs) - 1)))]
        dist = {k: {"n": len(v), "min": min(v),
                    "p25": pct(v, .25),
                    "median": pct(v, .5),
                    "p75": pct(v, .75), "max": max(v),
                    "mean": round(sum(v) / len(v), 3),
                    "within2": sum(1 for x in v
                                   if x >= full
                                   - NEAR_FULL_TOL)}
                for k, v in sorted(by_k.items())}
        min_full = min((k for k, v in by_k.items()
                        if max(v) >= full), default=None)
        min_near = min((k for k, v in by_k.items()
                        if max(v) >= full - NEAR_FULL_TOL),
                       default=None)
        tol_curve = {}
        for tol in range(0, full + 1):
            mk = min((k for k, v in by_k.items()
                      if max(v) >= full - tol),
                     default=None)
            tol_curve[str(tol)] = mk
            if mk == 0:
                break
        # necessity / sufficiency
        loo = {}
        fullvec = pm[255]["per_state_top1"]
        for blk in range(8):
            mid = 255 & ~(1 << blk)
            s = pm[mid]
            vec = s["per_state_top1"]
            loo[str(blk)] = {
                "top1_delta": s["top1"] - full,
                "lost": sum(1 for a2, b2 in zip(fullvec, vec)
                            if a2 and not b2),
                "gained": sum(1 for a2, b2 in
                              zip(fullvec, vec)
                              if b2 and not a2),
                "mrr_delta": round(s["mrr"]
                                   - pm[255]["mrr"], 4),
                "margin_delta": round(
                    s["mean_margin"]
                    - pm[255]["mean_margin"], 4)}
        singles = {str(blk): pm[1 << blk]["top1"]
                   for blk in range(8)}
        pairs = {f"{a2},{b2}": pm[(1 << a2) | (1 << b2)]
                 ["top1"]
                 for a2 in range(8) for b2 in range(a2 + 1, 8)}
        # redundancy
        n_within2 = sum(1 for s in pm.values()
                        if s["top1"] >= full - NEAR_FULL_TOL)
        n_match_total = sum(1 for s in pm.values()
                            if s["top1"] == full)
        n_match_vec = sum(
            1 for s in pm.values()
            if s["per_state_top1"] == fullvec)
        state_rob = [sum(1 for s in pm.values()
                         if s["per_state_top1"][i]) / 256
                     for i in range(96)]
        state_min_k = []
        for i in range(96):
            ks = [s["k"] for s in pm.values()
                  if s["per_state_top1"][i]]
            state_min_k.append(min(ks) if ks else None)
        return {"full_top1": full,
                "distribution_k": dist,
                "minimal_full_k": min_full,
                "minimal_near_full_k": min_near,
                "tolerance_curve": tol_curve,
                "leave_one_out": loo,
                "singletons": singles,
                "pairs": pairs,
                "frac_within2_all": round(n_within2 / 256, 4),
                "n_match_total": n_match_total,
                "n_match_state_vector": n_match_vec,
                "state_robustness_summary": {
                    "min": round(min(state_rob), 4),
                    "median": round(
                        sorted(state_rob)[48], 4),
                    "max": round(max(state_rob), 4)},
                "state_robustness": [round(x, 4)
                                     for x in state_rob],
                "state_min_k": state_min_k}

    cells = {f"{pop}:{arm}": cell_summary(pop, arm)
             for pop, arm in CELL_ORDER}

    # paired C-v-PF per population
    paired = {}
    for pop in ("heldout", "calibration"):
        dc = cell_results[(pop, "CANONICAL")]
        dp = cell_results[(pop, "PARAM_FIRST")]
        deltas = {mid: dc[mid]["top1"] - dp[mid]["top1"]
                  for mid in range(256)}
        by_k = {}
        for mid, d in deltas.items():
            by_k.setdefault(bin(mid).count("1"),
                            []).append(d)
        # per-block leave-one-out paired per-state
        # discordance (the registered anatomy: counts of
        # states where C loses and PF keeps, and v.v., under
        # each block deletion)
        loo_disc = {}
        for blk in range(8):
            mid = 255 & ~(1 << blk)
            vc = dc[mid]["per_state_top1"]
            vp = dp[mid]["per_state_top1"]
            fc = dc[255]["per_state_top1"]
            fp = dp[255]["per_state_top1"]
            loo_disc[str(blk)] = {
                "C_lost_PF_kept": sum(
                    1 for i in range(96)
                    if fc[i] and not vc[i]
                    and fp[i] and vp[i]),
                "PF_lost_C_kept": sum(
                    1 for i in range(96)
                    if fp[i] and not vp[i]
                    and fc[i] and vc[i])}
        paired[pop] = {
            "mean_delta_by_k": {
                str(k): round(sum(v) / len(v), 3)
                for k, v in sorted(by_k.items())},
            "n_masks_C_better": sum(
                1 for d in deltas.values() if d > 0),
            "n_masks_PF_better": sum(
                1 for d in deltas.values() if d < 0),
            "n_masks_equal": sum(
                1 for d in deltas.values() if d == 0),
            "leave_one_out_paired_discordance": loo_disc}

    receipt = {
        "prereg": "MATH-CYBER-1-RESIDUAL-PATH-ROUTING-"
                  "REPLICATION-PREREG-0",
        "prereg_commit": PREREG_COMMIT,
        "raw_census_sha": raw_sha,
        "coherence": coherence,
        "cells": {k: {kk: vv for kk, vv in v.items()
                      if kk not in ("state_robustness",
                                    "state_min_k", "pairs")}
                  for k, v in cells.items()},
        "paired_C_v_PF": paired,
        "run_wall_s": round(time.time() - t_run, 1),
        "device": "mps", "torch": torch.__version__,
        "pins": {**{a: h for a, (_, h) in CKPTS.items()},
                 "init": INIT_SHA,
                 **{k: h for k, (_, h) in POPS.items()},
                 **{f"rawref_{k}": h
                    for k, (_, h) in RAW_REFS.items()}},
        "start": START,
        "completion_commit": completion_commit()}
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT PIN {a}")
    (OUTDIR / "respath20_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    (OUTDIR / "riders.json").write_text(json.dumps(
        {k: {"pairs": cells[k]["pairs"],
             "state_robustness": cells[k]
             ["state_robustness"],
             "state_min_k": cells[k]["state_min_k"]}
         for k in cells}, indent=1))
    print(json.dumps({
        "coherence": coherence,
        "summary": {k: {kk: v[kk] for kk in
                        ("full_top1", "minimal_full_k",
                         "minimal_near_full_k",
                         "frac_within2_all",
                         "n_match_total")}
                    for k, v in cells.items()},
        "paired": paired}, indent=1), flush=True)
    print("[respath] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

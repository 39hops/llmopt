"""MATH-CYBER-1 SVP-ADJUDICATION-0 — mechanical scoring of the
completed STATE and PROGRAM births on the frozen 72-decision
primary band under the SVP-DESIGN-0 law. Zero training, zero
generation, zero world/sympy anywhere: frozen candidate BYTES
are teacher-force scored, nothing else.

score(a) = mean continuation logprob (eos in T); pessimistic
top-1 (ties count against the arm); exact two-sided McNemar
alpha .05; PROMOTE-PROGRAM / STATE-WIN / INCONCLUSIVE per the
frozen thresholds. Riders: MRR, summed-logprob top-1, u_choice /
term_index strata. Per-decision rows are PERSISTED to
scores.jsonl before any aggregate is computed.

    .venv/bin/python scratch/mathworld1_svpadj.py             (Mac)
    SVPADJ_SEED=10001 .venv/bin/python scratch/mathworld1_svpadj.py
    SVPADJ_SEED=11001 .venv/bin/python scratch/mathworld1_svpadj.py
"""
import hashlib
import json
import math
import os
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

SEED = int(os.environ.get("SVPADJ_SEED", "9001"))
CKPT_BY_SEED = {
    9001: {
        "checkpoints/svp_state.pt":
            "8e0a22f29074ee819a3936748f27939022ac9b974989c988fa"
            "1d3f6f0694c060",
        "checkpoints/svp_program.pt":
            "d9db0049b135f326eb8fa2d9f74e7c067516e49ae597ecaac1"
            "1ecae1dfc57853"},
    10001: {
        "checkpoints/svp_state_s10001.pt":
            "118551181a3f8904da0b6e6da9ef123a2038b83a90d585cb3d"
            "535739a3a3f686",
        "checkpoints/svp_program_s10001.pt":
            "395dfd535d4c446c3372baa8ea33ed2eef6f703c3c36db9715"
            "59019ab4e199a9"},
    11001: {
        "checkpoints/svp_state_s11001.pt":
            "041b3b047df80ff825ddbcff28c879de04586b9616600eb119"
            "6ff7cf69ad4973",
        "checkpoints/svp_program_s11001.pt":
            "3305782548b8e1f2b7a0b63d44ecf79a44c94301f692ba9b93"
            "fbdc255c3a13b7"},
}
gate(SEED in CKPT_BY_SEED, f"UNREGISTERED SVPADJ_SEED {SEED}")
CKPTS = CKPT_BY_SEED[SEED]
CK_STATE, CK_PROGRAM = list(CKPTS)
PINS = dict(CKPTS)
PINS.update({
    "logs/mathworld1/svpeval/episodes.jsonl":
        "cb90ff0f6d655cfe5dc20f091da0597b1bb0e23a4d0c23355a"
        "997e8849c61dd8",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f316524"
        "1406a86253858c6",
})
SUF = "" if SEED == 9001 else f"_s{SEED}"
OUTDIR = Path(f"logs/mathworld1/svpadj{SUF}")
S9001_ADJ_PROTECT = "logs/mathworld1/svpadj/svpadj_receipt.json"
TOK = ActionGCTok()


def protect_9001_adj():
    """The booked seed-9001 adjudication artifacts stay
    byte-frozen while a replication seed scores: the receipt pins
    its own scores.jsonl sha, so asserting the receipt bytes plus
    the scores sha it carries covers both files."""
    if SEED == 9001:
        return
    rec = json.loads(Path(S9001_ADJ_PROTECT).read_text())
    gate(rec["files"]["scores.jsonl"]
         == fsha("logs/mathworld1/svpadj/scores.jsonl"),
         "SEED-9001 svpadj scores.jsonl MUTATED")
    gate(rec["primary"]["STATE_top1"] == 45
         and rec["primary"]["PROGRAM_top1"] == 65,
         "SEED-9001 svpadj receipt MUTATED")


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def mcnemar_exact(b, c):
    """Exact two-sided binomial McNemar. b = only-A-correct,
    c = only-B-correct."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n
    return min(1.0, 2.0 * tail)


def score_decision(model, dev, cur, conts):
    """Teacher-force one decision: one batch, all candidates.
    Returns list of (mean_lp, sum_lp, T)."""
    pre = TOK.encode(f"Current: {cur}\nHints: none\nStep: ")
    enc = [pre + c for c in conts]
    L = max(len(e) for e in enc)
    ids = torch.full((len(enc), L), TOK.pad_id)
    for i, e in enumerate(enc):
        ids[i, :len(e)] = torch.tensor(e)
    ids = ids.to(dev)
    with torch.no_grad():
        logp = torch.log_softmax(model(ids)[:, :-1].float(), -1)
        tok_lp = logp.gather(
            -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu()
    out = []
    for i, c in enumerate(conts):
        s = tok_lp[i, len(pre) - 1:len(pre) - 1 + len(c)]
        gate(bool(torch.isfinite(s).all()), "NON-FINITE LOGPROB")
        out.append((float(s.mean()), float(s.sum()), len(c)))
    return out


def rank_metrics(scores, label_idx):
    """Pessimistic top-1 + rank of the labeled candidate.
    Ties for first count against; rank = 1 + #(strictly better)
    + #(ties with label excluding itself) [pessimistic]."""
    lab = scores[label_idx]
    better = sum(1 for i, s in enumerate(scores)
                 if i != label_idx and s > lab)
    ties = sum(1 for i, s in enumerate(scores)
               if i != label_idx and s == lab)
    rank = 1 + better + ties
    return (better + ties) == 0, rank


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    protect_9001_adj()
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    START = start_provenance(
        ["scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    rows = [json.loads(l)
            for l in open("logs/mathworld1/svpeval/decisions.jsonl")]
    prim = [r for r in rows if r.get("primary_eligible")]
    gate(len(prim) == 72, f"PRIMARY N {len(prim)}")
    gate(torch.backends.mps.is_available(), "MPS UNAVAILABLE")
    dev = torch.device("mps")
    arms = {}
    for view, ck in (("STATE", CK_STATE), ("PROGRAM", CK_PROGRAM)):
        m = build_model(TOK.vocab_size, ctx=4096)
        m.load_state_dict(torch.load(ck, weights_only=True))
        m.eval()
        arms[view] = m.to(dev)

    OUTDIR.mkdir(parents=True)
    t0 = time.time()
    per_dec = []
    with open(OUTDIR / "scores.jsonl", "w") as fo:
        for r in prim:
            cands = r["candidates"]
            gate(len(cands) == r["n_candidates"], "CAND COUNT")
            labs = [i for i, c in enumerate(cands) if c["is_label"]]
            gate(len(labs) == 1, "LABEL COUNT")
            li = labs[0]
            conts = {
                "STATE": [TOK.encode(c["child_sstr"] + "\n")
                          + [TOK.eos_id] for c in cands],
                "PROGRAM": [TOK.encode(c["program_text"])
                            + [TOK.eos_id] for c in cands]}
            rec = {"episode_id": r["episode_id"],
                   "decision_index": r["decision_index"],
                   "n_candidates": len(cands),
                   "label_index": li,
                   "exec_order": ["STATE", "PROGRAM"]}
            for view in ("STATE", "PROGRAM"):
                triples = score_decision(
                    arms[view], dev, r["cur"], conts[view])
                means = [t[0] for t in triples]
                sums = [t[1] for t in triples]
                top1, rank = rank_metrics(means, li)
                top1_sum, rank_sum = rank_metrics(sums, li)
                rec[view] = {
                    "mean_lp": means, "sum_lp": sums,
                    "T": [t[2] for t in triples],
                    "top1": top1, "rank": rank,
                    "top1_sum_rider": top1_sum,
                    "rank_sum_rider": rank_sum}
            fo.write(json.dumps(rec) + "\n")
            per_dec.append(rec)
    # rows are on disk; NOW the aggregates (mechanical)
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST-RUN PIN MISMATCH {p}")
    protect_9001_adj()

    def agg(key):
        s = sum(1 for d in per_dec if d["STATE"][key])
        p = sum(1 for d in per_dec if d["PROGRAM"][key])
        b = sum(1 for d in per_dec
                if d["STATE"][key] and not d["PROGRAM"][key])
        c = sum(1 for d in per_dec
                if d["PROGRAM"][key] and not d["STATE"][key])
        return s, p, b, c

    s1, p1, b, c = agg("top1")
    pval = mcnemar_exact(b, c)
    if p1 > s1 and pval < 0.05:
        verdict = "PROMOTE-PROGRAM"
    elif s1 > p1 and pval < 0.05:
        verdict = "STATE-WIN"
    else:
        verdict = "INCONCLUSIVE"
    mrr = {v: sum(1.0 / d[v]["rank"] for d in per_dec) / 72
           for v in ("STATE", "PROGRAM")}
    ss, ps, bs, cs = agg("top1_sum_rider")

    def stratum(pred):
        sub = [d for d, r in zip(per_dec, prim) if pred(r)]
        return {"n": len(sub),
                "STATE_top1": sum(1 for d in sub if d["STATE"]["top1"]),
                "PROGRAM_top1": sum(1 for d in sub
                                    if d["PROGRAM"]["top1"])}

    has_u = lambda r: any(c["param_kind"] == "u_choice"
                          for c in r["candidates"])
    has_t = lambda r: any(c["param_kind"] == "term_index"
                          for c in r["candidates"])
    lab_t = lambda r: r["candidates"][
        [i for i, c in enumerate(r["candidates"])
         if c["is_label"]][0]]["param_kind"] == "term_index"
    receipt = {
        "n_primary": 72,
        "seed": SEED,
        "checkpoints": {"STATE": CK_STATE, "PROGRAM": CK_PROGRAM},
        "device": str(dev),
        "exec_order_note": "per decision STATE scored first, then "
                           "PROGRAM; candidate order = frozen band "
                           "order",
        "primary": {
            "STATE_top1": s1, "PROGRAM_top1": p1,
            "discordant_state_only": b,
            "discordant_program_only": c,
            "mcnemar_exact_two_sided_p": pval,
            "alpha": 0.05,
            "verdict": verdict},
        "riders": {
            "MRR": mrr,
            "summed_lp_top1": {"STATE": ss, "PROGRAM": ps,
                               "discordant": [bs, cs]},
            "stratum_u_choice_present": stratum(has_u),
            "stratum_u_choice_absent":
                stratum(lambda r: not has_u(r)),
            "stratum_term_index_present": stratum(has_t),
            "stratum_labeled_term_index": stratum(lab_t)},
        "wall_s": round(time.time() - t0, 1),
        "files": {"scores.jsonl": fsha(OUTDIR / "scores.jsonl")},
        "pins": {p: fsha(p) for p in PINS},
        "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "svpadj_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("start", "pins")}, indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

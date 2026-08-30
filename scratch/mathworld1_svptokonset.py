"""MATH-CYBER-1 SVP-HASH-TOKEN-ONSET-PROBE-0 — ONE execution of
the analysis law frozen at TOKEN-ONSET-PROBE-DESIGN-0 (commit
b46bc71f): per-token cumulative-score anatomy of the SEVEN
existing checkpoints (HASH-P1 s16001/s17001/s18001, HASH-P2
s18001, FACTOR s16001/s17001/s18001 as the descriptive rider) on
the exact 96 strict heldout primary states, stored candidate
sets/order, standing prompt. POSTHOC anatomy on known outcomes —
no prediction, no causation, no new descriptor, no redesign.

Frozen objects (verbatim from the prereg):
  S_j(k) = sum_{t<=k} log p(c_{j,t} | prompt, c_{j,<t}), k=1..9.
  Family A: accuracy(k) = pessimistic gold top-1 count at each k
    (ties against gold); MRR(k) rider; per-state per-k
    gold-top flag + argmax-first winner persisted.
  Family B: decision class at k in {gold-top, tied, gold-beaten}
    (tied = best rival == gold); k* = smallest k with class
    constant through k=9 (k=9 qualifies trivially); reported
    separately for final-correct / final-incorrect; histogram,
    median, p25/p75, lock fractions k*<=2/4/6/8.
  Family C: r* = argmax-first non-gold candidate by S_j(9);
    delta_t = gold tok-lp[t] - r* tok-lp[t]; per-position
    median (primary) + mean (rider); cumulative trajectories of
    partial delta sums; for final-incorrect states the first k
    with cumsum < 0 and whether it recovers (>= 0 again by 9).
  Contrasts (never pooled): SEED (P1 across 16001/17001/18001,
    TOKEN-ONSET-MEASURED, no label); PERMUTATION (s18001 H1 v
    H2) with D(k) = acc_H1(k) - acc_H2(k) and the frozen label:
    EARLY-ONSET iff D(4) >= (2/3) D(9); LATE-ONSET iff
    D(4) <= (1/3) D(9); else DIFFUSE/MIXED-ONSET.
  INSTRUMENT GATE (numeric, frozen): the k=9 per-state
    pessimistic top-1 vectors for s18001 H1 and H2 must EXACTLY
    equal the booked heldout-18001 vectors (primary_scores
    pinned) — any per-state mismatch is INSTRUMENT FAILURE.

Anti-peek order: pins -> P2 re-derivation -> heldout structure
gates -> per-checkpoint blind scoring -> ALL raw
per-(checkpoint, state, candidate, k) scores persisted + hashed
-> instrument gate -> families A/B/C -> contrasts + label ->
receipt. No summary printed before the raw artifact is hashed.

Outputs under logs/mathworld1/svptokonset/ (refuse-if-exists):
raw_token_scores.jsonl, summaries.json,
svptokonset_receipt.json.

    .venv/bin/python scratch/mathworld1_svptokonset.py       (Mac)
"""
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_actiontok import (ActionGCTok,  # noqa: E402
                                          OPCODE_ORDER)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import (ORD_MAX,  # noqa: E402
                                        factor_decode,
                                        hash_decode)
from scratch.mathworld1_svpp2qual import (hash2_decode,  # noqa: E402
                                          hash2_symbols)

HELD = "logs/mathworld1/svpdiet3/heldout_test16.jsonl"
BOOKED = "logs/mathworld1/svpheldout18/primary_scores.jsonl"
PINS = {
    HELD:
        "a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec13881"
        "b4509df46ddb",
    BOOKED:
        "0b2c801405d6c8a34b929f9a658ac1e5b0711dbeb9b3abf24c7d"
        "f426b91f1df4",
    "logs/mathworld1/svpp2qual/svpp2qual_receipt.json":
        "47309f22e2be3fba57a28ea0937c985d9ae616121b4cfff608c1"
        "9371246c2337",
}
# (arm_kind, seed) -> (path, full sha) — all from booked birth
# receipts, disk-verified
CKPTS = {
    "H1_s16001": ("checkpoints/svp_grid_hash_s16001.pt",
                  "645fc24f6a829bd50d6e92ec02cc18d4f56f459b8d"
                  "f2c874f1d7351f66a474e0", "HASH_P1"),
    "H1_s17001": ("checkpoints/svp_grid_hash_s17001.pt",
                  "e24237b86314a3ba468f8f213d18f5d9300f4964ee"
                  "c530eabfd51fb625f163d6", "HASH_P1"),
    "H1_s18001": ("checkpoints/svp_grid_hashp1_s18001.pt",
                  "1ec2ea437737f75e57d754c992a75156f982614fb9"
                  "569e50385d4186751cbaac", "HASH_P1"),
    "H2_s18001": ("checkpoints/svp_grid_hashp2_s18001.pt",
                  "26e4afc35870df25a6381f68e65d90280e891399ce"
                  "4c639fd5c977644b8ee844", "HASH_P2"),
    "F_s16001": ("checkpoints/svp_grid_factor_s16001.pt",
                 "c3f7a3e974d92862478ac7a0fd48d57153f2d221db6"
                 "6f80859b59dc28f63949a", "FACTOR"),
    "F_s17001": ("checkpoints/svp_grid_factor_s17001.pt",
                 "12e19fae6fe74b0b5d10fd41ce95192fe324a1f7d46"
                 "6dac8d713d29eb0db882c", "FACTOR"),
    "F_s18001": ("checkpoints/svp_grid_factor_s18001.pt",
                 "ecf5be31f5f7a09d16f0f9a00217983ef2c34a76cce"
                 "4706332ca64a3339fcf5a", "FACTOR"),
}
P2_REALIZATION_SHA = (
    "952f332da4e25961b2dd52c786902e74ba4b33bbf8413f88496a"
    "0df952450ba9")
VOCAB = 340
CODE_BASE = 332
OUTDIR = Path("logs/mathworld1/svptokonset")
TOK = ActionGCTok()


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def rederive_p2():
    stream = hashlib.sha256()
    for r in OPCODE_ORDER:
        for sk in ("W", "I"):
            for so in range(-1, ORD_MAX + 1):
                for pk in ("none", "u_choice", "term_index"):
                    for pi in range(-1, ORD_MAX + 1):
                        stream.update(bytes(
                            hash2_symbols(r, sk, so, pk, pi)))
    return stream.hexdigest()


def token_lps(model, dev, cur, conts):
    """Per-token logprob vectors (len 9) per candidate — the
    same teacher-forced pass as svpadj.score_decision, returning
    the full token vector instead of (mean, sum, T)."""
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
        gate(bool(torch.isfinite(s).all()), "NON-FINITE LP")
        gate(len(c) == 9, "T!=9")
        out.append([float(x) for x in s])
    return out


def gold_class(cum, li, k):
    """Decision class at k: gold-top / tied / gold-beaten,
    pessimistic (tied is not-top)."""
    g = cum[li][k - 1]
    riv = max(cum[j][k - 1] for j in range(len(cum)) if j != li)
    if g > riv:
        return "gold-top"
    if g == riv:
        return "tied"
    return "gold-beaten"


def summarize_kstar(kstars):
    if not kstars:
        return {"n": 0}
    sv = sorted(kstars)
    n = len(sv)
    return {"n": n,
            "histogram": {str(k): sv.count(k)
                          for k in range(1, 10)},
            "median": sv[n // 2],
            "p25": sv[max(0, int(n * 0.25))],
            "p75": sv[min(n - 1, int(n * 0.75))],
            "lock_le": {str(b): sum(1 for x in sv if x <= b)
                        for b in (2, 4, 6, 8)}}


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    import subprocess
    fr = subprocess.run(["git", "rev-parse", "b46bc71f"],
                        capture_output=True, text=True)
    gate(fr.returncode == 0, "PREREG COMMIT NOT FOUND")
    subj = subprocess.run(
        ["git", "log", "-1", "--format=%s", "b46bc71f"],
        capture_output=True, text=True).stdout
    gate("TOKEN-ONSET-PROBE-DESIGN-0" in subj,
         f"PREREG SUBJECT: {subj!r}")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    for name, (p, h, _) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {name}")
    gate(rederive_p2() == P2_REALIZATION_SHA, "P2 DRIFT")
    START = start_provenance(
        ["scratch/mathworld1_svptokonset.py",
         "scratch/mathworld1_svpheldout18.py",
         "scratch/mathworld1_svpp2qual.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")

    allrows = [json.loads(l) for l in open(HELD)]
    pri = [r for r in allrows
           if r["site_role"] == "heldout-I1"]
    gate(len(pri) == 96, "PRIMARY ROWS")
    for r in pri:
        for c in r["candidates"]:
            t = ctup(c)
            gate(factor_decode(c["factor_code"]) == t, "F RT")
            gate(hash_decode(c["hash_code"]) == t, "P1 RT")
            h2 = hash2_symbols(*t)
            gate(hash2_decode(h2) == t, "P2 RT")
            c["_codes"] = {"FACTOR": c["factor_code"],
                           "HASH_P1": c["hash_code"],
                           "HASH_P2": h2}
    booked = {r["block_id"]: r for r in
              (json.loads(l) for l in open(BOOKED))}
    gate(len(booked) == 96, "BOOKED ROWS")

    # blind scoring, all seven checkpoints
    t0 = time.time()
    raw = {}  # name -> list of per-state dicts
    for name, (path, sha, arm) in CKPTS.items():
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(path, weights_only=True))
        gate(sum(p.numel() for p in m.parameters())
             == 19142016, f"PARAMS {name}")
        m.eval()
        m = m.to(dev)
        rows = []
        for r in pri:
            cands = r["candidates"]
            conts = [[CODE_BASE + s for s in c["_codes"][arm]]
                     + [TOK.eos_id] for c in cands]
            lps = token_lps(m, dev, r["cur"], conts)
            li = [i for i, c in enumerate(cands)
                  if c["is_label"]][0]
            rows.append({"block_id": r["block_id"],
                         "term": r["term_cell"],
                         "regime": r["regime"],
                         "label_index": li,
                         "token_lps": lps})
        raw[name] = rows
        del m
    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "raw_token_scores.jsonl", "w") as fo:
        for name in CKPTS:
            for row in raw[name]:
                fo.write(json.dumps(
                    {"checkpoint": name, **row}) + "\n")
    raw_sha = fsha(OUTDIR / "raw_token_scores.jsonl")
    gate(sum(1 for _ in open(OUTDIR / "raw_token_scores.jsonl"))
         == 7 * 96, "RAW ROWS")

    # cumulative scores
    def cums(row):
        out = []
        for v in row["token_lps"]:
            c = []
            s = 0.0
            for x in v:
                s += x
                c.append(s)
            out.append(c)
        return out

    # INSTRUMENT GATE: k=9 vectors reproduce the booked heldout
    mismatch = []
    for name, arm in (("H1_s18001", "HASH_P1"),
                      ("H2_s18001", "HASH_P2")):
        for row in raw[name]:
            cum = cums(row)
            li = row["label_index"]
            top = gold_class(cum, li, 9) == "gold-top"
            b = booked[row["block_id"]]
            gate(b["label_index"] == li,
                 f"LABEL INDEX MISMATCH {row['block_id']}")
            if top != b[arm]["top1"]:
                mismatch.append((name, row["block_id"]))
    gate(not mismatch, f"INSTRUMENT FAILURE k=9 {mismatch[:5]}")

    # families A/B/C per checkpoint
    summaries = {}
    for name in CKPTS:
        rows = raw[name]
        acc = {k: 0 for k in range(1, 10)}
        mrr = {k: 0.0 for k in range(1, 10)}
        winners = []
        kstars_c, kstars_i = [], []
        deltas = []          # per state: [delta_t]*9
        final_correct = []
        crossings = []
        for row in rows:
            cum = cums(row)
            li = row["label_index"]
            n_c = len(cum)
            wrow = {"block_id": row["block_id"]}
            classes = {}
            for k in range(1, 10):
                cls = gold_class(cum, li, k)
                classes[k] = cls
                if cls == "gold-top":
                    acc[k] += 1
                g = cum[li][k - 1]
                better = sum(1 for j in range(n_c)
                             if j != li
                             and cum[j][k - 1] >= g)
                mrr[k] += 1.0 / (1 + better)
                wrow[f"k{k}"] = {
                    "gold_top": cls == "gold-top",
                    "argmax_first": max(
                        range(n_c),
                        key=lambda j: (cum[j][k - 1], -j))}
            winners.append(wrow)
            final = classes[9]
            kstar = 9
            for k in range(1, 10):
                if all(classes[kk] == final
                       for kk in range(k, 10)):
                    kstar = k
                    break
            (kstars_c if final == "gold-top"
             else kstars_i).append(kstar)
            final_correct.append(final == "gold-top")
            # family C
            rivals = [j for j in range(n_c) if j != li]
            rstar = max(rivals,
                        key=lambda j: (cum[j][8], -j))
            d = [row["token_lps"][li][t]
                 - row["token_lps"][rstar][t]
                 for t in range(9)]
            deltas.append(d)
            if final != "gold-top":
                cs = 0.0
                first_cross = None
                recovered = False
                for t in range(9):
                    cs += d[t]
                    if first_cross is None and cs < 0:
                        first_cross = t + 1
                    elif first_cross is not None and cs >= 0:
                        recovered = True
                crossings.append(
                    {"block_id": row["block_id"],
                     "first_crossing": first_cross,
                     "recovered": recovered})
        def med(vals):
            sv = sorted(vals)
            return sv[len(sv) // 2] if sv else 0.0
        summaries[name] = {
            "accuracy_k": {str(k): acc[k]
                           for k in range(1, 10)},
            "mrr_k": {str(k): round(mrr[k] / 96, 4)
                      for k in range(1, 10)},
            "kstar_final_correct": summarize_kstar(kstars_c),
            "kstar_final_incorrect": summarize_kstar(kstars_i),
            "delta_median_by_t": [round(med(
                [d[t] for d in deltas]), 4)
                for t in range(9)],
            "delta_mean_by_t": [round(sum(
                d[t] for d in deltas) / len(deltas), 4)
                for t in range(9)],
            "cum_delta_median_by_k": [round(med(
                [sum(d[:k + 1]) for d in deltas]), 4)
                for k in range(9)],
            "cum_delta_mean_by_k": [round(sum(
                sum(d[:k + 1]) for d in deltas)
                / len(deltas), 4) for k in range(9)],
            "final_incorrect_crossings": {
                "n": len(crossings),
                "first_crossing_histogram": dict(Counter(
                    c["first_crossing"] for c in crossings)),
                "recovered_count": sum(
                    1 for c in crossings if c["recovered"])},
            "winners": winners}

    # contrasts + frozen label
    d4 = (summaries["H1_s18001"]["accuracy_k"]["4"]
          - summaries["H2_s18001"]["accuracy_k"]["4"])
    d9 = (summaries["H1_s18001"]["accuracy_k"]["9"]
          - summaries["H2_s18001"]["accuracy_k"]["9"])
    gate(d9 == 40, f"D9 {d9} != booked 40")
    if d4 >= (2.0 / 3.0) * d9:
        label = "EARLY-ONSET DIFFERENCE"
    elif d4 <= (1.0 / 3.0) * d9:
        label = "LATE-ONSET DIFFERENCE"
    else:
        label = "DIFFUSE/MIXED-ONSET"

    result = {
        "prereg": "MATH-CYBER-1-SVP-HASH-TOKEN-ONSET-PROBE-0",
        "permutation_contrast": {
            "D4": d4, "D9": d9,
            "onset_label": label},
        "seed_contrast": {
            "label": "TOKEN-ONSET-MEASURED",
            "H1_accuracy_k": {
                s: summaries[f"H1_{s}"]["accuracy_k"]
                for s in ("s16001", "s17001", "s18001")}},
        "raw_sha": raw_sha}
    summaries_out = {"result": result,
                     "per_checkpoint": summaries}
    (OUTDIR / "summaries.json").write_text(
        json.dumps(summaries_out, indent=1))
    receipt = {
        "prereg": result["prereg"],
        "design_prereg_commit": fr.stdout.strip(),
        "onset_label": label,
        "D4": d4, "D9": d9,
        "instrument_gate_k9_exact_match":
            len(mismatch) == 0,
        "raw_scores_sha": raw_sha,
        "summaries_sha": fsha(OUTDIR / "summaries.json"),
        "n_rows_raw": 7 * 96,
        "wall_s": round(time.time() - t0, 1),
        "device": str(dev),
        "pins": {p: fsha(p) for p in PINS},
        "ckpt_pins": {n: fsha(CKPTS[n][0]) for n in CKPTS},
        "p2_realization_rederived": rederive_p2(),
        "start": START,
        "completion_commit": completion_commit()}
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    (OUTDIR / "svptokonset_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({"result": result,
                      "accuracy_curves": {
                          n: summaries[n]["accuracy_k"]
                          for n in CKPTS},
                      "kstar_medians": {
                          n: {"correct": summaries[n][
                              "kstar_final_correct"].get(
                                  "median"),
                              "incorrect": summaries[n][
                              "kstar_final_incorrect"].get(
                                  "median")}
                          for n in CKPTS},
                      "wall_s": receipt["wall_s"]},
                     indent=1), flush=True)
    print("[svptokonset] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

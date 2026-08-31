"""MATH-CYBER-1 SVP-FIELD-ORDER-REPLICATION-KNOWN-SET-HELDOUT-
ONSET-SCORE-20001 — stage 3 (final adjudicating stage) of the
frozen seed-20001 replication protocol (replication prereg
732a5312; scoring law byte-adopted from the booked seed-19001
stage-3 instrument scratch/mathworld1_svpfoheld.py — deltas are
20001 pins/paths and replication provenance only), under live
calibration authority (FIELD-ORDER-REPLICATION CALIBRATION
FIRED, receipt byte-pinned). ONE joint
prefix-scoring run of the two sealed seed-20001 checkpoints
(CANONICAL / PARAM-FIRST) on the 96 frozen strict heldout
primary states; BOTH registered endpoints derive from the SAME
raw per-(arm, state, candidate, k) token log-probabilities:
  S_j(k) = sum_{t<=k} log p(token_t | prompt, token_<t),
  payload tokens 1..8, EOS = token 9, arm-specific
  serialization, equal prefix lengths, no normalization.
COMPLETION (k=9 only, frozen symmetric law): ORDER-DAMAGE iff
CANONICAL > PARAM-FIRST and exact two-sided McNemar p < .05;
ORDER-REVERSAL symmetric; else NO-DIRECTIONAL-SEPARATION (NOT
equivalence). MECHANISM: per arm accuracy(k) pessimistic,
gain(k) = accuracy(k) - accuracy(k-1) with accuracy(0)=0,
k_step = EARLIEST argmax gain(k); CANONICAL-ONSET-ALIGNED iff
k_step == 8; PARAM-FIRST-ONSET-ALIGNED iff k_step == 3;
ONSET-RELOCATED iff BOTH. INSTRUMENT COHERENCE: the k=9
per-state top-1 vector is computed ONCE and consumed by both
endpoints; any divergence between derivations is INSTRUMENT
FAILURE. INTERPRETIVE MAP (frozen, mechanical): A
NO-DIRECTIONAL-SEPARATION + ONSET-RELOCATED => SEMANTIC-SOCKET
(strongest available from this known-set intervention; null is
not equivalence); B ORDER-DAMAGE + ONSET-RELOCATED =>
ORDER-LOAD-BEARING + SOCKET-MOVES; C ORDER-DAMAGE +
NOT-RELOCATED => ORDER-LOAD-BEARING, no socket support; D
NO-DIRECTIONAL-SEPARATION + NOT-RELOCATED => MIXED; E
ORDER-REVERSAL => SIGNIFICANT ORDER REVERSAL, onset beside it.
Asymmetric single-arm alignment reported as its exact cell,
never called relocation.

No retraining, no checkpoint mutation, no new arm or
population, no candidate regeneration or filtering, no
historical-result-dependent branch, no new endpoint or
threshold. Riders (post-receipt, descriptive only): per-state
correctness transitions; accuracy(k) restricted to the two
PRE-REGISTERED structural groups (the 29 early / 67 relocated
states, read from the frozen census per_state maps, byte-
pinned); final-rival token contributions per the frozen
token-onset formulas (r* from the k=9 score, delta_t medians).

Anti-peek order: authority + pins -> P2-free structure gates ->
load checkpoints -> blind-score both arms -> persist + hash raw
token scores -> instrument-coherence gates -> completion
endpoint -> mechanism endpoint -> interpretive cell -> receipt
-> only then riders. No per-state value printed before the raw
artifact is hashed.

Outputs under logs/mathworld1/svpfoheld20/ (refuse-if-exists):
raw_token_scores.jsonl, svpfoheld20_receipt.json, riders.json.

    .venv/bin/python scratch/mathworld1_svpfohrepl.py        (Mac)
"""
import hashlib
import json
import math
import sys
import time
from collections import Counter
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

HELD = "logs/mathworld1/svpdiet3/heldout_test16.jsonl"
CAL_RECEIPT = ("logs/mathworld1/svpfocal20/"
               "svpfocal20_receipt.json")
CENSUS_JSON = "logs/mathworld1/svpforepl_census/census.json"
PINS = {
    HELD:
        "a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec13881"
        "b4509df46ddb",
    CAL_RECEIPT:
        "83d3220d90bd9a7adf1b3204c0b914399abd5f4a9a9c967a6fde"
        "d8a39afb0bf8",
    CENSUS_JSON:
        "7d8343b3f328b00a1147b675441181e897d3d96eaff33fe33d40"
        "3aa482dd71d5",
    "logs/mathworld1/svpforepl_s20001_receipt.json":
        "70ff3248a9b2f6e584c2d7bb8e9fc7b59853ebb7055f58193783"
        "fbd61eab78b3",
}
CKPTS = {
    "CANONICAL": ("checkpoints/svp_forder_canonical_s20001.pt",
                  "0a841a5f2a43b6f64b0dac8259c26fd79961e6ab91"
                  "359a54be9c2582815b3e34"),
    "PARAM_FIRST": ("checkpoints/svp_forder_paramfirst_s20001"
                    ".pt",
                    "b7198ff2e7b903ab5ed075fe947cb29142c5790e"
                    "c84831434c53a598e466c322"),
}
INIT_SHA = ("7c95e77f8d7ccea5f4dd71c989e4d3225e347a178032d435"
            "39a1ae6ef62c9452")
INIT_CK = "checkpoints/svp_forder_init_s20001.pt"
VOCAB = 340
CODE_BASE = 332
ARMS = ["CANONICAL", "PARAM_FIRST"]
ALPHA = 0.05
K_STRUCT = {"CANONICAL": 8, "PARAM_FIRST": 3}
OUTDIR = Path("logs/mathworld1/svpfoheld20")
TOK = ActionGCTok()


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def binom_minlik_p(k_obs, n):
    """Exact two-sided p under Binomial(n, 0.5)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) / 2.0 ** n for k in range(n + 1)]
    thresh = pmf[k_obs] * (1 + 1e-12)
    return min(1.0, sum(p for p in pmf if p <= thresh))


def token_lps(model, dev, cur, conts):
    """Per-token logprob vectors (len 9) per candidate."""
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


def cums(vecs):
    out = []
    for v in vecs:
        c, s = [], 0.0
        for x in v:
            s += x
            c.append(s)
        out.append(c)
    return out


def gold_top(cum, li, k):
    """Pessimistic: gold strictly above every rival at k."""
    g = cum[li][k - 1]
    return all(cum[j][k - 1] < g for j in range(len(cum))
               if j != li)


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    cal = json.loads(Path(CAL_RECEIPT).read_text())
    gate(cal["verdict"]
         == "FIELD-ORDER-REPLICATION CALIBRATION FIRED",
         "NO STAGE-3 AUTHORITY")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "INIT PIN")
    START = start_provenance(
        ["scratch/mathworld1_svpfohrepl.py",
         "scratch/mathworld1_svpfoheld.py",
         "scratch/mathworld1_svpfoclrepl.py",
         "scratch/mathworld1_svpforepl.py",
         "scratch/mathworld1_svpfocal.py",
         "scratch/mathworld1_svpforder.py",
         "scratch/mathworld1_svptokonset.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    allrows = [json.loads(l) for l in open(HELD)]
    pri = [r for r in allrows
           if r["site_role"] == "heldout-I1"]
    gate(len(pri) == 96, "PRIMARY ROWS")
    gate(Counter((r["term_cell"], r["regime"]) for r in pri)
         == Counter({(2, "IN"): 24, (2, "OUT"): 24,
                     (3, "IN"): 24, (3, "OUT"): 24}), "STRATA")
    for r in pri:
        labs = [c for c in r["candidates"] if c["is_label"]]
        gate(len(labs) == 1, "LABEL COUNT")
        lt = ctup(labs[0])
        gate(lt[0] == "i_unprod" and lt[1] == "I"
             and lt[2] == 1, "LABEL SEMANTICS")
        for c in r["candidates"]:
            t = ctup(c)
            cz = c["factor_code"]
            gate(factor_decode(cz) == t, "C RT")
            pz = pf_encode(t)
            gate(pf_decode(pz) == t, "PF RT")
            gate(pz == [cz[PERM[i]] for i in range(8)],
                 "PERM IDENTITY")
            c["_pf"] = pz
    # frozen structural groups from the byte-pinned census
    census = json.loads(Path(CENSUS_JSON).read_text())
    per_state_c = census["structural"]["CANONICAL"]["per_state"]
    early = {b for b, k in per_state_c.items() if k == 1}
    late = {b for b, k in per_state_c.items() if k == 8}
    gate(len(early) == 29 and len(late) == 67, "GROUP SIZES")

    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    arms = {}
    for a in ARMS:
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(CKPTS[a][0],
                                     weights_only=True))
        gate(sum(p.numel() for p in m.parameters())
             == 19142016, f"PARAM COUNT {a}")
        m.eval()
        arms[a] = m.to(dev)

    # blind-score both arms, persist raw
    t0 = time.time()
    raw = {a: [] for a in ARMS}
    for r in pri:
        cands = r["candidates"]
        gate(len(cands) == r["n_candidates"], "CAND COUNT")
        li = [i for i, c in enumerate(cands)
              if c["is_label"]][0]
        conts = {
            "CANONICAL": [[CODE_BASE + s
                           for s in c["factor_code"]]
                          + [TOK.eos_id] for c in cands],
            "PARAM_FIRST": [[CODE_BASE + s for s in c["_pf"]]
                            + [TOK.eos_id] for c in cands]}
        for a in ARMS:
            lps = token_lps(arms[a], dev, r["cur"], conts[a])
            raw[a].append({"block_id": r["block_id"],
                           "term": r["term_cell"],
                           "regime": r["regime"],
                           "label_index": li,
                           "token_lps": lps})
    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "raw_token_scores.jsonl", "w") as fo:
        for a in ARMS:
            for row in raw[a]:
                fo.write(json.dumps({"arm": a, **row}) + "\n")
    raw_sha = fsha(OUTDIR / "raw_token_scores.jsonl")
    gate(sum(1 for _ in open(OUTDIR / "raw_token_scores.jsonl"))
         == 2 * 96, "RAW ROWS")

    # ONE k-vector derivation per arm/state, consumed by BOTH
    # endpoints (instrument coherence by construction)
    top_by_k = {a: {} for a in ARMS}   # arm -> block -> [9 bool]
    for a in ARMS:
        for row in raw[a]:
            cum = cums(row["token_lps"])
            li = row["label_index"]
            top_by_k[a][row["block_id"]] = [
                gold_top(cum, li, k) for k in range(1, 10)]
    # coherence gate: k=9 totals derived twice agree
    k9 = {a: {b: v[8] for b, v in top_by_k[a].items()}
          for a in ARMS}
    coherence_checked = 0
    for a in ARMS:
        for row in raw[a]:
            v2 = gold_top(cums(row["token_lps"]),
                          row["label_index"], 9)
            gate(k9[a][row["block_id"]] == v2,
                 f"COHERENCE {a} {row['block_id']}")
            coherence_checked += 1

    # COMPLETION ENDPOINT (k=9 only)
    c_top = sum(1 for v in k9["CANONICAL"].values() if v)
    p_top = sum(1 for v in k9["PARAM_FIRST"].values() if v)
    c_only = sum(1 for b in k9["CANONICAL"]
                 if k9["CANONICAL"][b]
                 and not k9["PARAM_FIRST"][b])
    p_only = sum(1 for b in k9["CANONICAL"]
                 if k9["PARAM_FIRST"][b]
                 and not k9["CANONICAL"][b])
    n_disc = c_only + p_only
    mcnemar_p = binom_minlik_p(c_only, n_disc)
    if c_top > p_top and mcnemar_p < ALPHA:
        completion = "ORDER-DAMAGE"
    elif p_top > c_top and mcnemar_p < ALPHA:
        completion = "ORDER-REVERSAL"
    else:
        completion = "NO-DIRECTIONAL-SEPARATION"

    # MECHANISM ENDPOINT
    acc = {a: {k: sum(1 for v in top_by_k[a].values()
                      if v[k - 1]) for k in range(1, 10)}
           for a in ARMS}
    gains = {a: {k: acc[a][k] - (acc[a][k - 1] if k > 1 else 0)
                 for k in range(1, 10)} for a in ARMS}
    k_step = {a: min([k for k in range(1, 10)
                      if gains[a][k]
                      == max(gains[a].values())])
              for a in ARMS}
    aligned = {a: k_step[a] == K_STRUCT[a] for a in ARMS}
    relocated = all(aligned.values())

    # INTERPRETIVE CELL (frozen map, mechanical)
    if completion == "ORDER-REVERSAL":
        cell = "E: SIGNIFICANT ORDER REVERSAL"
    elif completion == "NO-DIRECTIONAL-SEPARATION" and relocated:
        cell = "A: SEMANTIC-SOCKET"
    elif completion == "ORDER-DAMAGE" and relocated:
        cell = "B: ORDER-LOAD-BEARING + SOCKET-MOVES"
    elif completion == "ORDER-DAMAGE":
        cell = "C: ORDER-LOAD-BEARING, no socket support"
    else:
        cell = "D: MIXED"

    # MRR at k=9 (descriptive, pessimistic rank)
    def mrr(a):
        s = 0.0
        for row in raw[a]:
            cum = cums(row["token_lps"])
            li = row["label_index"]
            g = cum[li][8]
            rank = 1 + sum(1 for j in range(len(cum))
                           if j != li and cum[j][8] >= g)
            s += 1.0 / rank
        return round(s / 96, 4)

    receipt = {
        "prereg": "MATH-CYBER-1-SVP-FIELD-ORDER-REPLICATION-"
                  "KNOWN-SET-HELDOUT-ONSET-SCORE-20001",
        "prereg_commit": "732a5312",
        "replication_of": "seed 19001 stage 3 (prereg "
                          "3ac5a70e)",
        "seed": 20001,
        "completion_endpoint": {
            "verdict": completion,
            "CANONICAL_top1": c_top,
            "PARAM_FIRST_top1": p_top,
            "canonical_only_discordant": c_only,
            "param_first_only_discordant": p_only,
            "n_discordant": n_disc,
            "mcnemar_p_two_sided": mcnemar_p,
            "alpha": ALPHA,
            "mrr_descriptive": {a: mrr(a) for a in ARMS}},
        "mechanism_endpoint": {
            "accuracy_k": {a: {str(k): acc[a][k]
                               for k in range(1, 10)}
                           for a in ARMS},
            "gain_k": {a: {str(k): gains[a][k]
                           for k in range(1, 10)}
                       for a in ARMS},
            "k_step": k_step,
            "k_struct": K_STRUCT,
            "CANONICAL_ONSET_ALIGNED": aligned["CANONICAL"],
            "PARAM_FIRST_ONSET_ALIGNED":
                aligned["PARAM_FIRST"],
            "ONSET_RELOCATED": relocated},
        "interpretive_cell": cell,
        "instrument_coherence_per_state_checks":
            coherence_checked,
        "raw_scores_sha": raw_sha,
        "calibration_authority": {
            "receipt_verdict": cal["verdict"],
            "receipt_sha_pinned": PINS[CAL_RECEIPT]},
        "n_rows_raw": 2 * 96,
        "wall_s": round(time.time() - t0, 1),
        "device": str(dev),
        "torch": torch.__version__,
        "pins": {p: fsha(p) for p in PINS},
        "ckpt_paths_pins": {a: {"path": CKPTS[a][0],
                                "sha256": fsha(CKPTS[a][0])}
                            for a in ARMS},
        "init_pin": fsha(INIT_CK),
        "start": START,
        "completion_commit": completion_commit()}
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "POST INIT PIN")
    (OUTDIR / "svpfoheld20_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("completion_endpoint",
                       "mechanism_endpoint",
                       "interpretive_cell", "wall_s")},
                     indent=1), flush=True)

    # riders (post-receipt, descriptive only)
    trans = Counter()
    for b in k9["CANONICAL"]:
        trans[(k9["CANONICAL"][b], k9["PARAM_FIRST"][b])] += 1
    def acc_group(a, grp):
        return {str(k): sum(1 for b in grp
                            if top_by_k[a][b][k - 1])
                for k in range(1, 10)}
    # final-rival token contributions (frozen token-onset law)
    deltas = {a: [] for a in ARMS}
    for a in ARMS:
        for row in raw[a]:
            cum = cums(row["token_lps"])
            li = row["label_index"]
            rivals = [j for j in range(len(cum)) if j != li]
            rstar = max(rivals, key=lambda j: (cum[j][8], -j))
            deltas[a].append([row["token_lps"][li][t]
                              - row["token_lps"][rstar][t]
                              for t in range(9)])
    def med(vals):
        sv = sorted(vals)
        return round(sv[len(sv) // 2], 4) if sv else None
    riders = {
        "state_transitions_C_PF": {
            "both_correct": trans[(True, True)],
            "C_only": trans[(True, False)],
            "PF_only": trans[(False, True)],
            "both_wrong": trans[(False, False)]},
        "accuracy_k_early29": {a: acc_group(a, early)
                               for a in ARMS},
        "accuracy_k_late67": {a: acc_group(a, late)
                              for a in ARMS},
        "delta_median_by_t": {a: [med([d[t]
                                       for d in deltas[a]])
                                  for t in range(9)]
                              for a in ARMS}}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    print("[svpfoheld20] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

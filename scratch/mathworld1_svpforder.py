"""MATH-CYBER-1 SVP-FIELD-ORDER-MATERIALIZATION-CENSUS-BIRTH-
19001 — stage 1 of the frozen FIELD-ORDER protocol (prereg
3ac5a70e): PARAM-FIRST implementation + qualification, the
HARD-STOP pre-training structural census, the descriptive
exposure census, and (only when every gate has passed) the
two-arm seed-19001 BIRTH. No calibration, no heldout scoring,
no token-onset scoring, no verdict.

ARMS: CANONICAL = the shipped factor payload
[r1,r2,k,o1,o2,pk,x1,x2]; PARAM_FIRST = the LITERAL positional
permutation with PERM = [5,6,7,0,1,2,3,4] (output position i
takes canonical position PERM[i]) and INV = [3,4,5,6,7,0,1,2]
(canonical position j sits at output position INV[j]); decode =
un-permute then the shipped factor_decode. No hashing, no value
renaming, no changed semantics; per-action payload token
MULTISET identical by construction and HARD-gated.

MODES:
  SVPFO_CENSUS=1      qualification + structural + exposure
                      census -> census receipt (HARD STOP on any
                      structural disagreement with the frozen
                      predictions: CANONICAL 29@k=1 + 67@k=8 max
                      8; PARAM-FIRST same 29@k=1 + same 67@k=3
                      max 3; group state-identity gated). MUST
                      run and pass BEFORE any init.
  SVPFO_MAKE_INIT=1   dedicated init pre-step (double-build
                      byte-identity, sha == design-time pin).
  (no env)            path-isolated smoke: both real encoders,
                      forward/backward/optimizer/scheduler,
                      ordinary + longest + tail batches, both
                      execution orders, no production writes.
  SVPFO_PRODUCTION=1  ONE production birth; refuse-if-exists;
                      requires green census receipt + smoke.

STRUCTURAL CENSUS LAW (frozen, from the prereg): per arm, per
state, the earliest prefix position at which the gold payload is
distinguishable from ALL rivals under the pessimistic
prefix-tie law = max over rivals of the first position where
gold and rival payloads differ; group census + single-step
maximum + cross-arm group state-identity.

TRAINING LAW: the booked two-arm paired law verbatim (74,860
rows a943ba7f/0ef3d8a8/897c8bf8, plan f55e9fee, vocab 340,
19,142,016 params/arm, BS=32, 3 epochs, 7,020 updates/arm,
2,021,220 continuation tokens/arm, masks, alternating arm-first
3,510/3,510, zero I1/t2-t3 labels, per-batch equal-compute
gates) PLUS the treatment-specific per-row hard gate:
PARAM_FIRST payload == PERM(CANONICAL payload) with identical
token multiset, gated on every row at load.

Paths (refuse-if-exists in production):
  checkpoints/svp_forder_init_s19001.pt   (pre-step, sha-gated)
  checkpoints/svp_forder_canonical_s19001.pt
  checkpoints/svp_forder_paramfirst_s19001.pt
  logs/mathworld1/svpforder_s19001_receipt.json
Census: logs/mathworld1/svpforder_census/ (refuse-if-exists)
Smoke: logs/mathworld1/smoke_svpforder19.json

    SVPFO_CENSUS=1 .venv/bin/python scratch/mathworld1_svpforder.py
    SVPFO_MAKE_INIT=1 .venv/bin/python scratch/mathworld1_svpforder.py
    .venv/bin/python scratch/mathworld1_svpforder.py         (smoke)
    SVPFO_PRODUCTION=1 .venv/bin/python scratch/mathworld1_svpforder.py
"""
import hashlib
import io
import json
import math
import os
import resource
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
from scratch.mathworld1_svpbirth import (batch_plan,  # noqa: E402
                                         gate, make_opt, sha_b,
                                         train_step)
from scratch.mathworld1_svpcode import (ORD_MAX,  # noqa: E402
                                        factor_decode,
                                        factor_symbols,
                                        in_domain)

SEED = 19001
VOCAB = 340
CODE_BASE = 332
SEQ_CAP = 512
PERM = [5, 6, 7, 0, 1, 2, 3, 4]
INV = [3, 4, 5, 6, 7, 0, 1, 2]
PAIRED = "data/matsub_paired.jsonl"
PAIRED_SHA = ("a943ba7fc581db743b07192e5d951fadd"
              "dd2ba19bca3225b75d8402351d468e8")
AUG = "logs/mathworld1/svpdiet/balanced_grid_train.jsonl"
AUG_SHA = ("0ef3d8a880a7e07712d8de757bc1670df12701e487b856"
           "b44c97f8db16cb3759")
MANIFEST = "logs/mathworld1/svpdiet/combined_train_manifest.jsonl"
MANIFEST_SHA = ("897c8bf8fd2b6d39e361ed541d3e14c53c1c1302eee"
                "d560f77ee8fb2f2477bdd")
PLAN_SHA = ("f55e9fee1b00f57256f3c8152be149fbb4c619479c14e4"
            "d8c01e1672afebad4d")
HELD = "logs/mathworld1/svpdiet3/heldout_test16.jsonl"
HELD_SHA = ("a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec"
            "13881b4509df46ddb")
CAL = "logs/mathworld1/svpdiet3/covered_calibration.jsonl"
CAL_SHA = ("af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b085"
           "09794d0fb7978db3")
EPOCHS = 3
TOTAL_STEPS = 7020
N_ROWS = 74860
TOK_CONT_EXPECT = 9 * N_ROWS * EPOCHS
ARMS = ["CANONICAL", "PARAM_FIRST"]
ORDER = {0: ["CANONICAL", "PARAM_FIRST"],
         1: ["PARAM_FIRST", "CANONICAL"]}
INIT_CK = Path("checkpoints/svp_forder_init_s19001.pt")
# pinned at design time by two independent byte-equal builds,
# 2026-08-30:
INIT_SHA = ("20751529f22e3f5da4bfdf1504fbc8e8f33200003de2eec7"
            "848a2a146120339e")
CKS = {"CANONICAL":
       Path("checkpoints/svp_forder_canonical_s19001.pt"),
       "PARAM_FIRST":
       Path("checkpoints/svp_forder_paramfirst_s19001.pt")}
RECEIPT = Path("logs/mathworld1/svpforder_s19001_receipt.json")
SMOKE_RECEIPT = Path("logs/mathworld1/smoke_svpforder19.json")
CENSUS_DIR = Path("logs/mathworld1/svpforder_census")
CENSUS_RECEIPT = CENSUS_DIR / "svpforder_census_receipt.json"
MAKE_INIT = os.environ.get("SVPFO_MAKE_INIT") == "1"
PRODUCTION = os.environ.get("SVPFO_PRODUCTION") == "1"
CENSUS = os.environ.get("SVPFO_CENSUS") == "1"
TOK = ActionGCTok()

# frozen structural predictions (prereg 3ac5a70e)
PRED = {"CANONICAL": {"1": 29, "8": 67},
        "PARAM_FIRST": {"1": 29, "3": 67}}
PRED_MAX = {"CANONICAL": 8, "PARAM_FIRST": 3}


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def pf_encode(tup):
    canon = factor_symbols(*tup)
    return [canon[PERM[i]] for i in range(8)]


def pf_decode(sym):
    canon = [sym[INV[j]] for j in range(8)]
    return factor_decode(canon)


def encode(arm, tup):
    return (factor_symbols(*tup) if arm == "CANONICAL"
            else pf_encode(tup))


def sep_position(gold, rivals):
    """Earliest prefix position at which gold is distinguishable
    from ALL rivals (pessimistic prefix-tie law) = max over
    rivals of the first differing position."""
    best = 0
    for r in rivals:
        diffs = [i + 1 for i in range(8) if gold[i] != r[i]]
        gate(bool(diffs), "IDENTICAL RIVAL PAYLOAD")
        best = max(best, diffs[0])
    return best


def load_rows():
    gate(fsha(PAIRED) == PAIRED_SHA, "PAIRED PIN")
    gate(fsha(AUG) == AUG_SHA, "AUG PIN")
    gate(fsha(MANIFEST) == MANIFEST_SHA, "MANIFEST PIN")
    rows = [json.loads(l) for l in open(PAIRED)]
    gate(len(rows) == 73324, "NATURAL ROWS")
    aug = [json.loads(l) for l in open(AUG)]
    gate(len(aug) == 1536, "AUG ROWS")
    rows += aug
    gate(len(rows) == N_ROWS, "COMBINED ROWS")
    man_ids = [json.loads(l)["row_id"] for l in open(MANIFEST)]
    gate(sorted(man_ids) == sorted(r["row_id"] for r in rows),
         "MANIFEST MISMATCH")
    heldout_hits = 0
    for r in rows:
        tup = (r["rule"], r["site_kind"], r["site_ordinal"],
               r["param_kind"], r["param_index"])
        gate(in_domain(*tup), "ROW OUT OF DOMAIN")
        if tup[0] == "i_unprod" and tup[1] == "I" \
                and tup[2] == 1 and tup[3] == "term_index" \
                and tup[4] in (2, 3):
            heldout_hits += 1
        cz = factor_symbols(*tup)
        pz = pf_encode(tup)
        gate(factor_decode(cz) == tup, "C INVERSE")
        gate(pf_decode(pz) == tup, "PF INVERSE")
        gate(pz == [cz[PERM[i]] for i in range(8)],
             "PERM IDENTITY ROW")
        gate(sorted(pz) == sorted(cz), "MULTISET ROW")
        r["_C"] = [CODE_BASE + s for s in cz] + [TOK.eos_id]
        r["_P"] = [CODE_BASE + s for s in pz] + [TOK.eos_id]
        gate(len(r["_C"]) == len(r["_P"]) == 9, "T!=9")
        r["_pre"] = TOK.encode(
            f"Current: {r['cur']}\nHints: none\nStep: ")
        gate(len(r["_pre"]) + 9 <= SEQ_CAP, "SEQ CAP")
    gate(heldout_hits == 0,
         f"HELD-OUT LABELS IN TRAINING {heldout_hits}")
    return rows


def run_census():
    if CENSUS_DIR.exists():
        raise SystemExit(f"REFUSING: {CENSUS_DIR} exists")
    gate(not INIT_CK.exists(),
         "CENSUS MUST PRECEDE INIT")
    gate(fsha(HELD) == HELD_SHA, "HELD PIN")
    gate(fsha(CAL) == CAL_SHA, "CAL PIN")
    START = start_provenance(
        ["scratch/mathworld1_svpforder.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/lab/provenance.py"])
    t0 = time.time()
    # QUALIFICATION: full domain, both arms
    n_dom = 0
    for r in OPCODE_ORDER:
        for sk in ("W", "I"):
            for so in range(-1, ORD_MAX + 1):
                for pk in ("none", "u_choice", "term_index"):
                    for pi in range(-1, ORD_MAX + 1):
                        n_dom += 1
                        tup = (r, sk, so, pk, pi)
                        cz = factor_symbols(*tup)
                        pz = pf_encode(tup)
                        gate(len(cz) == len(pz) == 8
                             and all(0 <= s < 8 for s in pz)
                             and all(0 <= s < 8 for s in cz),
                             "WIDTH/ALPHABET")
                        gate(factor_decode(cz) == tup, "C RT")
                        gate(pf_decode(pz) == tup, "PF RT")
                        gate(pz == [cz[PERM[i]]
                                    for i in range(8)],
                             "PERM IDENTITY")
                        gate(sorted(pz) == sorted(cz),
                             "MULTISET IDENTITY")
    gate(n_dom == 884736, "DOMAIN SIZE")
    gate(all(INV[PERM[i]] == i for i in range(8)), "INV LAW")

    # eval candidate coverage, both arms
    n_eval = 0
    pri = []
    for src in (CAL, HELD):
        for l in open(src):
            row = json.loads(l)
            for c in row["candidates"]:
                tup = ctup(c)
                gate(in_domain(*tup), "EVAL DOMAIN")
                gate(pf_decode(pf_encode(tup)) == tup,
                     "EVAL PF RT")
                n_eval += 1
            if src == HELD and row.get("site_role") \
                    == "heldout-I1":
                pri.append(row)
    gate(len(pri) == 96, "PRIMARY STATES")

    # STRUCTURAL CENSUS (HARD STOP)
    census = {}
    groups = {}
    for arm in ARMS:
        hist = Counter()
        per_state = {}
        for r in pri:
            cands = r["candidates"]
            li = [i for i, c in enumerate(cands)
                  if c["is_label"]][0]
            gold = encode(arm, ctup(cands[li]))
            rivals = [encode(arm, ctup(c))
                      for i, c in enumerate(cands) if i != li]
            k = sep_position(gold, rivals)
            hist[k] += 1
            per_state[r["block_id"]] = k
        gate(len(per_state) == 96, f"PER-STATE COUNT {arm}")
        census[arm] = {"histogram": {str(k): v for k, v
                                     in sorted(hist.items())},
                       "per_state": per_state}
        gains = {k: hist.get(k, 0) for k in range(1, 9)}
        kmax = max(gains, key=lambda k: (gains[k], -k))
        census[arm]["single_step_max"] = kmax
        groups[arm] = {k: {b for b, kk in per_state.items()
                           if kk == k} for k in hist}
    # HARD STOP gates v frozen predictions
    for arm in ARMS:
        gate(census[arm]["histogram"] == PRED[arm],
             f"STRUCTURAL CENSUS DISAGREES {arm}: "
             f"{census[arm]['histogram']} != {PRED[arm]}")
        gate(census[arm]["single_step_max"] == PRED_MAX[arm],
             f"STRUCTURAL MAX DISAGREES {arm}")
    gate(groups["CANONICAL"][1] == groups["PARAM_FIRST"][1],
         "EARLY GROUP IDENTITY")
    gate(groups["CANONICAL"][8] == groups["PARAM_FIRST"][3],
         "LATE GROUP IDENTITY")

    # DESCRIPTIVE EXPOSURE CENSUS over training targets
    rows = load_rows()
    exp_pos = {a: [Counter() for _ in range(8)] for a in ARMS}
    agg = {a: Counter() for a in ARMS}
    for r in rows:
        cz = [t - CODE_BASE for t in r["_C"][:8]]
        pz = [t - CODE_BASE for t in r["_P"][:8]]
        for k in range(8):
            exp_pos["CANONICAL"][k][cz[k]] += 1
            exp_pos["PARAM_FIRST"][k][pz[k]] += 1
        agg["CANONICAL"].update(cz)
        agg["PARAM_FIRST"].update(pz)
    gate(agg["CANONICAL"] == agg["PARAM_FIRST"],
         "AGGREGATE EXPOSURE DRIFT")
    # candidate prefix-sharing structure by k (descriptive)
    share = {}
    for arm in ARMS:
        by_k = Counter()
        for r in pri:
            cands = r["candidates"]
            li = [i for i, c in enumerate(cands)
                  if c["is_label"]][0]
            gold = encode(arm, ctup(cands[li]))
            for i, c in enumerate(cands):
                if i == li:
                    continue
                z = encode(arm, ctup(c))
                lcp = 0
                for x, y in zip(gold, z):
                    if x != y:
                        break
                    lcp += 1
                by_k[lcp] += 1
        share[arm] = {str(k): v for k, v
                      in sorted(by_k.items())}
    CENSUS_DIR.mkdir(parents=True)
    out = {"structural": {a: {"histogram":
                              census[a]["histogram"],
                              "single_step_max":
                              census[a]["single_step_max"],
                              "per_state":
                              census[a]["per_state"]}
                          for a in ARMS},
           "group_identity": {"early_29_identical": True,
                              "late_67_identical": True},
           "exposure_per_position": {
               a: [{str(k): v for k, v in sorted(c.items())}
                   for c in exp_pos[a]] for a in ARMS},
           "exposure_aggregate": {
               a: {str(k): v for k, v in
                   sorted(agg[a].items())} for a in ARMS},
           "prefix_sharing_by_lcp": share,
           "n_domain": n_dom, "n_eval_candidates": n_eval,
           "n_train_rows": len(rows)}
    (CENSUS_DIR / "census.json").write_text(
        json.dumps(out, indent=1))
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-FACTOR-FIELD-ORDER-"
                  "PREREG-0",
        "stage": "materialization/census",
        "verdict": "CENSUS QUALIFIED",
        "structural_match_frozen_predictions": True,
        "perm": PERM, "inv": INV,
        "census_sha": fsha(CENSUS_DIR / "census.json"),
        "hard_bars": {
            "domain_roundtrips_both_arms": n_dom,
            "perm_identity": n_dom,
            "multiset_identity": n_dom,
            "eval_candidates_covered": n_eval,
            "train_rows_gated": len(rows),
            "aggregate_exposure_equal": True},
        "wall_s": round(time.time() - t0, 1),
        "pins": {p: fsha(p) for p in
                 (PAIRED, AUG, MANIFEST, HELD, CAL)},
        "start": START,
        "completion_commit": completion_commit()}
    CENSUS_RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "hard_bars", "wall_s")},
                     indent=1), flush=True)
    print(json.dumps({"structural": {
        a: {"histogram": census[a]["histogram"],
            "single_step_max": census[a]["single_step_max"]}
        for a in ARMS}}, indent=1), flush=True)
    print("[svpforder:census] DONE", flush=True)


def state_bytes(seed):
    torch.manual_seed(seed)
    m = build_model(VOCAB, ctx=4096)
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in
                sorted(m.state_dict().items())}, buf)
    return buf.getvalue()


def make_init():
    gate(CENSUS_RECEIPT.exists(), "CENSUS NOT RUN")
    cr = json.loads(CENSUS_RECEIPT.read_text())
    gate(cr["verdict"] == "CENSUS QUALIFIED",
         "CENSUS NOT QUALIFIED")
    for pth, h in cr["start"]["file_sha256"].items():
        gate(fsha(pth) == h, f"CENSUS STALE {pth}")
    gate(not INIT_CK.exists(), "INIT ALREADY EXISTS")
    b1 = state_bytes(SEED)
    b2 = state_bytes(SEED)
    gate(b1 == b2, "INIT NONDETERMINISTIC")
    gate(hashlib.sha256(b1).hexdigest() == INIT_SHA,
         "INIT SHA != PIN")
    INIT_CK.write_bytes(b1)
    print(f"[svpforder] init written {INIT_SHA}", flush=True)


def ensure_init():
    gate(INIT_CK.exists(), "INIT ABSENT")
    gate(fsha(INIT_CK) == INIT_SHA, "INIT != PINNED SHA")
    return INIT_SHA


def make_batch(rows, view, dev):
    key = {"CANONICAL": "_C", "PARAM_FIRST": "_P"}[view]
    enc = [(r["_pre"], r[key]) for r in rows]
    L = max(len(p) + len(c) for p, c in enc)
    ids = torch.full((len(enc), L), TOK.pad_id)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    lens = []
    for i, (p, c) in enumerate(enc):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        mask[i, len(p):len(p) + len(c)] = 1
        lens.append(len(p) + len(c))
    return ids.to(dev), mask.to(dev), lens


def load_arm_models(dev, init_sha):
    gate(fsha(INIT_CK) == init_sha, "INIT MUTATED")
    sd = torch.load(INIT_CK, weights_only=True)
    arms = {}
    n_params = None
    n_bitwise = 0
    for a in ARMS:
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(sd)
        for k, v in m.state_dict().items():
            gate(torch.equal(v.cpu(), sd[k]), f"INIT DRIFT {a}")
            n_bitwise += 1
        np_ = sum(p.numel() for p in m.parameters())
        gate(np_ == 19142016, "PARAM COUNT")
        if n_params is None:
            n_params = np_
        gate(np_ == n_params, "PARAM DRIFT")
        arms[a] = m.to(dev)
    ptrs = [{p.data_ptr() for p in arms[a].parameters()}
            for a in ARMS]
    gate(not (ptrs[0] & ptrs[1]), "SHARED STORAGE")
    return arms, n_params, n_bitwise


def run(plan, rows_by_id, dev, init_sha, n_steps, tag):
    arms, n_params, n_bitwise = load_arm_models(dev, init_sha)
    opts = {a: make_opt(arms[a], n_steps) for a in ARMS}
    losses = {a: [] for a in ARMS}
    gnorms = {a: [] for a in ARMS}
    tok_padded = {a: 0 for a in ARMS}
    tok_cont = {a: 0 for a in ARMS}
    tok_nonpad = {a: 0 for a in ARMS}
    order_counts = {0: 0, 1: 0}
    wall = {a: 0.0 for a in ARMS}
    lockstep_checks = 0
    shape_checks = 0
    peak_mps = 0
    for step, (e, bi, ids_) in enumerate(plan[:n_steps]):
        rows = [rows_by_id[i] for i in ids_]
        batches = {}
        for a in ARMS:
            ids, mask, lens = make_batch(rows, a, dev)
            batches[a] = (ids, mask, lens)
        idC, mkC, lnC = batches["CANONICAL"]
        idP, mkP, lnP = batches["PARAM_FIRST"]
        gate(idC.shape == idP.shape, f"SHAPE DRIFT step {step}")
        gate(lnC == lnP, f"ROW LEN DRIFT step {step}")
        gate(int(mkC.sum()) == int(mkP.sum()),
             f"MASK DRIFT step {step}")
        gate(int((idC != TOK.pad_id).sum())
             == int((idP != TOK.pad_id).sum()),
             f"NONPAD DRIFT step {step}")
        shape_checks += 1
        order = ORDER[step % 2]
        order_counts[step % 2] += 1
        for a in order:
            t0 = time.monotonic()
            ids, mask, _ = batches[a]
            tok_padded[a] += int(ids.numel())
            tok_cont[a] += int(mask.sum())
            tok_nonpad[a] += int((ids != TOK.pad_id).sum())
            opt, sched = opts[a]
            l, gn = train_step(arms[a], opt, sched, ids, mask)
            losses[a].append(l)
            gnorms[a].append(gn)
            wall[a] += time.monotonic() - t0
        peak_mps = max(peak_mps,
                       torch.mps.current_allocated_memory())
        counts = {a: opts[a][1].last_epoch for a in ARMS}
        gate(len(set(counts.values())) == 1,
             f"STEP-COUNT DRIFT {counts}")
        lockstep_checks += 1
        if step % 200 == 0 or step == n_steps - 1:
            print(f"[svpforder:{tag}] step {step}/{n_steps} " +
                  " ".join(f"{a[0]}={losses[a][-1]:.4f}"
                           for a in ARMS), flush=True)
    finite_checked = 0
    for a in ARMS:
        gate(len(losses[a]) == n_steps, f"LOSS COUNT {a}")
        gate(all(math.isfinite(x) for x in losses[a]),
             f"NON-FINITE {a}")
        finite_checked += len(losses[a])
        gate(opts[a][1].last_epoch == n_steps - 1,
             f"SCHED TERMINAL {a}")
    gate(tok_padded["CANONICAL"] == tok_padded["PARAM_FIRST"],
         "PADDED TOTAL DRIFT")
    gate(tok_cont["CANONICAL"] == tok_cont["PARAM_FIRST"],
         "CONT TOTAL DRIFT")
    gate(tok_nonpad["CANONICAL"] == tok_nonpad["PARAM_FIRST"],
         "NONPAD TOTAL DRIFT")
    stats = {"lockstep_checks": lockstep_checks,
             "shape_checks": shape_checks,
             "finite_losses_checked": finite_checked,
             "bitwise_tensors_compared": n_bitwise,
             "sched_terminal": {a: int(opts[a][1].last_epoch)
                                for a in ARMS},
             "peak_mps_allocated_bytes": peak_mps}
    return arms, losses, gnorms, tok_padded, tok_cont, \
        tok_nonpad, order_counts, wall, n_params, stats


def main():
    if CENSUS:
        gate(not (MAKE_INIT or PRODUCTION), "MODE CONFLICT")
        run_census()
        return 0
    if MAKE_INIT:
        gate(not PRODUCTION, "MAKE_INIT WITH PRODUCTION")
        make_init()
        return 0
    # smoke and production both require a green census
    gate(CENSUS_RECEIPT.exists(), "CENSUS NOT RUN")
    cr = json.loads(CENSUS_RECEIPT.read_text())
    gate(cr["verdict"] == "CENSUS QUALIFIED",
         "CENSUS NOT QUALIFIED")
    for pth, h in cr["start"]["file_sha256"].items():
        gate(fsha(pth) == h, f"CENSUS STALE {pth}")
    if PRODUCTION:
        for p in list(CKS.values()) + [RECEIPT]:
            if p.exists():
                raise SystemExit(f"REFUSING: {p} exists")
    else:
        if SMOKE_RECEIPT.exists():
            raise SystemExit(f"REFUSING: {SMOKE_RECEIPT} exists")
        for p in list(CKS.values()) + [RECEIPT]:
            if p.exists():
                raise SystemExit(
                    f"REFUSING: production path {p} exists")
    START = start_provenance(
        ["scratch/mathworld1_svpforder.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    gate(torch.backends.mps.is_available(), "MPS UNAVAILABLE")
    dev = torch.device("mps")
    init_sha = ensure_init()
    rows = load_rows()
    rows_by_id = {r["row_id"]: r for r in rows}
    gate(len(rows_by_id) == N_ROWS, "DUPLICATE ROW_ID")
    plan = batch_plan(list(rows_by_id))
    gate(len(plan) == TOTAL_STEPS, "PLAN LEN")
    plan_sha = sha_b(json.dumps(plan).encode())
    gate(plan_sha == PLAN_SHA, "PLAN SHA")
    for e in range(EPOCHS):
        eb = [p for p in plan if p[0] == e]
        gate(len(eb) == 2340, f"EPOCH BATCHES e{e}")
        gate(len(eb[-1][2]) == 12, f"TAIL e{e}")

    t0 = time.time()
    if not PRODUCTION:
        maxlen = {i: len(r["_pre"]) for i, r in rows_by_id.items()}
        longest_step = max(
            range(2340),
            key=lambda s: max(maxlen[i] for i in plan[s][2]))
        ordinary = [s for s in (0, 1, 2)
                    if s != longest_step][:2]
        mini = ([plan[longest_step]]
                + [plan[s] for s in ordinary]
                + [plan[2339]])
        n = len(mini)
        gate(n == 4, "SMOKE PLAN SIZE")
        arms, losses, gnorms, tp, tc, tn, oc, wall, n_params, \
            stats = run(mini, rows_by_id, dev, init_sha, n,
                        "smoke")
        n_rows_smoke = sum(len(p[2]) for p in mini)
        receipt = {
            "mode": "smoke", "seed": SEED, "vocab": VOCAB,
            "n_params": n_params, "init_sha": init_sha,
            "plan_sha": plan_sha, "steps": n,
            "smoke_plan_steps": [longest_step] + ordinary
            + [2339],
            "losses": losses, "grad_norms": gnorms,
            "tokens_continuation": tc,
            "order_counts": {str(k): v for k, v in oc.items()},
            "bars": {
                "INIT_BITWISE_ALL_ARMS":
                    stats["bitwise_tensors_compared"] == 2 * 59,
                "VOCAB_340": VOCAB == 340,
                "PARAM_EQUAL": n_params == 19142016,
                "PLAN_SHA_MATCH": plan_sha == PLAN_SHA,
                "BOTH_ARMS_T9":
                    tc["CANONICAL"] == tc["PARAM_FIRST"]
                    == 9 * n_rows_smoke,
                "BOTH_ORDERS_EXERCISED":
                    oc[0] >= 1 and oc[1] >= 1,
                "SHAPE_GATES_EVERY_STEP":
                    stats["shape_checks"] == n,
                "FINITE":
                    stats["finite_losses_checked"] == 2 * n,
                "STEP_LOCKSTEP": stats["lockstep_checks"] == n,
                "NO_PRODUCTION_PATHS": not any(
                    p.exists() for p in CKS.values())
                and not RECEIPT.exists(),
            },
            "wall_s": round(time.time() - t0, 1),
            "start": START,
            "completion_commit": completion_commit()}
        SMOKE_RECEIPT.write_text(json.dumps(receipt, indent=1))
        print(json.dumps(receipt["bars"], indent=1), flush=True)
        return 0

    gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
    sr = json.loads(SMOKE_RECEIPT.read_text())
    gate(all(sr["bars"].values()), "SMOKE NOT GREEN")
    for pth, h in sr["start"]["file_sha256"].items():
        gate(fsha(pth) == h, f"SMOKE STALE {pth}")

    arms, losses, gnorms, tp, tc, tn, oc, wall, n_params, \
        stats = run(plan, rows_by_id, dev, init_sha,
                    TOTAL_STEPS, "prod")
    for a in ARMS:
        gate(len(losses[a]) == TOTAL_STEPS, f"INCOMPLETE {a}")
        gate(tc[a] == TOK_CONT_EXPECT, f"CONT TOKENS {a}")
    gate(oc[0] == 3510 and oc[1] == 3510,
         f"ORDER COUNTS {dict(oc)}")
    gate(stats["bitwise_tensors_compared"] == 2 * 59,
         "INIT BITWISE COUNT")
    for a in ARMS:
        torch.save({k: v.cpu() for k, v in
                    sorted(arms[a].state_dict().items())},
                   CKS[a])
    shas = {a: fsha(CKS[a]) for a in ARMS}
    gate(shas["CANONICAL"] != shas["PARAM_FIRST"],
         "IDENTICAL ARM CHECKPOINTS")
    epoch_mean = {a: [round(sum(losses[a][e*2340:(e+1)*2340])
                            / 2340, 5) for e in range(3)]
                  for a in ARMS}
    receipt = {
        "mode": "production", "seed": SEED, "vocab": VOCAB,
        "n_params": n_params,
        "init_sha": init_sha, "plan_sha": plan_sha,
        "total_steps": TOTAL_STEPS,
        "updates_per_arm": {a: len(losses[a]) for a in ARMS},
        "sched_terminal": stats["sched_terminal"],
        "finite_losses_checked":
            stats["finite_losses_checked"],
        "epoch_mean_loss": epoch_mean,
        "tokens_padded": tp, "tokens_continuation": tc,
        "tokens_nonpad": tn,
        "order_counts": {"CANONICAL_first": oc[0],
                         "PARAM_FIRST_first": oc[1]},
        "arm_wall_s_note": ("encode excluded; within-run-paired "
                            "only, never cross-run comparable"),
        "arm_wall_s": {a: round(wall[a], 1) for a in ARMS},
        "wall_s": round(time.time() - t0, 1),
        "losses": {a: [round(x, 5) for x in losses[a]]
                   for a in ARMS},
        "grad_norm_summary": {a: {
            "p50": round(sorted(gnorms[a])[len(gnorms[a]) // 2],
                         5),
            "max": round(max(gnorms[a]), 5)} for a in ARMS},
        "lockstep_checks": stats["lockstep_checks"],
        "shape_checks": stats["shape_checks"],
        "bitwise_tensors_compared":
            stats["bitwise_tensors_compared"],
        "peak_rss_mb": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024),
        "peak_mps_allocated_bytes":
            stats["peak_mps_allocated_bytes"],
        "env": {"torch": torch.__version__,
                "device": str(dev)},
        "perm": PERM, "inv": INV,
        "checkpoints": {str(CKS[a]): shas[a] for a in ARMS},
        "census_receipt_sha": fsha(CENSUS_RECEIPT),
        "pins": {PAIRED: fsha(PAIRED), AUG: fsha(AUG),
                 MANIFEST: fsha(MANIFEST),
                 str(INIT_CK): fsha(INIT_CK)},
        "smoke_receipt_sha": fsha(SMOKE_RECEIPT),
        "start": START, "completion_commit": completion_commit()}
    gate(fsha(INIT_CK) == INIT_SHA, "POST INIT PIN")
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print("[svpforder] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

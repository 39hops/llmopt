"""MATH-CYBER-1 SVP-GRID-THREE-ARM-BIRTH-18001 — the BIRTH stage
of the frozen permutation-replication protocol
(PERMUTATION-REPLICATION-PREREG-0, commit 7976613b): ONE shared
fresh seed-18001 initialization, THREE equal-compute arms over
the frozen GRID-DIET population (74,860 rows, batch plan
f55e9fee..., 7,020 updates/arm):
  FACTOR   = frozen factor code (unchanged)
  HASH_P1  = frozen P1 Feistel (unchanged)
  HASH_P2  = the immutable qualified P2 realization
             (svpp2qual; realization pin 952f332d..., re-derived
             and gated at entry — the salt/mapping can NEVER
             change)
ZERO evaluation scoring; no calibration/heldout/P-OUT access;
eval shas appear ONLY as opaque provenance literals. This is a
NEW three-arm instrument, not a mechanical sibling of the
two-arm births; its common training law (rows, plan, model,
optimizer, scheduler, masks, label censoring, per-batch
equal-compute gates, init pre-step, path-isolated smoke) is
carried from the booked BIRTH-17001 law with the two-arm order
law replaced by the frozen six-cycle.

SIX-CYCLE ORDER LAW: the prereg froze "cycle all six orderings
of {F, H1, H2} by global batch index mod 6" without fixing the
residue->ordering map; that implementation detail is resolved
HERE, before any initialization, as the LEXICOGRAPHIC cycle
(disclosed in the adopting prereg entry as an implementation
resolution, never outcome-based):
  0: F,H1,H2   1: F,H2,H1   2: H1,F,H2
  3: H1,H2,F   4: H2,F,H1   5: H2,H1,F
7,020 = 6 x 1,170 => every ordering exactly 1,170 times; every
arm in execution position 1/2/3 exactly 2,340 times each (all
gated).

EQUAL-COMPUTE GATES (per batch, INSTRUMENT FAILURE on any
mismatch, pairwise across all three arms): identical batch
tensor shape, per-row sequence lengths, target-mask totals,
padded totals, non-pad totals; all three arms consume the SAME
rows before the plan advances. Totals per arm: 7,020 updates;
2,021,220 continuation tokens exactly; identical padded/non-pad
totals; scheduler lockstep through terminal step. Zero
I1/t2-t3 labels re-gated at load. Init: dedicated pre-step
(SVPGB18_MAKE_INIT=1), double-build byte-identity, sha pinned
at design time, 3x59 bitwise init-equality gates, storage
disjointness across the three arms; no prior-seed weights
loaded anywhere.

Smoke (path-isolated; writes only its own receipt): six real
plan batches — the longest batch, ordinary batches, and the
epoch-0 12-row tail — with the smoke-local step index driving
the six-cycle so ALL SIX orderings execute at least once; all
three encode paths, backward, finite grads, clipping, optimizer
+ scheduler steps.

Paths (refuse-if-exists in production):
  checkpoints/svp_grid_init_s18001.pt   (pre-step, sha-gated)
  checkpoints/svp_grid_factor_s18001.pt
  checkpoints/svp_grid_hashp1_s18001.pt
  checkpoints/svp_grid_hashp2_s18001.pt
  logs/mathworld1/svpgbirth_s18001_receipt.json
Smoke: logs/mathworld1/smoke_svpgbirth18.json

    SVPGB18_MAKE_INIT=1 .venv/bin/python scratch/mathworld1_svpgbirth18.py
    .venv/bin/python scratch/mathworld1_svpgbirth18.py            (smoke)
    SVPGB18_PRODUCTION=1 .venv/bin/python scratch/mathworld1_svpgbirth18.py
"""
import hashlib
import io
import json
import math
import os
import resource
import sys
import time
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
                                        hash_decode,
                                        hash_symbols, in_domain)
from scratch.mathworld1_svpp2qual import (hash2_decode,  # noqa: E402
                                          hash2_symbols)

SEED = 18001
VOCAB = 340
CODE_BASE = 332
SEQ_CAP = 512
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
P2_QUAL_RECEIPT = "logs/mathworld1/svpp2qual/svpp2qual_receipt.json"
P2_QUAL_RECEIPT_SHA = (
    "47309f22e2be3fba57a28ea0937c985d9ae616121b4cfff608c1"
    "9371246c2337")
P2_REALIZATION_SHA = (
    "952f332da4e25961b2dd52c786902e74ba4b33bbf8413f88496a"
    "0df952450ba9")
# opaque provenance literals ONLY (frozen eval artifact shas,
# receipt copy-through; nothing here opens those artifacts):
EVAL_PROVENANCE = {
    "calibration_sha": ("af1a4aa1df7bf3224745e91a90e1a77c36e5"
                        "c54f7ff9b08509794d0fb7978db3"),
    "heldout_sha": ("a3f6103b3733d909281849dcb3fd6ba9fba3891f"
                    "2014bec13881b4509df46ddb"),
    "prereg_commit": "7976613b3fef18f54d953ac0404f377e5c74031b"}
EPOCHS = 3
TOTAL_STEPS = 7020
N_ROWS = 74860
TOK_CONT_EXPECT = 9 * N_ROWS * EPOCHS  # 2,021,220 per arm
ARMS = ["FACTOR", "HASH_P1", "HASH_P2"]
# frozen lexicographic six-cycle (implementation resolution of
# the prereg's "cycle all six orderings by batch index mod 6")
ORDER6 = {0: ["FACTOR", "HASH_P1", "HASH_P2"],
          1: ["FACTOR", "HASH_P2", "HASH_P1"],
          2: ["HASH_P1", "FACTOR", "HASH_P2"],
          3: ["HASH_P1", "HASH_P2", "FACTOR"],
          4: ["HASH_P2", "FACTOR", "HASH_P1"],
          5: ["HASH_P2", "HASH_P1", "FACTOR"]}
INIT_CK = Path("checkpoints/svp_grid_init_s18001.pt")
# pinned at design time by two independent byte-equal builds,
# 2026-08-29:
INIT_SHA = ("a7bb5b8839e78560b6648f7471c03827796309c990514fd4"
            "bdce949b00299fc4")
CKS = {"FACTOR": Path("checkpoints/svp_grid_factor_s18001.pt"),
       "HASH_P1": Path("checkpoints/svp_grid_hashp1_s18001.pt"),
       "HASH_P2": Path("checkpoints/svp_grid_hashp2_s18001.pt")}
RECEIPT = Path("logs/mathworld1/svpgbirth_s18001_receipt.json")
SMOKE_RECEIPT = Path("logs/mathworld1/smoke_svpgbirth18.json")
MAKE_INIT = os.environ.get("SVPGB18_MAKE_INIT") == "1"
TOK = ActionGCTok()
PRODUCTION = os.environ.get("SVPGB18_PRODUCTION") == "1"


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rederive_p2_realization_sha():
    """Re-derive the P2 realization pin over the canonical
    enumeration and gate it against the qualified value — the
    immutability check on P2 before any training."""
    stream = hashlib.sha256()
    for r in OPCODE_ORDER:
        for sk in ("W", "I"):
            for so in range(-1, ORD_MAX + 1):
                for pk in ("none", "u_choice", "term_index"):
                    for pi in range(-1, ORD_MAX + 1):
                        stream.update(bytes(
                            hash2_symbols(r, sk, so, pk, pi)))
    return stream.hexdigest()


def state_bytes(seed):
    torch.manual_seed(seed)
    m = build_model(VOCAB, ctx=4096)
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in
                sorted(m.state_dict().items())}, buf)
    return buf.getvalue()


def make_init():
    gate(not INIT_CK.exists(), "INIT ALREADY EXISTS")
    b1 = state_bytes(SEED)
    b2 = state_bytes(SEED)
    gate(b1 == b2, "INIT NONDETERMINISTIC")
    gate(hashlib.sha256(b1).hexdigest() == INIT_SHA,
         "INIT SHA != PIN")
    INIT_CK.write_bytes(b1)
    print(f"[svpgb18] init written {INIT_SHA}", flush=True)


def ensure_init():
    gate(INIT_CK.exists(),
         "INIT ABSENT (run SVPGB18_MAKE_INIT=1 pre-step)")
    gate(fsha(INIT_CK) == INIT_SHA, "INIT != PINNED SHA")
    return INIT_SHA


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
        fs = factor_symbols(*tup)
        h1 = hash_symbols(*tup)
        h2 = hash2_symbols(*tup)
        gate(factor_decode(fs) == tup, "F INVERSE")
        gate(hash_decode(h1) == tup, "P1 INVERSE")
        gate(hash2_decode(h2) == tup, "P2 INVERSE")
        r["_F"] = [CODE_BASE + s for s in fs] + [TOK.eos_id]
        r["_H1"] = [CODE_BASE + s for s in h1] + [TOK.eos_id]
        r["_H2"] = [CODE_BASE + s for s in h2] + [TOK.eos_id]
        gate(len(r["_F"]) == len(r["_H1"]) == len(r["_H2"])
             == 9, "T!=9")
        r["_pre"] = TOK.encode(
            f"Current: {r['cur']}\nHints: none\nStep: ")
        gate(len(r["_pre"]) + 9 <= SEQ_CAP, "SEQ CAP")
    gate(heldout_hits == 0,
         f"HELD-OUT LABELS IN TRAINING {heldout_hits}")
    return rows


def make_batch(rows, view, dev):
    key = {"FACTOR": "_F", "HASH_P1": "_H1",
           "HASH_P2": "_H2"}[view]
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
        gate(np_ == n_params, "PARAM COUNT DRIFT")
        arms[a] = m.to(dev)
    ptrs = [{p.data_ptr() for p in arms[a].parameters()}
            for a in ARMS]
    for i in range(len(ARMS)):
        for j in range(i + 1, len(ARMS)):
            gate(not (ptrs[i] & ptrs[j]), "SHARED STORAGE")
    return arms, n_params, n_bitwise


def run(plan, rows_by_id, dev, init_sha, n_steps, tag):
    arms, n_params, n_bitwise = load_arm_models(dev, init_sha)
    opts = {a: make_opt(arms[a], n_steps) for a in ARMS}
    losses = {a: [] for a in ARMS}
    gnorms = {a: [] for a in ARMS}
    tok_padded = {a: 0 for a in ARMS}
    tok_cont = {a: 0 for a in ARMS}
    tok_nonpad = {a: 0 for a in ARMS}
    order_counts = {i: 0 for i in range(6)}
    pos_counts = {a: [0, 0, 0] for a in ARMS}
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
        # equal-compute gates, pairwise across all three arms
        ref_a = ARMS[0]
        idR, mkR, lnR = batches[ref_a]
        npadR = int((idR != TOK.pad_id).sum())
        mkRs = int(mkR.sum())
        for a in ARMS[1:]:
            idA, mkA, lnA = batches[a]
            gate(idA.shape == idR.shape,
                 f"SHAPE DRIFT {a} step {step}")
            gate(lnA == lnR, f"ROW LEN DRIFT {a} step {step}")
            gate(int(mkA.sum()) == mkRs,
                 f"TARGET MASK DRIFT {a} step {step}")
            gate(int((idA != TOK.pad_id).sum()) == npadR,
                 f"NONPAD DRIFT {a} step {step}")
        shape_checks += 1
        order = ORDER6[step % 6]
        order_counts[step % 6] += 1
        for pos, a in enumerate(order):
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
            pos_counts[a][pos] += 1
        peak_mps = max(peak_mps,
                       torch.mps.current_allocated_memory())
        counts = {a: opts[a][1].last_epoch for a in ARMS}
        gate(len(set(counts.values())) == 1,
             f"STEP-COUNT DRIFT {counts}")
        lockstep_checks += 1
        if step % 200 == 0 or step == n_steps - 1:
            print(f"[svpgb18:{tag}] step {step}/{n_steps} " +
                  " ".join(f"{a}={losses[a][-1]:.4f}"
                           for a in ARMS), flush=True)
    finite_checked = 0
    for a in ARMS:
        gate(len(losses[a]) == n_steps, f"LOSS COUNT {a}")
        gate(all(math.isfinite(x) for x in losses[a]),
             f"NON-FINITE {a}")
        finite_checked += len(losses[a])
        gate(opts[a][1].last_epoch == n_steps - 1,
             f"SCHED TERMINAL {a}")
    ref = ARMS[0]
    for a in ARMS[1:]:
        gate(tok_padded[a] == tok_padded[ref],
             f"PADDED TOTAL DRIFT {a}")
        gate(tok_cont[a] == tok_cont[ref],
             f"CONT TOTAL DRIFT {a}")
        gate(tok_nonpad[a] == tok_nonpad[ref],
             f"NONPAD TOTAL DRIFT {a}")
    stats = {"lockstep_checks": lockstep_checks,
             "shape_checks": shape_checks,
             "finite_losses_checked": finite_checked,
             "bitwise_tensors_compared": n_bitwise,
             "sched_terminal": {a: int(opts[a][1].last_epoch)
                                for a in ARMS},
             "peak_mps_allocated_bytes": peak_mps}
    return arms, losses, gnorms, tok_padded, tok_cont, \
        tok_nonpad, order_counts, pos_counts, wall, n_params, \
        stats


def main():
    if MAKE_INIT:
        gate(not PRODUCTION, "MAKE_INIT WITH PRODUCTION")
        make_init()
        return 0
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
    # P2 immutability: pin the qualification receipt bytes AND
    # re-derive the realization sha before anything trains
    gate(fsha(P2_QUAL_RECEIPT) == P2_QUAL_RECEIPT_SHA,
         "P2 QUAL RECEIPT PIN")
    qr = json.loads(Path(P2_QUAL_RECEIPT).read_text())
    gate(qr["p2_realization_sha"] == P2_REALIZATION_SHA,
         "P2 RECEIPT REALIZATION MISMATCH")
    gate(rederive_p2_realization_sha() == P2_REALIZATION_SHA,
         "P2 REALIZATION DRIFT")
    START = start_provenance(
        ["scratch/mathworld1_svpgbirth18.py",
         "scratch/mathworld1_svpp2qual.py",
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
        # smoke mini-plan: 6 real batches so the smoke-local
        # step index exercises ALL SIX orderings — the longest
        # batch, four ordinary batches, the epoch-0 12-row tail
        maxlen = {i: len(r["_pre"]) for i, r in rows_by_id.items()}
        longest_step = max(
            range(2340),
            key=lambda s: max(maxlen[i] for i in plan[s][2]))
        ordinary = [s for s in (0, 1, 2, 3, 4)
                    if s != longest_step and s != 2339][:4]
        gate(longest_step != 2339 or 2339 not in ordinary,
             "SMOKE TAIL DUP")
        mini = ([plan[longest_step]]
                + [plan[s] for s in ordinary]
                + [plan[2339]])
        n = len(mini)
        gate(n == 6, "SMOKE PLAN SIZE")
        arms, losses, gnorms, tp, tc, tn, oc, pc, wall, \
            n_params, stats = run(mini, rows_by_id, dev,
                                  init_sha, n, "smoke")
        n_rows_smoke = sum(len(p[2]) for p in mini)
        receipt = {
            "mode": "smoke", "seed": SEED, "vocab": VOCAB,
            "n_params": n_params, "init_sha": init_sha,
            "plan_sha": plan_sha, "steps": n,
            "smoke_plan_steps": [longest_step] + ordinary
            + [2339],
            "losses": losses, "grad_norms": gnorms,
            "tokens_padded": tp,
            "tokens_continuation": tc,
            "tokens_nonpad": tn,
            "order_counts": {str(k): v for k, v in oc.items()},
            "position_counts": pc,
            "bars": {
                "INIT_BITWISE_ALL_ARMS":
                    stats["bitwise_tensors_compared"] == 3 * 59,
                "VOCAB_340": VOCAB == 340,
                "PARAM_EQUAL": n_params == 19142016,
                "PLAN_SHA_MATCH": plan_sha == PLAN_SHA,
                "ALL_ARMS_T9":
                    tc["FACTOR"] == tc["HASH_P1"]
                    == tc["HASH_P2"] == 9 * n_rows_smoke,
                "ALL_SIX_ORDERINGS_EXERCISED":
                    all(oc[i] >= 1 for i in range(6)),
                "SHAPE_GATES_EVERY_STEP":
                    stats["shape_checks"] == n,
                "FINITE":
                    stats["finite_losses_checked"] == 3 * n,
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

    arms, losses, gnorms, tp, tc, tn, oc, pc, wall, n_params, \
        stats = run(plan, rows_by_id, dev, init_sha,
                    TOTAL_STEPS, "prod")
    # completion gates BEFORE checkpoint writes
    for a in ARMS:
        gate(len(losses[a]) == TOTAL_STEPS, f"INCOMPLETE {a}")
        gate(tc[a] == TOK_CONT_EXPECT, f"CONT TOKENS {a} {tc[a]}")
        gate(pc[a] == [2340, 2340, 2340],
             f"POSITION CENSUS {a} {pc[a]}")
    gate(all(oc[i] == 1170 for i in range(6)),
         f"ORDER CENSUS {dict(oc)}")
    gate(stats["bitwise_tensors_compared"] == 3 * 59,
         "INIT BITWISE COUNT")
    # post-run P2 immutability re-gates (derived value kept for
    # the receipt)
    gate(fsha(P2_QUAL_RECEIPT) == P2_QUAL_RECEIPT_SHA,
         "POST P2 QUAL RECEIPT PIN")
    p2_post_sha = rederive_p2_realization_sha()
    gate(p2_post_sha == P2_REALIZATION_SHA,
         "POST P2 REALIZATION DRIFT")
    for a in ARMS:
        torch.save({k: v.cpu() for k, v in
                    sorted(arms[a].state_dict().items())}, CKS[a])
    shas = {a: fsha(CKS[a]) for a in ARMS}
    gate(len(set(shas.values())) == 3,
         "NON-DISTINCT ARM CHECKPOINTS")
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
        "order_counts": {str(k): v for k, v in oc.items()},
        "position_counts": pc,
        "order6_law": {str(k): v for k, v in ORDER6.items()},
        "arm_wall_s_note": ("encode excluded from arm wall; "
                            "within-run-paired only, never "
                            "cross-run comparable"),
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
        "checkpoints": {str(CKS[a]): shas[a] for a in ARMS},
        "p2": {"qual_receipt_sha": fsha(P2_QUAL_RECEIPT),
               "realization_sha_rederived_post": p2_post_sha,
               "realization_sha_pin": P2_REALIZATION_SHA},
        "pins": {PAIRED: fsha(PAIRED), AUG: fsha(AUG),
                 MANIFEST: fsha(MANIFEST),
                 str(INIT_CK): fsha(INIT_CK)},
        "frozen_eval_provenance_literals": EVAL_PROVENANCE,
        "smoke_receipt_sha": fsha(SMOKE_RECEIPT),
        "start": START, "completion_commit": completion_commit()}
    gate(fsha(INIT_CK) == INIT_SHA, "POST INIT PIN")
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print("[svpgb18] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

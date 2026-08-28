"""MATH-CYBER-1 SVP-GRID-BIRTH-15001 — the frozen PAIRED
FACTOR/HASH production birth over the GRID-DIET combined
population (74,860 rows = 73,324 natural + 1,536 balanced
covered-grid augmentation), one target-blind batch plan (7,020
updates), one frozen optimization law, ONE bit-identical shared
vocab-340 init (seed 15001). ZERO evaluation scoring; this file
has no path or string dependency on any evaluation artifact —
the two frozen eval shas appear ONLY as opaque provenance
literals copied into the receipt (EVAL_PROVENANCE below), never
as paths, and nothing here opens, parses, or counts them.

Law by IMPORT from the frozen svpbirth driver: batch_plan,
make_opt, train_step, gate (finite-loss/finite-grad abort
before mutation, clip error_if_nonfinite, per-arm AdamW +
OneCycle, scheduler lockstep). FACTOR/HASH encodings from the
qualified svpcode module; code atoms ids 332..339; targets are
exactly 8 code atoms + EOS (T=9, gated per row with exact
semantic inverse). CANONICAL is NOT trained under this GO.

TWO-ARM ORDER LAW (frozen): even global batch index -> FACTOR
then HASH; odd -> HASH then FACTOR. 7,020 batches => 3,510
FACTOR-first + 3,510 HASH-first, gated exactly. Both arms
consume the same batch before the plan advances.

EQUAL-COMPUTE GATES (per batch, INSTRUMENT FAILURE on any
mismatch): identical batch tensor shape F v H, identical
per-row sequence lengths, identical padded-token totals,
identical non-pad (attention) totals, identical target-mask
totals. Totals gated at end: continuation tokens per arm
2,021,220 = 9 * 74,860 * 3 exactly; updates 7,020; 2,340
batches/epoch; 12-row tail per epoch; every training sequence
<= 512 tokens (measured max 229).

LABEL RECHECK (frozen bytes, not regeneration): the combined
population carries ZERO rows labeled i_unprod I1 term_index 2
or 3, re-gated at load.

Paths (refuse-if-exists in production):
  checkpoints/svp_grid_init_s15001.pt          (shared init)
  checkpoints/svp_grid_factor_s15001.pt
  checkpoints/svp_grid_hash_s15001.pt
  logs/mathworld1/svpgbirth_s15001_receipt.json
Smoke (isolated): logs/mathworld1/smoke_svpgbirth.json — both
arms resident, the longest real training batch + an ordinary
batch + the epoch-0 12-row tail, shared-init equality, real
encode paths, backward, finite grads, clipping, optimizer +
scheduler steps, both arm orders; smoke may MATERIALIZE the
sha-pinned shared init (create-if-absent, double-build gated,
never overwritten) but writes no arm checkpoint and no
production receipt.

    .venv/bin/python scratch/mathworld1_svpgbirth.py            (smoke)
    SVPGB_PRODUCTION=1 .venv/bin/python scratch/mathworld1_svpgbirth.py
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
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpbirth import (batch_plan,  # noqa: E402
                                         gate, make_opt, sha_b,
                                         train_step)
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        factor_symbols,
                                        hash_decode, hash_symbols,
                                        in_domain)

SEED = 15001
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
# opaque provenance literals ONLY (frozen eval artifact shas,
# receipt copy-through; nothing here opens those artifacts):
EVAL_PROVENANCE = {
    "calibration_sha": ("90421e8b9bcab38648a20e7cd24f48e2d54c"
                        "dcd20a437b18b8f54e9e3f4d9977"),
    "heldout_sha": ("3346bd84e90751d43937ea9b15a47fbb1d8f0273"
                    "92139b0efba41e2a91eeffdd")}
EPOCHS = 3
TOTAL_STEPS = 7020
N_ROWS = 74860
TOK_CONT_EXPECT = 9 * N_ROWS * EPOCHS  # 2,021,220
ARMS = ["FACTOR", "HASH"]
ORDER = {0: ["FACTOR", "HASH"], 1: ["HASH", "FACTOR"]}
INIT_CK = Path("checkpoints/svp_grid_init_s15001.pt")
# pinned at double-build-gated creation, 2026-08-28:
INIT_SHA = ("4b085795f9e8b0be874cabdc6d58899a2a4554f8b42cb711"
            "f5154614e41797bc")
CKS = {"FACTOR": Path("checkpoints/svp_grid_factor_s15001.pt"),
       "HASH": Path("checkpoints/svp_grid_hash_s15001.pt")}
RECEIPT = Path("logs/mathworld1/svpgbirth_s15001_receipt.json")
SMOKE_RECEIPT = Path("logs/mathworld1/smoke_svpgbirth.json")
TOK = ActionGCTok()
PRODUCTION = os.environ.get("SVPGB_PRODUCTION") == "1"


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def state_bytes(seed):
    torch.manual_seed(seed)
    m = build_model(VOCAB, ctx=4096)
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in
                sorted(m.state_dict().items())}, buf)
    return buf.getvalue()


def ensure_init():
    if not INIT_CK.exists():
        b1 = state_bytes(SEED)
        b2 = state_bytes(SEED)
        gate(b1 == b2, "INIT NONDETERMINISTIC")
        gate(hashlib.sha256(b1).hexdigest() == INIT_SHA,
             "INIT SHA != PIN")
        INIT_CK.write_bytes(b1)
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
        hs = hash_symbols(*tup)
        gate(factor_decode(fs) == tup, "F INVERSE")
        gate(hash_decode(hs) == tup, "H INVERSE")
        r["_F"] = [CODE_BASE + s for s in fs] + [TOK.eos_id]
        r["_H"] = [CODE_BASE + s for s in hs] + [TOK.eos_id]
        gate(len(r["_F"]) == 9 and len(r["_H"]) == 9, "T!=9")
        r["_pre"] = TOK.encode(
            f"Current: {r['cur']}\nHints: none\nStep: ")
        gate(len(r["_pre"]) + 9 <= SEQ_CAP, "SEQ CAP")
    gate(heldout_hits == 0,
         f"HELD-OUT LABELS IN TRAINING {heldout_hits}")
    return rows


def make_batch(rows, view, dev):
    enc = [(r["_pre"], r["_F"] if view == "FACTOR" else r["_H"])
           for r in rows]
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
        # equal-compute gates F v H, per batch
        idF, mkF, lnF = batches["FACTOR"]
        idH, mkH, lnH = batches["HASH"]
        gate(idF.shape == idH.shape, f"SHAPE DRIFT step {step}")
        gate(lnF == lnH, f"ROW LEN DRIFT step {step}")
        gate(int(mkF.sum()) == int(mkH.sum()),
             f"TARGET MASK DRIFT step {step}")
        npadF = int((idF != TOK.pad_id).sum())
        npadH = int((idH != TOK.pad_id).sum())
        gate(npadF == npadH, f"NONPAD DRIFT step {step}")
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
            print(f"[svpgb:{tag}] step {step}/{n_steps} " +
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
    gate(tok_padded["FACTOR"] == tok_padded["HASH"],
         "PADDED TOTAL DRIFT")
    gate(tok_cont["FACTOR"] == tok_cont["HASH"],
         "CONT TOTAL DRIFT")
    gate(tok_nonpad["FACTOR"] == tok_nonpad["HASH"],
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
        ["scratch/mathworld1_svpgbirth.py",
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
        # smoke mini-plan: longest real batch, ordinary batch 0,
        # epoch-0 12-row tail — real plan batches, both orders
        maxlen = {i: len(r["_pre"]) for i, r in rows_by_id.items()}
        longest_step = max(
            range(2340),
            key=lambda s: max(maxlen[i] for i in plan[s][2]))
        mini = [plan[longest_step], plan[0], plan[2339]]
        n = len(mini)
        arms, losses, gnorms, tp, tc, tn, oc, wall, n_params, \
            stats = run(mini, rows_by_id, dev, init_sha, n,
                        "smoke")
        n_rows_smoke = sum(len(p[2]) for p in mini)
        receipt = {
            "mode": "smoke", "seed": SEED, "vocab": VOCAB,
            "n_params": n_params, "init_sha": init_sha,
            "plan_sha": plan_sha, "steps": n,
            "smoke_plan_steps": [longest_step, 0, 2339],
            "losses": losses, "grad_norms": gnorms,
            "tokens_padded": tp,
            "tokens_continuation": tc,
            "tokens_nonpad": tn,
            "order_counts": {str(k): v for k, v in oc.items()},
            "bars": {
                "INIT_BITWISE_ALL_ARMS":
                    stats["bitwise_tensors_compared"] == 2 * 59,
                "VOCAB_340": VOCAB == 340,
                "PARAM_EQUAL": n_params == 19142016,
                "PLAN_SHA_MATCH": plan_sha == PLAN_SHA,
                "FH_T9":
                    tc["FACTOR"] == tc["HASH"]
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
    # completion gates BEFORE checkpoint writes
    for a in ARMS:
        gate(len(losses[a]) == TOTAL_STEPS, f"INCOMPLETE {a}")
        gate(tc[a] == TOK_CONT_EXPECT, f"CONT TOKENS {a} {tc[a]}")
    gate(oc[0] == 3510 and oc[1] == 3510,
         f"ORDER COUNTS {dict(oc)}")
    gate(stats["bitwise_tensors_compared"] == 2 * 59,
         "INIT BITWISE COUNT")
    for a in ARMS:
        torch.save({k: v.cpu() for k, v in
                    sorted(arms[a].state_dict().items())}, CKS[a])
    gate(fsha(CKS["FACTOR"]) != fsha(CKS["HASH"]),
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
        "order_counts": {"FACTOR_first": oc[0],
                         "HASH_first": oc[1]},
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
        "checkpoints": {str(CKS[a]): fsha(CKS[a]) for a in ARMS},
        "pins": {PAIRED: fsha(PAIRED), AUG: fsha(AUG),
                 MANIFEST: fsha(MANIFEST),
                 str(INIT_CK): fsha(INIT_CK)},
        "frozen_eval_provenance_literals": EVAL_PROVENANCE,
        "smoke_receipt_sha": fsha(SMOKE_RECEIPT),
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print("[svpgb] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

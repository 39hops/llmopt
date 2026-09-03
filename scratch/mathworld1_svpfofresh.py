"""MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-FRESH-SEED-0
fresh seeded paired training realizations (RESULTS L65695).

Seed-parameterized adaptation of the seed-20001 replication
instrument scratch/mathworld1_svpforepl.py, ADOPT-NOT-FORK in the
strongest form: this file imports that module and EXECUTES ITS
OWN FUNCTIONS for everything scientific (load_rows: corpus pins,
row gates, encoders, T = 9, SEQ_CAP, heldout-leak census;
state_bytes: the init procedure; run: the paired lockstep
training loop with every per-step gate; batch_plan / make_opt /
train_step through it). The seed-20001 source file is never
edited, so the structural census receipt's sha over it stays
valid. The only things this file changes are: SEED (from
SVPFF_SEED, one of the four frozen 21001 / 22001 / 23001 /
24001), the init / checkpoint / receipt paths (seed-specific,
attempt-suffixed, refuse-if-exists), and the init pin class
(run-derived from a double build rather than a design-time
literal). Source identity of every executed training function
is recorded by sha256 of its source text in the receipt.

MODES (env):
  SVPFF_SEED=<seed>                 required for every mode
  SVPFF_MAKE_INIT=1                 double-build init, write, record sha
  (no other env)                    path-isolated smoke: 4-step mini
                                    plan from the smoke init, no
                                    production writes
  SVPFF_PRODUCTION=1                the ONE fixed 3-epoch paired birth
  SVPFF_ATTEMPT=2                   protocol-restoring retry paths
                                    (at most one per seed, L65695)

Usage:
    SVPFF_SEED=21001 SVPFF_MAKE_INIT=1 .venv/bin/python scratch/mathworld1_svpfofresh.py
    SVPFF_SEED=21001 .venv/bin/python scratch/mathworld1_svpfofresh.py
    SVPFF_SEED=21001 SVPFF_PRODUCTION=1 .venv/bin/python scratch/mathworld1_svpfofresh.py
"""
import hashlib
import inspect
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
import scratch.mathworld1_svpforepl as F  # noqa: E402
from scratch.mathworld1_svpbirth import batch_plan, gate, sha_b  # noqa: E402

PREREG = "MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-FRESH-SEED-0"
PREREG_COMMIT = "632e57dd5593cc33c3f5588066c8e6176b7b14dd"
ADOPTED = "scratch/mathworld1_svpforepl.py"
ADOPTED_SHA = "bc9174885f7360f6c407d4f03d6eb7f1af771f96f98674d9cb783b10c0a073b9"
FRESH_SEEDS = (21001, 22001, 23001, 24001)
SEED = int(os.environ.get("SVPFF_SEED", "0"))
ATTEMPT = int(os.environ.get("SVPFF_ATTEMPT", "1"))
MAKE_INIT = os.environ.get("SVPFF_MAKE_INIT") == "1"
PRODUCTION = os.environ.get("SVPFF_PRODUCTION") == "1"
SUF = "" if ATTEMPT == 1 else f"_a{ATTEMPT}"
INIT_CK = Path(f"checkpoints/svp_forder_init_s{SEED}{SUF}.pt")
SMOKE_INIT = Path(f"checkpoints/svp_forder_init_s{SEED}_smoke.pt")
CKS = {"CANONICAL": Path(f"checkpoints/svp_forder_canonical_s{SEED}{SUF}.pt"),
       "PARAM_FIRST": Path(f"checkpoints/svp_forder_paramfirst_s{SEED}{SUF}.pt")}
RDIR = Path("logs/mathworld1/prband2fresh_train")
RECEIPT = RDIR / f"train_s{SEED}{SUF}_receipt.json"
INIT_RECEIPT = RDIR / f"init_s{SEED}{SUF}_receipt.json"
SMOKE_RECEIPT = Path(f"logs/mathworld1/prband2fresh_smoke/smoke_s{SEED}.json")
CENSUS_RECEIPT = F.CENSUS_RECEIPT
# frozen constants re-asserted against the adopted module
FROZEN = {"VOCAB": 340, "N_ROWS": 74860, "EPOCHS": 3, "TOTAL_STEPS": 7020,
          "SEQ_CAP": 512, "CODE_BASE": 332,
          "PERM": [5, 6, 7, 0, 1, 2, 3, 4], "INV": [3, 4, 5, 6, 7, 0, 1, 2],
          "PAIRED_SHA": "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75d8402351d468e8",
          "AUG_SHA": "0ef3d8a880a7e07712d8de757bc1670df12701e487b856b44c97f8db16cb3759",
          "MANIFEST_SHA": "897c8bf8fd2b6d39e361ed541d3e14c53c1c1302eeed560f77ee8fb2f2477bdd",
          "PLAN_SHA": "f55e9fee1b00f57256f3c8152be149fbb4c619479c14e4d8c01e1672afebad4d",
          "TOK_CONT_EXPECT": 2021220}
EXEC_FUNCS = ("load_rows", "state_bytes", "make_batch", "load_arm_models",
              "run", "pf_encode", "pf_decode", "encode", "ctup", "fsha")
BS = 32


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def guard():
    """Source-identity + constant guard on the adopted module."""
    gate(SEED in FRESH_SEEDS, f"SEED {SEED} NOT FROZEN")
    gate(ATTEMPT in (1, 2), "ATTEMPT")
    gate(fsha(ADOPTED) == ADOPTED_SHA, "ADOPTED INSTRUMENT SHA")
    for k, v in FROZEN.items():
        gate(getattr(F, k) == v, f"CONSTANT DRIFT {k}")
    gate(F.ARMS == ["CANONICAL", "PARAM_FIRST"] and F.ORDER == {
        0: ["CANONICAL", "PARAM_FIRST"], 1: ["PARAM_FIRST", "CANONICAL"]},
        "ARM/ORDER LAW")
    gate(F.SEED == 20001, "ADOPTED SEED LITERAL")
    return {f: hashlib.sha256(inspect.getsource(getattr(F, f)).encode()
                              ).hexdigest() for f in EXEC_FUNCS}


def census_applicable():
    """Choice B (L65695): adopt the booked census; recheck every
    sha it records."""
    gate(CENSUS_RECEIPT.exists(), "CENSUS NOT RUN")
    cr = json.loads(CENSUS_RECEIPT.read_text())
    gate(cr["verdict"] == "CENSUS QUALIFIED", "CENSUS NOT QUALIFIED")
    for pth, h in cr["start"]["file_sha256"].items():
        gate(fsha(pth) == h, f"CENSUS STALE {pth}")
    for pth, h in cr.get("pins", {}).items():
        gate(fsha(pth) == h, f"CENSUS PIN STALE {pth}")
    return fsha(CENSUS_RECEIPT)


def make_init(path):
    gate(not path.exists(), f"INIT ALREADY EXISTS {path}")
    b1 = F.state_bytes(SEED)
    b2 = F.state_bytes(SEED)
    gate(b1 == b2, "INIT-INTEGRITY FAILURE: double build differs")
    path.write_bytes(b1)
    sha = hashlib.sha256(b1).hexdigest()
    gate(fsha(path) == sha, "INIT WRITE")
    return sha


def main():
    gate(SEED != 0, "SVPFF_SEED REQUIRED")
    func_shas = guard()
    census_sha = census_applicable()
    START = start_provenance(
        ["scratch/mathworld1_svpfofresh.py", ADOPTED,
         "scratch/mathworld1_svpbirth.py", "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py", "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    RDIR.mkdir(parents=True, exist_ok=True)
    SMOKE_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    if MAKE_INIT:
        gate(not PRODUCTION, "MAKE_INIT WITH PRODUCTION")
        gate(not INIT_RECEIPT.exists(), f"REFUSING: {INIT_RECEIPT} exists")
        sha = make_init(INIT_CK)
        INIT_RECEIPT.write_text(json.dumps({
            "prereg": PREREG, "seed": SEED, "attempt": ATTEMPT,
            "init_path": str(INIT_CK), "init_sha": sha,
            "double_build_identical": True, "census_receipt_sha": census_sha,
            "adopted_instrument": {"path": ADOPTED, "sha256": fsha(ADOPTED)},
            "executed_function_source_sha256": func_shas,
            "start": START, "completion_commit": completion_commit()}, indent=1))
        print(f"[svpfofresh] init s{SEED} written {sha}", flush=True)
        return 0
    gate(torch.backends.mps.is_available(), "MPS UNAVAILABLE")
    dev = torch.device("mps")
    if PRODUCTION:
        for p in list(CKS.values()) + [RECEIPT]:
            if p.exists():
                raise SystemExit(f"REFUSING: {p} exists")
        gate(INIT_RECEIPT.exists(), "INIT RECEIPT ABSENT")
        ir = json.loads(INIT_RECEIPT.read_text())
        gate(ir["seed"] == SEED and ir["attempt"] == ATTEMPT
             and ir["double_build_identical"], "INIT RECEIPT")
        init_path, init_sha = INIT_CK, ir["init_sha"]
        gate(init_path.exists() and fsha(init_path) == init_sha, "INIT != RECEIPT SHA")
        gate(SMOKE_RECEIPT.exists(), "SMOKE NOT RUN")
        sr = json.loads(SMOKE_RECEIPT.read_text())
        gate(all(sr["bars"].values()), "SMOKE NOT GREEN")
        for pth, h in sr["start"]["file_sha256"].items():
            gate(fsha(pth) == h, f"SMOKE STALE {pth}")
    else:
        if SMOKE_RECEIPT.exists():
            raise SystemExit(f"REFUSING: {SMOKE_RECEIPT} exists")
        for p in list(CKS.values()) + [RECEIPT]:
            if p.exists():
                raise SystemExit(f"REFUSING: production path {p} exists")
        if SMOKE_INIT.exists():
            SMOKE_INIT.unlink()
        init_path, init_sha = SMOKE_INIT, make_init(SMOKE_INIT)
    # the adopted module's training functions read INIT_CK as a
    # module global: point it at the seed-specific init (path only)
    F.INIT_CK = init_path
    rows = F.load_rows()
    rows_by_id = {r["row_id"]: r for r in rows}
    gate(len(rows_by_id) == F.N_ROWS, "DUPLICATE ROW_ID")
    plan = batch_plan(list(rows_by_id))
    gate(len(plan) == F.TOTAL_STEPS, "PLAN LEN")
    plan_sha = sha_b(json.dumps(plan).encode())
    gate(plan_sha == F.PLAN_SHA, "PLAN SHA")
    for e in range(F.EPOCHS):
        eb = [p for p in plan if p[0] == e]
        gate(len(eb) == 2340, f"EPOCH BATCHES e{e}")
        gate(len(eb[-1][2]) == 12, f"TAIL e{e}")
    gate(all(len(p[2]) == BS for p in plan if len(p[2]) != 12), "BATCH SIZE")
    t0 = time.time()
    common = {"prereg": PREREG, "prereg_commit": PREREG_COMMIT, "seed": SEED,
              "attempt": ATTEMPT, "vocab": F.VOCAB,
              "adopted_instrument": {"path": ADOPTED, "sha256": fsha(ADOPTED)},
              "executed_function_source_sha256": func_shas,
              "replication_of": (f"seed-20001 paired training law executed from "
                                 f"{ADOPTED} sha {fsha(ADOPTED)[:16]}; fresh seed {SEED}"),
              "census_receipt_sha": census_sha, "init_path": str(init_path),
              "init_sha": init_sha, "plan_sha": plan_sha,
              "env": {"torch": torch.__version__, "device": str(dev)},
              "perm": F.PERM, "inv": F.INV}
    if not PRODUCTION:
        maxlen = {i: len(r["_pre"]) for i, r in rows_by_id.items()}
        longest_step = max(range(2340), key=lambda s: max(maxlen[i] for i in plan[s][2]))
        ordinary = [s for s in (0, 1, 2) if s != longest_step][:2]
        mini = [plan[longest_step]] + [plan[s] for s in ordinary] + [plan[2339]]
        n = len(mini)
        gate(n == 4, "SMOKE PLAN SIZE")
        arms, losses, gnorms, tp, tc, tn, oc, wall, n_params, stats = F.run(
            mini, rows_by_id, dev, init_sha, n, f"smoke-s{SEED}")
        n_rows_smoke = sum(len(p[2]) for p in mini)
        receipt = {**common, "mode": "smoke", "n_params": n_params, "steps": n,
                   "smoke_plan_steps": [longest_step] + ordinary + [2339],
                   "losses": losses, "tokens_continuation": tc,
                   "bars": {"INIT_BITWISE_ALL_ARMS": stats["bitwise_tensors_compared"] == 2 * 59,
                            "VOCAB_340": F.VOCAB == 340,
                            "PARAM_EQUAL": n_params == 19142016,
                            "PLAN_SHA_MATCH": plan_sha == F.PLAN_SHA,
                            "BOTH_ARMS_T9": tc["CANONICAL"] == tc["PARAM_FIRST"] == 9 * n_rows_smoke,
                            "BOTH_ORDERS_EXERCISED": oc[0] >= 1 and oc[1] >= 1,
                            "SHAPE_GATES_EVERY_STEP": stats["shape_checks"] == n,
                            "FINITE": stats["finite_losses_checked"] == 2 * n,
                            "STEP_LOCKSTEP": stats["lockstep_checks"] == n,
                            "NO_PRODUCTION_PATHS": not any(p.exists() for p in CKS.values())
                            and not RECEIPT.exists() and not INIT_CK.exists()},
                   "wall_s": round(time.time() - t0, 1),
                   "start": START, "completion_commit": completion_commit()}
        SMOKE_RECEIPT.write_text(json.dumps(receipt, indent=1))
        print(json.dumps(receipt["bars"], indent=1), flush=True)
        return 0
    arms, losses, gnorms, tp, tc, tn, oc, wall, n_params, stats = F.run(
        plan, rows_by_id, dev, init_sha, F.TOTAL_STEPS, f"prod-s{SEED}")
    for a in F.ARMS:
        gate(len(losses[a]) == F.TOTAL_STEPS, f"INCOMPLETE {a}")
        gate(tc[a] == F.TOK_CONT_EXPECT, f"CONT TOKENS {a}")
        gate(all(math.isfinite(x) for x in losses[a]), f"NON-FINITE {a}")
    gate(oc[0] == 3510 and oc[1] == 3510, f"ORDER COUNTS {dict(oc)}")
    gate(stats["bitwise_tensors_compared"] == 2 * 59, "INIT BITWISE COUNT")
    for a in F.ARMS:
        torch.save({k: v.cpu() for k, v in sorted(arms[a].state_dict().items())}, CKS[a])
    shas = {a: fsha(CKS[a]) for a in F.ARMS}
    gate(shas["CANONICAL"] != shas["PARAM_FIRST"], "IDENTICAL ARM CHECKPOINTS")
    gate(fsha(init_path) == init_sha, "POST INIT PIN")
    epoch_mean = {a: [round(sum(losses[a][e * 2340:(e + 1) * 2340]) / 2340, 5)
                      for e in range(3)] for a in F.ARMS}
    receipt = {**common, "mode": "production", "n_params": n_params,
               "total_steps": F.TOTAL_STEPS,
               "updates_per_arm": {a: len(losses[a]) for a in F.ARMS},
               "sched_terminal": stats["sched_terminal"],
               "finite_losses_checked": stats["finite_losses_checked"],
               "epoch_mean_loss": epoch_mean, "tokens_padded": tp,
               "tokens_continuation": tc, "tokens_nonpad": tn,
               "order_counts": {"CANONICAL_first": oc[0], "PARAM_FIRST_first": oc[1]},
               "arm_wall_s_note": "encode excluded; within-run-paired only, never cross-run comparable",
               "arm_wall_s": {a: round(wall[a], 1) for a in F.ARMS},
               "wall_s": round(time.time() - t0, 1),
               "losses": {a: [round(x, 5) for x in losses[a]] for a in F.ARMS},
               "grad_norm_summary": {a: {"p50": round(sorted(gnorms[a])[len(gnorms[a]) // 2], 5),
                                         "max": round(max(gnorms[a]), 5)} for a in F.ARMS},
               "lockstep_checks": stats["lockstep_checks"],
               "shape_checks": stats["shape_checks"],
               "bitwise_tensors_compared": stats["bitwise_tensors_compared"],
               "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024),
               "peak_mps_allocated_bytes": stats["peak_mps_allocated_bytes"],
               "checkpoints": {str(CKS[a]): shas[a] for a in F.ARMS},
               "pins": {F.PAIRED: fsha(F.PAIRED), F.AUG: fsha(F.AUG),
                        F.MANIFEST: fsha(F.MANIFEST), str(init_path): fsha(init_path)},
               "smoke_receipt_sha": fsha(SMOKE_RECEIPT),
               "init_receipt_sha": fsha(INIT_RECEIPT),
               "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(f"[svpfofresh] DONE s{SEED} {shas}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

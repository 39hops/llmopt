"""SYNTHETIC-GRADIENT-WRITER-1 qualification birth driver. Sibling of the
results-cited FROZEN-BACKBONE-1 driver scratch/birth19m_fb.py (frozen;
the stock loop lines and the bp / dfa / hybrid / zero branches survive
here verbatim; the DFA branches are never selected under the SG law).
Arena (AMENDMENT -ARENA L70126, AMENDMENT -SEAL): emb + blocks 0..3 frozen
exactly at W_0 (dfa_credit.freeze_lower(model, 4), re-verified at the end),
blocks 4..7 credited by the synthetic-gradient law of scratch/sg_credit.py,
norm + head by the true CE gradient with x_8 detached.

  CONTROL  MODE=zero K_BP=4      the FROZEN-BACKBONE-1 arm at the SG seed
                                 (BP-top-over-frozen); its gate defines c.
  SG cell  MODE=sg FAMILY=linear|mlp256 PLR=3e-4|3e-5
                                 blocks 4..7 by hat_delta_l = s_l * G_phi_l
                                 (fixed arena constants from
                                 logs/sgaudit0/audit.json, digest asserted),
                                 predictors AdamW(PLR, wd 0) + OneCycle of
                                 the stock shape, FOLD A step order, FOLD B
                                 elementwise normalized MSE; predictor
                                 snapshots pred_step_{n:05d}.pt beside the
                                 model snapshots; per-200-step log of the
                                 CE loss, the predictor MSE per block and the
                                 alignment cos(hat_delta, delta^BP) per block.

SG=1: seed 27 only, model LR 3e-4, 15,420 steps, 17 snapshots; paths
checkpoints/sgwriter1/ and logs/sgwriter1/qual.jsonl. SMOKE=1: seed 11,
checkpoints/sgwriter1_smoke/, logs/sgwriter1/smoke.jsonl, SMOKE_STEPS
default 300. Everything else (QUAL / DISCOVERY / FB branches) is inherited
and refused under SG=1 or SMOKE=1 here. Non-finite loss: SG books the cell
UNSTABLE (receipt row, no final checkpoint, rc 0, ladder continues); SMOKE
aborts. Receipts derive every field from the artifacts this process opened
or wrote; HEAD is re-read at the receipt write and must equal the launch
commit.

Smoke-only extras: DRYRUN=1 (stream assertions only), EMIT=0, DEVICE=cpu,
TAG=<suffix>.

Usage: SG=1 MODE=sg FAMILY=linear PLR=3e-4 SEED=27 LR=3e-4 .venv/bin/python scratch/birth19m_sg.py
       SG=1 MODE=zero K_BP=4 SEED=27 LR=3e-4 .venv/bin/python scratch/birth19m_sg.py
"""
import datetime
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

MODE = os.environ.get("MODE", "")
SEED = int(os.environ["SEED"])
assert MODE in ("bp", "dfa", "hybrid", "zero", "sg"), MODE
K_BP = int(os.environ["K_BP"]) if MODE in ("hybrid", "zero") else (0 if MODE == "dfa" else (4 if MODE == "sg" else 8))
assert 0 <= K_BP <= 8, K_BP
QUAL = os.environ.get("QUAL") == "1"
DISCOVERY = os.environ.get("DISCOVERY") == "1"
SMOKE = os.environ.get("SMOKE") == "1"
FB = os.environ.get("FB") == "1"
SG = os.environ.get("SG") == "1"
assert QUAL + DISCOVERY + SMOKE + FB + SG == 1, "exactly one of QUAL / DISCOVERY / SMOKE / FB / SG"
assert not (QUAL or DISCOVERY or FB), "QUAL / DISCOVERY / FB branches are inherited and refused in the SG driver"
SG_SEED = 27
SG_K = 4
SG_BLOCKS = [4, 5, 6, 7]
SG_FAMILIES = ("linear", "mlp256")
SG_PLRS = (3e-4, 3e-5)
FAMILY = os.environ.get("FAMILY", "")
PLR = float(os.environ["PLR"]) if MODE == "sg" else None
DRYRUN = os.environ.get("DRYRUN") == "1"
EMIT = os.environ.get("EMIT", "1") != "0"
DEVICE_OVERRIDE = os.environ.get("DEVICE", "")
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "300"))
TAG = os.environ.get("TAG", "")
GRADDUMP = ""
S = float(os.environ.get("S", "0")) if MODE in ("dfa", "hybrid") else None
PEAK_LR = float(os.environ.get("LR", "3e-4"))

SMOKE_SEED = 11
ZERO_LR = 3e-4
AUDIT = Path("logs/sgaudit0/audit.json")
if MODE == "sg":
    assert FAMILY in SG_FAMILIES, f"FAMILY must be one of {SG_FAMILIES}"
    assert PLR in SG_PLRS, f"PLR must be one of {SG_PLRS}"
    assert K_BP == SG_K
if SMOKE:
    assert SEED == SMOKE_SEED, f"smoke seed is {SMOKE_SEED}, got {SEED}"
    assert MODE in ("zero", "sg"), "the SG driver smokes the control (zero K_BP=4) and the sg cells only"
    if MODE == "zero":
        assert K_BP == SG_K and PEAK_LR == ZERO_LR, "the control is MODE=zero K_BP=4 at LR 3e-4"
else:
    assert EMIT and not DEVICE_OVERRIDE and not TAG, "EMIT/DEVICE/TAG are smoke-only"
    assert SEED == SG_SEED, f"SYNTHETIC-GRADIENT-WRITER-1 qualification seed is {SG_SEED}, got {SEED}"
    assert MODE in ("zero", "sg"), "qualification arms are MODE=zero K_BP=4 (control) and MODE=sg (cells)"
    assert PEAK_LR == 3e-4, "the model recipe is the stock LR 3e-4"
    if MODE == "zero":
        assert K_BP == SG_K, "the control is emb + blocks 0..3 frozen (K_BP=4)"

os.environ["ARM"] = "off"       # frozen module import side-effects only
os.environ["BIRTH_SEED"] = str(SEED)

import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402  (frozen, import-only)
import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest, stream_digest  # noqa: E402
from dfa_credit import build_feedback, feedback_digest, dfa_objective, hybrid_objective, freeze_lower  # noqa: E402
from sg_credit import build_predictors, run_sg_step, sg_step_terms  # noqa: E402

EPOCHS, BS = C.EPOCHS, C.BS
STEPS_TOTAL_PIN = 15_420
SCHEDULE = [0, 463] + list(range(1_028, STEPS_TOTAL_PIN + 1, 1_028))
PHASE = "smoke" if SMOKE else "sgq"
CELL = f"{PHASE}_" + ("control" if MODE == "zero" else f"sg_{FAMILY}_plr{PLR:g}") + f"_s{SEED}_lr{PEAK_LR:g}" + TAG
ROOT = Path("checkpoints/sgwriter1_smoke" if SMOKE else "checkpoints/sgwriter1")
OUTDIR = ROOT / CELL
RECEIPTS = Path("logs/sgwriter1/smoke.jsonl" if SMOKE else "logs/sgwriter1/qual.jsonl")
MIN_FREE_BYTES = 15 * 1024 ** 3
LOG_EVERY = 200

# Pinned actual token-id batch-stream digests, stock arm (PRE-REG L66546,
# scratch/atomtraj_pins.py); the DFA rung trains on the stock stream only.
PINNED_STREAMS = ["18a6c14a6e7cd48e871e1645bfef1f6c89297681531ee34cc48740665ea003f7",
                  "a3bf1ab910faf09f1cdd1455dd54a16474eb4cf4433deb96b1e4607c7cd18324",
                  "7ff1b63ea45b343a94ba2092deeaf3da4abbd006085a190f0167781f579d433b"]
PINNED_COUNTS = {"enc_stock": 164_490, "steps_per_epoch": 5_140}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty():
    return bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def save_snapshot(model, step, record):
    """Save the state dict (CPU float32) as step_{n:05d}.pt and record the
    file sha256 and canonical state_digest read back from the file."""
    p = OUTDIR / f"step_{step:05d}.pt"
    if p.exists():
        raise SystemExit(f"REFUSING: {p} exists")
    sd = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    torch.save(sd, p)
    back = torch.load(p, map_location="cpu")
    record[str(step)] = {"file_sha256": sha256_file(p), "state_digest": state_digest(back)}


def save_pred_snapshot(preds, step, record):
    p = OUTDIR / f"pred_step_{step:05d}.pt"
    if p.exists():
        raise SystemExit(f"REFUSING: {p} exists")
    sd = {k: v.detach().to("cpu") for k, v in preds.state_dict().items()}
    torch.save(sd, p)
    back = torch.load(p, map_location="cpu")
    record[str(step)] = {"file_sha256": sha256_file(p), "state_digest": state_digest(back)}


def load_constants():
    """The fixed arena constants s_4..7 of OBSERVATION SG-PREDICTOR-AUDIT-0,
    read from the locked audit receipt; the receipt's sha256 is recorded."""
    audit = json.loads(AUDIT.read_text())
    consts = audit["normalization_constants"]["arena_frozen_4_7"]
    assert sorted(consts) == ["4", "5", "6", "7"] and all(v > 0 for v in consts.values())
    return consts, sha256_file(AUDIT)


def write_receipt(row, launch_commit):
    head = git_head()
    if not SMOKE and head != launch_commit:
        raise SystemExit(f"ABORT: HEAD {head} != launch commit {launch_commit} at receipt write")
    row["code_commit"] = head
    row["launch_commit"] = launch_commit
    row["tree_dirty"] = git_dirty()
    RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    with RECEIPTS.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    if OUTDIR.exists() and not DRYRUN:
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    launch_commit = git_head()
    tok = TM.MathTokenizer()
    stock_rows = C.load_excised_rows()
    enc_stock, _ = C.encode_with_levels(stock_rows, tok)
    enc = enc_stock
    print(f"[sg] mode={MODE} k_bp={K_BP} seed={SEED} cell={CELL}: stock {len(enc_stock)} seq", flush=True)

    C.assert_noop(enc_stock)    # precondition, fresh, in-process

    steps_per_epoch = len(C.stock_epoch_stream(len(enc_stock), 0))
    steps_total = EPOCHS * (len(enc_stock) // BS)
    assert steps_total == STEPS_TOTAL_PIN, steps_total
    assert len(enc_stock) == PINNED_COUNTS["enc_stock"]
    assert steps_per_epoch == PINNED_COUNTS["steps_per_epoch"]

    # stream identity: the actual token-id batches of every epoch, before any step
    streams, digests = [], []
    for ep in range(EPOCHS):
        stream = C.stock_epoch_stream(len(enc), ep)
        assert len(stream) == steps_per_epoch
        digests.append(stream_digest(enc, stream))
        streams.append(stream)
    for ep in range(EPOCHS):
        if digests[ep] != PINNED_STREAMS[ep]:
            raise SystemExit(f"REFUSING: stream digest mismatch stock e{ep}: {digests[ep]}")
    print(f"[sg] stream digests verified stock e0..e2", flush=True)
    consts, audit_sha = (load_constants() if MODE == "sg" else (None, None))
    if DRYRUN:
        print(json.dumps({"dryrun": True, "mode": MODE, "seed": SEED, "cell": CELL, "streams": digests, "consts": consts, "audit_sha256": audit_sha}))
        return
    free = shutil.disk_usage(".").free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"NOT-RUN: disk free {free / 1024**3:.1f} GB < 15 GB")
    if not SMOKE and git_dirty():
        raise SystemExit("REFUSING: registered birth on a dirty tree")

    dev = DEVICE_OVERRIDE or ("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
    if SG:
        assert dev == "mps", "SYNTHETIC-GRADIENT-WRITER-1 births are sealed on mps"
    torch.manual_seed(SEED)
    model = TM.build_model(len(tok.vocab), d=384, layers=8,
                           heads=6, ffn=1536).to(dev)
    assert len(tok.vocab) == 40, len(tok.vocab)
    frozen_names, trainable_names = ([], [n for n, _ in model.named_parameters()])
    if MODE in ("zero", "sg"):
        frozen_names, trainable_names = freeze_lower(model, K_BP)
        print(f"[sg] frozen backbone: {len(frozen_names)} frozen tensors, {len(trainable_names)} trainable", flush=True)
    model_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(model_params, lr=PEAK_LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=PEAK_LR, total_steps=steps_total, pct_start=0.03)
    preds, phi, pred_opt, pred_sched, pred_seed = None, None, None, None, None
    if MODE == "sg":
        pred_seed = 1000 + SEED
        preds = build_predictors(FAMILY, SG_BLOCKS, seed=pred_seed).to(dev)
        phi = list(preds.parameters())
        pred_opt = torch.optim.AdamW(phi, lr=PLR, weight_decay=0.0)
        pred_sched = torch.optim.lr_scheduler.OneCycleLR(pred_opt, max_lr=PLR, total_steps=steps_total, pct_start=0.03)
        print(f"[sg] predictors {FAMILY} x {len(SG_BLOCKS)} ({sum(p.numel() for p in phi)} params) plr {PLR:g} seed {pred_seed}; "
              f"constants {json.dumps(consts)} (audit sha {audit_sha[:12]})", flush=True)
    print(f"[sg] steps_total {steps_total} "
          f"({steps_per_epoch}/epoch) peak lr {PEAK_LR:g}", flush=True)

    cap = min(SMOKE_STEPS, steps_total) if SMOKE else steps_total
    schedule = [s for s in SCHEDULE if s <= cap]
    if cap not in schedule:
        schedule.append(cap)
    OUTDIR.mkdir(parents=True, exist_ok=False)
    RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    snapshots, pred_snapshots = {}, {}
    started = now()
    init_digest = state_digest(model.state_dict())
    pred_init_digest = state_digest(preds.state_dict()) if preds is not None else None
    if EMIT:
        save_snapshot(model, 0, snapshots)
        assert snapshots["0"]["state_digest"] == init_digest, "step_0 file digest v in-memory init"
        if preds is not None:
            save_pred_snapshot(preds, 0, pred_snapshots)

    feedback = None
    if MODE in ("dfa", "hybrid"):
        Bs_cpu = build_feedback(S)
        fb_p = OUTDIR / "feedback.pt"
        torch.save({"s": S, "seed_base": 31_000_000, "B": Bs_cpu}, fb_p)
        back = torch.load(fb_p, map_location="cpu")["B"]
        feedback = {"s": S, "digest": feedback_digest(back), "file_sha256": sha256_file(fb_p),
                    "shapes": [list(b.shape) for b in back], "seed_base": 31_000_000}
        Bs = [b.to(dev) for b in back]
        print(f"[sg] feedback digest {feedback['digest'][:16]} s={S:g}", flush=True)

    step = 0
    t0 = time.time()
    done = False
    nonfinite_step = None
    sg_log = []
    for ep in range(EPOCHS):
        stream = streams[ep]
        print(f"[sg] ep{ep}: {len(stream)} batches", flush=True)
        for a, b in stream:
            batch = enc[a:b]
            L = max(len(s) for s in batch)
            ids = torch.tensor([s + [tok.pad_id] * (L - len(s))
                                for s in batch], device=dev)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                                 for s in batch], device=dev)
            labels = ids[:, 1:].clone()
            labels[mask[:, 1:] == 0] = -100
            if MODE == "sg":
                # FOLD A: terms at (W_t, phi_t) with the teacher target cached before any backward;
                # run_sg_step performs the model backward, the predictor backward against the cached
                # target, the two separate clips, the two optimizer steps and the two schedulers
                T = sg_step_terms(model, preds, ids[:, :-1], mask[:, :-1], labels, SG_BLOCKS, consts)
                loss = T["loss"]
                run_sg_step(T, model_params, phi, opt, pred_opt, sched, pred_sched, steps_total)
                step += 1
                if step % LOG_EVERY == 0 or step == 1 or step in schedule:
                    sg_log.append({"step": step, "loss": float(loss.detach()), "loss_T": float(T["loss_T"]),
                                   "pred_mse": float(T["pred_loss"].detach()), "pred_mse_per_block": {str(k): v for k, v in T["pred_loss_per_block"].items()},
                                   "align": {str(k): v for k, v in T["align"].items()}})
            else:
                if MODE == "bp":
                    logits = model(ids[:, :-1], mask[:, :-1])
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1), ignore_index=-100)
                    loss.backward()
                elif MODE == "dfa":
                    ob = dfa_objective(model, Bs, ids[:, :-1], mask[:, :-1], labels)
                    loss = ob["loss"]
                    ob["total"].backward()
                elif MODE == "hybrid":
                    ob = hybrid_objective(model, Bs, ids[:, :-1], mask[:, :-1], labels, K_BP)
                    loss = ob["loss"]
                    ob["total"].backward()
                else:
                    # zero: forward_hybrid(k_bp=8) attaches nothing by detach; the DFA -> BP cut is realised by
                    # freeze_lower (requires_grad False on emb and blocks 0..7-k), so no graph exists below the
                    # segment and the segment gradients equal those through a detached boundary (same input values)
                    ob = hybrid_objective(model, [], ids[:, :-1], mask[:, :-1], labels, 8)
                    loss = ob["loss"]
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                if sched.last_epoch < steps_total - 1:
                    sched.step()
                opt.zero_grad()
                step += 1
            if not torch.isfinite(loss.detach()):
                if SG:
                    nonfinite_step = step
                    print(f"[sg] UNSTABLE: non-finite loss at step {step}", flush=True)
                    done = True
                    break
                raise SystemExit(f"ABORT: non-finite loss at step {step}")
            if step % LOG_EVERY == 0:
                extra = ""
                if MODE == "sg":
                    extra = (f" pred_mse {sg_log[-1]['pred_mse']:.3f} align "
                             + " ".join(f"b{k}:{(v if v is not None else float('nan')):.3f}" for k, v in sg_log[-1]["align"].items()))
                print(f"  step {step}/{steps_total} loss "
                      f"{float(loss.detach()):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s){extra}",
                      flush=True)
            if EMIT and step in schedule:
                save_snapshot(model, step, snapshots)
                if preds is not None:
                    save_pred_snapshot(preds, step, pred_snapshots)
            if step >= cap:
                done = True
                break
        if done:
            break
    wall = time.time() - t0

    row = {"kind": "birth", "prereg": "SYNTHETIC-GRADIENT-WRITER-1", "phase": PHASE, "arm": ("CONTROL" if MODE == "zero" else "SG"), "mode": MODE, "k_bp": K_BP, "seed": SEED, "cell": CELL,
           "family": (FAMILY if MODE == "sg" else None), "plr": PLR, "sg_blocks": (SG_BLOCKS if MODE == "sg" else None),
           "constants": consts, "audit_sha256": audit_sha, "pred_seed": pred_seed, "pred_init_state_digest": pred_init_digest,
           "pred_n_params": (sum(p.numel() for p in phi) if phi is not None else None),
           "frozen_tensors": frozen_names, "n_trainable_tensors": len(trainable_names),
           "s": S, "peak_lr": PEAK_LR, "steps": step, "steps_total": steps_total,
           "smoke": SMOKE, "emit": EMIT, "device": dev, "tag": TAG,
           "started_utc": started, "ended_utc": now(), "wall_s": round(wall, 1),
           "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "torch": torch.__version__, "vocab_len": len(tok.vocab), "param_dtype": str(next(model.parameters()).dtype),
           "enc_stock": len(enc_stock), "enc_train": len(enc), "stream_sha256": digests,
           "schedule": schedule, "outdir": str(OUTDIR), "init_state_digest": init_digest,
           "feedback": feedback, "snapshots": snapshots, "pred_snapshots": pred_snapshots, "sg_log": sg_log}
    if nonfinite_step is not None:
        row.update({"stable_training": False, "nonfinite_step": nonfinite_step, "final": None})
        write_receipt(row, launch_commit)
        print(f"[sg] receipt appended to {RECEIPTS} (UNSTABLE at step {nonfinite_step}; no final checkpoint)", flush=True)
        return

    if MODE in ("zero", "sg"):
        sd_now = model.state_dict()
        w0_back = torch.load(OUTDIR / "step_00000.pt", map_location="cpu") if EMIT else None
        if w0_back is not None:
            for n in frozen_names:
                if not torch.equal(sd_now[n].detach().to("cpu"), w0_back[n]):
                    raise SystemExit(f"ABORT: frozen tensor {n} moved from W_0")
            print(f"[sg] freeze law verified: {len(frozen_names)} tensors bit-identical to W_0", flush=True)
    final_p = OUTDIR / "final.pt"
    if final_p.exists():
        raise SystemExit(f"REFUSING: {final_p} exists")
    sd_final = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    torch.save(sd_final, final_p)
    final_back = torch.load(final_p, map_location="cpu")
    final_rec = {"file_sha256": sha256_file(final_p), "state_digest": state_digest(final_back)}
    if EMIT:
        last = torch.load(OUTDIR / f"step_{step:05d}.pt", map_location="cpu")
        if state_digest(last) != final_rec["state_digest"]:
            raise SystemExit("ABORT: state_digest(step_final) != state_digest(final)")
    pred_final_rec = None
    if preds is not None:
        pf = OUTDIR / "pred_final.pt"
        torch.save({k: v.detach().to("cpu") for k, v in preds.state_dict().items()}, pf)
        pred_final_rec = {"file_sha256": sha256_file(pf), "state_digest": state_digest(torch.load(pf, map_location="cpu"))}
    print(f"[sg] saved {final_p} after {step} steps ({wall:.0f}s)", flush=True)

    h = hashlib.sha256()
    for k, v in sorted(((str(k), v) for k, v in opt.state_dict()["state"].items()), key=lambda kv: kv[0]):
        for name in sorted(v):
            t = v[name]
            h.update(f"{k}.{name}".encode())
            h.update(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes() if torch.is_tensor(t) else str(t).encode())
    row.update({"stable_training": True, "final": final_rec, "pred_final": pred_final_rec, "final_loss": float(loss.detach()),
                "optimizer_state_digest": h.hexdigest(),
                "torch_rng_digest": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()})
    write_receipt(row, launch_commit)
    print(f"[sg] receipt appended to {RECEIPTS} (final state_digest {final_rec['state_digest'][:16]})", flush=True)


if __name__ == "__main__":
    main()

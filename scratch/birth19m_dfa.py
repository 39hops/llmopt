"""WRITER-DFA-1 birth driver (pre-reg RESULTS L68321, sealed by
AMENDMENT -SEAL L68543 and -PRECISION L68644). Sibling of the
results-cited ATOM-DIET-TRAJECTORY-1 driver scratch/birth19m_atoms_traj.py
(stock arm; not edited): the recipe, the pinned stock batch stream, the
17-snapshot cadence, the receipt law and the aborts are verbatim. The
only scientific switch is the credit writer:

  MODE=bp   backprop control: the stock loop lines unchanged
            (logits = model(...); loss.backward()).
  MODE=dfa  Direct Feedback Alignment (scratch/dfa_credit.py): every
            block boundary detached, head / norm on x_8.detach(),
            delta_l = B_l e_t with B_l fixed (seed 31_000_000 + l,
            U(-1, 1) * S / sqrt(40)); total = L + sum_l <delta_l, x_{l+1}>
            and total.backward(). AdamW, clip 1.0, OneCycle (peak LR),
            stream, cadence and device unchanged.

Seed law (S7): exactly one of QUAL=1 (seed 21, MODE=dfa, one of the
four sealed (S, LR) cells), DISCOVERY=1 (seed 2; MODE=bp with LR=3e-4,
or MODE=dfa with the (S, LR) frozen in logs/writerdfa1/qual_selection.json)
or SMOKE=1 (seed 11 only; checkpoints/writerdfa1_smoke/ and
logs/writerdfa1/smoke.jsonl only; SMOKE_STEPS default 300). All other
seeds refused. Non-finite loss: QUAL books the cell UNSTABLE (receipt
row, no final checkpoint, rc 0, ladder continues); DISCOVERY / SMOKE
abort (NOT-RUN). Receipts derive every field from the artifacts this
process opened or wrote; HEAD is re-read at the receipt write and must
equal the launch commit.

Smoke-only extras: DRYRUN=1 (stream assertions only), EMIT=0, DEVICE=cpu,
TAG=<suffix>, GRADDUMP=<path> (after step 1: post-clip gradients and
post-step parameters, for the writer-integrity smoke).

Usage: QUAL=1 MODE=dfa SEED=21 S=1 LR=3e-4 .venv/bin/python scratch/birth19m_dfa.py
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
assert MODE in ("bp", "dfa"), MODE
QUAL = os.environ.get("QUAL") == "1"
DISCOVERY = os.environ.get("DISCOVERY") == "1"
SMOKE = os.environ.get("SMOKE") == "1"
assert QUAL + DISCOVERY + SMOKE == 1, "exactly one of QUAL / DISCOVERY / SMOKE"
DRYRUN = os.environ.get("DRYRUN") == "1"
EMIT = os.environ.get("EMIT", "1") != "0"
DEVICE_OVERRIDE = os.environ.get("DEVICE", "")
SMOKE_STEPS = int(os.environ.get("SMOKE_STEPS", "300"))
TAG = os.environ.get("TAG", "")
GRADDUMP = os.environ.get("GRADDUMP", "")
S = float(os.environ.get("S", "0")) if MODE == "dfa" else None
PEAK_LR = float(os.environ.get("LR", "3e-4"))

QUAL_SEED, DISCOVERY_SEED, SMOKE_SEED = 21, 2, 11
QUAL_CELLS = [(1.0, 3e-4), (1.0, 1e-4), (0.25, 3e-4), (4.0, 1e-4)]
SELECTION = Path("logs/writerdfa1/qual_selection.json")
if SMOKE:
    assert SEED == SMOKE_SEED, f"smoke seed is {SMOKE_SEED}, got {SEED}"
    if MODE == "dfa":
        assert S > 0, "S required in dfa mode"
else:
    assert EMIT and not DEVICE_OVERRIDE and not TAG and not GRADDUMP, "EMIT/DEVICE/TAG/GRADDUMP are smoke-only"
    if QUAL:
        assert SEED == QUAL_SEED and MODE == "dfa", "qualification is MODE=dfa at seed 21 only"
        assert (S, PEAK_LR) in QUAL_CELLS, f"({S}, {PEAK_LR}) is not a sealed qualification cell"
    else:
        assert SEED == DISCOVERY_SEED, f"discovery seed is {DISCOVERY_SEED}, got {SEED}"
        if MODE == "bp":
            assert PEAK_LR == 3e-4, "the control is the stock recipe (LR 3e-4)"
        else:
            sel = json.loads(SELECTION.read_text())
            assert sel.get("selected"), "no qualification selection frozen"
            assert (S, PEAK_LR) == (float(sel["selected"]["s"]), float(sel["selected"]["lr"])), f"(S, LR) differ from the frozen selection {sel['selected']}"

os.environ["ARM"] = "off"       # frozen module import side-effects only
os.environ["BIRTH_SEED"] = str(SEED)

import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402  (frozen, import-only)
import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest, stream_digest  # noqa: E402
from dfa_credit import build_feedback, feedback_digest, dfa_objective  # noqa: E402

EPOCHS, BS = C.EPOCHS, C.BS
STEPS_TOTAL_PIN = 15_420
SCHEDULE = [0, 463] + list(range(1_028, STEPS_TOTAL_PIN + 1, 1_028))
PHASE = "smoke" if SMOKE else ("qual" if QUAL else "disc")
CELL = f"{PHASE}_{MODE}_s{SEED}" + (f"_S{S:g}_lr{PEAK_LR:g}" if MODE == "dfa" else f"_lr{PEAK_LR:g}") + TAG
ROOT = Path("checkpoints/writerdfa1_smoke" if SMOKE else "checkpoints/writerdfa1")
OUTDIR = ROOT / CELL
RECEIPTS = Path("logs/writerdfa1/smoke.jsonl" if SMOKE else ("logs/writerdfa1/qual.jsonl" if QUAL else "logs/writerdfa1/births.jsonl"))
MIN_FREE_BYTES = 15 * 1024 ** 3

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
    print(f"[dfa] mode={MODE} seed={SEED} cell={CELL}: stock {len(enc_stock)} seq", flush=True)

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
    print(f"[dfa] stream digests verified stock e0..e2", flush=True)
    if DRYRUN:
        print(json.dumps({"dryrun": True, "mode": MODE, "seed": SEED, "cell": CELL, "streams": digests}))
        return
    free = shutil.disk_usage(".").free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"NOT-RUN: disk free {free / 1024**3:.1f} GB < 15 GB")
    if not SMOKE and git_dirty():
        raise SystemExit("REFUSING: registered birth on a dirty tree")

    dev = DEVICE_OVERRIDE or ("mps" if torch.backends.mps.is_available() else
                              "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = TM.build_model(len(tok.vocab), d=384, layers=8,
                           heads=6, ffn=1536).to(dev)
    assert len(tok.vocab) == 40, len(tok.vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=PEAK_LR, total_steps=steps_total, pct_start=0.03)
    print(f"[dfa] steps_total {steps_total} "
          f"({steps_per_epoch}/epoch) peak lr {PEAK_LR:g}", flush=True)

    cap = min(SMOKE_STEPS, steps_total) if SMOKE else steps_total
    schedule = [s for s in SCHEDULE if s <= cap]
    if cap not in schedule:
        schedule.append(cap)
    OUTDIR.mkdir(parents=True, exist_ok=False)
    RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    started = now()
    init_digest = state_digest(model.state_dict())
    if EMIT:
        save_snapshot(model, 0, snapshots)
        assert snapshots["0"]["state_digest"] == init_digest, "step_0 file digest v in-memory init"

    feedback = None
    if MODE == "dfa":
        Bs_cpu = build_feedback(S)
        fb_p = OUTDIR / "feedback.pt"
        torch.save({"s": S, "seed_base": 31_000_000, "B": Bs_cpu}, fb_p)
        back = torch.load(fb_p, map_location="cpu")["B"]
        feedback = {"s": S, "digest": feedback_digest(back), "file_sha256": sha256_file(fb_p),
                    "shapes": [list(b.shape) for b in back], "seed_base": 31_000_000}
        Bs = [b.to(dev) for b in back]
        print(f"[dfa] feedback digest {feedback['digest'][:16]} s={S:g}", flush=True)

    step = 0
    t0 = time.time()
    done = False
    nonfinite_step = None
    for ep in range(EPOCHS):
        stream = streams[ep]
        print(f"[dfa] ep{ep}: {len(stream)} batches", flush=True)
        for a, b in stream:
            batch = enc[a:b]
            L = max(len(s) for s in batch)
            ids = torch.tensor([s + [tok.pad_id] * (L - len(s))
                                for s in batch], device=dev)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                                 for s in batch], device=dev)
            labels = ids[:, 1:].clone()
            labels[mask[:, 1:] == 0] = -100
            if MODE == "bp":
                logits = model(ids[:, :-1], mask[:, :-1])
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1), ignore_index=-100)
                loss.backward()
            else:
                ob = dfa_objective(model, Bs, ids[:, :-1], mask[:, :-1], labels)
                loss = ob["loss"]
                ob["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if GRADDUMP and step == 0:
                grads = {n: p.grad.detach().to("cpu").clone() for n, p in model.named_parameters()}
            opt.step()
            if sched.last_epoch < steps_total - 1:
                sched.step()
            opt.zero_grad()
            step += 1
            if GRADDUMP and step == 1:
                torch.save({"grads": grads, "params": {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()},
                            "loss": float(loss.detach()), "mode": MODE, "seed": SEED, "s": S, "lr": PEAK_LR,
                            "batch": [a, b], "lr_at_step1": opt.param_groups[0]["lr"]}, GRADDUMP)
            if not torch.isfinite(loss.detach()):
                if QUAL:
                    nonfinite_step = step
                    print(f"[dfa] UNSTABLE: non-finite loss at step {step}", flush=True)
                    done = True
                    break
                raise SystemExit(f"ABORT: non-finite loss at step {step}")
            if step % 200 == 0:
                print(f"  step {step}/{steps_total} loss "
                      f"{float(loss.detach()):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)",
                      flush=True)
            if EMIT and step in schedule:
                save_snapshot(model, step, snapshots)
            if step >= cap:
                done = True
                break
        if done:
            break
    wall = time.time() - t0

    row = {"kind": "birth", "prereg": "WRITER-DFA-1", "phase": PHASE, "mode": MODE, "seed": SEED, "cell": CELL,
           "s": S, "peak_lr": PEAK_LR, "steps": step, "steps_total": steps_total,
           "smoke": SMOKE, "emit": EMIT, "device": dev, "tag": TAG,
           "started_utc": started, "ended_utc": now(), "wall_s": round(wall, 1),
           "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "torch": torch.__version__, "vocab_len": len(tok.vocab),
           "enc_stock": len(enc_stock), "enc_train": len(enc), "stream_sha256": digests,
           "schedule": schedule, "outdir": str(OUTDIR), "init_state_digest": init_digest,
           "feedback": feedback, "snapshots": snapshots}
    if nonfinite_step is not None:
        row.update({"stable": False, "nonfinite_step": nonfinite_step, "final": None})
        write_receipt(row, launch_commit)
        print(f"[dfa] receipt appended to {RECEIPTS} (UNSTABLE at step {nonfinite_step}; no final checkpoint)", flush=True)
        return

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
    print(f"[dfa] saved {final_p} after {step} steps ({wall:.0f}s)", flush=True)

    h = hashlib.sha256()
    for k, v in sorted(((str(k), v) for k, v in opt.state_dict()["state"].items()), key=lambda kv: kv[0]):
        for name in sorted(v):
            t = v[name]
            h.update(f"{k}.{name}".encode())
            h.update(t.detach().to("cpu", torch.float32).contiguous().numpy().tobytes() if torch.is_tensor(t) else str(t).encode())
    row.update({"stable_training": True, "final": final_rec, "final_loss": float(loss.detach()),
                "optimizer_state_digest": h.hexdigest(),
                "torch_rng_digest": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()})
    write_receipt(row, launch_commit)
    print(f"[dfa] receipt appended to {RECEIPTS} (final state_digest {final_rec['state_digest'][:16]})", flush=True)


if __name__ == "__main__":
    main()

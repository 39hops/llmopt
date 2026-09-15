"""OPTIMIZER-MEMORY-ABLATION-1 instrument (PRE-REG RESULTS L74310, AMENDMENT
-PRE-INSTRUMENT L74620): selective first-moment erasure at a stored interior
milestone of the stock-OneCycle (A) and backward-SequenceLR (B) writers,
continued on the byte-identical future stream.

Three registered modes (MODE env), each refuse-if-exists on its receipt:
  desk-bar1  zero training: bind both anchors, one gradient on the real
             step-7201 batch, the BAR-1 expectation n_pred =
             ||a_carry_given_batch|| / ||u_C|| per writer (no optimizer
             step, no Z / E state).            -> logs/oma1/desk_bar1.json
  stage0     native-state preconditions: C and C' (CPU, deterministic)
             and MC_a / MC_b (mps) legs of LEG steps per writer; P0.a
             bit-exact repeat, P0.b scheduler parity, P0.c first-step
             real-v-virtual parity, P0.d endpoint envelope v the booked
             target.                             -> logs/oma1/stage0.json
  stage1     Z (exp_avg <- 0) and E (exp_avg <- 0.99 exp_avg) legs on CPU
             from the same bound state, readouts at the horizons, HELD-32
             CE, the 120 gate at the leg's end (descriptive), the sealed
             ladder.                             -> logs/oma1/stage1.json
Stage 1 refuses unless stage0.json exists with verdict PASS at the same
source sha. Stage 1 is a deterministic CPU continuation from an
mps-produced checkpoint; P0.d tests that substrate transfer.

Laws (asserted before any step): the future stream is the trainer's own
(scripts/train_mathnative.py, ORDER_SEED 0): enc = length-sorted encode
of the D2-excised gen4 diet, starts = BS-slices, epoch order =
random.Random(ep).shuffle(starts), step k (1-based) sits at epoch
(k - 1) // E, position (k - 1) % E with E = len(enc) // BS. The
scheduler is rebuilt fresh and stepped (anchor - 1) times on the live
optimizer, must equal the serialized group exactly, and its pending
step must yield the audited row anchor + 1 (OGD0 reconstruction law).
The optimizer is bound by OGD0's bind_state (module order asserted).
No torch RNG is consumed after init (no Dropout modules; asserted).

SMOKE=1: a synthetic non-target anchor (seed-6 W_0 + WARM real steps
on the epoch-0 stream, stock scheduler) under checkpoints/oma1_smoke/,
a stand-in "booked" target (native continuation on the smoke device),
LEG 3 with horizons 1 / 2 / 3, all three modes in sequence, receipts
logs/oma1/smoke<SMOKE_TAG>_*.json. Never writes a real path.

Usage: MODE=desk-bar1 .venv/bin/python scratch/optimizer_memory_ablation.py
       SMOKE=1 SMOKE_TAG=mech .venv/bin/python scratch/optimizer_memory_ablation.py
"""
import copy
import datetime
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
import update_geometry_census as UG  # noqa: E402  (flatten law, probe law, gradient law)
import optimizer_geometry_desk as OG  # noqa: E402  (bind_state, opt_state_digest, sched_rows, clip law)
import onecycle_component_audit as OA  # noqa: E402  (verbatim SequenceLR / _stock_lr_sequence, asserted)
from atomtraj_pins import state_digest  # noqa: E402
from llmopt.lab.gate import gate_eval  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")
MODE = os.environ.get("MODE", "smoke" if SMOKE else "")
THREADS = int(os.environ.get("OMA_THREADS", "8"))
LR, WD, PCT, TOTAL, BS = 3e-4, 0.01, 0.03, 15_420, 32
ANCHOR = 3 if SMOKE else 7200
LEG = 3 if SMOKE else 900
TARGET = ANCHOR + LEG
HORIZONS = [1, 2, 3] if SMOKE else [1, 5, 20, 100, 900]
EPS_TWIN = 0.01                       # E: exp_avg <- (1 - EPS_TWIN) exp_avg
WARM = 3                              # smoke anchor: real warm-up steps on the seed-6 W_0
# sealed thresholds (literal; AMENDMENT -PRE-INSTRUMENT L74620)
BAR1_TOL, BAR1_BAND = 0.02, (0.80, 1.00)
FORGOTTEN, PERSISTENT = 0.05, 0.25
SENSITIVE = 0.10
CE_FLOOR, CE_MULT = 0.005, 3.0
ENV_MULT, ENV_FLOOR, ENV_CAP, ENV_CE = 2.0, 0.02, 0.25, 0.01
P0C_OUTER = 5e-2                      # float32 real step v float64 virtual law (OGD0 envelope)
P0C_EXACT = 1e-6                      # float64 replay step v the law
_FAM = {"A": "checkpoints/oma1_smoke/phase19m" if SMOKE else "checkpoints/phase19m", "B": "checkpoints/oma1_smoke/backsched19m" if SMOKE else "checkpoints/backsched19m"}
WRITERS = {"A": {"kind": "stock", "anchor": f"{_FAM['A']}/m{ANCHOR:06d}.pt", "target": f"{_FAM['A']}/m{TARGET:06d}.pt"},
           "B": {"kind": "forward" if SMOKE else "backward", "anchor": f"{_FAM['B']}/m{ANCHOR:06d}.pt", "target": f"{_FAM['B']}/m{TARGET:06d}.pt"}}
# SMOKE B uses the SequenceLR path with the UNREVERSED sequence: the reversed sequence's first rows are the anneal floor
# (lr about 1e-9), where a float32 update is below weight resolution and no parity can be read.
for _w in WRITERS.values():
    assert ("oma1_smoke" in _w["anchor"]) == SMOKE, "smoke anchors must live under checkpoints/oma1_smoke only"
UGC0 = Path("logs/ugc0/census.json")
OUT_DIR = Path("logs/oma1")
CK_DIR = Path("checkpoints/oma1_smoke" if SMOKE else "checkpoints/oma1")
RECEIPT = {"desk-bar1": "desk_bar1.json", "stage0": "stage0.json", "stage1": "stage1.json"}


def receipt_path(mode):
    return OUT_DIR / (f"smoke{SMOKE_TAG}_{RECEIPT[mode]}" if SMOKE else RECEIPT[mode])


def stream_path(mode):
    return OUT_DIR / (f"smoke{SMOKE_TAG}_{mode}.jsonl" if SMOKE else f"{mode}.jsonl")


# ---------------------------------------------------------------- stream law
def epoch_position(step, n_enc, bs=BS):
    """1-based trainer step -> (epoch, position) under the stock nopack loop (E = n_enc // bs steps per epoch)."""
    E = n_enc // bs
    assert step >= 1 and E > 0
    return (step - 1) // E, (step - 1) % E


def epoch_order(starts, ep):
    idx = list(starts)
    random.Random(ep).shuffle(idx)      # the trainer's per-epoch shuffle (ORDER_SEED 0)
    return idx


def leg_slices(starts, n_enc, first_step, n_steps):
    """The (start, end) enc slices consumed by steps first_step .. first_step + n_steps - 1, epoch crossings handled."""
    out, cache = [], {}
    for k in range(first_step, first_step + n_steps):
        ep, pos = epoch_position(k, n_enc)
        if ep not in cache:
            cache[ep] = epoch_order(starts, ep)
        out.append(tuple(cache[ep][pos]))
    return out


def future_stream(tok):
    import birth19m_curric as C
    rows = C.load_excised_rows()
    enc, _levels = C.encode_with_levels(rows, tok)
    starts = [(i, i + BS) for i in range(0, len(enc) - BS, BS)]
    chosen = random.Random(UG.PROBE_SEED).sample(starts, 64)
    digest = hashlib.sha256(json.dumps({"slices": chosen, "ids": [[enc[j] for j in range(a, b)] for a, b in chosen]}).encode()).hexdigest()
    return enc, starts, {"n_enc": len(enc), "n_rows": len(rows), "steps_per_epoch": len(enc) // BS, "probe64_digest": digest}


def batch_tensors(tok, batch, dev):
    L = max(len(s) for s in batch)
    ids = torch.tensor([s + [tok.pad_id] * (L - len(s)) for s in batch], device=dev)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in batch], device=dev)
    return ids, mask


def loss_of(model, ids, mask):
    logits = model(ids[:, :-1], mask[:, :-1])
    labels = ids[:, 1:].clone()
    labels[mask[:, 1:] == 0] = -100
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


# ---------------------------------------------------------------- scheduler law
def make_sched(kind, opt):
    if kind == "stock":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=TOTAL, pct_start=PCT)
    seq = OA._stock_lr_sequence(opt, LR, TOTAL, PCT)
    if kind == "backward":
        seq = seq[::-1]
    for g in opt.param_groups:
        g["lr"] = LR          # SequenceLR divides by the group lr at construction; the birth constructed it at the trainer's LR
    return OA.SequenceLR(opt, seq)


def group_record(opt):
    g = opt.param_groups[0]
    return {"lr": float(g["lr"]), "beta1": float(g["betas"][0]), "beta2": float(g["betas"][1]), "wd": float(g["weight_decay"])}


def resume_sched(kind, opt, anchor, serialized):
    """Fresh scheduler stepped (anchor - 1) times must equal the serialized group EXACTLY; then the pending step (the tail of
    step `anchor`) is taken and the group for step anchor + 1 is returned. Mismatch raises before any gradient."""
    sched = make_sched(kind, opt)
    for _ in range(anchor - 1):
        sched.step()
    got = group_record(opt)
    if got != serialized:
        raise SystemExit(f"SCHEDULER PARITY FAILED at anchor {anchor}: reconstructed {got} v serialized {serialized}")
    if sched.last_epoch < TOTAL - 1:
        sched.step()
    return sched, group_record(opt)


# ---------------------------------------------------------------- arms
def apply_arm(opt, arm):
    """C: native. Z: selective first-moment erasure (exp_avg <- 0; exp_avg_sq, step counter, weights untouched).
    E: exp_avg <- (1 - EPS_TWIN) exp_avg. Returns the number of tensors touched."""
    if arm in ("C", "Cprime", "MC_a", "MC_b"):
        return 0
    n = 0
    with torch.no_grad():
        for p in opt.param_groups[0]["params"]:
            st = opt.state[p]
            if arm == "Z":
                st["exp_avg"].zero_()
            elif arm == "E":
                st["exp_avg"].mul_(1.0 - EPS_TWIN)
            else:
                raise ValueError(arm)
            n += 1
    return n


def assert_no_dropout(model):
    bad = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout)]
    assert not bad, f"Dropout modules present: {bad}"


def sd_cpu(model):
    return {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}


def flat(sd, segs, d):
    x = np.empty(d, dtype=np.float64)
    for k, a, b, _ in segs:
        x[a:b] = sd[k].to(torch.float64).reshape(-1).numpy()
    return x


def sd_equal(sd1, sd2):
    return all(torch.equal(sd1[k], sd2[k]) for k in sd1) and set(sd1) == set(sd2)


def run_leg(model, opt, sched, tok, enc, slices, dev, horizons, on_step=None):
    """The trainer's loop verbatim (backward, clip 1.0, opt.step, guarded sched.step, zero_grad) over `slices`; snapshots at
    the horizons (1-based step count within the leg). Returns {h: model sd (cpu)}, per-step losses, opt digest at the end."""
    snaps, losses = {}, []
    model.train()
    for i, (a, b) in enumerate(slices, start=1):
        ids, mask = batch_tensors(tok, enc[a:b], dev)
        loss = loss_of(model, ids, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if sched.last_epoch < TOTAL - 1:
            sched.step()
        opt.zero_grad()
        losses.append(float(loss.detach()))
        if i in horizons:
            snaps[i] = sd_cpu(model)
        if on_step:
            on_step(i, losses[-1])
    return snaps, losses, OG.opt_state_digest(opt)


@torch.no_grad()
def held_ce(model, tok, held, dev):
    model.eval()
    vals = [float(loss_of(model, *batch_tensors(tok, b, dev))) for b in held]
    model.train()
    return float(np.mean(vals)), vals


# ---------------------------------------------------------------- BAR-1 law
def clipped_grad(model, tok, batch, dev, segs, d):
    """The trainer's gradient on one batch (backward from the trainer's loss), flattened under the law in the MODEL's dtype
    carried to float64 (a float64 model gives a float64 gradient; UG.grad_row would round it to float32), then clipped."""
    ids, mask = batch_tensors(tok, batch, dev)
    model.zero_grad(set_to_none=True)
    loss = loss_of(model, ids, mask)
    loss.backward()
    params = dict(model.named_parameters())
    g = np.empty(d, dtype=np.float64)
    for k, a, b, _ in segs:
        gr = params[k].grad
        assert gr is not None, k
        g[a:b] = gr.detach().to("cpu", torch.float64).reshape(-1).numpy()
    model.zero_grad(set_to_none=True)
    norm = float(np.linalg.norm(g))
    return g * OG.clip_coef(norm), norm, float(loss.detach())


def bar1_law(model, opt, c, segs, step_next, grp, eps=1e-8):
    """a_carry_given_batch, u_C, a_C, a_Z per the amended law (float64, GLOBAL). Returns the norms and n_pred plus the c = 0
    approximation (OGD0's a_0 / u_0 with the zero-gradient denominator) for the record."""
    names = [n for n, _ in model.named_parameters()]
    pidx = {n: i for i, n in enumerate(names)}
    params = opt.param_groups[0]["params"]
    lr, b1, b2, wd = grp["lr"], grp["beta1"], grp["beta2"], grp["wd"]
    bc1, bc2 = 1.0 - b1 ** step_next, 1.0 - b2 ** step_next
    acc = {k: 0.0 for k in ("carry", "uC", "uZ", "aC", "aZ", "a0", "u0", "diff")}
    for k, a, b, _ in segs:
        p = params[pidx[k]]
        st = opt.state[p]
        W = p.detach().to("cpu", torch.float64).reshape(-1).numpy()
        m = st["exp_avg"].detach().to("cpu", torch.float64).reshape(-1).numpy()
        v = st["exp_avg_sq"].detach().to("cpu", torch.float64).reshape(-1).numpy()
        cc = c[a:b]
        v1 = b2 * v + (1 - b2) * cc * cc
        D = np.sqrt(v1 / bc2) + eps
        aC = -lr * (b1 * m + (1 - b1) * cc) / (bc1 * D)
        aZ = -lr * ((1 - b1) * cc) / (bc1 * D)
        carry = -lr * b1 * m / (bc1 * D)
        decay = -lr * wd * W
        D0 = np.sqrt(b2 * v / bc2) + eps
        a0 = -lr * (b1 * m / bc1) / D0
        acc["carry"] += float(carry @ carry); acc["uC"] += float((decay + aC) @ (decay + aC)); acc["uZ"] += float((decay + aZ) @ (decay + aZ))
        acc["aC"] += float(aC @ aC); acc["aZ"] += float(aZ @ aZ); acc["a0"] += float(a0 @ a0); acc["u0"] += float((decay + a0) @ (decay + a0))
        acc["diff"] += float((aC - aZ - carry) @ (aC - aZ - carry))
    n = {k: float(np.sqrt(v)) for k, v in acc.items()}
    return {"norms": n, "n_pred": n["carry"] / n["uC"], "n_c0_approx": n["u0"] / n["uC"], "identity_residual": n["diff"] / max(n["carry"], 1e-300),
            "step_next": step_next, "group_next": grp}


def virtual_u(model, opt, c, segs, d, step_next, grp, eps=1e-8):
    """The native (C) update u_C per tensor under the virtual law, flattened float64 (for P0.c)."""
    names = [n for n, _ in model.named_parameters()]
    pidx = {n: i for i, n in enumerate(names)}
    params = opt.param_groups[0]["params"]
    lr, b1, b2, wd = grp["lr"], grp["beta1"], grp["beta2"], grp["wd"]
    bc1, bc2 = 1.0 - b1 ** step_next, 1.0 - b2 ** step_next
    u = np.empty(d, dtype=np.float64)
    for k, a, b, _ in segs:
        p = params[pidx[k]]; st = opt.state[p]
        W = p.detach().to("cpu", torch.float64).reshape(-1).numpy()
        m = st["exp_avg"].detach().to("cpu", torch.float64).reshape(-1).numpy()
        v = st["exp_avg_sq"].detach().to("cpu", torch.float64).reshape(-1).numpy()
        cc = c[a:b]
        m1 = b1 * m + (1 - b1) * cc; v1 = b2 * v + (1 - b2) * cc * cc
        u[a:b] = -lr * wd * W - lr * (m1 / bc1) / (np.sqrt(v1 / bc2) + eps)
    return u


# ---------------------------------------------------------------- readouts and the ladder
def readout(W_anchor, W_C, W_X, segs):
    dX = W_X - W_C; dC = W_C - W_anchor; dXa = W_X - W_anchor
    n = float(np.linalg.norm(dX) / max(np.linalg.norm(dC), 1e-300))
    cos = float(dXa @ dC / max(np.linalg.norm(dXa) * np.linalg.norm(dC), 1e-300))
    tot = float(dX @ dX)
    share = {}
    for grp in sorted(set(s[3] for s in segs)):
        e = sum(float(dX[a:b] @ dX[a:b]) for k, a, b, g in segs if g == grp)
        share[grp] = e / max(tot, 1e-300)
    return {"n": n, "leg_cos": cos, "abs": float(np.linalg.norm(dX)), "group_share": share}


def adjudicate(w):
    """w: per-writer dict {n_Z: {h: v}, n_E: {h: v}, dCE_Z: {h: v}, dCE_E: {h: v}, n_pred: float, h_end: int}. Pure; literal thresholds; inclusive."""
    hE = w["h_end"]
    out = {}
    nz1 = w["n_Z"][1]
    out["bar1"] = "PASS" if (abs(nz1 - w["n_pred"]) <= BAR1_TOL and BAR1_BAND[0] <= nz1 <= BAR1_BAND[1]) else "INSTRUMENT-FAULT"
    nz, ne = w["n_Z"][hE], w["n_E"][hE]
    out["path"] = "FORGOTTEN" if nz <= FORGOTTEN else ("PERSISTENT" if nz >= PERSISTENT else "INTERMEDIATE")
    out["sensitivity"] = "SENSITIVE" if ne >= SENSITIVE else "SPECIFIC"
    dz, de = w["dCE_Z"][hE], w["dCE_E"][hE]
    bound = max(CE_FLOOR, CE_MULT * abs(de))
    out["function"] = "NEUTRAL" if abs(dz) <= bound else ("HARMED" if dz > 0 else "HELPED")
    out["r"] = {str(h): (w["n_Z"][h] / (100.0 * w["n_E"][h]) if w["n_E"][h] > 0 else None) for h in w["n_Z"]}
    out["argmin_h_nZ"] = min(w["n_Z"], key=lambda h: w["n_Z"][h])
    return out


def program_label(per_writer):
    labs = {w: (a["path"], a["sensitivity"], a["function"]) for w, a in per_writer.items()}
    if any(a["bar1"] != "PASS" for a in per_writer.values()):
        return "INSTRUMENT-FAULT"
    if len(set(labs.values())) == 1:
        p, s, f = next(iter(labs.values()))
        return f"{p}-{s}+FUNCTION-{f}"
    return "MIXED-BY-WRITER"


def p0d(rho_cpu, rho_env, dce):
    if rho_env > ENV_CAP:
        return "NOT-ADJUDICABLE"
    ok = rho_cpu <= max(ENV_MULT * rho_env, ENV_FLOOR) and rho_cpu <= ENV_CAP and abs(dce) <= ENV_CE
    return "PASS" if ok else "FAIL"


# ---------------------------------------------------------------- provenance
def git_state():
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    return commit, dirty


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def base_record(mode, tok, stream_info, dev_cpu, dev_mps):
    commit, dirty = git_state()
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered mode on a dirty tree")
    return {"prereg": "OPTIMIZER-MEMORY-ABLATION-1", "kind": f"optimizer_memory_ablation/{mode}", "smoke": SMOKE, "smoke_tag": SMOKE_TAG, "commit": commit, "tree_dirty": dirty,
            "source_sha256": UG.sha256_file(__file__), "ugc0_source_sha256": UG.sha256_file(UG.__file__), "ogd0_source_sha256": UG.sha256_file(OG.__file__), "audit_source_sha256": UG.sha256_file(OA.__file__),
            "torch_version": torch.__version__, "numpy_version": np.__version__, "threads": torch.get_num_threads(), "deterministic": torch.are_deterministic_algorithms_enabled(),
            "device_cpu": dev_cpu, "device_mps": dev_mps, "anchor": ANCHOR, "leg": LEG, "target": TARGET, "horizons": HORIZONS, "eps_twin": EPS_TWIN,
            "law": {"BAR1_TOL": BAR1_TOL, "BAR1_BAND": BAR1_BAND, "FORGOTTEN": FORGOTTEN, "PERSISTENT": PERSISTENT, "SENSITIVE": SENSITIVE, "CE_FLOOR": CE_FLOOR, "CE_MULT": CE_MULT,
                    "ENV_MULT": ENV_MULT, "ENV_FLOOR": ENV_FLOOR, "ENV_CAP": ENV_CAP, "ENV_CE": ENV_CE, "P0C_OUTER": P0C_OUTER, "P0C_EXACT": P0C_EXACT,
                    "bar1": "n_pred = ||-lr b1 m / (bc1 (sqrt(v'/bc2) + eps))|| / ||u_C||, v' = b2 v + (1 - b2) c^2, c = clipped real batch gradient, s = anchor + 1"},
            "stream": stream_info, "writers": {w: {"kind": s["kind"], "anchor": s["anchor"], "target": s["target"]} for w, s in WRITERS.items()}, "started_utc": now()}


def bind(writer, tok, dev):
    """Model + optimizer from the anchor milestone (OGD0 bind_state), serialized group captured BEFORE any scheduler exists."""
    path = WRITERS[writer]["anchor"]
    ck = torch.load(path, map_location="cpu")
    assert int(ck["step"]) == ANCHOR, (ck["step"], ANCHOR)
    model = UG.build(tok, dev)
    assert_no_dropout(model)
    opt, rec = OG.bind_state(model, ck, dev)
    assert rec["adam_step"] == ANCHOR
    serialized = {"lr": rec["lr"], "beta1": rec["beta1"], "beta2": rec["beta2"], "wd": rec["weight_decay"]}
    info = {"path": path, "file_sha256": UG.sha256_file(path), "state_digest": state_digest(ck["model"]), "opt_state_digest_bound": OG.opt_state_digest(opt), "group": rec, "serialized": serialized}
    return model, opt, ck, info


def load_target(writer):
    path = WRITERS[writer]["target"]
    ck = torch.load(path, map_location="cpu")
    assert int(ck["step"]) == TARGET, (ck["step"], TARGET)
    return ck["model"], {"path": path, "file_sha256": UG.sha256_file(path), "state_digest": state_digest(ck["model"])}


# ---------------------------------------------------------------- modes
def mode_desk_bar1(tok, enc, starts, info, segs, d, dev_cpu, rec, stream):
    slices = leg_slices(starts, info["n_enc"], ANCHOR + 1, 1)
    rec["batch_slice"] = slices[0]
    rec["cells"] = {}
    for w in WRITERS:
        model, opt, ck, binfo = bind(w, tok, dev_cpu)
        _sched, grp_next = resume_sched(WRITERS[w]["kind"], opt, ANCHOR, binfo["serialized"])
        if not SMOKE:
            _used, nxt = OG.sched_rows(WRITERS[w]["kind"], ANCHOR)
            assert grp_next == {"lr": nxt["lr"], "beta1": nxt["beta1"], "beta2": nxt["beta2"], "wd": nxt["wd"]}, (grp_next, nxt)
        a, b = slices[0]
        c, gnorm, loss = clipped_grad(model, tok, enc[a:b], dev_cpu, segs, d)
        law = bar1_law(model, opt, c, segs, ANCHOR + 1, grp_next)
        assert law["identity_residual"] < 1e-9, law["identity_residual"]
        assert OG.opt_state_digest(opt) == binfo["opt_state_digest_bound"], "optimizer state changed during the desk (no step may occur)"
        cell = {"bind": binfo, "grad_norm": gnorm, "clip_coef": OG.clip_coef(gnorm), "batch_loss": loss, "law": law, "opt_steps_taken": 0}
        rec["cells"][w] = cell
        stream({"writer": w, "n_pred": law["n_pred"], "n_c0_approx": law["n_c0_approx"], "norms": law["norms"]})
        print(f"[oma1 desk-bar1] {w}: n_pred {law['n_pred']:.4f} (c=0 approx {law['n_c0_approx']:.4f}) |carry| {law['norms']['carry']:.3e} |u_C| {law['norms']['uC']:.3e}", flush=True)
    return rec


def snapshot_dir(writer, arm):
    p = CK_DIR / writer / arm
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_snap(writer, arm, h, sd, opt=None):
    p = snapshot_dir(writer, arm) / f"h{h:04d}.pt"
    if p.exists():
        raise SystemExit(f"REFUSING: {p} exists")
    blob = {"model": sd, "step": ANCHOR + h, "arm": arm, "writer": writer}
    if opt is not None:
        blob["opt"] = copy.deepcopy(opt.state_dict())
    torch.save(blob, p)
    return {"path": str(p), "sha256": UG.sha256_file(str(p)), "state_digest": state_digest(sd)}


def one_leg(writer, arm, tok, enc, slices, dev, segs, d, stream, rec_cell, keep_opt_at_end=False):
    model, opt, ck, binfo = bind(writer, tok, dev)
    sched, grp_next = resume_sched(WRITERS[writer]["kind"], opt, ANCHOR, binfo["serialized"])
    touched = apply_arm(opt, arm)
    digest_after_arm = OG.opt_state_digest(opt)
    t0 = time.time()
    snaps, losses, opt_digest_end = run_leg(model, opt, sched, tok, enc, slices, dev, HORIZONS,
                                            on_step=lambda i, l: stream({"writer": writer, "arm": arm, "device": dev, "step": ANCHOR + i, "loss": l}))
    wall = time.time() - t0
    rec_cell.update({"bind": binfo, "group_next": grp_next, "arm": arm, "device": dev, "tensors_touched": touched, "opt_state_digest_after_arm": digest_after_arm,
                     "opt_state_digest_end": opt_digest_end, "wall_s": round(wall, 1), "it_per_s": round(len(slices) / max(wall, 1e-9), 2), "losses_first5": losses[:5], "loss_last": losses[-1]})
    rec_cell["snapshots"] = {str(h): save_snap(writer, arm, h, sd, opt if (keep_opt_at_end and h == LEG) else None) for h, sd in snaps.items()}
    return model, opt, snaps, binfo


def mode_stage0(tok, enc, starts, info, segs, d, dev_cpu, dev_mps, held, rec, stream):
    slices = leg_slices(starts, info["n_enc"], ANCHOR + 1, LEG)
    rec["leg_slices_digest"] = hashlib.sha256(json.dumps(slices).encode()).hexdigest()
    rec["leg_first_slice"], rec["leg_last_slice"] = slices[0], slices[-1]
    rec["cells"], verdicts = {}, {}
    for w in WRITERS:
        cell = {}
        # P0.c: first real CPU step v the virtual law on the same gradient (float32 continuation) and a float64 replay step
        model, opt, ck, binfo = bind(w, tok, dev_cpu)
        _s, grp_next = resume_sched(WRITERS[w]["kind"], opt, ANCHOR, binfo["serialized"])
        if not SMOKE:
            _used, nxt = OG.sched_rows(WRITERS[w]["kind"], ANCHOR)
            assert grp_next == {"lr": nxt["lr"], "beta1": nxt["beta1"], "beta2": nxt["beta2"], "wd": nxt["wd"]}, (grp_next, nxt)
        W0 = flat(sd_cpu(model), segs, d)
        a, b = slices[0]
        c, gnorm, _loss = clipped_grad(model, tok, enc[a:b], dev_cpu, segs, d)
        u_virt = virtual_u(model, opt, c, segs, d, ANCHOR + 1, grp_next)
        law = bar1_law(model, opt, c, segs, ANCHOR + 1, grp_next)
        cell["bar1_law"] = law
        # float64 replay of the same step: the float64 model's own gradient feeds both the virtual law and a real torch step
        m64 = copy.deepcopy(model).to(torch.float64); o64 = OG.build_opt(m64); o64.load_state_dict(copy.deepcopy(opt.state_dict()))   # load_state_dict shares tensors with its source
        for p in o64.param_groups[0]["params"]:
            for kk in ("exp_avg", "exp_avg_sq"):
                o64.state[p][kk] = o64.state[p][kk].to(torch.float64)
        o64.param_groups[0].update({"lr": grp_next["lr"], "betas": (grp_next["beta1"], grp_next["beta2"])})
        c64, _n64, _l64 = clipped_grad(m64, tok, enc[a:b], dev_cpu, segs, d)
        u_virt64 = virtual_u(m64, o64, c64, segs, d, ANCHOR + 1, grp_next)
        W0_64 = flat(sd_cpu(m64), segs, d)
        ids, mask = batch_tensors(tok, enc[a:b], dev_cpu)
        loss_of(m64, ids, mask).backward(); torch.nn.utils.clip_grad_norm_(m64.parameters(), 1.0); o64.step()
        u64 = flat(sd_cpu(m64), segs, d) - W0_64
        cell["p0c_exact_float64"] = float(np.max(np.abs(u64 - u_virt64)) / np.max(np.abs(u_virt64)))
        cell["p0c_float64_v_float32_law"] = float(np.max(np.abs(u_virt64 - u_virt)) / np.max(np.abs(u_virt)))
        del m64, o64
        # C leg (float32 cpu) and its first step v the law
        cC = {}
        modelC, optC, snapsC, _ = one_leg(w, "C", tok, enc, slices, dev_cpu, segs, d, stream, cC, keep_opt_at_end=True)
        u_real = flat(snapsC[1], segs, d) - W0
        cell["p0c_float32"] = float(np.max(np.abs(u_real - u_virt)) / np.max(np.abs(u_virt)))
        cell["C"] = cC
        # C' repeat
        cP = {}
        _mP, _oP, snapsP, _ = one_leg(w, "Cprime", tok, enc, slices, dev_cpu, segs, d, stream, cP)
        cell["Cprime"] = cP
        cell["p0a_bit_exact"] = {str(h): bool(sd_equal(snapsC[h], snapsP[h])) for h in HORIZONS}
        cell["p0a_opt_digest_equal"] = cC["opt_state_digest_end"] == cP["opt_state_digest_end"]
        # mps native resumes
        cell["MC"] = {}
        snapsM = {}
        for arm in ("MC_a", "MC_b"):
            cm = {}
            _m, _o, sn, _ = one_leg(w, arm, tok, enc, slices, dev_mps, segs, d, stream, cm)
            cell["MC"][arm] = cm; snapsM[arm] = sn
        # P0.d
        tsd, tinfo = load_target(w)
        WT = flat(tsd, segs, d); WC = flat(snapsC[LEG], segs, d)
        den = float(np.linalg.norm(WT - W0))
        rho_cpu = float(np.linalg.norm(WC - WT) / den)
        rho_mps = {arm: float(np.linalg.norm(flat(snapsM[arm][LEG], segs, d) - WT) / den) for arm in snapsM}
        rho_env = max(rho_mps.values())
        tm = UG.build(tok, dev_cpu); tm.load_state_dict(tsd)
        ce_t, _ = held_ce(tm, tok, held, dev_cpu)
        ce_c, _ = held_ce(modelC, tok, held, dev_cpu)
        cell["target"] = tinfo
        cell["p0d"] = {"rho_cpu": rho_cpu, "rho_mps": rho_mps, "rho_env": rho_env, "leg_norm": den, "ce_held_C": ce_c, "ce_held_target": ce_t, "dce": ce_c - ce_t,
                       "mps_pair_distance_over_leg": float(np.linalg.norm(flat(snapsM["MC_a"][LEG], segs, d) - flat(snapsM["MC_b"][LEG], segs, d)) / den), "verdict": p0d(rho_cpu, rho_env, ce_c - ce_t)}
        cell["p0b_scheduler_parity"] = "PASS"
        cell["p0c_verdict"] = "PASS" if (cell["p0c_float32"] <= P0C_OUTER and cell["p0c_exact_float64"] <= P0C_EXACT) else "FAIL"
        cell["p0a_verdict"] = "PASS" if all(cell["p0a_bit_exact"].values()) and cell["p0a_opt_digest_equal"] else "FAIL"
        verdicts[w] = "PASS" if (cell["p0a_verdict"] == "PASS" and cell["p0c_verdict"] == "PASS" and cell["p0d"]["verdict"] == "PASS") else ("NOT-ADJUDICABLE" if cell["p0d"]["verdict"] == "NOT-ADJUDICABLE" else "FAIL")
        rec["cells"][w] = cell
        print(f"[oma1 stage0] {w}: P0.a {cell['p0a_verdict']} P0.c f32 {cell['p0c_float32']:.2e} f64 {cell['p0c_exact_float64']:.2e} P0.d rho_cpu {rho_cpu:.4f} rho_env {rho_env:.4f} dCE {ce_c - ce_t:+.4f} -> {verdicts[w]}", flush=True)
    rec["verdict_per_writer"] = verdicts
    rec["verdict"] = "PASS" if all(v == "PASS" for v in verdicts.values()) else ("NOT-ADJUDICABLE" if "NOT-ADJUDICABLE" in verdicts.values() else "FAIL")
    return rec


def mode_stage1(tok, enc, starts, info, segs, d, dev_cpu, held, rec, stream, stage0):
    assert stage0["verdict"] == "PASS", "Stage 0 did not PASS"
    assert stage0["source_sha256"] == rec["source_sha256"], "instrument changed since Stage 0"
    assert stage0["commit"] == rec["commit"] or SMOKE, "code commit changed since Stage 0"
    slices = leg_slices(starts, info["n_enc"], ANCHOR + 1, LEG)
    assert hashlib.sha256(json.dumps(slices).encode()).hexdigest() == stage0["leg_slices_digest"]
    dev_gate = "mps" if torch.backends.mps.is_available() else "cpu"
    rec["gate_device"] = dev_gate
    rec["cells"], per = {}, {}
    for w in WRITERS:
        cell = {"C_from_stage0": stage0["cells"][w]["C"]["snapshots"]}
        snapsC = {}
        for h in HORIZONS:
            blob = torch.load(cell["C_from_stage0"][str(h)]["path"], map_location="cpu")
            assert state_digest(blob["model"]) == cell["C_from_stage0"][str(h)]["state_digest"]
            snapsC[h] = blob["model"]
        anchor_sd = torch.load(WRITERS[w]["anchor"], map_location="cpu")["model"]
        W0 = flat(anchor_sd, segs, d)
        WC = {h: flat(snapsC[h], segs, d) for h in HORIZONS}
        mC = UG.build(tok, dev_cpu)
        ceC = {}
        for h in HORIZONS:
            mC.load_state_dict(snapsC[h]); ceC[h] = held_ce(mC, tok, held, dev_cpu)[0]
        cell["ce_held_C"] = {str(h): ceC[h] for h in HORIZONS}
        wrec = {"n_Z": {}, "n_E": {}, "dCE_Z": {}, "dCE_E": {}, "n_pred": stage0["cells"][w]["bar1_law"]["n_pred"], "h_end": LEG}
        for arm in ("Z", "E"):
            ca = {}
            model, opt, snaps, _ = one_leg(w, arm, tok, enc, slices, dev_cpu, segs, d, stream, ca, keep_opt_at_end=(arm == "Z"))
            assert ca["tensors_touched"] == len(UG.KEYS)
            ca["readout"] = {}; ca["ce_held"] = {}
            for h in HORIZONS:
                r = readout(W0, WC[h], flat(snaps[h], segs, d), segs)
                ca["readout"][str(h)] = r
                model.load_state_dict(snaps[h]); ce = held_ce(model, tok, held, dev_cpu)[0]
                ca["ce_held"][str(h)] = ce
                wrec[f"n_{arm}"][h] = r["n"]; wrec[f"dCE_{arm}"][h] = ce - ceC[h]
            cell[arm] = ca
        # descriptive gate at the leg's end (C and Z)
        gate = {}
        for arm, sd in (("C", snapsC[LEG]), ("Z", torch.load(cell["Z"]["snapshots"][str(LEG)]["path"], map_location="cpu")["model"])):
            gm = UG.build(tok, dev_gate); gm.load_state_dict(sd); gm.eval()
            solves, valid = gate_eval(gm, tok, dev_gate, n=(2 if SMOKE else None))
            gate[arm] = {"solves": solves, "total": int(sum(solves.values())), "valid_pct": round(valid, 2)}
        cell["gate"] = gate
        cell["adjudication"] = adjudicate(wrec)
        cell["numbers"] = {k: ({str(h): v for h, v in val.items()} if isinstance(val, dict) else val) for k, val in wrec.items()}
        per[w] = cell["adjudication"]
        rec["cells"][w] = cell
        print(f"[oma1 stage1] {w}: {cell['adjudication']} gate C {gate['C']['total']} Z {gate['Z']['total']}", flush=True)
    rec["program_label"] = program_label(per)
    return rec


# ---------------------------------------------------------------- smoke anchor (non-target, path-isolated)
def build_smoke_anchor(tok, enc, starts, info, dev):
    """seed-6 W_0 + WARM real steps of the epoch-0 stream under the stock scheduler -> checkpoints/oma1_smoke/*/m000003.pt
    ({model, opt, step} as the milestone tee writes), then LEG more native steps on `dev` -> the stand-in booked target."""
    for w in WRITERS:
        for key in ("anchor", "target"):
            p = Path(WRITERS[w][key])
            if p.exists():
                raise SystemExit(f"REFUSING: smoke artifact {p} exists")
            p.parent.mkdir(parents=True, exist_ok=True)
    model = UG.build(tok, dev); model.load_state_dict(UG.w0_seed(6)); assert_no_dropout(model)
    opt = OG.build_opt(model)
    for w in WRITERS:
        m = copy.deepcopy(model); o = OG.build_opt(m); sched = make_sched(WRITERS[w]["kind"], o)
        run_leg(m, o, sched, tok, enc, leg_slices(starts, info["n_enc"], 1, WARM - 1), dev, [])
        # step WARM as the milestone tee sees it: saved INSIDE opt.step, i.e. after the update and BEFORE that step's sched.step
        (a, b), = leg_slices(starts, info["n_enc"], WARM, 1)
        m.train(); loss_of(m, *batch_tensors(tok, enc[a:b], dev)).backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()
        torch.save({"model": sd_cpu(m), "opt": copy.deepcopy(o.state_dict()), "step": WARM}, WRITERS[w]["anchor"])
        if sched.last_epoch < TOTAL - 1:
            sched.step()
        o.zero_grad()
        run_leg(m, o, sched, tok, enc, leg_slices(starts, info["n_enc"], WARM + 1, LEG), dev, [])
        torch.save({"model": sd_cpu(m), "opt": copy.deepcopy(o.state_dict()), "step": WARM + LEG}, WRITERS[w]["target"])
    del opt


def main():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)
    torch.manual_seed(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev_cpu = "cpu"
    dev_mps = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = TM.MathTokenizer()
    assert len(tok.vocab) == 40 and not os.environ.get("VOCAB_EXTRA") and not os.environ.get("SEQ_CAP") and not os.environ.get("BIRTH_BS") and TM.BS == BS
    OA.assert_verbatim()
    probe_model = UG.build(tok, "cpu")
    segs, d, flat_digest = UG.flatten_law(probe_model)
    assert d == 18_911_616
    enc, starts, info = future_stream(tok)
    ugc0 = json.loads(UGC0.read_text())
    assert info["n_enc"] == 164_490 and info["steps_per_epoch"] == 5_140 and 3 * info["steps_per_epoch"] == TOTAL, info
    assert info["probe64_digest"] == ugc0["probe"]["digest"], "reconstructed enc v the booked UGC0 probe digest"
    info["flatten_law_digest"] = flat_digest
    if not SMOKE:
        ep, pos = epoch_position(ANCHOR + 1, info["n_enc"]); ep2, pos2 = epoch_position(TARGET, info["n_enc"])
        assert (ep, pos, ep2, pos2) == (1, 2060, 1, 2959), (ep, pos, ep2, pos2)
        info["leg"] = {"epoch": ep, "first_pos": pos, "last_pos": pos2}
    batches, probe = UG.probe_batches(tok)
    held = [batches[i] for i in probe["held"]]
    info["held_panel"] = {"n": len(held), "probe_digest": probe["digest"], "ugc0_digest": ugc0["probe"]["digest"]}
    if not SMOKE:
        assert probe["digest"] == ugc0["probe"]["digest"]
    modes = ["desk-bar1", "stage0", "stage1"] if SMOKE else [MODE]
    assert all(m in RECEIPT for m in modes), MODE
    for m in modes:
        if receipt_path(m).exists() or stream_path(m).exists():
            raise SystemExit(f"REFUSING: {receipt_path(m)} or {stream_path(m)} exists")
    if SMOKE:
        build_smoke_anchor(tok, enc, starts, info, dev_mps)
    stage0 = None
    for m in modes:
        rec = base_record(m, tok, info, dev_cpu, dev_mps)

        def stream(row, _m=m):
            with stream_path(_m).open("a") as f:
                f.write(json.dumps(row) + "\n")

        t0 = time.time()
        if m == "desk-bar1":
            rec = mode_desk_bar1(tok, enc, starts, info, segs, d, dev_cpu, rec, stream)
        elif m == "stage0":
            rec = mode_stage0(tok, enc, starts, info, segs, d, dev_cpu, dev_mps, held, rec, stream)
            stage0 = rec
        else:
            if stage0 is None:
                stage0 = json.loads(receipt_path("stage0").read_text())
            rec = mode_stage1(tok, enc, starts, info, segs, d, dev_cpu, held, rec, stream, stage0)
        rec["wall_s"] = round(time.time() - t0, 1); rec["ended_utc"] = now()
        receipt_path(m).write_text(json.dumps(rec, indent=1) + "\n")
        print(f"[oma1] {m} done in {rec['wall_s']} s -> {receipt_path(m)}", flush=True)


if __name__ == "__main__":
    main()

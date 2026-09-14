"""OPTIMIZER-GEOMETRY-DESK-0 (PRE-REG in RESULTS): zero-training desk on the
geometric transformation the actual AdamW writer applies between the task
gradient and the parameter write, at stored (W_s, optimizer_state_s) of the
WRITER-TRAJECTORY-CENSUS-0 pair. No persistent parameter update, no birth.

Specimens (existing milestones with {"model", "opt", "step"}): A = seed-2
stock-OneCycle writer (checkpoints/phase19m/m{s}.pt), B = seed-2 backward-
SequenceLR writer (checkpoints/backsched19m/m{s}.pt); interior states
s in {900, 7200, 13500} (frozen), so a next scheduler step exists.
Scheduler law: the ONECYCLE-SCHEDULER-COMPONENT-AUDIT-0 tables
(logs/schedaudit0/audit.json: stock for A, backward for B); the milestone's
serialized lr / betas must equal row s (values used for step s) EXACTLY,
then row s + 1 gives the next-step lr / beta1 / beta2 / weight_decay.

Probe panel: EXACTLY the UPDATE-GEOMETRY-CENSUS-0 population (64 stock
batches, FIT = first 32, HELD = last 32; the run asserts its digest equals
the booked logs/ugc0/census.json probe digest).

Gradient law: as UPDATE-GEOMETRY-CENSUS-0 (model.eval(), trainer CE,
p.grad, sorted-key flatten law, one float32 memmap of raw gradients per
specimen-state; state digest asserted before and after).

Optimizer reconstruction: AdamW(model.parameters(), lr=LR, weight_decay=
0.01) built with the trainer's constructor call; opt.load_state_dict(ckpt
["opt"]) binds the saved state to the live parameters; asserted: 59
parameters in the sorted-key order, exp_avg / exp_avg_sq shapes, every
step counter == s, one param group with the audited fields (eps 1e-8,
weight_decay 0.01, amsgrad / maximize false, decoupled weight decay).

Virtual step law (per probe batch i, every batch from the IDENTICAL
frozen state; nothing persists):
  g_i      raw gradient (memmap row)
  c_i      = g_i * min(1, 1.0 / (||g_i|| + 1e-6))      (clip_grad_norm_ 1.0, global)
  m'_i     = beta1 * m + (1 - beta1) * c_i               (Adam first moment after the step)
  v'_i     = beta2 * v + (1 - beta2) * c_i^2
  bc1, bc2 = 1 - beta1^(s+1), 1 - beta2^(s+1)
  a_i      = - lr * (m'_i / bc1) / (sqrt(v'_i / bc2) + eps)   (adaptive update, no decay)
  u_i      = - lr * wd * W + a_i                          (full AdamW write; decay first, as torch)
  u_0      = the same with c = 0 (zero gradient TENSORS: m' = beta1 m, v' = beta2 v)
  b_i      = u_i - u_0  (= a_i - a_0 exactly: the decay term is batch-independent)
  mhat_i   = m'_i / bc1                                   (first-moment candidate, descriptive)
with lr / beta1 / beta2 / wd = the next-step scheduler row, in float64
from the float32 stored state. Families are DERIVED per tensor block from
the raw memmap and the stored state inside the Gram pass; only the raw
memmap is stored (one per specimen-state; A and B of one state coexist
for the same-batch cross-Grams: 9.7 GB peak).

Geometry: the validated UPDATE-GEOMETRY-CENSUS-0 Gram machinery
(update_geometry_census: unit_gram, centered_gram, geometry_from_grams,
pair_overlap_from_grams, projection_fraction) on each family's 64 x 64
Gram: FIT / HELD reliability S_k, HELD capture C_k (k in 1, 2, 4, 8, 16),
participation ratio, Q, per-group energy (BLOCK0..7 + OUTSIDE), A / B
same-batch overlap; plus the DISTORTION readouts: D(x -> y) = ||K_y - K_x||_F
/ ||K_x||_F on unit-row cosine Grams for g -> mhat, mhat -> a, a -> u, g -> u,
g -> b, a -> b; per-batch cos(g_i, a_i), cos(g_i, u_i), cos(g_i, b_i), cos(g_i, mhat_i);
norm decomposition (||g||, ||c||, ||beta1 m|| carried, ||mhat||, ||a||,
||lr wd W||, ||a_0||, ||u_0||, ||b||, ||u||); velocity projections of the same-writer next-grid
displacement (s -> next: 900 -> 3600, 7200 -> 10800, 13500 -> 15420) onto the
FIT subspaces of g, u and b (own writer), descriptive only.

Parity endpoints (before any target read; SMOKE=1 on the non-target
seed-6 W_0): (i) synthetic: random tensors + random gradient through a
real torch AdamW.step v the virtual law; (ii) real: one real trainer step
on the seed-6 model (backward, clip, AdamW.step on the device) from a
saved pre-step state, v the virtual law applied to the pre-step state
and the same gradient, every model tensor, exp_avg, exp_avg_sq and step
counter compared at the registered tolerances; done at Adam step 1 and
after three warm-up steps (step 4) so bias corrections are exercised.

Ladder (thresholds literal; adjudicate is pure): see PRE-REG.
Outputs: logs/ogd0/desk.json (refuse-if-exists), logs/ogd0/desk.jsonl
(streamed), memmaps under logs/ogd0/tmp (deleted after the reductions).
SMOKE=1: parity endpoints + the pipeline on the seed-6 W_0 with a
synthetic optimizer state at Adam step 4, 8 probe batches, outputs
logs/ogd0/smoke<SMOKE_TAG>.jsonl. Usage: .venv/bin/python scratch/optimizer_geometry_desk.py
"""
import copy
import datetime
import hashlib
import json
import os
import shutil
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
import update_geometry_census as UG  # noqa: E402  (probe law, gradient law, flatten law, Gram geometry; its SMOKE flag is read from the same env)
from atomtraj_pins import state_digest  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")
LR, WD, PCT, TOTAL = 3e-4, 0.01, 0.03, 15_420
STATES = [900, 7200, 13500]
NEXT = {900: 3600, 7200: 10800, 13500: 15420}
A_PATHS = {s: f"checkpoints/phase19m/m{s:06d}.pt" for s in STATES}
B_PATHS = {s: f"checkpoints/backsched19m/m{s:06d}.pt" for s in STATES}
A_NEXT = {s: (f"checkpoints/phase19m/m{n:06d}.pt" if n < TOTAL else "checkpoints/gallery19m_phase_s2.pt") for s, n in NEXT.items()}
B_NEXT = {s: (f"checkpoints/backsched19m/m{n:06d}.pt" if n < TOTAL else "checkpoints/gallery19m_backsched_s2.pt") for s, n in NEXT.items()}
SCHED = Path("logs/schedaudit0/audit.json")
UGC0 = Path("logs/ugc0/census.json")
K_LIST = UG.K_LIST
K_LADDER = UG.K_LADDER
SHARPEN, SMALL, REL_MIN = 0.15, 0.05, 0.25          # ladder thresholds (literal)
SHARED_FRAC, ROTATED_FRAC = 0.8, 0.5
CLIP, CLIP_EPS = 1.0, 1e-6
PARITY_TOL_PARAM = 1e-9      # exact-law endpoint (float64 / cpu): max |realized - virtual| relative to max |virtual| per tensor, updates and moments
ENVELOPE_TOL = 5e-2          # float32 / device rounding envelope of the real writer: the realized update W_post - W_pre is quantized at 2^-24 |W| against an update of lr scale (descriptive; recorded)
OUT_DIR = Path("logs/ogd0")
TMP = OUT_DIR / ("tmp_smoke" if SMOKE else "tmp")
OUT = OUT_DIR / (f"smoke{SMOKE_TAG}.jsonl" if SMOKE else "desk.json")
STREAM = OUT_DIR / (f"smoke{SMOKE_TAG}_cells.jsonl" if SMOKE else "desk.jsonl")
FAMILIES = ("g", "mhat", "a", "u", "b")


# ---------------------------------------------------------------- virtual AdamW law (float64, per tensor)
def clip_coef(row_norm):
    return min(1.0, CLIP / (row_norm + CLIP_EPS))


def virtual_families(gblk, W, m, v, step_next, lr, beta1, beta2, wd, eps=1e-8):
    """gblk: (n, d_t) float64 RAW gradients of one tensor; W, m, v: (d_t,) float64 stored state; rows already clipped by the caller.
    Returns dict of (n, d_t) arrays: mhat, a, u, b (and the batch-independent a0 as a (d_t,) vector)."""
    bc1, bc2 = 1.0 - beta1 ** step_next, 1.0 - beta2 ** step_next
    m1 = beta1 * m[None, :] + (1.0 - beta1) * gblk
    v1 = beta2 * v[None, :] + (1.0 - beta2) * gblk * gblk
    mhat = m1 / bc1
    a = -lr * mhat / (np.sqrt(v1 / bc2) + eps)
    m0 = beta1 * m; v0 = beta2 * v
    a0 = -lr * (m0 / bc1) / (np.sqrt(v0 / bc2) + eps)
    decay = -lr * wd * W
    u = decay[None, :] + a
    b = a - a0[None, :]
    return {"mhat": mhat, "a": a, "u": u, "b": b, "a0": a0, "u0": decay + a0, "decay": decay, "carried": beta1 * m}


def rel_err(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1); b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(a - b) / (np.abs(b) + 1e-8)))


# ---------------------------------------------------------------- optimizer binding
def build_opt(model):
    return torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)    # the trainer's constructor call


def bind_state(model, ckpt, dev):
    """Load model + optimizer state from a milestone; assert structure. Returns (opt, step, group_record)."""
    model.load_state_dict(ckpt["model"]); model.to(dev); model.eval()
    opt = build_opt(model)
    opt.load_state_dict(ckpt["opt"])
    names = [n for n, _ in model.named_parameters()]
    assert sorted(names) == UG.KEYS, "parameter set v the sorted-key law"     # optimizer state indices follow this module order
    assert len(opt.param_groups) == 1
    g = opt.param_groups[0]
    steps = set()
    for i, p in enumerate(g["params"]):
        st = opt.state[p]
        assert st["exp_avg"].shape == p.shape and st["exp_avg_sq"].shape == p.shape, names[i]
        steps.add(int(st["step"].item()) if torch.is_tensor(st["step"]) else int(st["step"]))
    assert len(steps) == 1, steps
    rec = {"lr": float(g["lr"]), "beta1": float(g["betas"][0]), "beta2": float(g["betas"][1]), "weight_decay": float(g["weight_decay"]), "eps": float(g["eps"]),
           "amsgrad": bool(g["amsgrad"]), "maximize": bool(g["maximize"]), "decoupled_weight_decay": bool(g.get("decoupled_weight_decay", True)), "n_params": len(g["params"]), "adam_step": steps.pop(),
           "param_order_digest": hashlib.sha256("|".join(names).encode()).hexdigest()[:16]}
    assert rec["eps"] == 1e-8 and rec["weight_decay"] == WD and not rec["amsgrad"] and not rec["maximize"] and rec["decoupled_weight_decay"] and rec["n_params"] == len(UG.KEYS)
    return opt, rec


def registered_probe_digest(tok):
    """The UGC0 probe law at its registered size (64 slices, FIT 32 / HELD 32), independent of the SMOKE constants."""
    import hashlib as _h
    import random as _r
    import birth19m_curric as C
    rows = C.load_excised_rows(); enc, _ = C.encode_with_levels(rows, tok)
    starts = [(i, i + TM.BS) for i in range(0, len(enc) - TM.BS, TM.BS)]
    chosen = _r.Random(UG.PROBE_SEED).sample(starts, 64)
    return _h.sha256(json.dumps({"slices": chosen, "ids": [[enc[j] for j in range(a, b)] for a, b in chosen]}).encode()).hexdigest()


def sched_rows(family_kind, s):
    """(used_for_step_s, for_step_s_plus_1) from the audited tables: rows are [step, lr, beta1, beta2, wd, last_epoch]."""
    t = json.loads(SCHED.read_text())["tables"][family_kind]
    used, nxt = t[s - 1], t[s]
    assert used[0] == s and nxt[0] == s + 1
    return {"lr": used[1], "beta1": used[2], "beta2": used[3], "wd": used[4]}, {"lr": nxt[1], "beta1": nxt[2], "beta2": nxt[3], "wd": nxt[4]}


def opt_state_digest(opt):
    h = hashlib.sha256()
    for p in opt.param_groups[0]["params"]:
        st = opt.state[p]
        for k in ("exp_avg", "exp_avg_sq"):
            h.update(st[k].detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
        h.update(str(int(st["step"].item()) if torch.is_tensor(st["step"]) else int(st["step"])).encode())
    return h.hexdigest()


# ---------------------------------------------------------------- Gram pass with derived families
def cosine_gram(K):
    d = np.sqrt(np.clip(np.diag(K), 1e-300, None))
    return K / np.outer(d, d)


def distortion(Kx, Ky):
    Cx, Cy = cosine_gram(Kx), cosine_gram(Ky)
    return float(np.linalg.norm(Cy - Cx) / max(np.linalg.norm(Cx), 1e-300))


def family_pass(specs, segs, n, d, deltas=None):
    """specs: {"A": {"mm": path, "sd": model_sd(cpu), "opt_sd": opt state dict (cpu), "next": {lr,beta1,beta2,wd}, "step": s, "norms": (n,) raw row norms}, "B": {...}}.
    One pass over the tensor blocks: per specimen per family the Gram (n x n), per group Grams, cross-family diagonals
    (dot(g_i, x_i)), norms, the A x B cross-Gram per family, and velocity dots c = X delta (own writer) per family."""
    names = list(specs)
    G = {nm: np.memmap(specs[nm]["mm"], dtype=np.float32, mode="r", shape=(n, d)) for nm in names}
    Z = lambda: np.zeros((n, n))  # noqa: E731
    K = {nm: {f: Z() for f in FAMILIES} for nm in names}
    Kg = {nm: {f: {g: Z() for g in UG.GROUPS} for f in FAMILIES} for nm in names}
    X = {f: Z() for f in FAMILIES} if len(names) == 2 else None
    diag = {nm: {f: np.zeros(n) for f in ("mhat", "a", "u", "b")} for nm in names}          # dot(g_i, x_i)
    norm2 = {nm: {f: np.zeros(n) for f in ("g", "c", "mhat", "a", "u", "b")} for nm in names}
    scal = {nm: {"carried": 0.0, "decay": 0.0, "a0": 0.0, "u0": 0.0, "W": 0.0} for nm in names}
    vel = {nm: {f: np.zeros(n) for f in ("g", "u", "b")} for nm in names} if deltas else None
    vel2 = {nm: 0.0 for nm in names}
    for k, a, bnd, grp in segs:
        blk = {}
        for nm in names:
            sp = specs[nm]
            raw = np.asarray(G[nm][:, a:bnd], dtype=np.float64)
            coef = np.array([clip_coef(x) for x in sp["norms"]])[:, None]
            c = raw * coef
            idx = sp["pidx"][k]; st = sp["opt_sd"]["state"][idx]
            W = sp["sd"][k].double().numpy().reshape(-1); m = st["exp_avg"].double().numpy().reshape(-1); v = st["exp_avg_sq"].double().numpy().reshape(-1)
            nx = sp["next"]
            fam = virtual_families(c, W, m, v, sp["step"] + 1, nx["lr"], nx["beta1"], nx["beta2"], nx["wd"])
            fam["g"] = raw
            blk[nm] = fam
            for f in FAMILIES:
                kk = fam[f] @ fam[f].T
                K[nm][f] += kk; Kg[nm][f][grp] += kk
            for f in ("mhat", "a", "u", "b"):
                diag[nm][f] += np.einsum("ij,ij->i", raw, fam[f])
            for f in ("g", "mhat", "a", "u", "b"):
                norm2[nm][f] += np.einsum("ij,ij->i", fam[f], fam[f])
            norm2[nm]["c"] += np.einsum("ij,ij->i", c, c)
            scal[nm]["carried"] += float(fam["carried"] @ fam["carried"]); scal[nm]["decay"] += float(fam["decay"] @ fam["decay"])
            scal[nm]["a0"] += float(fam["a0"] @ fam["a0"]); scal[nm]["u0"] += float(fam["u0"] @ fam["u0"]); scal[nm]["W"] += float(W @ W)
            if deltas:
                dv = deltas[nm][k]
                for f in ("g", "u", "b"):
                    vel[nm][f] += fam[f] @ dv
                vel2[nm] += float(dv @ dv)
        if X is not None:
            for f in FAMILIES:
                X[f] += blk[names[0]][f] @ blk[names[1]][f].T
    del G
    return {"K": K, "K_group": Kg, "X": X, "diag": diag, "norm2": norm2, "scal": scal, "vel": vel, "vel2": vel2}


def family_geometry(K, Kg, fit, held):
    out = {}
    for f in FAMILIES:
        geo = UG.geometry_from_grams(K[f], fit, held, K_LIST)
        out[f] = {"Q": geo["Q"], "raw": {"held_capture": geo["raw"]["held_capture"], "reliability_fit_held": geo["raw"]["reliability_fit_held"],
                                          "participation_ratio": geo["raw"]["fit_spectrum"]["participation_ratio"], "topk_energy": geo["raw"]["fit_spectrum"]["topk_energy"], "mean_cos_fit": geo["raw"]["mean_cos_fit"]},
                  "centered": {"held_capture": geo["centered"]["held_capture"], "reliability_fit_held": geo["centered"]["reliability_fit_held"], "participation_ratio": geo["centered"]["fit_spectrum"]["participation_ratio"]},
                  "energy_share_by_group": {g: float(np.diag(Kg[f][g]).mean() / max(np.diag(K[f]).mean(), 1e-300)) for g in UG.GROUPS},
                  "group_raw_C_ladder": {g: float(UG.geometry_from_grams(Kg[f][g], fit, held, [K_LADDER])["raw"]["held_capture"][K_LADDER]["median"]) for g in UG.GROUPS}}
    out["distortion"] = {"g->mhat": distortion(K["g"], K["mhat"]), "mhat->a": distortion(K["mhat"], K["a"]), "a->u": distortion(K["a"], K["u"]), "g->u": distortion(K["g"], K["u"]), "g->b": distortion(K["g"], K["b"]), "a->b": distortion(K["a"], K["b"])}
    return out


def adjudicate(cells, pairs_late):
    """cells: {(spec, s): family_geometry output}; pairs_late: pair overlaps at the late state {family: {"raw": {k: S}}}.
    Axes: thinness of u and b relative to g (SHARPEN 0.15 at >= 2 of 3 states in BOTH writers), writer overlap at the late state."""
    def c8(spec, s, f):
        return cells[(spec, s)][f]["raw"]["held_capture"][K_LADDER]["median"]
    def rel(spec, s, f):
        return cells[(spec, s)][f]["raw"]["reliability_fit_held"][K_LADDER]
    axes = {"per_state": {}}
    resolved = True
    for s in STATES:
        row = {}
        for spec in ("A", "B"):
            row[spec] = {"C8_g": c8(spec, s, "g"), "C8_u": c8(spec, s, "u"), "C8_b": c8(spec, s, "b"), "rel_g": rel(spec, s, "g"), "rel_u": rel(spec, s, "u"), "rel_b": rel(spec, s, "b")}
            row[spec]["d_u"] = row[spec]["C8_u"] - row[spec]["C8_g"]; row[spec]["d_b"] = row[spec]["C8_b"] - row[spec]["C8_g"]
        axes["per_state"][s] = row
        if min(row["A"]["rel_g"], row["B"]["rel_g"], row["A"]["rel_u"], row["B"]["rel_u"]) < REL_MIN:
            resolved = False
    axes["resolution"] = "RESOLVED" if resolved else "GEOMETRY-NOT-RESOLVED"
    if not resolved:
        return axes
    def count(cond):
        return sum(all(cond(axes["per_state"][s][spec]) for spec in ("A", "B")) for s in STATES)
    n_u = count(lambda r: r["d_u"] >= SHARPEN); n_b = count(lambda r: r["d_b"] >= SHARPEN and r["rel_b"] >= REL_MIN)
    n_b_small = count(lambda r: r["d_b"] < SMALL); n_diff = count(lambda r: r["d_b"] <= -SHARPEN); n_b_unres = count(lambda r: r["rel_b"] < REL_MIN)
    axes["counts"] = {"states_full_sharpened": n_u, "states_batch_sharpened_resolved": n_b, "states_batch_rise_below_small": n_b_small, "states_batch_diffused": n_diff, "states_b_unresolved": n_b_unres}
    axes["b_panel"] = "B-UNRESOLVED" if n_b_unres >= 2 else "B-RESOLVED"     # reported beside the label, never merged into it
    if n_b >= 2:
        axes["sharpening"] = "BATCH-WRITE-SHARPENED"
    elif n_u >= 2 and n_b_small >= 2:
        axes["sharpening"] = "HISTORY-DOMINATED"
    elif n_u >= 2:
        axes["sharpening"] = "WRITE-SHARPENED"
    elif n_diff >= 2:
        axes["sharpening"] = "OPTIMIZER-DIFFUSES"
    else:
        axes["sharpening"] = "NO-OPTIMIZER-SHARPENING"
    late = STATES[-1]
    axes["writer"] = {}
    for f in ("u", "b"):
        s_ab = pairs_late[f]["raw"][K_LADDER]; self_min = min(rel("A", late, f), rel("B", late, f))
        axes["writer"][f] = {"S8_AB": s_ab, "self_min": self_min, "label": ("WRITER-UPDATE-SHARED" if s_ab >= SHARED_FRAC * self_min else ("WRITER-UPDATE-ROTATED" if s_ab < ROTATED_FRAC * self_min else "WRITER-UPDATE-INDETERMINATE"))}
    return axes


# ---------------------------------------------------------------- parity endpoints (smoke only)
def synthetic_parity(seed=0):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(s)) for s in ((5, 7), (11,), (3, 4, 2))]
    opt = torch.optim.AdamW(ps, lr=3e-4, weight_decay=0.01)
    grads = [torch.randn_like(p) for p in ps]
    for step in range(1, 5):
        for p, g in zip(ps, grads):
            p.grad = (g * (0.5 + step)).clone()
        opt.step()
    # virtual from the current state
    st_p = [p.detach().double().numpy().reshape(-1).copy() for p in ps]
    st_m = [opt.state[p]["exp_avg"].double().numpy().reshape(-1).copy() for p in ps]
    st_v = [opt.state[p]["exp_avg_sq"].double().numpy().reshape(-1).copy() for p in ps]
    gs = [torch.randn_like(p) for p in ps]
    tot = float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs)))
    coef = clip_coef(tot)
    errs = []
    for p, g in zip(ps, gs):
        p.grad = g.clone()
    torch.nn.utils.clip_grad_norm_(ps, CLIP)
    opt.param_groups[0]["lr"] = 2.5e-4; opt.param_groups[0]["betas"] = (0.88, 0.999)
    opt.step()
    for i, p in enumerate(ps):
        fam = virtual_families(gs[i].double().numpy().reshape(-1)[None, :] * coef, st_p[i], st_m[i], st_v[i], 5, 2.5e-4, 0.88, 0.999, 0.01)
        errs.append(rel_err(st_p[i] + fam["u"][0], p.detach().double().numpy().reshape(-1)))
        errs.append(rel_err(0.88 * st_m[i] + 0.12 * gs[i].double().numpy().reshape(-1) * coef, opt.state[p]["exp_avg"].double().numpy().reshape(-1)))
    return {"max_rel_err": float(max(errs)), "ok": bool(max(errs) <= 1e-6)}   # float32 torch v float64 law on synthetic tensors: rounding-level agreement


def scale_err(diff, ref):
    """max |diff| relative to the scale of the reference tensor (max |ref|), the registered parity metric."""
    diff = np.asarray(diff, dtype=np.float64).reshape(-1); ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(diff)) / max(float(np.max(np.abs(ref))), 1e-300))


def real_parity(tok, dev, batches, dtype=torch.float64, device="cpu", tol=None):
    """One real trainer step on the non-target seed-6 model (backward, clip, AdamW.step, in `dtype` on `device`) from a saved
    pre-step state, v the virtual law applied to the pre-step state and the same raw gradient. Compared: the realized update
    W_post - W_pre v u (relative to max |u|), exp_avg and exp_avg_sq (relative to their max), step counters; at Adam step 1
    and after three warm-up steps (step 4). float64 / cpu is the registered exact-law endpoint; float32 / device is the
    rounding envelope of the actual writer (descriptive)."""
    tol = tol or PARITY_TOL_PARAM
    out = {}
    model = UG.build(tok, device).to(dtype); sd0 = UG.w0_seed(6); model.load_state_dict(sd0); model.train()
    opt = build_opt(model)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=100, pct_start=PCT)
    pidx = {nme: i for i, (nme, _) in enumerate(model.named_parameters())}
    for trial, warm in (("step1", 0), ("step4", 3)):
        for w in range(warm):
            _real_step(model, tok, batches[w % len(batches)], device, opt, sch)
        pre_sd = {k: v.detach().cpu().double().clone() for k, v in model.state_dict().items()}
        pre_m = {k: (opt.state[opt.param_groups[0]["params"][pidx[k]]]["exp_avg"].detach().cpu().double().clone() if opt.param_groups[0]["params"][pidx[k]] in opt.state else None) for k in UG.KEYS}
        pre_v = {k: (opt.state[opt.param_groups[0]["params"][pidx[k]]]["exp_avg_sq"].detach().cpu().double().clone() if opt.param_groups[0]["params"][pidx[k]] in opt.state else None) for k in UG.KEYS}
        g = opt.param_groups[0]; lr, (b1, b2), wd = float(g["lr"]), g["betas"], float(g["weight_decay"])
        pre_steps = {int(opt.state[q]["step"].item()) for q in g["params"] if q in opt.state}
        step_pre = pre_steps.pop() if pre_steps else 0
        assert not pre_steps, "inconsistent Adam step counters"
        grads = _real_step(model, tok, batches[warm % len(batches)], device, opt, sch, return_raw_grads=True)
        post_sd = {k: v.detach().cpu().double() for k, v in model.state_dict().items()}
        tot = float(np.sqrt(sum(float((gr.double() ** 2).sum()) for gr in grads.values())))
        coef = clip_coef(tot)
        errs_p, errs_m, errs_v = [], [], []
        for k in UG.KEYS:
            W = pre_sd[k].numpy().reshape(-1)
            m = np.zeros_like(W) if pre_m[k] is None else pre_m[k].numpy().reshape(-1)
            v = np.zeros_like(W) if pre_v[k] is None else pre_v[k].numpy().reshape(-1)
            gg = grads[k].double().numpy().reshape(-1)[None, :] * coef
            fam = virtual_families(gg, W, m, v, step_pre + 1, lr, float(b1), float(b2), wd)
            p = opt.param_groups[0]["params"][pidx[k]]
            errs_p.append(scale_err((post_sd[k].numpy().reshape(-1) - W) - fam["u"][0], fam["u"][0]))
            m_new = float(b1) * m + (1 - float(b1)) * gg[0]; v_new = float(b2) * v + (1 - float(b2)) * gg[0] * gg[0]
            errs_m.append(scale_err(opt.state[p]["exp_avg"].detach().cpu().double().numpy().reshape(-1) - m_new, m_new))
            errs_v.append(scale_err(opt.state[p]["exp_avg_sq"].detach().cpu().double().numpy().reshape(-1) - v_new, v_new))
        steps = {int(opt.state[p]["step"].item()) for p in opt.param_groups[0]["params"]}
        out[trial] = {"dtype": str(dtype).replace("torch.", ""), "device": device, "adam_step_before": step_pre, "lr": lr, "beta1": float(b1), "clip_coef": coef, "grad_norm": tot,
                      "max_err_update": float(max(errs_p)), "max_err_exp_avg": float(max(errs_m)), "max_err_exp_avg_sq": float(max(errs_v)), "adam_steps_after": sorted(steps),
                      "ok": bool(max(errs_p) <= tol and max(errs_m) <= tol and max(errs_v) <= tol and steps == {step_pre + 1})}
    return out


def _real_step(model, tok, batch, dev, opt, sch, return_raw_grads=False):
    L = max(len(s) for s in batch)
    ids = torch.tensor([s + [tok.pad_id] * (L - len(s)) for s in batch], device=dev)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in batch], device=dev)
    logits = model(ids[:, :-1], mask[:, :-1]); labels = ids[:, 1:].clone(); labels[mask[:, 1:] == 0] = -100
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    loss.backward()
    raw = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()} if return_raw_grads else None
    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
    opt.step(); sch.step(); opt.zero_grad()
    return raw


# ---------------------------------------------------------------- main
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists() or STREAM.exists():
        raise SystemExit(f"REFUSING: {OUT} or {STREAM} exists")
    if TMP.exists():
        raise SystemExit(f"REFUSING: {TMP} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered desk on a dirty tree")
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = TM.MathTokenizer()
    assert len(tok.vocab) == 40 and not os.environ.get("VOCAB_EXTRA") and not os.environ.get("SEQ_CAP") and not os.environ.get("BIRTH_BS") and TM.BS == 32
    model = UG.build(tok, dev)
    segs, d, flat_digest = UG.flatten_law(model)
    assert d == 18_911_616
    batches, probe = UG.probe_batches(tok)
    ugc0 = json.loads(UGC0.read_text())
    if not SMOKE:
        assert probe["digest"] == ugc0["probe"]["digest"], "probe digest v the booked UGC0 receipt"
    else:
        # exercise the registered-law assertion before any target run: the 64-batch draw under the registered constants must equal UGC0's digest
        full = registered_probe_digest(tok)
        assert full == ugc0["probe"]["digest"], f"registered probe law drifted: {full[:16]} v UGC0 {ugc0['probe']['digest'][:16]}"
    sched = json.loads(SCHED.read_text())
    rec = {"prereg": "OPTIMIZER-GEOMETRY-DESK-0", "kind": "optimizer_geometry_desk", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty, "source_sha256": UG.sha256_file(__file__), "ugc0_source_sha256": UG.sha256_file(UG.__file__),
           "device": dev, "torch_version": torch.__version__, "numpy_version": np.__version__, "smoke_receipts_sha256": {str(q): UG.sha256_file(str(q)) for q in sorted(OUT_DIR.glob("smoke*.jsonl")) if q != OUT}, "d": d, "flatten_law_digest": flat_digest, "probe": {k: probe[k] for k in ("digest", "n_enc", "n_rows", "fit", "held", "level_mix")}, "ugc0_probe_digest": ugc0["probe"]["digest"],
           "sched_receipt_sha256": UG.sha256_file(SCHED), "sched_source_commit": sched["commit"], "states": STATES, "next": NEXT, "paths": {"A": A_PATHS, "B": B_PATHS, "A_next": A_NEXT, "B_next": B_NEXT},
           "law": {"LR": LR, "WD": WD, "clip": CLIP, "clip_eps": CLIP_EPS, "k_list": K_LIST, "K_LADDER": K_LADDER, "SHARPEN": SHARPEN, "SMALL": SMALL, "REL_MIN": REL_MIN, "SHARED_FRAC": SHARED_FRAC, "ROTATED_FRAC": ROTATED_FRAC,
                   "parity_tol_exact": PARITY_TOL_PARAM, "envelope_tol_float32": ENVELOPE_TOL, "families": FAMILIES, "virtual_law": "c = g * min(1, 1/(||g||+1e-6)); m' = b1 m + (1-b1) c; v' = b2 v + (1-b2) c^2; a = -lr (m'/bc1) / (sqrt(v'/bc2) + eps); u = -lr wd W + a; u0 with c = 0; b = u - u0"},
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    free_gb = shutil.disk_usage(OUT_DIR).free / 1e9
    need_gb = (2 if not SMOKE else 1) * len(batches) * d * 4 / 1e9 + 2.0
    assert free_gb >= need_gb, f"disk preflight: {free_gb:.1f} GB free < {need_gb:.1f} GB needed"
    rec["disk_preflight"] = {"free_gb": round(free_gb, 1), "need_gb": round(need_gb, 1)}
    TMP.mkdir(parents=True)
    fit, held = probe["fit"], probe["held"]; n = len(batches)
    t0 = time.time()

    def stream(row):
        with STREAM.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # parity endpoints (smoke) -------------------------------------------------
    if SMOKE:
        rec["parity_synthetic"] = synthetic_parity()
        rec["parity_real_exact"] = real_parity(tok, dev, batches, dtype=torch.float64, device="cpu", tol=PARITY_TOL_PARAM)
        rec["parity_real_envelope_float32"] = real_parity(tok, dev, batches, dtype=torch.float32, device=dev, tol=ENVELOPE_TOL)
        print(f"[ogd0] parity synthetic {rec['parity_synthetic']} exact {json.dumps(rec['parity_real_exact'])} envelope {json.dumps(rec['parity_real_envelope_float32'])}", flush=True)

    def gradient_cell(name, s, path_or_sd, opt_sd):
        """Raw-gradient memmap for one specimen-state; asserts state digests; returns (mm, norms, model_sd cpu, opt_sd cpu, digests)."""
        if isinstance(path_or_sd, str):
            ck = torch.load(path_or_sd, map_location="cpu"); msd, osd, step = ck["model"], ck["opt"], int(ck["step"])
            assert step == s
        else:
            msd, osd, step = path_or_sd, opt_sd, s
        dg = state_digest(msd)
        model.load_state_dict(msd); model.to(dev); model.eval()
        opt, grp = bind_state(model, {"model": msd, "opt": osd}, dev)
        assert grp["adam_step"] == s, (grp["adam_step"], s)
        odg = opt_state_digest(opt)
        mm = str(TMP / f"{name}_{s}.f32")
        g = UG.cell_gradients(model, tok, batches, dev, segs, d, mm)
        dg2 = state_digest({k: v.detach().cpu() for k, v in model.state_dict().items()})
        assert dg2 == dg, "STATE MUTATED"
        assert opt_state_digest(opt) == odg, "OPTIMIZER STATE MUTATED"
        G_ = np.memmap(mm, dtype=np.float32, mode="r", shape=(n, d))
        norms = np.sqrt(np.array([float(np.dot(r64 := np.asarray(G_[i], dtype=np.float64), r64)) for i in range(n)]))   # one read pass; float32 rows as the trainer's clip saw them (to float32 rounding)
        del G_
        pidx = {nme: i for i, (nme, _) in enumerate(model.named_parameters())}
        return mm, norms, msd, osd, {"pidx": pidx, "state_digest": dg, "opt_state_digest": odg, "group": grp, "file_sha256": UG.sha256_file(path_or_sd) if isinstance(path_or_sd, str) else None, "losses": g["losses"], "memmap": {"shape": g["shape"], "dtype": g["dtype"], "bytes": g["bytes"]}}

    cells, pairs, velocity = {}, {}, {}
    if SMOKE:
        # synthetic optimizer state at Adam step 4 on the non-target seed-6 W_0: three real warm-up steps, then the pipeline
        m_ = UG.build(tok, dev); sd0 = UG.w0_seed(6); m_.load_state_dict(sd0); m_.train(); o_ = build_opt(m_)
        sch_ = torch.optim.lr_scheduler.OneCycleLR(o_, max_lr=LR, total_steps=100, pct_start=PCT)
        for w in range(4):
            _real_step(m_, tok, batches[w % len(batches)], dev, o_, sch_)
        msd = {k: v.detach().cpu().clone() for k, v in m_.state_dict().items()}; osd = copy.deepcopy(o_.state_dict())
        for st in osd["state"].values():
            for k2 in list(st):
                if torch.is_tensor(st[k2]):
                    st[k2] = st[k2].detach().cpu()
        nxt = {"lr": float(o_.param_groups[0]["lr"]), "beta1": float(o_.param_groups[0]["betas"][0]), "beta2": float(o_.param_groups[0]["betas"][1]), "wd": WD}
        mm, norms, msd, osd, info = gradient_cell("S6W0", 4, msd, osd)
        specs = {"S": {"mm": mm, "sd": msd, "opt_sd": osd, "next": nxt, "step": 4, "norms": norms, "pidx": info.pop("pidx")}}
        red = family_pass(specs, segs, n, d)
        cells[("S", 4)] = family_geometry(red["K"]["S"], red["K_group"]["S"], fit, held)
        cells[("S", 4)]["decomposition"] = {f: float(np.sqrt(np.median(red["norm2"]["S"][f]))) for f in red["norm2"]["S"]}
        cells[("S", 4)]["cos_g_family_median"] = {f: float(np.median(red["diag"]["S"][f] / np.sqrt(red["norm2"]["S"]["g"] * red["norm2"]["S"][f]))) for f in ("mhat", "a", "u", "b")}
        cells[("S", 4)]["info"] = info
        os.remove(mm)
        stream({"kind": "cell", "specimen": "S6W0", "step": 4, "geometry": cells[("S", 4)]})
        print(f"[ogd0] smoke cell: C8 g {cells[('S',4)]['g']['raw']['held_capture'][K_LADDER]['median']:.3f} u {cells[('S',4)]['u']['raw']['held_capture'][K_LADDER]['median']:.3f} b {cells[('S',4)]['b']['raw']['held_capture'][K_LADDER]['median']:.3f} distortion {cells[('S',4)]['distortion']} ({time.time()-t0:.0f}s)", flush=True)
    else:
        for s in STATES:
            t1 = time.time()
            specs = {}
            for name, paths, nexts, kind in (("A", A_PATHS, A_NEXT, "stock"), ("B", B_PATHS, B_NEXT, "backward")):
                used, nxt = sched_rows(kind, s)
                g0 = torch.load(paths[s], map_location="cpu")["opt"]["param_groups"][0]     # the parity gate reads the serialized group BEFORE any target gradient
                saved = {"lr": float(g0["lr"]), "beta1": float(g0["betas"][0]), "beta2": float(g0["betas"][1]), "wd": float(g0["weight_decay"])}
                assert saved == used, f"SCHEDULER PARITY FAILED at {name}@{s}: saved {saved} v used {used}"
                del g0
                mm, norms, msd, osd, info = gradient_cell(name, s, paths[s], None)
                grp = info["group"]
                assert grp["lr"] == used["lr"] and grp["beta1"] == used["beta1"] and grp["beta2"] == used["beta2"] and grp["weight_decay"] == used["wd"]
                info["sched_used"] = used; info["sched_next"] = nxt; info["path"] = paths[s]
                nsd = UG.load_sd(nexts[s])
                delta = {k: (nsd[k].double() - msd[k].double()).reshape(-1).numpy() for k in UG.KEYS}
                specs[name] = {"mm": mm, "sd": msd, "opt_sd": osd, "next": nxt, "step": s, "norms": norms, "pidx": info.pop("pidx"), "info": info, "delta": delta, "next_path": nexts[s], "next_digest": state_digest(nsd)}
            red = family_pass(specs, segs, n, d, deltas={nm: specs[nm]["delta"] for nm in specs})
            for nm in ("A", "B"):
                geo = family_geometry(red["K"][nm], red["K_group"][nm], fit, held)
                geo["decomposition_median_norm"] = {f: float(np.sqrt(np.median(red["norm2"][nm][f]))) for f in red["norm2"][nm]}
                geo["decomposition_scalars_norm"] = {k2: float(np.sqrt(v)) for k2, v in red["scal"][nm].items()}
                geo["cos_g_family"] = {f: {"median": float(np.median(red["diag"][nm][f] / np.sqrt(red["norm2"][nm]["g"] * red["norm2"][nm][f]))), "min": float(np.min(red["diag"][nm][f] / np.sqrt(red["norm2"][nm]["g"] * red["norm2"][nm][f]))), "max": float(np.max(red["diag"][nm][f] / np.sqrt(red["norm2"][nm]["g"] * red["norm2"][nm][f])))} for f in ("mhat", "a", "u", "b")}
                geo["clip_coef"] = {"median": float(np.median([clip_coef(x) for x in specs[nm]["norms"]])), "min": float(np.min([clip_coef(x) for x in specs[nm]["norms"]])), "frac_clipped": float(np.mean([clip_coef(x) < 1 for x in specs[nm]["norms"]]))}
                ref = next((c for c in ugc0["cells"] if c["specimen"] == nm and c["step"] == s), None)     # the booked UGC0 g-family cell on the same batches (S2: recorded reproduction, not a hard gate: mps run-level nondeterminism)
                if ref is not None:
                    geo["ugc0_reference_g"] = {"C8_median": ref["geometry"]["raw"]["held_capture"][str(K_LADDER)]["median"], "rel8": ref["geometry"]["raw"]["reliability_fit_held"][str(K_LADDER)],
                                               "delta_C8": geo["g"]["raw"]["held_capture"][K_LADDER]["median"] - ref["geometry"]["raw"]["held_capture"][str(K_LADDER)]["median"],
                                               "delta_rel8": geo["g"]["raw"]["reliability_fit_held"][K_LADDER] - ref["geometry"]["raw"]["reliability_fit_held"][str(K_LADDER)], "state_digest_equal": ref["state_digest"] == specs[nm]["info"]["state_digest"]}
                geo["info"] = specs[nm]["info"]; geo["next_state"] = {"path": specs[nm]["next_path"], "digest": specs[nm]["next_digest"]}
                v2 = red["vel2"][nm]
                geo["velocity_own"] = {f: UG.projection_fraction(UG.unit_gram(red["K"][nm][f], fit), red["vel"][nm][f][fit] / np.sqrt(np.diag(red["K"][nm][f])[fit]), v2, K_LIST) for f in ("g", "u", "b")}
                geo["velocity_norm"] = float(np.sqrt(v2))
                cells[(nm, s)] = geo
                stream({"kind": "cell", "specimen": nm, "step": s, "geometry": geo})
            pairs[s] = {f: UG.pair_overlap_from_grams(red["K"]["A"][f], red["K"]["B"][f], red["X"][f], fit, fit, K_LIST) for f in FAMILIES}
            stream({"kind": "pair", "step": s, "overlap": pairs[s]})
            for nm in specs:
                os.remove(specs[nm]["mm"])
            print(f"[ogd0] state {s}: " + " | ".join(f"{nm} C8 g {cells[(nm,s)]['g']['raw']['held_capture'][K_LADDER]['median']:.3f} u {cells[(nm,s)]['u']['raw']['held_capture'][K_LADDER]['median']:.3f} b {cells[(nm,s)]['b']['raw']['held_capture'][K_LADDER]['median']:.3f} rel_b {cells[(nm,s)]['b']['raw']['reliability_fit_held'][K_LADDER]:.3f} D(g->u) {cells[(nm,s)]['distortion']['g->u']:.3f}" for nm in ("A", "B")) + f" | S8 u {pairs[s]['u']['raw'][K_LADDER]:.3f} b {pairs[s]['b']['raw'][K_LADDER]:.3f} ({time.time()-t1:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
    if not SMOKE:
        axes = adjudicate(cells, pairs[STATES[-1]])
    else:
        axes = {"smoke": True}
    shutil.rmtree(TMP, ignore_errors=True)
    rec.update({"cells": {f"{k[0]}@{k[1]}": v for k, v in cells.items()}, "pairs": pairs, "axes": axes, "wall_s": round(time.time() - t0, 1), "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    if SMOKE:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[ogd0] smoke row appended: parity ok synthetic {rec['parity_synthetic']['ok']} exact {[v['ok'] for v in rec['parity_real_exact'].values()]} envelope {[v['ok'] for v in rec['parity_real_envelope_float32'].values()]}")
    else:
        OUT.write_text(json.dumps(rec, indent=1))
        print(f"[ogd0] written {OUT}: axes {json.dumps(axes)} wall {rec['wall_s']}s")


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        shutil.rmtree(TMP, ignore_errors=True)      # a failed run leaves no stranded memmaps (the streamed rows stay for the record)
        raise

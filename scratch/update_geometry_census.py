"""UPDATE-GEOMETRY-CENSUS-0 (PRE-REG in RESULTS): zero-training LOSS-GRADIENT
GEOMETRY census on the WRITER-TRAJECTORY-CENSUS-0 pair. No training, no
optimizer step, no parameter mutation, no generated data, no gate.

Specimens (existing checkpoints only): A = seed-2 forward OneCycle
(checkpoints/phase19m milestones, final checkpoints/gallery19m_phase_s2.pt),
B = seed-2 backward OneCycle (checkpoints/backsched19m, final
checkpoints/gallery19m_backsched_s2.pt); the shared W_0 is the canonical
seed-2 construction (torch.manual_seed(2); build_model), digest asserted
against logs/writertraj0/census.json. Time grid (frozen): steps 0 (W_0),
900, 3600, 7200, 10800, 13500, 15420 (final) for both writers; W_0 once.
Supplemental FINAL-STATE null: the provenance-clean same-writer pair N3 / N4
(stock_s7 first run v repair, WRITER-DEPENDENCE-NULL-2) at step 15420 only.

Probe population (frozen before any specimen gradient is read): the
D2-excised stock rows (birth19m_curric.load_excised_rows), encoded and
length-sorted by the trainer's encode_with_levels, cut into the trainer's
BS = 32 slices (the same slice starts as stock_epoch_stream); 64 slices
drawn by random.Random("ugc0-probe-v1").sample; FIT = the first 32 drawn,
HELD = the last 32. Digest of the batch index tuples and the encoded ids
is receipted. Identical batches for every specimen, state and tensor group.

Gradient law: model.eval() (the family has no dropout), zero grads, the
trainer's CE (mean over eligible label positions, ignore_index -100, the
trainer's padding / masking, logits = model(ids[:, :-1], mask[:, :-1])),
loss.backward(), dL/dtheta read from p.grad for the 59 trainable tensors
in the sorted-key order of WRITER-TRAJECTORY-CENSUS-0 (atomtraj_pins.
CLASSES), flattened into one row of d = 18,911,616 float32 entries of a
file-backed numpy.memmap (rows = probe batches, columns = the registered
flatten law). No clipping, no weight decay, no optimizer moments. The
state digest is asserted before and after every cell.

Reductions (float64, per tensor block, never a d x d matrix): the 64 x 64
Gram K over all probe rows, the same per group (the house 9-group
partition BLOCK0..7 + OUTSIDE of writertraj_depend), row norms, the
cross-Gram between two specimens at the same state, and the 64 inner
products with a velocity vector (the exact difference of adjacent grid
states). Parity: row 0 is kept in float64 RAM and its direct inner
product with every other row is compared to the memmap float32 Gram.

Geometry (pure Gram algebra, tests/test_update_geometry_census.py):
Q = ||mean g||^2 / mean ||g||^2 (coherent-energy fraction); RAW =
unit-normalized rows; CENTERED = FIT-panel mean subtracted from every row
(FIT and HELD) before unit normalization; FIT Gram spectrum, participation
ratio and top-k energy; HELD capture C_k(h) = ||P_k h||^2 / ||h||^2 with
P_k the top-k right singular subspace of the FIT rows; subspace overlap
S_k = tr(P_X P_Y) / k for (A_FIT, B_FIT), (X_FIT, X_HELD) reliability and
(N3_FIT, N4_FIT); velocity projection ||P_k v||^2 / ||v||^2 for v = the
own-writer and other-writer next-grid-state difference. k in {1, 2, 4, 8,
16}. Isotropic reference: a synthetic Gaussian 64 x d memmap through the
identical reductions, plus the analytic floor k / d.

Ladder (thresholds literal, frozen before data; adjudicate is pure):
GEOMETRY-NOT-RESOLVED if reliability S_8(FIT, HELD) < REL_MIN at either
final specimen (raw); else STABLE-THIN if raw HELD median C_8 >= C_THIN in
both final specimens; NO-THIN-GEOMETRY if raw HELD median C_16 < C_FLOOR
in either; else PARTIAL-THIN. MEAN-ONLY if STABLE-THIN and centered HELD
median C_8 < C_FLOOR in both; RESIDUAL-THIN if centered HELD median C_8
>= C_THIN in both. WRITER-SHARED if raw S_8(A, B) >= SHARED_FRAC x
min(self_A, self_B) at final; WRITER-ROTATED if < ROTATED_FRAC x min(self);
else WRITER-INDETERMINATE. LEARNED-DIFFERS if |C_8 final - C_8 W_0| >=
INIT_DELTA (raw HELD median) for both specimens in the same direction
(SHARPENED or DIFFUSED), else INIT-PRESENT. S_k over a rank-deficient
panel uses its non-degenerate components and still divides by k.

Outputs: logs/ugc0/census.json (refuse-if-exists), logs/ugc0/census.jsonl
(one line per finished cell, streamed), memmaps under logs/ugc0/tmp/
(untracked working artifacts; at most four on disk: the two writers at
the current state plus the previous state's pair until the velocity
projection is done; deleted after unless the parity check fails:
DEBUG-RETENTION clause, sha256 receipted).
SMOKE=1: specimen = the canonical seed-6 W_0 (non-target), 8 probe
batches (4 / 4), reference, outputs logs/ugc0/smoke<SMOKE_TAG>.jsonl.
Usage: .venv/bin/python scratch/update_geometry_census.py
"""
import collections
import datetime
import hashlib
import json
import os
import random
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

import birth19m_curric as C  # noqa: E402
import train_mathnative as TM  # noqa: E402
from atomtraj_pins import CLASSES, state_digest  # noqa: E402
from llmopt.lab.locator import resolve as _resolve  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")
KEYS = sorted(sum(CLASSES.values(), []))                 # the WRITER-TRAJECTORY-CENSUS-0 sorted-key tensor law (59 tensors)
GROUPS = {f"BLOCK{l}": sorted(k for k in KEYS if k.startswith(f"blocks.{l}.")) for l in range(8)}
GROUPS["OUTSIDE"] = sorted(k for k in KEYS if not k.startswith("blocks."))   # writertraj_depend GROUPS, literally
GRID = [0, 900, 3600, 7200, 10800, 13500, 15420]
A_PATHS = {s: f"checkpoints/phase19m/m{s:06d}.pt" for s in GRID if 0 < s < 15420}
A_PATHS[15420] = "checkpoints/gallery19m_phase_s2.pt"
B_PATHS = {s: f"checkpoints/backsched19m/m{s:06d}.pt" for s in GRID if 0 < s < 15420}
B_PATHS[15420] = "checkpoints/gallery19m_backsched_s2.pt"
NULL_LOCATORS = {"N3": {"worktree_role": "main", "relative_path": "checkpoints/atomtraj1/stock_s7/step_15420.pt"},
                 "N4": {"worktree_role": "repair", "relative_path": "checkpoints/atomtraj1/stock_s7/step_15420.pt"}}   # logical locators (llmopt.lab.locator); resolved at runtime, never a home path
NULL_PATHS = {k: str(_resolve(v)) for k, v in NULL_LOCATORS.items()}
DIGEST_PREFIX = {"A_final": "4633efe5d376f911", "B_final": "4beeedec5f9f5e91", "W0": "eb4b0bb427f86972"}
CENSUS = Path("logs/writertraj0/census.json")
N_FIT = 4 if SMOKE else 32
N_HELD = 4 if SMOKE else 32
N_PROBE = N_FIT + N_HELD
PROBE_SEED = "ugc0-probe-v1"
K_LIST = [1, 2, 4] if SMOKE else [1, 2, 4, 8, 16]
K_THIN, C_THIN, K_FLOOR, C_FLOOR = 8, 0.5, 16, 0.25
REL_MIN, SHARED_FRAC, ROTATED_FRAC, INIT_DELTA = 0.25, 0.8, 0.5, 0.15
K_LADDER = 4 if SMOKE else K_THIN
REF_SEED = 20260914
OUT_DIR = Path("logs/ugc0")
TMP = OUT_DIR / ("tmp_smoke" if SMOKE else "tmp")
OUT = OUT_DIR / (f"smoke{SMOKE_TAG}.jsonl" if SMOKE else "census.json")
STREAM = OUT_DIR / (f"smoke{SMOKE_TAG}_cells.jsonl" if SMOKE else "census.jsonl")
PARITY_TOL = 1e-4


# ---------------------------------------------------------------- pure geometry (Gram algebra)
def unit_gram(K, rows_a, rows_b=None):
    """Gram of unit-normalized rows: K_ab / sqrt(K_aa K_bb)."""
    rows_b = rows_a if rows_b is None else rows_b
    da = np.sqrt(np.diag(K)[rows_a]); db = np.sqrt(np.diag(K)[rows_b])
    return K[np.ix_(rows_a, rows_b)] / np.outer(da, db)


def centered_gram(K, fit_rows):
    """Gram of all rows after subtracting the FIT-panel mean m from every row:
    (g_i - m).(g_j - m) = K_ij - a_i - a_j + mm, a_i = mean_f K_if, mm = mean_ff' K_ff'."""
    a = K[:, fit_rows].mean(axis=1)
    mm = K[np.ix_(fit_rows, fit_rows)].mean()
    return K - a[:, None] - a[None, :] + mm


def cross_centered(Kxx, Kyy, Kxy, fit_x, fit_y):
    """Cross-Gram between X rows centered by X's FIT mean and Y rows centered by Y's FIT mean:
    (x_i - mx).(y_j - my) = Kxy_ij - mean_f Kxy_i,f(y) - mean_f Kxy_f(x),j + mean Kxy_fx,fy."""
    return Kxy - Kxy[:, fit_y].mean(axis=1)[:, None] - Kxy[fit_x, :].mean(axis=0)[None, :] + Kxy[np.ix_(fit_x, fit_y)].mean()


def coherent_fraction(K, rows):
    """Q = ||mean_i g_i||^2 / mean_i ||g_i||^2 over the given rows."""
    Kr = K[np.ix_(rows, rows)]
    return float(Kr.mean() / np.diag(Kr).mean())


def fit_spectrum(Kff, ks):
    lam = np.clip(np.linalg.eigvalsh(Kff)[::-1], 0, None)
    tot = lam.sum()
    pr = float(tot ** 2 / (lam ** 2).sum()) if tot > 0 else None
    return {"eigenvalues": [float(x) for x in lam], "participation_ratio": pr,
            "topk_energy": {k: float(lam[:k].sum() / tot) if tot > 0 else None for k in ks}}


def held_capture(Kff, Kfh, Khh_diag, ks):
    """C_k(h) = ||P_k h||^2 / ||h||^2, P_k = top-k right singular subspace of the FIT rows.
    With Kff = U L U^T: ||P_k h||^2 = sum_{j<=k} (u_j . c)^2 / lam_j, c = X_fit h (a column of Kfh)."""
    lam, U = np.linalg.eigh(Kff)
    order = np.argsort(lam)[::-1]; lam, U = lam[order], U[:, order]
    proj = U.T @ Kfh                                     # (n_fit, n_held): u_j . c_h
    out = {}
    for k in ks:
        keep = [j for j in range(k) if lam[j] > 1e-12 * max(lam[0], 1e-300)]
        cap = (proj[keep, :] ** 2 / lam[keep][:, None]).sum(axis=0) / Khh_diag
        out[k] = [float(x) for x in cap]
    return out


def subspace_overlap(Kxx, Kyy, Kxy, ks):
    """S_k = tr(P_X P_Y) / k = ||V_Xk^T V_Yk||_F^2 / k, V_X = X^T U_X L_X^{-1/2}:
    V_X^T V_Y = L_X^{-1/2} U_X^T (X Y^T) U_Y L_Y^{-1/2}."""
    lx, Ux = np.linalg.eigh(Kxx); ox = np.argsort(lx)[::-1]; lx, Ux = lx[ox], Ux[:, ox]
    ly, Uy = np.linalg.eigh(Kyy); oy = np.argsort(ly)[::-1]; ly, Uy = ly[oy], Uy[:, oy]
    M = Ux.T @ Kxy @ Uy
    tol_x, tol_y = 1e-12 * max(lx[0], 1e-300), 1e-12 * max(ly[0], 1e-300)
    out = {}
    for k in ks:
        kx = [j for j in range(k) if lx[j] > tol_x]; ky = [j for j in range(k) if ly[j] > tol_y]   # rank-deficient panels (a centered panel loses one dimension) use their non-degenerate components only
        B = M[np.ix_(kx, ky)] / np.outer(np.sqrt(lx[kx]), np.sqrt(ly[ky]))
        out[k] = float((B ** 2).sum() / k)
    return out


def projection_fraction(Kff, c, v_norm2, ks):
    """||P_k v||^2 / ||v||^2 for a vector v given c = X_fit v and ||v||^2."""
    lam, U = np.linalg.eigh(Kff); o = np.argsort(lam)[::-1]; lam, U = lam[o], U[:, o]
    proj = U.T @ c
    out = {}
    for k in ks:
        keep = [j for j in range(k) if lam[j] > 1e-12 * max(lam[0], 1e-300)]
        out[k] = float((proj[keep] ** 2 / lam[keep]).sum() / v_norm2) if v_norm2 > 0 else None
    return out


def geometry_from_grams(K, fit, held, ks, Kxy_self=None):
    """All single-specimen readouts from the full 64 x 64 Gram K (rows: fit then held indices)."""
    K = np.asarray(K, dtype=np.float64)
    fit = list(fit); held = list(held)
    out = {"Q": {"fit": coherent_fraction(K, fit), "held": coherent_fraction(K, held), "all": coherent_fraction(K, fit + held)},
           "row_norm2": {"fit_mean": float(np.diag(K)[fit].mean()), "held_mean": float(np.diag(K)[held].mean())}}
    for name, Kc in (("raw", K), ("centered", centered_gram(K, fit))):
        Kff = unit_gram(Kc, fit); Kfh = unit_gram(Kc, fit, held); Khh = unit_gram(Kc, held)
        cap = held_capture(Kff, Kfh, np.ones(len(held)), ks)
        out[name] = {"fit_spectrum": fit_spectrum(Kff, ks),
                     "held_capture": {k: {"median": float(np.median(v)), "mean": float(np.mean(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)), "min": float(np.min(v))} for k, v in cap.items()},
                     "held_capture_values": {k: v for k, v in cap.items()},
                     "reliability_fit_held": subspace_overlap(Kff, Khh, Kfh, ks),
                     "mean_cos_fit": float((Kff.sum() - np.trace(Kff)) / (len(fit) * (len(fit) - 1)))}
    return out


def pair_overlap_from_grams(KX, KY, KXY, fit_x, fit_y, ks):
    fit_x = list(fit_x); fit_y = list(fit_y)
    out = {}
    KXc = centered_gram(KX, fit_x); KYc = centered_gram(KY, fit_y); KXYc = cross_centered(KX, KY, KXY, fit_x, fit_y)
    for name, (kx, ky, kxy) in (("raw", (KX, KY, KXY)), ("centered", (KXc, KYc, KXYc))):
        dx = np.sqrt(np.diag(kx)[fit_x]); dy = np.sqrt(np.diag(ky)[fit_y])
        out[name] = subspace_overlap(unit_gram(kx, fit_x), unit_gram(ky, fit_y), kxy[np.ix_(fit_x, fit_y)] / np.outer(dx, dy), ks)
    return out


def adjudicate(final, w0, pair_final, ks_present):
    """Pure ladder. final: {"A": geom, "B": geom} (geometry_from_grams output), w0: geom, pair_final: pair_overlap (raw S_k A/B).
    Returns the fired axes literally."""
    kt, kf = K_LADDER, (K_FLOOR if K_FLOOR in ks_present else max(ks_present))
    rel = {x: final[x]["raw"]["reliability_fit_held"][kt] for x in ("A", "B")}
    axes = {"reliability_S_k": rel}
    if min(rel.values()) < REL_MIN:
        axes["resolution"] = "GEOMETRY-NOT-RESOLVED"
        return axes
    axes["resolution"] = "RESOLVED"
    c8 = {x: final[x]["raw"]["held_capture"][kt]["median"] for x in ("A", "B")}
    c16 = {x: final[x]["raw"]["held_capture"][kf]["median"] for x in ("A", "B")}
    cc8 = {x: final[x]["centered"]["held_capture"][kt]["median"] for x in ("A", "B")}
    axes.update({"raw_C_k_thin": c8, "raw_C_k_floor": c16, "centered_C_k": cc8})
    if min(c8.values()) >= C_THIN:
        axes["thinness"] = "STABLE-THIN"
    elif min(c16.values()) < C_FLOOR:
        axes["thinness"] = "NO-THIN-GEOMETRY"
    else:
        axes["thinness"] = "PARTIAL-THIN"
    if min(cc8.values()) >= C_THIN:
        axes["residual"] = "RESIDUAL-THIN"
    elif axes["thinness"] == "STABLE-THIN" and max(cc8.values()) < C_FLOOR:
        axes["residual"] = "MEAN-ONLY"
    else:
        axes["residual"] = "RESIDUAL-INDETERMINATE"
    s_ab = pair_final["raw"][kt]
    self_min = min(rel.values())
    axes["writer_S_k"] = {"AB": s_ab, "self_min": self_min}
    axes["writer"] = "WRITER-SHARED" if s_ab >= SHARED_FRAC * self_min else ("WRITER-ROTATED" if s_ab < ROTATED_FRAC * self_min else "WRITER-INDETERMINATE")
    c8_w0 = w0["raw"]["held_capture"][kt]["median"]
    d = {x: c8[x] - c8_w0 for x in ("A", "B")}
    axes["init_delta_C_k"] = {"W0": c8_w0, **d}
    axes["learned_v_init"] = ("LEARNED-DIFFERS-" + ("SHARPENED" if min(d.values()) > 0 else "DIFFUSED")) if min(abs(v) for v in d.values()) >= INIT_DELTA and (min(d.values()) > 0 or max(d.values()) < 0) else "INIT-PRESENT"
    return axes


# ---------------------------------------------------------------- model / probe / gradients
def load_sd(p):
    d = torch.load(p, map_location="cpu")
    return d["model"] if isinstance(d, dict) and "model" in d else d


def w0_seed(seed):
    tok = TM.MathTokenizer()
    torch.manual_seed(seed)
    m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def build(tok, dev):
    return TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536).to(dev)


def flatten_law(model):
    """(offsets, d, segments) for the sorted-key tensor law; segments = [(key, start, end, group)]."""
    shapes = {k: tuple(v.shape) for k, v in model.named_parameters()}
    assert sorted(shapes) == KEYS, "trainable tensors v the sorted-key law"
    segs, off = [], 0
    g_of = {k: g for g, ks in GROUPS.items() for k in ks}
    for k in KEYS:
        n = int(np.prod(shapes[k]))
        segs.append((k, off, off + n, g_of[k]))
        off += n
    digest = hashlib.sha256(json.dumps([(k, shapes[k]) for k in KEYS]).encode()).hexdigest()
    return segs, off, digest


def probe_batches(tok):
    rows = C.load_excised_rows()
    enc, levels = C.encode_with_levels(rows, tok)
    starts = [(i, i + TM.BS) for i in range(0, len(enc) - TM.BS, TM.BS)]     # the trainer's stock_epoch_stream slices
    chosen = random.Random(PROBE_SEED).sample(starts, N_PROBE)
    batches = [[enc[j] for j in range(a, b)] for a, b in chosen]
    h = hashlib.sha256(json.dumps({"slices": chosen, "ids": batches}).encode()).hexdigest()
    lv = [[levels[j] for j in range(a, b)] for a, b in chosen]
    return batches, {"slices": chosen, "digest": h, "n_enc": len(enc), "n_rows": len(rows), "fit": list(range(N_FIT)), "held": list(range(N_FIT, N_PROBE)),
                     "level_mix": dict(collections.Counter(x for b in lv for x in b)), "tokens_per_batch": [sum(len(s) for s in b) for b in batches]}


def grad_row(model, tok, batch, dev, segs, d):
    """One probe batch: the trainer's CE, backward, gradient flattened under the law. Returns (float32 row, loss, n_labels)."""
    L = max(len(s) for s in batch)
    ids = torch.tensor([s + [tok.pad_id] * (L - len(s)) for s in batch], device=dev)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in batch], device=dev)
    model.zero_grad(set_to_none=True)
    logits = model(ids[:, :-1], mask[:, :-1])
    labels = ids[:, 1:].clone()
    labels[mask[:, 1:] == 0] = -100
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
    loss.backward()
    params = dict(model.named_parameters())
    row = np.empty(d, dtype=np.float32)
    for k, a, b, _ in segs:
        g = params[k].grad
        assert g is not None, k
        row[a:b] = g.detach().to("cpu", torch.float32).reshape(-1).numpy()
    n_lab = int((labels != -100).sum())
    model.zero_grad(set_to_none=True)
    return row, float(loss.detach().item()), n_lab


def cell_gradients(model, tok, batches, dev, segs, d, path):
    """Write the (n, d) float32 memmap; keep row 0 in float64 for the parity check. Returns receipts."""
    n = len(batches)
    G = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, d))
    losses, nlab, row0 = [], [], None
    direct = []
    for i, b in enumerate(batches):
        row, loss, nl = grad_row(model, tok, b, dev, segs, d)
        G[i] = row
        losses.append(loss); nlab.append(nl)
        r64 = row.astype(np.float64)
        if i == 0:
            row0 = r64
        direct.append(float(row0 @ r64))
    G.flush(); del G
    return {"losses": losses, "n_labels": nlab, "direct_row0_dots": direct, "bytes": os.path.getsize(path), "shape": [n, d], "dtype": "float32"}


def reduce_grams(path, n, d, segs, other=None, vec=None):
    """Float64 per-tensor-block reductions over the memmap: K (n x n) global and per group; cross-Gram with `other`
    (another memmap of the same shape); c = G v and ||v||^2 (global and per group) for the segment-keyed vector `vec`."""
    G = np.memmap(path, dtype=np.float32, mode="r", shape=(n, d))
    H = np.memmap(other, dtype=np.float32, mode="r", shape=(n, d)) if other else None
    K = np.zeros((n, n)); Kg = {g: np.zeros((n, n)) for g in GROUPS}
    X = np.zeros((n, n)) if H is not None else None
    c = np.zeros(n) if vec is not None else None
    cg = {g: np.zeros(n) for g in GROUPS} if vec is not None else None
    v2 = 0.0; v2g = {g: 0.0 for g in GROUPS}
    for k, a, b, g in segs:
        blk = np.asarray(G[:, a:b], dtype=np.float64)
        kk = blk @ blk.T
        K += kk; Kg[g] += kk
        if H is not None:
            X += blk @ np.asarray(H[:, a:b], dtype=np.float64).T
        if vec is not None:
            v = vec[k]
            cc = blk @ v
            c += cc; cg[g] += cc
            vv = float(v @ v); v2 += vv; v2g[g] += vv
    del G, H
    out = {"K": K, "K_group": Kg}
    if X is not None:
        out["X"] = X
    if vec is not None:
        out["c"] = c; out["c_group"] = cg; out["v2"] = v2; out["v2_group"] = v2g
    return out


def write_reference(path, n, d, seed):
    """Synthetic isotropic Gaussian rows (float32) through the identical memmap + reductions."""
    rng = np.random.default_rng(seed)
    G = np.memmap(path, dtype=np.float32, mode="w+", shape=(n, d))
    step = 1_000_000
    for a in range(0, d, step):
        G[:, a:min(a + step, d)] = rng.standard_normal((n, min(a + step, d) - a), dtype=np.float32)
    G.flush(); del G


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def parity(direct, K):
    """Relative error of the memmap float32 Gram v the float64 direct row-0 inner products."""
    errs = [abs(K[0, j] - direct[j]) / max(abs(direct[j]), 1e-300) for j in range(len(direct))]
    return {"max_rel_err": float(max(errs)), "ok": bool(max(errs) <= PARITY_TOL)}


def flat_vec(sd_next, sd_prev):
    return {k: (sd_next[k].double() - sd_prev[k].double()).reshape(-1).numpy() for k in KEYS}


# ---------------------------------------------------------------- main
def specimen_plan():
    if SMOKE:
        return [("S6W0", 0, None)]
    plan = [("W0", 0, None)]
    for s in GRID[1:]:
        plan.append(("A", s, A_PATHS[s])); plan.append(("B", s, B_PATHS[s]))
    plan += [("N3", 15420, NULL_PATHS["N3"]), ("N4", 15420, NULL_PATHS["N4"])]
    return plan


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists() or STREAM.exists():
        raise SystemExit(f"REFUSING: {OUT} or {STREAM} exists")
    if TMP.exists():
        raise SystemExit(f"REFUSING: {TMP} exists (stale working directory)")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered census on a dirty tree")
    torch.manual_seed(0)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = TM.MathTokenizer()
    assert len(tok.vocab) == 40 and not os.environ.get("VOCAB_EXTRA")
    assert not os.environ.get("SEQ_CAP") and not os.environ.get("BIRTH_BS") and TM.BS == 32, "probe knobs must be the trainer's defaults"
    model = build(tok, dev)
    segs, d, flat_digest = flatten_law(model)
    assert d == 18_911_616, d
    batches, probe = probe_batches(tok)
    census = json.loads(CENSUS.read_text())
    for s_ in GRID[1:]:   # mid-grid milestones are bound to the census artifact's own A / B path lists (provenance), finals by digest below
        assert census["artifacts"]["A_paths"][str(s_)] == A_PATHS[s_] and census["artifacts"]["B_paths"][str(s_)] == B_PATHS[s_], f"grid step {s_} v census paths"
    rec = {"prereg": "UPDATE-GEOMETRY-CENSUS-0", "kind": "update_geometry_census", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty, "source_sha256": sha256_file(__file__),
           "device": dev, "torch_version": torch.__version__, "numpy_version": np.__version__, "n_keys": len(KEYS), "d": d, "flatten_law_digest": flat_digest, "groups": {g: len(v) for g, v in GROUPS.items()},
           "grid": GRID, "paths": {"A": A_PATHS, "B": B_PATHS, "null": NULL_LOCATORS}, "probe": probe, "n_fit": N_FIT, "n_held": N_HELD, "probe_seed": PROBE_SEED, "bs": TM.BS, "k_list": K_LIST,
           "thresholds": {"K_THIN": K_THIN, "C_THIN": C_THIN, "K_FLOOR": K_FLOOR, "C_FLOOR": C_FLOOR, "REL_MIN": REL_MIN, "SHARED_FRAC": SHARED_FRAC, "ROTATED_FRAC": ROTATED_FRAC, "INIT_DELTA": INIT_DELTA, "K_LADDER": K_LADDER},
           "parity_tol": PARITY_TOL, "ref_seed": REF_SEED, "census_sha256": sha256_file(CENSUS), "w0_seed2_state_digest_from_census": census["w0_seed2_state_digest"],
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    TMP.mkdir(parents=True)
    fit, held = probe["fit"], probe["held"]
    n = N_PROBE
    cells, Ks, retained = {}, {}, {}
    t0 = time.time()

    def stream(row):
        with STREAM.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def load_state(name, step, path):
        if name in ("W0", "S6W0"):
            return w0_seed(2 if name == "W0" else 6)
        return load_sd(path)

    def run_cell(name, step, path):
        """One specimen at one state: gradients -> memmap -> Grams -> geometry. Returns the memmap path (caller deletes)."""
        t1 = time.time()
        sd = load_state(name, step, path)
        dg_before = state_digest(sd)
        if name == "W0":
            assert dg_before.startswith(DIGEST_PREFIX["W0"]) and dg_before == census["w0_seed2_state_digest"], "W0 digest v census"
        if (name, step) == ("A", 15420):
            assert dg_before.startswith(DIGEST_PREFIX["A_final"]), "A final digest"
        if (name, step) == ("B", 15420):
            assert dg_before.startswith(DIGEST_PREFIX["B_final"]), "B final digest"
        model.load_state_dict(sd); model.eval()
        mm = str(TMP / f"{name}_{step}.f32")
        g = cell_gradients(model, tok, batches, dev, segs, d, mm)
        dg_after = state_digest({k: v.detach().cpu() for k, v in model.state_dict().items()})
        assert dg_after == dg_before, f"STATE MUTATED during {name}@{step}"
        red = reduce_grams(mm, n, d, segs)
        par = parity(g["direct_row0_dots"], red["K"])
        geom = geometry_from_grams(red["K"], fit, held, K_LIST)
        groups = {}
        for gn in GROUPS:
            gg = geometry_from_grams(red["K_group"][gn], fit, held, K_LIST)
            groups[gn] = {"Q": gg["Q"], "row_norm2": gg["row_norm2"]}
            for kind in ("raw", "centered"):
                groups[gn][kind] = {"held_capture": gg[kind]["held_capture"], "reliability_fit_held": gg[kind]["reliability_fit_held"],
                                    "participation_ratio": gg[kind]["fit_spectrum"]["participation_ratio"], "topk_energy": gg[kind]["fit_spectrum"]["topk_energy"]}
        cell = {"specimen": name, "step": step, "path": (NULL_LOCATORS[name] if name in NULL_LOCATORS else path), "state_digest": dg_before, "state_digest_after": dg_after, "file_sha256": sha256_file(path) if path else None,
                "loss_mean": float(np.mean(g["losses"])), "losses": g["losses"], "n_labels": g["n_labels"], "memmap": {"shape": g["shape"], "dtype": g["dtype"], "bytes": g["bytes"]},
                "parity": par, "geometry": geom, "geometry_groups": groups, "wall_s": round(time.time() - t1, 1)}
        if not par["ok"]:
            cell["memmap"]["retained_sha256"] = sha256_file(mm); retained[(name, step)] = mm
        cells[(name, step)] = cell; Ks[(name, step)] = red["K"]
        stream({"kind": "cell", **{k: v for k, v in cell.items() if k != "geometry_groups"}})
        print(f"[ugc0] cell {name}@{step}: loss {cell['loss_mean']:.4f} Q {geom['Q']['all']:.3f} rawC{K_LADDER} {geom['raw']['held_capture'][K_LADDER]['median']:.3f} centC{K_LADDER} {geom['centered']['held_capture'][K_LADDER]['median']:.3f} rel {geom['raw']['reliability_fit_held'][K_LADDER]:.3f} parity {par['max_rel_err']:.1e} ({cell['wall_s']}s, total {time.time() - t0:.0f}s)", flush=True)
        return mm

    def drop(key, mm):
        if key not in retained:
            os.remove(mm)

    def velocity_at(s_prev, s_next, mm_of):
        """Descriptive: v = theta_w(s_next) - theta_w(s_prev) projected on the FIT subspaces of the memmaps at s_prev
        (own writer, other writer; W0 shared at s_prev = 0). Next-state weights are read from disk (not censused here)."""
        out = {}
        for w, other in (("A", "B"), ("B", "A")):
            src = ("W0", 0) if s_prev == 0 else (w, s_prev)
            oth = ("W0", 0) if s_prev == 0 else (other, s_prev)
            nxt = load_sd((A_PATHS if w == "A" else B_PATHS)[s_next])
            prev = load_state(*src, (A_PATHS if w == "A" else B_PATHS).get(s_prev))
            v = flat_vec(nxt, prev)
            own = reduce_grams(mm_of[src], n, d, segs, vec=v)
            oth_r = reduce_grams(mm_of[oth], n, d, segs, vec=v)
            dn = np.sqrt(np.diag(Ks[src])[fit]); do = np.sqrt(np.diag(Ks[oth])[fit])
            out[f"{w}:{s_prev}->{s_next}"] = {"v_norm": float(np.sqrt(own["v2"])), "other_specimen": "W0(shared)" if s_prev == 0 else other,
                                              "own_subspace": projection_fraction(unit_gram(Ks[src], fit), own["c"][fit] / dn, own["v2"], K_LIST),
                                              "other_subspace": projection_fraction(unit_gram(Ks[oth], fit), oth_r["c"][fit] / do, oth_r["v2"], K_LIST),
                                              "cos_v_meangrad_own": float(own["c"].mean() / (np.sqrt(own["v2"]) * np.sqrt(Ks[src].mean()))) if own["v2"] > 0 and Ks[src].mean() > 0 else None}
        return out

    pairs, velocity = {}, {}
    if SMOKE:
        mm = run_cell("S6W0", 0, None)
        drop(("S6W0", 0), mm)
    else:
        mm_w0 = run_cell("W0", 0, None)
        mm_prev = {("W0", 0): mm_w0}
        s_prev = 0
        for s in GRID[1:]:
            mmA = run_cell("A", s, A_PATHS[s]); mmB = run_cell("B", s, B_PATHS[s])
            X = reduce_grams(mmA, n, d, segs, other=mmB)["X"]
            pairs[f"A_B@{s}"] = pair_overlap_from_grams(Ks[("A", s)], Ks[("B", s)], X, fit, fit, K_LIST)
            stream({"kind": "pair", "state": s, "overlap": pairs[f"A_B@{s}"]})
            velocity.update(velocity_at(s_prev, s, mm_prev))
            stream({"kind": "velocity", "interval_end": s, "velocity": {k: v for k, v in velocity.items() if k.endswith(f"->{s}")}})
            for key, mm in mm_prev.items():
                drop(key, mm)
            mm_prev = {("A", s): mmA, ("B", s): mmB}; s_prev = s
        for key, mm in mm_prev.items():
            drop(key, mm)
        mm3 = run_cell("N3", 15420, NULL_PATHS["N3"]); mm4 = run_cell("N4", 15420, NULL_PATHS["N4"])
        Xn = reduce_grams(mm3, n, d, segs, other=mm4)["X"]
        pairs["N3_N4@15420"] = pair_overlap_from_grams(Ks[("N3", 15420)], Ks[("N4", 15420)], Xn, fit, fit, K_LIST)
        stream({"kind": "pair", "state": "N3_N4@15420", "overlap": pairs["N3_N4@15420"]})
        drop(("N3", 15420), mm3); drop(("N4", 15420), mm4)
    # isotropic reference through the identical code (two independent draws for the random self-overlap)
    ref_mm, ref2 = str(TMP / "reference.f32"), str(TMP / "reference2.f32")
    write_reference(ref_mm, n, d, REF_SEED); write_reference(ref2, n, d, REF_SEED + 1)
    rr = reduce_grams(ref_mm, n, d, segs); rr2 = reduce_grams(ref2, n, d, segs)
    X12 = reduce_grams(ref_mm, n, d, segs, other=ref2)["X"]
    ref = {"geometry": geometry_from_grams(rr["K"], fit, held, K_LIST), "analytic_floor_k_over_d": {k: k / d for k in K_LIST},
           "self_pair_overlap_two_draws": pair_overlap_from_grams(rr["K"], rr2["K"], X12, fit, fit, K_LIST),
           "groups": {gn: {"raw_held_capture_median": {k: v["median"] for k, v in geometry_from_grams(rr["K_group"][gn], fit, held, K_LIST)["raw"]["held_capture"].items()},
                           "analytic_floor": {k: k / sum(b - a for _, a, b, g in segs if g == gn) for k in K_LIST}} for gn in GROUPS}}
    stream({"kind": "reference", "reference": ref})
    os.remove(ref_mm); os.remove(ref2)
    if not SMOKE:
        axes = adjudicate({"A": cells[("A", 15420)]["geometry"], "B": cells[("B", 15420)]["geometry"]}, cells[("W0", 0)]["geometry"], pairs["A_B@15420"], K_LIST)
    else:
        axes = {"smoke": True, "S6W0_raw_C": cells[("S6W0", 0)]["geometry"]["raw"]["held_capture"][K_LADDER]["median"]}
    kept = list(retained.values())
    if not kept:
        shutil.rmtree(TMP)
    rec.update({"cells": [c for c in cells.values()], "pairs": pairs, "velocity": velocity, "reference": ref, "axes": axes, "retained_memmaps": kept,
                "wall_s": round(time.time() - t0, 1), "ended_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")})
    if SMOKE:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[ugc0] smoke row appended: axes {axes} parity {[c['parity'] for c in cells.values()]} ref rawC{K_LADDER} {ref['geometry']['raw']['held_capture'][K_LADDER]['median']:.2e}")
    else:
        OUT.write_text(json.dumps(rec, indent=1))
        print(f"[ugc0] written {OUT}: axes {json.dumps(axes)} wall {rec['wall_s']}s")


if __name__ == "__main__":
    main()

"""MEZO-SIGNAL-DESK-0: zero-training feasibility desk for a zeroth-order
(MeZO / SPSA) credit writer over the frozen-random-backbone arena (PRE-REG
MEZO-SIGNAL-DESK-0). Nothing trains; no gate is read; no birth.

Arena and states: the retained paired-BP CONTROL states (emb + blocks 0..3
frozen at W_0; the trainable vector theta = blocks 4..7 + norm + head,
d = 9,456,000 in 30 tensors) of the seed-27 (checkpoints/sgwriter1) and
seed-28 (checkpoints/sgbb7) controls at steps 463, 3,084, 5,140, 15,420;
every state digest asserted against its qual.jsonl. Loss L(theta) = the
control's exact CE objective (dfa_credit.hybrid_objective(k=8) over
freeze_lower(model, 4)) on a FIXED evaluation batch (a 32-row chunk of the
frozen probe; BATCH_CHUNKS). fp32 CPU (the training dtype); a small fp64
reference for finite-difference fidelity.

Per (state, batch):
  g        exact BP gradient of L over theta (DIAGNOSTIC ORACLE only).
  Families of fixed perturbations z_k, k < K (perturbation seed
           PERT_SEED_BASE + 1000 * family_index + k, torch.Generator CPU;
           the SAME K directions in every cell: a paired design across
           states and batches, so the pooled bar sees 4 direction groups
           x 16 cells, not 64 independent draws):
    vanilla  Rademacher +-1 on every coordinate of theta (PRIMARY; vanilla
             full-trainable-vector SPSA / MeZO);
    rank1    the ONE registered secondary structured family: every 2-D
             tensor W (rows x cols) gets u v^T with u, v Rademacher (entries
             +-1, so the per-entry scale equals vanilla's; effective
             dimension rows + cols), every 1-D tensor gets Rademacher.
  d_BP_k   = <g, z_k>                                     (exact)
  d_FD_k   = [L(theta + eps z_k) - L(theta - eps z_k)] / (2 eps) at every
             eps in EPS (antithetic; EPS[0] = 1e-3 is PRIMARY), fp32; and
             at the primary eps in fp64 for the first K_F64 directions.
  (1) FD fidelity: per family and eps, over the K directions: Pearson
      correlation of d_FD v d_BP, median relative error
      |d_FD - d_BP| / |d_BP|, sign agreement rate.
  (2) Intrinsic SPSA variance: the IDEAL estimator g_hat_m = (1/m)
      sum_{i in group} d_BP_i z_i over disjoint groups of m of the K
      directions, m in M_ALL; cosine to g per group; median per m; the
      isotropic reference sqrt(m / d_family) is reported beside it.
  (3) Actual estimator: the same with d_FD_i (primary eps, fp32).
  (4) Virtual-step descent (non-persistent, the registered law): for a
      direction w, u = w / ||w||, D(u) = min over eta in ETA_GRID of
      L(theta - eta u) - L(theta) (theta restored after every evaluation);
      D_BP = D(g / ||g||); ratio R(w) = D(u) / D_BP (both <= 0 in
      practice; R is reported as None if D_BP >= 0). Computed for the
      first N_VIRTUAL estimator groups of every m in M_PRACTICAL, ideal and
      actual, per family.
Adjudication (adjudicate, pure):
  FD-FAITHFUL  iff over vanilla, primary eps, all states x batches x K:
               sign agreement >= 0.95 and median relative error <= 0.10;
               else NOT-RESOLVABLE-FD (nothing licensed, nothing parked).
  BAR-SIGNAL   per family: FIRES iff the pooled median of the ACTUAL
               (d_FD) estimator's descent ratio R at m = 4 (the largest
               practical budget) is >= R_LICENSE = 0.1.
  Verdict: vanilla FIRES -> MEZO-LICENSED; only rank1 FIRES ->
           STRUCTURED-ZO-LICENSED; neither -> PARK (the foreign-writer
           program parks). Larger m are descriptive only.
Writes logs/mezo0/desk.json (refuses to overwrite) + desk.jsonl. SMOKE=1: a
fresh seed-11 W_0 (no checkpoint), one batch (chunk 0), K 8, primary eps
only, logs/mezo0/smoke<SMOKE_TAG>.jsonl.
Usage: .venv/bin/python scratch/mezo_signal_desk.py
"""
import datetime
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402
from torch.nn.utils import parameters_to_vector, vector_to_parameters  # noqa: E402

import train_mathnative as TM  # noqa: E402
from atomtraj_pins import state_digest  # noqa: E402
from dfa_credit import freeze_lower, hybrid_objective  # noqa: E402
from dfa_probe import probe_rows, probe_tensors  # noqa: E402

SMOKE = os.environ.get("SMOKE", "0") == "1"
SMOKE_TAG = os.environ.get("SMOKE_TAG", "")
assert not SMOKE_TAG or SMOKE, "SMOKE_TAG is smoke-only"
OUT_DIR = Path("logs/mezo0")
OUT = OUT_DIR / (f"smoke{SMOKE_TAG}.jsonl" if SMOKE else "desk.json")
STATES = [] if SMOKE else [
    ("logs/sgwriter1/qual.jsonl", "sgq_control_s27_lr0.0003", 27, [463, 3084, 5140, 15420]),
    ("logs/sgbb7/qual.jsonl", "sgb7_control_s28_lr0.0003", 28, [463, 3084, 5140, 15420]),
]
BATCH_CHUNKS = [0] if SMOKE else [2, 4]        # frozen probe chunks (32 rows; lengths 44 / 61)
CHUNK = 32
EPS = [1e-3] if SMOKE else [1e-3, 1e-2, 1e-4]   # EPS[0] primary
K = 8 if SMOKE else 64
K_F64 = 2 if SMOKE else 8
M_PRACTICAL = [1, 2] if SMOKE else [1, 2, 4]
M_ALL = [1, 2, 4, 8] if SMOKE else [1, 2, 4, 8, 16, 32, 64]
N_VIRTUAL = 2 if SMOKE else 4
ETA_GRID = [1e-2, 1e-1] if SMOKE else [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0]
FAMILIES = ["vanilla", "rank1"]
PERT_SEED_BASE = 5000
R_LICENSE = 0.1
FD_SIGN_MIN, FD_RELERR_MAX = 0.95, 0.10
SMOKE_SEED = 11
THREADS = 5


def rademacher(n, g):
    return (torch.randint(0, 2, (n,), generator=g, dtype=torch.int8).to(torch.float32) * 2 - 1)


def perturbation(family, params, seed):
    """A flat +-1 perturbation over the trainable vector. vanilla: iid
    Rademacher; rank1: u v^T per 2-D tensor (u, v Rademacher), Rademacher
    per 1-D tensor. Deterministic in (family, seed)."""
    g = torch.Generator().manual_seed(seed)
    parts = []
    for p in params:
        if family == "vanilla" or p.dim() != 2:
            parts.append(rademacher(p.numel(), g))
        else:
            u, v = rademacher(p.shape[0], g), rademacher(p.shape[1], g)
            parts.append(torch.outer(u, v).reshape(-1))
    return torch.cat(parts)


def effective_dim(family, params):
    if family == "vanilla":
        return sum(p.numel() for p in params)
    return sum((p.shape[0] + p.shape[1]) if p.dim() == 2 else p.numel() for p in params)


class Evaluator:
    """L(theta) on the fixed batch with theta restored after every call."""

    def __init__(self, model, ids, mask, labels):
        self.model, self.ids, self.mask, self.labels = model, ids, mask, labels
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.theta = parameters_to_vector(self.params).detach().clone()
        self.n_eval = 0

    def loss(self, delta=None):
        with torch.no_grad():
            if delta is not None:
                vector_to_parameters(self.theta + delta, self.params)
            L = float(hybrid_objective(self.model, [], self.ids, self.mask, self.labels, 8)["loss"].detach())
            if delta is not None:
                vector_to_parameters(self.theta, self.params)
        self.n_eval += 1
        return L

    def grad(self):
        self.model.zero_grad(set_to_none=True)
        ob = hybrid_objective(self.model, [], self.ids, self.mask, self.labels, 8)
        ob["loss"].backward()
        g = torch.cat([p.grad.detach().reshape(-1) for p in self.params]).clone()
        self.model.zero_grad(set_to_none=True)
        return float(ob["loss"].detach()), g


def pearson(a, b):
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    den = float(a.norm() * b.norm())
    return float((a * b).sum() / den) if den > 0 else None


def fidelity(d_fd, d_bp):
    rel = [abs(f - b) / abs(b) for f, b in zip(d_fd, d_bp) if b != 0]
    return {"pearson": pearson(d_fd, d_bp), "median_rel_err": statistics.median(rel) if rel else None,
            "sign_agreement": sum(1 for f, b in zip(d_fd, d_bp) if (f > 0) == (b > 0)) / len(d_bp), "n": len(d_bp),
            "median_abs_d_bp": statistics.median(abs(b) for b in d_bp)}


def cos(a, b):
    na, nb = float(a.norm()), float(b.norm())
    return float((a * b).sum() / (na * nb)) if na > 0 and nb > 0 else None


def estimator(zs, ds, idx):
    return sum(ds[i] * zs[i] for i in idx) / len(idx)


def virtual_step(ev, w, base_loss):
    u = w / float(w.norm())
    curve = {f"{eta:g}": ev.loss(-eta * u) - base_loss for eta in ETA_GRID}
    best = min(curve.values())
    return {"curve": curve, "D": best, "eta_best": min(curve, key=curve.get)}


def run_state_batch(model, ids, mask, labels, label):
    ev = Evaluator(model, ids, mask, labels)
    base, g = ev.grad()
    base_fwd = ev.loss()
    rec = {"label": label, "base_loss": base, "base_loss_fwd": base_fwd, "grad_norm": float(g.norm()), "d": int(g.numel()), "families": {}}
    vs_bp = virtual_step(ev, g, base)
    rec["virtual_bp"] = vs_bp
    for fi, fam in enumerate(FAMILIES):
        zs = [perturbation(fam, ev.params, PERT_SEED_BASE + 1000 * fi + k) for k in range(K)]
        d_bp = [float((g * z).sum()) for z in zs]
        frec = {"d_eff": effective_dim(fam, ev.params), "d_bp": d_bp, "d_fd": {}, "fidelity": {}, "ideal_cos": {}, "actual_cos": {}, "virtual": {"ideal": {}, "actual": {}}}
        for eps in EPS:
            d_fd = [(ev.loss(eps * z) - ev.loss(-eps * z)) / (2 * eps) for z in zs]
            frec["d_fd"][f"{eps:g}"] = d_fd
            frec["fidelity"][f"{eps:g}"] = fidelity(d_fd, d_bp)
        # fp64 reference at the primary eps on the first K_F64 directions
        m64 = model.double()
        ev64 = Evaluator(m64, ids, mask, labels)
        _, g64 = ev64.grad()                                        # fp64 oracle for the fp64 reference only
        d_bp64 = [float((g64 * z.double()).sum()) for z in zs[:K_F64]]
        d_fd64 = [(ev64.loss(EPS[0] * z.double()) - ev64.loss(-EPS[0] * z.double())) / (2 * EPS[0]) for z in zs[:K_F64]]
        frec["d_fd64_primary"] = d_fd64
        frec["d_bp64"] = d_bp64
        frec["fidelity_f64_primary"] = fidelity(d_fd64, d_bp64)                       # fp64 FD v fp64 gradient
        frec["fidelity_f32_v_f64_primary"] = fidelity(frec["d_fd"][f"{EPS[0]:g}"][:K_F64], d_fd64)   # fp32 FD v fp64 FD
        frec["fidelity_bp32_v_bp64"] = fidelity(d_bp[:K_F64], d_bp64)                 # fp32 gradient v fp64 gradient
        model.float()
        n_prev = ev.n_eval + ev64.n_eval
        ev = Evaluator(model, ids, mask, labels)
        ev.n_eval = n_prev
        d_act = frec["d_fd"][f"{EPS[0]:g}"]
        for m in M_ALL:
            groups = [list(range(j * m, (j + 1) * m)) for j in range(K // m)]
            frec["ideal_cos"][str(m)] = [cos(estimator(zs, d_bp, idx), g) for idx in groups]
            frec["actual_cos"][str(m)] = [cos(estimator(zs, d_act, idx), g) for idx in groups]
            if m in M_PRACTICAL:
                for kind, ds in (("ideal", d_bp), ("actual", d_act)):
                    out = []
                    for idx in groups[:N_VIRTUAL]:
                        vs = virtual_step(ev, estimator(zs, ds, idx), base)
                        vs["R"] = (vs["D"] / vs_bp["D"]) if vs_bp["D"] < 0 else None
                        out.append(vs)
                    frec["virtual"][kind][str(m)] = out
        frec["ref_cos_sqrt_m_over_d"] = {str(m): math.sqrt(m / frec["d_eff"]) for m in M_ALL}
        rec["families"][fam] = frec
        print(f"[mezo] {label} {fam}: fid@{EPS[0]:g} sign {frec['fidelity'][f'{EPS[0]:g}']['sign_agreement']:.2f} relerr {frec['fidelity'][f'{EPS[0]:g}']['median_rel_err']:.3f} "
              f"| ideal cos m=1/4 {statistics.median(frec['ideal_cos']['1']):.2e}/{statistics.median(frec['ideal_cos'][str(M_PRACTICAL[-1])]):.2e} (ref {frec['ref_cos_sqrt_m_over_d']['1']:.2e}) "
              f"| R actual m={M_PRACTICAL[-1]} {[round(v['R'], 6) if v['R'] is not None else None for v in frec['virtual']['actual'][str(M_PRACTICAL[-1])]]} D_bp {vs_bp['D']:.4f}", flush=True)
    rec["n_eval"] = ev.n_eval
    return rec


def adjudicate(rec):
    """Pure. Returns the FD precondition, per-family BAR-SIGNAL and verdict."""
    prim = f"{EPS[0]:g}"
    sb = rec["state_batches"]
    van = [x["families"]["vanilla"] for x in sb]
    sign = sum(f["fidelity"][prim]["sign_agreement"] * f["fidelity"][prim]["n"] for f in van) / sum(f["fidelity"][prim]["n"] for f in van)
    rels = [abs(a - b) / abs(b) for f in van for a, b in zip(f["d_fd"][prim], f["d_bp"]) if b != 0]
    rel = statistics.median(rels) if rels else None
    out = {"fd_sign_agreement_pooled": sign, "fd_median_rel_err_pooled": rel, "fd_faithful": bool(sign >= FD_SIGN_MIN and rel is not None and rel <= FD_RELERR_MAX), "families": {}}
    out["fd_by_eps"] = {}
    for fam in FAMILIES:
        fs = [x["families"][fam] for x in sb]
        out["fd_by_eps"][fam] = {e: {"sign_agreement_pooled": sum(f["fidelity"][e]["sign_agreement"] * f["fidelity"][e]["n"] for f in fs) / sum(f["fidelity"][e]["n"] for f in fs),
                                     "median_rel_err_pooled": statistics.median([abs(a - b) / abs(b) for f in fs for a, b in zip(f["d_fd"][e], f["d_bp"]) if b != 0])}
                                 for e in fs[0]["fidelity"]}
    m_lic = str(M_PRACTICAL[-1])
    for fam in FAMILIES:
        fr = {}
        for kind in ("ideal", "actual"):
            fr[kind] = {}
            for m in M_PRACTICAL:
                Rs = [v["R"] for x in sb for v in x["families"][fam]["virtual"][kind][str(m)] if v["R"] is not None]
                fr[kind][str(m)] = {"median_R": statistics.median(Rs) if Rs else None, "n": len(Rs), "frac_descent": (sum(1 for x in sb for v in x["families"][fam]["virtual"][kind][str(m)] if v["D"] < 0) / max(1, sum(len(x["families"][fam]["virtual"][kind][str(m)]) for x in sb)))}
        fr["cos_median_by_m"] = {str(m): {"ideal": statistics.median(c for x in sb for c in x["families"][fam]["ideal_cos"][str(m)] if c is not None),
                                          "actual": statistics.median(c for x in sb for c in x["families"][fam]["actual_cos"][str(m)] if c is not None),
                                          "ref": sb[0]["families"][fam]["ref_cos_sqrt_m_over_d"][str(m)]} for m in M_ALL}
        R = fr["actual"][m_lic]["median_R"]
        fr["bar_signal"] = ("FIRES" if (R is not None and R >= R_LICENSE) else "NO-FIRE")
        out["families"][fam] = fr
    if not out["fd_faithful"]:
        out["verdict"] = "NOT-RESOLVABLE-FD"
    elif out["families"]["vanilla"]["bar_signal"] == "FIRES":
        out["verdict"] = "MEZO-LICENSED"
    elif out["families"]["rank1"]["bar_signal"] == "FIRES":
        out["verdict"] = "STRUCTURED-ZO-LICENSED"
    else:
        out["verdict"] = "PARK"
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT.exists() and (not SMOKE or not SMOKE_TAG):
        raise SystemExit(f"REFUSING: {OUT} exists (a booked smoke receipt is never appended to: rerun with SMOKE_TAG)")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    if not SMOKE and dirty:
        raise SystemExit("REFUSING: registered desk on a dirty tree")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(THREADS)          # pinned: fp32 CPU reductions are thread-count dependent
    tok = TM.MathTokenizer()
    _, rows, probe_digest, _ = probe_rows(tok)
    batches = {c: probe_tensors(rows[c * CHUNK:(c + 1) * CHUNK], tok) for c in BATCH_CHUNKS}
    rec = {"prereg": "MEZO-SIGNAL-DESK-0", "kind": "mezo_signal_desk", "smoke": SMOKE, "commit": commit, "tree_dirty": dirty,
           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "probe_token_digest": probe_digest,
           "batch_chunks": BATCH_CHUNKS, "eps": EPS, "K": K, "K_f64": K_F64, "m_practical": M_PRACTICAL, "m_all": M_ALL, "n_virtual": N_VIRTUAL, "eta_grid": ETA_GRID,
           "families": FAMILIES, "pert_seed_base": PERT_SEED_BASE, "r_license": R_LICENSE, "fd_sign_min": FD_SIGN_MIN, "fd_relerr_max": FD_RELERR_MAX,
           "dtype": "float32 (fp64 reference)", "device": "cpu", "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"), "states": [], "state_batches": []}
    t0 = time.time()
    states = []
    if SMOKE:
        torch.manual_seed(SMOKE_SEED)
        m = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
        states.append(("w0_seed11", {k: v.detach().clone() for k, v in m.state_dict().items()}, None, None))
    else:
        for receipt, cell, seed, steps in STATES:
            b = next(json.loads(l) for l in open(receipt) if '"kind": "birth"' in l and f'"cell": "{cell}"' in l)
            for step in steps:
                p = Path(b["outdir"]) / f"step_{step:05d}.pt"
                sd = torch.load(p, map_location="cpu")
                assert state_digest(sd) == b["snapshots"][str(step)]["state_digest"], f"{p}: digest v receipt"
                states.append((f"s{seed}_step{step}", sd, str(p), b["snapshots"][str(step)]["state_digest"]))
    for label, sd, path, digest in states:
        rec["states"].append({"label": label, "path": path, "state_digest": digest})
        for c in BATCH_CHUNKS:
            torch.manual_seed(0)
            model = TM.build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
            model.load_state_dict(sd)
            model.train()
            freeze_lower(model, 4)
            ids, mask, labels = batches[c]
            sb = run_state_batch(model, ids, mask, labels, f"{label}/chunk{c}")
            sb.update({"state": label, "chunk": c, "n_tokens": int((labels != -100).sum())})
            rec["state_batches"].append(sb)
            print(f"[mezo] {label}/chunk{c} done: {sb['n_eval']} loss evaluations ({time.time() - t0:.0f}s)", flush=True)
    rec["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["adjudication"] = adjudicate(rec)
    if SMOKE:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[mezo] smoke row appended; verdict (smoke, not booked) {rec['adjudication']['verdict']}")
    else:
        OUT.write_text(json.dumps(rec, indent=1))
        with (OUT_DIR / "desk.jsonl").open("a") as f:
            f.write(json.dumps({k: v for k, v in rec.items() if k != "state_batches"} | {"adjudication": rec["adjudication"]}) + "\n")
        print(f"[mezo] written {OUT}; verdict {rec['adjudication']['verdict']}")


if __name__ == "__main__":
    main()

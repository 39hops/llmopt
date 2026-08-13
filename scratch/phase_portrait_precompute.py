"""PHASE-PORTRAIT-1 precompute: milestones -> data/anim/phase.npz.

Per milestone m and gate neuron i (12,288 = 8 layers x 1536):
  angle[m,i]   position: angle in the FINAL milestone's whitened-PCA
               basis (one fixed frame across the whole trajectory,
               same construction as the morph scene)
  mag[m,i]     absolute row norm
  vel[m,i]     finite-difference speed: ||W_m - W_{m-1}||_row / dstep
               (per optimizer step, so uneven milestone gaps compare)
  mom[m,i]     Adam momentum magnitude: ||exp_avg||_row — the
               optimizer's own velocity estimate, read not derived
  steps[m]     optimizer step of each milestone

Portrait-only instrument data (birth19m_phase precedent): no
capability numbers here; any gate read of these milestones needs its
own pre-reg.

Usage: .venv/bin/python scratch/phase_portrait_precompute.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

MDIR = Path("checkpoints/phase19m")
OUT = Path("data/anim/phase.npz")
KEY = "gate.weight"


def gate_rows(sd: dict) -> torch.Tensor:
    ks = sorted(k for k in sd if KEY in k)
    return torch.cat([sd[k].float() for k in ks], dim=0)


def main() -> None:
    files = sorted(MDIR.glob("m*.pt"))
    assert len(files) >= 10, f"only {len(files)} milestones in {MDIR}"
    steps, W, M = [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu")
        steps.append(int(d["step"]))
        W.append(gate_rows(d["model"]))
        # exp_avg keyed by param order; rebuild name order from model
        ks = sorted(k for k in d["model"] if KEY in k)
        names = list(d["model"])
        st = d["opt"]["state"]
        rows = [st[names.index(k)]["exp_avg"].float() for k in ks]
        M.append(torch.cat(rows, dim=0))
        print(f"  {f.name}: step {steps[-1]}")
    n = W[0].shape[0]

    mu = W[-1].mean(0)
    _, S, V = torch.linalg.svd(W[-1] - mu, full_matrices=False)

    def ang(Wm):
        P = ((Wm - mu) @ V[:2].T) / S[:2].clamp(min=1e-12)
        return torch.complex(P[:, 0], P[:, 1]).angle()

    nm = len(files)
    angle = np.zeros((nm, n), np.float32)
    mag = np.zeros((nm, n), np.float32)
    vel = np.zeros((nm, n), np.float32)
    mom = np.zeros((nm, n), np.float32)
    for m in range(nm):
        angle[m] = ang(W[m]).numpy()
        mag[m] = W[m].norm(dim=1).numpy()
        mom[m] = M[m].norm(dim=1).numpy()
        if m:
            dstep = steps[m] - steps[m - 1]
            vel[m] = ((W[m] - W[m - 1]).norm(dim=1) / dstep).numpy()

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    from matplotlib.colors import to_hex

    from llmopt.figures import figstyle
    ramp = {mode: [to_hex(figstyle.continuous("magnitude", mode)(i / 15))
                   for i in range(16)] for mode in ("light", "dark")}
    meta = {
        "head": head,
        "provenance": ("checkpoints/phase19m milestones, "
                       "scratch/birth19m_phase.py SEED=2 · portrait-only, "
                       "no capability claims"),
        "basis": "final-milestone whitened PCA (fixed across trajectory)",
        "steps": steps,
        "vel_note": "||dW||_row per optimizer step between milestones",
        "mom_note": "||Adam exp_avg||_row, read from optimizer state",
        "ramp": ramp, "chrome": figstyle.CHROME,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, meta=np.array(json.dumps(meta)),
                        steps=np.array(steps, np.int64),
                        angle=angle, mag=mag, vel=vel, mom=mom)
    print(f"wrote {OUT}: {nm} milestones x {n} neurons")
    print(f"  vel mean by milestone: "
          f"{[round(float(v.mean()), 6) for v in vel[1:]]}")
    print(f"  mom mean by milestone: "
          f"{[round(float(m.mean()), 6) for m in mom]}")


if __name__ == "__main__":
    main()

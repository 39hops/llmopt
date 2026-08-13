"""Precompute animation scene data (lab venv side).

The anim venv (.venv-anim, manim) carries no torch and never imports
figstyle; this script does the heavy/typed work in the LAB venv and
writes one .npz per scene under data/anim/ (untracked, regenerable):

  crystal  pca/sphere/polar xy + rank order for gallery19m_s1.pt
  morph    angle (fixed final-checkpoint basis) + absolute row norm
           at ep0 / ep1 / final, with per-stage mean norms in meta
  crest    routing_crest arm labels+values from docs/figures.json

Every npz carries a `meta` JSON string: provenance (ckpt sha8s or
figures.json), repo HEAD, panel labels, and the house ramp sampled
from figstyle.continuous at 16 stops per mode — the anim side
interpolates hexes and never invents color.

  .venv/bin/python scripts/anim_precompute.py --scene crystal
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from llmopt.lab import anatomy

OUTDIR = Path("data/anim")


def _ramp_stops(n: int = 16) -> dict[str, list[str]]:
    from matplotlib.colors import to_hex

    from llmopt.figures import figstyle
    stops = {}
    for mode in ("light", "dark"):
        cmap = figstyle.continuous("magnitude", mode)
        stops[mode] = [to_hex(cmap(i / (n - 1))) for i in range(n)]
    return stops


def _chrome() -> dict:
    from llmopt.figures import figstyle
    return figstyle.CHROME


def _projections(ckpt: str, key: str = "gate.weight") -> dict:
    label, W = anatomy.neuron_rows(ckpt, key)
    order = anatomy.rank_scale(W.norm(dim=1)).numpy().astype(np.float32)
    arrays = {"order": order}
    for method in ("pca", "sphere", "polar"):
        xs, ys, _ = anatomy.project(W, method)
        arrays[f"{method}_x"] = xs.numpy().astype(np.float32)
        arrays[f"{method}_y"] = ys.numpy().astype(np.float32)
    arrays["label"] = np.array(label)
    return arrays


def scene_crystal() -> tuple[dict, dict]:
    ckpt = "checkpoints/gallery19m_s1.pt"
    arrays = _projections(ckpt)
    meta = {"provenance": anatomy.checkpoint_provenance(ckpt),
            "title": "Three views of the same weights"}
    return arrays, meta


def scene_morph() -> tuple[dict, dict]:
    """One FIXED coordinate system across checkpoints: every stage is
    projected through the FINAL checkpoint's whitened PCA basis, and
    the y axis is the absolute row norm (no per-stage rescale) — so
    scale growth between epochs is visible instead of normalized away.
    Per-stage mean row norm ships in meta as the on-screen statistic."""
    import torch
    ckpts = {"ep0": "checkpoints/gallery19m_s1_ep0.pt",
             "ep1": "checkpoints/gallery19m_s1_ep1.pt",
             "final": "checkpoints/gallery19m_s1.pt"}
    Ws = {tag: anatomy.neuron_rows(ckpt)[1] for tag, ckpt in ckpts.items()}
    mu = Ws["final"].mean(0)
    _, S, V = torch.linalg.svd(Ws["final"] - mu, full_matrices=False)
    arrays: dict = {}
    prov, mean_norm = [], {}
    for tag, ckpt in ckpts.items():
        P = ((Ws[tag] - mu) @ V[:2].T) / S[:2].clamp(min=1e-12)
        ang = torch.complex(P[:, 0], P[:, 1]).angle()
        mag = Ws[tag].norm(dim=1)
        arrays[f"{tag}_angle"] = ang.numpy().astype(np.float32)
        arrays[f"{tag}_mag"] = mag.numpy().astype(np.float32)
        mean_norm[tag] = float(mag.mean())
        prov.append(f"{tag}={anatomy.checkpoint_provenance(ckpt)}")
    meta = {"provenance": " ".join(prov),
            "title": "the crystal forming", "stages": list(ckpts),
            "mean_norm": mean_norm,
            "basis": "final-checkpoint whitened PCA (fixed across stages)"}
    return arrays, meta


def scene_crest() -> tuple[dict, dict]:
    spec = json.loads(Path("docs/figures.json").read_text())
    fig = spec["routing_crest"]
    controls = [a for a in fig["arms"] if a["value"] == 0]
    arrays = {
        "seed": np.array([p["seed"] for p in fig["seed_pairs"]],
                         dtype=np.int32),
        "full": np.array([p["full"] for p in fig["seed_pairs"]],
                         dtype=np.float32),
        "mask": np.array([p["mask"] for p in fig["seed_pairs"]],
                         dtype=np.float32),
        "ctl_values": np.array([a["value"] for a in controls],
                               dtype=np.float32),
        "ctl_labels": np.array([a["label"] for a in controls]),
    }
    meta = {"provenance": "docs/figures.json routing_crest",
            "title": fig["title"], "denominator": fig["denominator"],
            "fence": fig["fence"]}
    return arrays, meta


SCENES = {"crystal": scene_crystal, "morph": scene_morph,
          "crest": scene_crest}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, choices=sorted(SCENES))
    a = ap.parse_args()
    import subprocess
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    arrays, meta = SCENES[a.scene]()
    meta.update(head=head, ramp=_ramp_stops(), chrome=_chrome())
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{a.scene}.npz"
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    sizes = {k: getattr(v, "shape", None) for k, v in arrays.items()}
    print(f"wrote {out}  {sizes}")


if __name__ == "__main__":
    main()

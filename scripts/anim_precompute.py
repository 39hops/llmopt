"""Precompute animation scene data (lab venv side).

The anim venv (.venv-anim, manim) carries no torch and never imports
figstyle; this script does the heavy/typed work in the LAB venv and
writes one .npz per scene under data/anim/ (untracked, regenerable):

  crystal  pca/sphere/polar xy + rank order for gallery19m_s1.pt
  morph    the same projections at ep0 / ep1 / final
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
            "title": "19M gate-neuron geometry"}
    return arrays, meta


def scene_morph() -> tuple[dict, dict]:
    ckpts = {"ep0": "checkpoints/gallery19m_s1_ep0.pt",
             "ep1": "checkpoints/gallery19m_s1_ep1.pt",
             "final": "checkpoints/gallery19m_s1.pt"}
    arrays: dict = {}
    prov = []
    for tag, ckpt in ckpts.items():
        pj = _projections(ckpt)
        for k, v in pj.items():
            if k != "label":
                arrays[f"{tag}_{k}"] = v
        prov.append(f"{tag}={anatomy.checkpoint_provenance(ckpt)}")
    meta = {"provenance": " ".join(prov),
            "title": "the crystal forming", "stages": list(ckpts)}
    return arrays, meta


def scene_crest() -> tuple[dict, dict]:
    spec = json.loads(Path("docs/figures.json").read_text())
    fig = spec["routing_crest"]
    arrays = {
        "values": np.array([a["value"] for a in fig["arms"]],
                           dtype=np.float32),
        "labels": np.array([a["label"] for a in fig["arms"]]),
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

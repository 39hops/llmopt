"""CLI for the README hero: llmopt.lab.anatomy dot views.

Each dot one neuron (a row of a projection matrix), color = row-norm
magnitude (rank-scaled), three projections (pca/sphere/polar). The
render logic lives in llmopt/lab/anatomy.py — reuse it there for any
matrix (streamed big-model expert shards included); this script is
the checkpoint-in, README-hero-out wrapper.

Usage:
  .venv/bin/python scripts/render_hero_neurons.py \
      --ckpt checkpoints/gallery19m_s1.pt \
      --out docs/assets/neurons-19m
  (writes <out>-light.png and <out>-dark.png at 300 dpi)
"""
import argparse


from llmopt.lab import anatomy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--key", default="gate.weight")
    ap.add_argument("--title", default="THE CRYSTAL — weight-space "
                    "anatomy of a math-native 19M model")
    ap.add_argument("--out", required=True,
                    help="output stem; writes <out>-light.png and "
                         "<out>-dark.png")
    a = ap.parse_args()
    label, W = anatomy.neuron_rows(a.ckpt, a.key)
    outs = anatomy.render_dot_views(
        W, a.out, a.title, source_label=f"rows of {label}",
        provenance=(anatomy.checkpoint_provenance(a.ckpt)
                    + " · render_hero_neurons.py"))
    for o in outs:
        print(f"saved {o}")


if __name__ == "__main__":
    main()

"""Gen-6 arm B: grow the champion 45M -> ~55M, function-preserving.

FFN-only growth (d fixed): gate/up gain +GROW rows via template
spray (near-orthogonal directions + 3% tilt toward 5 random
anchors, norms drawn from the existing rows' distribution); down
gains +GROW ZERO columns => identical function at step 0 (the
identity pre-check gates this checkpoint and must print the
champion's exact 69/120 before training).

    python scripts/grow_mathnative.py \
        --src checkpoints/mathnative_45m_gen4_std.pt \
        --out checkpoints/mathnative_gen6_grown.pt --grow 256
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    import torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--grow", type=int, default=256)
    a = ap.parse_args()
    sd = torch.load(a.src, map_location="cpu")
    g = torch.Generator().manual_seed(6)
    new = {}
    for k, W in sd.items():
        if k.endswith("gate.weight") or k.endswith("up.weight"):
            n, d = W.shape
            anchors = torch.randn(5, d, generator=g)
            anchors = anchors / anchors.norm(dim=1, keepdim=True)
            fam = torch.randint(0, 5, (a.grow,), generator=g)
            rows = torch.randn(a.grow, d, generator=g)
            rows = rows / rows.norm(dim=1, keepdim=True)
            rows = rows + 0.03 * anchors[fam]
            rows = rows / rows.norm(dim=1, keepdim=True)
            src_norms = W.norm(dim=1)
            idx = torch.randint(0, n, (a.grow,), generator=g)
            new[k] = torch.cat([W, rows * src_norms[idx].unsqueeze(1)])
        elif k.endswith("down.weight"):
            d, n = W.shape
            new[k] = torch.cat(
                [W, torch.zeros(d, a.grow, dtype=W.dtype)], dim=1)
        else:
            new[k] = W
    torch.save(new, a.out)
    Path(a.out + ".ep").write_text("-1")
    tot = sum(v.numel() for v in new.values())
    print(f"grown -> {a.out}: {tot/1e6:.1f}M params "
          f"(+{a.grow}/layer FFN, down-cols zeroed)")


if __name__ == "__main__":
    main()

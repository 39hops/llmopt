"""Joint-permutation distance closure cell (banked 2026-07-26).

git-re-basin-style: ONE hidden-unit permutation per FFN layer,
chosen jointly over gate rows + up rows + down cols (the correct
gauge subgroup with residual basis fixed), applied consistently.
KILL CONDITION (pre-reg): seed pairs (wfloor/s2/s3) still >> the
same-init ball (0.31-0.47) => weight distance closes PERMANENTLY.
Close to ~same-init => the orbifold metric revives.
"""
import itertools

import torch
from scipy.optimize import linear_sum_assignment

MODELS = {
    "wfloor": "checkpoints/mathnative_wfloor_d256.pt",
    "s2": "checkpoints/mathnative_wfloor_d256_s2.pt",
    "s3": "checkpoints/mathnative_wfloor_d256_s3.pt",
    "stream4": "checkpoints/mathnative_wfloor_d256_stream4.pt",
    "clade2": "checkpoints/mathnative_wfloor_d256_clade2.pt",
}


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return [(sd[f"blocks.{li}.gate.weight"].float(),
             sd[f"blocks.{li}.up.weight"].float(),
             sd[f"blocks.{li}.down.weight"].float())
            for li in range(8)]


def joint_dist(A, B):
    """Per-layer joint perm over [gate row | up row | down col]."""
    num = den = 0.0
    for (ga, ua, da), (gb, ub, db) in zip(A, B):
        fa = torch.cat([ga, ua, da.T], dim=1)   # (ffn, 3d)
        fb = torch.cat([gb, ub, db.T], dim=1)
        an = torch.nn.functional.normalize(fa, dim=1)
        bn = torch.nn.functional.normalize(fb, dim=1)
        r, c = linear_sum_assignment(-(an @ bn.T).numpy())
        pi = torch.empty(fa.shape[0], dtype=torch.long)
        pi[r] = torch.tensor(c)
        num += (fa - fb[pi]).norm() ** 2
        den += ((fa.norm() + fb[pi].norm()) / 2) ** 2
    return float((num / den) ** 0.5)


def raw_dist(A, B):
    num = den = 0.0
    for (ga, ua, da), (gb, ub, db) in zip(A, B):
        fa = torch.cat([ga, ua, da.T], dim=1)
        fb = torch.cat([gb, ub, db.T], dim=1)
        num += (fa - fb).norm() ** 2
        den += ((fa.norm() + fb.norm()) / 2) ** 2
    return float((num / den) ** 0.5)


ws = {k: load(p) for k, p in MODELS.items()}
for m1, m2 in itertools.combinations(MODELS, 2):
    r, j = raw_dist(ws[m1], ws[m2]), joint_dist(ws[m1], ws[m2])
    tag = ("SEED-PAIR" if {m1, m2} <= {"wfloor", "s2", "s3"}
           else "same-init")
    print(f"{m1:8s} {m2:8s} raw={r:.4f} jointperm={j:.4f}  [{tag}]",
          flush=True)
print("\nkill bar: seed pairs >> 0.31-0.47 (same-init ball) "
      "=> weight distance closes permanently")

"""RUNG 2b-ROUTER (pre-reg V4-RUNG-R + 2B-ROUTER): the retest VERDICT
V4-RUNG-2B could not do.

2b compared expert 0 against experts 1-7 BY INDEX, which is arbitrary.
If experts organise by router key, the pairs most likely to share weight
structure are router NEIGHBOURS. This runs the identical residual
pipeline (IDENT / RAND / HUNG, exact integer residuals on the shared
dyadic lattice) on the five most-similar pairs by gate-key cosine,
against five randomly drawn pairs as the matched control.

Pair selection was fixed in the pre-registration before the router was
read, so it cannot be fitted to the answer.

Env: LAYER (default 22), SHARD (default 24), NPAIR (default 5).
Usage: .venv/bin/python scratch/v4flash_rung2b_router.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
from scipy.optimize import linear_sum_assignment  # noqa: E402

import v4flash_router as RT  # noqa: E402
import v4flash_rung2b as R2  # noqa: E402
import v4flash_rungA as RA  # noqa: E402

LAYER = int(os.environ.get("LAYER", "22"))
SHARD = int(os.environ.get("SHARD", "24"))
NPAIR = int(os.environ.get("NPAIR", "5"))
OUT = "logs/opus/v4_rung2b_router.jsonl"


def pairs_from_router():
    """Top-NPAIR by raw gate-key cosine (the pre-registered rule), plus
    NPAIR random pairs as the matched control."""
    W, _, _ = RT.read_router(SHARD, LAYER)
    n = W.shape[0]
    U = W / np.linalg.norm(W, axis=1, keepdims=True)
    iu = np.triu_indices(n, 1)
    cos = (U @ U.T)[iu]
    order = np.argsort(-cos)[:NPAIR]
    near = [(int(iu[0][k]), int(iu[1][k]), float(cos[k])) for k in order]
    rng = np.random.default_rng(2026_08_02)
    rand, seen = [], {(a, b) for a, b, _ in near}
    while len(rand) < NPAIR:
        a, b = sorted(rng.choice(n, 2, replace=False).tolist())
        if (a, b) in seen:
            continue
        seen.add((a, b))
        k = np.where((iu[0] == a) & (iu[1] == b))[0][0]
        rand.append((a, b, float(cos[k])))
    return near, rand


def measure(a, b, hdr, url, base, rng):
    """Residual entropy of expert a against expert b, three alignments."""
    ea = R2.load_expert(a, hdr, url, base)
    eb = R2.load_expert(b, hdr, url, base)
    nh = eb["w1"].shape[0]
    c = R2.hidden_cost(ea, eb)
    r, col = linear_sum_assignment(c)
    arms = {"IDENT": np.arange(nh), "RAND": rng.permutation(nh),
            "HUNG": col}
    out = {"raw_bits": R2.raw_entropy(ea),
           "cost_hung_ident": float(c[r, col].sum()
                                    / c[np.arange(nh), np.arange(nh)].sum())}
    for tag, perm in arms.items():
        out[f"resid_{tag}"] = R2.resid_entropy(ea, R2.permute(eb, perm))
    return out


def main():
    os.makedirs("logs/opus", exist_ok=True)
    R2.LAYER = LAYER
    RA.SHARD = SHARD
    hdr, url, base = RA.header()
    near, rand = pairs_from_router()
    print(f"[2bR] layer {LAYER} | NEIGHBOUR pairs "
          + ", ".join(f"e{a}~e{b} {v:.3f}" for a, b, v in near))
    print(f"[2bR] layer {LAYER} | CONTROL   pairs "
          + ", ".join(f"e{a}~e{b} {v:.3f}" for a, b, v in rand), flush=True)
    rng = np.random.default_rng(7)
    groups, sink = {}, open(OUT, "a")
    for tag, plist in (("NEIGHBOUR", near), ("CONTROL", rand)):
        rows = []
        for a, b, cs in plist:
            m = measure(a, b, hdr, url, base, rng)
            m.update(group=tag, layer=LAYER, a=a, b=b, key_cos=cs)
            rows.append(m)
            sink.write(json.dumps(m) + "\n")
            sink.flush()
            print(f"[2bR] {tag:9s} e{a:3d}~e{b:3d} cos {cs:.3f} | raw "
                  f"{m['raw_bits']:.3f} IDENT {m['resid_IDENT']:.3f} "
                  f"RAND {m['resid_RAND']:.3f} HUNG {m['resid_HUNG']:.3f}",
                  flush=True)
        groups[tag] = rows
    sink.close()
    mean = lambda g, k: float(np.mean([r[k] for r in groups[g]]))
    print(f"\n[2bR] group means ({NPAIR} pairs each)")
    for g in ("NEIGHBOUR", "CONTROL"):
        print(f"[2bR]   {g:9s} key-cos {mean(g,'key_cos'):.3f} | raw "
              f"{mean(g,'raw_bits'):.4f} | IDENT {mean(g,'resid_IDENT'):.4f} "
              f"RAND {mean(g,'resid_RAND'):.4f} HUNG {mean(g,'resid_HUNG'):.4f}")
    d = mean("NEIGHBOUR", "resid_IDENT") - mean("CONTROL", "resid_IDENT")
    dh = mean("NEIGHBOUR", "resid_HUNG") - mean("CONTROL", "resid_HUNG")
    print(f"[2bR] NEIGHBOUR - CONTROL: IDENT {d:+.4f} | HUNG {dh:+.4f} "
          f"bits/param   (registered bar: routing helps if < -0.05)")
    hi = mean("NEIGHBOUR", "resid_HUNG") - mean("NEIGHBOUR", "resid_IDENT")
    print(f"[2bR] within NEIGHBOUR, HUNG - IDENT {hi:+.4f} "
          f"(bar for alignment helping: < -0.2)")


if __name__ == "__main__":
    main()

"""RUNG 2b (pre-reg V4-RUNG-2B): are DeepSeek-V4-Flash experts closer to
each other UP TO A PERMUTATION than they are coordinate-wise?

N3 killed coordinate-aligned mu+delta on Qwen3-30B with entrywise
correlation — an instrument CLAUDE.md's gauge law says is invalid for
weight comparison, since permuting hidden units leaves a function
identical while zeroing every correlation. This cell asks the question
that instrument could not.

The gauge: permuting a SwiGLU expert's hidden units (rows of w1 and w3,
columns of w2, one shared permutation) leaves the expert's function
exactly invariant, and a permutation is a bijection, so residual coding
against a permuted reference stays lossless.

Three arms, identical residual pipeline so the alphabet expansion is
common and cancels: IDENT (no permutation), RAND (a random one), HUNG
(Hungarian on a float32 proxy cost). The alignment search is a
heuristic; the residual is coded in exact integers.

Env: LAYER (default 22), SHARD (default 24), NEXP (default 8).
Usage: .venv/bin/python scratch/v4flash_rung2b.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
from scipy.optimize import linear_sum_assignment  # noqa: E402

import v4flash_rungA as RA  # noqa: E402

LAYER = int(os.environ.get("LAYER", "22"))
NEXP = int(os.environ.get("NEXP", "8"))
OUT = "logs/opus/v4_rung2b.jsonl"


def load_expert(idx, hdr, url, base):
    """Integer weights on the shared dyadic lattice, per projection.

    Every value is codes2x * 2^(exp-1); shifting by (exp - emin) puts a
    whole tensor on one integer lattice with no rounding.
    """
    RA.EXPERT = f"layers.{LAYER}.ffn.experts.{idx}"
    out = {}
    for proj in ("w1", "w2", "w3"):
        codes2x, exps = RA.decode(proj, hdr, url, base)
        emin = int(exps.min())
        sh = (exps - emin).repeat(32, axis=1)
        out[proj] = (codes2x << sh).astype(np.int64)
    return out


def hidden_cost(a, b):
    """Squared-distance cost between hidden units of two experts.

    ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x.y, so the cross term is one
    matmul. Float32 by design: this only chooses a permutation, it
    never touches the exact residual arithmetic.
    """
    cost = np.zeros((a["w1"].shape[0], b["w1"].shape[0]), np.float64)
    for proj, axis in (("w1", 0), ("w3", 0), ("w2", 1)):
        x = a[proj].astype(np.float32)
        y = b[proj].astype(np.float32)
        if axis == 1:                      # hidden units are columns
            x, y = x.T.copy(), y.T.copy()
        cost += (np.square(x).sum(1)[:, None]
                 + np.square(y).sum(1)[None, :]
                 - 2.0 * (x @ y.T)).astype(np.float64)
    return cost


def permute(ref, perm):
    """Apply a hidden-unit permutation to a reference expert."""
    return {"w1": ref["w1"][perm], "w3": ref["w3"][perm],
            "w2": ref["w2"][:, perm]}


def resid_entropy(e, ref):
    """Order-0 entropy in bits/param of the exact integer residual."""
    tot, n = 0.0, 0
    for proj in ("w1", "w2", "w3"):
        d = (e[proj] - ref[proj]).ravel()
        _, cnt = np.unique(d, return_counts=True)
        p = cnt / cnt.sum()
        tot += float(-(p * np.log2(p)).sum()) * d.size
        n += d.size
    return tot / n


def raw_entropy(e):
    tot, n = 0.0, 0
    for proj in ("w1", "w2", "w3"):
        d = e[proj].ravel()
        _, cnt = np.unique(d, return_counts=True)
        p = cnt / cnt.sum()
        tot += float(-(p * np.log2(p)).sum()) * d.size
        n += d.size
    return tot / n


def main():
    os.makedirs("logs/opus", exist_ok=True)
    hdr, url, base = RA.header()
    print(f"[2b] layer {LAYER} | reference expert 0 | "
          f"{NEXP - 1} comparisons", flush=True)
    ref = load_expert(0, hdr, url, base)
    rng = np.random.default_rng(2026_08_02)
    nh = ref["w1"].shape[0]

    # (4) free necessary condition: do experts even share a sorted
    # hidden-unit norm profile? If not, no permutation could align them.
    prof = {}
    rows, sink = [], open(OUT, "a")
    for idx in range(NEXP):
        e = load_expert(idx, hdr, url, base) if idx else ref
        nrm = np.sqrt(np.square(e["w1"].astype(np.float64)).sum(1))
        prof[idx] = np.sort(nrm)
        if idx == 0:
            continue
        c = hidden_cost(e, ref)
        r, col = linear_sum_assignment(c)
        hung = col
        rand = rng.permutation(nh)
        arms = {"IDENT": np.arange(nh), "RAND": rand, "HUNG": hung}
        row = {"layer": LAYER, "expert": idx,
               "raw_bits": raw_entropy(e),
               "cost_ident": float(c[np.arange(nh), np.arange(nh)].sum()),
               "cost_rand": float(c[np.arange(nh), rand].sum()),
               "cost_hung": float(c[r, col].sum())}
        for tag, perm in arms.items():
            row[f"resid_{tag}"] = resid_entropy(e, permute(ref, perm))
        # sorted-profile agreement against the reference
        row["prof_rel_l2"] = float(
            np.linalg.norm(prof[idx] - prof[0]) / np.linalg.norm(prof[0]))
        rows.append(row)
        sink.write(json.dumps(row) + "\n")
        sink.flush()
        print(f"[2b] e{idx}: raw {row['raw_bits']:.3f} | resid "
              f"IDENT {row['resid_IDENT']:.3f} RAND {row['resid_RAND']:.3f} "
              f"HUNG {row['resid_HUNG']:.3f} | cost hung/ident "
              f"{row['cost_hung'] / row['cost_ident']:.4f} | "
              f"prof dL2 {row['prof_rel_l2']:.4f}", flush=True)
    sink.close()
    m = lambda k: float(np.mean([r[k] for r in rows]))
    print(f"\n[2b] means over {len(rows)} pairs")
    print(f"[2b] raw code entropy (integer lattice) {m('raw_bits'):.4f}")
    print(f"[2b] residual IDENT {m('resid_IDENT'):.4f} | "
          f"RAND {m('resid_RAND'):.4f} | HUNG {m('resid_HUNG'):.4f}")
    print(f"[2b] HUNG - IDENT = {m('resid_HUNG') - m('resid_IDENT'):+.4f} "
          f"bits/param   (registered bar: helps if < -0.2)")
    print(f"[2b] HUNG - RAND  = {m('resid_HUNG') - m('resid_RAND'):+.4f}")
    print(f"[2b] cost hung/ident {m('cost_hung') / m('cost_ident'):.4f} | "
          f"rand/ident {m('cost_rand') / m('cost_ident'):.4f}")
    print(f"[2b] sorted-profile relative L2 vs reference "
          f"{m('prof_rel_l2'):.4f}")


if __name__ == "__main__":
    main()

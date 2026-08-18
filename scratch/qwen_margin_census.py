"""Teacher margin-bin census (AMENDMENT QWEN-MODEL1-TREE-PINS
item 2) — teacher-only per-bin position counts, booked once with
the teacher lock, before any arm scores.

Margin m = (top1 - top2) logit gap from the LOCKED fp16 records
after fp32 upcast, at every scored position: all corpus positions,
all prefix positions, and rollout live positions (t <
gen_lengths[b]). Frozen edges [0, .02, .05, .1, .2, .5, 1, 2, 5,
inf); bins with fewer than 30 positions carry no directional claim
(the fence threshold is recorded in the receipt).

Every record's sha256 is verified against the manifest AND the
verified digests are recorded in the receipt (an unrecorded
verification is unfalsifiable — receipt-audit B2, 2026-08-17).
Refuses smoke manifests and refuses to overwrite.

    .venv/bin/python scratch/qwen_margin_census.py
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

TD = "logs/qwenteacher_v2"
OUT = os.path.join(TD, "margin_census.json")
EDGES = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")]
SMALL_N = 30


def margins(a):
    p = np.partition(a.astype(np.float32), -2, axis=-1)
    return p[..., -1] - p[..., -2]


def bins(m):
    h, _ = np.histogram(m, EDGES)
    return h.astype(int).tolist()


def main() -> int:
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    man = json.load(open(os.path.join(TD, "teacher_manifest.json")))
    if man["smoke"] is not False:
        raise SystemExit("REFUSING: smoke manifest")
    shas = {}

    def load_verified(fname, rec_key):
        a = np.load(os.path.join(TD, fname))
        h = hashlib.sha256(a.tobytes()).hexdigest()
        want = man["records"][rec_key]["sha256"]
        if h != want:
            raise SystemExit(f"REFUSING: {fname} sha {h[:12]} != "
                             f"manifest {want[:12]}")
        shas[fname] = h
        return a

    cl = load_verified("corpus_logits.npy", "corpus")
    pl = load_verified("prefix_logits.npy", "prefixes")
    rl = load_verified("rollout_logits.npy", "rollouts")
    gl = man["records"]["rollouts"]["gen_lengths"]

    mc, mp = margins(cl), margins(pl)
    mr = np.concatenate([margins(rl[:gl[b], b])
                         for b in range(len(gl))])
    allm = np.concatenate([mc, mp, mr])
    total = bins(allm)
    rec = {
        "gate": "teacher margin-bin census (TREE-PINS item 2)",
        "edges": [e if np.isfinite(e) else "inf" for e in EDGES],
        "counts": {"corpus": bins(mc), "prefixes": bins(mp),
                   "rollouts_live": bins(mr), "total": total},
        "n_positions": {"corpus": int(mc.size),
                        "prefixes": int(mp.size),
                        "rollouts_live": int(mr.size),
                        "total": int(allm.size)},
        "small_n_fence": SMALL_N,
        "small_n_bins_total": [i for i, c in enumerate(total)
                               if c < SMALL_N],
        "verified_record_sha256": shas,
        "teacher_code_commit": man["code_commit"],
        "revision": man["revision"],
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "--short",
             "HEAD"]).decode().strip(),
        "tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"]).strip()),
    }
    with open(OUT, "x") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps({k: rec[k] for k in
                      ("counts", "n_positions", "small_n_bins_total",
                       "code_commit", "tree_dirty")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

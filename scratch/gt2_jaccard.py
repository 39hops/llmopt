"""MOE-GT-2 coalition Jaccard analysis (D2/D3 readouts, committed
post-hoc so the booked numbers are re-derivable — they were desk
computations in-session on 2026-08-04).

Reads phase-tagged TRAJ v2 files, builds DECODE-ONLY gate-prompts-only
keep-sets at 45.3% (arm2's top-demand rule), and prints:
  - pairwise coalition Jaccard (math/phys/code), mean + min over layers
  - within-domain split-half nulls (even vs odd prompt ids)
  - coverage: fraction of each keep-set inside the other
  - count-weighted cross-coverage of demand (the D4-CROSS desk numbers)

Booked numbers this reproduces: VERDICT MOE-GT-2-D2 (0.767 full-data*,
0.930/0.871 nulls, 0.8671 coverage), D3 (0.804/0.543/0.539, 0.653 code
null), D4-CROSS (0.7761/0.8023 cross-coverage).
*D2's 0.767 included the probe prompt's decode rows (pred=all); the
 gate-prompts-only filter gives 0.804 — both filters are exposed here
 (GATE_ONLY env), and the D3 entry names the distinction.

Usage: .venv/bin/python scratch/gt2_jaccard.py   [FRAC=0.453,
       GATE_ONLY=1 env overrides]
"""

import json
import os
from collections import defaultdict

FRAC = float(os.environ.get("FRAC", "0.453"))
GATE_ONLY = os.environ.get("GATE_ONLY", "1") == "1"
TRAJ = {
    "math": "logs/opus/moe_gt1_traj_v2.jsonl",
    "phys": "logs/opus/gt2_phys_traj.jsonl",
    "code": "logs/opus/gt2_code_traj.jsonl",
}


DROP_TAIL = os.environ.get("DROP_TAIL", "1") == "1"


def decode_counts(path, pred=lambda r: True):
    """DROP_TAIL=1 (default) drops the FIRST decode-phase row per
    (prompt, layer): mlx_lm's prefill leaves the last prompt token to
    the first 1-token step, so that row is the chat-template tail
    mislabeled as decode (reviewer bug 2026-08-04; TRAJ v3 records it
    as phase=prompt_tail). DROP_TAIL=0 reproduces the originally
    booked D2/D3 numbers."""
    c = defaultdict(lambda: defaultdict(int))
    first_seen = set()
    for line in open(path):
        r = json.loads(line)
        if r["phase"] != "decode":
            continue
        if DROP_TAIL and isinstance(r["prompt"], int):
            key = (r["prompt"], r["layer"])
            if key not in first_seen:
                first_seen.add(key)
                continue
        if GATE_ONLY and not isinstance(r["prompt"], int):
            continue
        if not pred(r):
            continue
        for e in r["topk"]:
            c[r["layer"]][e] += 1
    return c


def keep(counts, n=128, top_k=8):
    out = {}
    for li, row in counts.items():
        full = [row.get(e, 0) for e in range(n)]
        k = max(top_k, round(FRAC * n))
        out[li] = set(sorted(range(n), key=lambda e: -full[e])[:k])
    return out


def jmean(ka, kb):
    js = [len(ka[li] & kb[li]) / len(ka[li] | kb[li])
          for li in ka if li in kb]
    return sum(js) / len(js), min(js)


def coverage(demand, kp):
    """Count-weighted fraction of `demand` routed inside keep-set kp."""
    hit = tot = 0
    for li, row in demand.items():
        ks = kp[li]
        for e, c in row.items():
            tot += c
            if e in ks:
                hit += c
    return hit / tot


def main():
    counts = {d: decode_counts(p) for d, p in TRAJ.items()}
    keeps = {d: keep(c) for d, c in counts.items()}
    doms = list(TRAJ)
    print(f"frac {FRAC} | gate_only {GATE_ONLY}")
    for i, a in enumerate(doms):
        for b in doms[i + 1:]:
            m, lo = jmean(keeps[a], keeps[b])
            print(f"Jaccard({a},{b}): mean {m:.4f} min {lo:.4f}")
    for d, p in TRAJ.items():
        half = lambda par: (lambda r: isinstance(r["prompt"], int)
                            and r["prompt"] % 2 == par)
        ke = keep(decode_counts(p, half(0)))
        ko = keep(decode_counts(p, half(1)))
        m, lo = jmean(ke, ko)
        print(f"{d} split-half null: mean {m:.4f} min {lo:.4f}")
    for a in doms:
        for b in doms:
            if a != b:
                cov = [len(keeps[a][li] & keeps[b][li]) / len(keeps[a][li])
                       for li in keeps[a] if li in keeps[b]]
                print(f"{a} keep-set covered by {b} keep-set: "
                      f"mean {sum(cov) / len(cov):.4f}")
    for a in doms:
        for b in doms:
            if a != b:
                print(f"{a} DEMAND covered by {b} keep-set: "
                      f"{coverage(counts[a], keeps[b]):.4f}")


if __name__ == "__main__":
    main()

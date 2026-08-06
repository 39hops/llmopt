"""MOE-GT-2 coalition Jaccard analysis (D2/D3 readouts, committed
post-hoc so the booked numbers are re-derivable — they were desk
computations in-session on 2026-08-04).

Reads phase-tagged TRAJ v2 files, builds DECODE-ONLY gate-prompts-only
keep-sets at 45.3% (arm2's top-demand rule), and prints:
  - pairwise coalition Jaccard (math/phys/code), mean + min over layers
  - within-domain split-half nulls (even vs odd prompt ids)
  - coverage: fraction of each keep-set inside the other
  - count-weighted cross-coverage of demand (the D4-CROSS desk numbers)

Default output (DROP_TAIL=1 GATE_ONLY=1) is the CORRECTED set booked
by AMENDMENT GT2-REVIEW-2: Jaccard 0.8013/0.5331/0.5280, nulls
0.9205/0.8670/0.6364, plus the GT2-CORE-0 core statistic (37.1/58,
containment 0.92). DROP_TAIL=0 reproduces the ORIGINAL D2/D3 booking
(0.804/0.543/0.539, code null 0.653); GATE_ONLY=0 additionally
restores probe rows (D2's headline 0.767 and 0.8671 coverage).
DUMP_DECODE=1 writes checkpoints/gt2_{dom}_arm0_decode.json; use
DUMP_DECODE=1 DROP_TAIL=0 to reproduce BYTE-IDENTICALLY the demand
logs the D4/PHYS-B/cross arms actually consumed (they were built
before the prompt-tail correction; verified identical 2026-08-04).

Usage: .venv/bin/python scratch/gt2_jaccard.py   [FRAC=0.453,
       GATE_ONLY=1 env overrides]
"""

import json
import os
from collections import defaultdict

TRAJ_DEFAULTS = {
    "math": "logs/opus/moe_gt1_traj_v2.jsonl",
    "phys": "logs/opus/gt2_phys_traj.jsonl",
    "code": "logs/opus/gt2_code_traj.jsonl",
}


def _traj():
    """TRAJ log paths, env-overridable per domain (TRAJ_MATH etc.)."""
    return {d: os.environ.get(f"TRAJ_{d.upper()}", p)
            for d, p in TRAJ_DEFAULTS.items()}


# Env is resolved at CALL time, not import time (lab spec F2: a
# consumer importing this module and setting FRAC afterwards used to
# get the default silently). None = "read the env now".
def _frac(frac=None):
    return float(os.environ.get("FRAC", "0.453")) if frac is None else frac


def _flag(name, default, value=None):
    if value is not None:
        return value
    return os.environ.get(name, default) == "1"


def decode_counts(path, pred=lambda r: True,
                  gate_only=None, drop_tail=None):
    """DROP_TAIL=1 (default) drops the FIRST decode-phase row per
    (prompt, layer): mlx_lm's prefill leaves the last prompt token to
    the first 1-token step, so that row is the chat-template tail
    mislabeled as decode (reviewer bug 2026-08-04; TRAJ v3 records it
    as phase=prompt_tail). DROP_TAIL=0 reproduces the originally
    booked D2/D3 numbers."""
    gate_only = _flag("GATE_ONLY", "1", gate_only)
    drop_tail = _flag("DROP_TAIL", "1", drop_tail)
    c = defaultdict(lambda: defaultdict(int))
    first_seen = set()
    for line in open(path):
        r = json.loads(line)
        if r["phase"] != "decode":
            continue
        if drop_tail and isinstance(r["prompt"], int):
            key = (r["prompt"], r["layer"])
            if key not in first_seen:
                first_seen.add(key)
                continue
        if gate_only and not isinstance(r["prompt"], int):
            continue
        if not pred(r):
            continue
        for e in r["topk"]:
            c[r["layer"]][e] += 1
    return c


def keep(counts, n=128, top_k=8, frac=None):
    frac = _frac(frac)
    out = {}
    for li, row in counts.items():
        full = [row.get(e, 0) for e in range(n)]
        k = max(top_k, round(frac * n))
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
    traj = _traj()
    counts = {d: decode_counts(p) for d, p in traj.items()}
    if os.environ.get("DUMP_DECODE") == "1":
        for d, c in counts.items():
            out = f"checkpoints/gt2_{d}_arm0_decode.json"
            json.dump({"counts": {str(li): [row.get(e, 0) for e in range(128)]
                                  for li, row in sorted(c.items())},
                       "source": f"{traj[d]} decode-only gate-prompts-only"},
                      open(out, "w"))
            print(f"wrote {out}")
    keeps = {d: keep(c) for d, c in counts.items()}
    doms = list(traj)
    print(f"frac {_frac()} | gate_only {_flag('GATE_ONLY', '1')}")
    for i, a in enumerate(doms):
        for b in doms[i + 1:]:
            m, lo = jmean(keeps[a], keeps[b])
            print(f"Jaccard({a},{b}): mean {m:.4f} min {lo:.4f}")
    for d, p in traj.items():
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
    # GT2-CORE-0: the three-domain core + containment (class hierarchy)
    core = {li: keeps["math"][li] & keeps["phys"][li] & keeps["code"][li]
            for li in keeps["math"]}
    sizes = [len(v) for v in core.values()]
    k = len(next(iter(keeps["math"].values())))
    print(f"three-domain CORE: mean {sum(sizes)/len(sizes):.1f}/{k} "
          f"per layer (min {min(sizes)} max {max(sizes)}; "
          f"independence null {k*(k/128)**2:.1f})")
    mc_in_p = sum(
        len((keeps["math"][li] & keeps["code"][li]) & keeps["phys"][li])
        / max(len(keeps["math"][li] & keeps["code"][li]), 1)
        for li in core) / len(core)
    print(f"containment: fraction of (math&code) also in phys "
          f"{mc_in_p:.2f} (1.0 = clean class hierarchy)")


if __name__ == "__main__":
    main()

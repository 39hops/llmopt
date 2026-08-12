"""lab.keepsets — keep-set / coalition algebra. CANONICAL BODY since
2026-08-12 (Phase 3 module 1); scratch/gt2_jaccard.py is a re-export
shim over these symbols and keeps only its CLI. Originally adopted
verbatim from that file 2026-08-06. Guarded by
tests/test_lab_keepsets.py (shim identity + synthetic battery +
full acceptance against the booked stats and the byte-frozen
checkpoints/gt2_*_arm0_decode.json dumps).

REGENERATION-SENSITIVE (the extraction spec's warning): the DROP_TAIL
first-row rule, GATE_ONLY filter, and the stable-sort tie-break at
the keep boundary must reproduce byte-identically — booked Jaccards
(0.8013/0.5331/0.5280; nulls 0.9205/0.8670/0.6364) depend on them.
Env is resolved at CALL time (lab spec F2).
"""
from __future__ import annotations

import os
from collections import defaultdict

import json


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

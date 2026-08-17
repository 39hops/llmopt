"""WHOLE-0T artifact qualification as a LIBRARY, not a workflow.

Consumers (runtime reference, cache sidecar, MODEL-1 scorer) call
load_manifest_verified() and cannot see a manifest before
qualification returns — "all consumers qualify first" holds in
code. scratch/qwen_qualify.py is a thin CLI over this module.

Rungs (failure-localizing, cheap first):
  0  identity: regenerate every file sha v the committed digest
     chain (fail-closed: no chain -> refuse unless the caller
     passes allow_unchained=True, which lands in the report)
  1a exact key conservation v the sha-pinned vendor index
     (duplicate-refusing parse on BOTH json documents)
  1b structure: codec census, exact payload lengths, exact span
     cover (no gaps, no trailing bytes)
  3  smallest-tensor decode spot check per codec
  preflight  estimated peak residency v available memory,
     refuse above SAFETY_FRACTION
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess

import numpy as np

from llmopt.lab.qcodec import decode_entry, expected_len

VENDOR_INDEX_SHA = ("77042094076611b69791a610065f28b7013b8c62"
                    "1795fa86ddccc8bac7d1b9df")
SAFETY_FRACTION = 0.8
CHAIN_ROWS = 19                     # manifest + 18 shards


class QualifyError(SystemExit):
    pass


def _no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise QualifyError(f"duplicate JSON key {k!r}")
        d[k] = v
    return d


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def verify_chain(art_dir: str, chain_path: str | None,
                 allow_unchained: bool = False) -> dict:
    """Rung 0. Returns {'checked': n} or raises."""
    if chain_path is None or not os.path.exists(chain_path):
        if allow_unchained:
            return {"checked": 0, "unchained": True}
        raise QualifyError(
            f"rung0: no digest chain for {art_dir} — identity "
            f"unprovable; pass allow_unchained=True to override "
            f"(the override lands in the report)")
    rows = [l.split() for l in open(chain_path) if l.strip()]
    if len(rows) != CHAIN_ROWS:
        raise QualifyError(f"rung0: chain has {len(rows)} rows, "
                           f"expected {CHAIN_ROWS}")
    for sha, fname in rows:
        p = os.path.join(art_dir, fname)
        got = _sha(p) if os.path.exists(p) else "ABSENT"
        if got != sha:
            raise QualifyError(f"rung0: {fname} sha {got[:12]} != "
                               f"chain {sha[:12]}")
    return {"checked": len(rows), "unchained": False}


def load_index(vendor_index: str) -> set:
    ib = open(vendor_index, "rb").read()
    got = hashlib.sha256(ib).hexdigest()
    if got != VENDOR_INDEX_SHA:
        raise QualifyError(f"vendor index sha {got[:12]} != pinned "
                           f"{VENDOR_INDEX_SHA[:12]}")
    return set(json.loads(ib.decode(),
                          object_pairs_hook=_no_dup_pairs)["weight_map"])


def estimate_runtime_peak(man: dict) -> int:
    """Shared cost model for the CPU reference runtime: compressed
    io resident + one fp32 layer x2 (decode temp + param) + 2 GiB
    overhead. Both the preflight gate and the runtime receipt use
    THIS function; a divergence between estimate and observation is
    the gate being uncalibrated, and both numbers land in one
    receipt so it shows."""
    io_keys = ("model.language_model.embed_tokens.weight",
               "lm_head.weight")
    io_payload = sum(man[k]["len"] for k in io_keys if k in man
                     and man[k]["codec"] != "excluded")
    biggest = 0
    per_layer = {}
    for k, e in man.items():
        if ".layers." in k and e["codec"] != "excluded":
            li = k.split(".layers.")[1].split(".")[0]
            per_layer[li] = per_layer.get(li, 0) \
                + int(np.prod(e["shape"]))
    if per_layer:
        biggest = max(per_layer.values())
    return io_payload + biggest * 4 * 2 + 2 * 2 ** 30


def available_memory() -> int | None:
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                return int(line.split()[1]) * 1024
    except OSError:
        try:
            out = subprocess.check_output(["vm_stat"]).decode()
            m = re.search(r"page size of (\d+)", out)
            page = int(m.group(1)) if m else 16384
            n = sum(int(l.split()[-1].rstrip("."))
                    for l in out.splitlines()
                    if l.startswith(("Pages free", "Pages inactive")))
            return n * page
        except Exception:
            return None
    return None


def resource_preflight(man: dict) -> dict:
    est = estimate_runtime_peak(man)
    avail = available_memory()
    if avail is None:
        raise QualifyError("preflight: available memory undiscoverable")
    if est > SAFETY_FRACTION * avail:
        raise QualifyError(
            f"preflight REFUSE: est {est/2**30:.2f} GiB > "
            f"{SAFETY_FRACTION} x available {avail/2**30:.2f} GiB")
    return {"estimated_peak_bytes": est, "available_bytes": avail,
            "safety_fraction": SAFETY_FRACTION}


def qualify_artifact(art_dir: str, vendor_index: str,
                     chain_path: str | None = None,
                     allow_unchained: bool = False,
                     preflight: bool = True) -> dict:
    """Full ladder; returns the manifest + a report dict. The ONLY
    sanctioned way for a consumer to obtain a manifest."""
    art_dir = os.path.expanduser(art_dir)
    report = {"art_dir": art_dir}
    report["rung0"] = verify_chain(art_dir, chain_path, allow_unchained)

    man = json.loads(open(os.path.join(art_dir, "manifest.json"),
                          "rb").read().decode(),
                     object_pairs_hook=_no_dup_pairs)

    expected = load_index(os.path.expanduser(vendor_index))
    missing = sorted(expected - set(man))
    extra = sorted(set(man) - expected)
    if missing or extra:
        raise QualifyError(f"conservation: missing {missing[:3]} "
                           f"extra {extra[:3]}")
    report["conservation"] = {"keys": len(man)}

    by_shard = {}
    counts = {"w4": 0, "s16": 0, "raw": 0, "excluded": 0}
    for name, e in man.items():
        codec = e["codec"]
        if codec not in counts:
            raise QualifyError(f"unknown codec {codec} on {name}")
        counts[codec] += 1
        if codec == "excluded":
            continue
        if e["len"] != expected_len(codec, e["shape"]):
            raise QualifyError(f"{name}: payload length wrong")
        by_shard.setdefault(e["shard"], []).append(
            (e["off"], e["off"] + e["len"], name))
    for sh, spans in by_shard.items():
        spans.sort()
        fsize = os.path.getsize(os.path.join(art_dir, sh + ".bin"))
        prev = 0
        for off, end, name in spans:
            if off != prev:
                raise QualifyError(f"{sh}: gap/overlap at {name}")
            prev = end
        if prev != fsize:
            raise QualifyError(f"{sh}: trailing bytes")
    report["census"] = counts

    for codec in ("w4", "s16", "raw"):
        cands = [(int(np.prod(e["shape"])), n)
                 for n, e in man.items() if e["codec"] == codec]
        if not cands:
            continue
        _, name = min(cands)
        e = man[name]
        with open(os.path.join(art_dir, e["shard"] + ".bin"), "rb") as f:
            f.seek(e["off"])
            W = decode_entry(f.read(e["len"]), e)
        if not np.isfinite(W).all():
            raise QualifyError(f"{name}: non-finite decode")
        mag = float(np.abs(W).max())
        if not (1e-6 < mag < 1e3):
            raise QualifyError(f"{name}: implausible magnitude {mag}")

    if preflight:
        report["preflight"] = resource_preflight(man)
    return {"manifest": man, "report": report}


def load_manifest_verified(art_dir: str, vendor_index: str,
                           chain_path: str | None = None,
                           **kw) -> dict:
    return qualify_artifact(art_dir, vendor_index, chain_path, **kw)

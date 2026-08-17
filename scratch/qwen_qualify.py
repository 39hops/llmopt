"""Artifact qualification ladder, rungs 1-3 (static, seconds each).

Runs BEFORE any model-scale execution touches an artifact
(engineering law 2026-08-17: no expensive run is the first test of
new code). The runtime reference, cache sidecar, and MODEL-1
scorer all call this first.

  rung 1  static artifact checks: manifest parses, key census,
          codec names, exact payload-length formulas, offsets
          non-overlapping and within file bounds, conservation
          (every text-tower key present exactly once)
  rung 2  golden codec fixtures: delegated to
          tests/test_qwen_codec.py (run it via pytest)
  rung 3  real-tensor decode spot-check: decode the SMALLEST
          coded tensor per codec through llmopt.lab.qcodec and
          assert finite values + plausible magnitude
  preflight  estimated peak residency for a given runtime plan
          (resident io dtype + one fp32 layer) v host RAM

    ART_DIR=~/qwen_whole0t/A python scratch/qwen_qualify.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, ".")
from llmopt.lab.qcodec import decode_entry, expected_len  # noqa: E402

ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/A"))


VENDOR_INDEX = os.path.expanduser(os.environ.get(
    "VENDOR_INDEX", "~/qwen_vendor/model.safetensors.index.json"))


def _no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise SystemExit(f"QUALIFY FAILED: duplicate manifest key {k}")
        d[k] = v
    return d


def main():
    fails = []
    # rung 0 — IDENTITY: regenerate the digest chain from the files
    # on disk and compare against the committed chain (transfer
    # rule: a corruption preserving offsets/lengths/finiteness
    # passes every later rung; this one catches it)
    import hashlib
    arm = os.path.basename(ART.rstrip("/"))
    chain_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "logs", "qwenwhole",
                              f"artifact_digest_{arm}.txt")
    if os.path.exists(chain_path):
        want = {l.split()[1]: l.split()[0]
                for l in open(chain_path) if l.strip()}
        for fname, sha in sorted(want.items()):
            p = os.path.join(ART, fname)
            got = hashlib.sha256(open(p, "rb").read()).hexdigest() \
                if os.path.exists(p) else "ABSENT"
            if got != sha:
                fails.append(f"digest mismatch {fname}: {got[:12]} "
                             f"!= {sha[:12]}")
        print(f"[q] rung0 identity: {len(want)} files v committed "
              f"chain, {len(fails)} mismatches")
        if fails:
            for f_ in fails[:5]:
                print(f"[q] FAIL {f_}")
            raise SystemExit("QUALIFY FAILED at rung 0 (identity)")
    else:
        print(f"[q] rung0 SKIP: no committed chain for arm {arm!r} "
              f"(non-A/B/C artifact)")

    man = json.load(open(os.path.join(ART, "manifest.json")),
                    object_pairs_hook=_no_dup_pairs)
    print(f"[q] {ART}: {len(man)} keys")

    # rung 1a — EXACT key conservation v the pinned vendor index
    # (same duplicate-refusing parse; sha pinned to rev 1d4bf0f2)
    VENDOR_INDEX_SHA = ("77042094076611b69791a610065f28b7013b8c62"
                        "1795fa86ddccc8bac7d1b9df")
    ib = open(VENDOR_INDEX, "rb").read()
    got_sha = __import__("hashlib").sha256(ib).hexdigest()
    if got_sha != VENDOR_INDEX_SHA:
        fails.append(f"vendor index sha {got_sha[:12]} != pinned "
                     f"{VENDOR_INDEX_SHA[:12]}")
    expected = set(json.loads(ib.decode(),
                              object_pairs_hook=_no_dup_pairs)
                   ["weight_map"])
    got = set(man)
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing:
        fails.append(f"missing {len(missing)} keys: {missing[:3]}")
    if extra:
        fails.append(f"extra {len(extra)} keys: {extra[:3]}")
    print(f"[q] rung1a conservation: expected {len(expected)} = "
          f"manifest {len(got)}, missing {len(missing)}, "
          f"extra {len(extra)}")

    # rung 1b — static structure
    by_shard = {}
    counts = {"w4": 0, "s16": 0, "raw": 0, "excluded": 0}
    for name, e in man.items():
        codec = e["codec"]
        if codec not in counts:
            fails.append(f"unknown codec {codec} on {name}")
            continue
        counts[codec] += 1
        if codec == "excluded":
            continue
        exp = expected_len(codec, e["shape"])
        if e["len"] != exp:
            fails.append(f"{name}: len {e['len']} != expected {exp}")
        by_shard.setdefault(e["shard"], []).append(
            (e["off"], e["off"] + e["len"], name))
    for sh, spans in by_shard.items():
        spans.sort()
        fsize = os.path.getsize(os.path.join(ART, sh + ".bin"))
        prev_end = 0
        for off, end, name in spans:
            if off != prev_end:               # exact cover: no gaps,
                fails.append(f"{sh}: {name} at {off}, expected "
                             f"{prev_end} (gap or overlap)")
            prev_end = end
        if prev_end != fsize:                 # no trailing bytes
            fails.append(f"{sh}: spans end {prev_end} != file "
                         f"size {fsize}")
    print(f"[q] rung1 census {counts}, {len(by_shard)} shards")

    # rung 3 — smallest coded tensor per codec, decoded for real
    for codec in ("w4", "s16", "raw"):
        cands = [(np.prod(e["shape"]), n) for n, e in man.items()
                 if e["codec"] == codec]
        if not cands:
            continue
        _, name = min(cands)
        e = man[name]
        with open(os.path.join(ART, e["shard"] + ".bin"), "rb") as f:
            f.seek(e["off"])
            buf = f.read(e["len"])
        W = decode_entry(buf, e)
        if not np.isfinite(W).all():
            fails.append(f"{name}: non-finite decode")
        mag = float(np.abs(W).max())
        if not (1e-6 < mag < 1e3):
            fails.append(f"{name}: implausible magnitude {mag}")
        print(f"[q] rung3 {codec}: {name} max|W|={mag:.4f}")

    # preflight — resident io fp16 + one fp32 layer + overhead
    n_io = sum(int(np.prod(e["shape"])) for k, e in man.items()
               if k in ("model.language_model.embed_tokens.weight",
                        "lm_head.weight"))
    biggest_layer = 0
    layer_tot = {}
    for k, e in man.items():
        if ".layers." in k and e["codec"] != "excluded":
            li = k.split(".layers.")[1].split(".")[0]
            layer_tot[li] = layer_tot.get(li, 0) \
                + int(np.prod(e["shape"]))
    if layer_tot:
        biggest_layer = max(layer_tot.values())
    est = n_io * 2 + biggest_layer * 4 * 2 + 2 * 2 ** 30
    avail = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                avail = int(line.split()[1]) * 1024
    except OSError:                        # macOS
        import subprocess
        try:
            free = subprocess.check_output(["vm_stat"]).decode()
            import re as _re
            m = _re.search(r"page size of (\d+)", free)
            page = int(m.group(1)) if m else 16384
            n = sum(int(l.split()[-1].rstrip("."))
                    for l in free.splitlines()
                    if l.startswith(("Pages free", "Pages inactive")))
            avail = n * page
        except Exception:
            pass
    frac = est / avail if avail else float("nan")
    print(f"[q] preflight: est peak {est/2**30:.2f} GiB v available "
          f"{(avail or 0)/2**30:.2f} GiB (fraction {frac:.2f}, "
          f"limit 0.80)")
    if avail is None:
        fails.append("preflight: available memory undiscoverable")
    elif est > 0.8 * avail:
        fails.append(f"preflight REFUSE: est {est/2**30:.2f} GiB > "
                     f"0.8 x available {avail/2**30:.2f} GiB")

    if fails:
        for f_ in fails[:10]:
            print(f"[q] FAIL {f_}")
        raise SystemExit(f"QUALIFY FAILED: {len(fails)} findings")
    print("[q] PASS")


if __name__ == "__main__":
    main()

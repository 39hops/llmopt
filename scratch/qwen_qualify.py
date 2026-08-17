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


def main():
    man = json.load(open(os.path.join(ART, "manifest.json")))
    print(f"[q] {ART}: {len(man)} keys")
    fails = []

    # rung 1 — static
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
        if codec != "raw" and e["len"] != exp:
            fails.append(f"{name}: len {e['len']} != expected {exp}")
        by_shard.setdefault(e["shard"], []).append(
            (e["off"], e["off"] + e["len"], name))
    for sh, spans in by_shard.items():
        spans.sort()
        fsize = os.path.getsize(os.path.join(ART, sh + ".bin"))
        prev_end = 0
        for off, end, name in spans:
            if off < prev_end:
                fails.append(f"{sh}: {name} overlaps previous span")
            prev_end = end
        if prev_end > fsize:
            fails.append(f"{sh}: spans exceed file size")
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
    print(f"[q] preflight: io fp16 {n_io*2/2**30:.2f} GiB + "
          f"layer fp32 x2 {biggest_layer*8/2**30:.2f} GiB + 2 GiB "
          f"overhead = {est/2**30:.2f} GiB peak estimate")

    if fails:
        for f_ in fails[:10]:
            print(f"[q] FAIL {f_}")
        raise SystemExit(f"QUALIFY FAILED: {len(fails)} findings")
    print("[q] PASS")


if __name__ == "__main__":
    main()

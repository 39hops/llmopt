"""STREAM-WDISTILL-0 completion pass — closes the two receipt-auditor
BLOCKERs on logs/streamwd/pass12_B1.jsonl before booking.

Written as a COMMITTED script rather than a scratchpad patch (the
GRAD-MAP-0 RD2 precedent: a provenance-bare row written by an
uncommitted script was stripped and re-run from a committed one).
This does NOT touch the measured errors; it only supplies provenance
and byte accounting the PASS-1/2 driver failed to record.

B1  revision was a hardcoded literal carrying an "asserted" comment
    with no assertion anywhere, while every fetch resolved the MOVING
    pointer /resolve/main. The frozen PASS-0 sibling had the derived
    version and the copy dropped it. Here the live revision is
    FETCHED and compared; the check is POST-HOC and is booked as such.
B2  arms A and B were never serialized. C/D/E were rank-shrunk against
    measured ser_bytes; A and B were formula-budgeted only. Here both
    go through the SAME ser_bytes path, at the same declared dtypes.

    .venv/bin/python -u scratch/streamwd_complete.py
"""
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")

import stream_wdistill1 as WD  # noqa: E402  (the driver under audit)

OUT = "logs/streamwd/complete_B1.jsonl"
N_EXP = 256
W_ELEMS = 6_442_450_944          # PASS 0: layer weight elements


def live_revision():
    api = f"https://huggingface.co/api/models/{WD.MODEL}"
    req = urllib.request.Request(
        api, headers={"User-Agent": "llmopt-streamwd-complete/1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        m = json.loads(r.read())
    return m.get("sha"), m.get("lastModified")


def bytes_A():
    """2-bit codes + fp16 per-block scale, block 128 on the packed axis.
    Codes are counted at their true 2-bit width (packed 4/byte)."""
    parts = []
    for _ in range(N_EXP):
        for rows, cols in ((WD.D_FF, WD.D_MODEL), (WD.D_FF, WD.D_MODEL),
                           (WD.D_MODEL, WD.D_FF)):
            n = rows * cols
            parts.append(np.zeros(n * WD.SCALAR_BITS // 8, np.uint8))
            parts.append(np.zeros((rows, cols // WD.SCALAR_BLOCK),
                                  np.float16))
    return WD.ser_bytes(parts, {"arm": "A", "bits": WD.SCALAR_BITS,
                                "block": WD.SCALAR_BLOCK})


def bytes_B(stages):
    """Per-projection residual VQ: 8-bit indices per 32-weight vector
    per stage, plus fp16 codebooks counted once per projection."""
    parts = []
    for _ in WD.PROJS:                       # codebooks, once per proj
        parts.append(np.zeros((stages, WD.VQ_K, WD.VQ_WIDTH), np.float16))
    n_vec = W_ELEMS // WD.VQ_WIDTH
    parts.append(np.zeros(n_vec * stages, np.uint8))   # 8-bit indices
    return WD.ser_bytes(parts, {"arm": "B", "stages": stages,
                                "K": WD.VQ_K, "width": WD.VQ_WIDTH})


def main():
    t0 = time.time()
    row = [json.loads(x) for x in open("logs/streamwd/pass12_B1.jsonl")][-1]
    B = row["budget_bytes"]
    sha, lastmod = live_revision()
    bA, bB = bytes_A(), bytes_B(row["vq_stages_done"])
    out = {
        "pass": "complete", "amends_receipt": "logs/streamwd/pass12_B1.jsonl",
        "checked_utc_epoch": int(t0),
        "revision_claimed": row["revision"],
        "revision_live_now": sha, "revision_lastModified": lastmod,
        "revision_matches": sha == row["revision"],
        "revision_check_is_post_hoc": True,
        "budget_bytes": B,
        "arm_bytes_A": bA, "arm_A_within_budget": bA <= B,
        "arm_bytes_B": bB, "arm_B_within_budget": bB <= B,
        "arm_bytes_CDE": row["arm_bytes"],
        "bits_per_weight": {"A": 8 * bA / W_ELEMS, "B": 8 * bB / W_ELEMS},
        "code_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
    }
    os.makedirs("logs/streamwd", exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(out) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    print(f"\n[complete] A {bA:,} B ({bA<=B}) | B {bB:,} B ({bB<=B}) | "
          f"budget {B:,} | revision match {out['revision_matches']}",
          flush=True)


if __name__ == "__main__":
    main()

"""HARDENING-P3 R4 wrapper: the cooldown small-delta cell of the
streaming 2x2 gets n=3 paired seeds — frozen driver
scratch/streaming_birth_d256.py untouched (import-and-override).

Two overrides, both BEFORE the driver's import executes it:
(1) scripts.train_mathnative.load_rows -> D2-excised (receipt
    printed; the P1 law);
(2) torch.save -> path-redirected: the driver writes SEED-FREE
    names (mathnative_wfloor_d256_stream.pt / _stream4.pt) that
    ALREADY EXIST from the 07-26 era — the redirect makes the
    frozen originals unreachable and gives each arm a
    seed-tagged OUT, with refuse-if-exists on the target.

Usage: ARM=hot|cool SEED=<n> .venv/bin/python scratch/p3_stream2x2.py
  hot  = v1 (mixed shuffled batches, surprise-gated LR, no tail)
  cool = v4 (same + final-10% cooldown)   [STREAM_V4=1]
Device: Mac/mps (device-of-origin of the whole streaming line).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

ARM = os.environ["ARM"]
SEED = os.environ["SEED"]
assert ARM in ("hot", "cool"), ARM
if ARM == "cool":
    os.environ["STREAM_V4"] = "1"
os.environ["BIRTH_SEED"] = SEED

target = Path(f"checkpoints/p3r4_stream_{ARM}_s{SEED}.pt")
if target.exists():
    raise SystemExit(f"REFUSING: {target} exists (use unspent SEED)")

sys.path.insert(0, "scratch")
import torch  # noqa: E402
import scripts.train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

_orig_rows = TM.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig_rows(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[p3-r4] D2 excision: {len(rows)} -> {len(kept)} rows "
          f"({len(rows) - len(kept)} excised)", flush=True)
    return kept


TM.load_rows = excised_load_rows

_orig_save = torch.save


def redirected_save(obj, path, *a, **kw):
    p = str(path)
    if "mathnative_wfloor_d256_stream" in p:
        print(f"[p3-r4] save redirect: {p} -> {target}", flush=True)
        p = str(target)
    return _orig_save(obj, p, *a, **kw)


torch.save = redirected_save

import streaming_birth_d256  # noqa: E402,F401  (import runs the birth)

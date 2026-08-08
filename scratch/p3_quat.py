"""HARDENING-P3 R7 wrapper: the quaternionic 4x conversion toll gets
n=3 warm-epoch seeds — frozen scratch/quat_convert.py untouched
(import-and-override; the driver runs at import).

Three pre-import overrides:
(1) torch.manual_seed REDIRECT: the driver hardcodes
    torch.manual_seed(1); the wrapper intercepts that exact call and
    substitutes SEED (plus random.seed(SEED) for any module-level
    random use). The projection is deterministic; only the warm
    epoch's RNG varies — which is exactly the replication variable.
(2) train_mathnative.load_rows -> D2-excised (receipt printed).
(3) torch.save REDIRECT: quat_convert_{a,b}.pt are cited frozen
    artifacts; every save goes to p3r7_quat_{ARM}_s{SEED}.pt with
    refuse-if-exists.

Usage: ARM=b SEED=<n> .venv/bin/python scratch/p3_quat.py
Device: Mac/mps (the symmetry ladder's device-of-origin).
"""
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SEED = int(os.environ["SEED"])
ARM = os.environ["ARM"]
target = Path(f"checkpoints/p3r7_quat_{ARM}_s{SEED}.pt")
if target.exists():
    raise SystemExit(f"REFUSING: {target} exists (use unspent SEED)")

import torch  # noqa: E402
import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

random.seed(SEED)

_orig_seed = torch.manual_seed


def redirected_seed(s):
    if s == 1:
        print(f"[p3-r7] manual_seed redirect: 1 -> {SEED}", flush=True)
        s = SEED
    return _orig_seed(s)


torch.manual_seed = redirected_seed

_orig_rows = TM.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig_rows(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[p3-r7] D2 excision: {len(rows)} -> {len(kept)} rows "
          f"({len(rows) - len(kept)} excised)", flush=True)
    return kept


TM.load_rows = excised_load_rows

_orig_save = torch.save


def redirected_save(obj, path, *a, **kw):
    p = str(path)
    if "quat_convert_" in p:
        print(f"[p3-r7] save redirect: {p} -> {target}", flush=True)
        p = str(target)
    return _orig_save(obj, p, *a, **kw)


torch.save = redirected_save

import quat_convert  # noqa: E402,F401  (import runs the conversion)

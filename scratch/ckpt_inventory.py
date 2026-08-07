"""Checkpoint triage INVENTORY (Artin GO 2026-08-07; banked 51GB
thread). READ-ONLY: walks checkpoints/ emitting one jsonl row per
file (path, bytes, mtime, sha256). Deletion decisions happen
elsewhere, with provenance, on Artin review — this script cannot
modify anything.

Usage: OUT=logs/triage/<host>_inventory.jsonl \
       .venv/bin/python scratch/ckpt_inventory.py
"""
import hashlib
import json
import os
import time
from pathlib import Path

out = Path(os.environ.get("OUT", "logs/triage/inventory.jsonl"))
out.parent.mkdir(parents=True, exist_ok=True)
n = tot = 0
with out.open("w") as f:
    for p in sorted(Path("checkpoints").rglob("*")):
        if not p.is_file():
            continue
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        st = p.stat()
        f.write(json.dumps({
            "path": str(p), "bytes": st.st_size,
            "mtime": int(st.st_mtime), "sha256": h.hexdigest()}) + "\n")
        n += 1; tot += st.st_size
        if n % 50 == 0:
            print(f"[inv] {n} files, {tot >> 30} GiB", flush=True)
print(f"[inv] DONE {n} files, {tot >> 30} GiB -> {out}", flush=True)

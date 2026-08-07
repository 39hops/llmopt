"""LEAN kernel-sample builder (provenance repair, 2026-08-07 audit:
the frozen 1000-id sample had no committed sampler — SEV-1 class).

Recipe per PRE-REG LEAN-KERNEL-SAMPLE: 1000 rows drawn
random.Random("lean-kernel-sample-0").sample from the 21,914-row
sidecar scratch/lean_real_corpus/parity_certs.jsonl (v1); the v2
sample is the SAME 1000 ids matched into parity_certs_v2.jsonl.
BYTE-IDENTITY against both existing sample files is asserted — this
builder aborts on mismatch and can never silently replace evidence.
Also emits the frozen id list (small text) for git tracking.

Usage: .venv/bin/python scratch/lean_sample_build.py
"""
import json
import random
from pathlib import Path

BASE = Path("scratch/lean_real_corpus")
rows = [l for l in (BASE / "parity_certs.jsonl").open()]
assert len(rows) == 21914, len(rows)
# TRUE recipe (re-derived 2026-08-07 against the frozen bytes; the
# naive sample-the-rows form MISMATCHES): the draw samples the ID
# LIST and emits rows in id-selection order.
by_id = {json.loads(l)["id"]: l for l in rows}
ids_all = [json.loads(l)["id"] for l in rows]
sel = random.Random("lean-kernel-sample-0").sample(ids_all, 1000)
# rows are RE-SERIALIZED via json.dumps (the frozen file's byte form;
# raw sidecar lines differ in whitespace only — verified 1000/1000
# semantically identical during the 2026-08-07 re-derivation)
picked = [json.dumps(json.loads(by_id[i])) + "\n" for i in sel]


def check(name, text):
    p = BASE / name
    if p.exists():
        if p.read_text() == text:
            print(f"{name}: BYTE-IDENTICAL")
            return
        raise SystemExit(f"{name}: MISMATCH — aborting, evidence frozen")
    p.write_text(text)
    print(f"{name}: written")


check("kernel_sample_1000.jsonl", "".join(picked))
ids = sel
idset = set(ids)
v2 = [l for l in (BASE / "parity_certs_v2.jsonl").open()
      if json.loads(l)["id"] in idset]
check("kernel_sample_1000_v2.jsonl", "".join(v2))
check("kernel_sample_1000_ids.txt", "\n".join(ids) + "\n")

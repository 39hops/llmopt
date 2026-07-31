"""ckpt_manifest.py — checkpoint manifest for the curated tree.

Default: walk checkpoints/confirmed/**, write/refresh
checkpoints/MANIFEST.jsonl (committed) with name, sha256, bytes,
mtime — PRESERVING curated fields (category, verdict, note) across
regens, keyed by relative path.
--all: scan ALL of checkpoints/ and print sha256/bytes to stdout
(per-machine forensics pass: dedupe exact twins, autopsy name-twins
that differ, cross-check scores against RESULTS — banked riff
2026-07-31).

Usage: python scripts/ckpt_manifest.py [--all]
"""
import hashlib
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
MANIFEST = os.path.join(ROOT, "MANIFEST.jsonl")


def sha256(path, bufsize=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan(base):
    for dirpath, _, names in os.walk(base):
        for n in sorted(names):
            if n.endswith((".pt", ".safetensors", ".npz", ".bin")):
                yield os.path.join(dirpath, n)


def main():
    if "--all" in sys.argv:
        for p in scan(ROOT):
            rel = os.path.relpath(p, ROOT)
            print(f"{sha256(p)}  {os.path.getsize(p):>12}  {rel}")
        return
    base = os.path.join(ROOT, "confirmed")
    old = {}
    if os.path.exists(MANIFEST):
        for line in open(MANIFEST):
            r = json.loads(line)
            old[r["path"]] = r
    rows = []
    for p in scan(base):
        rel = os.path.relpath(p, ROOT)
        r = {"path": rel, "sha256": sha256(p),
             "bytes": os.path.getsize(p)}
        for k in ("category", "verdict", "note"):  # curation survives
            if rel in old and k in old[rel]:
                r[k] = old[rel][k]
        rows.append(r)
    with open(MANIFEST, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} confirmed checkpoints -> {MANIFEST}")


if __name__ == "__main__":
    main()

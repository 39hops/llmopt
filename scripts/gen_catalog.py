"""gen_catalog.py — regenerate data/catalog/models.jsonl (EXHAUST, not evidence).

Walks checkpoints/*.pt (top level) + checkpoints/confirmed/** and
emits one row per file via llmopt.lab.catalog.scan_checkpoint, sorted
by path. The output is regenerable exhaust (logs doctrine
2026-08-06); the evidence record stays checkpoints/MANIFEST.jsonl +
RESULTS citations — never point a verdict at this file.

Flags:
  --no-sha      fast pass, sha256 left null
  --update      only (re)hash rows whose (bytes, mtime) changed vs the
                existing jsonl; unchanged rows keep their sha
  --limit N     scan only the first N files (tests)
  --out PATH    override output (tests)
  --root PATH   override repo root (tests)

Validation: if checkpoints/MANIFEST.jsonl (written by the frozen
scripts/ckpt_manifest.py — called, never rewritten) carries shas for
confirmed/ files we also hashed, mismatches RAISE.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llmopt.lab.catalog import scan_checkpoint  # noqa: E402


def cited_names(repo_root):
    p = os.path.join(repo_root, "docs", "RESULTS.md")
    if not os.path.exists(p):
        return set()
    with open(p, errors="replace") as f:
        text = f.read()
    names = set()
    for w in text.split():
        w = w.strip(".,;:`'\"()[]*")
        if w.endswith(".pt"):
            names.add(os.path.basename(w))
    return names


def walk_targets(ckpt_root):
    out = []
    if os.path.isdir(ckpt_root):
        for n in sorted(os.listdir(ckpt_root)):
            p = os.path.join(ckpt_root, n)
            if n.endswith(".pt") and os.path.isfile(p):
                out.append(p)
        conf = os.path.join(ckpt_root, "confirmed")
        for dirpath, dirs, names in os.walk(conf):
            dirs.sort()
            for n in sorted(names):
                if n.endswith(".pt"):
                    out.append(os.path.join(dirpath, n))
    return out


def load_jsonl(path):
    rows = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    rows[r["path"]] = r
    return rows


def cross_check_manifest(repo_root, rows_by_path):
    """Raise on sha disagreement with the frozen manifest (confirmed/)."""
    man = os.path.join(repo_root, "checkpoints", "MANIFEST.jsonl")
    if not os.path.exists(man):
        return 0
    checked = 0
    with open(man) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            rel = "checkpoints/" + m["path"]  # manifest paths are ckpt-relative
            r = rows_by_path.get(rel)
            if r and r.get("sha256") and m.get("sha256"):
                checked += 1
                if r["sha256"] != m["sha256"]:
                    raise RuntimeError(
                        f"sha mismatch vs MANIFEST.jsonl: {rel} "
                        f"catalog={r['sha256']} manifest={m['sha256']}")
    return checked


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-sha", action="store_true")
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=None)
    args = ap.parse_args(argv)

    repo_root = os.path.abspath(
        args.root or os.path.join(os.path.dirname(__file__), ".."))
    ckpt_root = os.path.join(repo_root, "checkpoints")
    out_path = args.out or os.path.join(repo_root, "data", "catalog",
                                        "models.jsonl")
    cited = cited_names(repo_root)
    old = load_jsonl(out_path) if args.update else {}

    targets = walk_targets(ckpt_root)
    if args.limit is not None:
        targets = targets[: args.limit]

    sib_cache = {}
    rows = []
    for p in targets:
        d = os.path.dirname(p)
        if d not in sib_cache:
            sib_cache[d] = os.listdir(d)
        rel = os.path.relpath(p, repo_root).replace(os.sep, "/")
        st = os.stat(p)
        prior = old.get(rel)
        reuse = (args.update and prior is not None
                 and prior.get("bytes") == st.st_size
                 and prior.get("mtime") == st.st_mtime)
        want_sha = (not args.no_sha) and not reuse
        row = scan_checkpoint(p, repo_root, cited, siblings=sib_cache[d],
                              want_sha=want_sha,
                              want_arch=not reuse)
        if reuse:
            row["sha256"] = prior.get("sha256")
            row["arch"] = prior.get("arch")
        rows.append(row)

    rows.sort(key=lambda r: r["path"])
    by_path = {r["path"]: r for r in rows}
    n = cross_check_manifest(repo_root, by_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows -> {out_path} "
          f"(manifest cross-checked: {n})")


if __name__ == "__main__":
    main()

"""QWEN-ATTN-ATTRIB-1 recomposer: build arm F/L/Q as a KEY-LEVEL byte
recomposition of two frozen WHOLE-0T artifacts (PRE-REG
QWEN-ATTN-ATTRIB-1; the MODEL1-TREE recomposition rules ride: no
Lloyd, no DP retraining, no recompression — every payload byte is
copied from an already-frozen artifact).

Recipes (registered):
  F: base B, donor C, keys ".self_attn." promoted in C  (64 tensors)
  L: base B, donor C, keys ".linear_attn." promoted     (144 tensors)
  Q: base A, donor C, keys ".linear_attn.in_proj_qkv."  (48 tensors)

Output: <root>/<NAME>/ shards mirroring vendor sharding (dense
offsets, qualify_artifact ladder must pass), manifest.json, a
19-row canonical digest chain (manifest + 18 shards) written both
into the artifact dir and to logs/qwenwhole/artifact_digest_<NAME>.txt,
and a compose receipt logs/qwenattrib/compose_<NAME>.json with
derived provenance (source chain shas, promoted bytes, code_commit).

    NAME=F .venv/bin/python scratch/qwen_recompose.py    (on the 3080)

The compose core is a pure function over manifests + payload
readers; tests/test_qwen_recompose.py pins it on synthetic
artifacts before any 27B byte moves (qualification ladder).
"""
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

RECIPES = {
    "F": {"base": "B", "donor": "C", "mark": ".self_attn.",
          "n_expected": 64},
    "L": {"base": "B", "donor": "C", "mark": ".linear_attn.",
          "n_expected": 144},
    "Q": {"base": "A", "donor": "C", "mark": ".linear_attn.in_proj_qkv.",
          "n_expected": 48},
}


def promoted_keys(base_man: dict, donor_man: dict, mark: str) -> list:
    """Keys the donor holds at a different codec than the base AND
    matching the recipe mark. Sorted for determinism."""
    return sorted(k for k in donor_man
                  if mark in k
                  and donor_man[k]["codec"] != base_man[k]["codec"])


def compose(base_man: dict, donor_man: dict, promote: set,
            read_base, read_donor, write_shard) -> dict:
    """Pure core: returns the new manifest. read_*(entry) -> bytes;
    write_shard(shard_name, [(key, entry_meta, payload), ...]) is
    called once per shard with DENSE offsets already assigned in
    the entry metas (off recomputed, len/codec/shape/meta copied
    from the chosen source entry)."""
    new_man = {}
    by_shard = {}
    for k, e in base_man.items():
        if e["codec"] == "excluded":
            # excluded tensors carry no shard/off/len (zero payload);
            # conservation counts them, the layout loop must not
            if k in promote:
                raise SystemExit(f"REFUSING: excluded key promoted {k}")
            new_man[k] = dict(e)
            continue
        by_shard.setdefault(e["shard"], []).append(k)
    for sh in sorted(by_shard):
        # deterministic layout: base's within-shard payload order
        keys = sorted(by_shard[sh], key=lambda k: (base_man[k]["off"],
                                                   k))
        rows, off = [], 0
        for k in keys:
            src_man, rd = ((donor_man, read_donor) if k in promote
                           else (base_man, read_base))
            e = dict(src_man[k])
            if e["shard"] != sh:
                raise SystemExit(f"REFUSING: {k} maps to {e['shard']} "
                                 f"in source but {sh} in base")
            payload = rd(src_man[k]) if e["codec"] != "excluded" else b""
            if len(payload) != (e["len"] if e["codec"] != "excluded"
                                else 0):
                raise SystemExit(f"REFUSING: {k} payload length "
                                 f"{len(payload)} != entry {e['len']}")
            e["off"] = off
            off += len(payload)
            rows.append((k, e, payload))
            new_man[k] = e
        write_shard(sh, rows)
    if set(new_man) != set(base_man):
        raise SystemExit("REFUSING: key-set conservation broke")
    return new_man


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    from llmopt.lab import qartifact
    name = os.environ["NAME"]
    rec = RECIPES[name]
    root = os.path.expanduser(os.environ.get("ART_ROOT",
                                             "~/qwen_whole0t"))
    vidx = os.path.expanduser(os.environ.get(
        "VENDOR_INDEX", "~/qwen_vendor/model.safetensors.index.json"))
    out = os.path.join(root, name)
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    t0 = time.time()
    srcs = {}
    for role in ("base", "donor"):
        arm = rec[role]
        d = os.path.join(root, arm)
        chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
        q = qartifact.qualify_artifact(d, vidx, chain)
        srcs[role] = {"dir": d, "man": q["manifest"],
                      "chain_sha": _sha(chain), "arm": arm}
        print(f"[rc] qualified {role}={arm}: {q['report']['census']}",
              flush=True)
    bm, dm = srcs["base"]["man"], srcs["donor"]["man"]
    promote = promoted_keys(bm, dm, rec["mark"])
    if len(promote) != rec["n_expected"]:
        raise SystemExit(f"REFUSING: {len(promote)} promoted keys, "
                         f"registered {rec['n_expected']}")
    added = sum(dm[k]["len"] - bm[k]["len"] for k in promote)
    print(f"[rc] {name}: {len(promote)} keys, +{added/2**30:.4f} GiB",
          flush=True)

    handles = {}

    def reader(art_dir):
        def rd(e):
            p = os.path.join(art_dir, e["shard"] + ".bin")
            if p not in handles:
                handles[p] = open(p, "rb")
            handles[p].seek(e["off"])
            return handles[p].read(e["len"])
        return rd

    os.makedirs(out)

    def write_shard(sh, rows):
        with open(os.path.join(out, sh + ".bin"), "wb") as f:
            for _, _, payload in rows:
                f.write(payload)

    new_man = compose(bm, dm, set(promote),
                      reader(srcs["base"]["dir"]),
                      reader(srcs["donor"]["dir"]), write_shard)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        f.write(json.dumps(new_man) + "\n")

    # canonical 19-row chain: manifest first, then shards in order
    shards = sorted({e["shard"] for e in new_man.values()
                     if e["codec"] != "excluded"})
    chain_rows = [(_sha(os.path.join(out, "manifest.json")),
                   "manifest.json")]
    chain_rows += [(_sha(os.path.join(out, s + ".bin")), s + ".bin")
                   for s in shards]
    chain_txt = "".join(f"{sha}  {fn}\n" for sha, fn in chain_rows)
    chain_path = f"logs/qwenwhole/artifact_digest_{name}.txt"
    os.makedirs("logs/qwenwhole", exist_ok=True)
    for p in (os.path.join(out, f"digest_{name}.txt"), chain_path):
        if os.path.exists(p):
            raise SystemExit(f"REFUSING: {p} exists")
        with open(p, "w") as f:
            f.write(chain_txt)

    # self-check: the new artifact must pass the full ladder
    q = qartifact.qualify_artifact(out, vidx, chain_path)
    print(f"[rc] self-qualified {name}: {q['report']['census']}",
          flush=True)

    os.makedirs("logs/qwenattrib", exist_ok=True)
    rcpt_path = f"logs/qwenattrib/compose_{name}.json"
    if os.path.exists(rcpt_path):
        raise SystemExit(f"REFUSING: {rcpt_path} exists")
    rcpt = {
        "name": name,
        "recipe": rec,
        "base": {"arm": srcs["base"]["arm"],
                 "dir": srcs["base"]["dir"],
                 "chain_sha256": srcs["base"]["chain_sha"]},
        "donor": {"arm": srcs["donor"]["arm"],
                  "dir": srcs["donor"]["dir"],
                  "chain_sha256": srcs["donor"]["chain_sha"]},
        "promoted_keys": len(promote),
        "bytes_added": added,
        "out_dir": out,
        "out_chain_sha256": _sha(chain_path),
        "self_qualify_census": q["report"]["census"],
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
        "tree_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain", "-uno"]).decode().strip()),
        "wall_s": round(time.time() - t0, 1)}
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rcpt) + "\n")
    print(f"[rc] receipt -> {rcpt_path} wall {rcpt['wall_s']}s",
          flush=True)


if __name__ == "__main__":
    main()

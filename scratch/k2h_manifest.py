"""K2-HORIZON-TRANSPORT-0 phase 0: resolve and freeze IMMUTABLE manifests
for the 3.7B and 7B checkpoint ladders before any model output. Tags are
mutable on the hub (re-pointed 2026-09-05), so every selected checkpoint
is frozen as (full commit, config.json sha256, tokenizer sha256s, index
sha256, per-shard LFS sha256 from the tree API, byte sizes). Also emits
the homology census: per-tag config (layers, hidden, vocab, rope,
max_position), relabel identities (tags whose shard-oid sets coincide),
and the tag graph per size. No weights are downloaded here.

Usage:
    .venv-k2/bin/python scratch/k2h_manifest.py  -> docs/preregs/k2h-transport-0.manifest.json
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs/preregs/k2h-transport-0.manifest.json")
SIZES = {"3.7B": "IFM/K2-Horizon-3.7B", "7B": "IFM/K2-Horizon-7B"}
PRETRAIN = [f"pretrain_{s}" for s in range(100000, 1100001, 100000)] + ["pretrain_final"]
FINALS = ["mid_1_final", "mid_2_final", "mid_3_final", "mid_4_final", "sft_1_final", "rl_merged"]
EXTRA = {"7B": ["sft_2_final"]}
SMALL = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json", "generation_config.json"]


def get(url):
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            time.sleep(2 + 2 * i)
            err = e
    raise SystemExit(f"fetch failed {url}: {err!r}")


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    man = {"frozen_on": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "sizes": {}}
    for size, repo in SIZES.items():
        refs = json.loads(get(f"https://huggingface.co/api/models/{repo}/refs"))
        tags = {t["name"]: t["targetCommit"] for t in refs["tags"]}
        branches = {b["name"]: b["targetCommit"] for b in refs["branches"]}
        sel = PRETRAIN + FINALS + EXTRA.get(size, [])
        entry = {"repo": repo, "all_tags": tags, "branches": branches, "selected": {}, "n_tags": len(tags)}
        for tag in sel:
            if tag not in tags:
                entry["selected"][tag] = {"status": "TAG-ABSENT"}
                print(size, tag, "ABSENT", flush=True)
                continue
            commit = tags[tag]
            tree = json.loads(get(f"https://huggingface.co/api/models/{repo}/tree/{commit}"))
            shards = {f["path"]: {"sha256": f["lfs"]["oid"], "size": f["size"]} for f in tree if f["path"].endswith(".safetensors")}
            small = {}
            for s in SMALL:
                if any(f["path"] == s for f in tree):
                    small[s] = hashlib.sha256(get(f"https://huggingface.co/{repo}/resolve/{commit}/{s}")).hexdigest()
            cfg = json.loads(get(f"https://huggingface.co/{repo}/resolve/{commit}/config.json"))
            entry["selected"][tag] = {"commit": commit, "shards": shards, "n_shards": len(shards), "bytes": sum(v["size"] for v in shards.values()),
                                      "small_sha256": small,
                                      "config": {k: cfg.get(k) for k in ("architectures", "num_hidden_layers", "hidden_size", "intermediate_size", "vocab_size",
                                                                          "num_attention_heads", "num_key_value_heads", "max_position_embeddings", "rope_parameters",
                                                                          "rope_scaling", "rope_theta", "tie_word_embeddings", "dtype", "torch_dtype")}}
            print(size, tag, commit[:8], len(shards), "shards", round(entry["selected"][tag]["bytes"] / 1e9, 2), "GB",
                  cfg.get("max_position_embeddings"), (cfg.get("rope_parameters") or {}).get("rope_type"), flush=True)
        # relabel identities: tags with identical shard-oid multisets
        oidsets = {}
        for tag, v in entry["selected"].items():
            if "shards" in v:
                key = tuple(sorted(s["sha256"] for s in v["shards"].values()))
                oidsets.setdefault(key, []).append(tag)
        entry["identical_weight_groups"] = [g for g in oidsets.values() if len(g) > 1]
        man["sizes"][size] = entry
    json.dump(man, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    for size, e in man["sizes"].items():
        print(size, "identical groups", e["identical_weight_groups"])


if __name__ == "__main__":
    main()

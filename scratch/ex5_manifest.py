"""EX5 mask manifest (desk): the committed identity record for the
six EX5 deletion masks, so the prereg's operands consume a COMMITTED
artifact rather than ephemeral gitignored checkpoint files.

Per mask: the exact 80 deleted (layer, expert) slots, the keepset
file sha256, and its deterministic string seed. Plus: sha256 of
ex5_build.py and of the two source artifacts the draws consumed
(checkpoints/ex3_inv_pooled.json named carriers,
checkpoints/moe_gt1_arm0.json pooled demand), and pairwise exact
slot overlap + Jaccard within each 3-mask family (the n_mask=3
audit: three near-identical draws are not three replicates).

Receipt: logs/ex5/mask_manifest.json (refuse-if-exists).

    .venv/bin/python scratch/ex5_manifest.py               (Mac desk)
"""
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

MASKS = {f"ex5_del_rank{j}": f"ex5-rank-{j}" for j in range(3)}
MASKS |= {f"ex5_del_layer{j}": f"ex5-layer-{j}" for j in range(3)}
SOURCES = ["scratch/ex5_build.py", "checkpoints/ex3_inv_pooled.json",
           "checkpoints/moe_gt1_arm0.json"]
OUT = Path("logs/ex5/mask_manifest.json")
N_EXPERTS = 128


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def deleted_slots(path):
    ks = json.loads(Path(path).read_text())
    return sorted([l, e] for l in range(48) for e in range(N_EXPERTS)
                  if e not in set(ks[str(l)]))


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/ex5_manifest.py",
                              "scratch/ex5_build.py",
                              "scratch/ex3_build.py",
                              "scratch/ex4_build.py"])
    masks = {}
    slots = {}
    for name, seed in MASKS.items():
        p = f"checkpoints/{name}.json"
        dele = deleted_slots(p)
        assert len(dele) == 80, (name, len(dele))
        slots[name] = {tuple(s) for s in dele}
        masks[name] = {"string_seed": seed,
                       "keepset_sha256": sha(p),
                       "deleted_slots": dele}
    families = {}
    for fam in ("rank", "layer"):
        members = [f"ex5_del_{fam}{j}" for j in range(3)]
        pairs = {}
        for a, b in combinations(members, 2):
            inter = len(slots[a] & slots[b])
            union = len(slots[a] | slots[b])
            pairs[f"{a}|{b}"] = {"exact_overlap": inter,
                                 "jaccard": round(inter / union, 4)}
        families[fam] = pairs

    rcpt = {"note": "EX5 mask manifest: exact deleted slots + shas "
                    "for the six frozen masks; pairwise family "
                    "overlap audits n_mask=3",
            "start": START, "completion_commit": completion_commit(),
            "sources_sha256": {p: sha(p) for p in SOURCES},
            "masks": masks,
            "family_pairwise": families}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rcpt, indent=1) + "\n")
    for fam, pairs in families.items():
        for k, v in pairs.items():
            print(f"{k}: overlap {v['exact_overlap']}/80 "
                  f"jaccard {v['jaccard']}")
    print(f"receipt -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

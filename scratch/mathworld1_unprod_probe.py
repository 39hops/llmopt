import glob
import hashlib
import json
import multiprocessing as mp
import sys
from collections import Counter

sys.path.insert(0, "/Users/artin/code/llmopt")


def sha(t):
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def probe(args):
    h, cur = args
    import sympy as sp

    import llmopt.search.derivation as derivation
    from llmopt.search.derivation import State, successors
    from scratch.mathworld1_actionsem import apply_at, sites_preorder
    parent = sp.sympify(cur)
    derivation._RULE_CACHE.clear()
    gen = sorted(successors(State(parent)),
                 key=lambda nc: (nc[0], nc[1].key()))
    acc = {c.key() for n, c in gen
           if (n.split("@", 1)[0] if "@" in n else n) == "i_unprod"}
    mx = 0
    for node in sites_preorder(parent, "I"):
        ck, _ = apply_at(parent, "i_unprod", node)
        mx = max(mx, len(set(ck) & acc))
    return h, mx


def main():
    recov = {}
    for l in open("/Users/artin/code/llmopt/logs/mathworld1/pdc_relabel.jsonl"):
        r = json.loads(l)
        for nxt, cl in r.get("rows", {}).items():
            if (isinstance(cl, dict)
                    and cl.get("class") == "unique_program"
                    and cl.get("rule") == "i_unprod"):
                recov.setdefault(r["cur_sha"], []).append(nxt)
    cur_of = {}
    files = sorted(glob.glob(
        "/Users/artin/code/llmopt/data/micromodel_chains_shard*.jsonl"))
    files.append("/Users/artin/code/llmopt/data/step_chains.jsonl")
    for f in files:
        for l in open(f):
            c = json.loads(l)["cur"]
            h = sha(c)
            if h in recov and h not in cur_of:
                cur_of[h] = c
    ctx = mp.get_context("fork")
    with ctx.Pool(9) as pool:
        res = dict(pool.imap_unordered(
            probe, list(cur_of.items()), chunksize=4))
    hist = Counter(res.values())
    branch_rows = sum(len(recov[h]) for h, m in res.items() if m > 1)
    out = {"i_unprod_rows": sum(len(v) for v in recov.values()),
           "unique_parents": len(recov),
           "max_siblings_per_site_hist":
               {str(k): v for k, v in sorted(hist.items())},
           "branch_gt0_exposed_rows": branch_rows}
    json.dump(out, open("/tmp/unprod_exposure.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()

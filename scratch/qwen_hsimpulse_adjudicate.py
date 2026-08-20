"""QWEN-HEADSWAP-IMPULSE-0 independent offline adjudicator: every
bar count, P1 check, and outcome class recomputed from the token-ID
sidecars alone (PRIMITIVE-EVIDENCE — the producer's in-run outcome
fields are non-authoritative and are only compared against, never
used). Also: orbit classification (ORIGINAL-ORBIT-REJOIN v
NEW-RECURRENCE v NO-RECURRENCE), CYCLE-comparable gap<=300 fields,
and full 18-shard BLe artifact verification against the qualified
digest chain.

    .venv/bin/python scratch/qwen_hsimpulse_adjudicate.py    (3080)
"""
import glob
import hashlib
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

OUT = "logs/qwenhsimpulse"
PARAMS = "docs/preregs/qwen-headswap-impulse-0.params.json"
PREREG = "docs/preregs/qwen-headswap-impulse-0.json"
FROZEN = "logs/qweneffort2_probe/traj_xhigh_{i}.json"
CHAIN = "logs/qwenwhole/artifact_digest_BLe.txt"
WIN, LAG_MAX, T_MIN = 32, 512, 544
ORIG_PERIOD = {0: 88, 4: 242}
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def detector_fires(ids, start_at):
    fires = []
    for t in range(max(T_MIN, start_at), len(ids) + 1):
        w = ids[t - WIN:t]
        lo = max(WIN, t - LAG_MAX)
        if any(ids[e - WIN:e] == w for e in range(lo, t - WIN + 1)):
            fires.append(t)
    return fires


def orbit_class(item, ids, pos, frozen_ids, fires_post):
    """ORIGINAL-ORBIT-REJOIN: the item's original exact cycle
    segment (one full frozen period, taken from the frozen tail)
    occurs verbatim in the branch after the injection. Item 3 has
    no exact frozen cycle; its rejoin anchor is the frozen G3
    restart landmark occurring post-injection."""
    if not fires_post:
        return "NO-RECURRENCE"
    if item in ORIG_PERIOD:
        L = ORIG_PERIOD[item]
        seg = frozen_ids[-L:]
        post = ids[pos:]
        rejoin = any(post[i:i + L] == seg
                     for i in range(len(post) - L))
    else:
        pj = json.load(open("docs/preregs/"
                            "qwen-loop-state-0.params.json"))
        g3 = pj["capture"]["item3"]["anchor_gram_g3"]
        post = ids[pos:]
        rejoin = any(post[i:i + 32] == g3
                     for i in range(len(post) - 32))
    return "ORIGINAL-ORBIT-REJOIN" if rejoin else "NEW-RECURRENCE"


def verify_shards():
    art = os.path.expanduser(os.environ.get(
        "ART_DIR", "~/qwen_whole0t/BLe"))
    chain = {}
    for line in open(CHAIN):
        parts = line.split()
        if len(parts) == 2:
            chain[parts[1]] = parts[0]
    checked, bad = 0, []
    for f in sorted(os.listdir(art)):
        if not f.endswith(".bin"):
            continue
        sha = hashlib.sha256(
            open(os.path.join(art, f), "rb").read()).hexdigest()
        checked += 1
        if chain.get(f) != sha:
            bad.append(f)
    return {"checked": checked, "mismatched": bad,
            "chain_file": CHAIN}


def _tt(ids):
    import torch
    return torch.tensor(ids)


def main():
    ep = _load("qwen_effort_probe", "scratch/qwen_effort_probe.py")
    tl = _load("qwen_tower_ladder", "scratch/qwen_tower_ladder.py")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tl.VDIR)
    inj = {(j["item"], j["pos"]): j
           for j in json.load(open(PARAMS))["injections"]}
    rows = [json.loads(l) for l in
            open(os.path.join(OUT, "impulse_rows.jsonl"))]
    prod = {(r["item"], r["inject_pos"]): r for r in rows}
    items = {i["id"]: i for i in ep.make_items(30)}
    eos = (248046, 248044)
    shards = verify_shards()
    branches, mismatches = [], []
    for sp in sorted(glob.glob(os.path.join(OUT, "traj_*.json"))):
        d = json.load(open(sp))
        r = d["row"]
        ids = d["gen_token_ids"]
        key = (r["item"], r["inject_pos"])
        j = inj[key]
        frozen = json.load(open(FROZEN.format(
            i=r["item"])))["gen_token_ids"]
        pos = j["pos"]
        p1 = (ids[:pos] == frozen[:pos]
              and ids[pos] == j["vendor_token"]
              and j["vendor_token"] != frozen[pos])
        fires = detector_fires(ids, pos + WIN)
        post = [f for f in fires if f > pos]
        eos_end = bool(ids and ids[-1] in eos)
        if post:
            outcome = "RECONVERGED"
        elif eos_end:
            outcome = "TERMINATED"
        else:
            outcome = "AMBIGUOUS"
        # independent correctness: decode the sidecar ids with the
        # vendor tokenizer and re-run parse + sympy oracle — the
        # producer's answer field is never used
        text = tok.decode(_tt(ids), skip_special_tokens=False)
        term = "</think>" in text
        vis = text.split("</think>", 1)[1] if term else text
        ans = ep.parse_answer(vis)
        ok = bool(ans and ep.check(ans, items[r["item"]]["truth"]))
        oc = orbit_class(r["item"], ids, pos, frozen, post)
        b = {"item": r["item"], "pos": pos, "p1": p1,
             "outcome": outcome, "orbit_class": oc,
             "return_gap": (post[0] - pos) if post else None,
             "gap_le_300": (post[0] - pos) <= 300 if post else None,
             "eos_terminated": eos_end, "correct": ok}
        branches.append(b)
        pr = prod[key]  # the rows-file row, which carries outcome
        for fld, mine in (("outcome", outcome),
                          ("p1_prefix_identical", p1),
                          ("correct", ok),
                          ("return_gap", b["return_gap"])):
            if fld not in pr:
                mismatches.append((os.path.basename(sp), fld,
                                   "ABSENT-IN-PRODUCER", mine))
            elif pr[fld] != mine:
                mismatches.append((os.path.basename(sp), fld,
                                   pr[fld], mine))
    n_rec = sum(b["outcome"] == "RECONVERGED" for b in branches)
    n_ok = sum(b["correct"] for b in branches)
    valid = (all(b["p1"] for b in branches)
             and len(branches) == 5
             and shards["checked"] == 18
             and not shards["mismatched"])
    obs = {
        "note": "independent offline recomputation from token-ID "
                "sidecars; producer outcome fields non-authoritative "
                "(compared, listed under producer_mismatches); full "
                "18-shard BLe verification against the qualified "
                "digest chain",
        "measurement_valid": valid,
        "arms": {"BLe": {
            "admissible": valid,
            "reason": f"{len(branches)}/5 sidecars, P1 "
                      f"{sum(b['p1'] for b in branches)}/5, BLe "
                      f"shards {shards['checked']}/18 chain-verified,"
                      f" mismatched {shards['mismatched']}"}},
        "measurements": {
            "1": {"value": n_rec,
                  "metric": "n_reconverged_branches",
                  "population": "branches:5 (one per registered "
                                "disagreement point)",
                  "aggregation": "count",
                  "provenance": "offline detector recomputation "
                                "from sidecars; "
                                + json.dumps([b["outcome"]
                                              for b in branches])},
            "2": {"value": n_ok,
                  "metric": "n_correct_branches",
                  "population": "branches:5 (one per registered "
                                "disagreement point)",
                  "aggregation": "count",
                  "provenance": "sympy oracle re-run offline on the "
                                "producer-parsed answers"}},
        "branches": branches,
        "producer_mismatches": mismatches,
        "ble_shard_verification": shards}
    p = os.path.join(OUT, "impulse_observations_offline.json")
    if os.path.exists(p) and os.environ.get("OBS_OVERWRITE") != "1":
        raise SystemExit(f"REFUSING: {p} exists")
    with open(p, "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation,
                                   load as load_prereg)
    doc = load_prereg(PREREG)
    outcomes = adjudicate_prereg(doc, obs)
    ref = adjudicate_refutation(doc, obs, bar_outcomes=outcomes)
    print(json.dumps({"bars": {o.bar_id: o.outcome
                               for o in outcomes},
                      "refutation": ref,
                      "branches": branches,
                      "producer_mismatches": mismatches,
                      "shards": shards}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

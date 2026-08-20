"""QWEN-HOMEO-ACTUATOR-0 independent offline adjudicator: every bar
count, the event table, the sanity-gate exactness, and both escape
counts recomputed from the token-ID sidecars alone
(PRIMITIVE-EVIDENCE — the producer's in-run outcome fields are
non-authoritative and only compared against, never used).
Correctness decodes each sidecar's ids with the vendor tokenizer
and re-runs parse + sympy independently. Both artifact chains (BLe
and BLem, 18 shards each) verified against their qualified digest
files.

    .venv/bin/python scratch/qwen_homeo_adjudicate.py        (3080)
"""
import glob
import hashlib
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

OUT = "logs/qwenhomeo"
PARAMS = "docs/preregs/qwen-homeo-actuator-0.params.json"
PREREG = "docs/preregs/qwen-homeo-actuator-0.json"
FROZEN = "logs/qweneffort2_probe/traj_xhigh_{i}.json"
WIN, LAG_MAX, T_MIN = 32, 512, 544
ESC_WIN = 300
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


def verify_chain(art_dir, chain_file):
    chain = {}
    for line in open(chain_file):
        parts = line.split()
        if len(parts) == 2:
            chain[parts[1]] = parts[0]
    checked, bad = 0, []
    for f in sorted(os.listdir(art_dir)):
        if not f.endswith(".bin"):
            continue
        sha = hashlib.sha256(
            open(os.path.join(art_dir, f), "rb").read()).hexdigest()
        checked += 1
        if chain.get(f) != sha:
            bad.append(f)
    return {"dir": art_dir, "checked": checked, "mismatched": bad,
            "chain_file": chain_file}


def main():
    import torch
    os.environ.setdefault("STEP", "n/a")
    root = os.path.expanduser(os.environ.get("ART_ROOT",
                                             "~/qwen_whole0t"))
    os.environ.setdefault("ART_DIR", os.path.join(root, "BLe"))
    ep = _load("qwen_effort_probe", "scratch/qwen_effort_probe.py")
    tl = _load("qwen_tower_ladder", "scratch/qwen_tower_ladder.py")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tl.VDIR)
    params = json.load(open(PARAMS))
    items_reg = params["items"]
    frozen = {}
    for i in items_reg:
        tr = json.load(open(FROZEN.format(i=i)))
        sha = hashlib.sha256(json.dumps(
            tr["gen_token_ids"]).encode()).hexdigest()
        assert sha == params["frozen_sidecar_sha256"][str(i)], i
        frozen[i] = tr["gen_token_ids"]
    # event table recomputed independently, then compared to receipt
    events = {i: detector_fires(frozen[i], 0)[0] for i in items_reg}
    ev = json.load(open(os.path.join(OUT, "homeo_event_table.json")))
    ev_match = {str(i): events[i] for i in items_reg} == \
        {str(k): v for k, v in ev["events"].items()}
    rows = [json.loads(l) for l in
            open(os.path.join(OUT, "homeo_rows.jsonl"))]
    prod = {(r["arm"], r["item"]): r for r in rows}
    items = {i["id"]: i for i in ep.make_items(30)}
    eos = (248046, 248044)
    chains = [verify_chain(os.path.join(root, a),
                           f"logs/qwenwhole/artifact_digest_{a}.txt")
              for a in ("BLe", "BLem")]
    branches, mismatches = [], []
    n_rl_exact = 0
    for sp in sorted(glob.glob(os.path.join(OUT, "traj_*.json"))):
        d = json.load(open(sp))
        r = d["row"]
        cont = d["gen_token_ids"]
        i = r["item"]
        t = events[i]
        arm = r["arm"]
        b = {"arm": arm, "item": i, "event_t": t,
             "cont_tokens": len(cont)}
        if arm == "REFRESH-LOW":
            n_cmp = min(len(cont), len(frozen[i]) - t)
            exact = (cont[:n_cmp] == frozen[i][t:t + n_cmp]
                     and len(cont) == len(frozen[i]) - t)
            b["exact"] = exact
            n_rl_exact += exact
            checks = (("exact", exact),)
        else:
            ids_full = frozen[i][:t] + cont
            fires = [f for f in detector_fires(ids_full, t + 1)
                     if f > t]
            escaped = not any(t < f <= t + ESC_WIN for f in fires)
            text = tok.decode(torch.tensor(cont),
                              skip_special_tokens=False)
            term = "</think>" in text
            vis = text.split("</think>", 1)[1] if term else text
            ans = ep.parse_answer(vis)
            ok = bool(ans and ep.check(ans, items[i]["truth"]))
            b.update({"escaped_300": escaped,
                      "post_fires_n": len(fires),
                      "first_post_fire": fires[0] if fires else None,
                      "eos_terminated": bool(cont and cont[-1] in eos),
                      "think_terminated": term, "correct": ok})
            checks = (("escaped_300", escaped), ("correct", ok),
                      ("first_post_fire", b["first_post_fire"]))
        branches.append(b)
        pr = prod[(arm, i)]
        for fld, mine in checks:
            if fld not in pr:
                mismatches.append((os.path.basename(sp), fld,
                                   "ABSENT-IN-PRODUCER", mine))
            elif pr[fld] != mine:
                mismatches.append((os.path.basename(sp), fld,
                                   pr[fld], mine))
    hot = [b for b in branches if b["arm"] == "HOT-HIGH"]
    ref = [b for b in branches if b["arm"] == "REFRESH-HIGH"]
    n_hot = sum(b["escaped_300"] for b in hot)
    n_ref = sum(b["escaped_300"] for b in ref)
    n_ok = sum(b["correct"] for b in hot + ref)
    valid = (n_rl_exact == len(items_reg) == 3 and ev_match
             and len(hot) == len(ref) == 3
             and all(c["checked"] == 18 and not c["mismatched"]
                     for c in chains))
    reason = (f"RL exact {n_rl_exact}/3, event table match "
              f"{ev_match}, sidecars {len(branches)}/9, chains "
              + "; ".join(f"{os.path.basename(c['dir'])} "
                          f"{c['checked']}/18 mismatched "
                          f"{c['mismatched']}" for c in chains))

    def meas(v, metric, pop, prov):
        return {"value": v, "metric": metric, "population": pop,
                "aggregation": "count", "provenance": prov}
    obs = {
        "note": "independent offline recomputation from token-ID "
                "sidecars; producer outcome fields non-authoritative "
                "(compared, listed under producer_mismatches); "
                "correctness independently decoded with the vendor "
                "tokenizer and re-run through parse + sympy; both "
                "artifact chains verified",
        "measurement_valid": valid,
        "arms": {a: {"admissible": valid, "reason": reason}
                 for a in ("REFRESH-LOW", "HOT-HIGH", "REFRESH-HIGH")},
        "measurements": {
            "1": meas(n_rl_exact, "n_refresh_low_exact",
                      "items:3 (frozen loop-state trajectories 0,3,4)",
                      "offline exact comparison of sidecar cont v "
                      "frozen ids"),
            "2": meas(n_hot, "n_hot_escaped",
                      "branches:3 (one HOT-HIGH branch per item)",
                      "offline detector recomputation; "
                      + json.dumps([b["escaped_300"] for b in hot])),
            "3": meas(n_ref, "n_refresh_escaped",
                      "branches:3 (one REFRESH-HIGH branch per item)",
                      "offline detector recomputation; "
                      + json.dumps([b["escaped_300"] for b in ref])),
            "4": meas(n_hot + n_ref, "n_high_escaped_total",
                      "branches:6 (3 HOT-HIGH + 3 REFRESH-HIGH)",
                      "sum of measurements 2 and 3"),
            "5": meas(n_ok, "n_correct_branches",
                      "branches:6 (3 HOT-HIGH + 3 REFRESH-HIGH)",
                      "sympy oracle on independently decoded "
                      "sidecar text")},
        "branches": branches,
        "producer_mismatches": mismatches,
        "artifact_verification": chains,
        "event_table_match": ev_match}
    p = os.path.join(OUT, "homeo_observations_offline.json")
    if os.path.exists(p) and os.environ.get("OBS_OVERWRITE") != "1":
        raise SystemExit(f"REFUSING: {p} exists")
    with open(p, "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation,
                                   load as load_prereg)
    doc = load_prereg(PREREG)
    outcomes = adjudicate_prereg(doc, obs)
    refv = adjudicate_refutation(doc, obs, bar_outcomes=outcomes)
    print(json.dumps({"bars": {o.bar_id: o.outcome for o in outcomes},
                      "refutation": refv,
                      "branches": branches,
                      "producer_mismatches": mismatches,
                      "chains": [{k: c[k] for k in
                                  ("dir", "checked", "mismatched")}
                                 for c in chains]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

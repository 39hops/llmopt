"""QWEN-HEADSWAP-IMPULSE-0 driver: single vendor-top1 token injected
at each of the 5 registered vendor/BLe disagreement points on BLe
greedy xhigh replays (PRE-REG in docs/RESULTS.md; frozen injection
table in docs/preregs/qwen-headswap-impulse-0.params.json).

Per branch: greedy replay to pos (prefix must sha-match the frozen
QWEN-LOOP-STATE-0 sidecar tokens, P1), emit the frozen vendor token
for that ONE position (recomputed from the attested slice at start;
REFUSES on mismatch with the frozen table), then BLe greedy to
MAX_TOK or eos. Full token-ID sidecar per branch. Outcome scoring
lives in the rows as primitives; bars are recomputed by the
adjudicator block at the end from the sidecars (cap-free frozen
detector; AMBIGUOUS fail-closed to not-reconverged).

    .venv/bin/python scratch/qwen_headswap_impulse.py        (3080)
    SMOKE=1 ... (smoke path logs/qwenhsimpulse_smoke, MAX_TOK 640,
        first branch only)
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SMOKE = os.environ.get("SMOKE") == "1"
os.environ.setdefault("ART_DIR", os.path.expanduser(
    "~/qwen_whole0t/BLe"))
# arm derived from the resolved artifact dir, never a free literal
ARM = os.path.basename(os.environ["ART_DIR"].rstrip("/"))
os.environ.setdefault("STEP", "n/a")
VSLICE = os.path.expanduser(os.environ.get(
    "VENDOR_SLICE", "~/qwen_vendor_lmhead"))
MAX_TOK = 640 if SMOKE else 3072
OUT = "logs/qwenhsimpulse_smoke" if SMOKE else "logs/qwenhsimpulse"
PARAMS = "docs/preregs/qwen-headswap-impulse-0.params.json"
LS_IN = "logs/qwenloopstate"
FROZEN = "logs/qweneffort2_probe/traj_xhigh_{i}.json"
WIN, LAG_MAX, T_MIN = 32, 512, 544
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def detector_fires(ids, start_at):
    """All fire positions of the frozen 32-gram detector over ids,
    cap-free, first evaluated position max(T_MIN, start_at)."""
    fires = []
    for t in range(max(T_MIN, start_at), len(ids) + 1):
        w = ids[t - WIN:t]
        lo = max(WIN, t - LAG_MAX)
        if any(ids[e - WIN:e] == w for e in range(lo, t - WIN + 1)):
            fires.append(t)
    return fires


def recheck_vendor_tokens(inj):
    """Recompute each vendor top1 from the attested slice + pinned
    npz h; REFUSE on any mismatch with the frozen table."""
    from safetensors import safe_open
    with safe_open(os.path.join(VSLICE, "lm_head.safetensors"),
                   framework="pt", device="cpu") as f:
        Wv = f.get_tensor("lm_head.weight").float().numpy()
    by_item = {}
    for j in inj:
        by_item.setdefault(j["item"], []).append(j)
    for rid, js in by_item.items():
        a = np.load(os.path.join(
            LS_IN, f"loopstate_arrays_id{rid}.npz"))
        pos = {int(p): i for i, p in enumerate(a["positions"])}
        h = a["h"].astype(np.float32)
        for j in js:
            hv = h[pos[j["pos"]]:pos[j["pos"]] + 1]
            bv, bi = -np.inf, -1
            for lo in range(0, Wv.shape[0], 16384):
                z = (hv @ Wv[lo:lo + 16384].T)[0]
                m = float(z.max())
                if m > bv:
                    bv, bi = m, int(z.argmax()) + lo
            if bi != j["vendor_token"]:
                raise SystemExit(
                    f"REFUSING: recomputed vendor token {bi} != "
                    f"frozen {j['vendor_token']} at item "
                    f"{j['item']} pos {j['pos']}")
            if int(a["gen_token_ids"][j["pos"]]) != j["ble_token"]:
                raise SystemExit(f"REFUSING: frozen ble_token "
                                 f"mismatch at {j}")
    print("[hi] injection table verified against slice + npz",
          flush=True)


def run_branch(model, tok, ep, it, frozen_ids, inj, eos):
    import torch
    text = tok.apply_chat_template(
        [{"role": "user", "content": it["prompt"]}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=True, reasoning_effort="xhigh")
    ids_in = tok(text, return_tensors="pt")["input_ids"].cuda()
    p_inj = inj["pos"]
    out_ids, past, cur = [], None, ids_in
    t0 = time.time()
    with torch.inference_mode():
        while len(out_ids) < MAX_TOK:
            o = model(input_ids=cur, past_key_values=past,
                      use_cache=True)
            past = o.past_key_values
            logits = o.logits[0, -1]
            p = len(out_ids)
            if p < p_inj:
                nxt = int(torch.argmax(logits).item())
                if nxt != frozen_ids[p]:
                    raise SystemExit(
                        f"P1 FAIL: replay diverged from frozen at "
                        f"{p}: {nxt} != {frozen_ids[p]}")
            elif p == p_inj:
                nxt = inj["vendor_token"]
            else:
                nxt = int(torch.argmax(logits).item())
            out_ids.append(nxt)
            if nxt in eos:
                break
            cur = torch.tensor([[nxt]], device="cuda")
    wall = time.time() - t0
    prefix_sha = hashlib.sha256(
        json.dumps(out_ids[:p_inj]).encode()).hexdigest()
    frozen_prefix_sha = hashlib.sha256(
        json.dumps(frozen_ids[:p_inj]).encode()).hexdigest()
    out = tok.decode(torch.tensor(out_ids), skip_special_tokens=False)
    terminated = "</think>" in out
    vis = out.split("</think>", 1)[1] if terminated else out
    ans = ep.parse_answer(vis)
    ok = bool(ans and ep.check(ans, it["truth"]))
    row = {"arm": ARM, "cell": "xhigh", "item": it["id"],
           "inject_pos": p_inj, "ble_token": inj["ble_token"],
           "vendor_token": inj["vendor_token"], "smoke": SMOKE,
           "p1_prefix_identical": prefix_sha == frozen_prefix_sha,
           "out_tokens": len(out_ids),
           "eos_terminated": bool(out_ids and out_ids[-1] in eos),
           "think_terminated": terminated,
           "correct": ok, "answer": ans, "truth": it["truth"],
           "gen_sha256": hashlib.sha256(
               json.dumps(out_ids).encode()).hexdigest(),
           "wall_s": round(wall, 1), "runtime": "qcuda_tower"}
    sp = os.path.join(OUT,
                      f"traj_i{it['id']}_p{p_inj}.json")
    if os.path.exists(sp):
        raise SystemExit(f"REFUSING: {sp} exists")
    with open(sp, "w") as f:
        f.write(json.dumps({"row": row, "gen_token_ids": out_ids})
                + "\n")
    row["sidecar"] = sp
    return row, out_ids


def main():
    import torch
    from transformers import AutoTokenizer
    ep = _load("qwen_effort_probe", "scratch/qwen_effort_probe.py")
    tl = _load("qwen_tower_ladder", "scratch/qwen_tower_ladder.py")
    inj = json.load(open(PARAMS))["injections"]
    if SMOKE:
        inj = inj[:1]
    frozen = {i: json.load(open(FROZEN.format(i=i)))["gen_token_ids"]
              for i in {j["item"] for j in inj}}
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "impulse_rows.jsonl")
    if os.path.exists(rows_path):
        raise SystemExit(f"REFUSING: {rows_path} exists")
    START = start_provenance(
        ["scratch/qwen_headswap_impulse.py",
         "scratch/qwen_tower_ladder.py",
         "scratch/qwen_effort_probe.py",
         "llmopt/lab/qcuda_tower.py", "llmopt/lab/qcuda.py",
         PARAMS, "docs/preregs/qwen-headswap-impulse-0.json"],
        artifacts={ARM: os.environ["ART_DIR"],
                   "vendor_slice": VSLICE,
                   "vendor_checkout": tl.VDIR})
    recheck_vendor_tokens(json.load(open(PARAMS))["injections"])
    tok = AutoTokenizer.from_pretrained(tl.VDIR)
    model, plan, routes, n_routes = tl.build_tower()
    torch.cuda.synchronize()
    print(f"[hi] tower up, routes {n_routes}", flush=True)
    items = {i["id"]: i for i in ep.make_items(30)}
    eos = (248046, 248044)
    branches = []
    for j in inj:
        row, ids = run_branch(model, tok, ep, items[j["item"]],
                              frozen[j["item"]], j, eos)
        # outcome primitives: fires recomputed cap-free from the
        # emitted ids, first evaluated at max(544, pos+32)
        fires = detector_fires(ids, j["pos"] + WIN)
        post = [f for f in fires if f > j["pos"]]
        if post:
            outcome = "RECONVERGED"
        elif row["eos_terminated"]:
            outcome = "TERMINATED"
        else:
            outcome = "AMBIGUOUS"
        row["post_fires_n"] = len(post)
        row["first_post_fire"] = post[0] if post else None
        row["return_gap"] = (post[0] - j["pos"]) if post else None
        row["outcome"] = outcome
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        branches.append(row)
        print(f"[hi] item{j['item']}@{j['pos']} {outcome} "
              f"gap={row['return_gap']} eos={row['eos_terminated']} "
              f"ok={row['correct']} gen={row['out_tokens']} "
              f"{row['wall_s']:.0f}s", flush=True)
    summ = {"start": START,
            "completion_commit": completion_commit(),
            "smoke": SMOKE, "n_branches": len(branches)}
    with open(os.path.join(OUT, "impulse_summary.json"), "w") as f:
        f.write(json.dumps(summ, indent=1) + "\n")
    if SMOKE:
        print("[hi] smoke done", flush=True)
        return 0
    n_rec = sum(b["outcome"] == "RECONVERGED" for b in branches)
    n_ok = sum(b["correct"] for b in branches)
    obs = {
        "note": "outcomes recomputed from emitted ids inside the "
                "driver (cap-free frozen detector; AMBIGUOUS "
                "fail-closed to not-reconverged); sidecars permit "
                "independent offline recomputation",
        "measurement_valid": all(b["p1_prefix_identical"]
                                 for b in branches),
        "arms": {"BLe": {
            "admissible": all(b["p1_prefix_identical"]
                              for b in branches),
            "reason": f"P1 prefix identity "
                      f"{sum(b['p1_prefix_identical'] for b in branches)}"
                      f"/{len(branches)}; injection table verified "
                      "against slice + npz at start"}},
        "measurements": {
            "1": {"value": n_rec,
                  "metric": "n_reconverged_branches",
                  "population": "branches:5 (one per registered "
                                "disagreement point)",
                  "aggregation": "count",
                  "provenance": "outcome field per row; "
                                + json.dumps(
                                    [b["outcome"] for b in branches])},
            "2": {"value": n_ok,
                  "metric": "n_correct_branches",
                  "population": "branches:5 (one per registered "
                                "disagreement point)",
                  "aggregation": "count",
                  "provenance": "sympy oracle via "
                                "qwen_effort_probe.check"}}}
    with open(os.path.join(OUT, "impulse_observations.json"),
              "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation,
                                   load as load_prereg)
    doc = load_prereg("docs/preregs/qwen-headswap-impulse-0.json")
    outcomes = adjudicate_prereg(doc, obs)
    ref = adjudicate_refutation(doc, obs, bar_outcomes=outcomes)
    print(json.dumps({"bars": {o.bar_id: o.outcome
                               for o in outcomes},
                      "refutation": ref,
                      "n_reconverged": n_rec, "n_correct": n_ok},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

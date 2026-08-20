"""QWEN-HOMEO-ACTUATOR-0 driver: one-band precision escalation
(BLe -> BLem) applied at the first frozen-detector fire of each
frozen loop trajectory (PRE-REG + AMENDMENT -REFRESH in
docs/RESULTS.md; constants and boundary semantics pinned in
docs/preregs/qwen-homeo-actuator-0.params.json).

Phase A (BLe tower): per item — derive event t (first detector
fire on the frozen sidecar ids), boundary fixture (teacher-forced
chunked prefill of frozen_ids[:t]; next-token prediction must equal
frozen_ids[t] via the REFRESH-LOW first token), serialize the
boundary state to CPU, REFRESH-LOW continuation from a RESTORED
copy (covers prefill + serializer roundtrip) which must reproduce
frozen_ids[t:] exactly. Phase B (BLem tower, plan-before-build):
per item — HOT-HIGH restores the saved BLe boundary state and
continues under BLem weights (F(W_BLem, S^BLe)); REFRESH-HIGH
teacher-forces the same frozen prefix through BLem and continues.
Both high arms predict token index t first (identical boundary).
Full token-ID sidecar per branch; outcomes are primitives in rows;
bars adjudicated at the end from the prereg machinery
(refutation_precedence: sanity bar 1 gates everything).

    .venv/bin/python scratch/qwen_homeo_actuator.py          (3080)
    SMOKE=1 ...   (smoke paths logs/qwenhomeo_smoke, item 0 only,
                   64-token continuations, REFRESH-LOW compared on
                   the first 64 tokens)
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SMOKE = os.environ.get("SMOKE") == "1"
os.environ.setdefault("STEP", "n/a")
ROOT_ART = os.path.expanduser(os.environ.get("ART_ROOT",
                                             "~/qwen_whole0t"))
BLE_DIR = os.path.join(ROOT_ART, "BLe")
BLEM_DIR = os.path.join(ROOT_ART, "BLem")
OUT = "logs/qwenhomeo_smoke" if SMOKE else "logs/qwenhomeo"
PARAMS = "docs/preregs/qwen-homeo-actuator-0.params.json"
PREREG = "docs/preregs/qwen-homeo-actuator-0.json"
FROZEN = "logs/qweneffort2_probe/traj_xhigh_{i}.json"
WIN, LAG_MAX, T_MIN = 32, 512, 544
ESC_WIN = 300
CAP = 64 if SMOKE else 3072
PREFILL_CHUNK = 512
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
    cap-free, first evaluated position max(T_MIN, start_at) — the
    HEADSWAP-IMPULSE implementation verbatim."""
    fires = []
    for t in range(max(T_MIN, start_at), len(ids) + 1):
        w = ids[t - WIN:t]
        lo = max(WIN, t - LAG_MAX)
        if any(ids[e - WIN:e] == w for e in range(lo, t - WIN + 1)):
            fires.append(t)
    return fires


def _move_state(obj, device, _seen=None):
    """Deep-copy a cache/state object with every tensor moved to
    device. Handles dicts/lists/tuples and plain attribute objects;
    refuses on containers it cannot rebuild."""
    import torch
    if _seen is None:
        _seen = {}
    oid = id(obj)
    if oid in _seen:
        return _seen[oid]
    if isinstance(obj, torch.nn.Module):
        raise SystemExit("REFUSING: nn.Module reachable from cache "
                         "state — serializer would copy the tower")
    if isinstance(obj, torch.Tensor):
        out = obj.to(device, copy=True)
    elif isinstance(obj, dict):
        out = {k: _move_state(v, device, _seen) for k, v in obj.items()}
    elif isinstance(obj, list):
        out = [_move_state(v, device, _seen) for v in obj]
    elif isinstance(obj, tuple):
        out = tuple(_move_state(v, device, _seen) for v in obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        out = obj
    elif hasattr(obj, "__dict__"):
        out = object.__new__(type(obj))
        _seen[oid] = out
        for k, v in vars(obj).items():
            setattr(out, k, _move_state(v, device, _seen))
        return out
    else:
        raise SystemExit(f"REFUSING: unserializable state member "
                         f"{type(obj)}")
    _seen[oid] = out
    return out


def prefill(model, ids_all):
    """Teacher-forced chunked prefill of ids_all[:-1]; returns
    (past, cur) where cur is the LAST token, so the next forward
    predicts the position after ids_all."""
    import torch
    past = None
    body = ids_all[:-1]
    with torch.inference_mode():
        for lo in range(0, len(body), PREFILL_CHUNK):
            chunk = torch.tensor([body[lo:lo + PREFILL_CHUNK]],
                                 device="cuda")
            o = model(input_ids=chunk, past_key_values=past,
                      use_cache=True)
            past = o.past_key_values
    return past, ids_all[-1]


def continue_greedy(model, past, cur, n_max, eos):
    import torch
    out_ids = []
    with torch.inference_mode():
        t = torch.tensor([[cur]], device="cuda")
        while len(out_ids) < n_max:
            o = model(input_ids=t, past_key_values=past,
                      use_cache=True)
            past = o.past_key_values
            nxt = int(torch.argmax(o.logits[0, -1]).item())
            out_ids.append(nxt)
            if nxt in eos:
                break
            t = torch.tensor([[nxt]], device="cuda")
    return out_ids


def sidecar(name, row, ids):
    sp = os.path.join(OUT, name)
    if os.path.exists(sp):
        raise SystemExit(f"REFUSING: {sp} exists")
    with open(sp, "w") as f:
        f.write(json.dumps({"row": row, "gen_token_ids": ids}) + "\n")
    return sp


def score_row(arm, item, t, cont, frozen_prefix, ep, tok, eos, wall):
    """Outcome primitives for one high-arm branch."""
    import torch
    ids_full = frozen_prefix + cont
    fires = [f for f in detector_fires(ids_full, t + 1) if f > t]
    escaped = not any(t < f <= t + ESC_WIN for f in fires)
    out = tok.decode(torch.tensor(cont), skip_special_tokens=False)
    terminated = "</think>" in out
    vis = out.split("</think>", 1)[1] if terminated else out
    ans = ep.parse_answer(vis)
    ok = bool(ans and ep.check(ans, item["truth"]))
    return {"arm": arm, "item": item["id"], "event_t": t,
            "smoke": SMOKE, "cont_tokens": len(cont),
            "eos_terminated": bool(cont and cont[-1] in eos),
            "think_terminated": terminated,
            "escaped_300": escaped,
            "post_fires_n": len(fires),
            "first_post_fire": fires[0] if fires else None,
            "correct": ok, "answer": ans, "truth": item["truth"],
            "gen_sha256": hashlib.sha256(
                json.dumps(cont).encode()).hexdigest(),
            "wall_s": round(wall, 1), "runtime": "qcuda_tower",
            "state_machine": ("F(W_BLem, S^BLe)" if arm == "HOT-HIGH"
                              else None)}


def main():
    import torch
    from transformers import AutoTokenizer
    params = json.load(open(PARAMS))
    assert (params["detector"]["WIN"], params["detector"]["LAG_MAX"],
            params["detector"]["T_MIN"]) == (WIN, LAG_MAX, T_MIN)
    assert params["detector"]["escape_window_tokens"] == ESC_WIN
    items_reg = [0] if SMOKE else params["items"]
    ep = _load("qwen_effort_probe", "scratch/qwen_effort_probe.py")
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "homeo_rows.jsonl")
    if os.path.exists(rows_path):
        raise SystemExit(f"REFUSING: {rows_path} exists")

    frozen = {}
    for i in items_reg:
        tr = json.load(open(FROZEN.format(i=i)))
        sha = hashlib.sha256(json.dumps(
            tr["gen_token_ids"]).encode()).hexdigest()
        if sha != params["frozen_sidecar_sha256"][str(i)]:
            raise SystemExit(f"REFUSING: frozen sidecar sha mismatch "
                             f"item {i}: {sha}")
        frozen[i] = tr["gen_token_ids"]

    os.environ["ART_DIR"] = BLE_DIR
    tlA = _load("qwen_tower_ladder_ble", "scratch/qwen_tower_ladder.py")
    START = start_provenance(
        ["scratch/qwen_homeo_actuator.py",
         "scratch/qwen_tower_ladder.py",
         "scratch/qwen_effort_probe.py",
         "llmopt/lab/qcuda_tower.py", "llmopt/lab/qcuda.py",
         PARAMS, PREREG],
        artifacts={"BLe": BLE_DIR, "BLem": BLEM_DIR,
                   "vendor_checkout": tlA.VDIR})
    tok = AutoTokenizer.from_pretrained(tlA.VDIR)
    items = {i["id"]: i for i in ep.make_items(30)}
    eos = (248046, 248044)

    # EVENT TABLE first (emitted before any continuation, per prereg)
    events = {}
    for i in items_reg:
        fires = detector_fires(frozen[i], 0)
        if not fires:
            raise SystemExit(f"REFUSING: no detector fire on frozen "
                             f"trajectory item {i}")
        t = fires[0]
        if not (T_MIN <= t < len(frozen[i])):
            raise SystemExit(f"REFUSING: event t={t} out of range "
                             f"item {i} (len {len(frozen[i])})")
        events[i] = t
    ev_path = os.path.join(OUT, "homeo_event_table.json")
    if os.path.exists(ev_path):
        raise SystemExit(f"REFUSING: {ev_path} exists")
    with open(ev_path, "w") as f:
        f.write(json.dumps({"events": events, "WIN": WIN,
                            "LAG_MAX": LAG_MAX, "T_MIN": T_MIN,
                            "escape_window": ESC_WIN,
                            "index_semantics":
                                params["detector"]["index_semantics"]},
                           indent=1) + "\n")
    print(f"[ha] event table {events}", flush=True)

    def prompt_ids(i):
        text = tok.apply_chat_template(
            [{"role": "user", "content": items[i]["prompt"]}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=True, reasoning_effort="xhigh")
        return tok(text)["input_ids"]

    # ---- Phase A: BLe — fixture + REFRESH-LOW + state capture ----
    model, plan, routes, n_routes = tlA.build_tower()
    torch.cuda.synchronize()
    print(f"[ha] BLe tower up, routes {n_routes}", flush=True)
    saved = {}      # item -> CPU boundary state (past, cur)
    rl_rows = []
    for i in items_reg:
        t = events[i]
        pids = prompt_ids(i)
        t0 = time.time()
        past, cur = prefill(model, pids + frozen[i][:t])
        cpu_state = _move_state(past, "cpu")
        saved[i] = (cpu_state, cur)
        # REFRESH-LOW from a RESTORED copy (serializer roundtrip in
        # the measured path)
        past_r = _move_state(cpu_state, "cuda")
        n_rl = min(CAP, len(frozen[i]) - t)
        cont = continue_greedy(model, past_r, cur, n_rl, eos)
        del past, past_r
        torch.cuda.empty_cache()
        wall = time.time() - t0
        exact = cont == frozen[i][t:t + n_rl]
        # boundary fixture: first restored-state prediction is the
        # frozen event-following token
        fixture_next = bool(cont) and cont[0] == frozen[i][t]
        row = {"arm": "REFRESH-LOW", "item": i, "event_t": t,
               "smoke": SMOKE, "cont_tokens": len(cont),
               "compared_tokens": n_rl, "exact": exact,
               "fixture_next_token_ok": fixture_next,
               "gen_sha256": hashlib.sha256(
                   json.dumps(cont).encode()).hexdigest(),
               "wall_s": round(wall, 1), "runtime": "qcuda_tower"}
        row["sidecar"] = sidecar(f"traj_RL_i{i}.json", row, cont)
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        rl_rows.append(row)
        print(f"[ha] RL item{i} t={t} exact={exact} "
              f"next_ok={fixture_next} n={len(cont)} "
              f"{wall:.0f}s", flush=True)
        if not (exact and fixture_next):
            raise SystemExit("INSTRUMENT-INVALID: sanity gate failed "
                             f"on item {i} — no treatment cell runs")
    del model
    torch.cuda.empty_cache()
    print("[ha] BLe tower freed", flush=True)

    # ---- Phase B: BLem — HOT-HIGH + REFRESH-HIGH ----
    os.environ["ART_DIR"] = BLEM_DIR
    tlB = _load("qwen_tower_ladder_blem",
                "scratch/qwen_tower_ladder.py")
    model, plan, routes, n_routes = tlB.build_tower()
    torch.cuda.synchronize()
    print(f"[ha] BLem tower up, routes {n_routes}", flush=True)
    hi_rows = []
    for i in items_reg:
        t = events[i]
        # HOT-HIGH: restore the saved BLe boundary state
        cpu_state, cur = saved[i]
        past_h = _move_state(cpu_state, "cuda")
        t0 = time.time()
        cont = continue_greedy(model, past_h, cur, CAP, eos)
        del past_h
        torch.cuda.empty_cache()
        row = score_row("HOT-HIGH", items[i], t, cont, frozen[i][:t],
                        ep, tok, eos, time.time() - t0)
        row["sidecar"] = sidecar(f"traj_HOT_i{i}.json", row, cont)
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        hi_rows.append(row)
        print(f"[ha] HOT item{i} esc={row['escaped_300']} "
              f"eos={row['eos_terminated']} ok={row['correct']} "
              f"n={row['cont_tokens']} {row['wall_s']:.0f}s",
              flush=True)
        # REFRESH-HIGH: same frozen prefix teacher-forced under BLem
        pids = prompt_ids(i)
        t0 = time.time()
        past_r, cur_r = prefill(model, pids + frozen[i][:t])
        cont = continue_greedy(model, past_r, cur_r, CAP, eos)
        del past_r
        torch.cuda.empty_cache()
        row = score_row("REFRESH-HIGH", items[i], t, cont,
                        frozen[i][:t], ep, tok, eos,
                        time.time() - t0)
        row["sidecar"] = sidecar(f"traj_RH_i{i}.json", row, cont)
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        hi_rows.append(row)
        print(f"[ha] RH item{i} esc={row['escaped_300']} "
              f"eos={row['eos_terminated']} ok={row['correct']} "
              f"n={row['cont_tokens']} {row['wall_s']:.0f}s",
              flush=True)

    summ = {"start": START, "completion_commit": completion_commit(),
            "smoke": SMOKE, "events": events,
            "n_rows": len(rl_rows) + len(hi_rows)}
    with open(os.path.join(OUT, "homeo_summary.json"), "w") as f:
        f.write(json.dumps(summ, indent=1) + "\n")

    hot = [r for r in hi_rows if r["arm"] == "HOT-HIGH"]
    ref = [r for r in hi_rows if r["arm"] == "REFRESH-HIGH"]
    n_pop = len(items_reg)

    def meas(v, metric, pop):
        return {"value": v, "metric": metric, "population": pop,
                "aggregation": "count",
                "provenance": "driver primitives; offline consumer "
                              "recomputes from sidecars before "
                              "booking"}
    obs = {
        "note": "outcomes computed from emitted ids inside the "
                "driver (frozen detector, 300-token escape window); "
                "sidecars permit independent offline recomputation; "
                "HOT-HIGH continuations compute F(W_BLem, S^BLe)",
        "measurement_valid": all(r["exact"] for r in rl_rows),
        "arms": {a: {"admissible": True, "reason":
                     "sanity gate 3/3 exact; event table + boundary "
                     "fixture receipted before treatment"}
                 for a in ("REFRESH-LOW", "HOT-HIGH", "REFRESH-HIGH")},
        "measurements": {
            "1": meas(sum(r["exact"] for r in rl_rows),
                      "n_refresh_low_exact",
                      "items:3 (frozen loop-state trajectories 0,3,4)"),
            "2": meas(sum(r["escaped_300"] for r in hot),
                      "n_hot_escaped",
                      "branches:3 (one HOT-HIGH branch per item)"),
            "3": meas(sum(r["escaped_300"] for r in ref),
                      "n_refresh_escaped",
                      "branches:3 (one REFRESH-HIGH branch per item)"),
            "4": meas(sum(r["escaped_300"] for r in hot + ref),
                      "n_high_escaped_total",
                      "branches:6 (3 HOT-HIGH + 3 REFRESH-HIGH)"),
            "5": meas(sum(r["correct"] for r in hot + ref),
                      "n_correct_branches",
                      "branches:6 (3 HOT-HIGH + 3 REFRESH-HIGH)")}}
    with open(os.path.join(OUT, "homeo_observations.json"), "w") as f:
        f.write(json.dumps(obs, indent=1) + "\n")
    if SMOKE:
        print(f"[ha] smoke done (obs block exercised, n_pop {n_pop})",
              flush=True)
        return 0
    from llmopt.lab.prereg import (adjudicate_prereg,
                                   adjudicate_refutation,
                                   load as load_prereg)
    doc = load_prereg(PREREG)
    outcomes = adjudicate_prereg(doc, obs)
    refv = adjudicate_refutation(doc, obs, bar_outcomes=outcomes)
    print(json.dumps({"bars": {o.bar_id: o.outcome for o in outcomes},
                      "refutation": refv}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

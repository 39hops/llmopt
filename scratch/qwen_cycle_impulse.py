"""QWEN-CYCLE-IMPULSE-0 driver: cycle-triggered temperature impulse
on BLe xhigh trajectories (PRE-REG frozen parameters — detector,
impulse, escape — live in docs/RESULTS.md; any change is a new
registration).

Greedy token-by-token until the recurrence detector fires (last 32
generated ids re-occur exactly with lag 32..512, checked from
position 544), then 8 tokens sampled at temperature T (top-p 1.0,
string-seeded generator), then greedy again; detector re-arms, cap
16 bursts. Full token-ID sidecar per run (TRAJECTORY-SIDECAR).

    ARM=BLe .venv/bin/python scratch/qwen_cycle_impulse.py   (3080)

Rows -> logs/qwencycle/impulse_rows.jsonl (refuse-if-exists);
per-run sidecars logs/qwencycle/traj_T<T>_s<seed>_id<id>.json.
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

ARM = os.environ.get("ARM", "BLe")
assert ARM in ("BLe",), ARM
os.environ.setdefault("ART_DIR", os.path.expanduser(
    f"~/qwen_whole0t/{ARM}"))
os.environ.setdefault("STEP", "n/a")
MAX_TOK = 3072
ITEM_IDS = (0, 3, 4)
TEMPS = (0.3, 0.7)
SEEDS = (0, 1, 2)
WIN, LAG_MAX, T_MIN = 32, 512, 544
BURST_LEN, BURST_CAP, ESCAPE_GAP = 8, 16, 300
OUT = "logs/qwencycle"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def detector_fire(ids, t):
    """PRE-REG detector: at generated position t (1-based count of
    generated ids), fire iff t >= T_MIN and ids[t-WIN:t] equals a
    WIN-gram ending at some position e with 32 <= t-e <= LAG_MAX."""
    if t < T_MIN:
        return False
    w = ids[t - WIN:t]
    lo = max(WIN, t - LAG_MAX)
    for e in range(lo, t - WIN + 1):
        if ids[e - WIN:e] == w:
            return True
    return False


def string_seed(tag):
    return int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)


def run_one(model, tok, ep, it, temp, seed, eos):
    import torch
    text = tok.apply_chat_template(
        [{"role": "user", "content": it["prompt"]}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=True, reasoning_effort="xhigh")
    ids_in = tok(text, return_tensors="pt")["input_ids"].cuda()
    gen = torch.Generator(device="cuda")
    gen.manual_seed(string_seed(f"cycle-impulse-{temp}-{seed}"))
    out_ids, bursts, fires = [], [], []
    burst_left, n_bursts = 0, 0
    past = None
    cur = ids_in
    t0 = time.time()
    with torch.inference_mode():
        while len(out_ids) < MAX_TOK:
            o = model(input_ids=cur, past_key_values=past,
                      use_cache=True)
            past = o.past_key_values
            logits = o.logits[0, -1]
            if burst_left > 0:
                probs = torch.softmax(logits / temp, dim=-1)
                nxt = int(torch.multinomial(probs, 1,
                                            generator=gen).item())
                burst_left -= 1
            else:
                nxt = int(torch.argmax(logits).item())
            out_ids.append(nxt)
            if nxt in eos:
                break
            t = len(out_ids)
            if (burst_left == 0 and n_bursts < BURST_CAP
                    and detector_fire(out_ids, t)):
                fires.append(t)
                bursts.append({"fire_at": t, "temp": temp})
                burst_left = BURST_LEN
                n_bursts += 1
            cur = torch.tensor([[nxt]], device="cuda")
    wall = time.time() - t0
    out = tok.decode(torch.tensor(out_ids), skip_special_tokens=False)
    terminated = "</think>" in out
    vis = out.split("</think>", 1)[1] if terminated else out
    ans = ep.parse_answer(vis)
    ok = bool(ans and ep.check(ans, it["truth"]))
    # PRE-REG escape: a burst escapes iff no detector fire within
    # ESCAPE_GAP tokens after its burst ends, or eos inside that gap.
    # A window that ended at the token cap without eos and without
    # covering the full gap counts NOT escaped (conservative).
    escapes = []
    eos_end = bool(out_ids and out_ids[-1] in eos)
    for b in bursts:
        end = b["fire_at"] + BURST_LEN
        refired = any(end < f <= end + ESCAPE_GAP for f in fires)
        window_complete = len(out_ids) > end + ESCAPE_GAP
        eos_in_gap = eos_end and len(out_ids) <= end + ESCAPE_GAP
        escapes.append((not refired) and
                       (window_complete or eos_in_gap))
    row = {"arm": ARM, "cell": "xhigh", "id": it["id"],
           "family": it["family"], "temp": temp, "seed": seed,
           "n_fires": len(fires), "fire_positions": fires,
           "n_bursts": n_bursts,
           "burst_escapes": escapes,
           "run_escaped": any(escapes),
           "detector_silent": len(fires) == 0,
           "eos_terminated": bool(out_ids and out_ids[-1] in eos),
           "think_terminated": terminated,
           "correct": ok, "answer": ans, "truth": it["truth"],
           "out_tokens": len(out_ids), "wall_s": round(wall, 1),
           "runtime": "qcuda_tower", "greedy_base": True}
    sidecar = {"row": row, "gen_token_ids": out_ids,
               "gen_sha256": hashlib.sha256(
                   json.dumps(out_ids).encode()).hexdigest(),
               "text": out}
    sp = os.path.join(OUT, f"traj_T{temp}_s{seed}_id{it['id']}.json")
    with open(sp, "w") as f:
        f.write(json.dumps(sidecar) + "\n")
    return row


def main():
    import torch
    from transformers import AutoTokenizer
    ep = _load("qwen_effort_probe", "scratch/qwen_effort_probe.py")
    tl = _load("qwen_tower_ladder", "scratch/qwen_tower_ladder.py")
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "impulse_rows.jsonl")
    if os.path.exists(rows_path):
        raise SystemExit(f"REFUSING: {rows_path} exists")
    START = start_provenance(
        ["scratch/qwen_cycle_impulse.py",
         "scratch/qwen_tower_ladder.py",
         "scratch/qwen_effort_probe.py",
         "llmopt/lab/qcuda_tower.py",
         "llmopt/lab/qcuda.py"])
    tok = AutoTokenizer.from_pretrained(tl.VDIR)
    model, plan, routes, n_routes = tl.build_tower()
    torch.cuda.synchronize()
    print(f"[ci] tower up, routes {n_routes}", flush=True)
    items = {i["id"]: i for i in ep.make_items(30)}
    eos = (248046, 248044)
    for temp in TEMPS:
        for seed in SEEDS:
            for rid in ITEM_IDS:
                row = run_one(model, tok, ep, items[rid],
                              temp, seed, eos)
                with open(rows_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"[ci] T={temp} s={seed} #{rid} "
                      f"fires={row['n_fires']} "
                      f"escaped={row['run_escaped']} "
                      f"eos={row['eos_terminated']} "
                      f"ok={row['correct']} gen={row['out_tokens']} "
                      f"{row['wall_s']:.0f}s", flush=True)
    summ = {"start": START, "completion_commit": completion_commit(),
            "n_rows": len(TEMPS) * len(SEEDS) * len(ITEM_IDS)}
    with open(os.path.join(OUT, "impulse_summary.json"), "w") as f:
        f.write(json.dumps(summ, indent=1) + "\n")
    print("[ci] done", flush=True)


if __name__ == "__main__":
    main()

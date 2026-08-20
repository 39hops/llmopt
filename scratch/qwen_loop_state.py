"""QWEN-LOOP-STATE-0 driver: observation-only greedy regeneration of
the three frozen BLe xhigh loop trajectories with sparse pre-head
hidden-state capture (PRE-REG in docs/RESULTS.md; frozen capture
sets, anchors, and precondition shas live in
docs/preregs/qwen-loop-state-0.json).

No impulses, no sampling: the run is valid only if the regenerated
token ids match the frozen autopsy sidecars sha-for-sha (P1). h_t is
the input to the fused s16 lm_head (post final norm, post last-pos
slice), captured fp32 at the registered positions only. Bars are
computed by the adjudicator from the persisted primitives, never
here (PRIMITIVE-EVIDENCE doctrine).

    .venv/bin/python scratch/qwen_loop_state.py           (3080)
    SMOKE=1 .venv/bin/python scratch/qwen_loop_state.py   (short cap,
        smoke paths only, P1 skipped: a 256-token prefix cannot match
        the 3072-token frozen sha)

Rows -> logs/qwenloopstate/loopstate_rows.jsonl (refuse-if-exists);
arrays -> logs/qwenloopstate/loopstate_arrays_id<id>.npz
(refuse-if-exists; untracked, sha recorded in the row).
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
ARM = "BLe"
os.environ.setdefault("ART_DIR", os.path.expanduser(
    f"~/qwen_whole0t/{ARM}"))
os.environ.setdefault("STEP", "n/a")
MAX_TOK = 256 if SMOKE else 3072
ITEM_IDS = (0, 4, 3)  # primary rigid orbits first, contrast last
OUT = "logs/qwenloopstate_smoke" if SMOKE else "logs/qwenloopstate"
PREREG = "docs/preregs/qwen-loop-state-0.json"
FROZEN = "logs/qweneffort2_probe/traj_xhigh_{i}.json"
WIN, LAG_MAX, T_MIN = 32, 512, 544  # frozen CYCLE detector (color)
EVENT_HALF = 16
TOPK = 256
N_FIXTURE = 8
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def first_fire(ids):
    """First position where the frozen 32-gram detector fires on the
    FROZEN ids (cap-free), or None. Positions are 1-based counts in
    the detector convention; returned as the 0-based index of the
    last token of the firing window."""
    for t in range(T_MIN, len(ids) + 1):
        w = ids[t - WIN:t]
        lo = max(WIN, t - LAG_MAX)
        if any(ids[e - WIN:e] == w for e in range(lo, t - WIN + 1)):
            return t - 1
    return None


def capture_positions(pj, frozen_ids):
    """Frozen capture set per item id -> sorted position list, plus
    the item-3 full-logit positions. Derived only from the prereg
    JSON and the FROZEN sidecar ids."""
    cap = {}
    full3 = set()
    cap[0] = list(range(1200, 1464))
    cap[4] = list(range(1400, 2126))
    a3 = pj["capture"]["item3"]["anchor_positions"]
    w3 = pj["capture"]["item3"]["anchor_window"]
    cap[3] = sorted({p for a in a3 for p in range(a, a + w3)})
    full3 = set(cap[3])
    for i in ITEM_IDS:
        f = first_fire(frozen_ids[i])
        if f is not None:
            lo, hi = max(0, f - EVENT_HALF), f + EVENT_HALF + 1
            cap[i] = sorted(set(cap[i]) | set(range(lo, hi)))
        if SMOKE:
            cap[i] = [p for p in cap[i] if p < MAX_TOK] or [8, 9, 10]
    return cap, full3


def run_item(model, tok, it, cap_pos, full_pos, eos):
    import numpy as np
    import torch
    text = tok.apply_chat_template(
        [{"role": "user", "content": it["prompt"]}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=True, reasoning_effort="xhigh")
    ids_in = tok(text, return_tensors="pt")["input_ids"].cuda()
    capset = set(cap_pos)
    hbox = {}
    hook = model.lm_head.inner.register_forward_pre_hook(
        lambda m, args: hbox.__setitem__("h", args[0]))
    H, P, T256i, T256v, ent, marg, lse = [], [], [], [], [], [], []
    fixture_logits, full_rows, full_rows_pos = [], [], []
    out_ids, past, cur = [], None, ids_in
    t0 = time.time()
    with torch.inference_mode():
        while len(out_ids) < MAX_TOK:
            o = model(input_ids=cur, past_key_values=past,
                      use_cache=True)
            past = o.past_key_values
            logits = o.logits[0, -1]
            p = len(out_ids)
            if p in capset:
                h = hbox["h"].reshape(-1, hbox["h"].shape[-1])[-1]
                hbox["dtype"] = h.dtype
                H.append(h.float().cpu().numpy())
                P.append(p)
                lf = logits.float()
                tv, ti = torch.topk(lf, TOPK)
                T256i.append(ti.cpu().numpy())
                T256v.append(tv.cpu().numpy())
                ls = torch.logsumexp(lf, 0)
                pr = torch.softmax(lf, 0)
                ent.append(float(-(pr * (pr.clamp_min(1e-30)).log())
                                 .sum()))
                marg.append(float(tv[0] - tv[1]))
                lse.append(float(ls))
                if len(fixture_logits) < N_FIXTURE:
                    fixture_logits.append(logits.detach().cpu()
                                          .numpy())
                if p in full_pos:
                    full_rows.append(logits.detach().half().cpu()
                                     .numpy())
                    full_rows_pos.append(p)
            nxt = int(torch.argmax(logits).item())
            out_ids.append(nxt)
            if nxt in eos:
                break
            cur = torch.tensor([[nxt]], device="cuda")
    hook.remove()
    wall = time.time() - t0
    # P2 fixture: captured h re-fed through the SAME fused head must
    # reproduce the in-run logits bit-exactly.
    fx = []
    with torch.inference_mode():
        for k in range(len(fixture_logits)):
            # fp32 roundtrip of a narrower float is lossless, so
            # casting back to the captured dtype reproduces the exact
            # head input.
            hh = torch.from_numpy(H[k]).cuda().to(hbox["dtype"])
            re = model.lm_head.inner(hh.view(1, 1, -1))[0, -1]
            ref = torch.from_numpy(fixture_logits[k]).cuda()
            fx.append({"pos": P[k],
                       "top1_identical": bool(int(re.argmax())
                                              == int(ref.argmax())),
                       "max_abs_diff": float((re.float()
                                              - ref.float()).abs()
                                             .max())})
    gen_sha = hashlib.sha256(json.dumps(out_ids).encode()).hexdigest()
    arr_path = os.path.join(OUT,
                            f"loopstate_arrays_id{it['id']}.npz")
    if os.path.exists(arr_path):
        raise SystemExit(f"REFUSING: {arr_path} exists")
    np.savez_compressed(
        arr_path, h=np.stack(H).astype(np.float32),
        positions=np.array(P, dtype=np.int32),
        top256_ids=np.stack(T256i), top256_logits=np.stack(T256v),
        entropy=np.array(ent), local_margin=np.array(marg),
        logsumexp=np.array(lse),
        fixture_logits=np.stack(fixture_logits),
        full_logits_fp16=(np.stack(full_rows) if full_rows
                          else np.zeros((0,), np.float16)),
        full_logits_pos=np.array(full_rows_pos, dtype=np.int32),
        gen_token_ids=np.array(out_ids, dtype=np.int64))
    arr_sha = hashlib.sha256(open(arr_path, "rb").read()).hexdigest()
    row = {"arm": ARM, "cell": "xhigh", "id": it["id"],
           "family": it["family"], "smoke": SMOKE,
           "out_tokens": len(out_ids), "gen_sha256": gen_sha,
           "n_captured": len(P), "capture_positions_sha": hashlib
           .sha256(json.dumps(cap_pos).encode()).hexdigest(),
           "fixture": fx, "wall_s": round(wall, 1),
           "arrays": arr_path, "arrays_sha256": arr_sha,
           "runtime": "qcuda_tower", "greedy": True,
           "observation_only": True}
    return row


def main():
    import torch
    from transformers import AutoTokenizer
    ep = _load("qwen_effort_probe", "scratch/qwen_effort_probe.py")
    tl = _load("qwen_tower_ladder", "scratch/qwen_tower_ladder.py")
    pj = json.load(open(PREREG))
    frozen = {i: json.load(open(FROZEN.format(i=i)))
              for i in ITEM_IDS}
    frozen_ids = {i: frozen[i]["gen_token_ids"] for i in ITEM_IDS}
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "loopstate_rows.jsonl")
    if os.path.exists(rows_path):
        raise SystemExit(f"REFUSING: {rows_path} exists")
    START = start_provenance(
        ["scratch/qwen_loop_state.py",
         "scratch/qwen_tower_ladder.py",
         "scratch/qwen_effort_probe.py",
         "llmopt/lab/qcuda_tower.py", "llmopt/lab/qcuda.py",
         PREREG])
    cap, full3 = capture_positions(pj, frozen_ids)
    tok = AutoTokenizer.from_pretrained(tl.VDIR)
    model, plan, routes, n_routes = tl.build_tower()
    torch.cuda.synchronize()
    print(f"[ls] tower up, routes {n_routes}", flush=True)
    items = {i["id"]: i for i in ep.make_items(30)}
    eos = (248046, 248044)
    for rid in ITEM_IDS:
        row = run_item(model, tok, items[rid], cap[rid],
                       full3 if rid == 3 else set(), eos)
        row["p1_identity_ok"] = (
            None if SMOKE else
            row["gen_sha256"] == pj["preconditions"]
            ["P1_trajectory_identity_sha256"][str(rid)])
        with open(rows_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        fx_ok = all(x["top1_identical"] and x["max_abs_diff"] == 0.0
                    for x in row["fixture"])
        print(f"[ls] #{rid} cap={row['n_captured']} "
              f"p1={row['p1_identity_ok']} fixture_bitexact={fx_ok} "
              f"gen={row['out_tokens']} {row['wall_s']:.0f}s",
              flush=True)
    summ = {"start": START, "completion_commit": completion_commit(),
            "smoke": SMOKE, "prereg": PREREG,
            "frozen_sidecar_shas": {i: frozen[i]["gen_sha256"]
                                    for i in ITEM_IDS}}
    with open(os.path.join(OUT, "loopstate_summary.json"), "w") as f:
        f.write(json.dumps(summ, indent=1) + "\n")
    print("[ls] done", flush=True)


if __name__ == "__main__":
    main()

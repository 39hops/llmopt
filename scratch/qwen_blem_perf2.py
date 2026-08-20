"""BLEM-DECODE-PERF phase-2 probe (observation-only): the exact
cross-tower reproduction. Build BLe, prefill the frozen item-0
prefix to the HOMEO boundary, serialize the state to CPU with value
statistics per tensor (fraction of fp32 subnormals, min |nonzero|,
max |x|, mean |x|); free BLe; build BLem; restore that BLe state
and decode 64 tokens (the HOMEO HOT configuration, timed); then
prefill the same prefix natively under BLem and decode 64 (the RH
configuration, timed); value stats for the native BLem state too.
Receipt: logs/qwenblemperf/perf2_receipt.json.

    .venv/bin/python scratch/qwen_blem_perf2.py              (3080)
"""
import importlib.util
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

os.environ.setdefault("STEP", "n/a")
ROOT_ART = os.path.expanduser(os.environ.get("ART_ROOT",
                                             "~/qwen_whole0t"))
OUT = "logs/qwenblemperf"
FROZEN = "logs/qweneffort2_probe/traj_xhigh_0.json"
PREFIX_N = 569
DECODE_N = 64
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def value_stats(past, st):
    import torch
    out = {}
    for p, t in st(past):
        if not t.is_floating_point() or t.numel() == 0:
            continue
        x = t.detach().float().abs()
        nz = x[x > 0]
        sub = float((nz < 1.17549435e-38).float().mean()) \
            if nz.numel() else 0.0
        out[p] = {"numel": t.numel(),
                  "frac_zero": float((x == 0).float().mean()),
                  "frac_subnormal_of_nonzero": sub,
                  "min_abs_nonzero": float(nz.min()) if nz.numel()
                  else None,
                  "max_abs": float(x.max()),
                  "mean_abs": float(x.mean())}
    return out


def main():
    import torch
    ha = _load("qwen_homeo_actuator", "scratch/qwen_homeo_actuator.py")
    bp = _load("qwen_blem_perf", "scratch/qwen_blem_perf.py")
    os.makedirs(OUT, exist_ok=True)
    rcpt_path = os.path.join(OUT, "perf2_receipt.json")
    if os.path.exists(rcpt_path):
        raise SystemExit(f"REFUSING: {rcpt_path} exists")
    START = start_provenance(
        ["scratch/qwen_blem_perf2.py", "scratch/qwen_blem_perf.py",
         "scratch/qwen_homeo_actuator.py",
         "scratch/qwen_tower_ladder.py", "llmopt/lab/qcuda_tower.py"],
        artifacts={"BLem": os.path.join(ROOT_ART, "BLem"),
                   "BLe": os.path.join(ROOT_ART, "BLe")})
    frozen = json.load(open(FROZEN))["gen_token_ids"]

    os.environ["ART_DIR"] = os.path.join(ROOT_ART, "BLe")
    tlA = _load("tl_ble_p2", "scratch/qwen_tower_ladder.py")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tlA.VDIR)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "warm"}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=True, reasoning_effort="xhigh")
    pids = tok(text)["input_ids"]

    model, plan, routes, n = tlA.build_tower()
    torch.cuda.synchronize()
    print("[b2] BLe tower up", flush=True)
    past, cur = ha.prefill(model, pids + frozen[:PREFIX_N])
    cpu_ble = ha._move_state(past, "cpu")
    stats_ble = value_stats(cpu_ble, bp.state_tensors)
    # BLe-native decode rate as the same-session baseline
    ids_b, wall = bp.decode(model, past, cur, DECODE_N)
    print(f"[b2] BLe native {DECODE_N/wall:.2f} tok/s", flush=True)
    ble_native = {"tok_s": DECODE_N / wall}
    del past, model
    torch.cuda.empty_cache()
    print("[b2] BLe freed", flush=True)

    os.environ["ART_DIR"] = os.path.join(ROOT_ART, "BLem")
    tlB = _load("tl_blem_p2", "scratch/qwen_tower_ladder.py")
    model, plan, routes, n = tlB.build_tower()
    torch.cuda.synchronize()
    print("[b2] BLem tower up", flush=True)
    # warm
    pw, cw = ha.prefill(model, pids + frozen[:64])
    bp.decode(model, pw, cw, 8)
    del pw
    torch.cuda.empty_cache()

    # HOT configuration: BLe state under BLem weights
    past_x = ha._move_state(cpu_ble, "cuda")
    ids_x, wall = bp.decode(model, past_x, cur, DECODE_N)
    print(f"[b2] cross (BLe state, BLem weights) "
          f"{DECODE_N/wall:.2f} tok/s", flush=True)
    cross = {"tok_s": DECODE_N / wall}
    del past_x
    torch.cuda.empty_cache()

    # RH configuration: native BLem state, same prefix
    past_m, cur_m = ha.prefill(model, pids + frozen[:PREFIX_N])
    cpu_blem = ha._move_state(past_m, "cpu")
    stats_blem = value_stats(cpu_blem, bp.state_tensors)
    ids_m, wall = bp.decode(model, past_m, cur_m, DECODE_N)
    print(f"[b2] BLem native {DECODE_N/wall:.2f} tok/s", flush=True)
    blem_native = {"tok_s": DECODE_N / wall}

    # stats deltas worth printing: any tensor whose subnormal or
    # min-abs profile differs strongly between the two states
    flagged = []
    for k in stats_ble:
        a, b = stats_ble[k], stats_blem.get(k)
        if b is None:
            continue
        if (a["frac_subnormal_of_nonzero"] > 1e-4
                or b["frac_subnormal_of_nonzero"] > 1e-4
                or (a["max_abs"] and b["max_abs"]
                    and max(a["max_abs"], b["max_abs"])
                    > 1e3 * max(1e-30, min(a["max_abs"],
                                           b["max_abs"])))):
            flagged.append({"path": k, "ble": a, "blem": b})
    print(f"[b2] flagged value-profile tensors: {len(flagged)}",
          flush=True)
    for f_ in flagged[:12]:
        print(f"[b2]   {f_['path']} ble_sub="
              f"{f_['ble']['frac_subnormal_of_nonzero']:.2e} "
              f"blem_sub={f_['blem']['frac_subnormal_of_nonzero']:.2e}"
              f" ble_max={f_['ble']['max_abs']:.3e} "
              f"blem_max={f_['blem']['max_abs']:.3e}", flush=True)

    rcpt = {"note": "BLEM-DECODE-PERF phase-2: exact cross-tower "
                    "reproduction with state value statistics",
            "start": START, "completion_commit": completion_commit(),
            "prefix_n": PREFIX_N, "decode_n": DECODE_N,
            "ble_native": ble_native, "cross_hot_config": cross,
            "blem_native": blem_native,
            "n_flagged": len(flagged), "flagged": flagged,
            "stats_ble": stats_ble, "stats_blem": stats_blem}
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rcpt, indent=1) + "\n")
    print(f"[b2] receipt -> {rcpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

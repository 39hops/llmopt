"""BLEM-DECODE-PERF phase-3 probe (observation-only): is the 7x
emergent with decode length or host-RAM pressure? BLem tower, the
HOMEO item-0 boundary state (cross: BLe-generated), decode 1024
tokens with per-64-token segment rates, two arms:
  A. cross state, no ballast
  B. cross state, WITH ballast — two extra deep copies of the CPU
     state resident (the HOMEO Phase-B RAM condition: three saved
     states on a 16GB host)
Then C: native BLem state, no ballast, 1024 tokens (control).
Receipt: logs/qwenblemperf/perf3_receipt.json.

    .venv/bin/python scratch/qwen_blem_perf3.py              (3080)
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
TOTAL_N = int(os.environ.get("TOTAL_N", "1024"))
SEG = 64
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def decode_segments(model, past, cur, total, seg):
    import torch
    rates = []
    with torch.inference_mode():
        t = torch.tensor([[cur]], device="cuda")
        done = 0
        while done < total:
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(min(seg, total - done)):
                o = model(input_ids=t, past_key_values=past,
                          use_cache=True)
                past = o.past_key_values
                nxt = int(torch.argmax(o.logits[0, -1]).item())
                t = torch.tensor([[nxt]], device="cuda")
            torch.cuda.synchronize()
            n = min(seg, total - done)
            rates.append(round(n / (time.time() - t0), 2))
            done += n
            print(f"[b3]   seg@{done}: {rates[-1]} tok/s", flush=True)
    return rates


def main():
    import torch
    ha = _load("qwen_homeo_actuator", "scratch/qwen_homeo_actuator.py")
    os.makedirs(OUT, exist_ok=True)
    rcpt_path = os.path.join(OUT, f"perf3_receipt_{TOTAL_N}"
                             + ("_rn" if os.environ.get("RESTORE_NATIVE") == "1" else "") + ".json")
    if TOTAL_N == 1024 and not os.path.exists(rcpt_path):
        rcpt_path = os.path.join(OUT, "perf3_receipt.json")
    if os.path.exists(rcpt_path):
        raise SystemExit(f"REFUSING: {rcpt_path} exists")
    START = start_provenance(
        ["scratch/qwen_blem_perf3.py",
         "scratch/qwen_homeo_actuator.py",
         "scratch/qwen_tower_ladder.py", "llmopt/lab/qcuda_tower.py"],
        artifacts={"BLem": os.path.join(ROOT_ART, "BLem"),
                   "BLe": os.path.join(ROOT_ART, "BLe")})
    frozen = json.load(open(FROZEN))["gen_token_ids"]

    os.environ["ART_DIR"] = os.path.join(ROOT_ART, "BLe")
    tlA = _load("tl_ble_p3", "scratch/qwen_tower_ladder.py")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tlA.VDIR)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "warm"}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=True, reasoning_effort="xhigh")
    pids = tok(text)["input_ids"]
    model, plan, routes, n = tlA.build_tower()
    past, cur = ha.prefill(model, pids + frozen[:PREFIX_N])
    cpu_ble = ha._move_state(past, "cpu")
    del past, model
    torch.cuda.empty_cache()
    print("[b3] BLe state captured, tower freed", flush=True)

    os.environ["ART_DIR"] = os.path.join(ROOT_ART, "BLem")
    tlB = _load("tl_blem_p3", "scratch/qwen_tower_ladder.py")
    model, plan, routes, n = tlB.build_tower()
    torch.cuda.synchronize()
    print("[b3] BLem tower up", flush=True)
    pw, cw = ha.prefill(model, pids + frozen[:64])
    decode_segments(model, pw, cw, 8, 8)
    del pw
    torch.cuda.empty_cache()

    if os.environ.get("SKIP_CROSS") == "1":
        rates_a = None
    else:
        print("[b3] ARM A: cross, no ballast", flush=True)
        past_a = ha._move_state(cpu_ble, "cuda")
        rates_a = decode_segments(model, past_a, cur, TOTAL_N, SEG)
        del past_a
        torch.cuda.empty_cache()

    if os.environ.get("SKIP_BALLAST") == "1":
        rates_b = None
    else:
        print("[b3] ARM B: cross, ballast x2 resident", flush=True)
        ballast = [ha._move_state(cpu_ble, "cpu") for _ in range(2)]
        past_b = ha._move_state(cpu_ble, "cuda")
        rates_b = decode_segments(model, past_b, cur, TOTAL_N, SEG)
        del past_b, ballast
        torch.cuda.empty_cache()

    if os.environ.get("RESTORE_NATIVE") == "1":
        print("[b3] ARM C': native state ROUNDTRIPPED", flush=True)
        past_c, cur_c = ha.prefill(model, pids + frozen[:PREFIX_N])
        cpu_n = ha._move_state(past_c, "cpu")
        del past_c
        torch.cuda.empty_cache()
        past_c = ha._move_state(cpu_n, "cuda")
    else:
        print("[b3] ARM C: native, no ballast", flush=True)
        past_c, cur_c = ha.prefill(model, pids + frozen[:PREFIX_N])
    rates_c = decode_segments(model, past_c, cur_c, TOTAL_N, SEG)

    rcpt = {"note": "BLEM-DECODE-PERF phase-3: length scaling + "
                    "RAM-pressure ballast on the cross-tower state",
            "start": START, "completion_commit": completion_commit(),
            "prefix_n": PREFIX_N, "total_n": TOTAL_N, "seg": SEG,
            "rates_cross": rates_a,
            "rates_cross_ballast": rates_b,
            "rates_native": rates_c}
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rcpt, indent=1) + "\n")
    print(f"[b3] receipt -> {rcpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

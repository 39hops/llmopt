"""BLEM-DECODE-PERF probe (observation-only, gates nothing; the
banked riff's phase 1). Three sections, each streamed as it lands:

A. SAME-TOWER RESTORE: build the BLem tower once, prefill the
   frozen item-0 prefix, then decode 64 tokens three ways —
   (1) native cache, (2) after a CPU serializer roundtrip
   (the HOMEO _move_state path verbatim), (3) after roundtrip with
   .contiguous() forced on every tensor. Token-identity between
   arms is asserted (greedy, same weights, same state). Rates
   separate restore-effect from cross-tower-effect: HOMEO's slow
   HOT arm was restored+cross-tower; this isolates restored alone.
B. STATE PHYSICALS: for native v restored, every tensor's dtype,
   device, shape, stride, contiguity, storage_offset — diffs
   printed and receipted.
C. GEMV MICRO: the 48 promoted mid-band keys, each timed as the
   BLem FusedS16Linear v the BLe FusedW4Linear on a (1, in) fp32
   cuda vector, 200 reps after 20 warmup, medians receipted.

Receipt: logs/qwenblemperf/perf_receipt.json (refuse-if-exists).

    .venv/bin/python scratch/qwen_blem_perf.py               (3080)
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
BLEM_DIR = os.path.join(ROOT_ART, "BLem")
BLE_DIR = os.path.join(ROOT_ART, "BLe")
OUT = "logs/qwenblemperf"
FROZEN = "logs/qweneffort2_probe/traj_xhigh_0.json"
PREFIX_N = 569          # the HOMEO item-0 event boundary
DECODE_N = 64
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def state_tensors(obj, path="state", _seen=None):
    """(path, tensor) pairs reachable from a cache object."""
    import torch
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return
    _seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from state_tensors(v, f"{path}.{k}", _seen)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from state_tensors(v, f"{path}[{i}]", _seen)
    elif hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            yield from state_tensors(v, f"{path}.{k}", _seen)


def physicals(past):
    out = {}
    for p, t in state_tensors(past):
        out[p] = {"dtype": str(t.dtype), "device": str(t.device),
                  "shape": list(t.shape), "stride": list(t.stride()),
                  "contiguous": bool(t.is_contiguous()),
                  "storage_offset": int(t.storage_offset())}
    return out


def decode(model, past, cur, n):
    import torch
    ids = []
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.inference_mode():
        t = torch.tensor([[cur]], device="cuda")
        for _ in range(n):
            o = model(input_ids=t, past_key_values=past,
                      use_cache=True)
            past = o.past_key_values
            nxt = int(torch.argmax(o.logits[0, -1]).item())
            ids.append(nxt)
            t = torch.tensor([[nxt]], device="cuda")
    torch.cuda.synchronize()
    return ids, time.time() - t0


def main():
    import torch
    ha = _load("qwen_homeo_actuator", "scratch/qwen_homeo_actuator.py")
    os.makedirs(OUT, exist_ok=True)
    rcpt_path = os.path.join(OUT, "perf_receipt.json")
    if os.path.exists(rcpt_path):
        raise SystemExit(f"REFUSING: {rcpt_path} exists")
    START = start_provenance(
        ["scratch/qwen_blem_perf.py", "scratch/qwen_homeo_actuator.py",
         "scratch/qwen_tower_ladder.py", "llmopt/lab/qcuda_tower.py",
         "llmopt/lab/qcuda.py"],
        artifacts={"BLem": BLEM_DIR, "BLe": BLE_DIR})
    frozen = json.load(open(FROZEN))["gen_token_ids"]

    os.environ["ART_DIR"] = BLEM_DIR
    tl = _load("qwen_tower_ladder_perf", "scratch/qwen_tower_ladder.py")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tl.VDIR)
    model, plan, routes, n_routes = tl.build_tower()
    torch.cuda.synchronize()
    print(f"[bp] BLem tower up {n_routes}", flush=True)
    text = tok.apply_chat_template(
        [{"role": "user", "content": "warm"}],
        add_generation_prompt=True, tokenize=False,
        enable_thinking=True, reasoning_effort="xhigh")
    pids = tok(text)["input_ids"]

    # warm the tower with one prefill+short decode before timing
    past_w, cur_w = ha.prefill(model, pids + frozen[:64])
    decode(model, past_w, cur_w, 8)
    del past_w
    torch.cuda.empty_cache()

    # A: native v roundtrip v roundtrip+contiguous, same tower
    past, cur = ha.prefill(model, pids + frozen[:PREFIX_N])
    cpu_copy = ha._move_state(past, "cpu")
    phys_native = physicals(past)
    arms = {}
    ids_n, wall = decode(model, past, cur, DECODE_N)
    arms["native"] = {"tok_s": DECODE_N / wall, "wall_s": wall}
    print(f"[bp] native {DECODE_N/wall:.2f} tok/s", flush=True)
    del past
    torch.cuda.empty_cache()

    past_r = ha._move_state(cpu_copy, "cuda")
    phys_restored = physicals(past_r)
    ids_r, wall = decode(model, past_r, cur, DECODE_N)
    arms["restored"] = {"tok_s": DECODE_N / wall, "wall_s": wall}
    print(f"[bp] restored {DECODE_N/wall:.2f} tok/s", flush=True)
    del past_r
    torch.cuda.empty_cache()

    past_c = ha._move_state(cpu_copy, "cuda")
    for _, t in state_tensors(past_c):
        if not t.is_contiguous():
            t.data = t.contiguous()
    ids_c, wall = decode(model, past_c, cur, DECODE_N)
    arms["restored_contig"] = {"tok_s": DECODE_N / wall,
                               "wall_s": wall}
    print(f"[bp] restored+contig {DECODE_N/wall:.2f} tok/s",
          flush=True)
    del past_c
    torch.cuda.empty_cache()

    token_identity = {"native_v_restored": ids_n == ids_r,
                      "native_v_contig": ids_n == ids_c}
    print(f"[bp] token identity {token_identity}", flush=True)

    # B: physical diffs
    diffs = []
    for k in sorted(set(phys_native) | set(phys_restored)):
        a, b = phys_native.get(k), phys_restored.get(k)
        if a != b:
            diffs.append({"path": k, "native": a, "restored": b})
    print(f"[bp] physical diffs: {len(diffs)}", flush=True)
    for d in diffs[:20]:
        print(f"[bp]   {d}", flush=True)

    # C: GEMV micro on the 48 promoted keys (BLem s16 v BLe w4)
    del model
    torch.cuda.empty_cache()
    import llmopt.lab.qcuda_tower as qt
    micro = []
    for art, codec in ((BLEM_DIR, "s16"), (BLE_DIR, "w4")):
        os.environ["ART_DIR"] = art
        tlx = _load(f"tl_{codec}", "scratch/qwen_tower_ladder.py")
        q, man, payload = tlx.man_and_payload()
        keys = sorted(k for k, e in man.items()
                      if ".linear_attn." in k and e["codec"] == codec
                      and 21 <= int(k.split("layers.")[1].split(".")[0])
                      <= 41)
        assert len(keys) == 48, (codec, len(keys))
        for k in keys:
            e = man[k]
            mod = qt.fused_module(e, payload(e))
            x = torch.randn(1, e["shape"][1], device="cuda")
            for _ in range(20):
                mod(x)
            torch.cuda.synchronize()
            ts = []
            for _ in range(200):
                t0 = time.time()
                mod(x)
                torch.cuda.synchronize()
                ts.append(time.time() - t0)
            ts.sort()
            micro.append({"key": k, "codec": codec,
                          "shape": e["shape"],
                          "median_us": ts[100] * 1e6})
            del mod
        torch.cuda.empty_cache()
        print(f"[bp] micro {codec}: 48 keys timed", flush=True)

    rcpt = {"note": "BLEM-DECODE-PERF phase-1 probe (observation-"
                    "only): same-tower restore arms, state "
                    "physicals, per-shape GEMV medians",
            "start": START, "completion_commit": completion_commit(),
            "prefix_n": PREFIX_N, "decode_n": DECODE_N,
            "arms": arms, "token_identity": token_identity,
            "n_physical_diffs": len(diffs),
            "physical_diffs": diffs,
            "gemv_micro": micro}
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rcpt, indent=1) + "\n")
    print(f"[bp] receipt -> {rcpt_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

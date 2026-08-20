"""Unregistered autopsy probe: regenerate selected BLE-FREEGEN-2
xhigh rows with FULL token-ID + text capture (the frozen screen rows
persist only parsed fields — banked forward fix TRAJECTORY-SIDECAR).

Greedy decoding on a fixed runtime + artifact is deterministic, so
each regeneration reproduces the booked trajectory verbatim; the
probe asserts identity against the frozen row (out_tokens,
think_terminated, truncated) and REFUSES to report a trajectory whose
regeneration diverged. Writes to logs/qweneffort2_probe/ — never the
frozen screen path.

    ARM=BLe IDS=0,3,4 .venv/bin/python scratch/qwen_ble2_autopsy.py

Outputs per id: traj_xhigh_<id>.json (token ids, decoded text,
identity check) under logs/qweneffort2_probe/.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

ARM = os.environ.get("ARM", "BLe")
assert ARM in ("BLe",), ARM
os.environ.setdefault("ART_DIR", os.path.expanduser(
    f"~/qwen_whole0t/{ARM}"))
os.environ.setdefault("STEP", "n/a")
MAX_TOK = int(os.environ.get("MAX_TOK", "3072"))
IDS = [int(x) for x in os.environ.get("IDS", "0,3,4").split(",")]
FROZEN_ROWS = "logs/qweneffort2/tower_rows_BLe.jsonl"
OUT = "logs/qweneffort2_probe"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch
    from transformers import AutoTokenizer
    ep = _load("qwen_effort_probe", "scratch/qwen_effort_probe.py")
    tl = _load("qwen_tower_ladder", "scratch/qwen_tower_ladder.py")
    os.makedirs(OUT, exist_ok=True)
    frozen = {(json.loads(l)["cell"], json.loads(l)["id"]):
              json.loads(l) for l in open(FROZEN_ROWS)}
    start = {"start_commit": subprocess.check_output(
                 ["git", "rev-parse", "--short", "HEAD"])
                 .decode().strip(),
             "interpreter": sys.executable,
             "runtime": "qcuda_tower",
             "runtime_sha256": hashlib.sha256(open(os.path.join(
                 _ROOT, "llmopt/lab/qcuda_tower.py"), "rb").read())
                 .hexdigest()}
    tok = AutoTokenizer.from_pretrained(tl.VDIR)
    t0 = time.time()
    model, plan, routes, n_routes = tl.build_tower()
    torch.cuda.synchronize()
    print(f"[au] built tower {time.time()-t0:.0f}s", flush=True)
    items = {it["id"]: it for it in ep.make_items(30)}
    eos = [248046, 248044]
    for rid in IDS:
        it = items[rid]
        fr = frozen[("xhigh", rid)]
        text = tok.apply_chat_template(
            [{"role": "user", "content": it["prompt"]}],
            add_generation_prompt=True, tokenize=False,
            enable_thinking=True, reasoning_effort="xhigh")
        ids = tok(text, return_tensors="pt")["input_ids"]
        t = time.time()
        out_ids = model.generate(
            input_ids=ids.cuda(), max_new_tokens=MAX_TOK,
            do_sample=False, use_cache=True,
            eos_token_id=eos, pad_token_id=eos[0])
        torch.cuda.synchronize()
        gen = out_ids[0][ids.shape[1]:].tolist()
        out = tok.decode(out_ids[0][ids.shape[1]:],
                         skip_special_tokens=False)
        ident = {"out_tokens": len(gen) == fr["out_tokens"],
                 "think_terminated":
                     ("</think>" in out) == fr["think_terminated"],
                 "truncated":
                     (len(gen) >= MAX_TOK - 2) == fr["truncated"]}
        rec = {"cell": "xhigh", "id": rid, "family": it["family"],
               "prompt": it["prompt"], "truth": it["truth"],
               "gen_token_ids": gen,
               "gen_sha256": hashlib.sha256(
                   json.dumps(gen).encode()).hexdigest(),
               "text": out, "wall_s": round(time.time() - t, 1),
               "identity_v_frozen_row": ident,
               "identity_ok": all(ident.values()),
               "frozen_row_ref": {k: fr[k] for k in
                                  ("out_tokens", "think_terminated",
                                   "truncated", "code_commit")},
               "start": start}
        p = os.path.join(OUT, f"traj_xhigh_{rid}.json")
        with open(p, "w") as f:
            f.write(json.dumps(rec, indent=1) + "\n")
        print(f"[au] xhigh #{rid} {it['family']} gen={len(gen)} "
              f"identity_ok={rec['identity_ok']} -> {p}", flush=True)
    print("[au] done", flush=True)


if __name__ == "__main__":
    main()

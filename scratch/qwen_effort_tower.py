"""Free-generation screen on the QCUDA-TOWER runtime (the
BLE-FREEGEN-2 instrument; spec 2026-08-19-qcuda-tower-runtime).

Same items, cells, and row schema as scratch/qwen_effort_quant.py
(imported generator, same string seed), but the model builds through
scratch/qwen_tower_ladder.build_tower() — the runtime whose
equivalence to rung4 is banked (OBSERVATION QWEN-TOWER-EQUIVALENCE-0)
— and rows land on a FRESH path (logs/qweneffort2/): old and new
runtime rows never merge.

    ARM=BLe ART_DIR=~/qwen_whole0t/BLe .venv/bin/python \
        scratch/qwen_effort_tower.py    (3080)

Rows stream to logs/qweneffort2/tower_rows_<ARM>.jsonl;
summary_tower_<ARM>.json at end.
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

ARM = os.environ["ARM"]
assert ARM in ("BLe",), ARM   # widen deliberately, arm by arm
os.environ.setdefault("ART_DIR", os.path.expanduser(
    f"~/qwen_whole0t/{ARM}"))
os.environ.setdefault("STEP", "n/a")   # ladder module requires it
MAX_TOK = int(os.environ.get("MAX_TOK", "3072"))
N = int(os.environ.get("N_ITEMS", "30"))
CELLS = ("nothink", "xhigh")
OUT = "logs/qweneffort2"
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
    rows_path = os.path.join(OUT, f"tower_rows_{ARM}.jsonl")
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
    print(f"[et] built {ARM} tower in {time.time()-t0:.0f}s "
          f"routes {n_routes}", flush=True)
    items = ep.make_items(N)
    done = set()
    if os.path.exists(rows_path):
        done = {(json.loads(l)["cell"], json.loads(l)["id"])
                for l in open(rows_path)}
    eos = [248046, 248044]
    for cell in CELLS:
        for it in items:
            if (cell, it["id"]) in done:
                continue
            kw = {"enable_thinking": False} if cell == "nothink" \
                else {"enable_thinking": True, "reasoning_effort": cell}
            text = tok.apply_chat_template(
                [{"role": "user", "content": it["prompt"]}],
                add_generation_prompt=True, tokenize=False, **kw)
            ids = tok(text, return_tensors="pt")["input_ids"]
            t = time.time()
            out_ids = model.generate(
                input_ids=ids.cuda(), max_new_tokens=MAX_TOK,
                do_sample=False, use_cache=True,
                eos_token_id=eos, pad_token_id=eos[0])
            torch.cuda.synchronize()
            wall = time.time() - t
            gen = out_ids[0][ids.shape[1]:]
            out = tok.decode(gen, skip_special_tokens=False)
            think, vis = ("", out)
            terminated = "</think>" in out
            if terminated:
                think, vis = out.split("</think>", 1)
            ans = ep.parse_answer(vis)
            ok = bool(ans and ep.check(ans, it["truth"]))
            row = {"arm": ARM, "cell": cell, "id": it["id"],
                   "family": it["family"], "correct": ok,
                   "answer": ans, "truth": it["truth"],
                   "think_terminated": terminated,
                   "think_tokens": (len(tok.encode(think))
                                    if terminated else None),
                   "out_tokens": int(len(gen)),
                   "truncated": bool(len(gen) >= MAX_TOK - 2),
                   "wall_s": round(wall, 1),
                   "tok_s": round(len(gen) / max(wall, 1e-9), 2),
                   "code_commit": start["start_commit"],
                   "runtime": "qcuda_tower",
                   "device_actual": torch.cuda.get_device_name(0),
                   "greedy": True}
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[et] {ARM} {cell} #{it['id']} {it['family']} "
                  f"ok={ok} term={terminated} "
                  f"gen={row['out_tokens']} {wall:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(rows_path)]
    summ = {"arm": ARM, "n_items": N, "max_tok": MAX_TOK,
            "start": start, "route_counts": n_routes,
            "residency_plan": plan, "cells": {}}
    for cell in CELLS:
        cr = [r for r in rows if r["cell"] == cell]
        summ["cells"][cell] = {
            "n": len(cr), "correct": sum(r["correct"] for r in cr),
            "truncated": sum(r["truncated"] for r in cr),
            "n_think_unterminated": sum(
                1 for r in cr if not r.get("think_terminated", True)),
            "mean_wall_s": round(sum(r["wall_s"] for r in cr)
                                 / max(len(cr), 1), 1)}
    with open(os.path.join(OUT, f"summary_tower_{ARM}.json"),
              "w") as f:
        f.write(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ["cells"], indent=1), flush=True)


if __name__ == "__main__":
    main()

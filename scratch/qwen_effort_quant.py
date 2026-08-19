"""QWEN-EFFORT-QUANT-0: does compression damage the deliberation
loop before it damages answers?

Runs the QWEN-EFFORT-0 item set (IDENTICAL items — imported from
scratch/qwen_effort_probe.make_items, same string seed) through a
compressed HOUSE arm on the 3080 CUDA runtime (rung-4 fused tower,
imported from scratch/qwen_cuda_rung4.build — qualification chain
enforced inside), cells {nothink, xhigh}, greedy, sympy
symbolic-equivalence scoring.

FENCES: free-generation accuracy on non-frozen prompts — COLOR for
every registered tree, gates nothing; CUDA leg, so no number here
compares against the Mac CPU X/K quantities (cross-device rule);
single seed, greedy. Books as an OBSERVATION beside QWEN-EFFORT-0's
external-q4 ceiling read.

    ARM=B .venv/bin/python scratch/qwen_effort_quant.py    (on 3080)

Rows stream to logs/qweneffort/quant_rows_<ARM>.jsonl (a killed run
keeps its finished class visible); summary_quant_<ARM>.json at end.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

ARM = os.environ["ARM"]
assert ARM in ("A", "B", "C", "F", "L", "Q", "BLe"), ARM
os.environ.setdefault("ART_DIR", os.path.expanduser(f"~/qwen_whole0t/{ARM}"))
MAX_TOK = int(os.environ.get("MAX_TOK", "3072"))
N = int(os.environ.get("N_ITEMS", "30"))
CELLS = ("nothink", "xhigh")
OUT = "logs/qweneffort"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    import torch
    from transformers import AutoTokenizer
    ep = _load("qwen_effort_probe", "scratch/qwen_effort_probe.py")
    r4 = _load("qwen_cuda_rung4", "scratch/qwen_cuda_rung4.py")
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, f"quant_rows_{ARM}.jsonl")
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    tok = AutoTokenizer.from_pretrained(r4.VDIR)
    t0 = time.time()
    model, trav, n_fused = r4.build()
    torch.cuda.synchronize()
    print(f"[eq] built {ARM} in {time.time()-t0:.0f}s "
          f"({n_fused} fused linears)", flush=True)
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
            # an UNTERMINATED think block books terminated=false with
            # think_tokens=null — never 0 (the whole generation was
            # deliberation; 0 encoded the opposite. EFFORT-QUANT-0's
            # frozen rows carry the old encoding, disclosed by
            # amendment)
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
                   "code_commit": commit,
                   "device_actual": torch.cuda.get_device_name(0),
                   "greedy": True}
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[eq] {ARM} {cell} #{it['id']} {it['family']} "
                  f"ok={ok} think={row['think_tokens']} "
                  f"gen={row['out_tokens']} {wall:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(rows_path)]
    summ = {"arm": ARM, "n_items": N, "max_tok": MAX_TOK,
            "code_commit": commit, "cells": {}}
    for cell in CELLS:
        cr = [r for r in rows if r["cell"] == cell]
        summ["cells"][cell] = {
            "n": len(cr), "correct": sum(r["correct"] for r in cr),
            "truncated": sum(r["truncated"] for r in cr),
            "n_think_unterminated": sum(
                1 for r in cr if not r.get("think_terminated", True)),
            "mean_think_tokens_terminated": round(
                sum(r["think_tokens"] for r in cr
                    if r.get("think_tokens") is not None)
                / max(sum(1 for r in cr
                          if r.get("think_tokens") is not None), 1), 1),
            "mean_wall_s": round(sum(r["wall_s"] for r in cr)
                                 / max(len(cr), 1), 1)}
    with open(os.path.join(OUT, f"summary_quant_{ARM}.json"),
              "w") as f:
        f.write(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ["cells"], indent=1), flush=True)


if __name__ == "__main__":
    main()

"""QWEN-EFFORT-0 overnight probe: does the reasoning_effort knob
change oracle-checked answer quality, and by how much?

SUBJECT: the mlx-community-style q4 conversion of the pinned vendor
checkpoint (~/qwen_mlx_q4, produced by mlx_lm convert — an EXTERNAL
quantization, NOT a house artifact). Chat/free-generation reads are
COLOR by standing law: nothing here gates any registered tree; this
books as an OBSERVATION with the external-artifact fence.

DESIGN: cells = {nothink, low, medium, xhigh} (the vendor template's
full effort vocabulary — the knob is a SYSTEM-TEXT instruction, not
an architectural budget) x N sympy-generated calculus/algebra items
(string-seeded, fresh — never the frozen MODEL-1 payload). Greedy
decode. Per row: sympy symbolic-equivalence verdict on the final
answer (never string match), think-token count, wall seconds.

    N_ITEMS=30 MAX_TOK=3072 .venv/bin/python scratch/qwen_effort_probe.py

Receipts: logs/qweneffort/rows.jsonl (streamed per row — a killed
run keeps its finished class visible) + summary.json at end.
"""
import json
import os
import random
import re
import subprocess
import time

import sympy as sp

X = sp.Symbol("x")
OUT = "logs/qweneffort"
MODEL = os.path.expanduser("~/qwen_mlx_q4")
N = int(os.environ.get("N_ITEMS", "30"))
MAX_TOK = int(os.environ.get("MAX_TOK", "3072"))
CELLS = ("nothink", "low", "medium", "xhigh")


def make_items(n):
    """String-seeded sympy items with machine-checkable answers.
    Three families: derivative, definite-ish integral (indefinite,
    compared modulo constant), expansion."""
    rng = random.Random(f"qwen-effort-0-{n}")
    items = []
    while len(items) < n:
        fam = rng.choice(("diff", "int", "expand"))
        a, b, c, d = (rng.randint(1, 9) for _ in range(4))
        k = rng.randint(2, 4)
        if fam == "diff":
            # second derivative of a product+composition nest
            f = a * X**k * sp.sin(b * X) + c * sp.exp(d * X) * sp.cos(X)
            truth = sp.diff(f, X, 2)
            q = ("Compute the SECOND derivative with respect to x of "
                 f"{sp.sstr(f)}.")
        elif fam == "int":
            f = a * X**k * sp.exp(b * X)
            truth = sp.integrate(f, X)
            q = (f"Compute an antiderivative with respect to x of "
                 f"{sp.sstr(f)} (constant of integration omitted).")
        else:
            f = (a * X + b) * (c * X + d) * (X + k) * (X - a)
            truth = sp.expand(f)
            q = f"Expand fully: {sp.sstr(f)}."
        q += (" Give the final answer on the last line as: "
              "ANSWER: <expression> (a single Python/SymPy-syntax "
              "expression in x, no words).")
        items.append({"id": len(items), "family": fam, "prompt": q,
                      "truth": sp.sstr(truth)})
    return items


def parse_answer(text):
    m = re.findall(r"ANSWER:\s*(.+)", text)
    return m[-1].strip().rstrip(".") if m else None


def check(ans, truth):
    """sympy symbolic equivalence, never string match; int family
    compares modulo an additive constant."""
    try:
        got = sp.sympify(ans, locals={"x": X})
        want = sp.sympify(truth, locals={"x": X})
        d = sp.simplify(got - want)
        if d == 0:
            return True
        return d.is_constant() is True
    except Exception:
        return False


def main():
    from mlx_lm import generate, load
    os.makedirs(OUT, exist_ok=True)
    rows_path = os.path.join(OUT, "rows.jsonl")
    model, tok = load(MODEL)
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    items = make_items(N)
    done = set()
    if os.path.exists(rows_path):
        for line in open(rows_path):
            r = json.loads(line)
            done.add((r["cell"], r["id"]))
    t00 = time.time()
    for cell in CELLS:
        for it in items:
            if (cell, it["id"]) in done:
                continue
            kw = {"enable_thinking": False} if cell == "nothink" \
                else {"enable_thinking": True, "reasoning_effort": cell}
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": it["prompt"]}],
                add_generation_prompt=True, tokenize=False, **kw)
            t0 = time.time()
            out = generate(model, tok, prompt=prompt,
                           max_tokens=MAX_TOK, verbose=False)
            wall = time.time() - t0
            think = ""
            vis = out
            if "</think>" in out:
                think, vis = out.split("</think>", 1)
            ans = parse_answer(vis)
            ok = bool(ans and check(ans, it["truth"]))
            row = {"cell": cell, "id": it["id"], "family": it["family"],
                   "correct": ok, "answer": ans, "truth": it["truth"],
                   "think_tokens": len(tok.encode(think)) if think else 0,
                   "out_tokens": len(tok.encode(out)),
                   "truncated": len(tok.encode(out)) >= MAX_TOK - 2,
                   "wall_s": round(wall, 1), "code_commit": commit,
                   "model_dir": MODEL, "greedy": True}
            with open(rows_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[eff] {cell} #{it['id']} {it['family']} "
                  f"ok={ok} think={row['think_tokens']} "
                  f"{wall:.0f}s", flush=True)
    rows = [json.loads(l) for l in open(rows_path)]
    summ = {"model": MODEL, "n_items": N, "max_tok": MAX_TOK,
            "code_commit": commit,
            "wall_total_s": round(time.time() - t00, 1), "cells": {}}
    for cell in CELLS:
        cr = [r for r in rows if r["cell"] == cell]
        summ["cells"][cell] = {
            "n": len(cr),
            "correct": sum(r["correct"] for r in cr),
            "truncated": sum(r["truncated"] for r in cr),
            "mean_think_tokens": round(
                sum(r["think_tokens"] for r in cr) / max(len(cr), 1), 1),
            "mean_wall_s": round(
                sum(r["wall_s"] for r in cr) / max(len(cr), 1), 1)}
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        f.write(json.dumps(summ, indent=1) + "\n")
    print(json.dumps(summ["cells"], indent=1), flush=True)


if __name__ == "__main__":
    main()

"""Adoption A/B: where do the step model's hints come from?

Three arms, same problems, same budget, same adapter (the promoted
step_lora.pt): (a) oracle hints — every INT_RULE actually runs
(~200ms fork per novel state, today's production), (b) predicted
hints — the frozen layer-15 probe (90.5% exact / 0.979 micro-F1,
RESULTS 2026-07-13 round 5) reading expression + orbital sketch,
(c) no hints ("none" in the prompt).

Pre-registered bar: arm (b) step validity >= arm (a) - 0.2pts and
solves >= (a) - 2 -> the oracle leaves the prompt path. If (c) ties
(a), hints were never load-bearing and BOTH leave (that verdict gets
recorded too — it would re-aim the syndrome-head spec at payoff 3
only).

Mechanism: monkeypatch bench_step_tokens._hints_isolated per arm —
solve_chain and the prompt format stay byte-identical otherwise.

    .venv/bin/python scripts/bench_hints_ab.py [--n-per 12]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LEVELS = (2, 3, 4, 5)
# 9.1M: the original race band. 9.2M+: confirmation bands.
# All outside eval (8.2M), mining (8M), labels (30M), estimator bands.


def _sketch_worker(s: str, q) -> None:
    import sympy as sp
    x = sp.Symbol("x")
    try:
        e = sp.sympify(s, locals={"Integral": sp.Integral, "x": x})
        node = max(e.atoms(sp.Integral), key=sp.count_ops, default=None)
        if node is None:
            q.put("no integral")
            return
        f = node.function
        trig = sorted({sp.sstr(t) for t in f.atoms(sp.sin, sp.cos)})
        expo = sorted({sp.sstr(t) for t in f.atoms(sp.exp)
                       if t.args[0].has(x)})
        logs = sorted({sp.sstr(t) for t in f.atoms(sp.log)})
        roots = sorted({sp.sstr(t) for t in f.atoms(sp.Pow)
                        if t.exp.is_Rational and not t.exp.is_Integer})
        num, den = f.as_numer_denom()
        try:
            deg = sp.degree(num, x) if num.is_polynomial(x) else -1
        except Exception:
            deg = -1
        parts = []
        if trig:
            parts.append("trig " + ", ".join(trig))
        if expo:
            parts.append("exp " + ", ".join(expo))
        if logs:
            parts.append("log " + ", ".join(logs))
        if roots:
            parts.append("root " + ", ".join(roots))
        parts.append(f"polydeg {deg}")
        if den != 1:
            parts.append(f"denom {sp.sstr(den)}")
        q.put("; ".join(parts))
    except Exception:
        q.put("?")


def make_predicted_hinter():
    """Layer-15 probe as a drop-in for _hints_isolated (same
    signature, same memoization discipline)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ck = torch.load("checkpoints/pred_syndromes_l15.pt",
                    weights_only=False)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    enc = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        dtype=torch.float16).to(dev).eval()
    net = torch.nn.Sequential(
        torch.nn.Linear(896, 96), torch.nn.ReLU(),
        torch.nn.Linear(96, 96), torch.nn.ReLU(),
        torch.nn.Linear(96, len(ck["vocab"])))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    cache: dict[str, list[str]] = {}
    ctx = mp.get_context("fork")

    def hinter(cur_s: str, wall: int = 15) -> list[str]:
        if cur_s in cache:
            return cache[cur_s]
        q = ctx.Queue()
        pr = ctx.Process(target=_sketch_worker, args=(cur_s, q))
        pr.start()
        try:
            sketch = q.get(timeout=10)
        except Exception:
            sketch = "?"
        pr.kill()
        pr.join()
        text = f"{cur_s} || orbitals: {sketch}"
        import torch
        with torch.no_grad():
            b = tok([text], return_tensors="pt", truncation=True,
                    max_length=384).to(dev)
            hs = enc.model(input_ids=b["input_ids"],
                           attention_mask=b["attention_mask"],
                           output_hidden_states=True).hidden_states[15]
            m = b["attention_mask"].unsqueeze(-1).float()
            z = ((hs * m).sum(1) / m.sum(1)).float().cpu()
            logits = net((z - ck["mu"]) / ck["sd"])[0]
        out = [ck["vocab"][i] for i in range(len(ck["vocab"]))
               if logits[i] > 0]
        cache[cur_s] = out
        return out

    return hinter


def main(n_per: int, budget: int, seed0: int,
         arm_names: list[str]) -> None:
    import sympy as sp

    import bench_step_tokens as bst
    from bench_step_tokens import _gen_isolated, load, solve_chain

    SEED0 = seed0
    tok, model = load("checkpoints/step_lora.pt")
    oracle = bst._hints_isolated
    all_arms = {
        "oracle": oracle,
        "predicted": None,  # built lazily (loads a second model)
        "none": lambda cur_s, wall=15: [],
    }
    arms = {}
    for a in arm_names:
        if a == "predicted" and all_arms[a] is None:
            all_arms[a] = make_predicted_hinter()
        arms[a] = all_arms[a]
    # same problems for every arm
    probs = []
    for lv in LEVELS:
        for i in range(n_per):
            p = _gen_isolated(lv, SEED0 + 1000 * lv + i)
            if p is not None:
                probs.append((lv, sp.sstr(p._expr)))
    print(f"# hints A/B — {len(probs)} problems, budget {budget}")
    results = {}
    for arm, fn in arms.items():
        bst._hints_isolated = fn
        bst.USE_HINTS = arm != "none"  # solve_chain gates on this now
        bst._HINT_CACHE.clear()
        solved = valid = tried = 0
        t0 = time.time()
        for k, (lv, s) in enumerate(probs):
            ok, _pairs, v, t = solve_chain(tok, model, s, budget,
                                           seed0=SEED0 + k)
            solved += ok
            valid += v
            tried += t
        wall = time.time() - t0
        results[arm] = (solved, 100 * valid / max(tried, 1), wall)
        print(f"{arm:>10s}: solves {solved}/{len(probs)} "
              f"validity {results[arm][1]:.2f}% wall {wall:.0f}s",
              flush=True)
    bst._hints_isolated = oracle
    if "oracle" in results and "predicted" in results:
        o, p = results["oracle"], results["predicted"]
        verdict = ("ADOPT" if p[1] >= o[1] - 0.2 and p[0] >= o[0] - 2
                   else "KEEP ORACLE")
        print(f"\nbar: predicted validity >= oracle-0.2pts AND solves"
              f" >= oracle-2 -> {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per", type=int, default=12)
    ap.add_argument("--budget", type=int, default=384)
    ap.add_argument("--seed0", type=int, default=9_100_000)
    ap.add_argument("--arms", type=str, default="oracle,predicted,none")
    a = ap.parse_args()
    main(a.n_per, a.budget, a.seed0, a.arms.split(","))

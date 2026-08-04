"""MOE-GT-1 arm 0: the full-residency oracle run (pre-reg 2026-08-03).

Runs Qwen3-30B-A3B-4bit FULLY RESIDENT on the 120-item mathgen gate
(seed 1234, the eval_pruned_moe split — disjoint from the router-stats
corpus seed 7) plus the F2 probe prompt, with the router instrumented
for three readouts:

  1. TRUE DEMAND: per-layer expert counts + router mass (RouterStats),
     collected across prefill AND decode — this is the oracle demand
     that arm 2's open-loop recall is judged against.
  2. FIRST-TOUCH ORDER (new instrument): per layer, the global routed
     token index at which each expert was FIRST selected. Descriptive
     this rung (no prefetch claim, per pre-reg fence); it answers
     "what actually comes in first" for any future streaming schedule.
  3. ROUTING TAIL (arm 1's number, computed from 1): per-layer share
     of router mass carried by the top-25% most-massive experts.
     Registered P1: >= 40% mean (heavier than uniform's 25%).

Also books the FULL-MODEL GATE BASELINE (P3's reference point) and the
F2 probe text (P4's non-degenerate reference), since arm 0 pays for
those tokens anyway.

Output: checkpoints/moe_gt1_arm0.json (counts, mass, first_touch,
tail, gate dict) + a row in logs/opus/moe_gt1.jsonl.

Usage: .venv/bin/python scratch/moe_gt1.py   [N_EVAL=120 MAX_TOKENS=96
       PROBE_TOKENS=64 env overrides for smoke tests]
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from llmopt.mathgen.evaluate import SYSTEM, extract_expression
from llmopt.mathgen.problems import make_dataset
from llmopt.moe.router_stats import RouterStats

MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
N_EVAL = int(os.environ.get("N_EVAL", 120))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 96))
PROBE_TOKENS = int(os.environ.get("PROBE_TOKENS", 64))
# the F2 probe, verbatim from the V4 arms (logs/opus/v4_f1d.jsonl)
PROBE = "The three most important ideas in computer science are"
OUT = Path("checkpoints/moe_gt1_arm0.json")
LOG = Path("logs/opus/moe_gt1.jsonl")


def instrument(model):
    """Class-patch every sparse-MoE block to record top-k picks, router
    mass, and FIRST-TOUCH order. Same dispatch pattern as
    scripts/moe_router_stats.py (obj(x) goes through type(obj).__call__,
    so the patch must be class-level with a per-instance registry)."""
    state = {"stats": None, "first": {}, "pos": {}}
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    layer_of = {id(block): li for li, block in moe_layers}
    cls = type(moe_layers[0][1])

    def wrapped(self, x):
        gates = mx.softmax(self.gate(x), axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        if state["stats"] is not None:
            li = layer_of[id(self)]
            flat_i = inds.reshape(-1, k).tolist()
            flat_s = scores.reshape(-1, k).tolist()
            state["stats"].update(li, flat_i, flat_s)
            first = state["first"].setdefault(li, {})
            pos = state["pos"].get(li, 0)
            for t, picks in enumerate(flat_i):
                for e in picks:
                    if e not in first:
                        first[e] = pos + t
            state["pos"][li] = pos + len(flat_i)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    cls.__call__ = wrapped
    n_exp = model.args.num_experts if hasattr(model, "args") else 128
    print(f"[gt1] instrumented {len(moe_layers)} MoE layers "
          f"({cls.__name__}), {n_exp} experts", flush=True)
    return state, n_exp


def tail_share(mass_row, frac=0.25):
    """Share of total router mass carried by the top-`frac` experts."""
    total = sum(mass_row)
    if total <= 0:
        return 0.0
    n_top = max(1, round(frac * len(mass_row)))
    return sum(sorted(mass_row, reverse=True)[:n_top]) / total


def main():
    from mlx_lm import load, generate

    model, tok = load(MODEL)
    state, n_experts = instrument(model)
    stats = RouterStats(n_experts=n_experts)
    state["stats"] = stats

    problems = make_dataset(N_EVAL, seed=1234)
    per_level, n_ok = {}, 0
    t0 = time.time()
    for i, p in enumerate(problems):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": p.prompt}]
        text = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        completion = generate(model, tok, prompt=text, max_tokens=MAX_TOKENS)
        ok = p.check(extract_expression(completion))
        n_ok += ok
        lvl = getattr(p, "level", "?")
        per_level[lvl] = per_level.get(lvl, 0) + int(ok)
        if (i + 1) % 20 == 0:
            print(f"[gt1] gate {i + 1}/{len(problems)} "
                  f"acc {n_ok / (i + 1):.1%} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    gate_s = time.time() - t0
    print(f"[gt1] GATE full model: {n_ok}/{len(problems)} "
          f"per-level {per_level} | {gate_s:.0f}s", flush=True)

    t0 = time.time()
    probe_text = generate(model, tok, prompt=PROBE, max_tokens=PROBE_TOKENS)
    print(f"[gt1] PROBE TEXT (verbatim): {probe_text!r} "
          f"| {time.time() - t0:.0f}s", flush=True)

    tails = {li: tail_share(stats.mass[li]) for li in sorted(stats.mass)}
    mean_tail = sum(tails.values()) / len(tails)
    print(f"[gt1] ROUTING TAIL top-25% mass share: mean {mean_tail:.3f} "
          f"min {min(tails.values()):.3f} max {max(tails.values()):.3f} "
          f"(uniform would be 0.250; P1 bar 0.40)", flush=True)
    ever = {li: sum(1 for c in stats.counts[li] if c > 0)
            for li in sorted(stats.counts)}
    print(f"[gt1] experts EVER touched per layer: "
          f"min {min(ever.values())} max {max(ever.values())} "
          f"of {n_experts}", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "model": MODEL, "n_experts": n_experts, "n_eval": N_EVAL,
        "seed": 1234, "max_tokens": MAX_TOKENS,
        "gate_ok": n_ok, "gate_per_level": per_level,
        "counts": stats.counts, "mass": stats.mass,
        "first_touch": state["first"],
        "tail_top25": tails, "probe_text": probe_text,
    }))
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps({
            "arm": 0, "model": MODEL, "gate_ok": n_ok, "n_eval": N_EVAL,
            "gate_per_level": per_level, "mean_tail_top25": mean_tail,
            "gate_s": gate_s, "probe_text": probe_text,
        }) + "\n")
    print(f"[gt1] saved {OUT} + log row", flush=True)


if __name__ == "__main__":
    main()

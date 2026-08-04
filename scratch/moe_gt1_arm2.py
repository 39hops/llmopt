"""MOE-GT-1 arm 2: residency replay at 50% / 25% / 12.5% (pre-reg
2026-08-03).

Keep-sets are TOP-DEMAND from arm 0 (per-layer expert selection counts
in checkpoints/moe_gt1_arm0.json — the declared rule). For each
residency fraction, the router is masked (kept experts only) and we
measure, per the pre-reg:

  (a) OPEN-LOOP recall — computed offline from arm-0 TRUE demand:
      the fraction of arm-0 routed picks that land inside the
      keep-set (count-weighted). No model run needed.
  (b) CLOSED-LOOP recall — measured DURING the masked run: per token,
      the unmasked top-8 (what the full router wants, from the same
      gate logits) intersected with the keep-set, /8. This is the F2
      instrument, now with a ground-truth twin.
  (c) GATE at N=120 (seed 1234, same split as arm 0) — P3's cliff.
  (d) F2 PROBE text — P4's degeneracy readout.

P2 registered: |closed - open| >= 0.10 at 25% residency.
P3 registered: solves within 1.5 sigma (~7) of 64/120 at 50%;
              collapse >= 3 sigma (~15) at 12.5%.
P4 registered: probe non-degenerate at 50%; either answer books at
              12.5%.

Output: per-arm rows in logs/opus/moe_gt1.jsonl.

Usage: .venv/bin/python scratch/moe_gt1_arm2.py   [N_EVAL, MAX_TOKENS,
       PROBE_TOKENS, FRACS="0.5,0.25,0.125" env overrides]
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

MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
# MOE-GT-2-D4 knobs: ARM0 selects the demand log the keep-sets come
# from; KINDS/LEVELS select the gate corpus (defaults preserve the
# certified math arms byte-for-byte).
ARM0 = Path(os.environ.get("ARM0", "checkpoints/moe_gt1_arm0.json"))
KINDS = tuple(k for k in os.environ.get("KINDS", "").split(",") if k) or None
LEVELS = tuple(
    int(x) for x in os.environ.get("LEVELS", "").split(",") if x) or None
N_EVAL = int(os.environ.get("N_EVAL", 120))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 96))
PROBE_TOKENS = int(os.environ.get("PROBE_TOKENS", 64))
FRACS = [float(f) for f in os.environ.get("FRACS", "0.5,0.25,0.125").split(",")]
SEED = int(os.environ.get("SEED", 1234))
PERPROB = os.environ.get("PERPROB", "") == "1"
PROBE = "The three most important ideas in computer science are"
LOG = Path("logs/opus/moe_gt1.jsonl")
PERPROB_LOG = Path("logs/opus/moe_gt1_perprob.jsonl")


def keep_sets_from_counts(counts, frac, top_k):
    """Per-layer keep-sets at fraction `frac`. RULE env selects the
    rule: 'top' (default) = top-demand by arm-0 selection count;
    'random' = uniform random per layer (RULESEED-seeded, the generic-
    sparsity control for the crest claim); 'anti' = bottom-demand.
    Floor at top_k so the masked router always has a full slate."""
    import random as _random

    rule = os.environ.get("RULE", "top")
    rng = _random.Random(f"gt1-rule-{os.environ.get('RULESEED', '0')}")
    keep = {}
    for li, row in counts.items():
        n = len(row)
        n_keep = max(top_k, round(frac * n))
        if rule == "random":
            order = list(range(n))
            rng.shuffle(order)
        elif rule == "anti":
            order = sorted(range(n), key=lambda e: row[e])
        else:
            order = sorted(range(n), key=lambda e: -row[e])
        keep[int(li)] = set(order[:n_keep])
    return keep


def open_loop_recall(counts, keep):
    """Count-weighted fraction of arm-0 TRUE demand inside the keep-set."""
    hit = tot = 0
    for li, row in counts.items():
        kept = keep[int(li)]
        for e, c in enumerate(row):
            tot += c
            if e in kept:
                hit += c
    return hit / tot


def instrument(model, keep):
    """Class-patch: masked routing (kept experts only) + closed-loop
    recall (unmasked top-k of the SAME logits vs the keep-set)."""
    state = {"hits": 0, "slots": 0}
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    masks, keepsets = {}, {}
    for li, block in moe_layers:
        kept = keep[li]
        n_exp = block.gate.weight.shape[0]
        assert len(kept) >= block.top_k
        masks[id(block)] = mx.array(
            [0.0 if e in kept else float("-inf") for e in range(n_exp)])
        keepsets[id(block)] = kept
    cls = type(moe_layers[0][1])
    original = cls.__call__

    def wrapped(self, x):
        logits = self.gate(x)
        k = self.top_k
        # closed-loop recall: what the UNMASKED router wants
        want = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
        kept = keepsets[id(self)]
        for picks in want.reshape(-1, k).tolist():
            state["slots"] += k
            state["hits"] += sum(1 for e in picks if e in kept)
        # actual routing: masked
        gates = mx.softmax(logits + masks[id(self)], axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    cls.__call__ = wrapped

    def restore():
        cls.__call__ = original

    return state, restore


def run_gate(model, tok, problems, frac, state=None):
    from mlx_lm import generate

    per_level, n_ok, rows = {}, 0, []
    for i, p in enumerate(problems):
        # per-problem closed recall: snapshot the recall counters around
        # each problem (CHURN-JUDGE-1 instrument; deploy-observable,
        # oracle-free — as are parse success and completion length)
        h0, s0 = (state["hits"], state["slots"]) if state else (0, 0)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": p.prompt}]
        text = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        completion = generate(model, tok, prompt=text, max_tokens=MAX_TOKENS)
        expr = extract_expression(completion)
        ok = p.check(expr)
        n_ok += ok
        lvl = getattr(p, "level", "?")
        per_level[lvl] = per_level.get(lvl, 0) + int(ok)
        from llmopt.mathgen.problems import parse_answer
        row = {"seed": SEED, "frac": frac, "idx": i, "level": lvl,
               "ok": bool(ok),
               "parsed": bool(expr) and parse_answer(expr) is not None,
               "gen_len": len(completion)}
        if state and state["slots"] > s0:
            row["recall"] = round(
                (state["hits"] - h0) / (state["slots"] - s0), 4)
        rows.append(row)
        if (i + 1) % 40 == 0:
            print(f"[gt1-2]   gate {i + 1}/{len(problems)} "
                  f"acc {n_ok / (i + 1):.1%}", flush=True)
    if PERPROB:
        with PERPROB_LOG.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return n_ok, per_level


def main():
    from mlx_lm import load, generate

    arm0 = json.loads(ARM0.read_text())
    counts = arm0["counts"]
    model, tok = load(MODEL)
    top_k = next(
        layer.mlp.top_k for layer in model.model.layers
        if hasattr(layer.mlp, "top_k"))
    kw = {}
    if KINDS:
        kw["kinds"] = KINDS
    if LEVELS:
        kw["levels"] = LEVELS
    problems = make_dataset(N_EVAL, seed=SEED, **kw)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    for frac in FRACS:
        # frac 1.0 = the paired FULL baseline (keep-all mask is a no-op
        # routing-wise; the recall instrument still runs, recall == 1)
        keep = keep_sets_from_counts(counts, frac, top_k)
        ol = open_loop_recall(counts, keep)
        n_keep = sum(len(v) for v in keep.values()) / len(keep)
        print(f"[gt1-2] === seed {SEED} frac {frac} "
              f"rule {os.environ.get('RULE', 'top')} | keep {n_keep:.0f}/128 "
              f"per layer | open-loop recall {ol:.4f} ===", flush=True)
        state, restore = instrument(model, keep)
        try:
            t0 = time.time()
            n_ok, per_level = run_gate(model, tok, problems, frac,
                                       state=state)
            gate_s = time.time() - t0
            probe_text = generate(
                model, tok, prompt=PROBE, max_tokens=PROBE_TOKENS)
            cl = state["hits"] / max(state["slots"], 1)
        finally:
            restore()
        print(f"[gt1-2] frac {frac} GATE {n_ok}/{len(problems)} "
              f"per-level {per_level} | closed-loop recall {cl:.4f} "
              f"(open {ol:.4f}, gap {abs(cl - ol):.4f}) | {gate_s:.0f}s",
              flush=True)
        print(f"[gt1-2] PROBE TEXT (verbatim): {probe_text!r}", flush=True)
        with LOG.open("a") as f:
            f.write(json.dumps({
                "arm": 2, "seed": SEED, "frac": frac, "n_eval": N_EVAL,
                "arm0": str(ARM0), "kinds": KINDS, "levels": LEVELS,
                "gate_ok": n_ok, "gate_per_level": per_level,
                "open_recall": ol, "closed_recall": cl,
                "gap": abs(cl - ol), "gate_s": gate_s,
                "probe_text": probe_text,
            }) + "\n")
    print("[gt1-2] all fractions done", flush=True)


if __name__ == "__main__":
    main()

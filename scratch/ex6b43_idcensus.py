"""EX6-B43 identity census (registered observable (a) of VERDICT
EX6-DEPTH-1): WHICH outside-keepset expert(s) does the B43 router
natively select at z1 — and what replacement enters under the mask?

OUTCOME-BLIND by construction: the NONE predicate (nothing masked),
native generation, NO oracle call anywhere — the driver runs
generation only long enough to reach z1 at every MoE block, i.e.
max_tokens=2 per problem (prefill + z1 routing happens inside the
first two generated steps' calls; z1 routing occurs on the first
T=1 call). No gate result exists to leak.

Per problem and per block-position in {41, 43, 46}, logs at phase
z1: the top-16 expert ids ranked by router logits, which of the
top-8 fall outside the named-80 keepset, and the would-be
replacement (the highest-ranked KEPT expert not already in the
top-8 — recoverable from the native ranking because the -inf mask
preserves kept-expert relative order).

Receipts: logs/ex6b43id/{PRE}idcensus.jsonl (refuse-if-exists;
SMOKE=1 -> smoke_ paths, 4 problems, seed 7001).

    .venv/bin/python scratch/ex6b43_idcensus.py               (Mac)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMOKE = os.environ.get("SMOKE") == "1"
PRE = "smoke_" if SMOKE else ""
DIR = Path("logs/ex6b43id")
OUT = DIR / f"{PRE}idcensus.jsonl"

import scratch.moe_gt1_arm2 as m  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SEEDS = [7001] if SMOKE else [7001, 8002, 9003]
N_EVAL = 4 if SMOKE else 120
KEEPSET = "checkpoints/ex3_del_invp.json"
WATCH = (41, 43, 46)


def instrument(model, keep):
    import mlx.core as mx
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    keepsets, t1_count, blk_band = {}, {}, {}
    for bp, (li, block) in enumerate(moe_layers):
        keepsets[id(block)] = keep[li]
        blk_band[id(block)] = bp
    cls = type(moe_layers[0][1])
    original = cls.__call__
    rows = []

    def phase_of(self, n_tokens):
        if n_tokens > 1:
            t1_count[id(self)] = 0
            return "prefill"
        c = t1_count.get(id(self), 0) + 1
        t1_count[id(self)] = c
        return f"z{c}" if c <= 3 else "decode"

    def wrapped(self, x):
        logits = self.gate(x)
        k = self.top_k
        n_tokens = 1
        for d in logits.shape[:-1]:
            n_tokens *= d
        phase = phase_of(self, n_tokens)
        bp = blk_band[id(self)]
        if phase == "z1" and bp in WATCH:
            import mlx.core as mx2
            order = mx2.argsort(logits.reshape(-1))[::-1][:16].tolist()
            kept = keepsets[id(self)]
            top8 = order[:8]
            outside = [e for e in top8 if e not in kept]
            repl = next((e for e in order[8:] if e in kept), None)
            rows.append({"bp": bp, "top16": order,
                         "outside_top8": outside,
                         "replacement_next_kept": repl})
        gates = None
        import mlx.core as mx3
        gates = mx3.softmax(logits, axis=-1, precise=True)
        inds = mx3.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx3.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx3.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        return (y * scores[..., None]).sum(axis=-2)

    cls.__call__ = wrapped

    def restore():
        cls.__call__ = original

    return rows, restore


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    DIR.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/ex6b43_idcensus.py", "scratch/moe_gt1_arm2.py",
         KEEPSET])
    from mlx_lm import generate, load

    from llmopt.mathgen.problems import make_dataset
    model, tok = load(m.MODEL)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(KEEPSET).read_text()).items()}
    f = OUT.open("a")
    t0 = time.time()
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        for i, p in enumerate(problems):
            msgs = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": p.prompt}]
            text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
            rows, restore = instrument(model, keep)
            try:
                generate(model, tok, prompt=text, max_tokens=2)
            finally:
                restore()
            f.write(json.dumps({"seed": seed, "idx": i,
                                "level": getattr(p, "level", None),
                                "z1": rows}) + "\n")
            f.flush()
        print(f"[idc] seed {seed} done {time.time()-t0:.0f}s",
              flush=True)
    f.write(json.dumps({"meta": {
        "note": "EX6-B43 identity census, outcome-blind "
                "(no oracle call; max_tokens=2)",
        "start": START,
        "completion_commit": completion_commit()}}) + "\n")
    f.close()
    print("[idc] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

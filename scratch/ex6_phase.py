"""EX6-PHASE-0 driver: phase-scoped named-80 deletion (PRE-REG
EX6-PHASE-0 in docs/RESULTS.md; machine copy
docs/preregs/ex6-phase-0.json). The frozen moe_gt1_arm2 machinery
is imported, never edited; the ONLY new instrument code is the
phase-scoped router patch below.

MODES (env MODE):
  NONE    router untouched by the mask (paired-full reproduction)
  ALL     mask in every phase (must reproduce the static named-80
          deletion arm exactly — the wrapper's second parity leg)
  PROMPT  mask during prefill AND the prompt_tail step (the one
          1-token call carrying the last prompt token)
  DECODE  mask during generated-token decode steps only

Phase detection is the traj capture's law (scratch/moe_gt1.py,
GT1-TRAJ-CORR): a router call seeing T > 1 tokens is prefill and
resets the per-layer tail flag; the first T == 1 call after that is
prompt_tail; every later T == 1 call is decode. Per-layer flags,
reset naturally by the next prompt's prefill.

Closed-loop recall is tracked ONLY over masked calls (in NONE mode
it stays 0/0 and the row reports recall None).

Env: MODE, ARM (row label), SEED, N_EVAL, PERPROB=1,
LOG/PERPROB_LOG (set by launcher; refuse-if-exists is the
launcher's job — the frozen row emitters append).

    MODE=DECODE ARM=ex6_decode SEED=7001 ... \
        .venv/bin/python scratch/ex6_phase.py                 (Mac)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scratch.moe_gt1_arm2 as m

MODE = os.environ["MODE"]
assert MODE in ("NONE", "ALL", "PROMPT", "DECODE"), MODE
ARM = os.environ["ARM"]
KEEPSET = os.environ.get("KEEPSET", "checkpoints/ex3_del_invp.json")


def instrument_phase(model, keep, mode):
    import mlx.core as mx
    state = {"hits": 0, "slots": 0}
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    masks, keepsets, tail_done = {}, {}, {}
    for li, block in moe_layers:
        kept = keep[li]
        n_exp = block.gate.weight.shape[0]
        assert len(kept) >= block.top_k
        masks[id(block)] = mx.array(
            [0.0 if e in kept else float("-inf") for e in range(n_exp)])
        keepsets[id(block)] = kept
    cls = type(moe_layers[0][1])
    original = cls.__call__

    def phase_of(self, n_tokens):
        if n_tokens > 1:
            tail_done[id(self)] = False
            return "prefill"
        if not tail_done.get(id(self), False):
            tail_done[id(self)] = True
            return "prompt_tail"
        return "decode"

    def mask_now(phase):
        if mode == "NONE":
            return False
        if mode == "ALL":
            return True
        if mode == "PROMPT":
            return phase in ("prefill", "prompt_tail")
        return phase == "decode"

    def wrapped(self, x):
        logits = self.gate(x)
        k = self.top_k
        n_tokens = 1
        for d in logits.shape[:-1]:
            n_tokens *= d
        phase = phase_of(self, n_tokens)
        if mask_now(phase):
            want = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
            kept = keepsets[id(self)]
            for picks in want.reshape(-1, k).tolist():
                state["slots"] += k
                state["hits"] += sum(1 for e in picks if e in kept)
            gates = mx.softmax(logits + masks[id(self)], axis=-1,
                               precise=True)
        else:
            gates = mx.softmax(logits, axis=-1, precise=True)
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


def main():
    from mlx_lm import load

    from llmopt.mathgen.problems import make_dataset

    model, tok = load(m.MODEL)
    problems = make_dataset(m.N_EVAL, seed=m.SEED)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(KEEPSET).read_text()).items()}
    m.LOG.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ex6] arm {ARM} mode {MODE} seed {m.SEED}", flush=True)
    state, restore = instrument_phase(model, keep, MODE)
    try:
        t0 = time.time()
        n_ok, per_level = m.run_gate(model, tok, problems, ARM,
                                     state=state)
        gate_s = time.time() - t0
    finally:
        restore()
    recall = (round(state["hits"] / state["slots"], 4)
              if state["slots"] else None)
    print(f"[ex6] arm {ARM} GATE {n_ok}/{len(problems)} "
          f"per-level {per_level} | masked-phase recall {recall} | "
          f"{gate_s:.0f}s", flush=True)
    with m.LOG.open("a") as f:
        f.write(json.dumps({
            "battery": "ex6", "arm": ARM, "mode": MODE,
            "seed": m.SEED, "n_eval": m.N_EVAL, "gate_ok": n_ok,
            "gate_per_level": per_level,
            "masked_phase_recall": recall, "keepset": KEEPSET,
            "gate_s": gate_s}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

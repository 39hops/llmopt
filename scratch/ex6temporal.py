"""EX6-TEMPORAL-0 driver (frozen pre-launch): isolated temporal
masks z1 v z2 v z3 per PRE-REG EX6-TEMPORAL-0 — is the launch step
a special locus, or the first point on an early-token sensitivity
curve?

Arms (mask predicate over temporal phase labels):
  NONE  nothing masked (anchor: LOC NONE 64/61/66)
  Z1    mask the FIRST T=1 router call per MoE block after prompt
        reset (identical position set to LOC TOKEN1_ONLY; anchor:
        74/66/72 cell-exact)
  Z2    mask the SECOND T=1 call per block (generated z2's routing)
  Z3    mask the THIRD T=1 call per block

Temporal law (extends the frozen ex6_phase/EX6-MED-0-SEMANTICS
phase law): a router call with n_tokens > 1 is the prompt batch
and RESETS a PER-MODULE T=1 counter; each n_tokens == 1 call
increments its own module's counter; count 1/2/3 = z1/z2/z3, later
= decode. The counter is keyed by id(block) exactly like the
frozen tail_done state — 48 independent counters, never one global
counter across the layer pass. Equal dose by construction: each
Zk arm masks ONE temporal position across ALL 48 MoE layers.

Stages, fail-closed in order:
  census     seed 7001 problem idx 2 through the NONE predicate
             with call-shape logging: every MoE block must record
             a prompt-batch reset followed by z1, z2, z3 in order
             then decode, 48/48. Outcome-blind (no gate result read or
             printed). Failure exits 3 before any anchor runs.
  qual       per seed: NONE, Z1. All 6 cells must match the
             booked LOC anchors CELL-EXACT. Any miss exits 3
             BEFORE any Z2/Z3 cell runs or prints (fail-closed
             printing; treatment receipts stay unwritten).
  treatment  per seed: Z2, Z3 (sealed-until-qualification
             receipts: treatment.jsonl + its perprob stream).

Receipts under logs/ex6temporal/ (refuse-if-exists; SMOKE=1 ->
smoke_* paths, 4 problems, seed 7001, anchors not enforced):
  census.json, qual.jsonl, qual_perprob.jsonl      ALWAYS-READABLE
  treatment.jsonl, treatment_perprob.jsonl         SEALED until
                                                   qual PASS (rc 0)

    .venv/bin/python scratch/ex6temporal.py                   (Mac)
"""
import contextlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMOKE = os.environ.get("SMOKE") == "1"
PRE = "smoke_" if SMOKE else ""
DIR = Path("logs/ex6temporal")
CENSUS = DIR / f"{PRE}census.json"
QUAL = DIR / f"{PRE}qual.jsonl"
TREAT = DIR / f"{PRE}treatment.jsonl"
QUAL_PP = DIR / f"{PRE}qual_perprob.jsonl"
TREAT_PP = DIR / f"{PRE}treatment_perprob.jsonl"
# Direct assignment (not setdefault): an inherited env var must not
# silently redirect receipts off the registered paths.
os.environ["LOG"] = str(QUAL)
os.environ["PERPROB"] = "1"
os.environ["PERPROB_LOG"] = str(QUAL_PP)

import scratch.moe_gt1_arm2 as m  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SEEDS = [7001] if SMOKE else [7001, 8002, 9003]
N_EVAL = 4 if SMOKE else 120
KEEPSET = "checkpoints/ex3_del_invp.json"
N_MOE = 48
# Anchors: LOC receipt logs/ex6loc/ex6loc.jsonl (VERDICT EX6-LOC-0).
ANCHORS = {"NONE": {7001: 64, 8002: 61, 9003: 66},
           "Z1": {7001: 74, 8002: 66, 9003: 72}}
PREDS = {"NONE": lambda ph: False,
         "Z1": lambda ph: ph == "z1",
         "Z2": lambda ph: ph == "z2",
         "Z3": lambda ph: ph == "z3"}


def instrument(model, keep, pred):
    """Frozen ex6_phase wrapped-gate math; mask decision = pred(phase),
    phase from a per-module T=1 counter (temporal law above)."""
    import mlx.core as mx
    state = {"hits": 0, "slots": 0}
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    masks, zeros, keepsets, t1_count = {}, {}, {}, {}
    blk_layer = {}
    for li, block in moe_layers:
        kept = keep[li]
        n_exp = block.gate.weight.shape[0]
        assert len(kept) >= block.top_k
        masks[id(block)] = mx.array(
            [0.0 if e in kept else float("-inf") for e in range(n_exp)])
        zeros[id(block)] = mx.array([0.0] * n_exp)
        keepsets[id(block)] = kept
        blk_layer[id(block)] = li
    state["n_moe"] = len(moe_layers)
    cls = type(moe_layers[0][1])
    original = cls.__call__
    census = state["census"] = {}

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
        masked = pred(phase)
        if state.get("log_census"):
            census.setdefault(blk_layer[id(self)], []).append(
                (n_tokens, phase, bool(masked)))
        want = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
        kept = keepsets[id(self)]
        for picks in want.reshape(-1, k).tolist():
            if masked:
                state["slots"] += k
                state["hits"] += sum(1 for e in picks if e in kept)
        gates = mx.softmax(
            logits + (masks[id(self)] if masked else zeros[id(self)]),
            axis=-1, precise=True)
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


def census_verdict(census, n_moe):
    """48/48 modules: prompt-batch reset then z1, z2, z3 in order."""
    per_module = {}
    for li, calls in sorted(census.items()):
        phases = [ph for _, ph, _ in calls]
        prefill_idx = [i for i, (n, ph, _) in enumerate(calls)
                       if ph == "prefill" and n > 1]
        after = phases[prefill_idx[-1] + 1:] if prefill_idx else []
        ok = (bool(prefill_idx) and len(after) >= 4
              and after[0] == "z1" and after[1] == "z2"
              and after[2] == "z3"
              and all(p == "decode" for p in after[3:]))
        per_module[li] = {"ok": ok, "n_calls": len(calls),
                          "first6_after_reset": after[:6]}
    n_ok = sum(1 for v in per_module.values() if v["ok"])
    return {"n_moe_seen": len(per_module), "n_ok": n_ok,
            "required": n_moe,
            "pass": len(per_module) == n_moe and n_ok == n_moe,
            "per_module": per_module}


def run_arm(model, tok, problems, keep, seed, arm, log_path,
            sealed_stdout=None):
    state, restore = instrument(model, keep, PREDS[arm])
    try:
        t0 = time.time()
        if sealed_stdout is not None:
            # run_gate prints running accuracy every 40 problems —
            # a treatment value. Sealed arms redirect it to a
            # sealed-until-qualification file, never the session.
            with sealed_stdout.open("a") as sf, \
                    contextlib.redirect_stdout(sf):
                n_ok, per_level = m.run_gate(model, tok, problems,
                                             f"tmp_{arm}",
                                             state=state)
        else:
            n_ok, per_level = m.run_gate(model, tok, problems,
                                         f"tmp_{arm}", state=state)
    finally:
        restore()
    recall = (round(state["hits"] / state["slots"], 4)
              if state["slots"] else None)
    with log_path.open("a") as f:
        f.write(json.dumps({
            "arm": f"tmp_{arm}", "seed": seed,
            "n_eval": len(problems), "gate_ok": n_ok,
            "gate_per_level": per_level,
            "masked_recall_named80": recall,
            "gate_s": round(time.time() - t0, 1)}) + "\n")
    return n_ok


def main():
    for pth in (CENSUS, QUAL, TREAT, QUAL_PP, TREAT_PP,
                DIR / f"{PRE}treatment_stdout.log"):
        if pth.exists():
            raise SystemExit(f"REFUSING: {pth} exists")
    DIR.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/ex6temporal.py", "scratch/moe_gt1_arm2.py",
         KEEPSET])
    from mlx_lm import load

    from llmopt.mathgen.problems import make_dataset
    model, tok = load(m.MODEL)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(KEEPSET).read_text()).items()}

    # Stage 1: call-position census — outcome-blind, problem idx 2
    # only (its completion runs past 3 generated tokens, so the
    # decode label after z3 is actually witnessed; problem 0's is
    # exactly 3 T=1 calls and cannot witness it). Perprob streaming
    # is disabled for this stage so no census outcome row lands in
    # the always-readable qual perprob stream.
    m.SEED = 7001
    probe = make_dataset(N_EVAL, seed=7001)[2:3]
    state, restore = instrument(model, keep, PREDS["NONE"])
    state["log_census"] = True
    perprob_save, m.PERPROB = m.PERPROB, False
    try:
        m.run_gate(model, tok, probe, "tmp_census", state=state)
    finally:
        restore()
        m.PERPROB = perprob_save
    verdict = census_verdict(state["census"], state["n_moe"])
    CENSUS.write_text(json.dumps(
        {"start": START, "verdict": verdict}, indent=1))
    print(f"[tmp] census: {verdict['n_ok']}/{verdict['required']} "
          f"modules ok, seen {verdict['n_moe_seen']}", flush=True)
    if not SMOKE and not verdict["pass"]:
        print("[tmp] CENSUS FAIL — no anchor or treatment cell runs",
              flush=True)
        sys.exit(3)

    # Stage 2: qualification anchors (always-readable receipts).
    miss = []
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        m.SEED = seed
        for arm in ("NONE", "Z1"):
            n_ok = run_arm(model, tok, problems, keep, seed, arm,
                           QUAL)
            booked = ANCHORS[arm][seed] if not SMOKE else None
            tag = ""
            if booked is not None and n_ok != booked:
                miss.append((seed, arm, n_ok, booked))
                tag = f"  ANCHOR MISS v booked {booked}"
            print(f"[tmp] seed {seed} {arm}: {n_ok}/{len(problems)}"
                  f"{tag}", flush=True)
    if miss:
        with QUAL.open("a") as f:
            f.write(json.dumps({"meta": {
                "note": "EX6-TEMPORAL-0 QUALIFICATION FAIL",
                "anchor_misses": miss, "start": START,
                "completion_commit": completion_commit()}}) + "\n")
        print(f"[tmp] QUALIFICATION FAIL {len(miss)} anchor cell(s)"
              " — Z2/Z3 do not run", flush=True)
        sys.exit(3)

    # Stage 3: treatment (sealed-until-qualification receipts).
    m.PERPROB_LOG = TREAT_PP
    sealed_out = DIR / f"{PRE}treatment_stdout.log"
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        m.SEED = seed
        for arm in ("Z2", "Z3"):
            run_arm(model, tok, problems, keep, seed, arm, TREAT,
                    sealed_stdout=sealed_out)
            print(f"[tmp] seed {seed} {arm}: done "
                  f"(value sealed to treatment.jsonl)", flush=True)
    with TREAT.open("a") as f:
        f.write(json.dumps({"meta": {
            "note": "EX6-TEMPORAL-0 run meta", "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    print("[tmp] done; qualification PASS, treatment sealed",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

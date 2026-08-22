"""EX6-DEPTH-0 driver (frozen pre-launch): z1 depth-band masks per
PRE-REG EX6-DEPTH-0 — where along the 48 MoE layers does
z1-routing healing live?

Arms (mask predicate over (phase, band position)):
  NONE     nothing masked (anchor: 64/61/66)
  Z1_ALL   z1 masked at ALL 48 MoE blocks (position-identical to
           EX6-TEMPORAL-0's Z1; anchor: 74/66/72)
  Z1_EARLY z1 masked at MoE blocks 0-15 (order of appearance)
  Z1_MID   z1 masked at MoE blocks 16-31
  Z1_LATE  z1 masked at MoE blocks 32-47

Temporal law identical to scratch/ex6temporal.py (per-module T=1
counter keyed by id(block), reset on any n_tokens>1 prompt batch).
Band arms are one temporal position x 16 layers: ONE THIRD of
Z1_ALL's temporal-call dose, mutually exclusive, summing to ALL.
Realized displacement per band is DISCLOSED via per-band
masked_recall (the -SCOPE lesson: call dose is not displacement).

DEMAND CENSUS (registered outcome-blind secondary artifact,
RIFF-LEDGER bank 2026-08-21): during the NONE arms the wrapper
counts, per MoE block and per temporal position z1/z2/z3, native
top-8 slots and how many fall OUTSIDE the named-80 keepset — a
48 x 3 unmasked-router demand map, no oracle involvement in the
counters. Written to demand.json.

Stages, fail-closed in order:
  census     problem idx 2, seed 7001, one short pass PER ARM
             PREDICATE: every MoE block must show the temporal
             law (reset, z1, z2, z3, decode), and the per-block
             masked flags must match the arm's registered band
             exactly (0 for NONE, 48 for Z1_ALL, the named 16
             for each band) at phase z1 and nowhere else.
  qual       per seed: NONE, Z1_ALL — all 6 cells CELL-EXACT
             against the TEMPORAL-0 anchors or exit 3 before any
             band cell runs or prints.
  treatment  per seed: Z1_EARLY, Z1_MID, Z1_LATE
             (sealed-until-qualification receipts + sealed
             stdout redirect).

Receipts under logs/ex6depth/ (refuse-if-exists; SMOKE=1 ->
smoke_* paths, 4 problems, seed 7001, anchors not enforced):
  census.json, demand.json, qual.jsonl, qual_perprob.jsonl
                                                   ALWAYS-READABLE
  treatment.jsonl, treatment_perprob.jsonl,
  treatment_stdout.log                    SEALED until qual PASS

    .venv/bin/python scratch/ex6depth.py                      (Mac)
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
DIR = Path("logs/ex6depth")
CENSUS = DIR / f"{PRE}census.json"
DEMAND = DIR / f"{PRE}demand.json"
QUAL = DIR / f"{PRE}qual.jsonl"
TREAT = DIR / f"{PRE}treatment.jsonl"
QUAL_PP = DIR / f"{PRE}qual_perprob.jsonl"
TREAT_PP = DIR / f"{PRE}treatment_perprob.jsonl"
SEALED_OUT = DIR / f"{PRE}treatment_stdout.log"
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
# Anchors: EX6-TEMPORAL-0 receipt logs/ex6temporal/qual.jsonl
# (identical position set: Z1_ALL == tmp_Z1 == LOC TOKEN1_ONLY).
ANCHORS = {"NONE": {7001: 64, 8002: 61, 9003: 66},
           "Z1_ALL": {7001: 74, 8002: 66, 9003: 72}}
BANDS = {"Z1_EARLY": range(0, 16), "Z1_MID": range(16, 32),
         "Z1_LATE": range(32, 48)}


def pred_for(arm):
    """(phase, band_pos) -> masked. band_pos = order of appearance
    of the MoE block in the layer stack (0..47)."""
    if arm == "NONE":
        return lambda ph, bp: False
    if arm == "Z1_ALL":
        return lambda ph, bp: ph == "z1"
    band = BANDS[arm]
    return lambda ph, bp: ph == "z1" and bp in band


def instrument(model, keep, pred):
    """Frozen ex6temporal wrapped-gate math; mask decision =
    pred(phase, band_pos); optional outcome-blind demand counters."""
    import mlx.core as mx
    state = {"hits": 0, "slots": 0}
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    masks, zeros, keepsets, t1_count = {}, {}, {}, {}
    blk_layer, blk_band = {}, {}
    for bp, (li, block) in enumerate(moe_layers):
        kept = keep[li]
        n_exp = block.gate.weight.shape[0]
        assert len(kept) >= block.top_k
        masks[id(block)] = mx.array(
            [0.0 if e in kept else float("-inf") for e in range(n_exp)])
        zeros[id(block)] = mx.array([0.0] * n_exp)
        keepsets[id(block)] = kept
        blk_layer[id(block)] = li
        blk_band[id(block)] = bp
    state["n_moe"] = len(moe_layers)
    cls = type(moe_layers[0][1])
    original = cls.__call__
    census = state["census"] = {}
    demand = state["demand"] = {}

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
        masked = pred(phase, blk_band[id(self)])
        if state.get("log_census"):
            census.setdefault(blk_band[id(self)], []).append(
                (n_tokens, phase, bool(masked)))
        want = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
        kept = keepsets[id(self)]
        for picks in want.reshape(-1, k).tolist():
            if masked:
                state["slots"] += k
                state["hits"] += sum(1 for e in picks if e in kept)
            if state.get("log_demand") and phase in ("z1", "z2", "z3"):
                key = (blk_band[id(self)], phase)
                tot, out = demand.get(key, (0, 0))
                demand[key] = (tot + k,
                               out + sum(1 for e in picks
                                         if e not in kept))
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


def census_verdict(census, n_moe, arm):
    """Temporal law per module AND masked flags match the arm's
    registered band at z1, nowhere else."""
    want_band = (set() if arm == "NONE"
                 else set(range(n_moe)) if arm == "Z1_ALL"
                 else set(BANDS[arm]))
    per_module = {}
    for bp, calls in sorted(census.items()):
        phases = [ph for _, ph, _ in calls]
        prefill_idx = [i for i, (n, ph, _) in enumerate(calls)
                       if ph == "prefill" and n > 1]
        after = phases[prefill_idx[-1] + 1:] if prefill_idx else []
        law_ok = (bool(prefill_idx) and len(after) >= 4
                  and after[:3] == ["z1", "z2", "z3"]
                  and all(p == "decode" for p in after[3:]))
        masked_phases = {ph for _, ph, mk in calls if mk}
        mask_ok = (masked_phases == ({"z1"} if bp in want_band
                                     else set()))
        per_module[bp] = {"ok": law_ok and mask_ok,
                          "law_ok": law_ok, "mask_ok": mask_ok,
                          "masked_phases": sorted(masked_phases)}
    n_ok = sum(1 for v in per_module.values() if v["ok"])
    return {"arm": arm, "n_moe_seen": len(per_module), "n_ok": n_ok,
            "required": n_moe,
            "pass": len(per_module) == n_moe and n_ok == n_moe,
            "per_module": per_module}


def run_arm(model, tok, problems, keep, seed, arm, log_path,
            sealed_stdout=None, log_demand=False):
    state, restore = instrument(model, keep, pred_for(arm))
    if log_demand:
        state["log_demand"] = True
    try:
        t0 = time.time()
        if sealed_stdout is not None:
            with sealed_stdout.open("a") as sf, \
                    contextlib.redirect_stdout(sf):
                n_ok, per_level = m.run_gate(model, tok, problems,
                                             f"dep_{arm}",
                                             state=state)
        else:
            n_ok, per_level = m.run_gate(model, tok, problems,
                                         f"dep_{arm}", state=state)
    finally:
        restore()
    recall = (round(state["hits"] / state["slots"], 4)
              if state["slots"] else None)
    with log_path.open("a") as f:
        f.write(json.dumps({
            "arm": f"dep_{arm}", "seed": seed,
            "n_eval": len(problems), "gate_ok": n_ok,
            "gate_per_level": per_level,
            "masked_recall_named80": recall,
            "gate_s": round(time.time() - t0, 1)}) + "\n")
    return n_ok, state


def main():
    for pth in (CENSUS, DEMAND, QUAL, TREAT, QUAL_PP, TREAT_PP,
                SEALED_OUT):
        if pth.exists():
            raise SystemExit(f"REFUSING: {pth} exists")
    DIR.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/ex6depth.py", "scratch/moe_gt1_arm2.py", KEEPSET])
    from mlx_lm import load

    from llmopt.mathgen.problems import make_dataset
    model, tok = load(m.MODEL)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(KEEPSET).read_text()).items()}

    # Stage 1: per-arm call/mask census — outcome-blind, problem
    # idx 2 only, perprob off.
    m.SEED = 7001
    probe = make_dataset(N_EVAL, seed=7001)[2:3]
    verdicts = {}
    perprob_save, m.PERPROB = m.PERPROB, False
    try:
        for arm in ("NONE", "Z1_ALL", "Z1_EARLY", "Z1_MID",
                    "Z1_LATE"):
            state, restore = instrument(model, keep, pred_for(arm))
            state["log_census"] = True
            try:
                m.run_gate(model, tok, probe, "dep_census",
                           state=state)
            finally:
                restore()
            verdicts[arm] = census_verdict(state["census"],
                                           state["n_moe"], arm)
            print(f"[dep] census {arm}: "
                  f"{verdicts[arm]['n_ok']}/{verdicts[arm]['required']}",
                  flush=True)
    finally:
        m.PERPROB = perprob_save
    CENSUS.write_text(json.dumps(
        {"start": START, "verdicts": {a: v for a, v in
                                      verdicts.items()}}, indent=1))
    if not SMOKE and not all(v["pass"] for v in verdicts.values()):
        print("[dep] CENSUS FAIL — no anchor or band cell runs",
              flush=True)
        sys.exit(3)

    # Stage 2: anchors (always-readable), NONE arms carry the
    # outcome-blind demand counters.
    miss = []
    demand_acc = {}
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        m.SEED = seed
        for arm in ("NONE", "Z1_ALL"):
            n_ok, state = run_arm(
                model, tok, problems, keep, seed, arm, QUAL,
                log_demand=(arm == "NONE"))
            if arm == "NONE":
                for key, (tot, out) in state["demand"].items():
                    t0, o0 = demand_acc.get(key, (0, 0))
                    demand_acc[key] = (t0 + tot, o0 + out)
            booked = ANCHORS[arm][seed] if not SMOKE else None
            tag = ""
            if booked is not None and n_ok != booked:
                miss.append((seed, arm, n_ok, booked))
                tag = f"  ANCHOR MISS v booked {booked}"
            print(f"[dep] seed {seed} {arm}: {n_ok}/{len(problems)}"
                  f"{tag}", flush=True)
    DEMAND.write_text(json.dumps({
        "note": "48x3 native-demand census from the NONE arms, "
                "outcome-blind (router reads only): per (band_pos, "
                "zk): [top8_slots, outside_named80]",
        "start": START,
        "demand": {f"{bp}:{ph}": v for (bp, ph), v in
                   sorted(demand_acc.items())}}, indent=1))
    if miss:
        with QUAL.open("a") as f:
            f.write(json.dumps({"meta": {
                "note": "EX6-DEPTH-0 QUALIFICATION FAIL",
                "anchor_misses": miss, "start": START,
                "completion_commit": completion_commit()}}) + "\n")
        print(f"[dep] QUALIFICATION FAIL {len(miss)} anchor cell(s)"
              " — band arms do not run", flush=True)
        sys.exit(3)

    # Stage 3: band treatment (sealed receipts + sealed stdout).
    m.PERPROB_LOG = TREAT_PP
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        m.SEED = seed
        for arm in ("Z1_EARLY", "Z1_MID", "Z1_LATE"):
            run_arm(model, tok, problems, keep, seed, arm, TREAT,
                    sealed_stdout=SEALED_OUT)
            print(f"[dep] seed {seed} {arm}: done "
                  f"(value sealed to treatment.jsonl)", flush=True)
    with TREAT.open("a") as f:
        f.write(json.dumps({"meta": {
            "note": "EX6-DEPTH-0 run meta", "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    print("[dep] done; qualification PASS, treatment sealed",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

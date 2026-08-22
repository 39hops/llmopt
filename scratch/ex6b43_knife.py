"""EX6-B43-KNIFE-0 driver (frozen pre-launch): exact
activation-component knife at (block-position 43, z1) per PRE-REG
EX6-B43-KNIFE-0 — is the deletion vector d_del sufficient and
necessary for the B43 +20?

Arms (intervention on the single (z1, bp43) MoE call, nowhere
else; everything else is the frozen ex6depth1 native math with
the float32 zeros-add on every gate):
  NATIVE       nothing intervened (anchor: 64/61/66)
  FULL_DIRECT  verbatim masked-path output — identical math to
               the booked dep_Z1_B43 arm (anchor: 70/70/71)
  D_ONLY       native + d_del                (deletion vector)
  NO_D         native + d_entrant + d_renorm (non-deletion pair)
  FULL_SUM     native + all three terms      (additive check)
with d_del = -p_71*E_71(h), d_entrant = p'_r*E_r(h),
d_renorm = sum_C (p'_i - p_i)*E_i(h), terms fp32, the combined
return cast to the native activation dtype (the cast is part of
the registered intervention). Temporal law identical to
scratch/ex6temporal.py (per-module T=1 counter keyed id(block)).

Stages, fail-closed in order:
  census     problem idx 2, seed 7001, one short pass PER ARM:
             every MoE block shows the temporal law (reset, z1,
             z2, z3, decode) and the knife/mask applied flags are
             exactly {(z1, bp43)} for intervened arms, empty for
             NATIVE, nowhere else. Failure exits 3.
  qual       per seed: NATIVE, FULL_DIRECT — all 6 cells
             CELL-EXACT against the EX6-DEPTH-1 receipts or exit
             3 before any knife cell runs or prints.
  treatment  per seed: D_ONLY, NO_D, FULL_SUM (sealed receipts +
             sealed stdout redirect).
hnorms.jsonl (always-readable, outcome-blind): per intervened
problem at the knife call, ||h_pre||, ||native out||,
|delta_direct|, |delta|/||h_pre|| — the -DECOMP-0-SCOPE
residual-stream dose debt.

Receipts under logs/ex6b43knife/ (refuse-if-exists; SMOKE=1 ->
smoke_* paths, 4 problems, seed 7001, anchors not enforced):
  census.json, qual.jsonl, qual_perprob.jsonl, hnorms.jsonl
                                                   ALWAYS-READABLE
  treatment.jsonl, treatment_perprob.jsonl,
  treatment_stdout.log                    SEALED until qual PASS

    .venv/bin/python scratch/ex6b43_knife.py                  (Mac)
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
DIR = Path("logs/ex6b43knife")
CENSUS = DIR / f"{PRE}census.json"
QUAL = DIR / f"{PRE}qual.jsonl"
TREAT = DIR / f"{PRE}treatment.jsonl"
QUAL_PP = DIR / f"{PRE}qual_perprob.jsonl"
TREAT_PP = DIR / f"{PRE}treatment_perprob.jsonl"
HNORMS = DIR / f"{PRE}hnorms.jsonl"
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
TARGET_BP = 43
ARMS = ("NATIVE", "FULL_DIRECT", "D_ONLY", "NO_D", "FULL_SUM")
# Anchors: NATIVE from logs/ex6depth1/qual.jsonl (dep_NONE),
# FULL_DIRECT from logs/ex6depth1/treatment.jsonl (dep_Z1_B43).
ANCHORS = {"NATIVE": {7001: 64, 8002: 61, 9003: 66},
           "FULL_DIRECT": {7001: 70, 8002: 70, 9003: 71}}


def instrument(model, keep, arm, hnorm_sink=None):
    """Frozen ex6depth1 gate math on every call (float32 zeros/mask
    add); the knife applies only at (z1, bp TARGET_BP)."""
    import mlx.core as mx
    state = {"hits": 0, "slots": 0, "knife_calls": 0}
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    masks, zeros, keepsets, t1_count, blk_band = {}, {}, {}, {}, {}
    for bp, (li, block) in enumerate(moe_layers):
        kept = keep[li]
        n_exp = block.gate.weight.shape[0]
        masks[id(block)] = mx.array(
            [0.0 if e in kept else float("-inf") for e in range(n_exp)])
        zeros[id(block)] = mx.array([0.0] * n_exp)
        keepsets[id(block)] = kept
        blk_band[id(block)] = bp
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

    def expert_out(self, x, e):
        inds = mx.full(x.shape[:-1] + (1,), e, dtype=mx.uint32)
        return self.switch_mlp(x, inds)[..., 0, :].astype(mx.float32)

    def norm(v):
        return float(mx.sqrt(mx.sum(v * v)))

    def wrapped(self, x):
        logits = self.gate(x)
        k = self.top_k
        n_tokens = 1
        for d in logits.shape[:-1]:
            n_tokens *= d
        phase = phase_of(self, n_tokens)
        bp = blk_band[id(self)]
        at_knife = (arm != "NATIVE" and phase == "z1"
                    and bp == TARGET_BP)
        if state.get("log_census"):
            census.setdefault(bp, []).append(
                (n_tokens, phase, bool(at_knife)))
        gates = mx.softmax(logits + zeros[id(self)],
                           axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        native = (y * scores[..., None]).sum(axis=-2)
        if not at_knife:
            return native
        state["knife_calls"] += 1
        kept = keepsets[id(self)]
        # masked-path math, verbatim ex6depth1 masked cell
        want = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
        for picks in want.reshape(-1, k).tolist():
            state["slots"] += k
            state["hits"] += sum(1 for e in picks if e in kept)
        mgates = mx.softmax(logits + masks[id(self)],
                            axis=-1, precise=True)
        minds = mx.argpartition(mgates, kth=-k, axis=-1)[..., -k:]
        mscores = mx.take_along_axis(mgates, minds, axis=-1)
        if self.norm_topk_prob:
            mscores = mscores / mx.sum(mscores, axis=-1,
                                       keepdims=True)
        my = self.switch_mlp(x, minds)
        masked_out = (my * mscores[..., None]).sum(axis=-2)
        if arm == "FULL_DIRECT":
            out = masked_out
        else:
            nat_p = {e: float(v) for e, v in zip(
                inds.reshape(-1).tolist(),
                scores.reshape(-1).tolist())}
            msk_p = {e: float(v) for e, v in zip(
                minds.reshape(-1).tolist(),
                mscores.reshape(-1).tolist())}
            d_del = mx.zeros_like(native).astype(mx.float32)
            for e in nat_p:
                if e not in kept:
                    d_del = d_del - nat_p[e] * expert_out(self, x, e)
            d_ent = mx.zeros_like(native).astype(mx.float32)
            for e in msk_p:
                if e not in nat_p:
                    d_ent = d_ent + msk_p[e] * expert_out(self, x, e)
            d_ren = mx.zeros_like(native).astype(mx.float32)
            for e in nat_p:
                if e in msk_p:
                    d_ren = d_ren + (msk_p[e] - nat_p[e]) * \
                        expert_out(self, x, e)
            combo = {"D_ONLY": d_del,
                     "NO_D": d_ent + d_ren,
                     "FULL_SUM": d_del + d_ent + d_ren}[arm]
            out = (native.astype(mx.float32) + combo).astype(
                native.dtype)
        if hnorm_sink is not None:
            delta = (masked_out.astype(mx.float32)
                     - native.astype(mx.float32))
            h_pre = norm(x.astype(mx.float32))
            hnorm_sink.append({
                "arm": arm, "norm_h_pre": h_pre,
                "norm_native_out": norm(native.astype(mx.float32)),
                "norm_delta_direct": norm(delta),
                "delta_over_h_pre": (norm(delta) / h_pre
                                     if h_pre else None)})
        return out

    cls.__call__ = wrapped

    def restore():
        cls.__call__ = original

    return state, restore


def census_verdict(census, n_moe, arm):
    """Temporal law per module AND knife flags exactly (z1, bp43)
    for intervened arms, nowhere for NATIVE, nowhere else."""
    want = set() if arm == "NATIVE" else {TARGET_BP}
    per_module = {}
    for bp, calls in sorted(census.items()):
        phases = [ph for _, ph, _ in calls]
        prefill_idx = [i for i, (n, ph, _) in enumerate(calls)
                       if ph == "prefill" and n > 1]
        after = phases[prefill_idx[-1] + 1:] if prefill_idx else []
        law_ok = (bool(prefill_idx) and len(after) >= 4
                  and after[:3] == ["z1", "z2", "z3"]
                  and all(p == "decode" for p in after[3:]))
        knife_phases = {ph for _, ph, kf in calls if kf}
        knife_ok = (knife_phases == ({"z1"} if bp in want
                                     else set()))
        per_module[bp] = {"ok": law_ok and knife_ok,
                          "law_ok": law_ok, "knife_ok": knife_ok,
                          "knife_phases": sorted(knife_phases)}
    n_ok = sum(1 for v in per_module.values() if v["ok"])
    return {"arm": arm, "n_moe_seen": len(per_module), "n_ok": n_ok,
            "required": n_moe,
            "pass": len(per_module) == n_moe and n_ok == n_moe,
            "per_module": per_module}


def run_arm(model, tok, problems, keep, seed, arm, log_path,
            sealed_stdout=None):
    hsink = []
    state, restore = instrument(model, keep, arm, hnorm_sink=hsink)
    try:
        t0 = time.time()
        if sealed_stdout is not None:
            with sealed_stdout.open("a") as sf, \
                    contextlib.redirect_stdout(sf):
                n_ok, per_level = m.run_gate(model, tok, problems,
                                             f"kn_{arm}",
                                             state=state)
        else:
            n_ok, per_level = m.run_gate(model, tok, problems,
                                         f"kn_{arm}", state=state)
    finally:
        restore()
    recall = (round(state["hits"] / state["slots"], 4)
              if state["slots"] else None)
    with log_path.open("a") as f:
        f.write(json.dumps({
            "arm": f"kn_{arm}", "seed": seed,
            "n_eval": len(problems), "gate_ok": n_ok,
            "gate_per_level": per_level,
            "masked_recall_named80": recall,
            "knife_calls": state["knife_calls"],
            "gate_s": round(time.time() - t0, 1)}) + "\n")
    if hsink:
        with HNORMS.open("a") as f:
            f.write(json.dumps({"arm": f"kn_{arm}", "seed": seed,
                                "rows": hsink}) + "\n")
    return n_ok, state


def main():
    for pth in (CENSUS, QUAL, TREAT, QUAL_PP, TREAT_PP, HNORMS,
                SEALED_OUT):
        if pth.exists():
            raise SystemExit(f"REFUSING: {pth} exists")
    DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    art_dir = snapshot_download(m.MODEL, allow_patterns=["*.json"])
    START = start_provenance(
        ["scratch/ex6b43_knife.py", "scratch/moe_gt1_arm2.py",
         KEEPSET],
        artifacts={"model": art_dir})
    from mlx_lm import load

    from llmopt.mathgen.problems import make_dataset
    model, tok = load(m.MODEL)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(KEEPSET).read_text()).items()}

    # Stage 1: per-arm knife-position census — outcome-blind,
    # problem idx 2 only, perprob off.
    m.SEED = 7001
    probe = make_dataset(N_EVAL, seed=7001)[2:3]
    verdicts = {}
    perprob_save, m.PERPROB = m.PERPROB, False
    try:
        for arm in ARMS:
            state, restore = instrument(model, keep, arm)
            state["log_census"] = True
            try:
                m.run_gate(model, tok, probe, "kn_census",
                           state=state)
            finally:
                restore()
            verdicts[arm] = census_verdict(state["census"],
                                           state["n_moe"], arm)
            print(f"[kn] census {arm}: "
                  f"{verdicts[arm]['n_ok']}/{verdicts[arm]['required']}",
                  flush=True)
    finally:
        m.PERPROB = perprob_save
    CENSUS.write_text(json.dumps(
        {"start": START, "verdicts": {a: v for a, v in
                                      verdicts.items()}}, indent=1))
    if not SMOKE and not all(v["pass"] for v in verdicts.values()):
        print("[kn] CENSUS FAIL — no anchor or knife cell runs",
              flush=True)
        sys.exit(3)

    # Stage 2: anchors (always-readable, cell-exact).
    miss = []
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        m.SEED = seed
        for arm in ("NATIVE", "FULL_DIRECT"):
            n_ok, _ = run_arm(model, tok, problems, keep, seed,
                              arm, QUAL)
            booked = ANCHORS[arm][seed] if not SMOKE else None
            tag = ""
            if booked is not None and n_ok != booked:
                miss.append((seed, arm, n_ok, booked))
                tag = f"  ANCHOR MISS v booked {booked}"
            print(f"[kn] seed {seed} {arm}: {n_ok}/{len(problems)}"
                  f"{tag}", flush=True)
    with QUAL.open("a") as f:
        f.write(json.dumps({"meta": {
            "note": "EX6-B43-KNIFE-0 qual meta",
            "anchor_misses": miss, "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    if miss:
        print(f"[kn] QUALIFICATION FAIL {len(miss)} anchor cell(s)"
              " — knife arms do not run", flush=True)
        sys.exit(3)

    # Stage 3: knife treatment (sealed receipts + stdout).
    m.PERPROB_LOG = TREAT_PP
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        m.SEED = seed
        for arm in ("D_ONLY", "NO_D", "FULL_SUM"):
            run_arm(model, tok, problems, keep, seed, arm, TREAT,
                    sealed_stdout=SEALED_OUT)
            print(f"[kn] seed {seed} {arm}: done "
                  f"(value sealed to treatment.jsonl)", flush=True)
    with TREAT.open("a") as f:
        f.write(json.dumps({"meta": {
            "note": "EX6-B43-KNIFE-0 run meta", "start": START,
            "completion_commit": completion_commit()}}) + "\n")
    print("[kn] done; qualification PASS, treatment sealed",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

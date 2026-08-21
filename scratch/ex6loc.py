"""EX6-LOC-0 driver (frozen pre-launch): native-path localization
2x2 per PRE-REG EX6-LOC-0 — which physical component of the
EX6-PROMPT mask carries the +47 crest.

Arms (mask predicate over the frozen wrapper's phase labels):
  NONE          nothing masked
  PREFILL_ONLY  mask phase == 'prefill'      (the prompt batch)
  TOKEN1_ONLY   mask phase == 'prompt_tail'  (generated z1's own
                routing; label kept for receipt compatibility —
                measured mlx_lm semantics: the prompt batch routes
                all prompt tokens and produces z1's logits, the
                first T=1 call routes generated z1 and produces
                z2's logits, AMENDMENT EX6-MED-0-SEMANTICS)
  PROMPT        mask both (the booked EX6-PROMPT arm)

Native generation only (m.run_gate), no forced tokens. Arm order
per seed: NONE, PROMPT (qualification: must reproduce 64/61/66 and
79/80/79 cell-exact), then PREFILL_ONLY, TOKEN1_ONLY — the two new
arms are skipped for a seed whose qualification arm misses.

The wrapper math is the frozen ex6_phase wrapped-gate body with
the mask decision parametrized (identical to scratch/ex6med.py's
instrument, imported from there; ex6med's BATCH predicate is not
used).

Receipts: logs/ex6loc/ex6loc.jsonl + perprob (refuse-if-exists;
SMOKE=1 -> smoke paths, 4 problems, seed 7001).

    .venv/bin/python scratch/ex6loc.py                        (Mac)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMOKE = os.environ.get("SMOKE") == "1"
PRE = "smoke_" if SMOKE else ""
os.environ.setdefault("LOG", f"logs/ex6loc/{PRE}ex6loc.jsonl")
os.environ.setdefault("PERPROB", "1")
os.environ.setdefault("PERPROB_LOG",
                      f"logs/ex6loc/{PRE}ex6loc_perprob.jsonl")

import scratch.ex6med as x  # noqa: E402  (instrument + PREDS)
import scratch.moe_gt1_arm2 as m  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SEEDS = [7001] if SMOKE else [7001, 8002, 9003]
N_EVAL = 4 if SMOKE else 120
BOOKED = x.BOOKED
ARMS = ("NONE", "PROMPT", "PREFILL_ONLY", "TOKEN1_ONLY")
PREDS = {"NONE": lambda ph: False,
         "PROMPT": lambda ph: ph in ("prefill", "prompt_tail"),
         "PREFILL_ONLY": lambda ph: ph == "prefill",
         "TOKEN1_ONLY": lambda ph: ph == "prompt_tail"}


def main():
    for pth in (Path(m.LOG), Path(os.environ["PERPROB_LOG"])):
        if pth.exists():
            raise SystemExit(f"REFUSING: {pth} exists")
    Path(m.LOG).parent.mkdir(parents=True, exist_ok=True)
    START = start_provenance(
        ["scratch/ex6loc.py", "scratch/ex6med.py",
         "scratch/moe_gt1_arm2.py", x.KEEPSET])
    from mlx_lm import load

    from llmopt.mathgen.problems import make_dataset
    model, tok = load(m.MODEL)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(x.KEEPSET).read_text()).items()}
    log_f = Path(m.LOG).open("a")
    invalid = set()
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        m.SEED = seed
        for arm in ARMS:
            if arm in ("PREFILL_ONLY", "TOKEN1_ONLY") \
                    and seed in invalid:
                print(f"[loc] seed {seed} INVALID — skip {arm}",
                      flush=True)
                continue
            state, restore = x.instrument(model, keep, PREDS[arm])
            try:
                t0 = time.time()
                n_ok, per_level = m.run_gate(model, tok, problems,
                                             f"loc_{arm}",
                                             state=state)
            finally:
                restore()
            recall = (round(state["hits"] / state["slots"], 4)
                      if state["slots"] else None)
            log_f.write(json.dumps({
                "arm": f"loc_{arm}", "seed": seed,
                "n_eval": len(problems), "gate_ok": n_ok,
                "gate_per_level": per_level,
                "masked_recall_named80": recall,
                "gate_s": round(time.time() - t0, 1)}) + "\n")
            log_f.flush()
            if not SMOKE and arm in ("NONE", "PROMPT"):
                if n_ok != BOOKED[arm][seed]:
                    invalid.add(seed)
                    print(f"[loc] QUAL MISMATCH seed {seed} {arm}: "
                          f"{n_ok} v booked {BOOKED[arm][seed]}",
                          flush=True)
            print(f"[loc] seed {seed} {arm}: {n_ok}/{len(problems)}"
                  f" {time.time() - t0:.0f}s", flush=True)
    log_f.write(json.dumps({"meta": {
        "note": "EX6-LOC-0 run meta", "start": START,
        "completion_commit": completion_commit(),
        "invalid_seeds": sorted(invalid)}}) + "\n")
    log_f.close()
    if len(invalid) >= 2:
        print(f"[loc] RUN INVALID: {sorted(invalid)}", flush=True)
        sys.exit(3)
    print(f"[loc] done; invalid_seeds={sorted(invalid)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

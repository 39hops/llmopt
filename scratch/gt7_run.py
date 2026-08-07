"""MOE-GT-7 gate driver (PRE-REG MOE-GT-7, Artin GO 2026-08-06).

Runs the 10 drawn arms (scratch/gt7_draw.py receipts) through the
FROZEN moe_gt1_arm2 machinery by import — instrument, run_gate,
boxed oracle (v3.2 line-server + RSS watchdog), probe — with one
model load for all arms (one-resident-30B; sequential).

Additions over the frozen path, both registered:
  - DEGENERACY READOUT (descriptive only): every extracted answer
    streams to ANSWERS_LOG as {arm, idx, expr}; distinct-count per
    arm printed and logged.
  - ORACLE TIMEOUT CENSUS per arm (run_gate.timeouts snapshots).

Env (set by launcher): N_EVAL=120 SEED=1234 PERPROB=1
  LOG=logs/gt7/gt7.jsonl PERPROB_LOG=logs/gt7/gt7_perprob.jsonl
The 0.00-coverage bin (gt6_novrb pair) is REUSED from booked GT-6
gates (0/7), not re-run.

Usage: .venv/bin/python scratch/gt7_run.py
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scratch.moe_gt1_arm2 as m

ARMS = [
    "gt7_ladder_c15_d0", "gt7_ladder_c15_d1",
    "gt7_ladder_c30_d0", "gt7_ladder_c30_d1",
    "gt7_ladder_c45_d0", "gt7_ladder_c45_d1",
    "gt7_anom_r75_0", "gt7_anom_r75_1",
    "gt7_anom_r80_0", "gt7_anom_r80_1",
]
if os.environ.get("ARMS"):
    ARMS = [a for a in os.environ["ARMS"].split(",") if a]
ANSWERS_LOG = Path(os.environ.get("ANSWERS_LOG",
                                  "logs/gt7/gt7_answers.jsonl"))


def main():
    from mlx_lm import generate, load

    from llmopt.mathgen.problems import make_dataset

    arm0 = json.loads(m.ARM0.read_text())
    counts = arm0["counts"]
    model, tok = load(m.MODEL)
    problems = make_dataset(m.N_EVAL, seed=m.SEED)
    m.LOG.parent.mkdir(parents=True, exist_ok=True)
    ANSWERS_LOG.parent.mkdir(parents=True, exist_ok=True)

    frozen_extract = m.extract_expression
    if getattr(frozen_extract, "_gt7_recording", False):
        raise SystemExit("extract_expression already patched — one "
                         "main() per process (code review 2026-08-07)")
    answers_f = ANSWERS_LOG.open("a")
    current = {"arm": None, "idx": 0, "exprs": []}

    def recording_extract(completion):
        expr = frozen_extract(completion)
        answers_f.write(json.dumps({
            "arm": current["arm"], "idx": current["idx"],
            "expr": expr}) + "\n")
        answers_f.flush()
        current["idx"] += 1
        current["exprs"].append(expr)
        return expr

    recording_extract._gt7_recording = True
    m.extract_expression = recording_extract

    for arm in ARMS:
        keep = {int(li): set(v) for li, v in
                json.loads(Path(f"checkpoints/{arm}.json").read_text()
                           ).items()}
        ol = m.open_loop_recall(counts, keep)
        n_keep = sum(len(v) for v in keep.values()) / len(keep)
        print(f"[gt7] === arm {arm} | keep {n_keep:.0f}/128 | "
              f"open-loop recall {ol:.4f} ===", flush=True)
        current.update(arm=arm, idx=0, exprs=[])
        t_before = getattr(m.run_gate, "timeouts", 0)
        state, restore = m.instrument(model, keep)
        try:
            t0 = time.time()
            n_ok, per_level = m.run_gate(model, tok, problems, arm,
                                         state=state)
            gate_s = time.time() - t0
            cl = state["hits"] / max(state["slots"], 1)
            probe_text = generate(model, tok, prompt=m.PROBE,
                                  max_tokens=m.PROBE_TOKENS)
        finally:
            restore()
        n_timeouts = getattr(m.run_gate, "timeouts", 0) - t_before
        distinct = len(set(e for e in current["exprs"] if e))
        n_empty = sum(1 for e in current["exprs"] if not e)
        print(f"[gt7] arm {arm} GATE {n_ok}/{len(problems)} "
              f"per-level {per_level} | closed recall {cl:.4f} "
              f"(open {ol:.4f}) | distinct answers {distinct}/{len(problems)} "
              f"(empty {n_empty}) | timeouts {n_timeouts} | "
              f"{gate_s:.0f}s", flush=True)
        print(f"[gt7] PROBE TEXT (verbatim): {probe_text!r}", flush=True)
        with m.LOG.open("a") as f:
            f.write(json.dumps({
                "battery": "gt7", "arm": arm, "seed": m.SEED,
                "n_eval": m.N_EVAL, "gate_ok": n_ok,
                "gate_per_level": per_level, "open_recall": ol,
                "closed_recall": cl, "distinct_answers": distinct,
                "empty_answers": n_empty, "oracle_timeouts": n_timeouts,
                "gate_s": gate_s, "probe_text": probe_text,
            }) + "\n")
    m.extract_expression = frozen_extract
    answers_f.close()
    print("[gt7] all arms done", flush=True)


if __name__ == "__main__":
    main()

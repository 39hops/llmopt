"""GENERAL-ROUTING-FACTORIAL capture (observation-only; the banked
riff's residue item 2). Runs the FROZEN corpus (logs/grf/corpus.json,
emitted by the committed scratch/grf_corpus.py before any model
call) through the pinned 4-bit MoE with the frozen moe_gt1 TRAJ
instrumentation (imported, never edited): per-token routing rows
(prompt id, layer, position, top-8, scores, phase, router entropy)
stream to the traj file per prompt; the generated text (for a
LATER, separately registered answer-identity leg — stored, not
scored here) streams to a rows file.

Passive interpretability of a frozen public model: no oracle, no
scoring, no adaptation of anything to any output.

Env: SMOKE=1 -> first 4 prompts, smoke paths. MAX_TOKENS default 96.

Receipts: logs/grf/traj.jsonl + logs/grf/rows.jsonl
(refuse-if-exists; smoke -> logs/grf/smoke_*.jsonl).

    TRAJ=1 .venv/bin/python scratch/grf_capture.py            (Mac)
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SMOKE = os.environ.get("SMOKE") == "1"
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "96"))
CORPUS = "logs/grf/corpus.json"
TRAJ_OUT = Path("logs/grf/smoke_traj.jsonl" if SMOKE
                else "logs/grf/traj.jsonl")
ROWS_OUT = Path("logs/grf/smoke_rows.jsonl" if SMOKE
                else "logs/grf/rows.jsonl")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, str(Path(__file__).resolve().parents[1] / rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    assert os.environ.get("TRAJ") == "1", "run with TRAJ=1"
    for p in (TRAJ_OUT, ROWS_OUT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    m = _load("moe_gt1", "scratch/moe_gt1.py")
    START = start_provenance(
        ["scratch/grf_capture.py", "scratch/grf_corpus.py",
         "scratch/moe_gt1.py", CORPUS])
    corpus = json.load(open(CORPUS))
    if SMOKE:
        corpus = corpus[:4]

    from mlx_lm import generate, load
    model, tok = load(m.MODEL)
    state, n_experts = m.instrument(model)
    state["stats"] = m.RouterStats(n_experts=n_experts)
    TRAJ_OUT.parent.mkdir(parents=True, exist_ok=True)
    traj_f = TRAJ_OUT.open("w")
    rows_f = ROWS_OUT.open("w")
    t00 = time.time()
    for r in corpus:
        state["prompt"] = r["pid"]
        state["tpos"] = {}
        state["tail_done"] = {}
        msgs = [{"role": "user", "content": r["prompt"]}]
        text = tok.apply_chat_template(msgs,
                                       add_generation_prompt=True)
        t0 = time.time()
        completion = generate(model, tok, prompt=text,
                              max_tokens=MAX_TOKENS)
        for row in state["traj"]:
            traj_f.write(json.dumps(row) + "\n")
        state["traj"].clear()
        traj_f.flush()
        rows_f.write(json.dumps({
            **{k: r[k] for k in ("pid", "topic", "prop_idx",
                                 "operation", "form")},
            "completion": completion,
            "gen_s": round(time.time() - t0, 2)}) + "\n")
        rows_f.flush()
        if r["pid"] % 20 == 0 or SMOKE:
            print(f"[grf] pid {r['pid']} ({r['topic']}/{r['form']}) "
                  f"{time.time() - t0:.1f}s "
                  f"total {time.time() - t00:.0f}s", flush=True)
    traj_f.close()
    meta = {"note": "GRF capture meta", "start": START,
            "completion_commit": completion_commit(),
            "n_prompts": len(corpus), "max_tokens": MAX_TOKENS}
    rows_f.write(json.dumps({"meta": meta}) + "\n")
    rows_f.close()
    print(f"[grf] done -> {TRAJ_OUT} + {ROWS_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""MOE-GT-2 arm D3: CODE arm-0 — decode-only demand log on the codegen
ladder (pre-reg MOE-GT-2, 2026-08-04).

Reuses scratch/moe_gt1.py's certified router instrument (class-patch,
TRAJ v2 with phase + scores) on a toolchain-scored code corpus:
build_ladder RungTasks, oracle-checked by each task's own check()
(assemble/run — llvm.py). 120-task selection rule, fixed here:
round-robin across non-empty rungs in RUNGS order, taking tasks in
build order within each rung, until 120 (the o2_asm rung produced 0
tasks at this seed — flagged, not silently absorbed).

SYSTEM prompt is the bench_ladder code prompt, NOT the mathgen one
(fence: this is a deliberate corpus-plus-prompt treatment; the
coalition comparison notes it).

Usage: TRAJ=1 TRAJ_OUT=logs/opus/gt2_code_traj.jsonl \
       .venv/bin/python scratch/gt2_code_arm0.py
       [N_EVAL=120, MAX_TOKENS=96, SEED=99 env overrides]
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llmopt.codegen.ladder import RUNGS, build_ladder

N_EVAL = int(os.environ.get("N_EVAL", 120))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 96))
SEED = int(os.environ.get("SEED", 99))
OUT = Path(os.environ.get("OUT", "checkpoints/gt2_code_arm0.json"))
LOG = Path("logs/opus/moe_gt1.jsonl")

SYSTEM = (
    "You are an assembly and C toolchain assistant. Answer with only "
    "the requested output, no explanation."
)


def select_tasks(by_rung, n):
    """Fixed 120-task rule: round-robin across non-empty rungs in RUNGS
    order, build order within a rung."""
    pools = {r: list(by_rung[r]) for r in RUNGS if by_rung[r]}
    order = [r for r in RUNGS if r in pools]
    out, i = [], 0
    while len(out) < n and any(pools.values()):
        r = order[i % len(order)]
        i += 1
        if pools[r]:
            out.append(pools[r].pop(0))
    return out


def main():
    from mlx_lm import load, generate

    import moe_gt1  # certified instrument (class-patch, TRAJ v2)
    from llmopt.moe.router_stats import RouterStats

    by_rung = build_ladder(40, seed=SEED)
    print("[gt2-code] rung pool:", {r: len(t) for r, t in by_rung.items()},
          flush=True)
    tasks = select_tasks(by_rung, N_EVAL)
    assert len(tasks) == N_EVAL, f"only {len(tasks)} tasks available"

    model, tok = load(moe_gt1.MODEL)
    # D4 cross-arm mode: ARM0+FRAC mask the router to a keep-set drawn
    # from the named demand log (arm2's rule) instead of instrumenting
    # for demand. Gate + closed-loop recall only, no traj.
    mask_arm0 = os.environ.get("ARM0")
    if mask_arm0:
        import moe_gt1_arm2 as arm2
        frac = float(os.environ.get("FRAC", "0.453"))
        counts = json.loads(Path(mask_arm0).read_text())["counts"]
        top_k = next(
            layer.mlp.top_k for layer in model.model.layers
            if hasattr(layer.mlp, "top_k"))
        keep = arm2.keep_sets_from_counts(counts, frac, top_k)
        ol = arm2.open_loop_recall(counts, keep)
        print(f"[gt2-code] MASKED: {mask_arm0} frac {frac} "
              f"keep {sum(len(v) for v in keep.values()) / len(keep):.0f}"
              f"/128 | open-loop recall vs own log {ol:.4f}", flush=True)
        recall_state, _restore = arm2.instrument(model, keep)
        state = {"traj": None}
        n_experts = 128
        stats = None
    else:
        state, n_experts = moe_gt1.instrument(model)
        stats = RouterStats(n_experts=n_experts)
        state["stats"] = stats

    traj_f = None
    if state["traj"] is not None:
        traj_path = Path(os.environ.get(
            "TRAJ_OUT", "logs/opus/gt2_code_traj.jsonl"))
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        traj_f = traj_path.open("w")

    per_rung, n_ok = {}, 0
    t0 = time.time()
    for i, t in enumerate(tasks):
        state["prompt"] = i
        state["tpos"] = {}
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": t.prompt}]
        text = tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        completion = generate(model, tok, prompt=text, max_tokens=MAX_TOKENS)
        ok = bool(t.check(completion))
        n_ok += ok
        per_rung[t.rung] = per_rung.get(t.rung, 0) + int(ok)
        if traj_f is not None:
            for row in state["traj"]:
                row["ok"] = ok
                traj_f.write(json.dumps(row) + "\n")
            state["traj"].clear()
        if (i + 1) % 20 == 0:
            print(f"[gt2-code] gate {i + 1}/{len(tasks)} "
                  f"acc {n_ok / (i + 1):.1%} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    gate_s = time.time() - t0
    if traj_f is not None:
        traj_f.close()
    print(f"[gt2-code] GATE {'MASKED' if mask_arm0 else 'full'} model: "
          f"{n_ok}/{len(tasks)} per-rung {per_rung} | {gate_s:.0f}s",
          flush=True)

    if mask_arm0:
        cl = recall_state["hits"] / max(recall_state["slots"], 1)
        print(f"[gt2-code] closed-loop recall {cl:.4f} "
              f"(open vs own log {ol:.4f})", flush=True)
        with LOG.open("a") as f:
            f.write(json.dumps({
                "arm": "gt2-code-masked", "arm0": mask_arm0, "frac": frac,
                "gate_ok": n_ok, "n_eval": N_EVAL, "seed": SEED,
                "gate_per_rung": per_rung, "open_recall": ol,
                "closed_recall": cl, "gate_s": gate_s,
            }) + "\n")
        _restore()
        return

    tails = {li: moe_gt1.tail_share(stats.mass[li])
             for li in sorted(stats.mass)}
    mean_tail = sum(tails.values()) / len(tails)
    print(f"[gt2-code] ROUTING TAIL top-25% mass share: mean "
          f"{mean_tail:.3f}", flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "model": moe_gt1.MODEL, "n_experts": n_experts, "n_eval": N_EVAL,
        "seed": SEED, "max_tokens": MAX_TOKENS, "corpus": "codegen-ladder",
        "gate_ok": n_ok, "gate_per_rung": per_rung,
        "counts": stats.counts, "mass": stats.mass,
        "tail_top25": tails,
    }))
    with LOG.open("a") as f:
        f.write(json.dumps({
            "arm": "gt2-code-0", "gate_ok": n_ok, "n_eval": N_EVAL,
            "gate_per_rung": per_rung, "mean_tail_top25": mean_tail,
            "gate_s": gate_s,
        }) + "\n")
    print(f"[gt2-code] saved {OUT} + log row", flush=True)


if __name__ == "__main__":
    main()

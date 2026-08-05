"""MOE-GT-3 arm-0: demand log over an arbitrary prompt list (pre-reg
MOE-GT-3, 2026-08-05 — the base-class discriminators).

Reuses the certified TRAJ instrument from scratch/moe_gt1.py on a JSON
prompt list (no oracle, no gate — coalition readouts only need
routing). Rows carry ok=None. Corpus files carry {"prompt", "kind",
"level"} dicts (checkpoints/gt3_proofs_prompts.json,
gt3_prose_prompts.json).

Usage: PROMPTS=checkpoints/gt3_proofs_prompts.json \
       TRAJ_OUT=logs/opus/gt3_proofs_traj.jsonl \
       .venv/bin/python scratch/gt3_probe_arm0.py   [MAX_TOKENS=96]
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from llmopt.mathgen.evaluate import SYSTEM

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 96))
PROMPTS = Path(os.environ["PROMPTS"])
TRAJ_OUT = Path(os.environ["TRAJ_OUT"])
# proofs rows use the math SYSTEM (held fixed, as in D2's design);
# prose rows use a neutral one (registered: the prose arm is a corpus
# PLUS prompt treatment, like D3's code arm — fence travels).
PROSE_SYSTEM = "You are a helpful writing assistant."


def main():
    from mlx_lm import load, generate

    import moe_gt1  # certified TRAJ instrument

    from llmopt.moe.router_stats import RouterStats

    os.environ["TRAJ"] = "1"
    rows = json.loads(PROMPTS.read_text())
    model, tok = load(moe_gt1.MODEL)
    state, n_experts = moe_gt1.instrument(model)
    # the wrapped router records (traj included) only when stats is set
    state["stats"] = RouterStats(n_experts=n_experts)

    TRAJ_OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with TRAJ_OUT.open("w") as traj_f:
        for i, r in enumerate(rows):
            state["prompt"] = i
            state["tpos"] = {}
            state["tail_done"] = {}
            sys_prompt = (PROSE_SYSTEM if r["kind"] in ("prose", "dialog")
                          else SYSTEM)
            msgs = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": r["prompt"]}]
            text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
            generate(model, tok, prompt=text, max_tokens=MAX_TOKENS)
            for row in state["traj"]:
                row["kind"] = r["kind"]
                traj_f.write(json.dumps(row) + "\n")
            state["traj"].clear()
            if (i + 1) % 20 == 0:
                print(f"[gt3] {i + 1}/{len(rows)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
    print(f"[gt3] wrote {TRAJ_OUT} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

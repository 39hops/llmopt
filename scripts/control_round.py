"""Control round: retrain on the EXACT rounds-2/3 diet, gate it.

Four consecutive rollbacks (manual r4 + autonomous takes 2/3 + hints-
off rounds 1/2) all retrained on post-r3 corpora and all failed the
gate — but round 2's candidate REALLOCATED (L2 7->15, L3 10->6)
rather than degrading. Competing hypotheses:
  H_diet:     the corpus additions since r3 are net harmful.
  H_variance: retraining itself is high-variance; the promoted model
              is a lucky balanced draw, and ANY retrain — even on its
              own diet — rolls a new allocation.
This round splits them: train on the corpus at 38c8c46 (the promoted
model's own diet, byte-identical), same recipe (3 epochs, lr 1e-4),
gate against the promoted model on a fresh band. Diet identical =>
any gate failure is variance.

    .venv/bin/python scripts/control_round.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

R23 = Path("data/step_chains_r23.jsonl")
CAND = Path("checkpoints/step_lora_control.pt")
SEED_BASE = 8_500_000  # fresh eval band


def main() -> None:
    import expert_iter_steps as eis
    from expert_loop import evaluate, gate_verdict
    if not R23.exists():
        blob = subprocess.run(
            ["git", "show", "38c8c46:data/step_chains.jsonl"],
            capture_output=True, text=True, check=True).stdout
        R23.write_text(blob)
    print(f"r23 diet: {sum(1 for _ in R23.open())} rows", flush=True)

    from bench_step_tokens import load
    tok, model = load("checkpoints/step_lora.pt")
    pre = evaluate(tok, model, levels=(2, 3, 4, 5), n_per=24,
                   seed_base=SEED_BASE, budget=512)
    print(f"promoted: {pre['solves']} validity "
          f"{pre['validity']:.2f}%", flush=True)
    del model

    eis.CHAINS = R23  # the byte-identical rounds-2/3 diet
    eis.phase_train(epochs=3, lr=1e-4, out=CAND)
    tok, model = load(str(CAND))
    post = evaluate(tok, model, levels=(2, 3, 4, 5), n_per=24,
                    seed_base=SEED_BASE, budget=512)
    print(f"control:  {post['solves']} validity "
          f"{post['validity']:.2f}%", flush=True)
    ok, reason = gate_verdict(
        {"solves": pre["solves"], "validity": pre["validity"]},
        {"solves": post["solves"], "validity": post["validity"]}, 5)
    print(f"\ncontrol gate: {'PASS' if ok else 'FAIL'} ({reason})")
    print("FAIL on its own diet => H_variance (tournament gate next); "
          "PASS => H_diet (the additions were the harm)")


if __name__ == "__main__":
    main()

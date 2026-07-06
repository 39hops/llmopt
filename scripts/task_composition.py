"""Task-arithmetic experiment 4: composition (see 2026-07-06 spec).

Two single-skill adapters (differentiate-only, integrate-only, trained
on the Windows box) applied to one Instruct model: each alone, then
both added. Success bar from the spec: both-added retains >= 90% of
each skill's alone-accuracy. Interference below that is the finding.

Eval is per-skill: differentiate problems score the diff skill,
integrate problems the int skill (same seed-99 protocol as the other
task-arithmetic experiments).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmopt.mathgen.evaluate import evaluate_model
from llmopt.mathgen.problems import make_dataset
from llmopt.train.task_vector import apply_task_vector, load_adapter

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
N_EVAL = 150  # per skill


def skill_acc(model, tok, problems) -> float:
    return evaluate_model(model, tok, problems, max_new_tokens=160)["overall"]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(device)
    model.eval()

    diff_ad = load_adapter("checkpoints/diff_only_lora.pt")
    int_ad = load_adapter("checkpoints/int_only_lora.pt")
    diff_probs = make_dataset(N_EVAL, kinds=("differentiate",), seed=99)
    int_probs = make_dataset(N_EVAL, kinds=("integrate",), seed=99)

    def both_skills(label):
        d = skill_acc(model, tok, diff_probs)
        i = skill_acc(model, tok, int_probs)
        print(f"  {label:14s} diff {d:6.1%}   int {i:6.1%}")
        return d, i

    print("composition (undo order is LIFO):")
    both_skills("baseline")

    undo_d = apply_task_vector(model, diff_ad, 1.0)
    d_alone, _ = both_skills("diff alone")
    undo_d()

    undo_i = apply_task_vector(model, int_ad, 1.0)
    _, i_alone = both_skills("int alone")
    undo_i()

    undo_d = apply_task_vector(model, diff_ad, 1.0)
    undo_i = apply_task_vector(model, int_ad, 1.0)
    d_both, i_both = both_skills("both added")
    undo_i()
    undo_d()

    print(f"\nretention: diff {d_both / max(d_alone, 1e-9):.1%}, "
          f"int {i_both / max(i_alone, 1e-9):.1%}  (bar: >= 90%)")


if __name__ == "__main__":
    main()

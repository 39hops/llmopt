"""Symbolic-equivalence eval for calculus problems.

Greedy-generate an answer per problem, parse it with sympy, score by
Problem.check() — algebraic equivalence, not string match. Reports
accuracy per (kind, level) plus overall.
"""

from __future__ import annotations

from typing import Sequence

from llmopt.mathgen.problems import Problem

SYSTEM = (
    "You are a calculus assistant. Answer with a single mathematical "
    "expression on one line, in Python/SymPy syntax (use ** for powers, "
    "sin/cos/exp/log). No explanation."
)


def format_chat(tok, problem: Problem) -> list[int]:
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": problem.prompt},
    ]
    text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return tok(text, add_special_tokens=False).input_ids


def extract_expression(text: str) -> str:
    """First non-empty line of the completion (the format we train for)."""
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def evaluate_model(
    model, tok, problems: Sequence[Problem], *, max_new_tokens: int = 64,
    batch_size: int = 16,
) -> dict:
    import torch

    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    results: dict[tuple, list[bool]] = {}
    n_ok = 0
    for i in range(0, len(problems), batch_size):
        batch = problems[i : i + batch_size]
        prompts = [format_chat(tok, p) for p in batch]
        width = max(len(p) for p in prompts)
        ids = torch.full((len(batch), width), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, p in enumerate(prompts):
            ids[j, width - len(p):] = torch.tensor(p)
            mask[j, width - len(p):] = 1
        ids, mask = ids.to(model.device), mask.to(model.device)
        with torch.inference_mode():
            out = model.generate(
                input_ids=ids, attention_mask=mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        for j, p in enumerate(batch):
            completion = tok.decode(out[j, width:], skip_special_tokens=True)
            ok = p.check(extract_expression(completion))
            results.setdefault((p.kind, p.level), []).append(ok)
            n_ok += ok
    report = {
        f"{kind}/L{level}": sum(v) / len(v)
        for (kind, level), v in sorted(results.items())
    }
    report["overall"] = n_ok / len(problems)
    return report

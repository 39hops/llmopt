"""Move proposer: a policy model in front of the classical searcher
(spec: 2026-07-07-move-proposer-design.md). The searcher enumerates
LEGAL moves; the model only ranks them — rank-not-generate keeps
legality by construction. Ranking = likelihood of each numbered choice's
answer tokens under the fine-tuned model."""

from __future__ import annotations

from typing import Callable

import sympy as sp

from llmopt.search.derivation import State

ScoreFn = Callable[[str, "list[str]"], "list[float]"]


def build_prompt(state_str: str, labels: list[str]) -> str:
    lines = [f"State: {state_str}", "Legal moves:"]
    lines += [f"{i + 1}. {lab}" for i, lab in enumerate(labels)]
    lines.append("Best move:")
    return "\n".join(lines)


def make_proposer(score_fn: ScoreFn):
    """Wrap a scoring function into the beam_search proposer callable.
    Higher score = better; sort is stable so ties keep enumeration order."""

    def proposer(state: State, children: list[tuple[str, State]]):
        if not children:
            return children
        labels = [name for name, _ in children]
        scores = score_fn(sp.sstr(state.expr), labels)
        order = sorted(range(len(children)), key=lambda i: -scores[i])
        return [children[i] for i in order]

    return proposer


def make_scoring_proposer(score_fn: ScoreFn):
    """Like make_proposer, but returns (ranked_children, scores_desc)
    so an adaptive-k policy can read the ranker's confidence."""

    def proposer(state: State, children: list[tuple[str, State]]):
        if not children:
            return children, []
        labels = [name for name, _ in children]
        scores = score_fn(sp.sstr(state.expr), labels)
        order = sorted(range(len(children)), key=lambda i: -scores[i])
        return [children[i] for i in order], [scores[i] for i in order]

    return proposer


def entropy_k(k_min: int = 1, k_max: int = 6, temperature: float = 1.0):
    """Confidence-gated branching: peaked ranking -> deep (k_min);
    flat ranking -> wide (k_max). H is normalized entropy of the
    softmax over child scores (spec: 2026-07-07-adaptive-k-design.md)."""
    import math

    def policy(state, ranked, scores) -> int:
        n = len(scores)
        if n <= 1:
            return max(1, k_min)
        m = max(s / temperature for s in scores)
        exps = [math.exp(s / temperature - m) for s in scores]
        z = sum(exps)
        ps = [e / z for e in exps]
        h = -sum(p * math.log(p) for p in ps if p > 0) / math.log(n)
        return k_min + round(h * (k_max - k_min))

    return policy


def hf_score_fn(model, tok, device: str) -> ScoreFn:
    """Score each candidate as the mean logprob of its answer tokens
    (' {i}') given the numbered-choice prompt. Batched; 1-2 answer
    tokens per candidate keeps this cheap even at ~30 candidates."""
    import torch

    def score(state_str: str, labels: list[str]) -> list[float]:
        prompt = build_prompt(state_str, labels)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        rows, spans = [], []
        for i in range(len(labels)):
            a_ids = tok(f" {i + 1}", add_special_tokens=False).input_ids
            rows.append(p_ids + a_ids)
            spans.append(len(a_ids))
        width = max(len(r) for r in rows)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(rows), width), pad, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for j, r in enumerate(rows):
            ids[j, : len(r)] = torch.tensor(r)
            mask[j, : len(r)] = 1
        ids, mask = ids.to(device), mask.to(device)
        # NEVER materialize full logits: the LM head to a 152k vocab on
        # batch x seq positions is ~11 GB on long states (OOM'd the
        # 3080 twice — first in log_softmax, then in the head matmul
        # itself). Run the transformer BODY (hidden states are MBs),
        # then project ONLY the answer positions through the head.
        pos_j, pos_t = [], []
        for j, r in enumerate(rows):
            for t in range(len(r) - spans[j], len(r)):
                pos_j.append(j)
                pos_t.append(t - 1)
        with torch.no_grad():
            hidden = model.model(input_ids=ids,
                                 attention_mask=mask).last_hidden_state
            sel_h = hidden[pos_j, pos_t]  # (n_positions, d_model)
            sel = model.lm_head(sel_h)    # (n_positions, vocab) — tiny
        sel = torch.log_softmax(sel.float(), dim=-1)
        out, idx = [], 0
        for j, r in enumerate(rows):
            n = spans[j]
            lp = 0.0
            for t in range(len(r) - n, len(r)):
                lp += float(sel[idx, r[t]])
                idx += 1
            out.append(lp / n)
        return out

    return score

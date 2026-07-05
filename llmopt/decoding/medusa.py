"""Medusa heads: extra LM heads drafting several lookahead tokens at once.

Head i predicts the token at offset i+1 ahead (head 0 = two tokens ahead
of the current position; the base LM head covers one ahead). Drafting
takes top-k per head and forms candidate paths; verification reuses the
tree-attention machinery from tree_verify, so output stays token-
identical to vanilla greedy regardless of head quality — bad heads only
cost speed.

train_medusa_heads fits heads on frozen base-model hidden states
(self-distillation on plain text), the standard Medusa-1 recipe.
"""

from __future__ import annotations

from itertools import product

from llmopt.decoding.tree_verify import (
    TokenTree,
    gather_kv,
    tree_attention_inputs,
)


def build_medusa_heads(hidden_size: int, vocab_size: int, num_heads: int = 3):
    """ResBlock + linear per head (Medusa-1 architecture)."""
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(hidden_size, hidden_size)
            self.act = nn.SiLU()

        def forward(self, x):
            return x + self.act(self.linear(x))

    return nn.ModuleList(
        nn.Sequential(ResBlock(), nn.Linear(hidden_size, vocab_size, bias=False))
        for _ in range(num_heads)
    )


def draft_candidates(heads, hidden, next_token: int, topk: int = 2) -> list[list[int]]:
    """Candidate continuations of ``next_token`` from head top-ks.

    Cartesian product of per-head top-k, best-first; each candidate is
    [next_token, head0_choice, head1_choice, ...]. Product size is
    topk**num_heads — keep topk small (Medusa uses sparse trees; the
    trie merge collapses shared prefixes anyway).
    """
    import torch

    per_head = [
        torch.topk(head(hidden), topk).indices.tolist() for head in heads
    ]
    return [[next_token, *combo] for combo in product(*per_head)]


def generate_medusa(
    model,
    heads,
    input_ids,
    *,
    max_new_tokens: int = 128,
    topk: int = 2,
    eos_token_id: int | None = None,
):
    """Greedy decoding with Medusa-head drafts, tree-verified.

    Token-identical to vanilla greedy. Returns (tokens, stats).
    """
    import torch

    from llmopt.decoding.kv import valid_len

    device = next(model.parameters()).device
    tokens = input_ids[0].tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    stats = {
        "drafted": 0, "accepted": 0, "forward_passes": 0,
        "prompt_len": len(tokens),
    }

    with torch.inference_mode():
        out = model(
            input_ids=torch.tensor([tokens], device=device),
            use_cache=True, output_hidden_states=True,
        )
        past = out.past_key_values
        stats["forward_passes"] += 1
        hidden = out.hidden_states[-1][0, -1]
        tokens.append(int(out.logits[0, -1].argmax()))
        produced = 1

        while produced < max_new_tokens:
            budget = max_new_tokens - produced - 1
            cands = [
                c[1 : 1 + budget]  # candidates start with tokens[-1] itself
                for c in draft_candidates(heads, hidden, tokens[-1], topk)
            ]
            tree = TokenTree.from_candidates([c for c in cands if c])
            stats["drafted"] += len(tree)

            prefix_len = valid_len(past)
            fed = [tokens[-1]] + tree.tokens
            mask, pos, cache_pos = tree_attention_inputs(
                tree, prefix_len, device, model.dtype
            )
            out = model(
                input_ids=torch.tensor([fed], device=device),
                past_key_values=past, attention_mask=mask,
                position_ids=pos, cache_position=cache_pos,
                use_cache=True, output_hidden_states=True,
            )
            past = out.past_key_values
            stats["forward_passes"] += 1
            preds = out.logits[0].argmax(-1).tolist()

            path: list[int] = []
            at = -1
            while True:
                want = preds[0] if at == -1 else preds[1 + at]
                nxt = next(
                    (j for j in tree.children_of(at) if tree.tokens[j] == want),
                    None,
                )
                if nxt is None:
                    break
                path.append(nxt)
                at = nxt
            bonus = preds[0] if at == -1 else preds[1 + at]
            stats["accepted"] += len(path)
            # hidden state at the last accepted fed position drives the
            # next draft (it predicted the bonus token)
            hidden = out.hidden_states[-1][0, 0 if at == -1 else 1 + at]

            new = [tree.tokens[j] for j in path] + [bonus]
            tokens.extend(new)
            produced += len(new)
            past = gather_kv(past, prefix_len, [0] + [1 + j for j in path])
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats


def train_medusa_heads(
    model, heads, token_seqs, *, epochs: int = 5, lr: float = 1e-3
) -> list[float]:
    """Medusa-1: freeze the base model, fit heads on its hidden states.

    Head i learns P(token[t+2+i] | hidden[t]). Returns per-epoch mean
    loss (summed over heads).
    """
    import torch
    import torch.nn.functional as F

    device = next(model.parameters()).device
    opt = torch.optim.Adam(heads.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for seq in token_seqs:
            ids = torch.tensor([seq], device=device)
            with torch.no_grad():  # not inference_mode: heads backprop through hidden
                hidden = model(input_ids=ids, output_hidden_states=True
                               ).hidden_states[-1][0]
            loss = torch.zeros((), device=device)
            for i, head in enumerate(heads):
                offset = 2 + i  # hidden[t] -> token[t + 2 + i]
                if len(seq) <= offset:
                    continue
                logits = head(hidden[: len(seq) - offset])
                targets = ids[0, offset:]
                loss = loss + F.cross_entropy(logits, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, count = total + float(loss.detach()), count + 1
        losses.append(total / max(count, 1))
    return losses

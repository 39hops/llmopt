"""EAGLE-2-style decoding: feature-level drafting + dynamic draft trees.

EAGLE's insight: predicting the target's next *hidden state* is easier
than predicting tokens — the draft head is a small autoregressive model
over (last hidden, next token embedding) pairs, and reuses the target's
own frozen lm_head to turn predicted hiddens into token distributions.

EAGLE-2's addition: the draft tree is *dynamic* — candidate nodes are
ranked by cumulative draft confidence and only the global top-M are
kept, so budget flows to whichever branch the draft is sure about,
instead of a fixed tree shape.

Verification reuses tree_verify: output is exactly target-greedy no
matter how good or bad the head is.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from llmopt.decoding.tree_verify import (
    TokenTree,
    gather_kv,
    tree_attention_inputs,
)


class EagleHead(nn.Module):
    """hidden[t], embed(x[t+1]) -> predicted hidden[t+1]."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fuse = nn.Linear(2 * hidden_size, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size), nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

    def forward(self, hidden, embed):
        h = self.fuse(torch.cat([hidden, embed], dim=-1))
        return h + self.mlp(h)


def dynamic_draft_tree(
    model, head, hidden, next_token: int, *,
    depth: int = 4, branch: int = 3, budget: int = 8,
) -> list[list[int]]:
    """Roll the head autoregressively, expanding a tree; keep the
    global top-``budget`` nodes by cumulative draft logprob (EAGLE-2's
    confidence ranking). Returns candidate paths for TokenTree."""
    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    # frontier entries: (cum_logprob, path_tokens, hidden_state)
    frontier = [(0.0, [next_token], hidden)]
    nodes: list[tuple[float, list[int]]] = []
    for _ in range(depth):
        nxt = []
        for cum, path, h in frontier:
            h2 = head(h, embed(torch.tensor(path[-1], device=h.device)))
            logp = torch.log_softmax(lm_head(h2), dim=-1)
            top = torch.topk(logp, branch)
            for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                cand = (cum + lp, path + [tok])
                nodes.append(cand)
                nxt.append((*cand, h2))
        nxt.sort(key=lambda t: -t[0])
        frontier = nxt[:budget]  # beam by draft confidence
    keep = sorted(nodes, key=lambda t: -t[0])[:budget]
    return [path for _, path in keep]


def generate_eagle(
    model,
    head,
    input_ids,
    *,
    max_new_tokens: int = 128,
    depth: int = 4,
    branch: int = 2,
    budget: int = 8,
    eos_token_id: int | None = None,
):
    """Greedy decoding with EAGLE drafts, tree-verified. Token-identical
    to target greedy. Returns (tokens, stats)."""
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
            budget_now = min(budget, max_new_tokens - produced - 1)
            cands = [
                c[1:]  # paths start with tokens[-1]
                for c in dynamic_draft_tree(
                    model, head, hidden, tokens[-1],
                    depth=depth, branch=branch, budget=max(budget_now, 1),
                )
            ]
            cands = [c[:budget_now] for c in cands if c]
            tree = TokenTree.from_candidates(cands)
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

            path, at = [], -1
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


def train_eagle_head(
    model, head, token_seqs, *, epochs: int = 10, lr: float = 1e-3
) -> list[float]:
    """Fit the head to predict the target's next hidden state (smooth L1)
    + next-token CE through the frozen lm_head."""
    import torch.nn.functional as F

    device = next(model.parameters()).device
    embed = model.get_input_embeddings()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for seq in token_seqs:
            ids = torch.tensor([seq], device=device)
            with torch.no_grad():
                hs = model(input_ids=ids, output_hidden_states=True
                           ).hidden_states[-1][0]
            # head sees (hidden[t], embed(x[t+1])), targets hidden[t+1]
            pred = head(hs[:-1], embed(ids[0, 1:]))
            loss = F.smooth_l1_loss(pred, hs[1:])
            if len(seq) > 2:
                logits = model.lm_head(pred[:-1])
                loss = loss + F.cross_entropy(logits, ids[0, 2:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, count = total + float(loss.detach()), count + 1
        losses.append(total / max(count, 1))
    return losses

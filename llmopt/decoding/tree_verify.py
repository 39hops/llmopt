"""Tree verification: verify several draft continuations in one pass.

Medusa-style tree attention, but drafts come from prompt-lookup (top-k
n-gram matches) instead of trained heads, so it works on any model.

Candidates are trie-merged into a token tree, linearized parents-first.
One forward pass sees the whole tree with a 4D attention mask (a node
attends the cached prefix + its ancestors + itself) and depth-based
position_ids (siblings share a position). The greedy walk then descends
the tree following argmax predictions; the accepted path's KV entries
are gathered out of the tree-shaped cache so the cache stays linear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def find_ngram_continuations(
    context: Sequence[int],
    *,
    max_ngram: int = 3,
    min_ngram: int = 1,
    num_draft: int = 10,
    num_candidates: int = 4,
) -> list[list[int]]:
    """Up to num_candidates distinct continuations of the context suffix,
    longest-ngram / most-recent matches first (see find_ngram_continuation).
    """
    ctx = list(context)
    n_total = len(ctx)
    out: list[list[int]] = []
    for n in range(min(max_ngram, n_total - 1), min_ngram - 1, -1):
        suffix = ctx[n_total - n :]
        for start in range(n_total - n - 1, -1, -1):
            if ctx[start : start + n] == suffix:
                cont = ctx[start + n : start + n + num_draft]
                if cont and cont not in out:
                    out.append(cont)
                    if len(out) == num_candidates:
                        return out
    return out


@dataclass
class TokenTree:
    """Trie of draft continuations, linearized parents-first.

    tokens[i] is the token at node i; parent[i] is the node index of its
    parent, or -1 for depth-0 nodes (children of the current last token).
    depth[i] = distance from the root (0 for parent == -1).
    """

    tokens: list[int]
    parent: list[int]
    depth: list[int]

    @classmethod
    def from_candidates(cls, candidates: Sequence[Sequence[int]]) -> "TokenTree":
        tokens: list[int] = []
        parent: list[int] = []
        depth: list[int] = []
        children: dict[int, dict[int, int]] = {-1: {}}  # node -> token -> node
        for cand in candidates:
            at = -1
            for tok in cand:
                nxt = children[at].get(tok)
                if nxt is None:
                    nxt = len(tokens)
                    tokens.append(tok)
                    parent.append(at)
                    depth.append(0 if at == -1 else depth[at] + 1)
                    children[at][tok] = nxt
                    children[nxt] = {}
                at = nxt
        return cls(tokens, parent, depth)

    def ancestors(self, i: int) -> list[int]:
        """Node indices from depth-0 down to i, inclusive."""
        path = [i]
        while self.parent[path[-1]] != -1:
            path.append(self.parent[path[-1]])
        return path[::-1]

    def children_of(self, i: int) -> list[int]:
        return [j for j, p in enumerate(self.parent) if p == i]

    def __len__(self) -> int:
        return len(self.tokens)


def tree_attention_inputs(tree: TokenTree, prefix_len: int, device, dtype):
    """(input-extension mask, position_ids, cache_position) for one pass.

    fed = [last_token] + tree.tokens. Row i of the mask lets fed[i]
    attend the whole prefix, fed[0], its tree ancestors, and itself —
    a complete mask (causality is implied), so backends that take a 4D
    mask verbatim need no internal causal mask.
    """
    import torch

    q = 1 + len(tree)
    k = prefix_len + q
    neg = torch.finfo(dtype).min
    mask = torch.full((1, 1, q, k), neg, device=device, dtype=dtype)
    mask[..., :, :prefix_len] = 0  # prefix visible to all
    mask[..., :, prefix_len] = 0  # last_token visible to all
    for i in range(len(tree)):
        for a in tree.ancestors(i):
            mask[..., 1 + i, prefix_len + 1 + a] = 0
    pos = torch.tensor(
        [[prefix_len] + [prefix_len + 1 + d for d in tree.depth]], device=device
    )
    cache_pos = torch.arange(prefix_len, prefix_len + q, device=device)
    return mask, pos, cache_pos


def gather_kv(past, prefix_len: int, keep: Sequence[int]):
    """Compact a tree-shaped cache: keep the prefix plus the fed slots in
    `keep` (indices into the fed block), in order. In-place per layer.
    """
    import torch

    idx = torch.arange(prefix_len).tolist() + [prefix_len + j for j in keep]

    def take(t):
        return t[:, :, idx]

    if hasattr(past, "layers"):  # transformers 5.x
        for layer in past.layers:
            layer.keys, layer.values = take(layer.keys), take(layer.values)
        return past
    if hasattr(past, "key_cache"):  # transformers 4.x DynamicCache
        for i in range(len(past.key_cache)):
            past.key_cache[i] = take(past.key_cache[i])
            past.value_cache[i] = take(past.value_cache[i])
        return past
    return tuple((take(k), take(v)) for k, v in past)


def generate_lookup_tree(
    model,
    input_ids,
    *,
    max_new_tokens: int = 128,
    num_draft: int = 10,
    max_ngram: int = 3,
    num_candidates: int = 4,
    eos_token_id: int | None = None,
):
    """Greedy decoding, prompt-lookup drafts verified as a tree.

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
        # prefill
        out = model(
            input_ids=torch.tensor([tokens], device=device), use_cache=True
        )
        past = out.past_key_values
        stats["forward_passes"] += 1
        tokens.append(int(out.logits[0, -1].argmax()))
        produced = 1

        while produced < max_new_tokens:
            cands = [
                c[: max_new_tokens - produced - 1]
                for c in find_ngram_continuations(
                    tokens, max_ngram=max_ngram, num_draft=num_draft,
                    num_candidates=num_candidates,
                )
            ]
            tree = TokenTree.from_candidates([c for c in cands if c])
            stats["drafted"] += len(tree)

            prefix_len = valid_len(past)
            assert prefix_len == len(tokens) - 1
            fed = [tokens[-1]] + tree.tokens
            mask, pos, cache_pos = tree_attention_inputs(
                tree, prefix_len, device, model.dtype
            )
            out = model(
                input_ids=torch.tensor([fed], device=device),
                past_key_values=past, attention_mask=mask,
                position_ids=pos, cache_position=cache_pos, use_cache=True,
            )
            past = out.past_key_values
            stats["forward_passes"] += 1
            preds = out.logits[0].argmax(-1).tolist()  # per fed node

            # greedy walk: from the root, follow the child matching argmax
            path: list[int] = []  # accepted node indices (fed offsets - 1)
            at = -1  # -1 = root (fed[0])
            while True:
                want = preds[0] if at == -1 else preds[1 + at]
                nxt = next(
                    (
                        j
                        for j in tree.children_of(at)
                        if tree.tokens[j] == want
                    ),
                    None,
                )
                if nxt is None:
                    break
                path.append(nxt)
                at = nxt
            bonus = preds[0] if at == -1 else preds[1 + at]
            stats["accepted"] += len(path)

            new = [tree.tokens[j] for j in path] + [bonus]
            tokens.extend(new)
            produced += len(new)
            # keep prefix + fed[0] + accepted path; drop other tree slots
            past = gather_kv(past, prefix_len, [0] + [1 + j for j in path])
            if eos_token_id is not None and eos_token_id in new:
                idx = tokens.index(eos_token_id, len(tokens) - len(new))
                tokens = tokens[: idx + 1]
                break

    return tokens, stats

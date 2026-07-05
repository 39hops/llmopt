"""Tree verification tests on a tiny random Llama (CPU, no download).

The strong invariant: tree-verified decoding is token-identical to
vanilla greedy, and per-node tree logits match feeding each root-to-node
path sequentially.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.tree_verify import (
    TokenTree,
    find_ngram_continuations,
    generate_lookup_tree,
    tree_attention_inputs,
)

VOCAB = 128


@pytest.fixture(scope="module")
def model():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=512, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


def _greedy(model, ids, n):
    out = list(ids)
    with torch.inference_mode():
        for _ in range(n):
            logits = model(input_ids=torch.tensor([out])).logits
            out.append(int(logits[0, -1].argmax()))
    return out


def test_token_tree_trie_merges_shared_prefixes():
    tree = TokenTree.from_candidates([[1, 2, 3], [1, 2, 9], [7]])
    assert tree.tokens == [1, 2, 3, 9, 7]
    assert tree.parent == [-1, 0, 1, 1, -1]
    assert tree.depth == [0, 1, 2, 2, 0]
    assert tree.ancestors(3) == [0, 1, 3]


def test_find_ngram_continuations_distinct():
    ctx = [1, 2, 5, 5, 1, 2, 7, 7, 1, 2]
    cands = find_ngram_continuations(ctx, max_ngram=2, num_draft=2)
    assert [7, 7] in cands and [5, 5] in cands
    assert cands.index([7, 7]) < cands.index([5, 5])  # recency order


def test_tree_logits_match_sequential_paths(model):
    prompt = [3, 14, 15, 9, 2, 6, 5, 3, 14]
    tree = TokenTree.from_candidates([[10, 20, 30], [10, 21], [40]])
    last = 15
    fed = [last] + tree.tokens

    with torch.inference_mode():
        prefix_out = model(input_ids=torch.tensor([prompt]), use_cache=True)
        mask, pos, cache_pos = tree_attention_inputs(
            tree, len(prompt), "cpu", model.dtype
        )
        tree_logits = model(
            input_ids=torch.tensor([fed]),
            past_key_values=prefix_out.past_key_values,
            attention_mask=mask, position_ids=pos, cache_position=cache_pos,
        ).logits[0]

        for i in range(len(tree)):
            path_tokens = [tree.tokens[j] for j in tree.ancestors(i)]
            seq = prompt + [last] + path_tokens
            ref = model(input_ids=torch.tensor([seq])).logits[0, -1]
            assert torch.allclose(tree_logits[1 + i], ref, atol=1e-4), i


def test_tree_lookup_matches_vanilla_greedy(model):
    torch.manual_seed(1)
    # repetitive prompt so candidates exist and partially accept
    prompt = [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6]
    n = 30
    ref = _greedy(model, prompt, n)

    out, stats = generate_lookup_tree(
        model, prompt, max_new_tokens=n, num_draft=4, max_ngram=3,
        num_candidates=3,
    )
    assert out == ref
    assert stats["forward_passes"] <= n + 1


def test_tree_lookup_no_candidates_prompt(model):
    prompt = [11, 23, 87, 41, 3]  # no repeated ngrams
    n = 8
    out, _ = generate_lookup_tree(model, prompt, max_new_tokens=n)
    assert out == _greedy(model, prompt, n)

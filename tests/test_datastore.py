"""REST-style datastore drafting: pure index behavior + engine integration."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.datastore import SuffixDatastore, make_draft_fn
from llmopt.decoding.stacked import StackedEngine

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


# --- pure index ---------------------------------------------------------------


def test_longest_suffix_wins():
    ds = SuffixDatastore(max_ngram=3)
    ds.add([1, 2, 3, 4, 5])      # "...2,3" -> 4
    ds.add([9, 2, 3, 7, 8])      # "...2,3" -> 7 but "9,2,3" -> 7 only via 3-gram
    # context ends 9,2,3: the 3-gram match must beat the more frequent 2-gram
    assert ds.draft([0, 9, 2, 3], num_draft=2) == [7, 8]


def test_majority_vote_on_next_token():
    ds = SuffixDatastore(max_ngram=2)
    ds.add([1, 2, 30, 40])
    ds.add([1, 2, 30, 41])
    ds.add([1, 2, 99, 42])
    # "1,2" continues with 30 twice, 99 once -> 30 wins; most recent 30-
    # occurrence supplies the tail
    assert ds.draft([5, 1, 2], num_draft=2) == [30, 41]


def test_empty_when_unseen():
    ds = SuffixDatastore()
    ds.add([1, 2, 3])
    assert ds.draft([50, 60], num_draft=4) == []


def test_draft_fn_falls_back_to_prompt_lookup():
    ds = SuffixDatastore()  # empty store
    fn = make_draft_fn(ds, max_ngram=2)
    # repetitive context: prompt-lookup finds the continuation
    ctx = [7, 8, 9, 7, 8]
    assert fn(ctx, 2) == [9, 7]


# --- engine integration -------------------------------------------------------


PROMPT_A = [5, 6, 7, 8, 9, 10, 11, 3]
PROMPT_B = [42, 43, 5, 6, 7, 8, 9, 10, 11, 3]  # different prefix, same tail


def test_stacked_with_datastore_matches_greedy(model):
    n = 16
    ds = SuffixDatastore()
    engine = StackedEngine(model, num_draft=6, max_ngram=3, datastore=ds)
    tokens, _ = engine.generate(PROMPT_A, max_new_tokens=n)
    assert tokens == _greedy(model, PROMPT_A, n)
    assert len(ds.seqs) == 1  # generation stored

    tokens_b, _ = engine.generate(PROMPT_B, max_new_tokens=n)
    assert tokens_b == _greedy(model, PROMPT_B, n)


def test_datastore_improves_drafting_on_repeat(model):
    # same prompt twice: second run drafts the previous generation from the
    # store and must not do worse than the cold run
    n = 24
    ds = SuffixDatastore()
    engine = StackedEngine(model, num_draft=6, max_ngram=3, datastore=ds)
    out1, cold = engine.generate(PROMPT_A, max_new_tokens=n)
    out2, warm = engine.generate(PROMPT_A, max_new_tokens=n)
    assert out2 == out1
    assert warm["forward_passes"] <= cold["forward_passes"]
    assert warm["accepted"] >= cold["accepted"]
    # the whole previous answer is in the store: near-perfect acceptance
    assert warm["forward_passes"] <= n // 6 + 2

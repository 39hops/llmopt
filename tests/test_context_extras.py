"""RULER-style eval machinery + gist tokens (tiny random Llama, CPU)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.context.gist import (
    compress_to_gist_kv,
    gist_attention_mask,
    gist_training_step,
    to_additive,
)
from llmopt.context.ruler import (
    Task,
    effective_context_length,
    evaluate,
    make_niah,
    make_suite,
    make_variable_tracking,
    score,
)

VOCAB = 64


def _llama(seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=256, attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg).eval()


# --- ruler ------------------------------------------------------------------


def test_niah_answer_is_retrievable_and_deterministic():
    t = make_niah(200, seed=7)
    assert f"is {t.answer}." in t.prompt  # needle present verbatim
    assert t.prompt.split("What is the magic number for ")[1].split("?")[0] in t.prompt
    assert make_niah(200, seed=7) == t  # seeded → reproducible


def test_niah_multikey_has_distractors():
    t = make_niah(300, num_keys=4, seed=3)
    assert t.kind == "niah_multikey"
    assert t.prompt.count("The magic number for") == 4
    assert t.prompt.count(t.answer) == 1  # only the target value queried


def test_variable_tracking_requires_all_hops():
    t = make_variable_tracking(300, hops=4, seed=5)
    assert t.prompt.count("Set VAR_") == 4
    # answer value appears exactly once (at the chain root, not the queried var)
    assert t.prompt.count(t.answer) == 1
    queried = t.prompt.split("value of ")[1].split("?")[0]
    assert f"Set {queried} = {t.answer}" not in t.prompt  # must follow chain


def test_scoring_and_evaluate_group_by_kind_and_length():
    suite = make_suite([100, 200], samples_per_length=2, tasks=("niah",))
    oracle = {t.prompt: t.answer for t in suite}
    results = evaluate(lambda p: f"The answer is {oracle[p]}", suite)
    assert set(results) == {("niah", 100), ("niah", 200)}
    assert all(acc == 1.0 for acc in results.values())

    broken = evaluate(lambda p: "no idea", suite)
    assert all(acc == 0.0 for acc in broken.values())


def test_effective_context_length_first_failure_wins():
    results = {
        ("niah", 100): 1.0, ("vt", 100): 1.0,
        ("niah", 200): 0.9, ("vt", 200): 0.9,
        ("niah", 400): 0.5, ("vt", 400): 0.2,
        ("niah", 800): 1.0, ("vt", 800): 1.0,  # past first failure — ignored
    }
    assert effective_context_length(results, threshold=0.85) == 200


def test_score_containment():
    t = Task("p", "12345", "niah", {"length_words": 10})
    assert score("the answer is 12345.", t)
    assert not score("the answer is 999", t)


# --- gist -------------------------------------------------------------------


def test_gist_mask_semantics():
    m = gist_attention_mask(6, 2, 4)  # inst [0,2), gists [2,4), answer [4,6)
    assert m[3, 0] and m[3, 1]        # gists read the instruction
    assert not m[4, 0] and not m[5, 1]  # answer blinded to instruction
    assert m[4, 2] and m[5, 3]        # answer reads gists
    assert not m[2, 3]                # still causal inside gist span
    assert m.tril().equal(m)          # nothing attends the future


def test_gist_kv_matches_masked_full_forward():
    # continuing from the gist-only cache must equal the full masked forward
    model = _llama()
    inst, gists, answer = [5, 9, 11, 3], [60, 61], [7, 8]
    cache, glen = compress_to_gist_kv(model, inst, gists)
    assert glen == 2
    assert cache.layers[0].keys.shape[2] == 2  # instruction rows dropped

    ids = inst + gists + answer
    full_mask = gist_attention_mask(len(ids), len(inst), len(inst) + 2)
    with torch.inference_mode():
        ref = model(
            input_ids=torch.tensor([ids]),
            attention_mask=to_additive(full_mask, model.dtype),
        ).logits[0, -len(answer):]
        got = model(
            input_ids=torch.tensor([answer]),
            past_key_values=cache,
            position_ids=torch.tensor([[len(ids) - 2, len(ids) - 1]]),
            attention_mask=torch.ones(1, 2 + len(answer), dtype=torch.long),
        ).logits[0]
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_gist_training_step_learns():
    # a few SGD steps on one example: loss must drop (gists are learnable)
    model = _llama().train()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    inst, gists, answer = [5, 9, 11, 3], [60, 61], [7, 8, 12]
    losses = []
    for _ in range(8):
        loss = gist_training_step(model, inst, gists, answer)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(torch.tensor(losses)))

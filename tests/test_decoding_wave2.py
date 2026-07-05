"""EAGLE-2, quality verify, sampler-aware speculative (CPU, tiny Llamas)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.eagle import (
    EagleHead,
    dynamic_draft_tree,
    generate_eagle,
    train_eagle_head,
)
from llmopt.decoding.quality_verify import generate_quality_verify
from llmopt.decoding.samplers import top_k
from llmopt.decoding.speculative_filtered import generate_speculative_filtered

VOCAB, HIDDEN = 64, 64


def _llama(layers, seed):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=512,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg)


@pytest.fixture(scope="module")
def target():
    return _llama(2, 0).eval()


def _greedy(model, ids, n):
    out = list(ids)
    with torch.inference_mode():
        for _ in range(n):
            logits = model(input_ids=torch.tensor([out])).logits
            out.append(int(logits[0, -1].argmax()))
    return out


# --- eagle -------------------------------------------------------------------


def test_dynamic_tree_respects_budget(target):
    torch.manual_seed(1)
    head = EagleHead(HIDDEN)
    with torch.inference_mode():
        h = target(
            input_ids=torch.tensor([[1, 2, 3]]), output_hidden_states=True
        ).hidden_states[-1][0, -1]
        paths = dynamic_draft_tree(
            target, head, h, 5, depth=3, branch=3, budget=6
        )
    assert 0 < len(paths) <= 6
    assert all(p[0] == 5 for p in paths)


def test_eagle_untrained_still_greedy_exact(target):
    torch.manual_seed(2)
    head = EagleHead(HIDDEN)
    prompt = [5, 6, 7, 8, 5, 6, 7, 8]
    n = 20
    with torch.inference_mode():
        out, stats = generate_eagle(
            target, head, prompt, max_new_tokens=n, depth=3, branch=2, budget=6
        )
    assert out == _greedy(target, prompt, n)
    assert stats["forward_passes"] <= n + 1


def test_eagle_training_lifts_acceptance(target):
    torch.manual_seed(3)
    head = EagleHead(HIDDEN)
    prompt = [9, 4, 2]
    n = 24
    base = _greedy(target, prompt, n)

    with torch.inference_mode():
        _, before = generate_eagle(target, head, prompt, max_new_tokens=n)
    losses = train_eagle_head(target, head, [base], epochs=40, lr=2e-3)
    with torch.inference_mode():
        out, after = generate_eagle(target, head, prompt, max_new_tokens=n)

    assert losses[-1] < losses[0]
    assert after["accepted"] > before["accepted"]
    assert out == base  # correctness untouched by training


# --- quality verify ----------------------------------------------------------


def test_quality_verify_accepts_more_as_criterion_loosens(target):
    draft = _llama(1, 7).eval()
    prompt = [5, 6, 7, 8, 5, 6, 7, 8]
    n = 30
    rates = {}
    for k in (1, 5, 50):
        _, st = generate_quality_verify(
            target, draft, prompt, max_new_tokens=n, criterion="top_k", top_k=k
        )
        rates[k] = st["accepted"] / max(st["drafted"], 1)
    assert rates[1] <= rates[5] <= rates[50]
    assert rates[50] > rates[1]  # loosening must actually buy acceptance


def test_quality_verify_topk_invariant_holds(target):
    draft = _llama(1, 7).eval()
    prompt = [1, 2, 3, 4]
    n = 20
    out, _ = generate_quality_verify(
        target, draft, prompt, max_new_tokens=n, criterion="top_k", top_k=3
    )
    # every emitted token is within target top-3 given its prefix
    with torch.inference_mode():
        for i in range(len(prompt), len(out)):
            logits = target(input_ids=torch.tensor([out[:i]])).logits[0, -1]
            assert out[i] in torch.topk(logits, 3).indices


# --- sampler-aware speculative ------------------------------------------------


def test_filtered_speculative_matches_direct_distribution(target):
    draft = _llama(1, 8).eval()
    prompt = [5, 6, 7, 8]
    procs = [top_k(4)]
    n_samples = 400

    with torch.inference_mode():
        logits = target(input_ids=torch.tensor([prompt])).logits[0, -1]
        for pr in procs:
            logits = pr(logits, prompt)
        direct = torch.softmax(logits, -1)

    counts = torch.zeros(VOCAB)
    for s in range(n_samples):
        out, _ = generate_speculative_filtered(
            target, draft, prompt, processors=procs,
            max_new_tokens=1, num_draft=3, seed=s,
        )
        counts[out[len(prompt)]] += 1
    emp = counts / n_samples

    assert (emp[direct == 0] == 0).all()  # never leaves the filtered support
    tv = 0.5 * float((emp - direct).abs().sum())
    assert tv < 0.12  # statistical tolerance at n=400

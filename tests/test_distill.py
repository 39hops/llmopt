"""Logit-KD and the draft-distillation payoff, on tiny random Llamas.

The headline test is the speculative-decoding pairing: distilling a
1-layer draft toward a 2-layer target must raise the accept rate that
generate_speculative reports, while output stays target-greedy-exact.
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.speculative import generate_speculative
from llmopt.distill.logit_kd import distill_logits, kd_loss

VOCAB = 64


def _llama(layers, seed):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=layers, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=512,
        attn_implementation="eager",
    )
    return LlamaForCausalLM(cfg)


def _greedy(model, ids, n):
    out = list(ids)
    with torch.inference_mode():
        for _ in range(n):
            logits = model(input_ids=torch.tensor([out])).logits
            out.append(int(logits[0, -1].argmax()))
    return out


def test_kd_loss_zero_iff_identical():
    logits = torch.randn(7, VOCAB)
    assert float(kd_loss(logits, logits)) == pytest.approx(0.0, abs=1e-6)
    assert float(kd_loss(logits + torch.randn_like(logits), logits)) > 0.01


def test_kd_loss_temperature_scaling_keeps_magnitude():
    a, b = torch.randn(5, VOCAB), torch.randn(5, VOCAB)
    hot = float(kd_loss(a, b, temperature=4.0))
    assert 0 < hot  # T^2 scaling keeps it in a trainable range
    assert hot != float(kd_loss(a, b, temperature=1.0))


def test_draft_distillation_lifts_accept_rate():
    target = _llama(2, seed=0).eval()
    draft = _llama(1, seed=1)  # different init: bad drafter out of the box
    prompt = [5, 6, 7, 8, 9, 10]
    n = 30

    # corpus = the target's own greedy continuations (deployment dist)
    corpus = [_greedy(target, [s, s + 1, s + 2], 24) for s in range(1, 9)]

    def accept_rate():
        out, st = generate_speculative(
            target, draft.eval(), prompt, max_new_tokens=n, num_draft=4
        )
        assert out == _greedy(target, prompt, n)  # exactness is free
        return st["accepted"] / max(st["drafted"], 1)

    before = accept_rate()
    losses = distill_logits(draft.train(), target, corpus, epochs=25, lr=2e-3)
    after = accept_rate()

    assert losses[-1] < losses[0] * 0.5
    assert after > before, (before, after)

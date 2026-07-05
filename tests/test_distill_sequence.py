"""Sequence-KD and GKD on tiny random Llamas (CPU, no download)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.distill.sequence_kd import generalized_jsd, gkd, sequence_kd

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


def test_generalized_jsd_zero_iff_identical_and_beta_limits():
    logits = torch.randn(6, VOCAB)
    assert float(generalized_jsd(logits, logits, 0.5)) == pytest.approx(0.0, abs=1e-5)
    other = torch.randn(6, VOCAB)
    for beta in (0.1, 0.5, 0.9):
        assert float(generalized_jsd(logits, other, beta)) > 0


def test_sequence_kd_loss_decreases_and_student_mimics():
    teacher, student = _llama(2, 0).eval(), _llama(1, 1)
    prompts = [[s, s + 3] for s in range(1, 7)]
    losses = sequence_kd(
        student, teacher, prompts, gen_len=16, epochs=20, lr=2e-3
    )
    assert losses[-1] < losses[0] * 0.5


def test_gkd_on_policy_loss_decreases():
    teacher, student = _llama(2, 0).eval(), _llama(1, 2)
    prompts = [[3, 9], [12, 5]]
    losses = gkd(
        student, teacher, prompts, gen_len=12, steps=30, lr=2e-3, beta=0.5
    )
    assert sum(losses[-5:]) < sum(losses[:5])

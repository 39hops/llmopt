"""ECE, TTFT/TPOT, lookahead decoding, scheduler (CPU, tiny Llama)."""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from llmopt.decoding.lookahead import generate_lookahead
from llmopt.decoding.scheduler import Scheduler
from llmopt.eval.calibration import ece, model_ece
from llmopt.eval.latency import measure_latency

VOCAB = 64


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


def test_ece_known_values():
    # perfectly calibrated: 80% confidence, 80% accurate
    conf = torch.full((1000,), 0.8)
    correct = torch.zeros(1000, dtype=torch.bool)
    correct[:800] = True
    assert ece(conf, correct) == pytest.approx(0.0, abs=0.01)
    # overconfident: 90% confidence, 50% accurate -> ECE ~0.4
    conf = torch.full((1000,), 0.9)
    correct[:] = False
    correct[:500] = True
    assert ece(conf, correct) == pytest.approx(0.4, abs=0.01)


def test_model_ece_in_range(model):
    seqs = [[5, 6, 7, 8, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6]]
    e = model_ece(model, seqs)
    assert 0.0 <= e <= 1.0


def test_measure_latency(model):
    r = measure_latency(model, [1, 2, 3, 4, 5], max_new_tokens=8)
    assert r.ttft_s > 0 and r.tpot_s > 0
    assert r.total_s == pytest.approx(r.ttft_s + 7 * r.tpot_s)
    assert r.decode_tok_s > 0


def test_lookahead_matches_greedy(model):
    for prompt in ([5, 6, 7, 8, 5, 6, 7, 8, 5, 6], [11, 23, 42, 7]):
        n = 24
        out, stats = generate_lookahead(model, prompt, max_new_tokens=n, window=6)
        assert out == _greedy(model, prompt, n)
        assert stats["forward_passes"] <= n + 1


def test_scheduler_priority_admission(model):
    s = Scheduler(model, max_batch=1, chunk_size=64)
    low = s.submit([1, 2, 3], max_new_tokens=4, priority=0)
    high = s.submit([4, 5, 6], max_new_tokens=4, priority=5)
    finish_order = []
    while s.waiting or s.running:
        s.step()
        for r in [r for r in s.running if r.done]:
            finish_order.append(r.rid)
            s._release(r)
    assert finish_order == [high, low]  # high priority admitted first


def test_scheduler_preemption_keeps_output_exact(model):
    n = 12
    prompts = {"low": [5, 6, 7, 8, 9], "high": [40, 41, 42]}
    s = Scheduler(model, max_batch=1, chunk_size=64)
    low = s.submit(prompts["low"], max_new_tokens=n, priority=0)
    # let the low-priority request start decoding, then submit high
    for _ in range(4):
        s.step()
    assert s.running and s.running[0].rid == low
    high = s.submit(prompts["high"], max_new_tokens=n, priority=9)
    results = s.run()

    assert s.stats["preemptions"] == 1
    assert results[high] == _greedy(model, prompts["high"], n)
    assert results[low] == _greedy(model, prompts["low"], n)  # exact despite preempt

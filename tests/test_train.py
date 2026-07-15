"""LoRA family, sequence packing, preference losses (CPU)."""

import math

import pytest

torch = pytest.importorskip("torch")

from llmopt.train.lora import (
    DoRALinear,
    LoRALinear,
    apply_lora,
    trainable_fraction,
)
from llmopt.train.packing import pack_batch, pack_greedy
from llmopt.train.preference import (
    dpo_loss,
    grpo_advantages,
    grpo_loss,
    ipo_loss,
    kto_loss,
    orpo_loss,
    simpo_loss,
)


# --- lora -------------------------------------------------------------------


def test_lora_starts_at_base_and_merges():
    torch.manual_seed(0)
    base = torch.nn.Linear(16, 8)
    lora = LoRALinear(base, r=4)
    x = torch.randn(5, 16)
    assert torch.allclose(lora(x), base(x))  # B zero-init: identical start
    with torch.no_grad():
        lora.b.copy_(torch.randn_like(lora.b))
    assert not torch.allclose(lora(x), base(x))
    assert torch.allclose(lora.merge()(x), lora(x), atol=1e-5)


def test_dora_starts_at_base():
    torch.manual_seed(1)
    base = torch.nn.Linear(16, 8, bias=False)
    dora = DoRALinear(base, r=4)
    x = torch.randn(5, 16)
    assert torch.allclose(dora(x), base(x), atol=1e-5)


def test_apply_lora_targets_and_freezes():
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
    )
    model = LlamaForCausalLM(cfg)
    n = apply_lora(model, ("q_proj", "v_proj"), r=4)
    assert n == 4  # 2 layers x (q, v)
    assert trainable_fraction(model) < 0.05
    out = model(input_ids=torch.tensor([[1, 2, 3]]))
    assert out.logits.shape == (1, 3, 64)


# --- packing ----------------------------------------------------------------


def test_pack_greedy_fits_and_covers():
    lengths = [7, 2, 5, 3, 3, 8]
    bins = pack_greedy(lengths, capacity=10)
    packed = sorted(i for b in bins for i in b)
    assert packed == list(range(6))
    assert all(sum(lengths[i] for i in b) <= 10 for b in bins)
    assert len(bins) == 3  # FFD: [8,2] [7,3] [5,3]


def test_pack_batch_mask_blocks_cross_document_attention():
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    batch = pack_batch(seqs, capacity=5)
    ids, pos, mask = (
        batch["input_ids"], batch["position_ids"], batch["attention_mask"]
    )
    assert ids.shape[1] == 5
    for row, spans in enumerate(batch["doc_spans"]):
        for s, e in spans:
            assert pos[row, s] == 0  # positions reset per document
            for q in range(s, e):
                assert (mask[row, 0, q, s : q + 1] == 0).all()  # causal self
                assert (mask[row, 0, q, :s] < -1e30).all()  # no cross-doc


# --- preference losses -------------------------------------------------------


def _pairs(margin):
    t = lambda v: torch.tensor([v])
    return t(margin / 2), t(-margin / 2), t(0.0), t(0.0)  # pc, pr, rc, rr


def test_dpo_value_and_monotonicity():
    assert float(dpo_loss(*_pairs(0.0), beta=0.1)) == pytest.approx(math.log(2))
    assert float(dpo_loss(*_pairs(10.0))) < float(dpo_loss(*_pairs(0.0)))


def test_ipo_zero_at_target_margin():
    beta = 0.1
    assert float(ipo_loss(*_pairs(1 / (2 * beta)), beta=beta)) == pytest.approx(0.0)
    assert float(ipo_loss(*_pairs(100.0), beta=beta)) > 1  # DPO would reward this


def test_kto_prefers_high_reward_on_desirable():
    p = torch.tensor([2.0, 2.0])
    r = torch.zeros(2)
    loss_good = kto_loss(p, r, torch.tensor([True, True]))
    loss_bad = kto_loss(p, r, torch.tensor([False, False]))
    assert float(loss_good) < float(loss_bad)


def test_orpo_and_simpo_prefer_chosen():
    pc, pr = torch.tensor([-0.5]), torch.tensor([-2.0])
    flat = torch.tensor([-1.0])
    assert float(orpo_loss(pc, pr, torch.tensor(1.0))) < float(
        orpo_loss(flat, flat, torch.tensor(1.0))
    )
    assert float(simpo_loss(pc, pr)) < float(simpo_loss(flat, flat))


def test_grpo_advantages_and_clip():
    rewards = torch.tensor([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    groups = torch.tensor([0, 0, 0, 1, 1, 1])
    adv = grpo_advantages(rewards, groups)
    for g in (0, 1):
        assert float(adv[groups == g].mean()) == pytest.approx(0.0, abs=1e-6)
        assert float(adv[groups == g].std()) == pytest.approx(1.0, abs=1e-3)

    # clipping caps the incentive for large ratio moves
    logp_old = torch.zeros(3)
    adv3 = torch.ones(3)
    small = grpo_loss(torch.full((3,), 0.1), logp_old, adv3)
    huge = grpo_loss(torch.full((3,), 5.0), logp_old, adv3)
    assert float(huge) == pytest.approx(-1.2)  # clipped at 1 + 0.2
    assert float(small) > float(huge)


def test_grpo_dual_clip_bounds_negative_advantage():
    # the cycle-5 spike: policy drifts TOWARD a failed sample —
    # ratio explodes, A<0, and vanilla PPO clip leaves -ratio*A
    # unbounded. Dual-clip floors the term at dual_clip*A.
    logp_old = torch.zeros(3)
    adv_neg = torch.full((3,), -1.0)
    runaway = grpo_loss(torch.full((3,), 5.0), logp_old, adv_neg)  # ratio e^5
    assert float(runaway) == pytest.approx(3.0)  # dual_clip * |A|, not ~148
    # gradient must be finite and bounded too
    lp = torch.full((3,), 5.0, requires_grad=True)
    grpo_loss(lp, logp_old, adv_neg).backward()
    assert torch.isfinite(lp.grad).all()
    assert float(lp.grad.abs().max()) == 0.0  # fully floored: no push

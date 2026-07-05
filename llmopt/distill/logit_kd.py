"""Logit knowledge distillation: match the student's next-token
distribution to a frozen teacher's, token position by token position.

The highest-leverage use here is draft-model distillation: a draft
distilled toward the target model raises speculative-decoding accept
rates, which is the dominant term in speculative speedup. The same
trainer serves quantization-recovery (teacher = fp model, student =
quantized) — only the (student, teacher) pair changes.

Loss is forward KL(teacher || student) at temperature T, scaled by T^2
(Hinton et al.) so gradients stay comparable across temperatures.
"""

from __future__ import annotations

from typing import Sequence


def kd_loss(student_logits, teacher_logits, temperature: float = 1.0):
    """Mean forward KL over positions, temperature-scaled. Inputs are
    [N, vocab] logits; returns a scalar tensor."""
    import torch.nn.functional as F

    t = temperature
    # softmax/KL in fp32: fp16 log_softmax under-/overflows enough to NaN
    # Adam on real models (seen with a fp16 Qwen teacher)
    return (
        F.kl_div(
            F.log_softmax(student_logits.float() / t, dim=-1),
            F.log_softmax(teacher_logits.float() / t, dim=-1),
            log_target=True,
            reduction="batchmean",
        )
        * t * t
    )


def distill_logits(
    student,
    teacher,
    token_seqs: Sequence[Sequence[int]],
    *,
    epochs: int = 5,
    lr: float = 1e-3,
    temperature: float = 1.0,
) -> list[float]:
    """Fit ``student`` to ``teacher``'s per-position logits on the given
    token sequences (teacher frozen). Returns per-epoch mean loss.

    Vocabularies must match. For draft distillation, ``token_seqs``
    should be drawn from the deployment distribution — ideally the
    teacher's own generations (sequence-level KD sampling).
    """
    import torch

    device = next(student.parameters()).device
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        total, count = 0.0, 0
        for seq in token_seqs:
            ids = torch.tensor([seq], device=device)
            with torch.no_grad():
                t_logits = teacher(input_ids=ids).logits[0]
            s_logits = student(input_ids=ids).logits[0]
            loss = kd_loss(s_logits, t_logits, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, count = total + float(loss.detach()), count + 1
        losses.append(total / max(count, 1))
    return losses

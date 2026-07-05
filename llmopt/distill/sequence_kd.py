"""Sequence-level KD and on-policy GKD.

Sequence-KD (Kim & Rush 2016): the teacher *generates* the training
corpus; the student learns plain cross-entropy on those hard targets.
Captures the teacher's search behavior, not just per-token marginals.

GKD (Agarwal et al. 2023): fixes the train/inference mismatch — the
student is trained on its *own* generations, scored by the teacher
(generalized Jensen-Shannon between the two distributions). beta=0 is
forward KL(teacher||student), beta=1 reverse KL, 0.5 symmetric.
"""

from __future__ import annotations

from typing import Sequence


def _greedy(model, prompt: list[int], n: int) -> list[int]:
    import torch

    out = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            logits = model(input_ids=torch.tensor([out], device=model.device)).logits
            out.append(int(logits[0, -1].argmax()))
    return out


def _sample(model, prompt: list[int], n: int, gen) -> list[int]:
    import torch

    out = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            logits = model(input_ids=torch.tensor([out], device=model.device)).logits
            p = torch.softmax(logits[0, -1], dim=-1)
            out.append(int(torch.multinomial(p, 1, generator=gen)))
    return out


def generalized_jsd(student_logits, teacher_logits, beta: float = 0.5):
    """Generalized JSD between per-position distributions ([N, vocab]).

    Mixture m = beta*student + (1-beta)*teacher;
    loss = beta*KL(s||m) + (1-beta)*KL(t||m). beta->0 tends to forward
    KL(t||s), beta->1 to reverse KL(s||t).
    """
    import torch
    import torch.nn.functional as F

    s = F.log_softmax(student_logits, dim=-1)
    t = F.log_softmax(teacher_logits, dim=-1)
    m = torch.logsumexp(
        torch.stack([s + torch.log(torch.tensor(beta + 1e-8)),
                     t + torch.log(torch.tensor(1 - beta + 1e-8))]),
        dim=0,
    )
    kl = lambda a: F.kl_div(m, a, log_target=True, reduction="batchmean")
    return beta * kl(s) + (1 - beta) * kl(t)


def sequence_kd(
    student,
    teacher,
    prompts: Sequence[Sequence[int]],
    *,
    gen_len: int = 32,
    epochs: int = 5,
    lr: float = 1e-3,
) -> list[float]:
    """Teacher greedy-generates from each prompt; student trains with
    cross-entropy on the generated tokens (prompt positions excluded).
    Returns per-epoch mean loss."""
    import torch
    import torch.nn.functional as F

    corpus = [
        (len(p), _greedy(teacher, list(p), gen_len)) for p in prompts
    ]
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []
    for _ in range(epochs):
        total = 0.0
        for plen, seq in corpus:
            ids = torch.tensor([seq], device=student.device)
            logits = student(input_ids=ids).logits[0]
            # predict generated tokens only: position i predicts seq[i+1]
            loss = F.cross_entropy(
                logits[plen - 1 : -1], ids[0, plen:]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach())
        losses.append(total / len(corpus))
    return losses


def gkd(
    student,
    teacher,
    prompts: Sequence[Sequence[int]],
    *,
    gen_len: int = 32,
    steps: int = 20,
    lr: float = 1e-3,
    beta: float = 0.5,
    seed: int = 0,
) -> list[float]:
    """On-policy GKD: each step, the student samples a continuation of a
    prompt, and both models score it; the student minimizes generalized
    JSD against the teacher on those self-generated positions.
    Returns per-step loss."""
    import torch

    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    losses = []
    for step in range(steps):
        prompt = list(prompts[step % len(prompts)])
        student.eval()
        seq = _sample(student, prompt, gen_len, g)
        student.train()
        ids = torch.tensor([seq], device=student.device)
        with torch.no_grad():
            t_logits = teacher(input_ids=ids).logits[0]
        s_logits = student(input_ids=ids).logits[0]
        sl = slice(len(prompt) - 1, len(seq) - 1)  # generated positions
        loss = generalized_jsd(s_logits[sl], t_logits[sl], beta)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses
